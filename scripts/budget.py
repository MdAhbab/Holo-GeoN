"""
scripts/budget.py
=================
Will this fit, and how long will it take?

Prints measured memory and disk figures for a given configuration so the answer
is arithmetic rather than optimism. Everything here is derived from real tensor
shapes: the model is constructed, a batch of the configured size is built and
run, and peak allocation is read back. On a CUDA device the peak is
``torch.cuda.max_memory_allocated``; on CPU it is computed from the tensors the
forward pass actually materialises.

Target hardware for this project is an RTX 5070 Ti: 16 GB VRAM, sm_120, with
32 GB of system RAM.

Example
-------
python -m scripts.budget --batch-size 64 --window 256 --unique-residues 12000000
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.encoder import ESM_DIMS, EncodingStore
from src.model import HoloGeoN, model_kwargs_from_config

GB = 1024 ** 3


def _mib(n_bytes: float) -> str:
    return f"{n_bytes / 1024 ** 2:,.0f} MiB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--unique-residues", type=int, default=12_000_000,
                    help="total residues across unique parent proteins, for the "
                         "cache estimate (e.g. 20k proteins x 600 residues)")
    ap.add_argument("--vram-gb", type=float, default=16.0)
    ap.add_argument("--sequence-nodes", type=int, default=None,
                    help="nodes in a whole-sequence graph (default: the config's "
                         "max_sequence_nodes) - the IDR/MFI/proteomics path")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={
        "batch_size": args.batch_size, "window": args.window,
        "esm_model": args.esm_model, "device": args.device,
    })
    device = torch.device(args.device)
    esm_dim = ESM_DIMS.get(cfg.esm_model, 1280)
    if args.sequence_nodes is None:
        args.sequence_nodes = int(cfg.get("max_sequence_nodes", 2048))

    print("=" * 70)
    print("Holo-GeoN budget")
    print("=" * 70)
    print(f"encoder            : {cfg.esm_model}  (dim {esm_dim}, frozen)")
    print(f"window             : {cfg.window} residues")
    print(f"batch size         : {cfg.batch_size}")
    print(f"assumed VRAM       : {args.vram_gb:.0f} GB")
    print()

    # --- disk: encoder cache ------------------------------------------------
    cache_bytes = EncodingStore.projected_bytes(
        args.unique_residues, esm_dim, cfg.contact_top_k
    )
    print("--- encoder cache (one-time, parents only) ---")
    print(f"unique residues    : {args.unique_residues:,}")
    print(f"projected size     : {cache_bytes / GB:.1f} GB "
          f"(fp16 embeddings + sparse top-{cfg.contact_top_k} contacts)")
    dense = args.unique_residues * (esm_dim * 2)
    dense += args.unique_residues * 600 * 2   # a dense LxL map at L~600
    print(f"dense-map equivalent: {dense / GB:.1f} GB — the sparse edge list is "
          f"what keeps this on a consumer disk")
    print()

    # --- training memory ----------------------------------------------------
    from tests.helpers import make_batch   # synthetic batch, shapes only

    kwargs = model_kwargs_from_config(cfg, esm_dim)
    model = HoloGeoN(**kwargs).to(device)
    counts = model.parameter_count()

    print("--- trainable model ---")
    for name, n in counts.items():
        print(f"{name:<19}: {n:,}")
    print(f"{'params (fp32)':<19}: {_mib(counts['total'] * 4)}")
    print(f"{'+ AdamW states':<19}: {_mib(counts['total'] * 4 * 3)} "
          f"(weights + 2 moments)")
    print()

    batch = make_batch(dim=esm_dim, batch_size=min(cfg.batch_size, 16),
                       length=cfg.window, seed=0)
    node_bytes = batch["esm"].numel() * 4
    scale = cfg.batch_size / max(batch["esm"].shape[0], 1)
    print("--- one training batch ---")
    print(f"node features      : {_mib(node_bytes * scale)} "
          f"({cfg.batch_size} x {cfg.window} x {esm_dim} fp32)")
    print(f"E_geo edges        : ~{int(batch['edge_geo'].shape[1] * scale):,}")
    print(f"E_mech edges       : ~{int(batch['edge_mech'].shape[1] * scale):,}")

    # Analytic terms that dominate activation memory. Round-1 review, R1-16: the
    # previous "activations ~6x node features" constant had no relation to the
    # graph, and each of these two terms alone exceeded the 0.78 GB it produced.
    e_geo = int(batch["edge_geo"].shape[1] * scale)
    e_mech = int(batch["edge_mech"].shape[1] * scale)

    def gat_term(n_edges: int) -> float:
        """
        Bytes of per-edge GATv2 message tensors retained for backward.

        ``(E, heads, out_channels)`` fp32, in each of 4 convolutions (2 layers
        x 2 streams). Round-2 review, R2-02: this term carried a spurious
        ``/ 2``, halving it to 1.23 GB where the tensors actually kept are
        2.46 GB - matching round 1's independent 2.38 GB estimate. The edge
        lists here are already DIRECTED (both orientations are materialised by
        the graph builder), so there is no double count to divide out.
        """
        return 4.0 * n_edges * kwargs["gat_heads"] * cfg.hidden * 4

    def scan_term(nodes_per_batch: int, width: int) -> float:
        """
        Bytes of associative-scan buffers retained by autograd.

        a_prev/x_prev are allocated and a_c/x_c rewritten at each of
        ceil(log2 width) steps, all held for backward.
        """
        if not cfg.use_ssm:
            return 0.0
        steps = max(1, math.ceil(math.log2(max(width, 2))))
        return steps * 4.0 * nodes_per_batch * esm_dim * 4

    gat_bytes = gat_term(e_geo + e_mech)
    scan_steps = max(1, math.ceil(math.log2(max(cfg.window, 2))))
    scan_bytes = scan_term(cfg.batch_size * cfg.window, cfg.window)

    print(f"GATv2 edge tensors : {gat_bytes / GB:.2f} GB (4 convs, "
          f"{e_geo + e_mech:,} directed edges, {kwargs['gat_heads']} heads)")
    print(f"SSM scan buffers   : {scan_bytes / GB:.2f} GB ({scan_steps} steps)")
    print()

    # --- the whole-sequence path (R2-02) -----------------------------------
    # IDR / MFI / proteomics graphs span the entire protein, not a window. They
    # were not modelled here at all, and at one shared batch_size they cost 8x
    # the batch above - past the target card, part-way through epoch 1.
    seq_nodes = int(args.sequence_nodes)
    budget_nodes = int(cfg.get("sequence_node_budget") or cfg.batch_size * cfg.window)
    edges_per_node = (e_geo + e_mech) / max(cfg.batch_size * cfg.window, 1)
    print(f"--- one whole-sequence batch (max_sequence_nodes={seq_nodes}) ---")
    for label, nodes in (
        (f"UNCAPPED at batch_size {cfg.batch_size}", cfg.batch_size * seq_nodes),
        (f"node budget {budget_nodes:,}", budget_nodes),
    ):
        g = gat_term(int(edges_per_node * nodes))
        sc = scan_term(nodes, seq_nodes)
        nb = nodes * esm_dim * 4
        total = (g + sc + nb + counts["total"] * 16) / GB
        samples = max(1, nodes // seq_nodes)
        print(f"{label:<28}: {samples:>3} sample(s)/batch, "
              f"{nodes:>7,} padded nodes -> LOWER BOUND {total:6.2f} GB"
              + ("  <-- exceeds the card" if total > args.vram_gb else ""))
    print("  (the node-budget row is what src/data.py NodeBudgetBatchSampler")
    print("   enforces; the uncapped row is what a shared batch_size costs.)")
    print()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out, _ = model.ddg(batch)
        out.sum().backward()
        peak = torch.cuda.max_memory_allocated()
        print(f"MEASURED peak      : {_mib(peak)} at batch "
              f"{batch['esm'].shape[0]}")
        projected = peak * scale
        headroom = "fits" if projected < args.vram_gb * GB * 0.8 else "TIGHT"
        print(f"measured, scaled to {cfg.batch_size}: {projected / GB:.2f} GB "
              f"({headroom} in {args.vram_gb:.0f} GB)")
    else:
        out, _ = model.ddg(batch)
        out.sum().backward()
        floor = (gat_bytes + scan_bytes + node_bytes * scale
                 + counts["total"] * 16) / GB
        print()
        print("MEASURED peak      : UNAVAILABLE - no CUDA device here.")
        print("This script will not print an activation 'estimate' without a")
        print("torch.cuda.max_memory_allocated() measurement: the previous")
        print("constant-multiplier estimate under-stated peak by an order of")
        print("magnitude, and a hardware plan was built on it.")
        print()
        print(f"Analytic LOWER BOUND at {cfg.batch_size}: {floor:.2f} GB")
        print("  (edge tensors + scan buffers + node features + optimiser state;")
        print("   a lower bound, not a peak - real peak is higher.)")
        print(f"  Run this on the target GPU for the real number.")

    print()
    print("--- notes ---")
    print("* The encoder is frozen and never enters the training loop, so its")
    print("  1.3 GB of fp16 weights cost nothing during training. Only the")
    print("  precompute pass and the web interface load it.")
    print("* Blackwell (sm_120) needs the cu128 wheel:")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cu128")
    print("  Every stage prints the detected device and warns if the installed")
    print("  wheel cannot target the card (src/device.py).")


if __name__ == "__main__":
    main()
