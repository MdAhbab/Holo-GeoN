"""
scripts/precompute_embeddings.py
================================
The one heavy GPU pass: encode every unique **parent** protein once.

Each cache entry holds the frozen ESM-2 per-residue embeddings, a sparse
contact-head edge list, and the seven mechanistic channels. Everything
downstream reads the cache and never loads ESM-2, which is what lets the
trainable model use large batches on a 16 GB card.

Two things make the cache tractable:

  * **Parents only.** A mutant is a residue identity handed to the head, not a
    sequence to embed, so the cache is keyed by unique proteins rather than by
    unique mutants. On MegaScale that is ~10^3 domains instead of ~10^5 mutants.
  * **Sparse contacts.** Top-k neighbours per residue rather than a dense L x L
    map that would be thresholded away immediately.

Mechanistic channels are computed here too, not per training item: ViennaRNA
folding is far too slow to repeat every epoch, and folding energy depends only
on the coding sequence, so it is cached alongside the embeddings.

The projected cache size is printed before anything is written, and
``--max-cache-gb`` aborts rather than filling a disk mid-run.

Example
-------
python -m scripts.precompute_embeddings \
    --data data/megascale_ddg_pairs.parquet data/clinvar_protein_pairs.parquet \
    --esm-model facebook/esm2_t33_650M_UR50D --cache cache/esm2_650M --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config
from src.data import detect_columns
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore, EsmEncoder
from src.features import VIENNA_AVAILABLE, mechanistic_features


def collect_parents(paths: list[str], cfg) -> dict[str, str | None]:
    """
    Unique parent sequences across the given tables, mapped to their coding
    sequence where one is present.

    Only wild-type / reference sequences are collected: mutants never enter the
    cache.
    """
    parents: dict[str, str | None] = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"[precompute] skipping missing file: {path}")
            continue
        df = pd.read_parquet(path)
        key = os.path.splitext(os.path.basename(path))[0]
        col = detect_columns(df, cfg.column_overrides(key))
        if col["wt_seq"] is None:
            print(f"[precompute] {path}: no wild-type sequence column; skipped")
            continue
        seqs = df[col["wt_seq"]].astype(str).str.upper()
        cds = df[col["cds"]].astype(str) if col["cds"] else None
        for i, s in enumerate(seqs):
            if not s or s == "NAN" or len(s) > cfg.max_protein_length:
                continue
            dna = None
            if cds is not None:
                v = cds.iat[i]
                dna = v if v and v.lower() != "nan" else None
            # A real CDS wins over a None seen earlier for the same protein.
            if s not in parents or (parents[s] is None and dna):
                parents[s] = dna
    return parents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--contact-top-k", type=int, default=None)
    ap.add_argument("--contact-min-sep", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--max-cache-gb", type=float, default=None,
                    help="abort if the projected cache exceeds this many GB")
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        overrides={
            "esm_model": args.esm_model, "cache": args.cache, "device": args.device,
            "contact_top_k": args.contact_top_k, "contact_min_sep": args.contact_min_sep,
        },
    )

    parents = collect_parents(args.data, cfg)
    if args.max_seqs:
        parents = dict(list(parents.items())[: args.max_seqs])
    if not parents:
        raise SystemExit("no parent sequences found in the given tables")

    store = EncodingStore(cfg.cache)
    todo = {s: d for s, d in parents.items() if not store.has(s)}

    dim = ESM_DIMS.get(cfg.esm_model, 1280)
    residues = sum(len(s) for s in todo)
    projected = EncodingStore.projected_bytes(residues, dim, cfg.contact_top_k)
    projected_gb = projected / 1024 ** 3

    n_with_cds = sum(1 for d in parents.values() if d)
    print(f"[precompute] model={cfg.esm_model} (dim {dim})  cache={cfg.cache}")
    print(f"[precompute] unique parents={len(parents)}  to encode={len(todo)}  "
          f"residues={residues:,}")
    print(f"[precompute] parents with a real coding sequence: {n_with_cds}/{len(parents)}")
    if not VIENNA_AVAILABLE:
        print("[precompute] ViennaRNA not installed: mRNA/CAI channels stay 0 and "
              "mech_valid stays 0, so E_mech will be empty. Install ViennaRNA to "
              "enable the mechanistic stream.")
    print(f"[precompute] projected cache size: {projected_gb:.2f} GB")
    if args.max_cache_gb and projected_gb > args.max_cache_gb:
        raise SystemExit(
            f"projected cache {projected_gb:.2f} GB exceeds --max-cache-gb "
            f"{args.max_cache_gb}. Use a smaller --esm-model, or cap --max-seqs."
        )
    def record_manifest(encoded_ok: int, failed: int) -> None:
        """
        Write what this cache is, so downstream artifacts can cite it.

        ``cds_coverage`` is the number that matters most: the mechanistic stream
        is only fed for parents that carried a real transcript AND had ViennaRNA
        available to fold it. If it reads 0, E_mech is self-loops only and the
        ``no_mech_stream`` ablation is measuring nothing (R1-05).
        """
        active = n_with_cds if VIENNA_AVAILABLE else 0
        store.write_manifest({
            "esm_model": cfg.esm_model,
            "dim": dim,
            "contact_top_k": cfg.contact_top_k,
            "contact_min_sep": cfg.contact_min_sep,
            "parents_requested": len(parents),
            "parents_encoded_this_pass": encoded_ok,
            "encode_failures_this_pass": failed,
            "parents_with_cds_column": n_with_cds,
            "vienna_available": VIENNA_AVAILABLE,
            "cds_active": active,
            "cds_coverage": round(active / len(parents), 4) if parents else 0.0,
        })

    if not todo:
        print("[precompute] cache already complete.")
        record_manifest(0, 0)
        return

    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    encoder = EsmEncoder(cfg.esm_model, device=str(device))

    failures = 0
    encoded = 0
    for seq, dna in tqdm(list(todo.items()), desc="encoding"):
        try:
            rec = encoder.encode(seq, top_k=cfg.contact_top_k, min_sep=cfg.contact_min_sep)
            rec["mech"] = mechanistic_features(seq, dna, len(seq)).half()
            rec["has_cds"] = bool(dna) and VIENNA_AVAILABLE
            # The coding sequence travels with the record so a single-codon
            # substitution can rebuild a mutant's mechanistic channels from the
            # SAME transcript (scripts/calibration.py reversibility check).
            rec["cds"] = dna
            store.put(rec)
            encoded += 1
        except Exception as exc:  # keep going; a single bad row must not end the pass
            failures += 1
            print(f"  ! failed (len={len(seq)}): {type(exc).__name__}: {exc}")

    record_manifest(encoded, failures)
    print(f"[precompute] done. cache={cfg.cache}  failures={failures}")
    print(f"[precompute] manifest written: {store.manifest_path()}")
    if failures:
        # Round-1 review, R1-06: a swallowed encoding failure left a parent with
        # no cache entry, which downstream became a FileNotFoundError from inside
        # a DataLoader worker - possibly twenty epochs into a two-day run. The
        # pass reports its failures with a non-zero status so `run.py all` stops
        # here rather than training on a silently incomplete cache.
        raise SystemExit(
            f"{failures} sequence(s) failed to encode. Their parents have no "
            f"cache entry, so any row referencing them would be dropped (and "
            f"counted) at load time. Investigate the errors above, then re-run "
            f"precompute - it skips what is already cached."
        )


if __name__ == "__main__":
    main()
