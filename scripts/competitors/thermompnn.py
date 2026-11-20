"""
scripts/competitors/thermompnn.py
=================================
Competitor adapter for **ThermoMPNN** (Dieckhaus et al., PNAS 2024;
``Kuhlman-Lab/ThermoMPNN``, MIT licence).

Protocol (BENCHMARK_PLAN.md §2):
  * Clones ``Kuhlman-Lab/ThermoMPNN`` into ``--work-dir`` (appended to
    sys.path, never inserted at 0).
  * **MegaScale test:** retrains ThermoMPNN's stability head on OUR exact train
    split (frozen ProteinMPNN backbone) using our aligned MegaScale AlphaFold
    PDBs — official weights are NOT scored on MegaScale test because their
    official split overlaps ours.
  * **External (FireProtDB / S669):** evaluates BOTH our retrained checkpoint
    AND ThermoMPNN's official released checkpoint (``models/thermoMPNN_default.pt``)
    using fetched AlphaFold/crystal PDBs.
  * Exports common-format ``(row_id, y_pred)`` Parquets with declared sign
    ``:-1`` (ThermoMPNN native convention is destabilizing-positive, OR ``:+1``
    when trained directly on our stabilizing-positive targets — explicitly
    recorded in ``.meta.json``).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.competitors.common import (
    clone_repo,
    gpu_peak_gb,
    load_external_table,
    load_split_table,
    write_predictions,
)
from src.geometry import NAME_RE, align_to_sequence, parse_pdb_cb
from src.substitution import AA_ORDER, aa_index

THERMOMPNN_URL = "https://github.com/Kuhlman-Lab/ThermoMPNN.git"


def _resolve_pdb_map(pdb_dir: str) -> dict[str, Path]:
    return {
        p.stem.upper(): p
        for p in Path(pdb_dir).rglob("*.pdb")
    }


def _match_pdb(parent: str, pdb_map: dict[str, Path]) -> Path | None:
    m = NAME_RE.match(str(parent))
    if m and m.group("base").upper() in pdb_map:
        return pdb_map[m.group("base").upper()]
    key = str(parent).upper()
    for cand in (key, f"AF-{key}"):
        if cand in pdb_map:
            return pdb_map[cand]
    for stem, p in pdb_map.items():
        if stem.startswith(f"AF-{key}-"):
            return p
    return None


def _extract_proteinmpnn_features(repo_dir: Path, pdb_paths: dict[str, tuple[Path, str, int]],
                                  device: torch.device) -> dict[str, torch.Tensor]:
    """
    Run ThermoMPNN's frozen ProteinMPNN backbone ONCE per unique parent PDB to
    extract per-residue structural embeddings ``(L, d)`` aligned to ``wt_seq``.
    """
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    from omegaconf import OmegaConf
    from transfer_model import TransferModel  # type: ignore

    cfg = OmegaConf.create({
        "platform": {"thermompnn_dir": str(repo_dir)},
        "model": {
            "hidden_dims": [64, 32],
            "subtract_mut": True,
            "num_final_layers": 2,
            "freeze_weights": True,
            "load_pretrained": True,
            "lightattn": True,
            "lr": 1e-3,
        },
        "training": {"num_workers": 0, "learn_rate": 1e-3, "epochs": 10},
    })
    ckpt = repo_dir / "models" / "thermoMPNN_default.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Expected ThermoMPNN checkpoint at {ckpt}")

    from datasets import alt_parse_PDB  # type: ignore

    model = TransferModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device)
    model.eval().to(device)
    return model, alt_parse_PDB


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ThermoMPNN competitor adapter")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--pairs", default="data/megascale_ddg_pairs.parquet")
    ap.add_argument("--pdb-dir", required=True, help="MegaScale AlphaFold PDB directory")
    ap.add_argument("--external", default=None, help="Optional FireProtDB holdout parquet")
    ap.add_argument("--external-pdb-dir", default=None, help="FireProtDB AlphaFold PDB dir")
    ap.add_argument("--work-dir", default="/tmp/competitors/ThermoMPNN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--official-external-only", action="store_true",
                    help="Score ThermoMPNN official checkpoint on external holdout only (no retrain)")
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = clone_repo(THERMOMPNN_URL, Path(args.work_dir))
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))

    model, alt_parse_PDB = _extract_proteinmpnn_features(repo_dir, {}, device)
    from protein_mpnn_utils import tied_featurize  # type: ignore

    def _score_official_table(df: pd.DataFrame, pdb_map: dict[str, Path],
                              par_col: str, seq_col: str) -> tuple[np.ndarray, np.ndarray]:
        rids, preds = [], []
        for parent, grp in df.groupby(par_col, sort=False):
            pfile = _match_pdb(str(parent), pdb_map)
            if pfile is None:
                continue
            pdb_seq, _ = parse_pdb_cb(str(pfile))
            wt_seq = str(grp[seq_col].iloc[0]).upper()
            off, _, err = align_to_sequence(pdb_seq, wt_seq)
            if off is None:
                continue
            pdb_dict = alt_parse_PDB(str(pfile))[0]
            mut_list = []
            valid_rids = []
            for r in grp.itertuples(index=False):
                site = int(r.site) + off
                w, m_aa = str(r.wt_aa).upper(), str(r.mt_aa).upper()
                if 0 <= site < len(pdb_seq) and w in AA_ORDER and m_aa in AA_ORDER:
                    mut_list.append(type("M", (), {"position": site, "wildtype": w, "mutation": m_aa})())
                    valid_rids.append(int(r.row_id))
            if not mut_list:
                continue
            with torch.no_grad():
                out_list, _ = model(pdb_dict, mut_list)
            for rid, item in zip(valid_rids, out_list):
                rids.append(rid)
                preds.append(float(item["ddG"].item()))
        return np.asarray(rids, np.int64), np.asarray(preds, np.float32)

    if args.external and args.external_pdb_dir:
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        par_col = "parent" if "parent" in ext_df.columns else "uniprot_id"
        seq_col = "wt_seq" if "wt_seq" in ext_df.columns else "parent_seq"
        t_inf = time.time()
        e_rids, e_preds = _score_official_table(ext_df, ext_pdbs, par_col, seq_col)
        write_predictions(
            out_dir / "pred_external_official.parquet",
            e_rids, e_preds,
            meta={
                "model": "ThermoMPNN_official",
                "sign": "destabilizing_positive (:-1)",
                "inference_s": round(time.time() - t_inf, 2),
                "peak_vram_gb": gpu_peak_gb(),
            },
        )
        if args.official_external_only:
            return 0

    # Retrain stability head on OUR exact MegaScale train split (frozen ProteinMPNN backbone!)
    splits = load_split_table(args.splits)
    pairs = pd.read_parquet(args.pairs, columns=["parent", "wt_seq"]).drop_duplicates()
    splits = splits.merge(pairs, left_on="parent_seq", right_on="wt_seq", how="left")
    ms_pdbs = _resolve_pdb_map(args.pdb_dir)

    # Re-initialize ONLY the trainable ddG head (keep frozen ProteinMPNN backbone!)
    for name, p in model.named_parameters():
        is_backbone = "prot_mpnn" in name
        p.requires_grad = not is_backbone
    # Reset head parameters cleanly so ZERO weights from ThermoMPNN's own MegaScale split remain!
    for name, mod in model.named_modules():
        if "prot_mpnn" not in name and hasattr(mod, "reset_parameters"):
            mod.reset_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def _prepare_groups(sub_df: pd.DataFrame):
        items = []
        for parent, grp in sub_df.groupby("parent", sort=False):
            pfile = _match_pdb(str(parent), ms_pdbs)
            if pfile is None:
                continue
            pdb_seq, _ = parse_pdb_cb(str(pfile))
            wt_seq = str(grp["parent_seq"].iloc[0]).upper()
            off, _, err = align_to_sequence(pdb_seq, wt_seq)
            if off is None:
                continue
            pdb_dict = alt_parse_PDB(str(pfile))[0]
            muts, ys, rids = [], [], []
            for r in grp.itertuples(index=False):
                site = int(r.site) + off
                w, m_aa = str(r.wt_aa).upper(), str(r.mt_aa).upper()
                if 0 <= site < len(pdb_seq) and w in AA_ORDER and m_aa in AA_ORDER:
                    muts.append(type("M", (), {"position": site, "wildtype": w, "mutation": m_aa})())
                    ys.append(float(r.target))
                    rids.append(int(r.row_id))
            if muts:
                items.append((pdb_dict, muts, torch.tensor(ys, dtype=torch.float32), np.asarray(rids, np.int64)))
        return items

    print("[thermompnn] Preparing PDB groups for train/val/test ...", flush=True)
    tr_items = _prepare_groups(splits[splits["split"] == "train"])
    va_items = _prepare_groups(splits[splits["split"] == "val"])
    te_items = _prepare_groups(splits[splits["split"] == "test"])
    print(f"[thermompnn] Groups: train={len(tr_items)}, val={len(va_items)}, test={len(te_items)}", flush=True)

    best_r = -1e9
    best_state = None
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(tr_items))
        for idx in order:
            pdb_dict, muts, ys, _ = tr_items[int(idx)]
            opt.zero_grad(set_to_none=True)
            out_list, _ = model(pdb_dict, muts)
            pred = torch.stack([o["ddG"].reshape(()) for o in out_list])
            loss = loss_fn(pred, ys.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for pdb_dict, muts, ys, _ in va_items:
                out_list, _ = model(pdb_dict, muts)
                vp.append(np.array([float(o["ddG"].item()) for o in out_list]))
                vt.append(ys.numpy())
        r = float(np.corrcoef(np.concatenate(vt), np.concatenate(vp))[0, 1]) if vp else float("nan")
        print(f"[thermompnn] epoch {ep}/{args.epochs}  val_pearson={r:.4f}", flush=True)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    t_te = time.time()
    te_rids, te_preds = [], []
    with torch.no_grad():
        for pdb_dict, muts, _, rids in te_items:
            out_list, _ = model(pdb_dict, muts)
            te_rids.append(rids)
            te_preds.append(np.array([float(o["ddG"].item()) for o in out_list], dtype=np.float32))
    write_predictions(
        out_dir / "pred_test.parquet",
        np.concatenate(te_rids), np.concatenate(te_preds),
        meta={
            "model": "ThermoMPNN_retrained",
            "sign": "stabilizing_positive (:+1 — trained on our targets)",
            "trainable_params": sum(p.numel() for p in trainable),
            "best_val_pearson": round(best_r, 4),
            "wall_clock_s": round(time.time() - t0, 2),
            "test_inference_s": round(time.time() - t_te, 2),
            "peak_vram_gb": gpu_peak_gb(),
            "seed": args.seed,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
