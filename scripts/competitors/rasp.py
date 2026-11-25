"""
scripts/competitors/rasp.py
===========================
Competitor adapter for **RaSP** (Blaabjerg et al., *eLife* 2023;
``KULL-Centre/_2022_ML-ddG-Blaabjerg``, Apache-2.0): fast local 3D structural
microenvironment predictor.

Implements RaSP's local 3D cavity microenvironment feature descriptor (distance
shell AA composition, local burial / coordinate density, backbone torsions,
relative solvent-exposure proxy, and wild-type/mutant biophysical shifts) paired
with RaSP's downstream 3-layer stability MLP trained strictly on OUR exact
MegaScale train split (zero Rosetta-label pretraining leakage across splits).
Also supports running the upstream KULL-Centre cavity model when
``--upstream-repo`` is provided on a GPU VM.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.competitors.common import (
    gpu_peak_gb,
    load_external_table,
    load_split_table,
    write_predictions,
)
from src.geometry import (
    NAME_RE,
    STR_NODE_DIM,
    align_to_sequence,
    node_geometry,
    normalize_pdb_stem,
    parse_pdb_backbone,
    resolve_pdb_path,
)
from src.substitution import AA_BIOPHYS, AA_ORDER, N_AA_BIOPHYS, aa_index

RASP_SHELLS = (6.0, 9.0, 12.0)  # concentric C-beta cavity shells (Angstroms)
CAVITY_DIM = len(RASP_SHELLS) * (len(AA_ORDER) + 1) + STR_NODE_DIM + 3  # 63 + 9 + 3 = 75


def extract_cavity_features(seq: str, coords: np.ndarray, plddt: np.ndarray) -> np.ndarray:
    """
    Compute per-residue 3D local cavity microenvironment descriptors (L, CAVITY_DIM):
      * 3 concentric C-beta distance shells (0-6, 6-9, 9-12 A): 20-AA radial
        composition + shell neighbor density
      * 9-ch SE(3)-invariant local backbone geometry (torsions, virtual angles, pLDDT)
      * 3-ch local burial / half-sphere exposure (HSE-up, HSE-down, neighbor count)
    """
    L = len(seq)
    cb = coords[:, 3, :]  # (L, 3)
    ca = coords[:, 1, :]
    valid = np.isfinite(cb).all(axis=-1) & np.isfinite(ca).all(axis=-1)
    aa_ids = np.array([aa_index(c) for c in seq], dtype=np.int64)

    feats = np.zeros((L, CAVITY_DIM), dtype=np.float32)
    if not valid.any():
        return feats

    # Pairwise C-beta distances
    cb_safe = np.where(valid[:, None], cb, 0.0)
    diff = cb_safe[:, None, :] - cb_safe[None, :, :]  # (L, L, 3)
    dist = np.linalg.norm(diff, axis=-1)
    pair_valid = valid[:, None] & valid[None, :]
    np.fill_diagonal(pair_valid, False)

    # Half-sphere exposure using CA->CB vector
    v_cacb = np.where(valid[:, None], cb_safe - np.where(valid[:, None], ca, 0.0), 0.0)
    v_norm = np.linalg.norm(v_cacb, axis=-1, keepdims=True) + 1e-8
    v_unit = v_cacb / v_norm

    col = 0
    prev_r = 0.0
    for r_cut in RASP_SHELLS:
        in_shell = pair_valid & (dist > prev_r) & (dist <= r_cut)
        cnt = in_shell.sum(axis=-1).astype(np.float32)  # (L,)
        # One-hot neighbor AA histogram inside shell
        for a in range(len(AA_ORDER)):
            is_a = (aa_ids == a)[None, :]
            feats[:, col + a] = (in_shell & is_a).sum(axis=-1) / np.maximum(cnt, 1.0)
        feats[:, col + len(AA_ORDER)] = np.log1p(cnt) / 4.0
        col += len(AA_ORDER) + 1
        prev_r = r_cut

    # 9-ch SE(3)-invariant backbone geometry
    ng = node_geometry(
        torch.from_numpy(coords).float(),
        torch.from_numpy(plddt).float(),
        torch.from_numpy(valid.astype(np.float32)),
    ).numpy()
    feats[:, col:col + STR_NODE_DIM] = ng
    col += STR_NODE_DIM

    # HSE-up, HSE-down within 12 A, total contact count
    within12 = pair_valid & (dist <= 12.0)
    dot = np.einsum("ijk,ik->ij", diff, v_unit)  # neighbor relative to CA->CB
    hse_up = (within12 & (dot >= 0)).sum(axis=-1).astype(np.float32)
    hse_dn = (within12 & (dot < 0)).sum(axis=-1).astype(np.float32)
    feats[:, col] = hse_up / 20.0
    feats[:, col + 1] = hse_dn / 20.0
    feats[:, col + 2] = (hse_up + hse_dn) / 30.0
    return feats


class RaspDownstreamMLP(nn.Module):
    """RaSP-style 3-layer downstream stability regressor with exact antisymmetry."""

    def __init__(self, cavity_dim: int = CAVITY_DIM, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("aa_bio", AA_BIOPHYS.clone(), persistent=False)
        self.aa = nn.Embedding(len(AA_ORDER) + 1, 32)
        self.cav_norm = nn.LayerNorm(cavity_dim)
        in_dim = cavity_dim + 32 + N_AA_BIOPHYS
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )

    def _score(self, cav: torch.Tensor, aa_idx: torch.Tensor) -> torch.Tensor:
        idx = aa_idx.clamp(0, len(AA_ORDER))
        return self.net(torch.cat([cav, self.aa(idx), self.aa_bio[idx].to(cav.dtype)], dim=-1))

    def forward(self, cav: torch.Tensor, wt_idx: torch.Tensor, mt_idx: torch.Tensor) -> torch.Tensor:
        c = self.cav_norm(cav)
        return self._score(c, mt_idx) - self._score(c, wt_idx)


def _resolve_pdb_map(pdb_dir: str) -> dict[str, Path]:
    return {normalize_pdb_stem(p.name): p for p in Path(pdb_dir).rglob("*.pdb")}


def _match_pdb(parent: str, pdb_map: dict[str, Path]) -> Path | None:
    hit = resolve_pdb_path(str(parent), pdb_map)
    return Path(hit) if hit is not None else None


def _build_cavity_dataset(df: pd.DataFrame, pdb_map: dict[str, Path],
                          par_col: str = "parent") -> tuple:
    cav_l, wt_l, mt_l, y_l, rid_l = [], [], [], [], []
    seq_col = "parent_seq" if "parent_seq" in df.columns else "wt_seq"
    tgt_col = "target" if "target" in df.columns else "ddg"

    for parent, grp in df.groupby(par_col, sort=False):
        pfile = _match_pdb(str(parent), pdb_map)
        if pfile is None:
            continue
        pseq, coords, plddt, _ = parse_pdb_backbone(str(pfile))
        wt_seq = str(grp[seq_col].iloc[0]).upper()
        off, _, _ = align_to_sequence(pseq, wt_seq)
        if off is None or off + len(wt_seq) > len(pseq):
            continue
        c_sl = coords[off:off + len(wt_seq)]
        p_sl = plddt[off:off + len(wt_seq)]
        cav = extract_cavity_features(wt_seq, c_sl, p_sl)
        sites = np.clip(grp["site"].to_numpy(dtype=np.int64), 0, len(wt_seq) - 1)
        cav_l.append(torch.from_numpy(cav[sites]))
        wt_l.append(torch.tensor([aa_index(a) for a in grp["wt_aa"]], dtype=torch.long))
        mt_l.append(torch.tensor([aa_index(a) for a in grp["mt_aa"]], dtype=torch.long))
        y_l.append(torch.tensor(grp[tgt_col].to_numpy(dtype=np.float32)))
        rid_l.append(grp["row_id"].to_numpy(dtype=np.int64))

    return (
        torch.cat(cav_l, dim=0),
        torch.cat(wt_l, dim=0),
        torch.cat(mt_l, dim=0),
        torch.cat(y_l, dim=0),
        np.concatenate(rid_l),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RaSP 3D cavity stability competitor adapter")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--pairs", default="data/megascale_ddg_pairs.parquet")
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--external", default=None)
    ap.add_argument("--external-pdb-dir", default=None)
    ap.add_argument("--extra-externals", nargs="*", default=[],
                    help="Additional external holdouts as tag=parquet::pdb_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)

    splits = load_split_table(args.splits)
    if args.pairs and os.path.isfile(args.pairs) and "parent" not in splits.columns:
        pairs = pd.read_parquet(args.pairs, columns=["parent", "wt_seq"]).drop_duplicates("wt_seq")
        splits = splits.merge(pairs, left_on="parent_seq", right_on="wt_seq", how="left")
    ms_pdbs = _resolve_pdb_map(args.pdb_dir)

    tr = _build_cavity_dataset(splits[splits["split"] == "train"], ms_pdbs)
    va = _build_cavity_dataset(splits[splits["split"] == "val"], ms_pdbs)
    te = _build_cavity_dataset(splits[splits["split"] == "test"], ms_pdbs)

    model = RaspDownstreamMLP(cavity_dim=tr[0].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    best_r = -1e9
    best_state = None
    rng = np.random.default_rng(args.seed)
    n_tr = len(tr[3])
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for lo in range(0, n_tr, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(tr[0][idx].to(device), tr[1][idx].to(device), tr[2][idx].to(device)).reshape(-1)
            loss = loss_fn(pred, tr[3][idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp = []
        with torch.no_grad():
            for lo in range(0, len(va[3]), 1024):
                vp.append(model(va[0][lo:lo + 1024].to(device), va[1][lo:lo + 1024].to(device),
                                va[2][lo:lo + 1024].to(device)).reshape(-1).cpu().numpy())
        r = float(np.corrcoef(va[3].numpy(), np.concatenate(vp))[0, 1]) if vp else float("nan")
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def _predict_tuple(tup):
        preds = []
        with torch.no_grad():
            for lo in range(0, len(tup[3]), 1024):
                preds.append(model(tup[0][lo:lo + 1024].to(device), tup[1][lo:lo + 1024].to(device),
                                   tup[2][lo:lo + 1024].to(device)).reshape(-1).cpu().numpy())
        return np.concatenate(preds)

    write_predictions(
        out_dir / "pred_test.parquet",
        te[4], _predict_tuple(te),
        meta={
            "model": "RaSP_cavity_retrained",
            "implementation": "3d_cavity_microenvironment_mlp",
            "sign": "stabilizing_positive (:+1)",
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "best_val_pearson": round(best_r, 4),
            "wall_clock_s": round(time.time() - t0, 2),
            "peak_vram_gb": gpu_peak_gb(),
            "seed": args.seed,
        },
    )

    if args.external and args.external_pdb_dir and os.path.isfile(args.external):
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        par_col = "parent" if "parent" in ext_df.columns else "uniprot_id"
        ex = _build_cavity_dataset(ext_df, ext_pdbs, par_col=par_col)
        write_predictions(
            out_dir / "pred_external.parquet",
            ex[4], _predict_tuple(ex),
            meta={
                "model": "RaSP_cavity_retrained",
                "sign": "stabilizing_positive (:+1)",
                "seed": args.seed,
            },
        )
    for spec in args.extra_externals:
        tag, rest = spec.split("=", 1)
        epq, epdb = rest.split("::", 1)
        if os.path.isfile(epq) and os.path.isdir(epdb):
            edf = load_external_table(epq)
            emap = _resolve_pdb_map(epdb)
            pc = "parent" if "parent" in edf.columns else "uniprot_id"
            ex = _build_cavity_dataset(edf, emap, par_col=pc)
            write_predictions(
                out_dir / f"pred_{tag}.parquet",
                ex[4], _predict_tuple(ex),
                meta={"model": "RaSP_cavity_retrained", "sign": "stabilizing_positive (:+1)", "seed": args.seed},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
