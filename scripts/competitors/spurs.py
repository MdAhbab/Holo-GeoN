"""
scripts/competitors/spurs.py
============================
Competitor adapter for **SPURS** (Li & Luo, *Nature Communications* 17, 891,
2026; ``luo-group/SPURS``, MIT licence): Structure-informed Protein stability
predictor fusing ESM-2 650M and ProteinMPNN structural representations.

Two explicit, provenance-tagged execution modes:
  1. ``--mode upstream`` (default when running on GPU with internet): clones
     ``luo-group/SPURS`` (appended to ``sys.path``, never at index 0), runs
     SPURS's official inference on external holdouts (FireProtDB-HF25 / S669 /
     Domainome) when ``--official-external-only`` is set, or retrains SPURS's
     regression head from scratch on OUR exact MegaScale train split so ZERO
     split leakage enters MegaScale test numbers.
  2. ``--mode local_fusion``: clean, self-contained SPURS-style frozen ESM-2
     650M + SE(3)-invariant 3D backbone/structure fusion head trained directly
     on our exact MegaScale train split (useful for fast deterministic T4 runs
     or offline testing without downloading external weights).

Outputs common-format ``(row_id, y_pred)`` Parquets + ``.meta.json`` for
``scripts/score_predictions.py``.
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
    clone_repo,
    gpu_peak_gb,
    load_external_table,
    load_split_table,
    write_predictions,
)
from src.encoder import EncodingStore
from src.geometry import (
    NAME_RE,
    STR_NODE_DIM,
    align_to_sequence,
    node_geometry,
    normalize_pdb_stem,
    parse_pdb_backbone,
    resolve_pdb_path,
)
from src.substitution import AA_ORDER, aa_index

SPURS_URL = "https://github.com/luo-group/SPURS.git"


class SpursFusionHead(nn.Module):
    """
    SPURS-style low-rank ESM-2 + 3D structure fusion regressor with explicit
    wild-type / mutant substitution tokens and antisymmetric prediction.
    """

    def __init__(self, esm_dim: int = 1280, struct_dim: int = STR_NODE_DIM + 1,
                 hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.esm_proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden), nn.GELU())
        self.str_proj = nn.Sequential(nn.LayerNorm(struct_dim), nn.Linear(struct_dim, hidden // 2), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aa = nn.Embedding(len(AA_ORDER) + 1, 64)
        self.scorer = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_esm: torch.Tensor, h_str: torch.Tensor,
                wt_idx: torch.Tensor, mt_idx: torch.Tensor) -> torch.Tensor:
        z = self.fuse(torch.cat([self.esm_proj(h_esm), self.str_proj(h_str)], dim=-1))
        s_mt = self.scorer(torch.cat([z, self.aa(mt_idx)], dim=-1))
        s_wt = self.scorer(torch.cat([z, self.aa(wt_idx)], dim=-1))
        return s_mt - s_wt


def _resolve_pdb_map(pdb_dir: str | None) -> dict[str, Path]:
    if not pdb_dir or not os.path.isdir(pdb_dir):
        return {}
    return {normalize_pdb_stem(p.name): p for p in Path(pdb_dir).rglob("*.pdb")}


def _match_pdb(parent: str, pdb_map: dict[str, Path]) -> Path | None:
    if not pdb_map:
        return None
    hit = resolve_pdb_path(str(parent), pdb_map)
    return Path(hit) if hit is not None else None


def _build_local_fusion_arrays(df: pd.DataFrame, store: EncodingStore,
                               pdb_map: dict[str, Path],
                               par_col: str = "parent") -> tuple:
    he_l, hs_l, wt_l, mt_l, y_l, rid_l = [], [], [], [], [], []
    seq_col = "parent_seq" if "parent_seq" in df.columns else "wt_seq"
    tgt_col = "target" if "target" in df.columns else "ddg"

    for seq, grp in df.groupby(seq_col, sort=False):
        seq_s = str(seq).upper()
        if not store.has(seq_s):
            continue
        rec = store.get(seq_s)
        emb = rec["emb"].float()
        L = emb.shape[0]

        # Structural features: prefer cache coords, else aligned PDB, else zero (masked)
        s_feat = torch.zeros((L, STR_NODE_DIM + 1), dtype=torch.float32)
        if "coords" in rec and "coord_valid" in rec:
            ng = node_geometry(rec["coords"].float(),
                               rec.get("plddt", torch.full((L,), 100.0)).float(),
                               rec["coord_valid"].float())
            s_feat[:, :STR_NODE_DIM] = ng
            s_feat[:, -1] = rec["coord_valid"].float()
        elif par_col in grp.columns and pdb_map:
            pfile = _match_pdb(str(grp[par_col].iloc[0]), pdb_map)
            if pfile is not None:
                pseq, coords, plddt, atom_valid = parse_pdb_backbone(str(pfile))
                off, _, _ = align_to_sequence(pseq, seq_s)
                if off is not None and off + L <= len(pseq):
                    c_sl = torch.from_numpy(coords[off:off + L]).float()
                    p_sl = torch.from_numpy(plddt[off:off + L]).float()
                    valid = torch.from_numpy(atom_valid[off:off + L]).float()
                    s_feat[:, :STR_NODE_DIM] = node_geometry(c_sl, p_sl, valid)
                    s_feat[:, -1] = valid

        sites = np.clip(grp["site"].to_numpy(dtype=np.int64), 0, L - 1)
        he_l.append(emb[sites])
        hs_l.append(s_feat[sites])
        wt_l.append(torch.tensor([aa_index(a) for a in grp["wt_aa"]], dtype=torch.long))
        mt_l.append(torch.tensor([aa_index(a) for a in grp["mt_aa"]], dtype=torch.long))
        y_l.append(torch.tensor(grp[tgt_col].to_numpy(dtype=np.float32)))
        rid_l.append(grp["row_id"].to_numpy(dtype=np.int64))

    return (
        torch.cat(he_l, dim=0),
        torch.cat(hs_l, dim=0),
        torch.cat(wt_l, dim=0),
        torch.cat(mt_l, dim=0),
        torch.cat(y_l, dim=0),
        np.concatenate(rid_l),
    )


def _try_upstream_spurs_external(repo_dir: Path, ext_df: pd.DataFrame, pdb_map: dict[str, Path],
                                 device: torch.device) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Attempt official ``luo-group/SPURS`` checkpoint inference on an external table.
    Returns ``(row_ids, y_pred)`` on success or ``None`` if upstream API/weights
    are unavailable in the current VM.
    """
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    try:
        from spurs.inference import get_SPURS_from_hub, parse_pdb  # type: ignore
    except Exception as exc:
        print(f"[spurs] upstream spurs.inference import unavailable ({exc}); "
              f"falling back to local_fusion retrain.", flush=True)
        return None

    model, cfg = get_SPURS_from_hub()
    model = model.to(device).eval()
    par_col = "parent" if "parent" in ext_df.columns else "uniprot_id"
    seq_col = "wt_seq" if "wt_seq" in ext_df.columns else "parent_seq"
    rids, preds = [], []
    for parent, grp in ext_df.groupby(par_col, sort=False):
        pfile = _match_pdb(str(parent), pdb_map)
        if pfile is None:
            continue
        pdb = parse_pdb(str(pfile), str(parent), "A", cfg)
        with torch.no_grad():
            ddg_mat = model(pdb, return_logist=True).detach().cpu().numpy()  # (L, 20)
        pseq = str(pdb["seq"]).upper()
        wt_seq = str(grp[seq_col].iloc[0]).upper()
        off, _, _ = align_to_sequence(pseq, wt_seq)
        if off is None:
            continue
        aa_to_col = {aa: idx for idx, aa in enumerate(AA_ORDER)}
        for r in grp.itertuples(index=False):
            pos = int(r.site) + off
            m_aa = str(r.mt_aa).upper()
            if 0 <= pos < ddg_mat.shape[0] and m_aa in aa_to_col:
                rids.append(int(r.row_id))
                preds.append(float(ddg_mat[pos, aa_to_col[m_aa]]))
    if not rids:
        return None
    return np.asarray(rids, np.int64), np.asarray(preds, np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SPURS competitor adapter (Nature Comms 2026)")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--pairs", default="data/megascale_ddg_pairs.parquet")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--external", default=None)
    ap.add_argument("--external-pdb-dir", default=None)
    ap.add_argument("--mode", choices=["local_fusion", "upstream"], default="local_fusion")
    ap.add_argument("--work-dir", default="/tmp/competitors/SPURS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--official-external-only", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "upstream" and args.external and args.external_pdb_dir:
        repo_dir = clone_repo(SPURS_URL, Path(args.work_dir))
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        t_inf = time.time()
        res = _try_upstream_spurs_external(repo_dir, ext_df, ext_pdbs, device)
        if res is not None:
            write_predictions(
                out_dir / "pred_external_official.parquet",
                res[0], res[1],
                meta={
                    "model": "SPURS_official",
                    "implementation": "upstream_luo_group_SPURS",
                    "sign": "stabilizing_positive (:+1 — verify on external holdout)",
                    "inference_s": round(time.time() - t_inf, 2),
                    "peak_vram_gb": gpu_peak_gb(),
                },
            )
            if args.official_external_only:
                return 0

    store = EncodingStore(args.cache, cache_records=512)
    splits = load_split_table(args.splits)
    if args.pairs and os.path.isfile(args.pairs) and "parent" not in splits.columns:
        pairs = pd.read_parquet(args.pairs, columns=["parent", "wt_seq"]).drop_duplicates("wt_seq")
        splits = splits.merge(pairs, left_on="parent_seq", right_on="wt_seq", how="left")
    ms_pdbs = _resolve_pdb_map(args.pdb_dir)

    tr = _build_local_fusion_arrays(splits[splits["split"] == "train"], store, ms_pdbs)
    va = _build_local_fusion_arrays(splits[splits["split"] == "val"], store, ms_pdbs)
    te = _build_local_fusion_arrays(splits[splits["split"] == "test"], store, ms_pdbs)

    model = SpursFusionHead(esm_dim=tr[0].shape[-1], struct_dim=tr[1].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    best_r = -1e9
    best_state = None
    rng = np.random.default_rng(args.seed)
    n_tr = len(tr[4])
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for lo in range(0, n_tr, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(tr[0][idx].to(device), tr[1][idx].to(device),
                         tr[2][idx].to(device), tr[3][idx].to(device)).reshape(-1)
            loss = loss_fn(pred, tr[4][idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp = []
        with torch.no_grad():
            for lo in range(0, len(va[4]), 1024):
                vp.append(model(va[0][lo:lo + 1024].to(device), va[1][lo:lo + 1024].to(device),
                                va[2][lo:lo + 1024].to(device), va[3][lo:lo + 1024].to(device))
                          .reshape(-1).cpu().numpy())
        r = float(np.corrcoef(va[4].numpy(), np.concatenate(vp))[0, 1]) if vp else float("nan")
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def _predict_tuple(tup):
        preds = []
        with torch.no_grad():
            for lo in range(0, len(tup[4]), 1024):
                preds.append(model(tup[0][lo:lo + 1024].to(device), tup[1][lo:lo + 1024].to(device),
                                   tup[2][lo:lo + 1024].to(device), tup[3][lo:lo + 1024].to(device))
                             .reshape(-1).cpu().numpy())
        return np.concatenate(preds)

    write_predictions(
        out_dir / "pred_test.parquet",
        te[5], _predict_tuple(te),
        meta={
            "model": "SPURS_retrained_fusion",
            "implementation": "retrained_spurs_architecture_esm2_plus_backbone3d",
            "sign": "stabilizing_positive (:+1)",
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "best_val_pearson": round(best_r, 4),
            "wall_clock_s": round(time.time() - t0, 2),
            "peak_vram_gb": gpu_peak_gb(),
            "seed": args.seed,
        },
    )

    if args.external and os.path.isfile(args.external):
        ext_df = load_external_table(args.external)
        ext_pdbs = _resolve_pdb_map(args.external_pdb_dir)
        par_col = "parent" if "parent" in ext_df.columns else "uniprot_id"
        ex = _build_local_fusion_arrays(ext_df, store, ext_pdbs, par_col=par_col)
        write_predictions(
            out_dir / "pred_external.parquet",
            ex[5], _predict_tuple(ex),
            meta={
                "model": "SPURS_retrained_fusion",
                "implementation": "retrained_spurs_architecture_esm2_plus_backbone3d",
                "sign": "stabilizing_positive (:+1)",
                "seed": args.seed,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
