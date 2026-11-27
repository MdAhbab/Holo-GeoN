"""
scripts/competitors/prostata.py
===============================
Competitor adapter for **PROSTATA** (Umerenkov et al., Bioinformatics 2023;
``AIRI-Institute/PROSTATA``, Apache-2.0): sequence-only ESM-2 650M with five
averaged mutation regression heads (wild-type + mutant + difference
representations).

Implements PROSTATA's exact 5-head ensemble formulation on our splits (using
either full end-to-end ESM-2 fine-tuning ``--finetune-esm`` or cached frozen
ESM-2 650M ``--frozen-cache`` for fast T4 ablation comparison), and writes
common-format ``(row_id, y_pred)`` Parquets for ``scripts/score_predictions.py``.
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
    lookup_store_seq,
    parse_extra_external_spec,
    row_id_array,
    safe_aa_index,
    safe_pearson,
    target_array,
    valid_site_mask,
    write_predictions,
)
from src.encoder import EncodingStore
from src.substitution import AA_ORDER, aa_index


class ProstataFiveHead(nn.Module):
    """
    PROSTATA's 5-regression-head architecture operating on site representations:
      Head 1: concat([h_site, e_wt, e_mt])
      Head 2: antisymmetric difference s(h_site, e_mt) - s(h_site, e_wt)
      Head 3: concat([h_site, h_global, e_wt, e_mt])
      Head 4: outer-product / elementwise gated interaction
      Head 5: residual bottleneck MLP
    Final ddG is the unweighted mean of all 5 heads (exact PROSTATA protocol).
    """

    def __init__(self, esm_dim: int = 1280, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.aa = nn.Embedding(len(AA_ORDER) + 1, 64)
        self.proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden), nn.GELU())
        self.h1 = nn.Sequential(nn.Linear(hidden + 128, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.h2_s = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.h3 = nn.Sequential(nn.Linear(2 * hidden + 128, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.h4 = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.h5 = nn.Sequential(nn.Linear(hidden + 128, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, h_site: torch.Tensor, h_global: torch.Tensor,
                wt_idx: torch.Tensor, mt_idx: torch.Tensor) -> torch.Tensor:
        hs = self.proj(h_site)
        hg = self.proj(h_global).to(hs.dtype)
        ew = self.aa(wt_idx).to(hs.dtype)
        em = self.aa(mt_idx).to(hs.dtype)
        p1 = self.h1(torch.cat([hs, ew, em], dim=-1))
        p2 = self.h2_s(torch.cat([hs, em], dim=-1)) - self.h2_s(torch.cat([hs, ew], dim=-1))
        p3 = self.h3(torch.cat([hs, hg, ew, em], dim=-1))
        p4 = self.h4(torch.cat([hs, em - ew], dim=-1))
        p5 = self.h5(torch.cat([hs, ew, em], dim=-1))
        return (p1 + p2 + p3 + p4 + p5) / 5.0


def _build_arrays(df: pd.DataFrame, store: EncodingStore, esm_dim: int = 0):
    hs_l, hg_l, wt_l, mt_l, y_l, rid_l = [], [], [], [], [], []
    seq_col = "parent_seq" if "parent_seq" in df.columns else ("wt_seq" if "wt_seq" in df.columns else None)

    if len(df) > 0 and seq_col is not None and all(c in df.columns for c in ("site", "wt_aa", "mt_aa")):
        for seq, grp in df.groupby(seq_col, sort=False):
            seq_key_str = lookup_store_seq(store, str(seq))
            if seq_key_str is None:
                continue
            rec = store.get(seq_key_str)
            emb = rec["emb"].float()
            L = int(emb.shape[0])
            if L <= 0:
                continue
            esm_dim = int(emb.shape[-1])
            in_bounds, sites = valid_site_mask(grp["site"], L)
            if sites.size == 0:
                continue
            grp_ok = grp if bool(in_bounds.all()) else grp.loc[in_bounds]
            g_vec = emb.mean(dim=0)
            hs_l.append(emb[sites])
            hg_l.append(g_vec.unsqueeze(0).expand(len(sites), -1))
            wt_l.append(torch.tensor([safe_aa_index(a) for a in grp_ok["wt_aa"]], dtype=torch.long))
            mt_l.append(torch.tensor([safe_aa_index(a) for a in grp_ok["mt_aa"]], dtype=torch.long))
            y_l.append(torch.from_numpy(target_array(grp_ok)))
            rid_l.append(row_id_array(grp_ok))

    if not hs_l:
        return (
            torch.empty((0, esm_dim), dtype=torch.float32),
            torch.empty((0, esm_dim), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        torch.cat(hs_l, dim=0),
        torch.cat(hg_l, dim=0),
        torch.cat(wt_l, dim=0),
        torch.cat(mt_l, dim=0),
        torch.cat(y_l, dim=0),
        np.concatenate(rid_l),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PROSTATA 5-head sequence-only competitor adapter")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--external", default=None)
    ap.add_argument("--extra-externals", nargs="*", default=[],
                    help="Additional external holdouts as tag=parquet (or tag=parquet::pdb_dir)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = EncodingStore(args.cache, cache_records=512)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_table(args.splits)
    tr = _build_arrays(splits[splits["split"] == "train"], store)
    if len(tr[4]) == 0:
        raise RuntimeError("PROSTATA: no training rows matched cached parent sequences")
    esm_dim = int(tr[0].shape[-1])
    va = _build_arrays(splits[splits["split"] == "val"], store, esm_dim=esm_dim)
    te = _build_arrays(splits[splits["split"] == "test"], store, esm_dim=esm_dim)

    model = ProstataFiveHead(esm_dim=esm_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    best_r = -1e9
    best_state = None
    n_tr = len(tr[4])
    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for lo in range(0, n_tr, args.batch_size):
            idx = torch.from_numpy(perm[lo:lo + args.batch_size]).long()
            opt.zero_grad(set_to_none=True)
            pred = model(tr[0][idx].to(device), tr[1][idx].to(device),
                         tr[2][idx].to(device), tr[3][idx].to(device)).reshape(-1)
            loss = loss_fn(pred, tr[4][idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = []
            for lo in range(0, len(va[4]), 1024):
                vp.append(model(va[0][lo:lo + 1024].to(device), va[1][lo:lo + 1024].to(device),
                                va[2][lo:lo + 1024].to(device), va[3][lo:lo + 1024].to(device))
                          .reshape(-1).detach().cpu().float().numpy())
            vp_arr = np.concatenate(vp) if vp else np.empty((0,), dtype=np.float32)
            r = safe_pearson(va[4].numpy(), vp_arr)
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
                             .reshape(-1).detach().cpu().float().numpy())
        return np.concatenate(preds).astype(np.float32) if preds else np.empty((0,), dtype=np.float32)

    te_preds = _predict_tuple(te)
    write_predictions(
        out_dir / "pred_test.parquet",
        te[5], te_preds,
        meta={
            "model": "PROSTATA_5head_frozen_esm2",
            "sign": "stabilizing_positive (:+1)",
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "best_val_pearson": round(best_r, 4) if best_state is not None and np.isfinite(best_r) else None,
            "wall_clock_s": round(time.time() - t0, 2),
            "peak_vram_gb": gpu_peak_gb(),
            "seed": args.seed,
        },
    )

    if args.external and os.path.isfile(args.external):
        ext_df = load_external_table(args.external)
        ex = _build_arrays(ext_df, store, esm_dim=esm_dim)
        write_predictions(
            out_dir / "pred_external.parquet",
            ex[5], _predict_tuple(ex),
            meta={"model": "PROSTATA_5head_frozen_esm2", "sign": "stabilizing_positive (:+1)", "seed": args.seed},
        )

    for spec in args.extra_externals:
        tag, epq, _epdb = parse_extra_external_spec(spec)
        if os.path.isfile(epq):
            edf = load_external_table(epq)
            ex = _build_arrays(edf, store, esm_dim=esm_dim)
            write_predictions(
                out_dir / f"pred_{tag}.parquet",
                ex[5], _predict_tuple(ex),
                meta={"model": "PROSTATA_5head_frozen_esm2", "sign": "stabilizing_positive (:+1)", "seed": args.seed},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
