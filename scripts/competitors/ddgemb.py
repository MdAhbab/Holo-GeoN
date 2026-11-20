"""
scripts/competitors/ddgemb.py
=============================
Competitor adapter for **DDGemb** (Savojardo et al., *Bioinformatics* 2024):
structure-free sequence-only stability predictor operating on ESM-2 650M
embeddings via a Transformer encoder over site-centered context plus
substitution difference tokens.

Directly comparable to structure-free Holo-GeoN v2 on identical splits and
identical frozen ESM-2 650M cache, without requiring PDB coordinates or MSAs.
Supports single-mutation training and evaluation, exporting common-format
``(row_id, y_pred)`` Parquets for ``scripts/score_predictions.py``.
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
from src.encoder import EncodingStore
from src.substitution import AA_ORDER, aa_index


class DdgEmbTransformer(nn.Module):
    """
    DDGemb-style sequence-only Transformer encoder over local ESM-2 residue
    windows (radius ``R`` around the mutated site) + explicit WT/MT embeddings.
    """

    def __init__(self, esm_dim: int = 1280, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, radius: int = 15, dropout: float = 0.2):
        super().__init__()
        self.radius = radius
        win_len = 2 * radius + 1
        self.proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, d_model))
        self.pos_emb = nn.Parameter(torch.randn(1, win_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.aa = nn.Embedding(len(AA_ORDER) + 1, 64)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model + 64),
            nn.Linear(2 * d_model + 64, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, win_emb: torch.Tensor, win_mask: torch.Tensor,
                wt_idx: torch.Tensor, mt_idx: torch.Tensor) -> torch.Tensor:
        # win_emb: (B, 2R+1, esm_dim), win_mask: (B, 2R+1) True=valid
        h = self.proj(win_emb) + self.pos_emb
        h = self.encoder(h, src_key_padding_mask=~win_mask)
        h_center = h[:, self.radius, :]
        w = win_mask.float().unsqueeze(-1)
        h_pool = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        ctx = torch.cat([h_center, h_pool], dim=-1)
        s_mt = self.head(torch.cat([ctx, self.aa(mt_idx)], dim=-1))
        s_wt = self.head(torch.cat([ctx, self.aa(wt_idx)], dim=-1))
        return s_mt - s_wt


def _extract_windows(df: pd.DataFrame, store: EncodingStore, radius: int = 15) -> tuple:
    win_len = 2 * radius + 1
    emb_l, mask_l, wt_l, mt_l, y_l, rid_l = [], [], [], [], [], []
    seq_col = "parent_seq" if "parent_seq" in df.columns else "wt_seq"
    tgt_col = "target" if "target" in df.columns else "ddg"

    for seq, grp in df.groupby(seq_col, sort=False):
        seq_s = str(seq).upper()
        if not store.has(seq_s):
            continue
        emb = store.get(seq_s)["emb"].float()  # (L, D)
        L, D = emb.shape
        padded = torch.zeros((L + 2 * radius, D), dtype=torch.float32)
        padded[radius:radius + L] = emb
        pmask = torch.zeros((L + 2 * radius,), dtype=torch.bool)
        pmask[radius:radius + L] = True

        sites = np.clip(grp["site"].to_numpy(dtype=np.int64), 0, L - 1)
        offsets = torch.arange(win_len, dtype=torch.long).unsqueeze(0)
        idx = torch.from_numpy(sites).unsqueeze(1) + offsets  # (M, 2R+1)
        emb_l.append(padded[idx])
        mask_l.append(pmask[idx])
        wt_l.append(torch.tensor([aa_index(a) for a in grp["wt_aa"]], dtype=torch.long))
        mt_l.append(torch.tensor([aa_index(a) for a in grp["mt_aa"]], dtype=torch.long))
        y_l.append(torch.tensor(grp[tgt_col].to_numpy(dtype=np.float32)))
        rid_l.append(grp["row_id"].to_numpy(dtype=np.int64))

    return (
        torch.cat(emb_l, dim=0),
        torch.cat(mask_l, dim=0),
        torch.cat(wt_l, dim=0),
        torch.cat(mt_l, dim=0),
        torch.cat(y_l, dim=0),
        np.concatenate(rid_l),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DDGemb sequence-only Transformer competitor adapter")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--external", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = EncodingStore(args.cache, cache_records=512)
    out_dir = Path(args.out)

    splits = load_split_table(args.splits)
    tr = _extract_windows(splits[splits["split"] == "train"], store, args.radius)
    va = _extract_windows(splits[splits["split"] == "val"], store, args.radius)
    te = _extract_windows(splits[splits["split"] == "test"], store, args.radius)

    model = DdgEmbTransformer(esm_dim=tr[0].shape[-1], radius=args.radius).to(device)
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
            for lo in range(0, len(va[4]), 512):
                vp.append(model(va[0][lo:lo + 512].to(device), va[1][lo:lo + 512].to(device),
                                va[2][lo:lo + 512].to(device), va[3][lo:lo + 512].to(device))
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
            for lo in range(0, len(tup[4]), 512):
                preds.append(model(tup[0][lo:lo + 512].to(device), tup[1][lo:lo + 512].to(device),
                                   tup[2][lo:lo + 512].to(device), tup[3][lo:lo + 512].to(device))
                             .reshape(-1).cpu().numpy())
        return np.concatenate(preds)

    write_predictions(
        out_dir / "pred_test.parquet",
        te[5], _predict_tuple(te),
        meta={
            "model": "DDGemb_transformer_frozen_esm2",
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
        ex = _extract_windows(ext_df, store, args.radius)
        write_predictions(
            out_dir / "pred_external.parquet",
            ex[5], _predict_tuple(ex),
            meta={
                "model": "DDGemb_transformer_frozen_esm2",
                "sign": "stabilizing_positive (:+1)",
                "seed": args.seed,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
