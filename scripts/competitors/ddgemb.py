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
from dataclasses import dataclass
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


@dataclass
class _WindowIndex:
    """
    Compact parent-indexed storage for site-centered windows.

    Stores each unique parent embedding ONCE (padded by ``radius`` on both sides)
    in fp16 (~90 MB for all of MegaScale) and gathers ``(B, 2R+1, D)`` windows
    on-the-fly per minibatch instead of pre-materialising ``(N_mut, 2R+1, D)``
    (~48 GB RAM).
    """

    flat_emb: torch.Tensor    # (sum(L_i + 2R), D) float16
    flat_mask: torch.Tensor   # (sum(L_i + 2R),) bool
    start_idx: torch.Tensor   # (N,) int64
    wt_idx: torch.Tensor      # (N,) int64
    mt_idx: torch.Tensor      # (N,) int64
    y: torch.Tensor           # (N,) float32
    row_id: np.ndarray        # (N,) int64
    radius: int
    esm_dim: int

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def gather_batch(self, idx: torch.Tensor | np.ndarray | slice, device: torch.device):
        if isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx).long()
        starts = self.start_idx[idx]
        offsets = torch.arange(2 * self.radius + 1, dtype=torch.long).unsqueeze(0)
        g = starts.unsqueeze(1) + offsets
        win_emb = self.flat_emb[g].to(device=device, dtype=torch.float32)
        win_mask = self.flat_mask[g].to(device=device)
        return (
            win_emb,
            win_mask,
            self.wt_idx[idx].to(device=device),
            self.mt_idx[idx].to(device=device),
            self.y[idx].to(device=device),
        )


def _extract_windows(df: pd.DataFrame, store: EncodingStore, radius: int = 15) -> _WindowIndex:
    emb_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    start_l: list[torch.Tensor] = []
    wt_l: list[torch.Tensor] = []
    mt_l: list[torch.Tensor] = []
    y_l: list[torch.Tensor] = []
    rid_l: list[np.ndarray] = []
    seq_col = "parent_seq" if "parent_seq" in df.columns else "wt_seq"
    tgt_col = "target" if "target" in df.columns else "ddg"

    cursor = 0
    esm_dim = 0
    for seq, grp in df.groupby(seq_col, sort=False):
        seq_s = str(seq).upper()
        if not store.has(seq_s):
            continue
        emb = store.get(seq_s)["emb"].to(dtype=torch.float16)  # (L, D)
        L, D = emb.shape
        esm_dim = D
        padded = torch.zeros((L + 2 * radius, D), dtype=torch.float16)
        padded[radius:radius + L] = emb
        pmask = torch.zeros((L + 2 * radius,), dtype=torch.bool)
        pmask[radius:radius + L] = True

        sites = np.clip(grp["site"].to_numpy(dtype=np.int64), 0, L - 1)
        emb_chunks.append(padded)
        mask_chunks.append(pmask)
        start_l.append(torch.from_numpy(sites + cursor))
        wt_l.append(torch.tensor([aa_index(a) for a in grp["wt_aa"]], dtype=torch.long))
        mt_l.append(torch.tensor([aa_index(a) for a in grp["mt_aa"]], dtype=torch.long))
        y_l.append(torch.tensor(grp[tgt_col].to_numpy(dtype=np.float32)))
        rid_l.append(grp["row_id"].to_numpy(dtype=np.int64))
        cursor += L + 2 * radius

    if not emb_chunks:
        raise RuntimeError("DDGemb: no cached parent sequences found for table")

    return _WindowIndex(
        flat_emb=torch.cat(emb_chunks, dim=0),
        flat_mask=torch.cat(mask_chunks, dim=0),
        start_idx=torch.cat(start_l, dim=0),
        wt_idx=torch.cat(wt_l, dim=0),
        mt_idx=torch.cat(mt_l, dim=0),
        y=torch.cat(y_l, dim=0),
        row_id=np.concatenate(rid_l),
        radius=radius,
        esm_dim=esm_dim,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DDGemb sequence-only Transformer competitor adapter")
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--external", default=None)
    ap.add_argument("--extra-externals", nargs="*", default=[],
                    help="Additional external holdouts as tag=parquet")
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

    model = DdgEmbTransformer(esm_dim=tr.esm_dim, radius=args.radius).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    best_r = -1e9
    best_state = None
    rng = np.random.default_rng(args.seed)
    n_tr = len(tr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for lo in range(0, n_tr, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            opt.zero_grad(set_to_none=True)
            we, wm, wi, mi, yb = tr.gather_batch(idx, device)
            pred = model(we, wm, wi, mi).reshape(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp = []
        with torch.no_grad():
            for lo in range(0, len(va), 512):
                we, wm, wi, mi, _ = va.gather_batch(slice(lo, lo + 512), device)
                vp.append(model(we, wm, wi, mi).reshape(-1).cpu().numpy())
        r = float(np.corrcoef(va.y.numpy(), np.concatenate(vp))[0, 1]) if vp else float("nan")
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def _predict_index(widx: _WindowIndex):
        preds = []
        with torch.no_grad():
            for lo in range(0, len(widx), 512):
                we, wm, wi, mi, _ = widx.gather_batch(slice(lo, lo + 512), device)
                preds.append(model(we, wm, wi, mi).reshape(-1).cpu().numpy())
        return np.concatenate(preds)

    write_predictions(
        out_dir / "pred_test.parquet",
        te.row_id, _predict_index(te),
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
            ex.row_id, _predict_index(ex),
            meta={
                "model": "DDGemb_transformer_frozen_esm2",
                "sign": "stabilizing_positive (:+1)",
                "seed": args.seed,
            },
        )

    for spec in args.extra_externals:
        tag, epq = spec.split("=", 1)
        if os.path.isfile(epq):
            edf = load_external_table(epq)
            ex = _extract_windows(edf, store, args.radius)
            write_predictions(
                out_dir / f"pred_{tag}.parquet",
                ex.row_id, _predict_index(ex),
                meta={"model": "DDGemb_transformer_frozen_esm2", "sign": "stabilizing_positive (:+1)", "seed": args.seed},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
