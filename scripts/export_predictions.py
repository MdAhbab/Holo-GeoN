"""
scripts/export_predictions.py
=============================
Per-row predictions from a trained Holo-GeoN checkpoint, in the common format
scripts/score_predictions.py reads, so Holo-GeoN and every competitor are scored
by the SAME code on the SAME rows.

Output (parquet): row_id, y_pred, y_sigma. ``row_id`` is the row's position in
the splits table, so the truth, the protein, the cluster and the split never
have to be copied around and cannot drift.

    python -m scripts.export_predictions --checkpoint runs/full_s42/model_best.pt \
        --splits data/splits_megascale.parquet --split test --out runs/full_s42/pred_test.parquet

The sign is the internal one (positive = stabilising). Rows without a cached
encoding are dropped and counted, never imputed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import apply_checkpoint_graph_config, load_config
from src.data import LoadReport, MutationDataset, collate_graphs, filter_to_cache, move_batch
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore
from src.model import HoloGeoN, model_kwargs_from_config


@torch.no_grad()
def _predict(model, loader, device):
    mu, sig = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        ddg, logvar = model.ddg(batch)
        mu.append(ddg.reshape(-1).float().cpu().numpy())
        sig.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
    return np.concatenate(mu), np.concatenate(sig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={"cache": args.cache, "device": args.device})
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    df = pd.read_parquet(args.splits)
    df["row_id"] = np.arange(len(df))
    part = df[df["split"] == args.split].copy()
    n_before = len(part)
    part = filter_to_cache(part, store, LoadReport(source=args.splits, rows_in=n_before))
    part = part.sort_values("row_id")          # loader order == this order
    if part.empty:
        raise SystemExit("no rows with a cached encoding; run precompute first")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    apply_checkpoint_graph_config(cfg, state)
    esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.esm_model, 1280))
    kwargs = state.get("model_kwargs") or model_kwargs_from_config(state.get("config", {}), esm_dim)
    model = HoloGeoN(**kwargs).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    loader = DataLoader(MutationDataset(part, store, cfg, target="target"),
                        batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_graphs)
    mu, sigma = _predict(model, loader, device)
    if len(mu) != len(part):
        raise SystemExit(f"predicted {len(mu)} rows for {len(part)} inputs: refusing to "
                         f"write misaligned predictions")

    out = pd.DataFrame({"row_id": part["row_id"].to_numpy(), "y_pred": mu, "y_sigma": sigma})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)
    meta = {"checkpoint": args.checkpoint, "splits": args.splits, "split": args.split,
            "rows_written": int(len(out)), "rows_dropped_no_cache": int(n_before - len(part)),
            "sign": "positive = stabilising (internal convention)"}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[export] wrote {len(out):,} rows to {args.out}  (dropped {meta['rows_dropped_no_cache']})")


if __name__ == "__main__":
    main()
