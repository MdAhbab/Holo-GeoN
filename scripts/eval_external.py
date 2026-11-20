"""
scripts/eval_external.py
========================
Score a trained checkpoint on an INDEPENDENT stability table it never saw.

Why this stage exists
---------------------
The shipped training set is 190 identity clusters of 32-72-residue designed and
natural domains, all measured by one proteolysis assay in one publication. Every
in-run number - the test split, the baselines, the ablation grid - is computed
inside that world. A held-out split of it answers "does the model generalise to
held-out domains of this kind"; it cannot answer "does any of this transfer to
real, full-length proteins measured by classical biophysics", because no such
protein is anywhere in the run.

``data/fireprotdb_holdout.parquet`` is that second question: FireProtDB with the
604,920 MegaScale-sourced rows removed, leaving ~4.5k mutations over 182
proteins from 67 organisms at a median length of 228. It shares no measurement
with the training table by construction.

Two things this reports that a plain metric would hide
------------------------------------------------------
* **Rows dropped for want of a cached encoding.** An external table's proteins
  are not in the run's cache unless they were deliberately encoded, and quietly
  scoring the subset that happens to be cached would report a metric for a set
  nobody chose. The count is in the output.
* **The sign convention actually applied.** This table is
  destabilizing-positive while the training table is stabilizing-positive
  (see ``scripts/build_fireprot_table.py``); the flip comes from
  ``configs/data.yaml`` via the loader, and the applied sign is echoed into the
  report so the number cannot be read without it.

Usage
-----
python -m scripts.eval_external --checkpoint runs/main/model_best.pt
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

from src.config import load_config
from src.data import (
    MutationDataset,
    collate_graphs,
    filter_to_cache,
    load_mutation_table,
    move_batch,
)
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore
from src.metrics import format_report, regression_metrics
from src.model import HoloGeoN
from src.provenance import provenance
from src.seeding import seed_everything

try:                                    # same fallback calibration.py uses
    from scripts.train import model_kwargs_from_config
except Exception:                       # pragma: no cover
    model_kwargs_from_config = None


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus, sigmas, ys = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        ddg, logvar = model.ddg(batch)
        mus.append(ddg.reshape(-1).float().cpu().numpy())
        sigmas.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
        ys.append(batch["y"].reshape(-1).float().cpu().numpy())
    if not mus:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(mus), np.concatenate(sigmas), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.path.join("runs", "main", "model_best.pt"))
    ap.add_argument("--data", default=os.path.join("data", "fireprotdb_holdout.parquet"))
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join("runs", "main"))
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={"cache": args.cache, "device": args.device})
    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"[external] no checkpoint at {args.checkpoint}. Train first: "
            f"python run.py train"
        )
    os.makedirs(args.out, exist_ok=True)
    seed_everything(cfg.seed)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    key = os.path.splitext(os.path.basename(args.data))[0]
    overrides = cfg.column_overrides(key)
    raw_df = pd.read_parquet(args.data)
    raw_idx_map = len(raw_df)
    norm, report = load_mutation_table(
        args.data, kind="stability", overrides=overrides,
        max_protein_length=cfg.max_protein_length,
    )
    norm["row_id"] = np.arange(len(norm), dtype=np.int64)
    del raw_df, raw_idx_map
    print(report.summary())
    sign_applied = report.notes.get("ddg_sign", "stabilizing_positive")
    print(f"[external] ddg_sign applied: {sign_applied}  "
          f"(mean ddG after normalisation: {norm['target'].mean():+.3f})")
    if sign_applied == "stabilizing_positive" and norm["target"].mean() > 0:
        print("[external] WARNING: mean ddG is positive under the internal "
              "convention. Random substitutions are destabilising on average, "
              "so this table may need ddg_sign: destabilizing_positive under "
              f"the key {key!r} in configs/data.yaml.")

    n_before = len(norm)
    norm = filter_to_cache(norm, store, report)
    n_uncached = n_before - len(norm)
    if n_uncached:
        print(f"[external] {n_uncached}/{n_before} rows have no cached encoding "
              f"and are NOT scored. Encode them first:\n"
              f"    python -m scripts.precompute_embeddings --data {args.data} "
              f"--cache {cfg.cache} --device {cfg.device}")
    if norm.empty:
        raise SystemExit(
            "[external] no rows have a cached encoding; nothing to score.")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.esm_model, 1280))
    kwargs = state.get("model_kwargs")
    if kwargs is None and model_kwargs_from_config is not None:
        kwargs = model_kwargs_from_config(state.get("config", {}), esm_dim)
    model = HoloGeoN(**kwargs).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    loader = DataLoader(
        MutationDataset(norm, store, cfg, target="target"),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_graphs,
    )
    mu, sigma, y = predict(model, loader, device)
    metrics = regression_metrics(y, mu)

    # Also write common-format per-row predictions so scripts/score_predictions.py
    # can compute cluster-bootstrap CIs and paired competitor comparisons directly.
    if "row_id" in norm.columns:
        row_ids = norm["row_id"].to_numpy(dtype=np.int64)
    else:
        row_ids = np.arange(len(norm), dtype=np.int64)
    pred_path = os.path.join(args.out, f"pred_{key}.parquet")
    pd.DataFrame({
        "row_id": row_ids,
        "y_true": y.astype(np.float32),
        "y_pred": mu.astype(np.float32),
        "y_sigma": sigma.astype(np.float32),
    }).to_parquet(pred_path, index=False)

    out = {
        "provenance": provenance(),
        "checkpoint": args.checkpoint,
        "data": args.data,
        "predictions_parquet": pred_path,
        "what_this_measures": (
            "transfer to proteins the run never saw: FireProtDB with all "
            "MegaScale-sourced rows removed, so no measurement is shared with "
            "the training table"
        ),
        "ddg_sign_applied": sign_applied,
        "rows_scored": int(len(y)),
        "rows_without_cached_encoding": int(n_uncached),
        "distinct_proteins": int(norm["parent_seq"].nunique()),
        "load_report": report.as_dict(),
        "metrics": metrics,
        "predicted_sigma": {
            "mean": float(np.mean(sigma)) if len(sigma) else float("nan"),
            "median": float(np.median(sigma)) if len(sigma) else float("nan"),
        },
    }
    path = os.path.join(args.out, "external_fireprotdb.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + format_report(metrics, title="External test — FireProtDB (no MegaScale rows)"))
    print(f"\n[external] {len(y)} rows over {out['distinct_proteins']} proteins")
    print(f"[external] wrote {path}")


if __name__ == "__main__":
    main()
