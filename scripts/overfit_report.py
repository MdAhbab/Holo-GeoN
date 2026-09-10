"""
scripts/overfit_report.py
=========================
Did the trained model memorise its training set? Measure it directly.

The training loop records validation metrics per epoch, and the loss it prints is
a sum of three differently-scaled terms, one of them a Gaussian NLL that is
unbounded below. None of that answers the question. This scores the FINAL
selected checkpoint on the training split and the held-out splits with the same
code path, so train and test sit next to each other in one table.

Three things it reports that a plain train/test comparison does not:

* **A cluster bootstrap.** The stability test split is 33,226 rows but only 37
  distinct proteins, and rows from one protein are not independent. Resampling
  ROWS would give a confidence interval roughly sqrt(33226/37) ~ 30x too narrow
  and make every difference look significant. It resamples PROTEINS.
* **The train subsample is drawn to match the test set's size**, so the two
  Pearson values are computed on comparable n and the gap is not an artefact of
  one being noisier.
* **The uncertainty channel separately.** sigma is fit on training residuals, so
  `rmse / mean_predicted_sigma` diverging from 1.0 on held-out data is a
  memorisation signal that the mean channel alone can hide.

Usage
-----
python -m scripts.overfit_report
python -m scripts.overfit_report --n-train 30000 --bootstrap 2000
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
from src.data import MutationDataset, collate_graphs, move_batch
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore
from src.metrics import classification_metrics, regression_metrics
from src.model import HoloGeoN, model_kwargs_from_config
from src.seeding import seed_everything


@torch.no_grad()
def predict(model, loader, device, task: str):
    """Return (prediction, target, sigma, group) arrays for one split."""
    preds, targets, sigmas = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        if task == "stability":
            mu, logvar = model.ddg(batch)
            preds.append(mu.reshape(-1).float().cpu().numpy())
            sigmas.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
        else:
            logit = model.pathogenicity(batch)
            preds.append(torch.sigmoid(logit).reshape(-1).float().cpu().numpy())
            sigmas.append(np.zeros(logit.reshape(-1).shape[0], dtype=np.float32))
        targets.append(batch["y"].reshape(-1).float().cpu().numpy())
    if not preds:
        return (np.array([]),) * 3
    return (np.concatenate(preds), np.concatenate(targets), np.concatenate(sigmas))


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                      stat, n_boot: int, seed: int = 0) -> tuple[float, float]:
    """
    Percentile CI for `stat`, resampling GROUPS with replacement.

    Rows within a protein share its sequence, its embedding and its structure, so
    they are not independent draws. A row-level bootstrap would understate the
    interval by roughly sqrt(rows per group).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in picked])
        try:
            value = stat(y[idx], p[idx])
        except Exception:
            continue
        if np.isfinite(value):
            out.append(value)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=os.path.join("runs", "main"))
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--splits", default=os.path.join("data", "splits_megascale.parquet"))
    ap.add_argument("--clinvar-splits",
                    default=os.path.join("data", "splits_clinvar.parquet"))
    ap.add_argument("--n-train", type=int, default=33000,
                    help="training rows to score, matched to the test split size")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache, int(cfg.get("record_cache_size", 0)))

    ckpt = os.path.join(args.run, "model_best.pt")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.esm_model, 1280))
    kwargs = state.get("model_kwargs") or model_kwargs_from_config(
        state.get("config", {}), esm_dim)
    model = HoloGeoN(**kwargs).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[overfit] checkpoint {ckpt} (epoch {state.get('epoch')})")

    def loader_for(frame, target):
        return DataLoader(MutationDataset(frame, store, cfg, target=target),
                          batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=collate_graphs)

    report: dict = {"checkpoint": ckpt}

    # ---------------- stability ----------------
    sp = pd.read_parquet(args.splits)
    rows = {}
    for split in ("train", "val", "test"):
        part = sp[sp.split == split]
        if split == "train" and len(part) > args.n_train:
            part = part.sample(n=args.n_train, random_state=int(cfg.seed))
        rows[split] = part.reset_index(drop=True)

    print("\n" + "=" * 78)
    print("  STABILITY (ddG) — the same model on each split")
    print("=" * 78)
    head = (f"  {'split':<6} {'n':>7} {'proteins':>9} {'Pearson':>9} "
            f"{'95% CI (cluster)':>22} {'RMSE':>8} {'R2':>7} {'sig/err':>8}")
    print(head)
    stab: dict = {}
    for split, frame in rows.items():
        if frame.empty:
            continue
        p, y, sigma = predict(model, loader_for(frame, "target"), device, "stability")
        m = regression_metrics(y, p)
        groups = frame["parent_seq"].to_numpy()
        lo, hi = cluster_bootstrap(
            y, p, groups,
            lambda a, b: float(np.corrcoef(a, b)[0, 1]), args.bootstrap)
        ratio = float(m["rmse"] / np.mean(sigma)) if np.mean(sigma) else float("nan")
        stab[split] = {**m, "n_proteins": int(frame["parent_seq"].nunique()),
                       "pearson_ci95_cluster": [lo, hi],
                       "rmse_over_predicted_sigma": ratio}
        print(f"  {split:<6} {len(y):>7} {stab[split]['n_proteins']:>9} "
              f"{m['pearson']:>9.4f} {f'[{lo:.4f}, {hi:.4f}]':>22} "
              f"{m['rmse']:>8.4f} {m['r2']:>7.4f} {ratio:>7.2f}x")
    report["stability"] = stab

    # ---------------- pathogenicity ----------------
    if os.path.exists(args.clinvar_splits):
        cv = pd.read_parquet(args.clinvar_splits)
        print("\n" + "=" * 78)
        print("  PATHOGENICITY (ClinVar)")
        print("=" * 78)
        print(f"  {'split':<6} {'n':>7} {'genes':>9} {'AUROC':>9} "
              f"{'95% CI (cluster)':>22} {'AUPRC':>8}")
        patho: dict = {}
        from sklearn.metrics import roc_auc_score
        for split in ("train", "val", "test"):
            part = cv[cv.split == split]
            if part.empty:
                continue
            if len(part) > args.n_train:
                part = part.sample(n=args.n_train, random_state=int(cfg.seed))
            part = part.reset_index(drop=True)
            p, y, _ = predict(model, loader_for(part, "label"), device, "pathogenicity")
            m = classification_metrics(y, p)
            groups = part["group"].to_numpy()
            lo, hi = cluster_bootstrap(
                y, p, groups, lambda a, b: float(roc_auc_score(a, b)), args.bootstrap)
            patho[split] = {**m, "n_groups": int(part["group"].nunique()),
                            "auroc_ci95_cluster": [lo, hi]}
            print(f"  {split:<6} {len(y):>7} {patho[split]['n_groups']:>9} "
                  f"{m['auroc']:>9.4f} {f'[{lo:.4f}, {hi:.4f}]':>22} "
                  f"{m['auprc']:>8.4f}")
        report["pathogenicity"] = patho

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    tr, te = stab.get("train", {}), stab.get("test", {})
    if tr and te:
        d_r = tr["pearson"] - te["pearson"]
        d_rmse = te["rmse"] / tr["rmse"] if tr["rmse"] else float("nan")
        lo, hi = te["pearson_ci95_cluster"]
        overlaps = lo <= tr["pearson"] <= hi
        print(f"  train Pearson {tr['pearson']:.4f} vs test {te['pearson']:.4f}"
              f"   gap {d_r:+.4f}")
        print(f"  train RMSE    {tr['rmse']:.4f} vs test {te['rmse']:.4f}"
              f"   ratio {d_rmse:.2f}x")
        print(f"  test 95% cluster CI on Pearson: [{lo:.4f}, {hi:.4f}]"
              f"  -> train value {'INSIDE' if overlaps else 'OUTSIDE'} it")
        report["verdict"] = {
            "pearson_gap_train_minus_test": d_r,
            "rmse_ratio_test_over_train": d_rmse,
            "train_pearson_within_test_ci": bool(overlaps),
        }
    out = os.path.join(args.run, "overfit_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
