"""
scripts/baselines.py
====================
Controls, scored on the identical split the model is scored on.

v1 reported one baseline: ESM zero-shot (|r| = 0.281) against the trained model
(r = 0.527). That comparison is supervised-versus-unsupervised — it shows that
training helps, which nobody doubts. It cannot answer the question a reviewer
actually asks: **what does the graph buy over an MLP on the same embeddings?**
Nothing in v1 could answer it.

Three controls, in increasing strength:

1. ``substitution_frequency`` — predicts from the amino-acid pair alone
   (mean target per wt->mt substitution type, fit on train). Anything that does
   not beat this has learned no protein context at all.
2. ``esm_zero_shot`` — the wild-type-marginal masked-LM score
   ``log p(mt | context) - log p(wt | context)`` (Meier et al., 2021). Training-free.
   Higher score means more tolerated, so the expected correlation with a
   destabilising ddG is negative; both signed and absolute r are reported.
3. ``esm_mlp`` — **the control that matters.** A supervised MLP on the same
   cached embeddings the model consumes: mean-pooled window plus the embedding
   at the mutated site plus the two residue identities. Same split, same target,
   same budget. The gap between this and the full model is the graph's
   contribution, and it is the number the architecture has to earn.

Example
-------
python -m scripts.baselines --splits data/splits_megascale.parquet \
    --cache cache/esm2_650M --out runs/baselines
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config
from src.device import describe_device
from src.encoder import EncodingStore, EsmEncoder
from src.metrics import classification_metrics, regression_metrics
from src.substitution import AA_ORDER, aa_index
from src.windowing import window_bounds


# ---------------------------------------------------------------------------
# 1. Substitution-frequency control
# ---------------------------------------------------------------------------
def substitution_frequency_baseline(train: pd.DataFrame, test: pd.DataFrame,
                                    target: str) -> dict:
    """Per-substitution-type mean fitted on train, applied to test."""
    key = train["wt_aa"].astype(str) + ">" + train["mt_aa"].astype(str)
    table = train.groupby(key)[target].mean()
    grand = float(train[target].mean())
    test_key = test["wt_aa"].astype(str) + ">" + test["mt_aa"].astype(str)
    pred = test_key.map(table).fillna(grand).to_numpy()
    truth = test[target].to_numpy(dtype=float)
    metrics = (classification_metrics(truth, pred) if target == "label"
               else regression_metrics(truth, pred))
    metrics["n"] = int(len(truth))
    return metrics


# ---------------------------------------------------------------------------
# 2. ESM zero-shot
# ---------------------------------------------------------------------------
def esm_zero_shot_baseline(test: pd.DataFrame, cfg, device, limit: int | None) -> dict:
    """Wild-type-marginal masked-LM score at the mutated position."""
    encoder = EsmEncoder(cfg.esm_model, device=str(device))
    rows = test if limit is None else test.iloc[:limit]
    scores, truth = [], []
    target = "label" if "label" in rows.columns else "target"

    for _, row in tqdm(list(rows.iterrows()), desc="esm zero-shot"):
        seq, site = row["parent_seq"], int(row["site"])
        # Score inside the same window the model sees, so the comparison is fair.
        start, end = window_bounds(len(seq), site, cfg.window)
        sub, local = seq[start:end], site - start
        marg = encoder.masked_marginals(sub, [local])[local]
        vocab = encoder.tokenizer.get_vocab()
        wt_tok, mt_tok = vocab.get(row["wt_aa"]), vocab.get(row["mt_aa"])
        if wt_tok is None or mt_tok is None:
            continue
        scores.append(float(marg[mt_tok] - marg[wt_tok]))
        truth.append(float(row[target]))

    p = np.asarray(scores)
    t = np.asarray(truth)
    metrics = classification_metrics(t, p) if target == "label" else regression_metrics(t, p)
    metrics["n"] = int(len(p))
    if "pearson" in metrics:
        metrics["abs_pearson"] = abs(metrics["pearson"])
    metrics["note"] = (
        "higher score = more tolerated, so a negative correlation with a "
        "destabilising ddG is the expected direction"
    )
    return metrics


# ---------------------------------------------------------------------------
# 3. Supervised ESM-only MLP  (the graph-vs-no-graph control)
# ---------------------------------------------------------------------------
class _EsmFeatureDataset(Dataset):
    """Mean-pooled window + site embedding + residue identities. No graph."""

    def __init__(self, df: pd.DataFrame, store: EncodingStore, window: int, target: str):
        self.df = df.reset_index(drop=True)
        self.store = store
        self.window = window
        self.target = target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        rec = self.store.get(row["parent_seq"])
        emb = rec["emb"].float()
        site = int(row["site"])
        start, end = window_bounds(emb.shape[0], min(site, emb.shape[0] - 1), self.window)
        win = emb[start:end]
        feats = torch.cat([win.mean(0), emb[min(site, emb.shape[0] - 1)]])
        aa = torch.zeros(2 * len(AA_ORDER))
        wi, mi = aa_index(row["wt_aa"]), aa_index(row["mt_aa"])
        if wi < len(AA_ORDER):
            aa[wi] = 1.0
        if mi < len(AA_ORDER):
            aa[len(AA_ORDER) + mi] = 1.0
        return torch.cat([feats, aa]), torch.tensor(float(row[self.target]))


def esm_mlp_baseline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                     store: EncodingStore, cfg, device, target: str,
                     epochs: int, batch_size: int) -> dict:
    """Train an MLP on cached embeddings only, then score the test split."""
    sets = {
        "train": _EsmFeatureDataset(train, store, cfg.window, target),
        "val": _EsmFeatureDataset(val, store, cfg.window, target),
        "test": _EsmFeatureDataset(test, store, cfg.window, target),
    }
    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"),
                      num_workers=cfg.num_workers)
        for k, v in sets.items() if len(v)
    }
    in_dim = sets["train"][0][0].numel()
    model = nn.Sequential(
        nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    classification = target == "label"
    loss_fn = (nn.BCEWithLogitsLoss() if classification else nn.MSELoss())

    best_state, best_score = None, -float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x).reshape(-1), y)
            loss.backward()
            opt.step()

        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for x, y in loaders.get("val", loaders["test"]):
                out = model(x.to(device)).reshape(-1).cpu()
                preds.append(torch.sigmoid(out).numpy() if classification else out.numpy())
                truth.append(y.numpy())
        p, t = np.concatenate(preds), np.concatenate(truth)
        m = classification_metrics(t, p) if classification else regression_metrics(t, p)
        score = m.get("auroc" if classification else "pearson", float("nan"))
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  [esm_mlp] epoch {epoch}/{epochs} val={score:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            out = model(x.to(device)).reshape(-1).cpu()
            preds.append(torch.sigmoid(out).numpy() if classification else out.numpy())
            truth.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(truth)
    metrics = classification_metrics(t, p) if classification else regression_metrics(t, p)
    metrics["n"] = int(len(p))
    metrics["best_val_score"] = float(best_score)
    metrics["note"] = (
        "supervised MLP on the same cached embeddings, no graph; the gap to the "
        "full model is the message-passing contribution"
    )
    return metrics


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/baselines")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--zero-shot-limit", type=int, default=None,
                    help="cap rows scored by the masked-LM baseline (it is slow)")
    ap.add_argument("--skip-zero-shot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={
        "cache": args.cache, "esm_model": args.esm_model, "device": args.device})
    os.makedirs(args.out, exist_ok=True)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    df = pd.read_parquet(args.splits)
    target = "label" if "label" in df.columns else "target"
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    if val.empty:
        val = test
    print(f"[baselines] target={target}  train={len(train)} val={len(val)} test={len(test)}")

    results: dict[str, dict] = {}

    print("[baselines] substitution frequency ...")
    results["substitution_frequency"] = substitution_frequency_baseline(train, test, target)

    print("[baselines] supervised ESM-only MLP ...")
    results["esm_mlp"] = esm_mlp_baseline(
        train, val, test, store, cfg, device, target, args.epochs, cfg.batch_size)

    if not args.skip_zero_shot:
        print("[baselines] ESM zero-shot ...")
        results["esm_zero_shot"] = esm_zero_shot_baseline(
            test, cfg, device, args.zero_shot_limit)

    payload = {
        "splits": args.splits,
        "target": target,
        "esm_model": cfg.esm_model,
        "window": cfg.window,
        "baselines": results,
    }
    out_path = os.path.join(args.out, "baselines.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    for name, metrics in results.items():
        print(f"[baselines] {name}: "
              + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                          if isinstance(v, float) and np.isfinite(v)))
    print(f"[baselines] wrote {out_path}")


if __name__ == "__main__":
    main()
