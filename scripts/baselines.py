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
   at the mutated site plus the two residue identities.

   Round-2 review, R2-16: "same budget" was documented and not delivered. The
   control ran 20 epochs to the model's 60, with no LR schedule, no gradient
   clipping and no early stopping, at 1.46M parameters against 20.6M — so the
   gap the paper's central empirical claim rests on conflated the graph with a
   3x training budget and a 14x capacity difference. It now takes the model's
   epoch count, warmup+cosine schedule, gradient clipping and early-stopping
   patience from the same config, and runs in TWO variants:

     * ``esm_mlp`` — matched training budget, the original narrow width.
     * ``esm_mlp_matched_capacity`` — the same, widened to the full model's own
       parameter count.

   Reporting both is what makes the comparison interpretable. If the model beats
   the narrow MLP but not the wide one, the gap was capacity; if it beats both by
   the same margin, capacity is not the explanation and the architecture is.

   The gap between this and the full model is the contribution of everything the
   full model adds — the graph, the site-anchored readout, the SSM, the
   cross-attention fusion and the mechanistic channels together. Attributing it
   to message passing alone needs ``scripts/ablations.py``, which separates
   them.

Example
-------
python -m scripts.baselines --splits data/splits_megascale.parquet \
    --cache cache/esm2_650M --out runs/baselines
"""
from __future__ import annotations

import argparse
import json
import math
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
from src.encoder import ESM_DIMS, EncodingStore, EsmEncoder
from src.metrics import classification_metrics, regression_metrics
from src.provenance import provenance
from src.seeding import loader_kwargs, seed_everything
from src.substitution import AA_ORDER, aa_index
from src.encoder import MAX_CONTEXT_RESIDUES
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
    truncated = 0
    target = "label" if "label" in rows.columns else "target"

    for _, row in tqdm(list(rows.iterrows()), desc="esm zero-shot"):
        seq, site = row["parent_seq"], int(row["site"])
        # Score on the FULL sequence, which is the context the model's own
        # inputs carry: embeddings are computed by ESM-2 over the whole protein
        # and cached at full length, and the window is sliced from that. Round-1
        # review, R1-12: scoring the baseline on a 256-residue window "so the
        # comparison is fair" gave it strictly less context than the model,
        # handicapping the control that the central empirical claim rests on.
        #
        # Sequences beyond the encoder's context are scored on the largest
        # window the encoder can take, centred on the site, and counted.
        if len(seq) <= MAX_CONTEXT_RESIDUES:
            sub, local = seq, site
        else:
            start, end = window_bounds(len(seq), site, MAX_CONTEXT_RESIDUES)
            sub, local = seq[start:end], site - start
            truncated += 1
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
    metrics["scored_on_full_sequence"] = int(len(p) - truncated)
    metrics["scored_on_truncated_context"] = int(truncated)
    metrics["note"] = (
        "wild-type-marginal log-odds: a higher score means the substitution is "
        "more tolerated. Under the internal convention (positive ddG is "
        "STABILISING, see src/data.DDG_CONVENTION) a more-tolerated substitution "
        "has a HIGHER ddG, so the expected correlation is POSITIVE. Round-1 "
        "review, R1-19: this docstring previously asserted the opposite, which "
        "would have made a correctly-signed result look like a bug"
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


def mlp_width_for(in_dim: int, target_params: int) -> int:
    """
    Hidden width making a ``[in -> h -> h/2 -> 1]`` MLP hold ~``target_params``.

    Round-2 review, R2-16 asked for a matched parameter count OR both a
    matched-capacity and a matched-budget variant. Solving
    ``h^2/2 + h*(in_dim + 2) ~= target`` gives the width; the caller reports the
    count it actually achieved rather than claiming the match.
    """
    a, b, c = 0.5, float(in_dim + 2), -float(target_params)
    h = (-b + (b * b - 4 * a * c) ** 0.5) / (2 * a)
    return max(16, int(round(h)))


def _mlp(in_dim: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, 1),
    )


def esm_mlp_baseline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                     store: EncodingStore, cfg, device, target: str,
                     epochs: int, batch_size: int, hidden: int = 512) -> dict:
    """
    Train an MLP on cached embeddings only, then score the test split.

    Given the same epoch budget, warmup+cosine LR schedule, gradient clipping
    and early-stopping patience as ``scripts/train.py``, so the gap to the full
    model is not a training-budget artifact (R2-16).
    """
    sets = {
        "train": _EsmFeatureDataset(train, store, cfg.window, target),
        "val": _EsmFeatureDataset(val, store, cfg.window, target),
        "test": _EsmFeatureDataset(test, store, cfg.window, target),
    }
    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"),
                      num_workers=cfg.num_workers,
                      **loader_kwargs(cfg.seed, cfg.num_workers))
        for k, v in sets.items() if len(v)
    }
    in_dim = sets["train"][0][0].numel()
    # Seeded so the control is as reproducible as the model it is compared with.
    torch.manual_seed(cfg.seed)
    model = _mlp(in_dim, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    classification = target == "label"
    loss_fn = (nn.BCEWithLogitsLoss() if classification else nn.MSELoss())

    # The same schedule the model gets, so the comparison is about architecture.
    total_steps = max(1, len(loaders["train"]) * epochs)
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best_state, best_score = None, -float("inf")
    best_epoch, patience = -1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x).reshape(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for x, y in loaders["val"]:
                out = model(x.to(device)).reshape(-1).cpu()
                preds.append(torch.sigmoid(out).numpy() if classification else out.numpy())
                truth.append(y.numpy())
        p, t = np.concatenate(preds), np.concatenate(truth)
        m = classification_metrics(t, p) if classification else regression_metrics(t, p)
        score = m.get("auroc" if classification else "pearson", float("nan"))
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        print(f"  [esm_mlp] epoch {epoch}/{epochs} val={score:.4f}")
        if patience >= cfg.early_stop_patience:
            print(f"  [esm_mlp] early stop at epoch {epoch} "
                  f"(best {best_epoch}, val={best_score:.4f})")
            break

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
    metrics["best_epoch"] = int(best_epoch)
    metrics["parameters"] = int(sum(q.numel() for q in model.parameters()))
    metrics["training_budget"] = {
        "epochs": epochs,
        "early_stop_patience": cfg.early_stop_patience,
        "lr_schedule": "warmup + cosine, as scripts/train.py",
        "grad_clip": cfg.grad_clip,
        "matched_to_model": True,
    }
    # Round-1 review R1-27 asked for this sentence in three places and it was
    # changed in none. With five arms in the ablation grid, the accurate version
    # is easy to write and the causal one was never supportable: this control
    # differs from the full model in the graph AND the readout AND the mixer AND
    # the fusion AND the mechanistic channels (R2-16).
    metrics["note"] = (
        "supervised MLP on the same cached embeddings, no graph, matched "
        "training budget. The gap to the full model is the joint contribution of "
        "the graph, the site-anchored readout, the SSM, the cross-attention "
        "fusion and the mechanistic channels; scripts/ablations.py separates "
        "them. Compare against esm_mlp_matched_capacity before attributing any "
        "of it to the architecture rather than to parameter count."
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
    ap.add_argument("--epochs", type=int, default=None,
                    help="epoch budget for the MLP control (default: the model's "
                         "own `epochs` from the config, so the comparison is not "
                         "a training-budget artifact - R2-16)")
    ap.add_argument("--zero-shot-limit", type=int, default=None,
                    help="cap rows scored by the masked-LM baseline (it is slow)")
    ap.add_argument("--skip-zero-shot", action="store_true")
    ap.add_argument("--skip-matched-capacity", action="store_true",
                    help="skip the second MLP variant sized to the model's own "
                         "parameter count (R2-16)")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={
        "cache": args.cache, "esm_model": args.esm_model, "device": args.device})
    os.makedirs(args.out, exist_ok=True)
    seed_everything(cfg.seed)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    df = pd.read_parquet(args.splits)
    target = "label" if "label" in df.columns else "target"
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    if val.empty:
        # Round-1 review, R1-12: this used to fall back to `val = test`, so the
        # MLP baseline - described everywhere as "the control that matters" -
        # was early-stopped on the very split it was then scored on. The gap to
        # the full model is the paper's central empirical claim; leaking into
        # the control makes it unmeasurable in the direction it is claimed.
        raise SystemExit(
            f"{args.splits}: no validation rows. The MLP baseline needs a val "
            f"split for model selection - selecting on test would inflate it. "
            f"Re-generate the split table with scripts.make_splits."
        )
    print(f"[baselines] target={target}  train={len(train)} val={len(val)} test={len(test)}")

    results: dict[str, dict] = {}

    print("[baselines] substitution frequency ...")
    results["substitution_frequency"] = substitution_frequency_baseline(train, test, target)

    print("[baselines] supervised ESM-only MLP ...")
    epochs = args.epochs if args.epochs is not None else int(cfg.epochs)
    print(f"[baselines] esm_mlp budget: {epochs} epochs, patience "
          f"{cfg.early_stop_patience}, warmup+cosine LR, grad clip "
          f"{cfg.grad_clip} - matched to scripts/train.py")
    results["esm_mlp"] = esm_mlp_baseline(
        train, val, test, store, cfg, device, target, epochs, cfg.batch_size)

    # Round-2 review, R2-16: a matched TRAINING budget still leaves a 14x
    # capacity gap, so the remaining difference could be either the architecture
    # or the parameter count. The second variant closes the capacity gap and
    # keeps everything else identical, which is what makes the pair
    # interpretable: if both land in the same place, capacity is not the
    # explanation.
    if not args.skip_matched_capacity:
        from src.model import HoloGeoN, model_kwargs_from_config

        esm_dim = ESM_DIMS.get(cfg.esm_model, 1280)
        full = HoloGeoN(**model_kwargs_from_config(cfg, esm_dim))
        target_params = full.parameter_count()["total"]
        del full
        feat_dim = _EsmFeatureDataset(train.head(1), store, cfg.window,
                                      target)[0][0].numel()
        width = mlp_width_for(feat_dim, target_params)
        print(f"[baselines] matched-capacity MLP: hidden {width} targeting the "
              f"model's {target_params:,} parameters ...")
        results["esm_mlp_matched_capacity"] = esm_mlp_baseline(
            train, val, test, store, cfg, device, target, epochs,
            cfg.batch_size, hidden=width)
        results["esm_mlp_matched_capacity"]["training_budget"]["matched_capacity"] = {
            "model_parameters": int(target_params),
            "hidden_width": width,
        }

    if not args.skip_zero_shot:
        print("[baselines] ESM zero-shot ...")
        results["esm_zero_shot"] = esm_zero_shot_baseline(
            test, cfg, device, args.zero_shot_limit)

    payload = {
        "provenance": provenance(),
        "splits": args.splits,
        "target": target,
        "epochs": epochs,
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
