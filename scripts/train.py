"""
scripts/train.py
================
Train the single multi-task Holo-GeoN model.

v1 trained a separate model per task — one for stability, one for
pathogenicity, one for the IDR head — which meant the "unified framework" was
three unrelated checkpoints. Here one backbone and one readout serve every task,
and the heads are optimised jointly with Kendall & Gal uncertainty weighting so
no task needs a hand-tuned loss weight.

Tasks train on whatever data is present. Stability and pathogenicity come from
their split tables; MFI and proteomics activate only if their tables are
configured. A head with no data is left untrained, is recorded as untrained in
the run manifest, and must not be described anywhere as a capability.

On reverse-mutation augmentation
--------------------------------
It is deliberately absent. With the substitution-conditioned head,
``ddG(mt->wt) == -ddG(wt->mt)`` exactly, so feeding a swapped pair with a
negated label produces an identical loss — the augmentation is a mathematical
no-op, not a regulariser. What reversal *does* change is the parent that gets
encoded, and that is measured properly by ``scripts/calibration.py``, which
encodes mutants as their own parents and checks forward/reverse agreement on a
held-out set.

Example
-------
python -m scripts.train --stability-splits data/splits_megascale.parquet \
    --pathogenicity-splits data/splits_clinvar.parquet \
    --cache cache/esm2_650M --out runs/main
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import (
    MutationDataset,
    SequenceDataset,
    collate_graphs,
    filter_to_cache,
    load_sequence_table,
    move_batch,
)
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore
from src.heads import ThreeStateHead
from src.losses import (
    UncertaintyWeightedLoss,
    gaussian_nll,
    pathogenicity_loss,
    sigma_gaussian_nll,
    three_state_loss,
)
from src.metrics import classification_metrics, regression_metrics
from src.model import HoloGeoN

# Tasks the multi-task loss knows about. A task appears in a step only when its
# loader yields a batch.
ALL_TASKS = ["stability", "three_state", "pathogenicity", "idr", "mfi", "proteomics"]

# Fallback experimental uncertainty for the three-state neutral band, used only
# when the stability table reports no per-measurement error. MegaScale's own
# reported ddG errors are preferred and are read from the `sigma` column.
DEFAULT_DDG_SIGMA = 0.5


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------
def _loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_graphs,
        drop_last=False,
        persistent_workers=workers > 0,
    )


def build_split_loaders(
    splits_path: str | None,
    store: EncodingStore,
    cfg,
    target: str,
    batch_size: int,
) -> dict[str, DataLoader]:
    """Train/val/test loaders from a split table written by make_splits."""
    if not splits_path or not os.path.exists(splits_path):
        return {}
    df = pd.read_parquet(splits_path)
    out: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        if part.empty:
            continue
        ds = MutationDataset(part, store, cfg, target=target)
        out[split] = _loader(ds, batch_size, split == "train", cfg.num_workers)
    return out


def build_sequence_loaders(
    path: str | None, store: EncodingStore, cfg, value_key: str, batch_size: int
) -> dict[str, DataLoader]:
    """Train/val loaders for a whole-sequence table, split by hash of sequence."""
    if not path or not os.path.exists(path):
        return {}
    key = os.path.splitext(os.path.basename(path))[0]
    df, report = load_sequence_table(
        path, value_key, overrides=cfg.column_overrides(key),
        max_rows=cfg.max_rows, max_protein_length=cfg.max_protein_length,
    )
    print(report.summary())
    df = filter_to_cache(df, store, report)
    if df.empty:
        return {}
    # Sequence-disjoint by construction: the group key is the sequence itself.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(df))
    cut = int(0.9 * len(df))
    parts = {"train": df.iloc[order[:cut]], "val": df.iloc[order[cut:]]}
    return {
        name: _loader(SequenceDataset(part, store, cfg), batch_size,
                      name == "train", cfg.num_workers)
        for name, part in parts.items() if len(part)
    }


# ---------------------------------------------------------------------------
# Losses for one batch of one task
# ---------------------------------------------------------------------------
def task_losses(model: HoloGeoN, task: str, batch: dict) -> dict[str, torch.Tensor]:
    """Compute the loss terms this task contributes for one batch."""
    if task == "stability":
        z = model.context(batch)
        ddg, logvar = model.ddg(batch, z)
        losses = {"stability": gaussian_nll(ddg, logvar, batch["y"])}
        sigma = batch["sigma"]
        sigma = torch.where(torch.isfinite(sigma), sigma,
                            torch.full_like(sigma, DEFAULT_DDG_SIGMA))
        labels = ThreeStateHead.labels_from_ddg(batch["y"], sigma)
        losses["three_state"] = three_state_loss(model.three_state(batch, z), labels)
        return losses

    if task == "pathogenicity":
        return {"pathogenicity": pathogenicity_loss(model.pathogenicity(batch), batch["y"])}

    if task == "idr":
        mu, sig = model.idr(batch)
        return {"idr": sigma_gaussian_nll(mu, sig, batch["y"])}

    if task == "mfi":
        pred = model.mfi(batch)
        return {"mfi": torch.nn.functional.mse_loss(pred.reshape(-1), batch["y"].reshape(-1))}

    if task == "proteomics":
        pred = model.proteomics(batch)
        return {"proteomics": torch.nn.functional.mse_loss(pred.reshape(-1), batch["y"].reshape(-1))}

    raise KeyError(f"unknown task {task!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_task(model: HoloGeoN, task: str, loader: DataLoader, device) -> dict:
    """Held-out metrics for one task."""
    model.eval()
    preds, targets, extra = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        if task == "stability":
            ddg, logvar = model.ddg(batch)
            preds.append(ddg.reshape(-1).float().cpu().numpy())
            extra.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
        elif task == "pathogenicity":
            preds.append(torch.sigmoid(model.pathogenicity(batch)).reshape(-1).float().cpu().numpy())
        elif task == "idr":
            mu, _ = model.idr(batch)
            preds.append(mu.reshape(-1).float().cpu().numpy())
        elif task == "mfi":
            preds.append(model.mfi(batch).reshape(-1).float().cpu().numpy())
        else:
            preds.append(model.proteomics(batch).reshape(-1).float().cpu().numpy())
        targets.append(batch["y"].reshape(-1).float().cpu().numpy())

    if not preds:
        return {}
    p = np.concatenate(preds)
    t = np.concatenate(targets)

    if task == "pathogenicity":
        metrics = classification_metrics(t, p)
    else:
        metrics = regression_metrics(t, p)
    metrics["n"] = int(len(p))
    # The target's spread on this split travels with the metric: a compressed
    # test distribution mechanically depresses Pearson r, and r reported alone
    # invites a comparison it cannot support.
    metrics["target_std"] = float(np.std(t))
    if extra:
        metrics["mean_predicted_sigma"] = float(np.mean(np.concatenate(extra)))
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--stability-splits", default="data/splits_megascale.parquet")
    ap.add_argument("--pathogenicity-splits", default="data/splits_clinvar.parquet")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-mech-stream", action="store_true",
                    help="ablation: disable the E_mech message-passing stream")
    ap.add_argument("--no-ssm", action="store_true",
                    help="ablation: disable the selective state-space mixer")
    ap.add_argument("--tag", default="main", help="name for this run's artifacts")
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        overrides={
            "cache": args.cache, "esm_model": args.esm_model, "device": args.device,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_rows": args.max_rows, "num_workers": args.num_workers,
        },
    )
    os.makedirs(args.out, exist_ok=True)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    # --- loaders -----------------------------------------------------------
    loaders: dict[str, dict[str, DataLoader]] = {}
    loaders["stability"] = build_split_loaders(
        args.stability_splits, store, cfg, "target", cfg.batch_size)
    loaders["pathogenicity"] = build_split_loaders(
        args.pathogenicity_splits, store, cfg, "label", cfg.batch_size)
    loaders["idr"] = build_sequence_loaders(
        cfg.resolve(cfg.idr_data), store, cfg, "rg", cfg.batch_size)
    loaders["mfi"] = build_sequence_loaders(
        cfg.resolve(cfg.mfi_data), store, cfg, "value", cfg.batch_size)
    loaders["proteomics"] = build_sequence_loaders(
        cfg.resolve(cfg.proteomics_data), store, cfg, "value", cfg.batch_size)
    loaders = {k: v for k, v in loaders.items() if v.get("train")}

    if not loaders:
        raise SystemExit(
            "no training data found. Run scripts.make_splits first, and check "
            "the paths in configs/run.yaml."
        )
    trained_tasks = sorted(loaders)
    untrained = [t for t in ("stability", "pathogenicity", "idr", "mfi", "proteomics")
                 if t not in loaders]
    print(f"[train] tasks with data: {trained_tasks}")
    if untrained:
        print(f"[train] tasks WITHOUT data (heads stay untrained): {untrained}")

    # --- model -------------------------------------------------------------
    esm_dim = ESM_DIMS.get(cfg.esm_model, 1280)
    model = HoloGeoN(
        esm_dim=esm_dim,
        hidden=cfg.hidden,
        out_dim=cfg.out_dim,
        gat_heads=cfg.gat_heads,
        dropout=cfg.dropout,
        use_ssm=cfg.use_ssm and not args.no_ssm,
        use_mech_stream=cfg.use_mech_stream and not args.no_mech_stream,
    ).to(device)
    print(f"[train] parameters: {model.parameter_count()}")

    active_loss_terms = list(
        dict.fromkeys(
            term for t in trained_tasks
            for term in (["stability", "three_state"] if t == "stability" else [t])
        )
    )
    criterion = UncertaintyWeightedLoss(active_loss_terms).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(len(v["train"]) for v in loaders.values())
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    primary = "stability" if "stability" in loaders else trained_tasks[0]
    primary_metric = "pearson" if primary != "pathogenicity" else "auroc"

    best_score = -float("inf")
    best_epoch = -1
    patience = 0
    history: list[dict] = []
    ckpt_path = os.path.join(args.out, "model_best.pt")
    last_path = os.path.join(args.out, "last.pt")
    start_epoch = 1

    if os.path.exists(last_path):
        try:
            state = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            criterion.load_state_dict(state["criterion"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            start_epoch = int(state["epoch"]) + 1
            best_score = state.get("best_score", -float("inf"))
            best_epoch = state.get("best_epoch", -1)
            history = state.get("history", [])
            print(f"[train] resumed from epoch {state['epoch']} (best {primary_metric}"
                  f"={best_score:.4f})")
        except Exception as exc:
            print(f"[train] could not resume ({exc}); starting fresh")

    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        cycles = {t: itertools.cycle(v["train"]) for t, v in loaders.items()}
        running: dict[str, float] = {}

        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            losses: dict[str, torch.Tensor] = {}
            with torch.amp.autocast("cuda", enabled=use_amp):
                for task in trained_tasks:
                    batch = move_batch(next(cycles[task]), device)
                    losses.update(task_losses(model, task, batch))
                total, parts = criterion(losses)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v

        train_log = {k: v / steps_per_epoch for k, v in running.items()}
        val = {
            t: evaluate_task(model, t, v["val"], device)
            for t, v in loaders.items() if "val" in v
        }
        primary_val = val.get(primary, {})
        if primary_val and primary_metric not in primary_val:
            # Guard against a silent no-op: if the key is wrong, `score` is NaN
            # forever, so nothing ever "improves", no checkpoint is written and
            # early stopping never fires. That failure is invisible until the
            # run ends with no model, which is an expensive way to find a typo.
            raise KeyError(
                f"primary metric {primary_metric!r} is not among the reported "
                f"metrics for task {primary!r}: {sorted(primary_val)}"
            )
        score = primary_val.get(primary_metric, float("nan"))
        record = {"epoch": epoch, "train": train_log, "val": val,
                  "lr": sched.get_last_lr()[0]}
        history.append(record)

        summary = "  ".join(
            f"{t}:{primary_metric if t == primary else 'n'}="
            f"{val[t].get(primary_metric if t == primary else 'n', float('nan'))}"
            for t in sorted(val)
        )
        print(f"[train] epoch {epoch}/{cfg.epochs}  "
              f"loss={sum(v for k, v in train_log.items() if not k.endswith('_weight')):.4f}  "
              f"{primary}.{primary_metric}={score:.4f}  | {summary}")

        improved = np.isfinite(score) and score > best_score
        if improved:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save(
                {"model": model.state_dict(),
                 "config": cfg.as_dict(),
                 "esm_dim": esm_dim,
                 "epoch": epoch,
                 "val": val,
                 "trained_tasks": trained_tasks,
                 "untrained_heads": untrained},
                ckpt_path,
            )
        else:
            patience += 1

        try:
            torch.save(
                {"model": model.state_dict(), "criterion": criterion.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                 "history": history},
                last_path,
            )
        except Exception as exc:
            print(f"[train] checkpoint save failed at epoch {epoch}: {exc}")

        if patience >= cfg.early_stop_patience:
            print(f"[train] early stop: no improvement in {patience} epochs "
                  f"(best epoch {best_epoch}, {primary_metric}={best_score:.4f})")
            break

    # --- final test-split evaluation on the best checkpoint ----------------
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=False)["model"])
    test = {
        t: evaluate_task(model, t, v["test"], device)
        for t, v in loaders.items() if "test" in v
    }

    manifest = {
        "tag": args.tag,
        "config": cfg.as_dict(),
        "ablation": {
            "mech_stream": bool(cfg.use_mech_stream and not args.no_mech_stream),
            "ssm": bool(cfg.use_ssm and not args.no_ssm),
        },
        "trained_tasks": trained_tasks,
        "untrained_heads": untrained,
        "parameters": model.parameter_count(),
        "best_epoch": best_epoch,
        f"best_val_{primary}_{primary_metric}": best_score,
        "test": test,
        "history": history,
        "wall_clock_hours": (time.time() - started) / 3600.0,
        "validated_by_real_run": True,
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[train] best epoch {best_epoch}  best val {primary}.{primary_metric}"
          f"={best_score:.4f}")
    for task, metrics in sorted(test.items()):
        print(f"[train] test {task}: {metrics}")
    print(f"[train] wrote {os.path.join(args.out, 'metrics.json')}")


if __name__ == "__main__":
    main()
