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

from src import splits as S
from src.config import load_config
from src.data import (
    LoadReport,
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
from src.seeding import loader_kwargs, restore_rng_state, rng_state, seed_everything
from src.model import HoloGeoN, model_kwargs_from_config

# Tasks the multi-task loss knows about. A task appears in a step only when its
# loader yields a batch.
ALL_TASKS = ["stability", "three_state", "pathogenicity", "idr", "mfi", "proteomics"]

#: Consecutive non-finite primary-metric epochs tolerated before aborting.
MAX_NAN_EPOCHS = 3

# Fallback experimental uncertainty for the three-state neutral band, used only
# when the stability table reports no per-measurement error. MegaScale's own
# reported ddG errors are preferred and are read from the `sigma` column.
DEFAULT_DDG_SIGMA = 0.5


def endless(loader):
    """
    Yield batches forever, re-creating the iterator at the end of each pass.

    Round-1 review, R1-03: this replaces ``itertools.cycle``, which saves every
    element it yields so it can replay them. Each element here is a collated
    batch whose ``esm`` tensor alone is 83.9 MB at the shipped shape, so one
    pass over a MegaScale-scale loader retained ~354 GB of host RAM on a machine
    documented as having 32 GB - an OOM part-way through epoch 1, invisible to
    the 300-row pipeline test. Re-creating the iterator holds one batch at a
    time and reshuffles on each pass, where ``cycle`` replayed the identical
    batches in the identical order.
    """
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------
def _loader(dataset, batch_size: int, shuffle: bool, workers: int,
            seed: int = 42) -> DataLoader:
    """A DataLoader whose shuffle order and worker RNG are reproducible."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_graphs,
        drop_last=False,
        persistent_workers=workers > 0,
        **loader_kwargs(seed, workers),
    )


def build_split_loaders(
    splits_path: str | None,
    store: EncodingStore,
    cfg,
    target: str,
    batch_size: int,
    task: str = "",
) -> tuple[dict[str, DataLoader], dict]:
    """
    Train/val/test loaders from a split table written by make_splits.

    Rows whose parent protein has no cached encoding are filtered out **and
    counted** here. Round-1 review, R1-06: this filter existed but was wired
    only into the three optional whole-sequence tasks. For stability and
    pathogenicity - the two tasks the paper reports - an uncached parent instead
    raised ``FileNotFoundError`` from inside a DataLoader worker, which can land
    twenty epochs into a two-day run, and no exclusion was ever counted.

    Returns ``(loaders, report_dict)``; the report goes into ``metrics.json`` so
    the exclusion is visible in the artifacts rather than only in a crash.
    """
    if not splits_path or not os.path.exists(splits_path):
        return {}, {}
    df = pd.read_parquet(splits_path)

    report = LoadReport(source=f"{splits_path} ({task or target})", rows_in=len(df))
    df = filter_to_cache(df, store, report)
    if report.dropped.get("no_cached_encoding"):
        print(report.summary())
    if df.empty:
        return {}, report.as_dict()

    out: dict[str, DataLoader] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        counts[split] = int(len(part))
        if part.empty:
            continue
        ds = MutationDataset(part, store, cfg, target=target)
        out[split] = _loader(ds, batch_size, split == "train",
                             cfg.num_workers, cfg.seed)
    info = report.as_dict()
    info["split_counts"] = counts
    return out, info


def build_sequence_loaders(
    path: str | None, store: EncodingStore, cfg, value_key: str, batch_size: int
) -> tuple[dict[str, DataLoader], dict]:
    """
    Train/val/test loaders for a whole-sequence table, split by unique sequence.

    Round-1 review, R1-08: this previously permuted **row indices** and cut at
    90%, under a comment claiming it was "sequence-disjoint by construction". It
    was not - a sequence appearing in two rows landed on both sides, which is
    the norm in retention-time and expression corpora where the same peptide
    recurs across runs. There was also no test split at all, so any number these
    heads produced was measured on the split early stopping ran on.

    Grouping is now by unique sequence, whole groups to one side, with a
    three-way split and a report. It reuses the same assignment logic as
    ``scripts/make_splits.py`` so both paths agree.
    """
    if not path or not os.path.exists(path):
        return {}, {}
    key = os.path.splitext(os.path.basename(path))[0]
    df, report = load_sequence_table(
        path, value_key, overrides=cfg.column_overrides(key),
        max_rows=cfg.max_rows, max_protein_length=cfg.max_protein_length,
    )
    print(report.summary())
    df = filter_to_cache(df, store, report)
    if df.empty:
        return {}, report.as_dict()

    # Whole groups to one split, group = the sequence itself.
    groups = df["parent_seq"].tolist()
    row_splits = S.row_splits_from_groups(
        groups, ratios=(0.8, 0.1, 0.1), seed=cfg.seed
    )
    if not S.verify_no_leakage(groups, row_splits):
        raise SystemExit(f"{path}: leakage detected in the whole-sequence split")
    df = df.copy()
    df["split"] = row_splits

    out: dict[str, DataLoader] = {}
    counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        counts[name] = int(len(part))
        if part.empty:
            continue
        out[name] = _loader(SequenceDataset(part, store, cfg), batch_size,
                            name == "train", cfg.num_workers, cfg.seed)
    info = report.as_dict()
    info["split_counts"] = counts
    info["grouping"] = "unique parent sequence, whole groups to one split"
    return out, info


# ---------------------------------------------------------------------------
# Losses for one batch of one task
# ---------------------------------------------------------------------------
def three_state_labels(batch: dict) -> torch.Tensor:
    """Class labels from measured ddG and its per-measurement uncertainty."""
    sigma = batch["sigma"]
    sigma = torch.where(torch.isfinite(sigma), sigma,
                        torch.full_like(sigma, DEFAULT_DDG_SIGMA))
    return ThreeStateHead.labels_from_ddg(batch["y"], sigma)


def task_losses(model: HoloGeoN, task: str, batch: dict,
                weights: dict | None = None) -> dict[str, torch.Tensor]:
    """Compute the loss terms this task contributes for one batch."""
    weights = weights or {}
    if task == "stability":
        z = model.context(batch)
        ddg, logvar = model.ddg(batch, z)
        losses = {"stability": gaussian_nll(ddg, logvar, batch["y"])}
        labels = three_state_labels(batch)
        losses["three_state"] = three_state_loss(
            model.three_state(batch, z), labels,
            class_weights=weights.get("three_state"),
        )
        return losses

    if task == "pathogenicity":
        return {"pathogenicity": pathogenicity_loss(
            model.pathogenicity(batch), batch["y"],
            pos_weight=weights.get("pathogenicity"),
        )}

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


def _json_safe(obj):
    """Recursively replace non-finite floats with None so the JSON is valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    return obj


def compute_class_weights(loaders: dict, device) -> dict:
    """
    Inverse-frequency class weights from the training split.

    Scans labels only - no model forward - so it is cheap even on a large table.
    """
    weights: dict = {}

    if "stability" in loaders:
        counts = np.zeros(3, dtype=np.int64)
        for batch in loaders["stability"]["train"]:
            labels = three_state_labels(batch).reshape(-1).numpy()
            counts += np.bincount(labels, minlength=3)
        if counts.sum() and (counts > 0).all():
            inv = counts.sum() / (3.0 * counts)
            weights["three_state"] = torch.tensor(inv, dtype=torch.float32, device=device)

    if "pathogenicity" in loaders:
        pos = neg = 0
        for batch in loaders["pathogenicity"]["train"]:
            y = batch["y"].reshape(-1).numpy()
            pos += int((y > 0.5).sum())
            neg += int((y <= 0.5).sum())
        if pos:
            weights["pathogenicity"] = torch.tensor(neg / pos, dtype=torch.float32,
                                                    device=device)
    return weights


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_task(model: HoloGeoN, task: str, loader: DataLoader, device) -> dict:
    """Held-out metrics for one task."""
    model.eval()
    preds, targets, extra = [], [], []
    ts_pred, ts_true = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        if task == "stability":
            z = model.context(batch)
            ddg, logvar = model.ddg(batch, z)
            preds.append(ddg.reshape(-1).float().cpu().numpy())
            extra.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
            # Round-1 review, R1-15: the three-state head was trained, listed in
            # the README head table and served by the API, but evaluated
            # nowhere - so nothing could tell whether the swap-equivariant
            # parameterisation buys anything. It rides along with stability,
            # on the same split.
            ts_pred.append(model.three_state(batch, z).argmax(-1).cpu().numpy())
            ts_true.append(three_state_labels(batch).reshape(-1).cpu().numpy())
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
    if ts_pred:
        metrics["three_state"] = three_state_metrics(
            np.concatenate(ts_true), np.concatenate(ts_pred)
        )
    return metrics


def three_state_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro-F1, per-class recall and the confusion matrix for the 3-class head."""
    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    names = ["destabilizing", "neutral", "stabilizing"]
    labels = [0, 1, 2]
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "recall_per_class": {n: float(r) for n, r in zip(names, recalls)},
        "support_per_class": {n: int((y_true == i).sum()) for i, n in enumerate(names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_order": names,
    }


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
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides configs/run.yaml; the ablation runner uses "
                         "this to repeat an arm across seeds")
    ap.add_argument("--no-mech-stream", action="store_true",
                    help="ablation: disable the E_mech message-passing stream")
    ap.add_argument("--no-ssm", action="store_true",
                    help="ablation: disable the selective state-space mixer")
    ap.add_argument("--no-geo-stream", action="store_true",
                    help="ablation: disable the E_geo (contact graph) stream")
    ap.add_argument("--global-pool-readout", action="store_true",
                    help="ablation: replace the site-anchored readout with the "
                         "v1 global-pool formulation")
    ap.add_argument("--tag", default="main", help="name for this run's artifacts")
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        overrides={
            "cache": args.cache, "esm_model": args.esm_model, "device": args.device,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_rows": args.max_rows, "num_workers": args.num_workers,
            "seed": args.seed,
        },
    )
    os.makedirs(args.out, exist_ok=True)
    # Round-1 review, R1-01: nothing in the training path was seeded, and two
    # runs of the same command on identical data differed by 0.030 test Pearson
    # - larger than most ablation effects, which made every single-run
    # delta_vs_full uninterpretable.
    seed_everything(cfg.seed, strict=bool(cfg.get("strict_determinism", False)))
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    cache_manifest = store.manifest()
    mech_on = bool(cfg.use_mech_stream and not args.no_mech_stream)
    if mech_on and cache_manifest and not cache_manifest.get("cds_active"):
        # Not fatal — a geometry-only run is a legitimate configuration — but it
        # must not be mistaken for a dual-stream one. R1-05 was exactly this
        # confusion, invisible because nothing printed or recorded it.
        print("[train] WARNING: the mechanistic stream is enabled but the encoder "
              "cache has no active coding sequences "
              f"(cds_coverage={cache_manifest.get('cds_coverage')}). E_mech will "
              "hold self-loops only, the stream contributes nothing, and its "
              "ablation delta will read ~0 for plumbing reasons rather than "
              "scientific ones. Run `python run.py etl --cds-source ...` and "
              "re-run precompute, or train with --no-mech-stream and say so.")

    # --- loaders -----------------------------------------------------------
    loaders: dict[str, dict[str, DataLoader]] = {}
    data_reports: dict[str, dict] = {}
    loaders["stability"], data_reports["stability"] = build_split_loaders(
        args.stability_splits, store, cfg, "target", cfg.batch_size, "stability")
    loaders["pathogenicity"], data_reports["pathogenicity"] = build_split_loaders(
        args.pathogenicity_splits, store, cfg, "label", cfg.batch_size, "pathogenicity")
    loaders["idr"], data_reports["idr"] = build_sequence_loaders(
        cfg.resolve(cfg.idr_data), store, cfg, "rg", cfg.batch_size)
    loaders["mfi"], data_reports["mfi"] = build_sequence_loaders(
        cfg.resolve(cfg.mfi_data), store, cfg, "value", cfg.batch_size)
    loaders["proteomics"], data_reports["proteomics"] = build_sequence_loaders(
        cfg.resolve(cfg.proteomics_data), store, cfg, "value", cfg.batch_size)
    loaders = {k: v for k, v in loaders.items() if v.get("train")}
    data_reports = {k: v for k, v in data_reports.items() if v}

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
    model_kwargs = model_kwargs_from_config(
        cfg, esm_dim,
        use_ssm=cfg.use_ssm and not args.no_ssm,
        use_mech_stream=cfg.use_mech_stream and not args.no_mech_stream,
        use_geo_stream=not args.no_geo_stream,
        site_anchored_readout=not args.global_pool_readout,
    )
    model = HoloGeoN(**model_kwargs).to(device)
    print(f"[train] parameters: {model.parameter_count()}")

    active_loss_terms = list(
        dict.fromkeys(
            term for t in trained_tasks
            for term in (["stability", "three_state"] if t == "stability" else [t])
        )
    )
    criterion = UncertaintyWeightedLoss(active_loss_terms).to(device)

    # Round-1 review, R1-15: three_state_loss and pathogenicity_loss both accept
    # class balancing and no caller ever supplied it, while the class
    # distributions are severely skewed (most substitutions are destabilising;
    # ClinVar benign/pathogenic is far from 1:1). Weights are computed once from
    # the training split by inverse frequency.
    class_weights = compute_class_weights(loaders, device)
    if class_weights:
        print("[train] class balancing: "
              + ", ".join(f"{k}={v.tolist() if v.numel() > 1 else float(v):.4}"
                          if v.numel() == 1 else f"{k}={[round(x, 3) for x in v.tolist()]}"
                          for k, v in class_weights.items()))
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Round-1 review, R1-10: stepping every task every iteration, with
    # steps_per_epoch = max(len(loader)), forcibly equalised them - a 300-row
    # experimental Rg table was replayed ~844 times per epoch and received the
    # same gradient budget as a 270,000-row ddG table. Tasks are now sampled in
    # proportion to len(loader) ** temperature, so a small table is seen in
    # proportion to its size. temperature=1 is strictly proportional; 0.5 is a
    # square-root schedule that gives small tasks some extra weight without
    # letting them dominate; 0 reproduces the old equal treatment.
    task_sizes = {t: len(v["train"]) for t, v in loaders.items()}
    temperature = float(cfg.get("task_sampling_temperature", 0.5))
    weights = {t: max(n, 1) ** temperature for t, n in task_sizes.items()}
    weight_total = sum(weights.values())
    task_names = list(loaders)
    task_probs = np.array([weights[t] / weight_total for t in task_names])

    steps_per_epoch = max(task_sizes.values())
    total_steps = steps_per_epoch * cfg.epochs
    print(f"[train] task sizes (batches): {task_sizes}")
    print(f"[train] sampling probabilities (temperature={temperature}): "
          + ", ".join(f"{t}={p:.3f}" for t, p in zip(task_names, task_probs)))
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

    # Round-1 review, R1-02: with no val split for the primary task, the score
    # was NaN every epoch, so nothing ever "improved", model_best.pt was never
    # written, the best-checkpoint reload silently no-opped, and the reported
    # test metrics came from whatever the last epoch left in memory - while the
    # process exited 0 and the manifest still claimed the run was validated.
    # There is no sensible way to continue, so it fails here rather than
    # producing a plausible-looking manifest.
    if "val" not in loaders[primary]:
        raise SystemExit(
            f"the primary task {primary!r} has no validation split, so model "
            f"selection and early stopping cannot run. Re-generate the split "
            f"table with scripts.make_splits (it writes train/val/test), or "
            f"check that its val side is non-empty."
        )

    best_score = -float("inf")
    best_epoch = -1
    patience = 0
    nan_epochs = 0
    history: list[dict] = []
    ckpt_path = os.path.join(args.out, "model_best.pt")
    last_path = os.path.join(args.out, "last.pt")
    start_epoch = 1

    if os.path.exists(last_path):
        # Round-1 review, R1-18: resume restored strictly less than it saved.
        # `patience` reset to 0 every time, so a run resumed more often than
        # early_stop_patience could never early-stop; RNG state and the AMP loss
        # scale were not saved at all, so a resumed run saw a different data
        # order, different dropout masks and a re-converging scaler. A broad
        # `except ... starting fresh` could also leave a loaded model paired
        # with a fresh optimizer, announced in one line of a 60-epoch log.
        try:
            state = torch.load(last_path, map_location=device, weights_only=False)
        except Exception as exc:
            raise SystemExit(
                f"{last_path} exists but could not be read ({exc}). Refusing to "
                f"start fresh silently: delete it deliberately to restart, so a "
                f"discarded run is never mistaken for a resumed one."
            )
        model.load_state_dict(state["model"])
        criterion.load_state_dict(state["criterion"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_score = state.get("best_score", -float("inf"))
        best_epoch = state.get("best_epoch", -1)
        patience = int(state.get("patience", 0))
        nan_epochs = int(state.get("nan_epochs", 0))
        history = state.get("history", [])
        rng_ok = restore_rng_state(state.get("rng"))
        print(f"[train] resumed from epoch {state['epoch']} (best {primary_metric}"
              f"={best_score:.4f}, patience={patience}, rng_restored={rng_ok})")
        if not rng_ok:
            print("[train] NOTE: RNG state was not restored, so this is a "
                  "continuation with a different data order than an "
                  "uninterrupted run would have seen.")

    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        cycles = {t: endless(v["train"]) for t, v in loaders.items()}
        running: dict[str, float] = {}

        step_rng = np.random.default_rng(cfg.seed + epoch)
        realised_steps: dict[str, int] = {t: 0 for t in task_names}

        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            task = str(step_rng.choice(task_names, p=task_probs))
            realised_steps[task] += 1
            losses: dict[str, torch.Tensor] = {}
            with torch.amp.autocast("cuda", enabled=use_amp):
                batch = move_batch(next(cycles[task]), device)
                losses.update(task_losses(model, task, batch, class_weights))
                total, parts = criterion(losses)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v

        # Average each term over the steps that actually contributed it, not
        # over every step in the epoch - with proportional sampling a task no
        # longer appears in every step.
        counts: dict[str, int] = {}
        for t, n in realised_steps.items():
            for term in (["stability", "three_state"] if t == "stability" else [t]):
                counts[term] = counts.get(term, 0) + n
        train_log = {k: v / max(counts.get(k.replace("_weight", ""), 1), 1)
                     for k, v in running.items()}
        train_log["realised_steps"] = dict(realised_steps)
        val = {
            t: evaluate_task(model, t, v["val"], device)
            for t, v in loaders.items() if "val" in v
        }
        primary_val = val.get(primary, {})
        if primary_metric not in primary_val:
            raise KeyError(
                f"primary metric {primary_metric!r} is not among the reported "
                f"metrics for task {primary!r}: {sorted(primary_val)}"
            )
        score = primary_val.get(primary_metric, float("nan"))

        # Round-1 review, R1-02: the key-presence guard above could never fire,
        # because the metrics helpers always emit the key - as NaN. A NaN score
        # means model selection is not happening at all, so it is caught by
        # VALUE here. One NaN epoch can be a degenerate batch; several in a row
        # means the run cannot select a model and would end with none.
        if not np.isfinite(score):
            nan_epochs += 1
            print(f"[train] WARNING: {primary}.{primary_metric} is not finite "
                  f"(epoch {epoch}, {nan_epochs} in a row). Model selection "
                  f"cannot proceed while this holds.")
            if nan_epochs >= MAX_NAN_EPOCHS:
                raise SystemExit(
                    f"{primary}.{primary_metric} has been non-finite for "
                    f"{nan_epochs} consecutive epochs, so no checkpoint can ever "
                    f"be selected. Most often the validation split is degenerate "
                    f"(constant target, or a single class). Aborting rather than "
                    f"finishing with no model and reporting test metrics anyway."
                )
        else:
            nan_epochs = 0
        record = {"epoch": epoch, "train": train_log, "val": val,
                  "lr": sched.get_last_lr()[0]}
        history.append(record)

        summary = "  ".join(
            f"{t}:{primary_metric if t == primary else 'n'}="
            f"{val[t].get(primary_metric if t == primary else 'n', float('nan'))}"
            for t in sorted(val)
        )
        print(f"[train] epoch {epoch}/{cfg.epochs}  "
              f"loss={sum(v for k, v in train_log.items() if isinstance(v, float) and not k.endswith('_weight')):.4f}  "
              f"{primary}.{primary_metric}={score:.4f}  | {summary}")

        improved = np.isfinite(score) and score > best_score
        if improved:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save(
                {"model": model.state_dict(),
                 # The checkpoint carries exactly how the model was built, so
                 # loading never has to re-derive constructor arguments from
                 # loose config keys and cannot drift out of sync with them.
                 "model_kwargs": model_kwargs,
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
                 "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                 "patience": patience, "nan_epochs": nan_epochs,
                 "rng": rng_state(),
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
    # Test metrics are reported ONLY for a model that was actually selected. If
    # no epoch ever improved there is no best checkpoint, and scoring whatever
    # the last epoch happened to leave in memory - as this did before - produces
    # a headline number attributable to no saved model at all.
    selected = best_epoch > 0 and os.path.exists(ckpt_path)
    if selected:
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=False)["model"])
        test = {
            t: evaluate_task(model, t, v["test"], device)
            for t, v in loaders.items() if "test" in v
        }
    else:
        test = {}
        print("[train] no checkpoint was ever selected, so no test metrics are "
              "reported. metrics.json records best_checkpoint: null.")

    manifest = {
        "tag": args.tag,
        "config": cfg.as_dict(),
        "seed": cfg.seed,
        "ablation": {
            "mech_stream": bool(cfg.use_mech_stream and not args.no_mech_stream),
            "geo_stream": not args.no_geo_stream,
            "ssm": bool(cfg.use_ssm and not args.no_ssm),
            "site_anchored_readout": not args.global_pool_readout,
        },
        "trained_tasks": trained_tasks,
        "untrained_heads": untrained,
        "data_reports": data_reports,
        # What the cache this run actually read contains. `cds_coverage` is the
        # honest state of the mechanistic stream: with 0 coverage E_mech carries
        # self-loops only, so the stream contributes nothing and its ablation
        # delta measures nothing (R1-05). Recording it here means no write-up can
        # claim a mechanistic contribution without the number that contradicts it.
        "encoder_cache": store.manifest(),
        "task_sizes_batches": task_sizes,
        "task_sampling_probabilities": dict(zip(task_names, task_probs.tolist())),
        "model_kwargs": model_kwargs,
        "parameters": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_checkpoint": ckpt_path if selected else None,
        f"best_val_{primary}_{primary_metric}": best_score,
        "loss_weight_saturation": criterion.saturated(),
        "test": test,
        "history": history,
        "wall_clock_hours": (time.time() - started) / 3600.0,
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        # Non-finite floats are serialised as null: json.dump would otherwise
        # emit the bare token `-Infinity`, which is invalid per RFC 8259 and
        # rejected by strict parsers (round-1 review, R1-25).
        json.dump(_json_safe(manifest), fh, indent=2)

    print(f"\n[train] best epoch {best_epoch}  best val {primary}.{primary_metric}"
          f"={best_score:.4f}  selected={selected}")
    for task, metrics in sorted(test.items()):
        print(f"[train] test {task}: {metrics}")
    print(f"[train] wrote {os.path.join(args.out, 'metrics.json')}")


if __name__ == "__main__":
    main()
