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
    NodeBudgetBatchSampler,
    ProteinBatchMutationDataset,
    SequenceDataset,
    collate_graphs,
    collate_protein_batch,
    filter_to_cache,
    load_sequence_table,
    move_batch,
)
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore
from src.heads import ThreeStateHead
from src.losses import (
    UncertaintyWeightedLoss,
    beta_gaussian_nll,
    gaussian_nll,
    pathogenicity_loss,
    sigma_gaussian_nll,
    three_state_loss,
)
from src.metrics import classification_metrics, regression_metrics
from src.provenance import provenance
from src.seeding import (
    loader_kwargs,
    reseed_loaders_for_epoch,
    restore_rng_state,
    rng_state,
    seed_everything,
)
from src.model import HoloGeoN, model_kwargs_from_config

# Tasks the multi-task loss knows about. A task appears in a step only when its
# loader yields a batch.
ALL_TASKS = ["stability", "three_state", "pathogenicity", "idr", "mfi", "proteomics"]

#: Rows of the TRAINING split scored each epoch to measure the train/val gap.
#: Fixed and deterministic, so the series is comparable across epochs; a
#: subsample because the point is the gap, not a precise training score.
TRAIN_EVAL_ROWS = 20000

#: Consecutive non-finite primary-metric epochs tolerated before aborting.
MAX_NAN_EPOCHS = 3

#: Consecutive epochs a task may sit at a log-variance clamp bound before the
#: run stops. Round-2 review, R2-07: `criterion.saturated()` recorded this in
#: the manifest and nothing acted on it, so a task pinned at the bound for 60
#: epochs - receiving either 400x or 1/400th of the shared gradient the whole
#: time - finished and reported its metrics like any other.
MAX_SATURATED_EPOCHS = 10

#: Fraction of held-out proteins allowed to have a near-identical training
#: protein before a whole-sequence split is refused. Zero: the grouping relation
#: is supposed to make this impossible, so any positive value is a defect in the
#: grouping rather than an acceptable rate (round-2 review, R2-04).
NEAR_DUPLICATE_TOLERANCE = 0.0

# Fallback experimental uncertainty for the three-state neutral band, used only
# when the stability table reports no per-measurement error. MegaScale's own
# reported ddG errors are preferred and are read from the `sigma` column.
#
# Round-2 review, R2-11: src/heads.py states that "the neutral band is not a
# hardcoded constant ... which is a noise model rather than an arbitrary
# threshold". With no `sigma` column - and the override is commented out for
# every shipped dataset - it is exactly a hardcoded constant, and it decides the
# class of ~21% of a realistic ddG distribution. Moving it from 0.2 to 1.0 moves
# that to 8.6% and 40.9%, taking the three-state macro-F1 and confusion matrix
# with it. It is now a config key (`three_state_sigma`), it is recorded in
# metrics.json, and a sensitivity row measures what it did.
DEFAULT_DDG_SIGMA = 0.5

#: Multipliers on the band used for the sensitivity row written to metrics.json.
BAND_SENSITIVITY_FACTORS = (0.4, 1.0, 2.0)


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
def _fork_workers() -> bool:
    """True when DataLoader workers are forked (they then inherit the parent's
    in-memory graph cache copy-on-write, so ``prewarm()`` in the parent is enough)."""
    import multiprocessing as mp
    method = mp.get_start_method(allow_none=True)
    if method is None:
        method = "fork" if sys.platform.startswith("linux") else "spawn"
    return method == "fork"


def _loader(dataset, batch_size: int, shuffle: bool, workers: int,
            seed: int = 42, collate_fn=collate_graphs) -> DataLoader:
    """A DataLoader whose shuffle order and worker RNG are reproducible.

    Workers are kept alive across epochs for the shuffling train loader (its
    in-worker graph cache survives) and - on platforms that cannot fork - for
    the evaluation loaders too, otherwise every epoch would re-spawn them and
    rebuild every parent graph from scratch. Where workers are forked the
    caller pre-warms the parent-graph cache instead (``prewarm_dataset``).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=workers > 0 and (shuffle or not _fork_workers()),
        **loader_kwargs(seed, workers),
    )


def prewarm_dataset(ds, workers: int, label: str = "") -> None:
    """Build every parent graph in THIS process before workers fork (see
    ``ProteinBatchMutationDataset.prewarm``). Pure caching: outputs unchanged."""
    if workers > 0 and _fork_workers() and hasattr(ds, "prewarm"):
        t0 = time.time()
        n = ds.prewarm()
        print(f"[data] prewarmed {n} parent graphs{(' (' + label + ')') if label else ''} "
              f"in {time.time() - t0:.1f}s; forked workers inherit them", flush=True)


def _sequence_loader(dataset, node_budget: int, max_batch: int, shuffle: bool,
                     workers: int, seed: int = 42) -> DataLoader:
    """
    A whole-sequence DataLoader batched by node budget rather than sample count.

    Round-2 review, R2-02: whole-sequence graphs span the entire protein, so at
    one shared ``batch_size`` they cost up to 8x the ddG batch and put the run
    past the documented 16 GB card. The budget is the same node count the
    windowed loaders spend (``batch_size * window``), which is the shape
    ``scripts/budget.py`` models, so no task can silently demand more than
    another.
    """
    sampler = NodeBudgetBatchSampler(
        [len(s) for s in dataset.df["parent_seq"]],
        node_budget=node_budget, max_batch_size=max_batch,
        shuffle=shuffle, seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        collate_fn=collate_graphs,
        persistent_workers=workers > 0,
        **{k: v for k, v in loader_kwargs(seed, workers).items()
           if k != "generator"},
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
    use_pb = bool(cfg.get("protein_batch", False)) and int(df["parent_seq"].str.len().max()) <= int(cfg.window)
    pb_prots = int(cfg.get("proteins_per_batch", 8))
    pb_muts = int(cfg.get("muts_per_protein", 64))

    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        counts[split] = int(len(part))
        if part.empty:
            continue
        noise = float(cfg.get("embed_noise_std", 0.0)) if split == "train" else 0.0
        cds_drop = float(cfg.get("cds_dropout", 0.0)) if split == "train" else 0.0
        if use_pb:
            ds = ProteinBatchMutationDataset(
                part, store, cfg, target=target,
                muts_per_protein=pb_muts, train=(split == "train"),
                noise_std=noise, cds_dropout=cds_drop, seed=int(cfg.seed),
            )
            prewarm_dataset(ds, cfg.num_workers, split)
            out[split] = _loader(
                ds, pb_prots, split == "train", cfg.num_workers, cfg.seed,
                collate_fn=collate_protein_batch,
            )
        else:
            ds = MutationDataset(
                part, store, cfg, target=target,
                noise_std=noise, cds_dropout=cds_drop,
            )
            out[split] = _loader(ds, batch_size, split == "train",
                                 cfg.num_workers, cfg.seed)
        if split == "train":
            n = min(TRAIN_EVAL_ROWS, len(part))
            sub = part.sample(n=n, random_state=int(cfg.seed))
            if use_pb:
                sub_ds = ProteinBatchMutationDataset(
                    sub, store, cfg, target=target,
                    muts_per_protein=pb_muts, train=False, seed=int(cfg.seed),
                )
                prewarm_dataset(sub_ds, cfg.num_workers, "train_eval")
                out["train_eval"] = _loader(
                    sub_ds, pb_prots, False, cfg.num_workers, cfg.seed,
                    collate_fn=collate_protein_batch,
                )
            else:
                out["train_eval"] = _loader(
                    MutationDataset(sub, store, cfg, target=target),
                    batch_size, False, cfg.num_workers, cfg.seed)
    info = report.as_dict()
    info["split_counts"] = counts
    info["protein_batch_active"] = bool(use_pb)
    info["window_coverage"] = window_coverage(df, int(cfg.window))
    return out, info


def window_coverage(df: pd.DataFrame, window: int) -> dict:
    """
    How often the site-centred crop covers the whole protein.

    Round-2 review, R2-14: the ``global_pool_readout`` arm is handed a
    ``window``-long crop centred on the mutation, so for a protein longer than
    the window the crop still encodes where the mutation is and the arm is not
    site-blind; for a protein shorter than the window it is. The arm therefore
    measures a different thing on each dataset, and the delta alone cannot
    separate them. This records the mix so the ablation table can say which.
    """
    if df.empty or "parent_seq" not in df:
        return {}
    lengths = df["parent_seq"].str.len()
    covered = int((lengths <= window).sum())
    return {
        "window": window,
        "rows": int(len(lengths)),
        "fraction_protein_shorter_than_window": round(covered / len(lengths), 4),
        "median_protein_length": int(lengths.median()),
        "note": (
            "rows where the protein is shorter than the window get a crop equal "
            "to the whole protein, so the global_pool_readout ablation arm is "
            "fully site-blind for them; for the rest the site-centred crop still "
            "encodes the mutation's location and the arm is not (R2-14)"
        ),
    }


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

    Round-2 review, R2-04: grouping by the exact parent sequence is not a
    homology relation for the three tables this path serves. An MFI table is a
    deep mutational scan - thousands of single mutants of one protein, every one
    a distinct sequence and therefore its own group - so ``verify_no_leakage``
    passed on a split where 100% of test rows had a training row 1-2 residues
    away. It now runs the same ``build_groups`` union-find the primary tasks use
    (identity clustering unioned with a declared parent/construct or gene
    column) and then AUDITS the result by comparing sequences across the
    boundary, which is the check that can actually fail.
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

    # Whole groups to one split, under the same relation the primary tasks use.
    groups, grouping = S.build_groups(
        df["parent_seq"].tolist(), declared=df["group"].tolist(),
        identity=cfg.identity, coverage=cfg.coverage,
    )
    row_splits = S.row_splits_from_groups(
        groups, ratios=(0.8, 0.1, 0.1), seed=cfg.seed
    )
    if not S.verify_no_leakage(groups, row_splits):
        raise SystemExit(f"{path}: leakage detected in the whole-sequence split")
    df = df.copy()
    df["split"] = row_splits

    # The check that can fail. verify_no_leakage above only re-states how the
    # groups were built; this measures the sequences themselves (R2-04).
    audit = S.near_duplicate_audit(
        df.loc[df["split"] == "train", "parent_seq"].tolist(),
        df.loc[df["split"] != "train", "parent_seq"].tolist(),
    )
    leaked = audit["fraction_with_near_train_neighbour"]
    grouping["near_duplicate_audit"] = audit
    if leaked > NEAR_DUPLICATE_TOLERANCE:
        raise SystemExit(
            f"{path}: {leaked:.1%} of held-out proteins have a near-identical "
            f"protein in train (estimated k-mer Jaccard >= "
            f"{audit['jaccard_threshold']}), above the "
            f"{NEAR_DUPLICATE_TOLERANCE:.0%} tolerance. Any metric this head "
            f"reports would be an interpolation score presented as "
            f"generalisation.\n"
            f"  grouping used: {report.notes.get('group_source')}\n"
            f"  Declare the parent protein each row is a variant of - a "
            f"parent/construct column, set under this dataset's key in "
            f"configs/data.yaml - so whole constructs go to one side. Installing "
            f"MMseqs2 additionally groups homologs. Examples of the overlap: "
            f"{audit['examples']}"
        )
    if leaked:
        print(f"[load] {path}: near-duplicate audit {leaked:.1%} of held-out "
              f"proteins (tolerance {NEAR_DUPLICATE_TOLERANCE:.0%})")

    out: dict[str, DataLoader] = {}
    counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        counts[name] = int(len(part))
        if part.empty:
            continue
        out[name] = _sequence_loader(
            SequenceDataset(part, store, cfg),
            # `or` not a get() default: the key exists and is null in the
            # shipped config, which get() would hand back as None.
            node_budget=int(cfg.get("sequence_node_budget")
                            or cfg.batch_size * cfg.window),
            max_batch=batch_size, shuffle=(name == "train"),
            workers=cfg.num_workers, seed=cfg.seed,
        )
    info = report.as_dict()
    info["split_counts"] = counts
    info["grouping"] = grouping
    info["node_cap"] = {
        "max_sequence_nodes": int(cfg.get("max_sequence_nodes", 2048)),
        "rows_over_cap": int(
            (df["parent_seq"].str.len() > int(cfg.get("max_sequence_nodes", 2048))).sum()
        ),
        "rows": int(len(df)),
        "note": (
            "rows whose protein exceeds the cap have their graph truncated, so "
            "the head sees a prefix. R1-07 asked for the cap to be applied and "
            "REPORTED; the per-sample `truncated` flag now reaches the batch and "
            "this counts it up front (R2, R1-07 follow-up)"
        ),
    }
    return out, info


# ---------------------------------------------------------------------------
# Losses for one batch of one task
# ---------------------------------------------------------------------------
def three_state_labels(batch: dict, fallback_sigma: float = DEFAULT_DDG_SIGMA
                       ) -> torch.Tensor:
    """
    Class labels from measured ddG and its per-measurement uncertainty.

    Rows with no reported error fall back to ``fallback_sigma``. A non-positive
    reported sigma is treated as missing rather than used: with sigma = 0 only
    an exactly-zero ddG is neutral, and with a negative sigma (a corrupt column)
    nothing is ever neutral and no error was raised (R2-11).
    """
    sigma = batch["sigma"]
    usable = torch.isfinite(sigma) & (sigma > 0)
    sigma = torch.where(usable, sigma, torch.full_like(sigma, fallback_sigma))
    return ThreeStateHead.labels_from_ddg(batch["y"], sigma)


def band_sensitivity(df: pd.DataFrame, fallback_sigma: float) -> dict:
    """
    How the three-state class balance moves with the neutral band (R2-11).

    The three-state macro-F1 is the only reported evidence that the
    swap-equivariant parameterisation buys anything, and it is a function of a
    constant that decides roughly a fifth of the labels. Reporting the class
    distribution at several bands turns that dependence into something a reader
    can see rather than something they would have to derive.

    Reads the split TABLE, not the loader: the labels are a function of
    ``target`` and ``sigma`` alone, so pulling them through the loader would
    rebuild every graph in the training split - once per band - to count
    integers.
    """
    if df is None or df.empty or "target" not in df:
        return {}
    y = torch.tensor(df["target"].to_numpy(dtype=float), dtype=torch.float32)
    if "sigma" in df:
        sigma = torch.tensor(
            pd.to_numeric(df["sigma"], errors="coerce").to_numpy(dtype=float),
            dtype=torch.float32)
    else:
        sigma = torch.full_like(y, float("nan"))

    # Scale the EFFECTIVE band, not the fallback.
    #
    # This loop used to multiply `fallback_sigma` only. The fallback applies
    # solely to rows whose own sigma is missing or non-positive - and on the
    # shipped MegaScale table every one of the 389,068 rows carries a finite
    # positive sigma (median 0.0513), so the fallback governs NO row and all
    # three factors produced byte-identical fractions under three different
    # `band_kcal_per_mol` labels. The table read as "the result is insensitive
    # to the band" when the band had never been varied.
    #
    # Scaling the per-row sigma as well makes the factor mean what its name
    # says, and the recorded band statistics say what the band actually was.
    usable = torch.isfinite(sigma) & (sigma > 0)
    effective = torch.where(usable, sigma, torch.full_like(sigma, fallback_sigma))

    out: dict = {
        "band_source": {
            "rows_using_their_own_sigma": int(usable.sum()),
            "rows_using_the_fallback": int((~usable).sum()),
            "fallback_sigma_kcal_per_mol": round(float(fallback_sigma), 4),
            "effective_band_median_kcal_per_mol": round(
                float(effective.median()), 4),
            "effective_band_p10_p90_kcal_per_mol": [
                round(float(effective.quantile(0.10)), 4),
                round(float(effective.quantile(0.90)), 4),
            ],
        }
    }
    for factor in BAND_SENSITIVITY_FACTORS:
        band = effective * factor
        labels = ThreeStateHead.labels_from_ddg(y, band)
        counts = np.bincount(labels.reshape(-1).numpy(), minlength=3)
        total = int(counts.sum()) or 1
        out[f"sigma_x{factor:g}"] = {
            "band_median_kcal_per_mol": round(float(band.median()), 4),
            "fraction_destabilizing": round(float(counts[0]) / total, 4),
            "fraction_neutral": round(float(counts[1]) / total, 4),
            "fraction_stabilizing": round(float(counts[2]) / total, 4),
        }
    return out


def task_losses(model: HoloGeoN, task: str, batch: dict,
                weights: dict | None = None,
                fallback_sigma: float = DEFAULT_DDG_SIGMA,
                nll_beta: float = 0.0) -> dict[str, torch.Tensor]:
    """Compute the loss terms this task contributes for one batch."""
    weights = weights or {}
    if task == "stability":
        z = model.context(batch)
        ddg, logvar = model.ddg(batch, z)
        losses = {"stability": beta_gaussian_nll(ddg, logvar, batch["y"], beta=nll_beta)}
        labels = three_state_labels(batch, fallback_sigma)
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


#: Cached parents opened to measure mechanistic coverage. A census would be
#: O(cache size) torch.loads at startup; a sample of this many is enough to tell
#: "the stream is fed" from "the stream is empty", which is the question.
MECH_COVERAGE_SAMPLE = 64


def measure_mech_coverage(loaders: dict, store: EncodingStore,
                          sample: int = MECH_COVERAGE_SAMPLE) -> dict:
    """
    Mechanistic coverage over the parents this run will read (R2-01).

    Distinct from the cache manifest, which describes whatever precompute last
    saw. This opens records belonging to the split tables in hand, so a claim
    about the mechanistic stream can be checked against the run that made it.
    """
    seen: list[str] = []
    for task in sorted(loaders):
        for split in ("train", "val", "test"):
            loader = loaders[task].get(split)
            if loader is None:
                continue
            df = getattr(loader.dataset, "df", None)
            if df is None or "parent_seq" not in df:
                continue
            for seq in df["parent_seq"].drop_duplicates():
                if seq not in seen:
                    seen.append(seq)
                if len(seen) >= sample:
                    break
            if len(seen) >= sample:
                break
        if len(seen) >= sample:
            break
    if not seen:
        return {}
    out = store.mech_coverage(seen)
    out["sampled"] = True
    out["sample_limit"] = sample
    return out


def compute_class_weights(loaders: dict, device,
                          fallback_sigma: float = DEFAULT_DDG_SIGMA) -> dict:
    """
    Inverse-frequency class weights from the training split.

    Reads the labels straight off each dataset's dataframe. ``fallback_sigma``
    is the three-state neutral band for rows with no reported error; the weights
    inherit it, so it travels here too (R2-11).

    This used to iterate the DataLoaders. That involves no model forward, which
    is what its docstring claimed made it "cheap" - but iterating the loader
    builds every batch in full: one cached parent record deserialised PER ROW
    (~2.2 MB for a ClinVar-sized protein), windowed, with both edge sets
    constructed. On the shipped tables that is a complete pass over ~460k rows,
    close to a terabyte of reads, before the first training step - and it is
    paid again by every baseline and by all 5 seeds x 5 arms of the ablation
    grid.

    Nothing in that work is used. The labels are columns on the dataframe the
    dataset already holds, and the rows are the same rows in the same order, so
    the counts are identical.
    """
    weights: dict = {}

    def frame(task: str) -> pd.DataFrame:
        return loaders[task]["train"].dataset.df

    if "stability" in loaders:
        df = frame("stability")
        y = torch.as_tensor(df["target"].to_numpy(dtype=np.float32))
        sigma = (torch.as_tensor(df["sigma"].to_numpy(dtype=np.float32))
                 if "sigma" in df.columns
                 else torch.full_like(y, float("nan")))
        labels = three_state_labels({"y": y, "sigma": sigma},
                                    fallback_sigma).reshape(-1).numpy()
        counts = np.bincount(labels, minlength=3)
        if counts.sum() and (counts > 0).all():
            inv = counts.sum() / (3.0 * counts)
            weights["three_state"] = torch.tensor(inv, dtype=torch.float32, device=device)

    if "pathogenicity" in loaders:
        y = frame("pathogenicity")["label"].to_numpy(dtype=np.float32)
        pos = int((y > 0.5).sum())
        neg = int((y <= 0.5).sum())
        if pos:
            weights["pathogenicity"] = torch.tensor(neg / pos, dtype=torch.float32,
                                                    device=device)
    return weights


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_task(model: HoloGeoN, task: str, loader: DataLoader, device,
                  fallback_sigma: float = DEFAULT_DDG_SIGMA) -> dict:
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
            ts_true.append(
                three_state_labels(batch, fallback_sigma).reshape(-1).cpu().numpy())
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
    ap.add_argument("--arch", choices=["v1", "v2"], default=None,
                    help="model architecture: v1 (legacy 20.6M) or v2 (Holo-GeoN v2 ~2.5M)")
    ap.add_argument("--nll-beta", type=float, default=None,
                    help="beta-NLL exponent (0.0 = standard NLL, 0.5 = v2 anti-collapse)")
    ap.add_argument("--cds-dropout", type=float, default=None,
                    help="training-only CDS/E_mech modality dropout probability")
    ap.add_argument("--no-esm-logits-prior", action="store_true",
                    help="ablation (v2): disable WT-marginal ESM log-odds anchor")
    ap.add_argument("--use-multi-layer", action="store_true",
                    help="ablation (v2): enable per-layer-LN multi-layer ESM scalar-mix")
    ap.add_argument("--no-multi-layer", action="store_true",
                    help="ablation (v2): use last ESM layer only (no scalar-mix)")
    ap.add_argument("--no-contact-burial", action="store_true",
                    help="ablation (v2): disable 6-ch contact-topology burial features")
    ap.add_argument("--use-str-stream", action="store_true",
                    help="GEOMETRIC_PLAN Arms B/C/D: enable invariant 3D coordinate stream")
    ap.add_argument("--str-topology-only", action="store_true",
                    help="GEOMETRIC_PLAN Arm D: 3-ch coordinate k-NN topology only")
    ap.add_argument("--true-contact-edges", action="store_true",
                    help="GEOMETRIC_PLAN Arm E: replace ESM contacts with real C-beta contacts")
    ap.add_argument("--true-contacts-dir", default=None,
                    help="directory of <sha1>.npy C-beta contact maps for Arm E")
    ap.add_argument("--protein-batch", dest="protein_batch", action="store_true", default=None,
                    help="enable Protein-Batch multi-mutation training (15-40x speedup)")
    ap.add_argument("--no-protein-batch", dest="protein_batch", action="store_false",
                    help="disable Protein-Batch training (row-level batching)")
    ap.add_argument("--no-mech-stream", action="store_true",
                    help="ablation: disable the E_mech message-passing stream")
    ap.add_argument("--no-ssm", action="store_true",
                    help="ablation: disable the selective state-space mixer")
    ap.add_argument("--no-geo-stream", action="store_true",
                    help="ablation: disable the E_geo (contact graph) stream")
    ap.add_argument("--global-pool-readout", action="store_true",
                    help="ablation: replace the site-anchored readout with the "
                         "v1 global-pool formulation")
    ap.add_argument("--strict-determinism", action="store_true",
                    help="deterministic kernels and cuDNN determinism: slower, "
                         "but makes a rerun bit-reproducible. scripts/ablations.py "
                         "sets it, because comparing arms across seeds is exactly "
                         "where run-to-run noise is the confound (R2-15)")
    ap.add_argument("--tag", default="main", help="name for this run's artifacts")
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        overrides={
            "arch": args.arch,
            "cache": args.cache, "esm_model": args.esm_model, "device": args.device,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_rows": args.max_rows, "num_workers": args.num_workers,
            "seed": args.seed,
            "nll_beta": args.nll_beta,
            "cds_dropout": args.cds_dropout,
            "protein_batch": args.protein_batch,
            "use_str_stream": True if args.use_str_stream else None,
            "str_topology_only": True if args.str_topology_only else None,
            "true_contact_edges": True if args.true_contact_edges else None,
            "true_contacts_dir": args.true_contacts_dir,
            "use_esm_logits_prior": False if args.no_esm_logits_prior else None,
            "use_multi_layer": (
                False if args.no_multi_layer else (True if args.use_multi_layer else None)
            ),
            "use_contact_burial": False if args.no_contact_burial else None,
            "strict_determinism": True if args.strict_determinism else None,
        },
    )
    os.makedirs(args.out, exist_ok=True)
    # Round-1 review, R1-01: nothing in the training path was seeded, and two
    # runs of the same command on identical data differed by 0.030 test Pearson
    # - larger than most ablation effects, which made every single-run
    # delta_vs_full uninterpretable.
    seed_everything(cfg.seed, strict=bool(cfg.get("strict_determinism", False)))
    # The three-state neutral band for rows with no reported measurement error.
    # Declared in the config and recorded in metrics.json, so the ~21% of labels
    # it decides is a stated choice rather than a constant in a module (R2-11).
    three_state_sigma = float(cfg.get("three_state_sigma") or DEFAULT_DDG_SIGMA)
    if three_state_sigma <= 0:
        raise SystemExit(
            f"three_state_sigma must be positive; got {three_state_sigma}. With "
            f"0 only an exactly-zero ddG is neutral, and with a negative value "
            f"nothing is ever neutral."
        )
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache,
                          cache_records=int(cfg.get("record_cache_size", 0) or 0))

    cache_manifest = store.manifest()
    mech_on = bool(cfg.use_mech_stream and not args.no_mech_stream)
    if mech_on:
        # Round-2 review, R2-01: this warning was gated on `cache_manifest` being
        # truthy, so a cache with no manifest.json - every cache that existed
        # when round 1 ran - took the SILENT branch. The loud case was exactly
        # backwards: a cache that cannot describe itself is the one a reported
        # run must not read quietly. Missing manifest now warns; a manifest that
        # reports zero coverage warns; only a manifest reporting real coverage
        # is silent.
        if not cache_manifest:
            print(f"[train] WARNING: the encoder cache at {cfg.cache} has no "
                  "manifest.json, so nothing can say whether the mechanistic "
                  "stream is fed. E_mech may be self-loops only, in which case "
                  "the stream contributes nothing and its ablation delta is a "
                  "plumbing artifact. Re-run `python -m scripts.precompute_"
                  "embeddings` (it writes a manifest without re-encoding what "
                  "is already cached), or train with --no-mech-stream.")
        elif not cache_manifest.get("cds_active"):
            # Not fatal — a geometry-only run is a legitimate configuration — but
            # it must not be mistaken for a dual-stream one. R1-05 was exactly
            # this confusion, invisible because nothing printed or recorded it.
            print("[train] WARNING: the mechanistic stream is enabled but the "
                  "encoder cache has no records with populated mechanistic "
                  f"channels (cds_coverage={cache_manifest.get('cds_coverage')}). "
                  "E_mech will hold self-loops only, the stream contributes "
                  "nothing, and its ablation delta will read ~0 for plumbing "
                  "reasons rather than scientific ones. Run `python run.py etl "
                  "--cds-source ...` and re-run precompute, or train with "
                  "--no-mech-stream and say so.")

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
    # Round-2 review, R2-01: measure the mechanistic stream's coverage on the
    # records this run will actually read, rather than trusting a manifest that
    # may predate them. Sampled, because opening every record on a 10^4-parent
    # cache is not worth the startup cost; the sample size travels with the
    # number so nobody reads it as a census.
    mech_coverage: dict = {}
    if mech_on:
        mech_coverage = measure_mech_coverage(loaders, store)
        if mech_coverage and mech_coverage.get("cds_active") == 0:
            print("[train] WARNING: sampled "
                  f"{mech_coverage['parents_present_in_cache']} cached parent(s) "
                  "this run will read; NONE has populated mechanistic channels. "
                  "E_mech is self-loops only for them, so --no-mech-stream would "
                  "train the same model. Do not report a mechanistic contribution "
                  "from this run.")
        elif mech_coverage:
            print(f"[train] mechanistic coverage (sampled from the cache): "
                  f"{mech_coverage['cds_coverage']:.3f} over "
                  f"{mech_coverage['parents_present_in_cache']} parent(s)")

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
        arch=cfg.get("arch", "v1"),
        use_ssm=cfg.use_ssm and not args.no_ssm,
        use_mech_stream=cfg.use_mech_stream and not args.no_mech_stream,
        use_geo_stream=cfg.use_geo_stream and not args.no_geo_stream,
        use_str_stream=bool(cfg.get("use_str_stream", False)),
        str_topology_only=bool(cfg.get("str_topology_only", False)),
        use_multi_layer=bool(cfg.get("use_multi_layer", True)),
        use_contact_burial=bool(cfg.get("use_contact_burial", True)),
        use_esm_logits_prior=bool(cfg.get("use_esm_logits_prior", True)),
        site_anchored_readout=not args.global_pool_readout,
    )
    model = HoloGeoN(**model_kwargs).to(device)
    nll_beta = float(cfg.get("nll_beta", 0.0))
    print(f"[train] parameters: {model.parameter_count()}  (nll_beta={nll_beta}, "
          f"cds_dropout={float(cfg.get('cds_dropout', 0.0))}, "
          f"protein_batch={bool(cfg.get('protein_batch', False))})")

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
    class_weights = compute_class_weights(loaders, device, three_state_sigma)
    if class_weights:
        print("[train] class balancing: "
              + ", ".join(f"{k}={v.tolist() if v.numel() > 1 else float(v):.4}"
                          if v.numel() == 1 else f"{k}={[round(x, 3) for x in v.tolist()]}"
                          for k, v in class_weights.items()))
    # Weight decay applies to WEIGHT MATRICES ONLY.
    #
    # This was one flat parameter group, so `weight_decay` also decayed every
    # LayerNorm gain, every bias, and - worst - the Kendall & Gal `log_var`
    # parameters in `criterion`. Decaying a loss-weight parameter pulls each
    # task weight toward exp(0) = 1; there is no prior that justifies shrinking
    # it, and it means the learned weighting is not the stationary point the
    # method defines.
    #
    # At the old `weight_decay: 1e-4` the whole effect was a 0.8% shrink across
    # 270k steps, so it changed nothing and hid the defect. That value was also
    # not regularising anything: AdamW's decoupled decay is `p -= lr*wd*p`, i.e.
    # 3e-8 per step at lr 3e-4. Raising it to a value that actually constrains a
    # 20.6M-parameter model fitting only 322 distinct proteins is what makes
    # this separation load-bearing rather than cosmetic.
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    loss_weights = [p for p in criterion.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": loss_weights, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    # Gradient clipping still spans every trainable tensor.
    params = decay + no_decay + loss_weights
    print(f"[train] weight decay {cfg.weight_decay:g} on {len(decay)} weight "
          f"tensors; 0.0 on {len(no_decay)} norm/bias tensors and "
          f"{len(loss_weights)} task-weight parameters")

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

    # Round-2 review, R2-10: `steps_per_epoch = max(len(loader))` stopped
    # denoting a pass over anything once tasks were sampled proportionally. At
    # MegaScale/ClinVar scale and the shipped temperature 0.5, the PRIMARY task
    # saw 63% of its own training batches per "epoch", so `epochs: 60` bought
    # 37.7 passes over the ddG data, and `early_stop_patience: 8` counted units
    # whose size depended on the OTHER tasks' row counts - adding a proteomics
    # corpus silently changed what an epoch meant for stability.
    #
    # An epoch is now one expected pass over the primary task: it is sampled
    # with probability p, so it takes len(loader)/p steps to see its own data
    # once. Every other task is then seen in proportion to its sampling weight,
    # which is the point of the schedule.
    primary_for_epoch = "stability" if "stability" in loaders else task_names[0]
    p_primary = float(task_probs[task_names.index(primary_for_epoch)])
    steps_per_epoch = max(1, int(round(task_sizes[primary_for_epoch] / max(p_primary, 1e-9))))
    total_steps = steps_per_epoch * cfg.epochs
    print(f"[train] task sizes (batches): {task_sizes}")
    print(f"[train] sampling probabilities (temperature={temperature}): "
          + ", ".join(f"{t}={p:.3f}" for t, p in zip(task_names, task_probs)))
    print(f"[train] steps_per_epoch={steps_per_epoch} = one expected pass over "
          f"{primary_for_epoch!r} (its {task_sizes[primary_for_epoch]} batches "
          f"at sampling probability {p_primary:.3f}). `epochs: {cfg.epochs}` "
          f"therefore means {cfg.epochs} passes over the primary task, and "
          f"early stopping counts those.")
    replay = {t: (steps_per_epoch * float(task_probs[i]) / max(task_sizes[t], 1))
              for i, t in enumerate(task_names)}
    heavy = {t: round(r, 1) for t, r in replay.items() if r > 5.0}
    if heavy:
        print(f"[train] NOTE: small tables replayed per epoch: {heavy}. A table "
              f"seen many times per epoch overfits long before the primary task "
              f"has converged; its head's metrics should be read with that in "
              f"mind.")
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
    saturated_epochs = 0
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
        saturated_epochs = int(state.get("saturated_epochs", 0))
        history = state.get("history", [])
        rng_ok = restore_rng_state(state.get("rng"))
        print(f"[train] resumed from epoch {state['epoch']} (best {primary_metric}"
              f"={best_score:.4f}, patience={patience}, rng_restored={rng_ok})")
        if not rng_ok:
            print("[train] NOTE: RNG state was not restored, so dropout masks "
                  "and any other global-RNG draw differ from an uninterrupted "
                  "run. The DATA order is unaffected: it is derived from "
                  "(seed, epoch) rather than from accumulated generator "
                  "state (R2-12).")

    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        # The batch order for THIS epoch, derived from (seed, epoch) alone, so
        # a resume reproduces it exactly rather than restarting the loaders
        # from epoch 1 (R2-12).
        reseed_loaders_for_epoch(loaders, cfg.seed, epoch)
        for v in loaders.values():
            ds = getattr(v.get("train"), "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
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
                losses.update(task_losses(model, task, batch, class_weights,
                                          three_state_sigma, nll_beta=nll_beta))
                total, parts = criterion(losses)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.step(opt)
            scaler.update()
            if (not use_amp) or (scaler.get_scale() >= scale_before):
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
        # The weight each task's gradient is actually multiplied by. R2-07: the
        # objective's per-task weighting was only inferable from the loss parts,
        # and an 8.5x imbalance created by the floors rather than by the data was
        # invisible in the log.
        train_log["task_weights"] = criterion.current_weights()
        val = {
            t: evaluate_task(model, t, v["val"], device, three_state_sigma)
            for t, v in loaders.items() if "val" in v
        }

        # The train/val gap, measured rather than inferred.
        #
        # Nothing in this run's artifacts could show overfitting: the only
        # training number recorded was a sum of three differently-scaled losses,
        # one of them a Gaussian NLL that is unbounded below and log-scale, so
        # "train loss fell 64%" was not comparable to any validation quantity.
        # The gap had to be reverse-engineered from the predicted sigma after
        # the fact. This scores the SAME metric on a fixed subsample of the
        # training split, so train and val sit side by side in `history`.
        #
        # A fixed subsample (not the full split) keeps the cost to a few percent
        # of an epoch; the same rows every epoch, so the series is comparable.
        if primary in loaders and "train_eval" in loaders[primary]:
            train_metrics = evaluate_task(model, primary,
                                          loaders[primary]["train_eval"],
                                          device, three_state_sigma)
            train_log["eval_on_train_subsample"] = train_metrics
            v_rmse = val.get(primary, {}).get("rmse")
            t_rmse = train_metrics.get("rmse")
            if v_rmse and t_rmse:
                train_log["generalisation_gap_rmse"] = round(v_rmse / t_rmse, 4)

        # Is the predicted uncertainty honest on held-out data? sigma is fit on
        # TRAINING residuals, so this ratio grows as the mean channel memorises:
        # it went 1.30 -> 2.39 over 11 epochs of the previous run while nothing
        # in the log said so. mu +/- 1.96*sigma is only a real interval near 1.0.
        v_stab = val.get("stability", {})
        if v_stab.get("mean_predicted_sigma"):
            v_stab["rmse_over_predicted_sigma"] = round(
                v_stab["rmse"] / v_stab["mean_predicted_sigma"], 4)

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
        saturated_now = criterion.saturated()
        if saturated_now:
            saturated_epochs += 1
            print(f"[train] WARNING: task log-variance at a clamp bound: "
                  f"{saturated_now} ({saturated_epochs} epoch(s) in a row). "
                  f"Those tasks are pinned at the extreme of the weight range, "
                  f"so the objective is no longer balancing them.")
            if saturated_epochs >= MAX_SATURATED_EPOCHS:
                raise SystemExit(
                    f"{saturated_now} have sat at a log-variance clamp bound for "
                    f"{saturated_epochs} consecutive epochs. The multi-task "
                    f"weighting has stopped adapting, and whichever task is "
                    f"pinned at the low bound is receiving 1/400th of the shared "
                    f"gradient the others get. Aborting rather than finishing a "
                    f"run whose 'multi-task' objective is effectively "
                    f"single-task. Re-scale that task's loss, drop it from the "
                    f"run, or widen TASK_LOGVAR_MIN/MAX deliberately."
                )
        else:
            saturated_epochs = 0

        record = {"epoch": epoch, "train": train_log, "val": val,
                  "lr": sched.get_last_lr()[0]}
        history.append(record)

        # Every task reports ITS OWN headline metric. This line used to print
        # the primary task's metric for the primary task and the literal row
        # count `n` for every other one, so a ClinVar AUROC of 0.95 was computed
        # each epoch, written into `history` and metrics.json, and never shown -
        # the second of the two trained tasks looked like it was reporting
        # nothing. The per-task metric is the one already declared in
        # `ablation_metrics`, so the log agrees with what the ablation grid
        # compares.
        headline = cfg.get("ablation_metrics", {}) or {}
        parts = []
        for t in sorted(val):
            key = primary_metric if t == primary else headline.get(t, "n")
            value = val[t].get(key)
            if value is None:                       # metric absent for this task
                key, value = "n", val[t].get("n", float("nan"))
            parts.append(f"{t}:{key}={value:.4f}" if isinstance(value, float)
                         else f"{t}:{key}={value}")
        summary = "  ".join(parts)
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
                 "saturated_epochs": saturated_epochs,
                 "rng": rng_state(),
                 "history": history},
                last_path,
            )
        except Exception as exc:
            # Round-2 review, R2-12: this printed one line and continued, so a
            # disk-full at epoch 3 of 60 was discovered at the end - with no
            # resumable state written for 57 epochs. The resume checkpoint is
            # the only thing that makes a 2-3 day run survivable, so losing it
            # stops the run.
            raise SystemExit(
                f"could not write the resume checkpoint {last_path} at epoch "
                f"{epoch}: {exc}. Continuing would leave the run unresumable "
                f"while it looks healthy. Free space (or fix permissions) and "
                f"restart - the best checkpoint so far is at {ckpt_path}."
            ) from exc

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
            t: evaluate_task(model, t, v["test"], device, three_state_sigma)
            for t, v in loaders.items() if "test" in v
        }
    else:
        test = {}
        print("[train] no checkpoint was ever selected, so no test metrics are "
              "reported. metrics.json records best_checkpoint: null.")

    manifest = {
        "tag": args.tag,
        "config": cfg.as_dict(),
        # Which code, on which machine, with which packages (R2-15).
        "provenance": provenance(),
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
        # An empty dict means the cache carried no manifest - which is itself the
        # answer to "was the mechanistic stream fed?", namely "unknown" (R2-01).
        "encoder_cache": cache_manifest or {"status": "no manifest.json in the cache"},
        # Measured here rather than trusted from the manifest: the manifest can
        # predate the records. Empty when no mech-stream run needs it.
        "encoder_cache_measured": mech_coverage,
        "task_sizes_batches": task_sizes,
        "task_sampling_probabilities": dict(zip(task_names, task_probs.tolist())),
        "model_kwargs": model_kwargs,
        "parameters": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_checkpoint": ckpt_path if selected else None,
        f"best_val_{primary}_{primary_metric}": best_score,
        "three_state_band": {
            "fallback_sigma_kcal_per_mol": three_state_sigma,
            "n_sigma": 1.0,
            "note": (
                "rows carrying a positive `sigma` use their own reported error; "
                "the rest use this fallback. It decides the class of a large "
                "fraction of a realistic ddG distribution, so the sensitivity "
                "table below says how large (R2-11)"
            ),
            "sensitivity": band_sensitivity(
                loaders["stability"]["train"].dataset.df, three_state_sigma
            ) if "stability" in loaders else {},
        },
        "loss_weight_saturation": criterion.saturated(),
        "loss_task_weights_final": criterion.current_weights(),
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
