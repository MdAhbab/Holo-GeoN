"""
scripts/figures/loaders.py
==========================
Artifact discovery and readers. Every reader fails FAST with a message that names
the file and what it is for, instead of letting a figure silently draw from nothing.

Results layout read by the builders (all optional; a builder whose inputs are absent
is skipped with a message, never faked):

    <results>/scores/<split>.json              scripts.score_predictions --out, split in {id30, id25, fold_tm60}
    <results>/ablations/summary.json           scripts.remote_job aggregate of kaggle_v2_ablations
                                               (or ablations.json from scripts.ablations)
    <results>/geometry/summary.json            same, for kaggle_v2_geometry
    <results>/calibration/<arm>_s<seed>.json   calibration.json of each run
                                               (or <arm>_s<seed>/calibration.json)
    <results>/epistasis/<arm>_s<seed>/epistasis_metrics.json  (+ pred_epistasis_test.parquet)
    <results>/efficiency/benchmark_speed.json  speed / VRAM benchmark (schema in result_figs.build_f9)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_VARIANTS = ("id30", "id25", "fold_tm60")
RUN_RE = re.compile(r"^(?P<arm>.+?)_s(?:eed)?(?P<seed>\d+)$")


class MissingArtifact(RuntimeError):
    """An input a builder needs is not there (yet)."""


@dataclass(frozen=True)
class Ctx:
    release: Path            # Holo-StabBench release folder (splits/, tables/, features/ ...)
    results: Path            # scorer / ablation / calibration / epistasis outputs
    figures: Path            # where figures are written
    tables: Path             # where tables are written
    demo: bool = False       # inputs are synthetic fixtures: watermark + _DEMO suffix
    primary_split: str = "id30"


def need(path, what: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise MissingArtifact(f"{what}: {p} not found")
    return p


def read_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_parquet(path, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def split_docs(ctx: Ctx) -> dict[str, dict]:
    """Scorer outputs found under <results>/scores/<split>.json, keyed by split."""
    out: dict[str, dict] = {}
    for v in SPLIT_VARIANTS:
        p = ctx.results / "scores" / f"{v}.json"
        if p.is_file():
            out[v] = read_json(p)
    return out


def find_runs(folder: Path, filename: str) -> list[tuple[str, int, Path]]:
    """(arm, seed, path) for '<arm>_s<seed>.json' files or '<arm>_s<seed>/<filename>' folders."""
    out: list[tuple[str, int, Path]] = []
    folder = Path(folder)
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        stem = p.stem if p.is_file() else p.name
        m = RUN_RE.match(stem)
        if not m:
            continue
        target = p if (p.is_file() and p.suffix == ".json") else (p / filename if p.is_dir() else None)
        if target is not None and target.is_file():
            out.append((m["arm"], int(m["seed"]), target))
    return out


# --- scorer output -> tidy frame ------------------------------------------------------
POOLED = ("pearson", "spearman", "rmse")
PER_PROTEIN = ("pearson", "spearman")


def _ci(e: dict, key: str):
    v = e.get(f"{key}_ci95")
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo, hi = v
        return (np.nan if lo is None else float(lo)), (np.nan if hi is None else float(hi))
    return np.nan, np.nan


def score_frame(doc: dict, split: str) -> pd.DataFrame:
    """One row per (model, group, metric): value and the cluster-bootstrap 95% interval.

    'pooled' and 'per-protein median' are separate groups on purpose: they are not
    comparable numbers (SPURS reports a per-protein median over 28 proteins,
    ThermoMPNN a pooled figure over 28,312 mutants) and must never share a column.
    """
    rows = []
    for model, e in (doc.get("models") or {}).items():
        for m in POOLED:
            lo, hi = _ci(e, m)
            rows.append({"split": split, "model": model, "group": "pooled", "metric": m,
                         "value": _f(e.get(m)), "lo": lo, "hi": hi, "n_proteins": np.nan})
        for m in PER_PROTEIN:
            key = f"per_protein_median_{m}"
            lo, hi = _ci(e, key)
            rows.append({"split": split, "model": model, "group": "per-protein median", "metric": m,
                         "value": _f(e.get(key)), "lo": lo, "hi": hi,
                         "n_proteins": _f(e.get("n_proteins_scored"))})
    return pd.DataFrame(rows)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --- paired ablation statistics -> one table -----------------------------------------
def _mde(sd, n, family) -> float:
    if not sd or not n or not family:
        return float("nan")
    from scripts.ablations import minimum_detectable_effect
    return float(minimum_detectable_effect(sd, n, family))


def _effect_row(arm: str, st: dict, arm_stats: dict | None, full_stats: dict | None) -> dict:
    p_holm = st.get("p_value_holm")
    n = st.get("n_seeds")
    mde = st.get("minimum_detectable_effect_80pct_power")
    if mde is None:
        mde = _mde(st.get("paired_diff_sd"), n, st.get("family_size"))
    sig = st.get("significant_after_holm")
    if sig is None and p_holm is not None:
        sig = bool(p_holm < 0.05)
    return {
        "arm": arm, "n_seeds": n, "delta": _f(st.get("delta_mean")),
        "lo": _f(st.get("ci95_low")), "hi": _f(st.get("ci95_high")),
        "p_holm": _f(p_holm), "family_size": st.get("family_size"), "mde": _f(mde),
        "significant": sig, "arm_mean": _f(st.get("arm_mean")), "arm_sd": _f(st.get("arm_sd")),
        "full_mean": _f(st.get("full_mean")), "full_sd": _f(st.get("full_sd")),
    }


def effect_table(doc: dict, metric: str) -> pd.DataFrame:
    """Normalise either aggregate format into one row per arm for one metric.

    * remote_job ``summary.json``: ``paired_vs_full[arm][metric]`` (metric e.g. 'test_pearson')
    * scripts.ablations ``ablations.json``: ``delta_vs_full[arm][task][m]`` (metric 'stability.pearson')
    """
    rows = []
    if "paired_vs_full" in doc:
        pv = doc.get("paired_vs_full") or {}
        for arm, per in pv.items():
            if arm == "unavailable" or not isinstance(per, dict) or metric not in per:
                continue
            rows.append(_effect_row(arm, per[metric], None, None))
    elif "delta_vs_full" in doc:
        task, _, m = metric.partition(".")
        for arm, tasks in (doc.get("delta_vs_full") or {}).items():
            st = ((tasks or {}).get(task) or {}).get(m)
            if st:
                rows.append(_effect_row(arm, st, None, None))
    return pd.DataFrame(rows)


def arm_means(doc: dict, metric: str) -> dict[str, tuple[float, float, int]]:
    """{arm: (mean, sd, n_seeds)} for 'test_pearson'-style metrics from a remote_job summary."""
    out = {}
    for arm, e in (doc.get("arms") or {}).items():
        v = e.get(metric) or {}
        mean = v.get("mean")
        if mean is not None and not (isinstance(mean, float) and math.isnan(mean)):
            out[arm] = (float(mean), _f(v.get("sd")), int(e.get("n_seeds") or 0))
    return out


def load_suite(ctx: Ctx, suite: str) -> dict:
    for name in ("summary.json", "ablations.json"):
        p = ctx.results / suite / name
        if p.is_file():
            return read_json(p)
    raise MissingArtifact(f"{suite} aggregate: no summary.json or ablations.json under {ctx.results / suite}")
