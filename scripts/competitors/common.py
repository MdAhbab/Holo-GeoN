"""
scripts/competitors/common.py
=============================
Shared contract and utilities for every competitor adapter (ThermoMPNN, SPURS,
RaSP, PROSTATA, DDGemb) so every baseline in the paper is trained on the exact
same split rows, evaluated with identical alignment checks, and scored by
``scripts/score_predictions.py``.

Design rules (from extras/kaggle-colab-lessons.md & BENCHMARK_PLAN.md):
  * NEVER insert a third-party repo at sys.path[0] (it can shadow our `src/`).
    Always append third-party paths or run isolated subprocesses with PYTHONPATH.
  * Every prediction file carries ``row_id`` (exact positional index in the
    splits/holdout parquet) and ``y_pred``, plus a companion ``.meta.json``
    recording sign convention, trainable parameters, peak GPU VRAM, training
    wall-clock time, inference latency, and git/library provenance.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_split_table(splits_path: str, split: str | None = None) -> pd.DataFrame:
    """Load a splits parquet with explicit positional ``row_id`` attached."""
    df = pd.read_parquet(splits_path)
    df = df.copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    if split is not None:
        df = df[df["split"] == split].copy()
    return df


def load_external_table(path: str) -> pd.DataFrame:
    """
    Load an external holdout table (FireProtDB / S669 / S1724 / Domainome) with
    explicit positional ``row_id`` and guaranteed normalised mutation columns
    (``parent_seq``, ``wt_seq``, ``site``, ``wt_aa``, ``mt_aa``, ``target``).
    """
    from src.windowing import MutationParseError, mutation_position

    df = pd.read_parquet(path).copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    if "wt_seq" not in df.columns and "parent_seq" in df.columns:
        df["wt_seq"] = df["parent_seq"]
    if "parent_seq" not in df.columns and "wt_seq" in df.columns:
        df["parent_seq"] = df["wt_seq"]
    if "wt_seq" in df.columns:
        df["wt_seq"] = df["wt_seq"].astype(str).str.upper()
    if "parent_seq" in df.columns:
        df["parent_seq"] = df["parent_seq"].astype(str).str.upper()
    if "target" not in df.columns and "ddg" in df.columns:
        df["target"] = df["ddg"].astype(np.float32)

    needs_site = any(c not in df.columns for c in ("site", "wt_aa", "mt_aa"))
    if needs_site and "wt_seq" in df.columns and "mt_seq" in df.columns:
        sites: list[int] = []
        wt_aas: list[str] = []
        mt_aas: list[str] = []
        keep: list[bool] = []
        wt_col = df["wt_seq"].astype(str).str.upper()
        mt_col = df["mt_seq"].astype(str).str.upper()
        for w, m in zip(wt_col, mt_col):
            try:
                pos = mutation_position(w, m)
                sites.append(pos)
                wt_aas.append(w[pos])
                mt_aas.append(m[pos])
                keep.append(True)
            except MutationParseError:
                keep.append(False)
        df = df.loc[keep].copy()
        df["site"] = np.asarray(sites, dtype=np.int64)
        df["wt_aa"] = wt_aas
        df["mt_aa"] = mt_aas
    return df


def clone_repo(url: str, dest: Path, rev: str | None = None) -> Path:
    """Clone a competitor repository if not already present (shallow when unpinned)."""
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    if rev:
        cmd = ["git", "clone", url, str(dest)]
    subprocess.run(cmd, check=True)
    if rev:
        subprocess.run(["git", "-C", str(dest), "checkout", rev], check=True)
    return dest


def write_predictions(
    out_path: Path,
    row_ids: np.ndarray,
    y_pred: np.ndarray,
    meta: dict,
) -> Path:
    """Validate finiteness and write common-format ``(row_id, y_pred)`` parquet + meta."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if len(row_ids) != len(y_pred):
        raise ValueError(f"misaligned lengths: {len(row_ids)} row_ids vs {len(y_pred)} preds")
    finite = np.isfinite(y_pred)
    if not bool(finite.all()):
        n_bad = int((~finite).sum())
        print(f"[competitor] WARNING: dropping {n_bad} non-finite prediction(s) before write")
        row_ids = row_ids[finite]
        y_pred = y_pred[finite]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_ids, "y_pred": y_pred}).to_parquet(out_path, index=False)

    full_meta = {
        **meta,
        "rows_written": int(len(row_ids)),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(full_meta, indent=2, default=str), encoding="utf-8")
    print(f"[competitor] wrote {len(row_ids):,} predictions -> {out_path}")
    return out_path


def gpu_peak_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    except Exception:
        pass
    return None
