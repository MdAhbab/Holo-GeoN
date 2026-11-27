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
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_MUT_CODE_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def _clean_seq_series(s: pd.Series) -> pd.Series:
    return s.where(s.notna(), "").astype(str).str.strip().str.upper()


def safe_aa_index(aa) -> int:
    """Index of a one-letter amino acid; tolerates ``None``/``NaN`` and non-canonical residues."""
    from src.substitution import UNKNOWN_AA_IDX, aa_index

    if aa is None or (isinstance(aa, float) and not np.isfinite(aa)):
        return UNKNOWN_AA_IDX
    try:
        if pd.isna(aa):
            return UNKNOWN_AA_IDX
    except (TypeError, ValueError):
        pass
    s = str(aa).strip()
    if not s:
        return UNKNOWN_AA_IDX
    return aa_index(s)


def valid_site_mask(site_series: pd.Series, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(boolean_mask, valid_int64_sites)`` keeping only finite integer
    0-based sites strictly inside ``[0, seq_len)``.
    """
    raw = pd.to_numeric(site_series, errors="coerce").to_numpy(dtype=np.float64)
    if seq_len <= 0 or raw.size == 0:
        return np.zeros((raw.size,), dtype=bool), np.empty((0,), dtype=np.int64)
    mask = np.isfinite(raw) & (raw == np.floor(raw)) & (raw >= 0) & (raw < seq_len)
    return mask, raw[mask].astype(np.int64)


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Warning-free Pearson correlation returning ``nan`` on empty/short/constant inputs."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.size != p.size or y.size < 2:
        return float("nan")
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 2:
        return float("nan")
    y = y[mask]
    p = p[mask]
    ys = y - y.mean()
    ps = p - p.mean()
    denom = float(np.sqrt(np.dot(ys, ys) * np.dot(ps, ps)))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.clip(np.dot(ys, ps) / denom, -1.0, 1.0))


def parse_extra_external_spec(spec: str) -> tuple[str, str, str | None]:
    """
    Parse ``--extra-externals`` items formatted as ``tag=parquet`` or
    ``tag=parquet::pdb_dir`` into ``(tag, parquet_path, pdb_dir_or_None)``.
    """
    if "=" not in spec:
        raise ValueError(
            f"invalid --extra-externals spec {spec!r}; expected 'tag=parquet' or 'tag=parquet::pdb_dir'"
        )
    tag, rest = spec.split("=", 1)
    tag = tag.strip()
    if "::" in rest:
        epq, epdb = rest.split("::", 1)
        epdb_clean = epdb.strip() or None
    else:
        epq, epdb_clean = rest, None
    return tag, epq.strip(), epdb_clean


def lookup_store_seq(store, seq: str) -> str | None:
    """Resolve ``seq`` in ``EncodingStore`` (normalized uppercase first, then raw fallback)."""
    raw = "" if seq is None else str(seq)
    norm = raw.strip().upper()
    if norm and store.has(norm):
        return norm
    if raw and raw != norm and store.has(raw):
        return raw
    return None


def row_id_array(df: pd.DataFrame) -> np.ndarray:
    """Extract ``row_id`` int64 array, falling back to DataFrame index if absent."""
    if "row_id" in df.columns:
        return df["row_id"].to_numpy(dtype=np.int64)
    return df.index.to_numpy(dtype=np.int64)


def target_array(df: pd.DataFrame) -> np.ndarray:
    """Extract float32 targets from ``target``/``ddg``/``ddG``/``normalized_fitness``, or NaN."""
    for col in ("target", "ddg", "ddG", "normalized_fitness"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
    return np.full((len(df),), np.nan, dtype=np.float32)


def match_group_pdb(parent, grp: pd.DataFrame, pdb_map: dict[str, Path], matcher) -> Path | None:
    """
    Match a protein group to a PDB path, checking ``parent`` first and falling
    back to ``uniprot_id`` / ``wt_pdb`` (e.g. Domainome ``A0PJY2_PF00096_289`` -> ``A0PJY2``).
    """
    if not pdb_map:
        return None
    cands: list[str] = []
    if parent is not None and not (isinstance(parent, float) and not np.isfinite(parent)):
        s = str(parent).strip()
        if s and s.lower() != "nan":
            cands.append(s)
    for col in ("uniprot_id", "wt_pdb"):
        if col in grp.columns and len(grp) > 0:
            val = grp[col].iloc[0]
            if pd.notna(val):
                s = str(val).strip()
                if s and s.lower() != "nan" and s not in cands:
                    cands.append(s)
    for cand in cands:
        hit = matcher(cand, pdb_map)
        if hit is not None:
            return hit
    return None


def load_split_table(splits_path: str, split: str | None = None) -> pd.DataFrame:
    """Load a splits parquet with explicit positional ``row_id`` attached."""
    df = pd.read_parquet(splits_path)
    df = df.copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    if "parent_seq" not in df.columns and "wt_seq" in df.columns:
        df["parent_seq"] = df["wt_seq"]
    if "wt_seq" not in df.columns and "parent_seq" in df.columns:
        df["wt_seq"] = df["parent_seq"]
    if "parent_seq" in df.columns:
        df["parent_seq"] = _clean_seq_series(df["parent_seq"])
    if "wt_seq" in df.columns:
        df["wt_seq"] = _clean_seq_series(df["wt_seq"])
    if "target" not in df.columns:
        for col in ("ddg", "ddG"):
            if col in df.columns:
                df["target"] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
                break
    if split is not None and "split" in df.columns:
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
        df["wt_seq"] = _clean_seq_series(df["wt_seq"])
    if "parent_seq" in df.columns:
        df["parent_seq"] = _clean_seq_series(df["parent_seq"])
    if "target" not in df.columns:
        for col in ("ddg", "ddG", "normalized_fitness"):
            if col in df.columns:
                df["target"] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
                break
        else:
            df["target"] = np.full(len(df), np.nan, dtype=np.float32)

    needs_site = any(c not in df.columns for c in ("site", "wt_aa", "mt_aa"))
    if needs_site and "wt_seq" in df.columns and "mt_seq" in df.columns:
        sites: list[int] = []
        wt_aas: list[str] = []
        mt_aas: list[str] = []
        keep: list[bool] = []
        wt_col = df["wt_seq"]
        mt_col = _clean_seq_series(df["mt_seq"])
        for w, m in zip(wt_col, mt_col):
            if not w or not m:
                keep.append(False)
                continue
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
    elif needs_site and "wt_seq" in df.columns:
        mut_col = next((c for c in ("mutation", "substitution") if c in df.columns), None)
        if mut_col is not None:
            sites = []
            wt_aas = []
            mt_aas = []
            keep = []
            for w, code in zip(df["wt_seq"], df[mut_col]):
                if not w or pd.isna(code):
                    keep.append(False)
                    continue
                m = _MUT_CODE_RE.match(str(code).strip().upper())
                if not m:
                    keep.append(False)
                    continue
                wa, pos1, ma = m.group(1), int(m.group(2)), m.group(3)
                idx = pos1 - 1
                if 0 <= idx < len(w) and w[idx] == wa and wa != ma:
                    sites.append(idx)
                    wt_aas.append(wa)
                    mt_aas.append(ma)
                    keep.append(True)
                else:
                    keep.append(False)
            df = df.loc[keep].copy()
            df["site"] = np.asarray(sites, dtype=np.int64)
            df["wt_aa"] = wt_aas
            df["mt_aa"] = mt_aas

    for col in ("wt_aa", "mt_aa"):
        if col in df.columns:
            df[col] = _clean_seq_series(df[col])
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
    row_ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
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
