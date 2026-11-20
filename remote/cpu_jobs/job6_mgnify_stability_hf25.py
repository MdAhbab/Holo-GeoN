"""
remote/cpu_jobs/job6_mgnify_stability_hf25.py
===========================================
Kaggle CPU Worker 6 (Account: mdahbabhamidkhan) — MGnify Stability (Zenodo 19411306)
Streaming Cleaning, Deduplication, and <25%-Identity Leakage Filtering.

Downloads into /tmp/holo_job6/ (1 TB free on Kaggle CPU):
  1. ``230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv`` (2.23 GB)
  2. ``dmsv4_filtered_train_splits.csv`` (60.0 MB)
  3. ``mgnify_training_index.csv`` (137.9 MB)

Pipeline:
  * Inspects exact CSV headers dynamically and records all raw columns in
    ``job6_summary.json``.
  * Streams the 2.23 GB recalibrated CSV in chunks (``chunksize=200_000``),
    keeping only essential columns (sequence, identifier/name, recalibrated dG,
    dG uncertainty/CI if present, library/split/domain tags) for rows with
    finite dG and canonical 20-AA sequences of length >= 20, deduplicating by
    exact ``aa_seq`` (averaging finite dG across replicates).
  * Builds the heldout/external exclusion FASTA from ALL heldout proteins in
    ``/kaggle/input`` (``holo-geon-v2-gpu-seed``):
      - ``val`` + ``test`` parents of ``splits_megascale.parquet``,
        ``splits_megascale_id25.parquet``, ``splits_megascale_fold_tm60.parquet``
      - ALL parents of ``fireprotdb_holdout.parquet`` and ``s669_holdout.parquet``
  * Installs static MMseqs2 Linux AVX2 into ``/tmp/holo_job6/`` and runs
    ``mmseqs easy-search`` (``-s 5.7``, ``--min-seq-id 0.20``, ``-e 10``,
    ``--threads 4``), excluding any MGnify sequence with ``fident >= 0.25`` and
    ``max(qcov, tcov) >= 0.50``, PLUS excluding all exact-match and
    exact-substring hits.
  * Deletes large temporary downloads/intermediates and emits compact outputs in
    ``/kaggle/working``:
      - ``mgnify_stability_clean_hf25.parquet``
      - ``mgnify_leakage_audit.json``
      - ``job6_summary.json``
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job6")
TMP = Path("/tmp/holo_job6") if Path("/tmp").is_dir() else WORK / "_tmp"
WORK.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Holo-GeoN/2.0"
Z95 = 1.959963985
CANONICAL_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{20,}$")

ZENODO_FILES = {
    "concat": {
        "filename": "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv",
        "url": "https://zenodo.org/api/records/19411306/files/230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv/content",
    },
    "dmsv4_splits": {
        "filename": "dmsv4_filtered_train_splits.csv",
        "url": "https://zenodo.org/api/records/19411306/files/dmsv4_filtered_train_splits.csv/content",
    },
    "mgnify_index": {
        "filename": "mgnify_training_index.csv",
        "url": "https://zenodo.org/api/records/19411306/files/mgnify_training_index.csv/content",
    },
}


def ensure_packages() -> None:
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow")):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[job6] Installing missing packages: {missing}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def find_input(name: str, roots=("/kaggle/input", ".")) -> Path | None:
    """Locate an input file by name, flattening Kaggle's X/X extracted folder."""
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for hit in rp.rglob(name):
            if hit.is_file():
                return hit
            if hit.is_dir():
                inner = hit / name
                if inner.is_file():
                    return inner
                if name.endswith(".parquet") and list(hit.glob("*.parquet")):
                    return hit
    return None


def file_hashes(path: Path) -> dict[str, object]:
    h_sha = hashlib.sha256()
    h_md5 = hashlib.md5()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h_sha.update(chunk)
            h_md5.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": h_sha.hexdigest(), "md5": h_md5.hexdigest()}


def download_file(url: str, dest: Path, tries: int = 5, timeout: float = 180.0) -> dict[str, object]:
    """Stream-download a file with retry, User-Agent, and Content-Length check."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None

    for attempt in range(1, tries + 1):
        try:
            if part.exists():
                part.unlink()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            h_sha = hashlib.sha256()
            h_md5 = hashlib.md5()
            written = 0
            expected_len: int | None = None
            t_start = time.time()
            next_log = 250 * 1024 * 1024

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit():
                    expected_len = int(cl)
                with open(part, "wb") as out_fh:
                    while True:
                        buf = resp.read(8 * 1024 * 1024)
                        if not buf:
                            break
                        out_fh.write(buf)
                        h_sha.update(buf)
                        h_md5.update(buf)
                        written += len(buf)
                        if written >= next_log:
                            mb = written / 1e6
                            dt = max(time.time() - t_start, 0.1)
                            print(
                                f"  [dl] {dest.name}: {mb:.1f} MB ({mb / dt:.1f} MB/s)",
                                flush=True,
                            )
                            next_log += 250 * 1024 * 1024

            if expected_len is not None and written != expected_len:
                raise RuntimeError(
                    f"Incomplete download for {dest.name}: got {written} bytes, expected {expected_len}"
                )
            if written == 0:
                raise RuntimeError(f"Empty download for {dest.name}")

            os.replace(part, dest)
            mb_total = written / 1e6
            dt_total = max(time.time() - t_start, 0.1)
            print(
                f"[job6] Downloaded {dest.name}: {mb_total:.2f} MB in {dt_total:.1f}s",
                flush=True,
            )
            return {
                "url": url,
                "bytes": written,
                "sha256": h_sha.hexdigest(),
                "md5": h_md5.hexdigest(),
                "elapsed_s": round(dt_total, 2),
            }
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                break
        except Exception as exc:
            last_exc = exc
        if attempt < tries:
            pause = min(60.0, 5.0 * (2 ** (attempt - 1)))
            print(
                f"  [dl] Retry {attempt}/{tries} failed for {dest.name} ({last_exc}); "
                f"sleeping {pause:.0f}s",
                flush=True,
            )
            time.sleep(pause)

    if part.exists():
        part.unlink()
    raise RuntimeError(f"Download failed for {url}: {last_exc}")


def read_csv_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
    return [str(c) for c in header]


def pick_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        hit = by_lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def install_mmseqs() -> str:
    mm_dir = TMP / "mmseqs_install"
    mm_bin = mm_dir / "mmseqs" / "bin" / "mmseqs"
    if mm_bin.is_file():
        return str(mm_bin)
    mm_dir.mkdir(parents=True, exist_ok=True)
    url = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
    tgz = TMP / "mmseqs-linux-avx2.tar.gz"
    download_file(url, tgz, tries=4, timeout=120.0)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(mm_dir)
    if tgz.exists():
        tgz.unlink()
    if not mm_bin.is_file():
        found = list(mm_dir.rglob("mmseqs"))
        mm_bin = next((p for p in found if p.is_file() and p.parent.name == "bin"), mm_bin)
    os.chmod(mm_bin, 0o755)
    ver = subprocess.check_output([str(mm_bin), "version"], text=True).strip()
    print(f"[job6] MMseqs2 version: {ver}", flush=True)
    return str(mm_bin)


def inspect_aux_csv(path: Path, header: list[str]) -> tuple[dict[str, object], set[str]]:
    import pandas as pd

    seq_col = pick_col(header, ("aa_seq", "seq", "sequence", "wt_seq"))
    split_col = pick_col(header, ("split", "train_split"))
    usecols = [c for c in (seq_col, split_col) if c is not None]

    total_rows = 0
    split_counts: dict[str, int] = {}
    seqs: set[str] = set()

    if usecols:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000, low_memory=False):
            total_rows += len(chunk)
            if seq_col:
                s_vals = chunk[seq_col].dropna().astype(str).str.strip().str.upper()
                seqs.update(v for v in s_vals if CANONICAL_AA_RE.match(v))
            if split_col:
                vc = chunk[split_col].fillna("NA").astype(str).value_counts().to_dict()
                for k, v in vc.items():
                    split_counts[str(k)] = split_counts.get(str(k), 0) + int(v)
    else:
        for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
            total_rows += len(chunk)

    info = {
        "raw_columns": header,
        "seq_col": seq_col,
        "split_col": split_col,
        "rows": total_rows,
        "unique_valid_aa_seqs": len(seqs),
        "split_counts": split_counts,
    }
    return info, seqs


def stream_clean_concat(
    concat_path: Path,
    header: list[str],
    mgnify_idx_seqs: set[str],
    dmsv4_seqs: set[str],
) -> tuple[object, dict[str, object]]:
    import numpy as np
    import pandas as pd

    seq_col = pick_col(header, ("aa_seq", "seq", "sequence", "wt_seq", "train_seq"))
    name_col = pick_col(header, ("name", "train_name", "id", "domain_id", "pdb_name"))
    dg_col = pick_col(header, ("deltaG", "dG", "recalibrated_dG", "dG_ML", "dg"))
    ci_col = pick_col(header, ("deltaG_95CI", "dG_95CI", "deltaG_std", "dG_std", "dg_ci"))
    ci_low_col = pick_col(header, ("deltaG_95CI_low", "dG_95CI_low"))
    ci_high_col = pick_col(header, ("deltaG_95CI_high", "dG_95CI_high"))
    lib_col = pick_col(header, ("lib", "library", "dms_lib", "dataset"))
    split_col = pick_col(header, ("split", "train_split"))
    mgnify_col = pick_col(header, ("mgnify",))
    design_col = pick_col(header, ("dm_design",))

    if seq_col is None or dg_col is None:
        raise RuntimeError(
            f"Missing required sequence/dG columns in {concat_path.name}: "
            f"seq_col={seq_col}, dg_col={dg_col}, columns={header}"
        )

    usecols = []
    for c in (
        name_col, seq_col, dg_col, ci_col, ci_low_col, ci_high_col,
        lib_col, split_col, mgnify_col, design_col,
    ):
        if c is not None and c not in usecols:
            usecols.append(c)

    print(
        f"[job6] Streaming {concat_path.name} with usecols={usecols} ...",
        flush=True,
    )

    chunks_kept = []
    stats = {
        "detected_columns": {
            "name": name_col,
            "aa_seq": seq_col,
            "dg": dg_col,
            "dg_95ci": ci_col,
            "dg_95ci_low": ci_low_col,
            "dg_95ci_high": ci_high_col,
            "lib": lib_col,
            "split": split_col,
            "mgnify": mgnify_col,
            "dm_design": design_col,
        },
        "raw_rows_read": 0,
        "dropped_invalid_or_short_seq": 0,
        "dropped_non_finite_dg": 0,
        "valid_rows_kept": 0,
    }

    for chunk_idx, chunk in enumerate(
        pd.read_csv(concat_path, usecols=usecols, chunksize=200_000, low_memory=False),
        1,
    ):
        n_raw = len(chunk)
        stats["raw_rows_read"] += n_raw

        seq_s = chunk[seq_col].fillna("").astype(str).str.strip().str.upper()
        seq_ok = seq_s.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{20,}", na=False)
        n_bad_seq = int((~seq_ok).sum())
        stats["dropped_invalid_or_short_seq"] += n_bad_seq

        dg_num = pd.to_numeric(chunk[dg_col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        dg_ok = np.isfinite(dg_num)
        stats["dropped_non_finite_dg"] += int((seq_ok.to_numpy() & (~dg_ok)).sum())

        keep = seq_ok.to_numpy() & dg_ok
        if not keep.any():
            continue

        out = pd.DataFrame({"aa_seq": seq_s[keep].to_numpy()})
        if name_col is not None:
            out["name"] = chunk.loc[keep, name_col].fillna("").astype(str).str.strip().to_numpy()
        else:
            out["name"] = ""
        out["dG"] = dg_num[keep]

        if ci_col is not None:
            ci_num = pd.to_numeric(chunk.loc[keep, ci_col], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            out["dG_95CI"] = np.where(np.isfinite(ci_num), ci_num, np.nan)
        elif ci_low_col is not None and ci_high_col is not None:
            lo = pd.to_numeric(chunk.loc[keep, ci_low_col], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            hi = pd.to_numeric(chunk.loc[keep, ci_high_col], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            diff = hi - lo
            out["dG_95CI"] = np.where(np.isfinite(diff), diff, np.nan)

        if lib_col is not None:
            out["lib"] = chunk.loc[keep, lib_col].fillna("").astype(str).str.strip().to_numpy()
        if split_col is not None:
            out["split"] = chunk.loc[keep, split_col].fillna("").astype(str).str.strip().to_numpy()
        if mgnify_col is not None:
            out["mgnify"] = (
                chunk.loc[keep, mgnify_col].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
            ).to_numpy()
        if design_col is not None:
            out["dm_design"] = (
                chunk.loc[keep, design_col].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
            ).to_numpy()

        stats["valid_rows_kept"] += len(out)
        chunks_kept.append(out)
        if chunk_idx % 5 == 0:
            print(
                f"  [chunk {chunk_idx}] rows_read={stats['raw_rows_read']:,}, "
                f"valid_kept={stats['valid_rows_kept']:,}",
                flush=True,
            )

    if not chunks_kept:
        raise RuntimeError("No valid canonical-AA finite-dG rows remained from concat CSV")

    df = pd.concat(chunks_kept, ignore_index=True)
    del chunks_kept
    gc.collect()

    # Fast deduplication by exact aa_seq: only groupby sequences that appear >1x
    dup_mask = df.duplicated("aa_seq", keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        uniq_df = df.loc[~dup_mask].copy()
        uniq_df["n_measurements"] = np.int32(1)

        dup_df = df.loc[dup_mask]
        agg_spec: dict[str, tuple[str, str]] = {
            "name": ("name", "first"),
            "dG": ("dG", "mean"),
            "n_measurements": ("dG", "size"),
        }
        if "dG_95CI" in dup_df.columns:
            agg_spec["dG_95CI"] = ("dG_95CI", "mean")
        for col in ("lib", "split", "mgnify", "dm_design"):
            if col in dup_df.columns:
                agg_spec[col] = (col, "first")

        grouped = dup_df.groupby("aa_seq", sort=False, as_index=False).agg(**agg_spec)
        grouped["n_measurements"] = grouped["n_measurements"].astype(np.int32)
        df = pd.concat([uniq_df, grouped], ignore_index=True)
        del uniq_df, dup_df, grouped
    else:
        df["n_measurements"] = np.int32(1)

    gc.collect()
    df["dG"] = df["dG"].astype(np.float32)
    if "dG_95CI" in df.columns:
        df["dG_95CI"] = df["dG_95CI"].astype(np.float32)
        df["dg_std"] = (df["dG_95CI"] / np.float32(2.0 * Z95)).astype(np.float32)
    df["seq_len"] = df["aa_seq"].str.len().astype(np.int32)

    if mgnify_idx_seqs:
        df["in_mgnify_training_index"] = df["aa_seq"].isin(mgnify_idx_seqs)
    if dmsv4_seqs:
        df["in_dmsv4_filtered_splits"] = df["aa_seq"].isin(dmsv4_seqs)

    stats["unique_seqs_after_dedup"] = int(len(df))
    stats["duplicate_rows_collapsed"] = int(stats["valid_rows_kept"] - len(df))
    if "in_mgnify_training_index" in df.columns:
        stats["overlap_mgnify_training_index"] = int(df["in_mgnify_training_index"].sum())
    if "in_dmsv4_filtered_splits" in df.columns:
        stats["overlap_dmsv4_filtered_splits"] = int(df["in_dmsv4_filtered_splits"].sum())

    print(
        f"[job6] Deduplicated {stats['valid_rows_kept']:,} valid rows -> "
        f"{stats['unique_seqs_after_dedup']:,} unique sequences",
        flush=True,
    )
    return df, stats


def collect_heldout_targets() -> tuple[list[str], dict[str, set[str]], dict[str, int]]:
    import pandas as pd
    import pyarrow.parquet as pq

    seq_sources: dict[str, set[str]] = {}
    source_counts: dict[str, int] = {}

    def add_seqs(seqs, tag: str) -> None:
        uniq = {
            str(s).strip().upper()
            for s in seqs
            if s is not None and str(s).strip() and str(s).strip().lower() not in ("nan", "none")
        }
        source_counts[tag] = len(uniq)
        for s in uniq:
            seq_sources.setdefault(s, set()).add(tag)

    split_specs = (
        ("splits_megascale.parquet", "megascale_id30_val_test"),
        ("splits_megascale_id25.parquet", "megascale_id25_val_test"),
        ("splits_megascale_fold_tm60.parquet", "megascale_fold_tm60_val_test"),
    )
    for fname, tag in split_specs:
        p = find_input(fname)
        if p is None:
            raise RuntimeError(f"Missing required split file in /kaggle/input: {fname}")
        cols = set(pq.ParquetFile(p).schema_arrow.names)
        key_col = "parent_seq" if "parent_seq" in cols else "wt_seq"
        sp_df = pd.read_parquet(p, columns=[key_col, "split"])
        sub = sp_df[sp_df["split"].astype(str).isin(("val", "test"))]
        add_seqs(sub[key_col].tolist(), tag)

    for fname, tag in (
        ("fireprotdb_holdout.parquet", "fireprotdb_holdout"),
        ("s669_holdout.parquet", "s669_holdout"),
    ):
        p = find_input(fname)
        if p is None:
            raise RuntimeError(f"Missing required holdout file in /kaggle/input: {fname}")
        cols = set(pq.ParquetFile(p).schema_arrow.names)
        seq_col = "wt_seq" if "wt_seq" in cols else "parent_seq"
        hdf = pd.read_parquet(p, columns=[seq_col])
        add_seqs(hdf[seq_col].tolist(), tag)

    targets = sorted(seq_sources.keys())
    print(
        f"[job6] Collected {len(targets)} unique heldout/external target sequences: {source_counts}",
        flush=True,
    )
    return targets, seq_sources, source_counts


def find_exact_and_substring_hits(
    query_seqs: list[str],
    targets: list[str],
    seq_sources: dict[str, set[str]],
) -> tuple[set[int], dict[str, int]]:
    """
    Exact check (O(L_q)) for:
      1) exact equality (q == t)
      2) query substring of target (q in t, since len(q) >= 20)
      3) target substring of query (t in q, via pigeonhole k-mer index)
    """
    if not targets:
        return set(), {}

    target_set = set(targets)
    k_q = 20
    t_20mers: dict[str, list[int]] = {}
    for ti, t in enumerate(targets):
        if len(t) >= k_q:
            for pos in range(len(t) - k_q + 1):
                km = t[pos : pos + k_q]
                t_20mers.setdefault(km, []).append(ti)

    short_targets = [ti for ti, t in enumerate(targets) if len(t) < 12]
    long_targets = [ti for ti, t in enumerate(targets) if len(t) >= 12]
    t_k12: dict[str, list[int]] = {}
    stride = 1
    k_t = 12
    if long_targets:
        min_t_len = min(len(targets[ti]) for ti in long_targets)
        k_t = min(12, min_t_len)
        stride = max(1, min_t_len - k_t + 1)
        for ti in long_targets:
            t = targets[ti]
            for pos in range(len(t) - k_t + 1):
                km = t[pos : pos + k_t]
                t_k12.setdefault(km, []).append(ti)

    hit_q_idxs: set[int] = set()
    hits_by_source: dict[str, int] = {}

    for qi, q in enumerate(query_seqs):
        matched_t: set[int] = set()
        if q in target_set:
            for ti, t in enumerate(targets):
                if q == t:
                    matched_t.add(ti)
                    break

        if len(q) >= k_q:
            cands = t_20mers.get(q[:k_q])
            if cands:
                for ti in cands:
                    if q in targets[ti]:
                        matched_t.add(ti)

        if long_targets and len(q) >= k_t:
            for pos in range(0, len(q) - k_t + 1, stride):
                cands = t_k12.get(q[pos : pos + k_t])
                if cands:
                    for ti in cands:
                        if targets[ti] in q:
                            matched_t.add(ti)

        if short_targets:
            for ti in short_targets:
                if targets[ti] in q:
                    matched_t.add(ti)

        if matched_t:
            hit_q_idxs.add(qi)
            srcs: set[str] = set()
            for ti in matched_t:
                srcs.update(seq_sources[targets[ti]])
            for s in srcs:
                hits_by_source[s] = hits_by_source.get(s, 0) + 1

    return hit_q_idxs, hits_by_source


def run_mmseqs_exclusion(
    mmseqs: str,
    query_seqs: list[str],
    targets: list[str],
    seq_sources: dict[str, set[str]],
) -> tuple[set[int], dict[str, object]]:
    sdir = TMP / "mmseqs_search"
    sdir.mkdir(parents=True, exist_ok=True)
    q_fa = sdir / "queries.fasta"
    t_fa = sdir / "targets.fasta"
    hits_tsv = sdir / "hits.tsv"
    mm_tmp = sdir / "tmp"
    mm_tmp.mkdir(exist_ok=True)

    with open(q_fa, "w", encoding="utf-8") as fh:
        for i, seq in enumerate(query_seqs):
            fh.write(f">Q{i}\n{seq}\n")
    with open(t_fa, "w", encoding="utf-8") as fh:
        for i, seq in enumerate(targets):
            fh.write(f">T{i}\n{seq}\n")

    t_mm = time.time()
    print(
        f"[job6] Running mmseqs easy-search: {len(query_seqs):,} queries vs {len(targets)} targets ...",
        flush=True,
    )
    subprocess.run(
        [
            mmseqs, "easy-search", str(q_fa), str(t_fa), str(hits_tsv), str(mm_tmp),
            "-s", "5.7", "--min-seq-id", "0.20", "-e", "10", "--threads", "4",
            "--format-output", "query,target,fident,alnlen,qcov,tcov",
        ],
        check=True,
    )
    mm_elapsed = round(time.time() - t_mm, 2)

    raw_hit_rows = 0
    excluded_mmseqs: set[int] = set()
    mm_hits_by_source: dict[str, set[int]] = {}
    max_ident_per_q: dict[int, float] = {}
    top_examples: list[dict[str, object]] = []

    if hits_tsv.is_file():
        with open(hits_tsv, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 6:
                    continue
                raw_hit_rows += 1
                q_id, t_id = parts[0], parts[1]
                fident = float(parts[2])
                alnlen = int(parts[3])
                qcov = float(parts[4])
                tcov = float(parts[5])
                qi = int(q_id[1:])
                ti = int(t_id[1:])
                if fident >= 0.25 and max(qcov, tcov) >= 0.50:
                    excluded_mmseqs.add(qi)
                    prev = max_ident_per_q.get(qi, 0.0)
                    if fident > prev:
                        max_ident_per_q[qi] = fident
                    for src_tag in seq_sources[targets[ti]]:
                        mm_hits_by_source.setdefault(src_tag, set()).add(qi)
                    if len(top_examples) < 25:
                        top_examples.append({
                            "query_idx": qi,
                            "target_idx": ti,
                            "target_sources": sorted(seq_sources[targets[ti]]),
                            "fident": round(fident, 4),
                            "alnlen": alnlen,
                            "qcov": round(qcov, 4),
                            "tcov": round(tcov, 4),
                        })

    exact_sub_hits, exact_by_source = find_exact_and_substring_hits(
        query_seqs, targets, seq_sources
    )
    excluded_all = excluded_mmseqs | exact_sub_hits

    audit_info = {
        "mmseqs_params": {
            "sensitivity": 5.7,
            "min_seq_id": 0.20,
            "evalue": 10,
            "threads": 4,
            "exclusion_rule": "fident >= 0.25 and max(qcov, tcov) >= 0.50, plus exact/substring match",
        },
        "mmseqs_elapsed_s": mm_elapsed,
        "mmseqs_raw_hit_rows": raw_hit_rows,
        "excluded_mmseqs_ge25_cov50": len(excluded_mmseqs),
        "excluded_exact_or_substring": len(exact_sub_hits),
        "excluded_only_exact_or_substring": len(exact_sub_hits - excluded_mmseqs),
        "excluded_total_unique_seqs": len(excluded_all),
        "excluded_mmseqs_by_target_source": {k: len(v) for k, v in sorted(mm_hits_by_source.items())},
        "excluded_exact_by_target_source": dict(sorted(exact_by_source.items())),
        "sample_excluded_hits": top_examples,
    }
    shutil.rmtree(sdir, ignore_errors=True)
    return excluded_all, audit_info


def main() -> int:
    t0 = time.time()
    errors: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "job": "job6_mgnify_stability_hf25",
        "zenodo_record": "19411306",
        "raw_headers": {},
        "downloads": {},
        "auxiliary_files": {},
        "cleaning_stats": {},
        "outputs": {},
        "errors": errors,
    }

    def record_error(stage: str, exc: Exception) -> None:
        msg = f"[job6] ERROR in {stage}: {exc}"
        print(msg, flush=True)
        errors.append({
            "stage": stage,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })

    try:
        ensure_packages()
    except Exception as exc:
        record_error("ensure_packages", exc)

    dl_paths: dict[str, Path] = {}
    try:
        dl_dir = TMP / "downloads"
        for key, spec in ZENODO_FILES.items():
            dst = dl_dir / spec["filename"]
            dl_info = download_file(spec["url"], dst)
            dl_paths[key] = dst
            summary["downloads"][spec["filename"]] = dl_info
    except Exception as exc:
        record_error("download_zenodo_19411306", exc)

    mgnify_idx_seqs: set[str] = set()
    dmsv4_seqs: set[str] = set()
    try:
        for key in ("dmsv4_splits", "mgnify_index"):
            p = dl_paths.get(key)
            if p is None or not p.is_file():
                continue
            hdr = read_csv_header(p)
            summary["raw_headers"][p.name] = hdr
            aux_info, seqs = inspect_aux_csv(p, hdr)
            summary["auxiliary_files"][p.name] = aux_info
            if key == "mgnify_index":
                mgnify_idx_seqs = seqs
            else:
                dmsv4_seqs = seqs
    except Exception as exc:
        record_error("inspect_auxiliary_csvs", exc)

    df_clean = None
    try:
        concat_p = dl_paths.get("concat")
        if concat_p is None or not concat_p.is_file():
            raise FileNotFoundError("Missing downloaded concat CSV")
        concat_hdr = read_csv_header(concat_p)
        summary["raw_headers"][concat_p.name] = concat_hdr
        df_clean, clean_stats = stream_clean_concat(
            concat_p, concat_hdr, mgnify_idx_seqs, dmsv4_seqs
        )
        summary["cleaning_stats"] = clean_stats
    except Exception as exc:
        record_error("stream_clean_concat", exc)
    finally:
        # Remove large downloaded CSVs from /tmp immediately after reading
        shutil.rmtree(TMP / "downloads", ignore_errors=True)
        del mgnify_idx_seqs, dmsv4_seqs
        gc.collect()

    targets: list[str] = []
    seq_sources: dict[str, set[str]] = {}
    source_counts: dict[str, int] = {}
    try:
        targets, seq_sources, source_counts = collect_heldout_targets()
    except Exception as exc:
        record_error("collect_heldout_targets", exc)

    mmseqs_bin: str | None = None
    try:
        mmseqs_bin = install_mmseqs()
    except Exception as exc:
        record_error("install_mmseqs", exc)

    try:
        if df_clean is not None and targets and mmseqs_bin is not None:
            import numpy as np

            q_seqs = df_clean["aa_seq"].tolist()
            excluded_idxs, mm_audit = run_mmseqs_exclusion(
                mmseqs_bin, q_seqs, targets, seq_sources
            )
            keep_mask = np.ones(len(df_clean), dtype=bool)
            if excluded_idxs:
                keep_mask[list(excluded_idxs)] = False
            hf25 = df_clean.loc[keep_mask].reset_index(drop=True)
            del df_clean
            gc.collect()

            out_pq = WORK / "mgnify_stability_clean_hf25.parquet"
            hf25.to_parquet(out_pq, index=False)

            dg_vals = hf25["dG"].to_numpy(dtype=np.float64)
            dg_stats = {
                "mean": round(float(np.mean(dg_vals)), 5),
                "std": round(float(np.std(dg_vals)), 5),
                "min": round(float(np.min(dg_vals)), 5),
                "p50": round(float(np.median(dg_vals)), 5),
                "max": round(float(np.max(dg_vals)), 5),
            }
            extra_counts = {}
            if "lib" in hf25.columns:
                extra_counts["lib_counts"] = {str(k): int(v) for k, v in hf25["lib"].value_counts().items()}
            if "split" in hf25.columns:
                extra_counts["split_counts"] = {str(k): int(v) for k, v in hf25["split"].value_counts().items()}
            if "mgnify" in hf25.columns:
                extra_counts["mgnify_true"] = int(hf25["mgnify"].sum())
            if "dm_design" in hf25.columns:
                extra_counts["dm_design_true"] = int(hf25["dm_design"].sum())

            leakage_audit = {
                "job": "job6_mgnify_stability_hf25",
                "target_sources_unique_seqs": source_counts,
                "total_unique_target_seqs": len(targets),
                "mgnify_unique_seqs_before_hf25": len(q_seqs),
                "mgnify_clean_hf25_seqs": int(len(hf25)),
                "clean_dg_stats": dg_stats,
                "clean_breakdown": extra_counts,
                **mm_audit,
            }
            audit_p = WORK / "mgnify_leakage_audit.json"
            audit_p.write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")

            summary["leakage_audit_summary"] = {
                "unique_seqs_before_hf25": len(q_seqs),
                "excluded_total": mm_audit["excluded_total_unique_seqs"],
                "excluded_mmseqs_ge25_cov50": mm_audit["excluded_mmseqs_ge25_cov50"],
                "excluded_exact_or_substring": mm_audit["excluded_exact_or_substring"],
                "clean_hf25_seqs": int(len(hf25)),
                "clean_dg_stats": dg_stats,
            }
            summary["outputs"][out_pq.name] = file_hashes(out_pq)
            summary["outputs"][audit_p.name] = file_hashes(audit_p)
        else:
            audit_p = WORK / "mgnify_leakage_audit.json"
            audit_p.write_text(
                json.dumps(
                    {
                        "status": "incomplete",
                        "target_sources_unique_seqs": source_counts,
                        "total_unique_target_seqs": len(targets),
                        "mmseqs_available": mmseqs_bin is not None,
                        "errors": errors,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary["outputs"][audit_p.name] = file_hashes(audit_p)
    except Exception as exc:
        record_error("mmseqs_filter_and_write", exc)
    finally:
        shutil.rmtree(TMP / "mmseqs_install", ignore_errors=True)
        shutil.rmtree(TMP / "mmseqs_search", ignore_errors=True)

    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["status"] = "ok" if not errors else "completed_with_errors"
    sum_p = WORK / "job6_summary.json"
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
