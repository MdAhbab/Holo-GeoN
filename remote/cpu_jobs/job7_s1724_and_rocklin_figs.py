"""
remote/cpu_jobs/job7_s1724_and_rocklin_figs.py
==============================================
Kaggle CPU Worker 7 (Account: mhmehedi26) — S1724 ThermoMutDB Benchmark
Standardization, <25%-Identity Homology Filtering, and Rocklin 2026 Figure
Prediction Baselines (Zenodo 19411306).

Downloads into /tmp/holo_job7/ (1 TB free on Kaggle CPU):
  1. ``benchmarks.zip`` (871.6 MB)
  2. ``figures.zip`` (715.6 MB)

Pipeline:
  * Inspects ``benchmarks.zip`` entries, extracts ``S1724_thermomutdb_cleaned_withseq.csv``
    and ``S1724_PDB_sequences.csv``, records exact raw columns, joins WT sequences,
    verifies ``wt_seq[site] == wt_aa`` on 100% of standardized rows, checks sign
    convention via mean ddG (standardizing to Holo-GeoN ``stabilizing_positive``),
    and writes ``s1724_thermomutdb_holdout.parquet``.
  * Installs static MMseqs2 Linux AVX2 into ``/tmp/holo_job7/`` and runs
    ``mmseqs easy-search`` (``-s 7.5``) against ALL MegaScale proteins in
    ``megascale_ddg_pairs.parquet`` (plus exact substring checks) -> writes
    ``s1724_homology_audit.parquet`` and ``s1724_thermomutdb_hf25.parquet``
    (``<25%`` identity to MegaScale).
  * Inspects ``figures.zip`` (listing all CSV filenames in ``job7_summary.json``),
    extracts ``fig3__megascale_point_mutants_predictions.csv`` and
    ``fig4__thermomut_predictions.csv``, records columns/baseline correlations,
    and writes ``rocklin2026_fig3_megascale_preds.parquet`` and
    ``rocklin2026_fig4_s1724_preds.parquet``.
  * Writes ``job7_summary.json`` with SHA-256 digests and full verification metrics,
    deletes large temporary files from ``/tmp/holo_job7/``, and always exits 0.
"""
from __future__ import annotations

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
import zipfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job7")
TMP = Path("/tmp/holo_job7") if Path("/tmp").is_dir() else WORK / "_tmp"
WORK.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Holo-GeoN/2.0"
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

ZENODO_ZIPS = {
    "benchmarks.zip": "https://zenodo.org/api/records/19411306/files/benchmarks.zip/content",
    "figures.zip": "https://zenodo.org/api/records/19411306/files/figures.zip/content",
}


def ensure_packages() -> None:
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow")):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[job7] Installing missing packages: {missing}", flush=True)
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
            next_log = 200 * 1024 * 1024

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
                            print(f"  [dl] {dest.name}: {mb:.1f} MB ({mb / dt:.1f} MB/s)", flush=True)
                            next_log += 200 * 1024 * 1024

            if expected_len is not None and written != expected_len:
                raise RuntimeError(
                    f"Incomplete download for {dest.name}: got {written} bytes, expected {expected_len}"
                )
            if written == 0:
                raise RuntimeError(f"Empty download for {dest.name}")

            os.replace(part, dest)
            dt_total = max(time.time() - t_start, 0.1)
            print(
                f"[job7] Downloaded {dest.name}: {written / 1e6:.2f} MB in {dt_total:.1f}s",
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


def clean_zip_entries(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    out = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        norm = info.filename.replace("\\", "/")
        base = norm.rsplit("/", 1)[-1]
        if norm.startswith("__MACOSX/") or base.startswith("._"):
            continue
        out.append(info)
    return out


def find_zip_member(entries: list[zipfile.ZipInfo], target_basename: str) -> zipfile.ZipInfo:
    target_l = target_basename.lower()
    exact = [i for i in entries if i.filename.replace("\\", "/").rsplit("/", 1)[-1].lower() == target_l]
    if not exact:
        avail = [i.filename for i in entries]
        raise FileNotFoundError(f"{target_basename} not found in zip; available={avail}")
    return max(exact, key=lambda i: i.file_size)


def pick_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        hit = by_lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def pick_ddg_col(columns: list[str]) -> str | None:
    non_pred = [
        c for c in columns
        if not c.lower().startswith(("predicted_", "pred_", "std_", "err_", "ci_"))
    ]
    cand = pick_col(
        non_pred,
        (
            "ddg_exp", "experimental_ddg", "exp_ddg", "experimental_ddg_dir",
            "ddg", "ddg_dir", "delta_delta_g", "ddg_value", "ddg_mean", "target",
            "ddg_ml", "dg_diff", "experiment",
        ),
    )
    if cand is not None:
        return cand
    for c in non_pred:
        cl = c.lower()
        if "ddg" in cl and not any(bad in cl for bad in ("std", "err", "ci", "pred", "sign")):
            return c
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
    print(f"[job7] MMseqs2 version: {ver}", flush=True)
    return str(mm_bin)


def _clean_aa_seq(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip().upper()
    if not s or s in ("NAN", "NONE", "NULL"):
        return None
    # Strip whitespace/newlines inside FASTA strings
    s = re.sub(r"\s+", "", s)
    if len(s) >= 10 and all(c in AA20 for c in s):
        return s
    return None


def build_pdb_seq_lookup(pdb_df) -> tuple[dict[str, list[str]], dict[str, object]]:
    cols = list(pdb_df.columns)
    pdb_col = pick_col(
        cols,
        ("pdb_chain", "pdb_id", "pdb", "wt_pdb", "pdb_code", "name", "id", "protein", "pdb_name"),
    )
    chain_col = pick_col(cols, ("chain", "chain_id"))
    seq_cols: list[str] = []
    for cand in (
        "paper", "paper_seq", "wt_seq", "parent_seq", "gabetrimmed",
        "gjr_trim_seq", "pdbfasta", "pdb_seq", "aa_seq", "sequence", "seq",
    ):
        hit = pick_col(cols, (cand,))
        if hit is not None and hit not in seq_cols:
            seq_cols.append(hit)

    lookup: dict[str, list[str]] = {}

    def add_key(k: str, seq: str) -> None:
        if not k:
            return
        bucket = lookup.setdefault(k, [])
        if seq not in bucket:
            bucket.append(seq)

    if pdb_col and seq_cols:
        for row in pdb_df.to_dict(orient="records"):
            raw_p = row.get(pdb_col)
            if raw_p is None:
                continue
            p_str = str(raw_p).strip().upper()
            if not p_str or p_str in ("NAN", "NONE"):
                continue
            seqs = []
            for sc in seq_cols:
                s_clean = _clean_aa_seq(row.get(sc))
                if s_clean and s_clean not in seqs:
                    seqs.append(s_clean)
            if not seqs:
                continue

            keys = [p_str]
            if p_str.endswith(".PDB"):
                keys.append(p_str[:-4])
            # Split combined PDB_chain like "1AJ3_A" or "1AJ3A"
            if "_" in p_str:
                base, ch_part = p_str.split("_", 1)
                keys.extend([base, f"{base}{ch_part}"])
            elif len(p_str) == 5 and p_str[:4].isalnum():
                keys.extend([p_str[:4], f"{p_str[:4]}_{p_str[4]}"])
            if chain_col:
                ch = str(row.get(chain_col) or "").strip().upper()
                if ch and ch not in ("NAN", "NONE"):
                    keys.extend([f"{p_str}_{ch}", f"{p_str}{ch}"])
            for k in keys:
                for sq in seqs:
                    add_key(k, sq)

    info = {
        "raw_columns": cols,
        "rows": int(len(pdb_df)),
        "detected_columns": {"pdb": pdb_col, "chain": chain_col, "seq_cols": seq_cols},
        "unique_keys_indexed": len(lookup),
    }
    return lookup, info


def _diff_wt_and_mt(wt_seq: str, mt_seq: str, wt_aa: str, mt_aa: str) -> int | None:
    """Return 0-based site if wt_seq and mt_seq differ at exactly one site matching wt_aa->mt_aa."""
    if len(wt_seq) != len(mt_seq):
        return None
    diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mt_seq)) if a != b]
    if len(diffs) == 1:
        site = diffs[0]
        if wt_seq[site] == wt_aa and mt_seq[site] == mt_aa:
            return site
    return None


def standardize_s1724(df, pdb_lookup: dict[str, list[str]]) -> tuple[object, dict[str, object]]:
    import numpy as np
    import pandas as pd

    cols = list(df.columns)
    sample_rows = df.head(2).astype(str).to_dict(orient="records")
    pdb_col = pick_col(
        cols,
        ("pdb", "pdb_chain", "wt_pdb", "pdb_id", "pdb_wild", "pdb_code", "protein", "pdb_name", "name", "id"),
    )
    chain_col = pick_col(cols, ("chain", "chain_id", "pdb_chain"))
    if chain_col == pdb_col:
        chain_col = None
    wt_seq_col = pick_col(cols, ("wt_seq", "parent_seq", "wildtype_seq", "wt_sequence"))
    # Mutant or ambiguous sequence columns (Rocklin S1724 stores mutant seq in paper_seq / GJR_trim_seq)
    row_seq_cols: list[str] = []
    for cand in ("paper_seq", "gjr_trim_seq", "aa_seq", "mt_seq", "mut_seq", "sequence", "seq", "pdb_seq"):
        hit = pick_col(cols, (cand,))
        if hit is not None and hit not in row_seq_cols:
            row_seq_cols.append(hit)

    mut_col = pick_col(cols, ("seq_mut", "mutation", "mut", "mut_type", "substitution", "pdb_mut", "variant"))
    pos_col = pick_col(cols, ("site", "pos", "position", "seq_pos", "resnum"))
    wt_col = pick_col(cols, ("wt_aa", "wt", "ref_aa", "wild_type"))
    mt_col = pick_col(cols, ("mt_aa", "mut_aa", "alt_aa", "mutant_aa"))
    name_col = pick_col(cols, ("name", "variant_id", "id"))
    ddg_col = pick_ddg_col(cols)
    temp_col = pick_col(cols, ("temperature_k", "temperature", "temp", "t"))
    ph_col = pick_col(cols, ("ph",))

    if ddg_col is None:
        raise RuntimeError(
            f"Could not locate experimental ddG column in S1724 CSV: cols={cols} sample={sample_rows}"
        )

    def pdb_candidates_for_row(row_dict: dict) -> list[str]:
        seqs: list[str] = []
        raw_p = str(row_dict.get(pdb_col) or "").strip().upper() if pdb_col else ""
        ch = str(row_dict.get(chain_col) or "").strip().upper() if chain_col else ""
        if ch in ("NAN", "NONE"):
            ch = ""
        keys: list[str] = []
        if raw_p and raw_p not in ("NAN", "NONE"):
            keys.extend([f"{raw_p}_{ch}" if ch else "", f"{raw_p}{ch}" if ch else "", raw_p])
            if raw_p.endswith(".PDB"):
                keys.append(raw_p[:-4])
            if "_" in raw_p:
                keys.append(raw_p.split("_", 1)[0])
            elif len(raw_p) == 5:
                keys.extend([raw_p[:4], f"{raw_p[:4]}_{raw_p[4]}"])
        for k in keys:
            if k and k in pdb_lookup:
                for sq in pdb_lookup[k]:
                    if sq not in seqs:
                        seqs.append(sq)
        return seqs

    def extract_mut_parts(row_dict: dict) -> tuple[str, int, str] | None:
        if wt_col is not None and mt_col is not None and pos_col is not None:
            w = str(row_dict.get(wt_col) or "").strip().upper()
            mt = str(row_dict.get(mt_col) or "").strip().upper()
            raw_p = row_dict.get(pos_col)
            if w and mt and raw_p is not None and pd.notna(raw_p):
                try:
                    pos = int(float(raw_p))
                    return w, pos, mt
                except (TypeError, ValueError):
                    pass
        if mut_col is not None:
            raw_m = row_dict.get(mut_col)
            if raw_m is not None and pd.notna(raw_m):
                m = MUT_RE.match(str(raw_m).strip().upper())
                if m:
                    return m.group(1), int(m.group(2)), m.group(3)
        if name_col is not None:
            raw_n = str(row_dict.get(name_col) or "").strip().upper()
            if "_" in raw_n:
                tail = raw_n.rsplit("_", 1)[-1]
                m = MUT_RE.match(tail)
                if m:
                    return m.group(1), int(m.group(2)), m.group(3)
        return None

    records_raw = df.to_dict(orient="records")

    # Also build per-(pdb, length) consensus WT sequence from mutant row_seq_cols as fallback
    from collections import Counter
    consensus_by_pdb: dict[str, list[str]] = {}
    if pdb_col and row_seq_cols:
        grouped_seqs: dict[tuple[str, int], list[str]] = {}
        for rd in records_raw:
            raw_p = str(rd.get(pdb_col) or "").strip().upper()
            if not raw_p or raw_p in ("NAN", "NONE"):
                continue
            for sc in row_seq_cols:
                sq = _clean_aa_seq(rd.get(sc))
                if sq:
                    grouped_seqs.setdefault((raw_p, len(sq)), []).append(sq)
        for (raw_p, seq_len), seq_list in grouped_seqs.items():
            if len(seq_list) >= 3:
                cons = "".join(
                    Counter(s[i] for s in seq_list).most_common(1)[0][0]
                    for i in range(seq_len)
                )
                bucket = consensus_by_pdb.setdefault(raw_p, [])
                if cons not in bucket:
                    bucket.append(cons)

    # Pre-scan 1-based vs 0-based when WT sequence is directly provided
    hits_1based = 0
    hits_0based = 0
    for rd in records_raw:
        mp = extract_mut_parts(rd)
        if mp is None:
            continue
        wt_aa, raw_pos, _ = mp
        wt_cands = []
        if wt_seq_col:
            sq = _clean_aa_seq(rd.get(wt_seq_col))
            if sq:
                wt_cands.append(sq)
        wt_cands.extend(pdb_candidates_for_row(rd))
        if not wt_cands:
            continue
        seq = wt_cands[0]
        if 1 <= raw_pos <= len(seq) and seq[raw_pos - 1] == wt_aa:
            hits_1based += 1
        if 0 <= raw_pos < len(seq) and seq[raw_pos] == wt_aa:
            hits_0based += 1

    use_zero_based = (mut_col is None) and (hits_0based > hits_1based)
    pos_indexing = "0-based" if use_zero_based else "1-based"

    dropped = {
        "missing_sequence": 0,
        "bad_or_non_single_mutation": 0,
        "non_canonical_or_synonymous": 0,
        "non_finite_ddg": 0,
        "out_of_bounds_or_wt_mismatch": 0,
    }
    resolution_methods: dict[str, int] = {}
    clean_rows: list[dict[str, object]] = []
    raw_ddg_vals: list[float] = []

    for rd in records_raw:
        mp = extract_mut_parts(rd)
        if mp is None:
            dropped["bad_or_non_single_mutation"] += 1
            continue
        wt_aa, raw_pos, mt_aa = mp
        if wt_aa not in AA20 or mt_aa not in AA20 or wt_aa == mt_aa:
            dropped["non_canonical_or_synonymous"] += 1
            continue

        try:
            raw_ddg = float(rd.get(ddg_col))
        except (TypeError, ValueError):
            raw_ddg = float("nan")
        if not np.isfinite(raw_ddg):
            dropped["non_finite_ddg"] += 1
            continue

        raw_pdb = str(rd.get(pdb_col) or "").strip().upper() if pdb_col else ""
        if raw_pdb in ("NAN", "NONE"):
            raw_pdb = ""
        raw_ch = str(rd.get(chain_col) or "").strip() if chain_col else ""
        if raw_ch.lower() in ("nan", "none"):
            raw_ch = ""

        # Collect WT candidates (explicit wt_seq_col -> S1724_PDB_sequences lookup -> consensus)
        wt_cands: list[tuple[str, str]] = []
        if wt_seq_col:
            sq = _clean_aa_seq(rd.get(wt_seq_col))
            if sq:
                wt_cands.append(("wt_seq_col", sq))
        for sq in pdb_candidates_for_row(rd):
            wt_cands.append(("pdb_lookup", sq))
        for sq in consensus_by_pdb.get(raw_pdb, []):
            wt_cands.append(("consensus_wt", sq))

        # Collect row mutant sequence candidates (Rocklin paper_seq / GJR_trim_seq / aa_seq)
        mt_cands: list[tuple[str, str]] = []
        for sc in row_seq_cols:
            sq = _clean_aa_seq(rd.get(sc))
            if sq:
                mt_cands.append((sc, sq))

        if not wt_cands and not mt_cands:
            dropped["missing_sequence"] += 1
            continue

        resolved_wt: str | None = None
        resolved_site: int | None = None
        method_used = ""

        # Strategy 1 (most exact): diff WT candidate vs row mutant sequence -> exact single-site match
        for wsrc, wsq in wt_cands:
            for mcol, msq in mt_cands:
                s_idx = _diff_wt_and_mt(wsq, msq, wt_aa, mt_aa)
                if s_idx is not None:
                    resolved_wt = wsq
                    resolved_site = s_idx
                    method_used = f"diff:{wsrc}_vs_{mcol}"
                    break
            if resolved_wt is not None:
                break

        # Strategy 2: direct 1-based / 0-based index into WT candidate
        if resolved_wt is None:
            site_direct = raw_pos if use_zero_based else (raw_pos - 1)
            for wsrc, wsq in wt_cands:
                if 0 <= site_direct < len(wsq) and wsq[site_direct] == wt_aa:
                    resolved_wt = wsq
                    resolved_site = site_direct
                    method_used = f"direct_index:{wsrc}"
                    break

        # Strategy 3: row_seq_col IS actually WT (not mutant) at direct index
        if resolved_wt is None:
            site_direct = raw_pos if use_zero_based else (raw_pos - 1)
            for mcol, msq in mt_cands:
                if 0 <= site_direct < len(msq) and msq[site_direct] == wt_aa:
                    resolved_wt = msq
                    resolved_site = site_direct
                    method_used = f"row_seq_as_wt:{mcol}"
                    break

        if resolved_wt is None or resolved_site is None:
            dropped["out_of_bounds_or_wt_mismatch"] += 1
            continue

        resolution_methods[method_used] = resolution_methods.get(method_used, 0) + 1
        seq = resolved_wt
        site = resolved_site

        if raw_pdb and raw_ch and not raw_pdb.endswith("_" + raw_ch.upper()):
            parent = f"{raw_pdb}_{raw_ch}"
        elif raw_pdb:
            parent = raw_pdb
        else:
            parent = f"S1724_{hashlib.sha1(seq.encode()).hexdigest()[:8]}"

        mt_seq = seq[:site] + mt_aa + seq[site + 1 :]
        temp_val = float("nan")
        if temp_col is not None:
            try:
                temp_val = float(rd.get(temp_col))
            except (TypeError, ValueError):
                temp_val = float("nan")
        ph_val = float("nan")
        if ph_col is not None:
            try:
                ph_val = float(rd.get(ph_col))
            except (TypeError, ValueError):
                ph_val = float("nan")

        raw_ddg_vals.append(raw_ddg)
        clean_rows.append({
            "parent": parent,
            "wt_pdb": raw_pdb,
            "chain": raw_ch,
            "wt_seq": seq,
            "parent_seq": seq,
            "site": int(site),
            "wt_aa": wt_aa,
            "mt_aa": mt_aa,
            "mutation": f"{wt_aa}{site + 1}{mt_aa}",
            "pdb_position": int(raw_pos),
            "mt_seq": mt_seq,
            "raw_ddg": raw_ddg,
            "temperature_k": temp_val,
            "ph": ph_val,
            "source": "S1724_Zenodo_19411306",
        })

    if not clean_rows:
        raise RuntimeError(
            f"0 valid S1724 rows survived standardization; cols={cols}; "
            f"sample={sample_rows}; dropped={dropped}"
        )

    raw_arr = np.asarray(raw_ddg_vals, dtype=np.float64)
    raw_mean = float(np.mean(raw_arr))
    raw_median = float(np.median(raw_arr))
    flipped = bool(raw_mean > 0.0)
    sign_mult = -1.0 if flipped else 1.0

    for row in clean_rows:
        std_ddg = float(row["raw_ddg"]) * sign_mult
        row["ddg"] = std_ddg
        row["target"] = std_ddg
        row["sign_convention"] = "stabilizing_positive"

    out_df = pd.DataFrame(clean_rows)
    assert all(
        s[i] == w for s, i, w in zip(out_df["wt_seq"], out_df["site"], out_df["wt_aa"])
    ), "wt_seq[site] == wt_aa invariant failed"

    std_mean = float(out_df["ddg"].mean())
    info = {
        "raw_columns": cols,
        "raw_rows": int(len(df)),
        "detected_columns": {
            "pdb": pdb_col,
            "chain": chain_col,
            "wt_seq": wt_seq_col,
            "row_seq_cols": row_seq_cols,
            "mut": mut_col,
            "pos": pos_col,
            "wt_aa": wt_col,
            "mt_aa": mt_col,
            "ddg": ddg_col,
            "temperature": temp_col,
            "ph": ph_col,
        },
        "position_indexing_detected": pos_indexing,
        "indexing_probe_counts": {"hits_1based": hits_1based, "hits_0based": hits_0based},
        "resolution_methods": resolution_methods,
        "rows_verified": int(len(out_df)),
        "unique_parents": int(out_df["parent"].nunique()),
        "unique_wt_seqs": int(out_df["wt_seq"].nunique()),
        "wt_seq_site_match_fraction": 1.0,
        "sign_check": {
            "raw_mean_ddg": round(raw_mean, 5),
            "raw_median_ddg": round(raw_median, 5),
            "raw_positive_count": int((raw_arr > 0).sum()),
            "raw_negative_count": int((raw_arr < 0).sum()),
            "raw_sign_convention": "destabilizing_positive" if flipped else "stabilizing_positive",
            "flipped_to_stabilizing_positive": flipped,
            "standardized_mean_ddg": round(std_mean, 5),
            "standardized_convention": "stabilizing_positive",
        },
        "dropped": dropped,
    }
    return out_df, info


def run_s1724_homology(mmseqs: str, s1724_df, megascale_pairs_p: Path) -> tuple[object, object, dict[str, object]]:
    import pandas as pd

    ms_df = pd.read_parquet(megascale_pairs_p, columns=["parent", "wt_seq"]).drop_duplicates("wt_seq")
    ms_seqs = [str(s).strip().upper() for s in ms_df["wt_seq"]]
    ms_parents = [str(p).strip() for p in ms_df["parent"]]

    s_uniq = s1724_df[["parent", "wt_seq"]].drop_duplicates().reset_index(drop=True)
    q_seqs = [str(s).strip().upper() for s in s_uniq["wt_seq"]]
    q_parents = [str(p).strip() for p in s_uniq["parent"]]

    sdir = TMP / "mmseqs_s1724"
    sdir.mkdir(parents=True, exist_ok=True)
    q_fa = sdir / "s1724_q.fasta"
    t_fa = sdir / "megascale_t.fasta"
    out_tsv = sdir / "hits.tsv"
    mm_tmp = sdir / "tmp"
    mm_tmp.mkdir(exist_ok=True)

    with open(q_fa, "w", encoding="utf-8") as fh:
        for i, seq in enumerate(q_seqs):
            fh.write(f">S{i}\n{seq}\n")
    with open(t_fa, "w", encoding="utf-8") as fh:
        for i, seq in enumerate(ms_seqs):
            fh.write(f">MS{i}\n{seq}\n")

    subprocess.run(
        [
            mmseqs, "easy-search", str(q_fa), str(t_fa), str(out_tsv), str(mm_tmp),
            "-s", "7.5", "--min-seq-id", "0.15", "-e", "10", "--threads", "4",
            "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        ],
        check=True,
    )

    hit_rows: list[dict[str, object]] = []
    max_cov50_by_seq: dict[str, float] = {}
    max_raw_by_seq: dict[str, float] = {}

    if out_tsv.is_file():
        with open(out_tsv, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 8:
                    continue
                qi = int(parts[0][1:])
                ti = int(parts[1][2:])
                fid = float(parts[2])
                aln = int(parts[3])
                qcov = float(parts[4])
                tcov = float(parts[5])
                ev = float(parts[6])
                bits = float(parts[7])
                q_seq = q_seqs[qi]
                t_seq = ms_seqs[ti]
                exact_sub = (q_seq in t_seq) or (t_seq in q_seq)
                max_cov = max(qcov, tcov)
                hit_rows.append({
                    "query": q_parents[qi],
                    "query_seq_len": len(q_seq),
                    "target": ms_parents[ti],
                    "target_seq_len": len(t_seq),
                    "seq_identity": fid,
                    "aln_len": aln,
                    "qcov": qcov,
                    "tcov": tcov,
                    "max_cov": max_cov,
                    "evalue": ev,
                    "bits": bits,
                    "exact_substring": bool(exact_sub),
                })
                max_raw_by_seq[q_seq] = max(max_raw_by_seq.get(q_seq, 0.0), fid)
                if max_cov >= 0.50 or exact_sub:
                    eff = 1.0 if exact_sub else fid
                    max_cov50_by_seq[q_seq] = max(max_cov50_by_seq.get(q_seq, 0.0), eff)

    # Exhaustive exact substring check (in case MMseqs missed short hits)
    exact_pairs: set[str] = set()
    for qi, q_seq in enumerate(q_seqs):
        for ti, t_seq in enumerate(ms_seqs):
            if q_seq in t_seq or t_seq in q_seq:
                exact_pairs.add(q_seq)
                max_cov50_by_seq[q_seq] = 1.0
                max_raw_by_seq[q_seq] = 1.0
                break

    if hit_rows:
        audit_df = pd.DataFrame(hit_rows)
    else:
        audit_df = pd.DataFrame(
            columns=[
                "query", "query_seq_len", "target", "target_seq_len",
                "seq_identity", "aln_len", "qcov", "tcov", "max_cov",
                "evalue", "bits", "exact_substring",
            ]
        )

    hf_annotated = s1724_df.copy()
    hf_annotated["max_megascale_seq_id"] = hf_annotated["wt_seq"].map(
        lambda s: float(max_cov50_by_seq.get(str(s).upper(), 0.0))
    )
    hf_annotated["max_megascale_seq_id_raw"] = hf_annotated["wt_seq"].map(
        lambda s: float(max_raw_by_seq.get(str(s).upper(), 0.0))
    )
    hf25 = hf_annotated[hf_annotated["max_megascale_seq_id"] < 0.25].reset_index(drop=True)

    unique_q_seqs = sorted(set(q_seqs))
    ge25_cov50 = sum(1 for s in unique_q_seqs if max_cov50_by_seq.get(s, 0.0) >= 0.25)
    ge25_raw = sum(1 for s in unique_q_seqs if max_raw_by_seq.get(s, 0.0) >= 0.25)

    shutil.rmtree(sdir, ignore_errors=True)
    info = {
        "megascale_unique_proteins": len(ms_seqs),
        "s1724_unique_seqs_queried": len(unique_q_seqs),
        "mmseqs_hit_rows": int(len(audit_df)),
        "s1724_seqs_exact_substring_megascale": len(exact_pairs),
        "s1724_seqs_ge_25pct_cov50": ge25_cov50,
        "s1724_seqs_ge_25pct_raw": ge25_raw,
        "s1724_hf25_rows": int(len(hf25)),
        "s1724_hf25_unique_parents": int(hf25["parent"].nunique()),
        "s1724_hf25_unique_wt_seqs": int(hf25["wt_seq"].nunique()),
    }
    return audit_df, hf25, info


def convert_fig_csv_to_parquet(
    zf: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
    out_parquet: Path,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    with zf.open(entry) as raw_fh:
        df = pd.read_csv(raw_fh, low_memory=False)

    raw_cols = list(df.columns)
    pred_cols = [c for c in raw_cols if c.lower().startswith("predicted_")]
    for c in pred_cols:
        num = pd.to_numeric(df[c], errors="coerce")
        df[c] = num.astype(np.float32)

    # Ensure object columns do not fail pyarrow serialization on mixed scalar types
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string")

    df.to_parquet(out_parquet, index=False)

    exp_col = pick_ddg_col(raw_cols)
    corrs: dict[str, dict[str, float]] = {}
    if exp_col is not None:
        exp_num = pd.to_numeric(df[exp_col], errors="coerce").to_numpy(dtype=np.float64)
        for pc in pred_cols[:20]:
            p_num = df[pc].to_numpy(dtype=np.float64)
            mask = np.isfinite(exp_num) & np.isfinite(p_num)
            if int(mask.sum()) >= 10:
                ea = exp_num[mask]
                pa = p_num[mask]
                if np.std(ea) > 0 and np.std(pa) > 0:
                    pearson_r = float(np.corrcoef(ea, pa)[0, 1])
                    corrs[pc] = {
                        "n": int(mask.sum()),
                        "pearson_r": round(pearson_r, 5),
                    }

    info = {
        "source_csv": entry.filename,
        "output_parquet": out_parquet.name,
        "rows": int(len(df)),
        "raw_columns": raw_cols,
        "n_predicted_columns": len(pred_cols),
        "predicted_columns": pred_cols,
        "detected_exp_column": exp_col,
        "pearson_vs_exp": corrs,
    }
    del df
    gc.collect()
    return info


def main() -> int:
    t0 = time.time()
    errors: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "job": "job7_s1724_and_rocklin_figs",
        "zenodo_record": "19411306",
        "downloads": {},
        "benchmarks_zip_entries": [],
        "figures_zip_csvs": [],
        "s1724_pdb_sequences": {},
        "s1724_standardization": {},
        "s1724_homology_audit": {},
        "rocklin_figures": {},
        "outputs": {},
        "errors": errors,
    }

    def record_error(stage: str, exc: Exception) -> None:
        print(f"[job7] ERROR in {stage}: {exc}", flush=True)
        errors.append({
            "stage": stage,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })

    try:
        ensure_packages()
    except Exception as exc:
        record_error("ensure_packages", exc)

    dl_dir = TMP / "downloads"
    bench_zip = dl_dir / "benchmarks.zip"
    figs_zip = dl_dir / "figures.zip"

    for fname, url in ZENODO_ZIPS.items():
        try:
            dl_info = download_file(url, dl_dir / fname)
            summary["downloads"][fname] = dl_info
        except Exception as exc:
            record_error(f"download_{fname}", exc)

    # Stage 1: benchmarks.zip -> S1724 holdout standardization
    s1724_df = None
    try:
        import pandas as pd

        if not bench_zip.is_file():
            raise FileNotFoundError(f"Missing {bench_zip}")
        with zipfile.ZipFile(bench_zip) as zf:
            entries = clean_zip_entries(zf)
            summary["benchmarks_zip_entries"] = [
                {"filename": e.filename, "bytes": e.file_size} for e in entries
            ]
            pdb_entry = find_zip_member(entries, "S1724_PDB_sequences.csv")
            mut_entry = find_zip_member(entries, "S1724_thermomutdb_cleaned_withseq.csv")
            with zf.open(pdb_entry) as fh:
                pdb_df = pd.read_csv(fh, low_memory=False)
            with zf.open(mut_entry) as fh:
                raw_s1724_df = pd.read_csv(fh, low_memory=False)

        pdb_lookup, pdb_info = build_pdb_seq_lookup(pdb_df)
        summary["s1724_pdb_sequences"] = pdb_info

        s1724_df, std_info = standardize_s1724(raw_s1724_df, pdb_lookup)
        summary["s1724_standardization"] = std_info

        out_holdout = WORK / "s1724_thermomutdb_holdout.parquet"
        s1724_df.to_parquet(out_holdout, index=False)
        summary["outputs"][out_holdout.name] = file_hashes(out_holdout)
        print(
            f"[job7] Standardized S1724: {len(s1724_df)} rows, "
            f"{std_info['unique_parents']} parents -> {out_holdout.name}",
            flush=True,
        )
    except Exception as exc:
        record_error("s1724_benchmarks_etl", exc)
    finally:
        if bench_zip.exists():
            bench_zip.unlink()

    # Stage 2: MMseqs2 (-s 7.5) homology audit vs MegaScale
    try:
        pairs_p = find_input("megascale_ddg_pairs.parquet")
        if s1724_df is None:
            raise RuntimeError("Skipping homology audit because S1724 standardization failed")
        if pairs_p is None:
            raise FileNotFoundError("megascale_ddg_pairs.parquet not found under /kaggle/input")

        mmseqs = install_mmseqs()
        audit_df, hf25_df, hom_info = run_s1724_homology(mmseqs, s1724_df, pairs_p)
        summary["s1724_homology_audit"] = hom_info

        out_audit = WORK / "s1724_homology_audit.parquet"
        out_hf25 = WORK / "s1724_thermomutdb_hf25.parquet"
        audit_df.to_parquet(out_audit, index=False)
        hf25_df.to_parquet(out_hf25, index=False)
        summary["outputs"][out_audit.name] = file_hashes(out_audit)
        summary["outputs"][out_hf25.name] = file_hashes(out_hf25)
        print(
            f"[job7] S1724 HF25: {len(hf25_df)}/{len(s1724_df)} rows across "
            f"{hom_info['s1724_hf25_unique_parents']} parents (<25% id to MegaScale)",
            flush=True,
        )
    except Exception as exc:
        record_error("s1724_mmseqs_homology", exc)
    finally:
        shutil.rmtree(TMP / "mmseqs_install", ignore_errors=True)
        shutil.rmtree(TMP / "mmseqs_s1724", ignore_errors=True)

    # Stage 3: figures.zip -> Rocklin 2026 baseline prediction Parquets
    try:
        if not figs_zip.is_file():
            raise FileNotFoundError(f"Missing {figs_zip}")
        with zipfile.ZipFile(figs_zip) as zf:
            entries = clean_zip_entries(zf)
            csv_entries = [e for e in entries if e.filename.lower().endswith(".csv")]
            summary["figures_zip_csvs"] = [
                {"filename": e.filename, "bytes": e.file_size} for e in csv_entries
            ]

            fig3_entry = find_zip_member(entries, "fig3__megascale_point_mutants_predictions.csv")
            fig4_entry = find_zip_member(entries, "fig4__thermomut_predictions.csv")

            out_fig3 = WORK / "rocklin2026_fig3_megascale_preds.parquet"
            out_fig4 = WORK / "rocklin2026_fig4_s1724_preds.parquet"

            fig3_info = convert_fig_csv_to_parquet(zf, fig3_entry, out_fig3)
            fig4_info = convert_fig_csv_to_parquet(zf, fig4_entry, out_fig4)

            summary["rocklin_figures"] = {
                "fig3_megascale": fig3_info,
                "fig4_s1724": fig4_info,
            }
            summary["outputs"][out_fig3.name] = file_hashes(out_fig3)
            summary["outputs"][out_fig4.name] = file_hashes(out_fig4)
            print(
                f"[job7] Extracted Rocklin fig3 ({fig3_info['rows']:,} rows) "
                f"and fig4 ({fig4_info['rows']:,} rows)",
                flush=True,
            )
    except Exception as exc:
        record_error("figures_zip_etl", exc)
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)

    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["status"] = "ok" if not errors else "completed_with_errors"
    sum_p = WORK / "job7_summary.json"
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
