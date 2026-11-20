"""
remote/cpu_jobs/job8_epistasis_etl_zen7844779.py
===============================================
Kaggle CPU Worker 8 (Account: wali0754) — Complete 39-Column Zenodo 7844779
Double-Mutant Epistasis ETL Across All 3 Split Tiers (id25, fold_tm60, id30).

Downloads into /tmp/holo_job8/ (1 TB free on Kaggle CPU):
  1. ``Processed_K50_dG_datasets.zip`` (1017 MB)
  2. ``Data_tables_for_figs.zip`` (24 MB)

Pipeline:
  * Streams ``Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv``
    directly via ``zipfile.ZipFile.open`` + ``io.TextIOWrapper`` + ``csv.DictReader``
    (skipping ``__MACOSX/`` resource forks).
  * Enforces all invariants from ``scripts/build_epistasis_table.py``:
      - Canonical ``site_i < site_j`` (0-based into ``parent_seq``)
      - Exact ``wt_seq[site_i] == wt_i`` and ``wt_seq[site_j] == wt_j``
      - Reading-frame-verified ``wt_dna`` CDS slicing
      - ``ddg_a`` / ``ddg_b`` matched from the SAME parent construct
      - ``dddg = ddg_ab - ddg_a - ddg_b``
      - Internal stabilizing-positive sign checks + numerical parity check
        against ``megascale_ddg_pairs.parquet`` from ``/kaggle/input``
  * Resolves MegaScale PDBs from ``/kaggle/input`` (``megascale_pdbs`` /
    ``megascale_pdbs.tar.gz``) with ``|`` and ``:`` normalized to ``_`` on both
    construct base names and PDB stems (recovering all ``EA|...`` constructs),
    and computes exact 3D C-beta distance ``dist_cb``.
  * Emits 3 split-aligned double-mutant Parquets in ``/kaggle/working``:
      - ``megascale_double_mutants_id25.parquet``
      - ``megascale_double_mutants_fold_tm60.parquet``
      - ``megascale_double_mutants_id30.parquet``
  * Extracts ``Data_tables_for_figs/dG_for_double_mutants_Fig4.csv`` summary
    stats, cleans up temporary archives in ``/tmp/holo_job8/``, writes
    ``epistasis_etl_summary.json`` and ``job8_summary.json`` (with ``<8 A`` vs
    ``>=8 A`` contact breakdown and SHA-256 digests), and always exits 0.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import io
import json
import math
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

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job8")
TMP = Path("/tmp/holo_job8") if Path("/tmp").is_dir() else WORK / "_tmp"
WORK.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Holo-GeoN/2.0"
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_ORDER)
MAX_MISMATCH_FRACTION = 0.15
NAME_RE = re.compile(r"^(?P<base>[^.]+)\.pdb(?:_(?P<mut>[A-Z]\d+[A-Z]|pross\d+))?$")
SINGLE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S", "CCT": "P", "CCC": "P",
    "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R",
    "CGG": "R", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G",
    "GGG": "G", "TAA": "*", "TAG": "*", "TGA": "*",
}

OUTPUT_COLUMNS = [
    "parent",
    "parent_seq",
    "wt_dna",
    "site_i",
    "wt_i",
    "mt_i",
    "site_j",
    "wt_j",
    "mt_j",
    "ddg_a",
    "ddg_b",
    "ddg_ab",
    "dddg",
    "dist_cb",
    "split",
    "group_used",
]

ZENODO_ZIPS = {
    "Processed_K50_dG_datasets.zip": "https://zenodo.org/api/records/7844779/files/Processed_K50_dG_datasets.zip/content",
    "Data_tables_for_figs.zip": "https://zenodo.org/api/records/7844779/files/Data_tables_for_figs.zip/content",
}


def ensure_packages() -> None:
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow")):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[job8] Installing missing packages: {missing}", flush=True)
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
                f"[job8] Downloaded {dest.name}: {written / 1e6:.2f} MB in {dt_total:.1f}s",
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


def find_zip_member(zf: zipfile.ZipFile, suffix: str) -> zipfile.ZipInfo:
    suf_l = suffix.lower()
    cands = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        norm = info.filename.replace("\\", "/")
        base = norm.rsplit("/", 1)[-1]
        if norm.startswith("__MACOSX/") or base.startswith("._"):
            continue
        if norm.lower().endswith(suf_l):
            cands.append(info)
    if not cands:
        raise FileNotFoundError(f"No zip entry matching *{suffix}")
    return max(cands, key=lambda i: i.file_size)


def translate_dna(dna: str) -> str:
    dna = dna.upper().replace("U", "T")
    return "".join(CODON_TABLE.get(dna[i : i + 3], "X") for i in range(0, len(dna) - 2, 3))


def domain_cds(aa_seq: str, aa_seq_full: str, dna_seq: str) -> str | None:
    if not isinstance(dna_seq, str) or not isinstance(aa_seq_full, str):
        return None
    offset = aa_seq_full.find(aa_seq)
    if offset < 0:
        return None
    cds = dna_seq[offset * 3 : (offset + len(aa_seq)) * 3]
    if len(cds) != 3 * len(aa_seq) or translate_dna(cds) != aa_seq:
        return None
    return cds


def _parse_float(value) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s in ("-", "NA", "na", "NaN", "nan", "None", "none", "<NA>"):
        return float("nan")
    try:
        v = float(s)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _parse_single_code(code: str) -> tuple[int, str, str] | None:
    m = SINGLE_RE.match(code.strip().upper())
    if not m:
        return None
    wt_aa, pos1, mt_aa = m.group(1), int(m.group(2)), m.group(3)
    if wt_aa not in AA_SET or mt_aa not in AA_SET:
        return None
    return pos1, wt_aa, mt_aa


def _parse_double_code(
    mut_type: str,
) -> tuple[tuple[tuple[int, str, str], tuple[int, str, str]] | None, str | None]:
    low = mut_type.lower()
    if "ins" in low or "del" in low:
        return None, "double_contains_indel"
    parts = [p.strip() for p in mut_type.split(":")]
    if len(parts) != 2:
        return None, "not_two_point_substitution"
    p1 = _parse_single_code(parts[0])
    p2 = _parse_single_code(parts[1])
    if p1 is None or p2 is None:
        return None, "unparseable_double_substitution"
    pos1, wt1, mt1 = p1
    pos2, wt2, mt2 = p2
    if wt1 == mt1 or wt2 == mt2:
        return None, "synonymous_residue_in_double"
    site1 = pos1 - 1
    site2 = pos2 - 1
    if site1 == site2:
        return None, "same_site_in_double"
    if site2 < site1:
        return ((site2, wt2, mt2), (site1, wt1, mt1)), None
    return ((site1, wt1, mt1), (site2, wt2, mt2)), None


def normalize_pdb_stem(name: str) -> str:
    """Strip trailing .pdb (case-insensitive), uppercase, and normalize | and : to _."""
    s = name.strip()
    if s.lower().endswith(".pdb"):
        s = s[:-4]
    return s.upper().replace("|", "_").replace(":", "_")


def parse_pdb_cb(path: Path):
    import numpy as np

    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    chain0 = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            if len(line) < 54:
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            chain = line[21]
            if chain0 is None:
                chain0 = chain
            if chain != chain0:
                continue
            resname = line[17:20].strip().upper()
            if resname not in THREE_TO_ONE:
                continue
            atom = line[12:16].strip()
            if atom not in ("CA", "CB"):
                continue
            key = (chain, line[22:27])
            if key not in best:
                best[key] = {"res": resname}
                order.append(key)
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            if atom == "CB" or "xyz" not in best[key]:
                if atom == "CB" or best[key].get("atom") != "CB":
                    best[key]["xyz"] = xyz
                    best[key]["atom"] = atom

    seq = []
    coords = []
    for key in order:
        rec = best[key]
        if "xyz" not in rec:
            continue
        seq.append(THREE_TO_ONE[rec["res"]])
        coords.append(rec["xyz"])
    return "".join(seq), np.asarray(coords, dtype=np.float64)


def align_to_sequence(
    pdb_seq: str,
    target_seq: str,
    max_mismatch_fraction: float = MAX_MISMATCH_FRACTION,
) -> tuple[int | None, int, str | None]:
    if pdb_seq == target_seq:
        return 0, 0, None
    if len(pdb_seq) < len(target_seq):
        return None, 0, "structure_shorter_than_sequence"

    budget = max(1, int(max_mismatch_fraction * len(target_seq)))
    scored = [
        (sum(a != b for a, b in zip(pdb_seq[o : o + len(target_seq)], target_seq)), o)
        for o in range(len(pdb_seq) - len(target_seq) + 1)
    ]
    best_mismatches, best_offset = min(scored)
    if best_mismatches > budget:
        return None, best_mismatches, "too_many_mismatches_vs_structure"
    if sum(1 for m, _ in scored if m == best_mismatches) > 1:
        return None, best_mismatches, "ambiguous_alignment"
    return best_offset, best_mismatches, None


def collect_megascale_pdbs() -> tuple[dict[str, Path], int]:
    available: dict[str, Path] = {}
    n_files = 0

    def index_pdb(p: Path) -> None:
        nonlocal n_files
        n_files += 1
        available[p.stem.upper()] = p
        available[normalize_pdb_stem(p.stem)] = p

    # 1. Check extracted megascale_pdbs folder in /kaggle/input or local workspace
    for root in ("/kaggle/input", "."):
        rp = Path(root)
        if not rp.exists():
            continue
        for hit in rp.rglob("megascale_pdbs"):
            if hit.is_dir():
                for p in hit.rglob("*.pdb"):
                    index_pdb(p)
        if n_files > 0:
            return available, n_files

    # 2. Check megascale_pdbs.tar.gz archive
    tar_p = find_input("megascale_pdbs.tar.gz")
    if tar_p is not None and tar_p.is_file() and tarfile.is_tarfile(tar_p):
        ext_dir = TMP / "ms_pdbs"
        ext_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:gz") as tf:
            tf.extractall(ext_dir)
        for p in ext_dir.rglob("*.pdb"):
            index_pdb(p)
        if n_files > 0:
            return available, n_files

    # 3. Fallback: scan /kaggle/input for *.pdb excluding alphafold_fireprot*
    for root in ("/kaggle/input", "DATASETS"):
        rp = Path(root)
        if not rp.exists():
            continue
        for p in rp.rglob("*.pdb"):
            low_parts = [part.lower() for part in p.parts]
            if any("alphafold_fireprot" in part for part in low_parts):
                continue
            index_pdb(p)
    return available, n_files


def resolve_pdb_path(parent: str, available: dict[str, Path]) -> Path | None:
    m = NAME_RE.match(parent)
    if m:
        base = m.group("base")
        for key in (base.upper(), normalize_pdb_stem(base)):
            hit = available.get(key)
            if hit is not None:
                return hit
    if ".pdb" in parent.lower():
        prefix = parent[: parent.lower().find(".pdb")]
        for key in (prefix.upper(), normalize_pdb_stem(prefix)):
            hit = available.get(key)
            if hit is not None:
                return hit
    return available.get(normalize_pdb_stem(parent))


def align_parent_pdbs(
    parents_needed: list[str],
    wt_info: dict[str, dict],
    available: dict[str, Path],
) -> tuple[dict[str, object], dict[str, int], dict[int, int], dict[str, int]]:
    aligned: dict[str, object] = {}
    skip_reasons: dict[str, int] = {}
    mismatch_hist: dict[int, int] = {}
    pipe_stats = {"pipe_or_colon_parents_considered": 0, "pipe_or_colon_parents_aligned": 0}

    def note(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if not available:
        for _ in parents_needed:
            note("pdb_dir_missing")
        return aligned, skip_reasons, mismatch_hist, pipe_stats

    for parent in parents_needed:
        is_special = ("|" in parent) or (":" in parent)
        if is_special:
            pipe_stats["pipe_or_colon_parents_considered"] += 1

        info = wt_info.get(parent)
        if info is None:
            note("no_wildtype_sequence")
            continue

        pfile = resolve_pdb_path(parent, available)
        if pfile is None:
            note("no_pdb_for_construct")
            continue

        pdb_seq, coords = parse_pdb_cb(pfile)
        if len(coords) == 0:
            note("pdb_had_no_usable_residues")
            continue

        wt_seq = info["aa_seq"]
        offset, mismatches, err = align_to_sequence(pdb_seq, wt_seq, MAX_MISMATCH_FRACTION)
        if offset is None:
            note(err or "alignment_failed")
            continue

        sub = coords[offset : offset + len(wt_seq)]
        if len(sub) != len(wt_seq):
            note("structure_shorter_than_sequence")
            continue

        mismatch_hist[mismatches] = mismatch_hist.get(mismatches, 0) + 1
        aligned[parent] = sub
        if is_special:
            pipe_stats["pipe_or_colon_parents_aligned"] += 1

    return aligned, skip_reasons, mismatch_hist, pipe_stats


def verify_singles_parity(single_ddg: dict[tuple[str, str], float], singles_path: Path) -> dict[str, object]:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    schema_cols = set(pq.ParquetFile(singles_path).schema_arrow.names)
    ddg_col = "ddG" if "ddG" in schema_cols else ("ddg" if "ddg" in schema_cols else None)
    if not {"parent", "mut_type"}.issubset(schema_cols) or ddg_col is None:
        raise RuntimeError(f"Singles parquet {singles_path} missing parent/mut_type/ddG")

    ref = pd.read_parquet(singles_path, columns=["parent", "mut_type", ddg_col])
    csv_vals: list[float] = []
    ref_vals: list[float] = []
    for row in ref.itertuples(index=False):
        key = (str(row.parent).strip(), str(row.mut_type).strip().upper())
        v = single_ddg.get(key)
        if v is not None and math.isfinite(v):
            rv = float(getattr(row, ddg_col))
            if math.isfinite(rv):
                csv_vals.append(v)
                ref_vals.append(rv)

    n_shared = len(csv_vals)
    if n_shared == 0:
        raise RuntimeError(f"0 shared single mutants found between CSV and {singles_path}")

    a = np.asarray(csv_vals, dtype=np.float64)
    b = np.asarray(ref_vals, dtype=np.float64)
    nz = (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6)
    sign_agree = float((np.sign(a[nz]) == np.sign(b[nz])).mean()) if nz.any() else 1.0
    max_diff = float(np.max(np.abs(a - b)))
    pearson_r = (
        float(np.corrcoef(a, b)[0, 1])
        if (n_shared >= 3 and np.std(a) > 0 and np.std(b) > 0)
        else float("nan")
    )
    if sign_agree < 0.99 or (math.isfinite(pearson_r) and pearson_r < 0.99):
        raise RuntimeError(
            f"Sign/value parity failure vs {singles_path}: "
            f"sign_agreement={sign_agree:.4f}, pearson_r={pearson_r:.4f}"
        )
    return {
        "n_shared_singles": n_shared,
        "sign_agreement": round(sign_agree, 6),
        "pearson_r": round(pearson_r, 6) if math.isfinite(pearson_r) else None,
        "max_abs_diff": round(max_diff, 6),
    }


def load_splits_lookup(splits_path: Path) -> tuple[dict[str, tuple[str, str]], str]:
    import pandas as pd
    import pyarrow.parquet as pq

    schema_cols = set(pq.ParquetFile(splits_path).schema_arrow.names)
    if "split" not in schema_cols:
        raise RuntimeError(f"Splits file missing 'split' column: {splits_path}")
    use_key = "parent" if "parent" in schema_cols else "parent_seq"
    if use_key not in schema_cols:
        raise RuntimeError(f"Splits file needs 'parent' or 'parent_seq': {splits_path}")

    cols = [use_key, "split"]
    has_group = "group_used" in schema_cols
    if has_group:
        cols.append("group_used")

    sp = pd.read_parquet(splits_path, columns=cols).drop_duplicates()
    if sp[use_key].duplicated().any():
        raise RuntimeError(f"Splits file {splits_path} maps {use_key} to multiple splits/groups")

    lookup: dict[str, tuple[str, str]] = {}
    for row in sp.itertuples(index=False):
        k = str(getattr(row, use_key))
        if use_key == "parent_seq":
            k = k.upper()
        sp_val = str(row.split)
        grp_val = str(row.group_used) if has_group and pd.notna(row.group_used) else ""
        lookup[k] = (sp_val, grp_val)
    return lookup, use_key


def stream_tsuboyama_csv(proc_zip: Path) -> dict[str, object]:
    """Stream Dataset2_Dataset3 CSV from zip and extract validated dicts."""
    dropped: dict[str, int] = {}

    def drop(reason: str, n: int = 1) -> None:
        if n:
            dropped[reason] = dropped.get(reason, 0) + n

    wt_info: dict[str, dict] = {}
    single_acc: dict[tuple[str, str], list] = {}
    double_acc: dict[tuple[str, int, str, str, int, str, str], list] = {}

    rows_in = 0
    raw_singles_kept = 0
    raw_doubles_kept = 0
    n_flagged = 0
    n_flagged_pos = 0
    min_flagged_ddg = float("inf")
    single_sum_all = 0.0
    single_n_all = 0

    with zipfile.ZipFile(proc_zip) as zf:
        member = find_zip_member(zf, "Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
        print(f"[job8] Streaming {member.filename} ({member.file_size / 1e6:.1f} MB) ...", flush=True)
        with zf.open(member) as raw_fh:
            text_fh = io.TextIOWrapper(raw_fh, encoding="utf-8", errors="replace", newline="")
            reader = csv.DictReader(text_fh)
            if not reader.fieldnames:
                raise RuntimeError(f"Empty CSV header in {member.filename}")
            raw_columns = list(reader.fieldnames)
            fset = set(raw_columns)
            if not {"WT_name", "mut_type", "aa_seq"}.issubset(fset):
                raise RuntimeError(f"Missing WT_name/mut_type/aa_seq in {raw_columns}")
            has_ddg_ml = "ddG_ML" in fset
            dg_col = "dG_ML" if "dG_ML" in fset else ("deltaG_ML" if "deltaG_ML" in fset else None)
            if not has_ddg_ml and dg_col is None:
                raise RuntimeError("CSV needs ddG_ML and/or dG_ML/deltaG_ML")
            has_cds_cols = {"aa_seq_full", "dna_seq"}.issubset(fset)
            has_stab_col = "Stabilizing_mut" in fset

            for rows_in, row in enumerate(reader, 1):
                if rows_in % 200_000 == 0:
                    print(
                        f"  [chunk] {rows_in:,} rows | {len(wt_info)} wt | "
                        f"{len(single_acc):,} singles | {len(double_acc):,} doubles",
                        flush=True,
                    )
                mut_type = (row.get("mut_type") or "").strip()
                parent = (row.get("WT_name") or "").strip()
                if not parent:
                    drop("missing_parent_name")
                    continue

                if mut_type == "wt":
                    aa_seq = (row.get("aa_seq") or "").strip().upper()
                    if not aa_seq:
                        drop("wt_missing_aa_seq")
                        continue
                    cds = (
                        domain_cds(
                            aa_seq,
                            (row.get("aa_seq_full") or "").strip().upper(),
                            (row.get("dna_seq") or "").strip(),
                        )
                        if has_cds_cols
                        else None
                    )
                    dg_val = _parse_float(row.get(dg_col)) if dg_col else float("nan")
                    rec = wt_info.get(parent)
                    if rec is None:
                        wt_info[parent] = {
                            "aa_seq": aa_seq,
                            "cds": cds,
                            "dg_sum": dg_val if math.isfinite(dg_val) else 0.0,
                            "dg_count": 1 if math.isfinite(dg_val) else 0,
                        }
                    else:
                        if rec["cds"] is None and cds is not None:
                            rec["cds"] = cds
                        if math.isfinite(dg_val):
                            rec["dg_sum"] += dg_val
                            rec["dg_count"] += 1
                    continue

                if ":" not in mut_type:
                    parsed_s = _parse_single_code(mut_type)
                    if parsed_s is None:
                        drop("not_single_or_double_substitution")
                        continue
                    pos1, wt_aa, mt_aa = parsed_s
                    if wt_aa == mt_aa:
                        drop("synonymous_single")
                        continue
                    ddg_val = _parse_float(row.get("ddG_ML")) if has_ddg_ml else float("nan")
                    dg_val = _parse_float(row.get(dg_col)) if dg_col else float("nan")
                    if has_ddg_ml:
                        if not math.isfinite(ddg_val) or (dg_col is not None and not math.isfinite(dg_val)):
                            drop("non_finite_single_dg_ddg")
                            continue
                        val_to_store = ddg_val
                    else:
                        if not math.isfinite(dg_val):
                            drop("non_finite_single_dg_ddg")
                            continue
                        val_to_store = dg_val

                    raw_singles_kept += 1
                    skey = (parent, f"{wt_aa}{pos1}{mt_aa}")
                    acc = single_acc.get(skey)
                    if acc is None:
                        single_acc[skey] = [val_to_store, 1]
                    else:
                        acc[0] += val_to_store
                        acc[1] += 1

                    if has_ddg_ml:
                        single_sum_all += val_to_store
                        single_n_all += 1
                        if has_stab_col and str(row.get("Stabilizing_mut") or "").strip().lower() == "true":
                            n_flagged += 1
                            if val_to_store > 0:
                                n_flagged_pos += 1
                            if val_to_store < min_flagged_ddg:
                                min_flagged_ddg = val_to_store
                    continue

                parsed_d, err_reason = _parse_double_code(mut_type)
                if parsed_d is None:
                    drop(err_reason or "invalid_double_substitution")
                    continue
                (site_i, wt_i, mt_i), (site_j, wt_j, mt_j) = parsed_d
                ddg_ab = _parse_float(row.get("ddG_ML")) if has_ddg_ml else float("nan")
                dg_ab = _parse_float(row.get(dg_col)) if dg_col else float("nan")
                if has_ddg_ml:
                    if not math.isfinite(ddg_ab) or (dg_col is not None and not math.isfinite(dg_ab)):
                        drop("non_finite_double_dg_ddg")
                        continue
                    dval_to_store = ddg_ab
                else:
                    if not math.isfinite(dg_ab):
                        drop("non_finite_double_dg_ddg")
                        continue
                    dval_to_store = dg_ab

                mseq = (row.get("aa_seq") or "").strip().upper()
                if mseq and (site_j >= len(mseq) or mseq[site_i] != mt_i or mseq[site_j] != mt_j):
                    drop("double_mutant_seq_mismatch")
                    continue

                raw_doubles_kept += 1
                dkey = (parent, site_i, wt_i, mt_i, site_j, wt_j, mt_j)
                dacc = double_acc.get(dkey)
                if dacc is None:
                    double_acc[dkey] = [dval_to_store, 1]
                else:
                    dacc[0] += dval_to_store
                    dacc[1] += 1

    if not double_acc:
        raise RuntimeError("0 valid double substitutions found in Tsuboyama CSV")

    wt_mean_dg: dict[str, float] = {
        p: (rec["dg_sum"] / rec["dg_count"])
        for p, rec in wt_info.items()
        if rec["dg_count"] > 0
    }
    single_ddg: dict[tuple[str, str], float] = {}
    for skey, acc in single_acc.items():
        parent = skey[0]
        mean_val = acc[0] / acc[1]
        if has_ddg_ml:
            single_ddg[skey] = mean_val
        elif parent in wt_mean_dg:
            single_ddg[skey] = mean_val - wt_mean_dg[parent]

    if not has_ddg_ml and single_ddg:
        vals = list(single_ddg.values())
        single_sum_all = float(sum(vals))
        single_n_all = len(vals)

    mean_single_ddg = (single_sum_all / single_n_all) if single_n_all else float("nan")
    implied_stab = (n_flagged_pos / n_flagged) if n_flagged else None
    if implied_stab is not None and implied_stab < 0.99:
        raise RuntimeError(
            f"Only {implied_stab:.2%} of Stabilizing_mut rows have ddG > 0; sign mismatch"
        )
    if single_n_all >= 20 and math.isfinite(mean_single_ddg) and mean_single_ddg >= 0.0:
        raise RuntimeError(
            f"Mean single-mutant ddG is {mean_single_ddg:+.3f}; expected < 0 under stabilizing_positive"
        )

    return {
        "csv_member": member.filename,
        "raw_columns": raw_columns,
        "rows_in_csv": rows_in,
        "wt_info": wt_info,
        "wt_mean_dg": wt_mean_dg,
        "has_ddg_ml": has_ddg_ml,
        "single_acc": single_acc,
        "single_ddg": single_ddg,
        "double_acc": double_acc,
        "raw_singles_kept": raw_singles_kept,
        "raw_doubles_kept": raw_doubles_kept,
        "sign_check_internal": {
            "n_stabilizing_flagged": n_flagged,
            "implied_positive_fraction": implied_stab,
            "min_flagged_ddg": min_flagged_ddg if n_flagged else None,
            "mean_single_ddg": round(mean_single_ddg, 5) if math.isfinite(mean_single_ddg) else None,
        },
        "dropped": dropped,
    }


def emit_split_parquet(
    base_records: list[dict[str, object]],
    splits_path: Path,
    out_path: Path,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    lookup, key_kind = load_splits_lookup(splits_path)
    split_rows: list[dict[str, object]] = []
    absent_parents: dict[str, int] = {}

    for rec in base_records:
        lookup_key = str(rec["parent"]) if key_kind == "parent" else str(rec["parent_seq"])
        sp_entry = lookup.get(lookup_key)
        if sp_entry is None:
            p = str(rec["parent"])
            absent_parents[p] = absent_parents.get(p, 0) + 1
            continue
        row = dict(rec)
        row["split"] = sp_entry[0]
        row["group_used"] = sp_entry[1]
        split_rows.append(row)

    if not split_rows:
        raise RuntimeError(f"0 rows matched splits in {splits_path}")

    df = pd.DataFrame.from_records(split_rows, columns=OUTPUT_COLUMNS)
    df["wt_dna"] = df["wt_dna"].astype("object")
    for col in ("site_i", "site_j"):
        df[col] = df[col].astype(np.int64)
    for col in ("ddg_a", "ddg_b", "ddg_ab", "dddg", "dist_cb"):
        df[col] = df[col].astype(np.float64)

    assert bool((df["site_i"] < df["site_j"]).all()), "canonical site_i < site_j violated"
    assert all(s[i] == w for s, i, w in zip(df["parent_seq"], df["site_i"], df["wt_i"]))
    assert all(s[j] == w for s, j, w in zip(df["parent_seq"], df["site_j"], df["wt_j"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    dist_vals = df["dist_cb"].to_numpy(dtype=np.float64)
    finite_dist = np.isfinite(dist_vals)
    n_lt8 = int((finite_dist & (dist_vals < 8.0)).sum())
    n_ge8 = int((finite_dist & (dist_vals >= 8.0)).sum())
    hashes = file_hashes(out_path)

    return {
        "file": out_path.name,
        "splits_source": splits_path.name,
        "splits_key": key_kind,
        **hashes,
        "rows_written": int(len(df)),
        "unique_parents_written": int(df["parent"].nunique()),
        "cds_coverage_fraction": round(float(df["wt_dna"].notna().mean()), 5),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "parents_per_split": {str(k): int(v) for k, v in df.groupby("split")["parent"].nunique().items()},
        "absent_from_splits": {
            "n_parents": int(len(absent_parents)),
            "n_rows": int(sum(absent_parents.values())),
            "parents": sorted(absent_parents.keys()),
        },
        "contact_breakdown": {
            "rows_with_finite_dist_cb": int(finite_dist.sum()),
            "rows_contact_lt_8A": n_lt8,
            "rows_distant_ge_8A": n_ge8,
            "rows_missing_dist_cb": int((~finite_dist).sum()),
        },
        "dddg_stats": {
            "mean": round(float(df["dddg"].mean()), 5),
            "std": round(float(df["dddg"].std(ddof=1)), 5) if len(df) > 1 else 0.0,
            "min": round(float(df["dddg"].min()), 5),
            "max": round(float(df["dddg"].max()), 5),
        },
    }


def extract_fig4_stats(fig_zip: Path) -> dict[str, object]:
    import numpy as np
    import pandas as pd

    with zipfile.ZipFile(fig_zip) as zf:
        member = find_zip_member(zf, "dG_for_double_mutants_Fig4.csv")
        with zf.open(member) as fh:
            df = pd.read_csv(fh, low_memory=False)

    cols = list(df.columns)
    name_col = next((c for c in cols if "name" in c.lower()), None)
    numeric_stats: dict[str, dict[str, object]] = {}
    for c in cols:
        if c == name_col:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        fin = vals[np.isfinite(vals)]
        if len(fin) > 0:
            numeric_stats[c] = {
                "finite_count": int(len(fin)),
                "mean": round(float(np.mean(fin)), 5),
                "std": round(float(np.std(fin)), 5),
                "min": round(float(np.min(fin)), 5),
                "max": round(float(np.max(fin)), 5),
            }

    return {
        "source_csv": member.filename,
        "raw_columns": cols,
        "rows": int(len(df)),
        "name_column": name_col,
        "unique_names": int(df[name_col].nunique()) if name_col else None,
        "numeric_stats": numeric_stats,
    }


def main() -> int:
    t0 = time.time()
    errors: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "job": "job8_epistasis_etl_zen7844779",
        "zenodo_record": "7844779",
        "downloads": {},
        "fig4_double_mutants_csv": {},
        "epistasis_etl": {},
        "split_outputs": {},
        "outputs": {},
        "errors": errors,
    }

    def record_error(stage: str, exc: Exception) -> None:
        print(f"[job8] ERROR in {stage}: {exc}", flush=True)
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
    proc_zip = dl_dir / "Processed_K50_dG_datasets.zip"
    fig_zip = dl_dir / "Data_tables_for_figs.zip"

    for fname, url in ZENODO_ZIPS.items():
        try:
            dl_info = download_file(url, dl_dir / fname)
            summary["downloads"][fname] = dl_info
        except Exception as exc:
            record_error(f"download_{fname}", exc)

    # Stage 1: Extract Fig 4 double-mutant CSV summary stats from Data_tables_for_figs.zip
    try:
        if not fig_zip.is_file():
            raise FileNotFoundError(f"Missing {fig_zip}")
        summary["fig4_double_mutants_csv"] = extract_fig4_stats(fig_zip)
        print(
            f"[job8] Fig4 CSV rows={summary['fig4_double_mutants_csv']['rows']:,}",
            flush=True,
        )
    except Exception as exc:
        record_error("extract_fig4_stats", exc)
    finally:
        if fig_zip.exists():
            fig_zip.unlink()

    # Stage 2: Stream 39-col Tsuboyama CSV + parity check + normalized PDB alignment
    etl_summary: dict[str, object] = {}
    try:
        import numpy as np

        if not proc_zip.is_file():
            raise FileNotFoundError(f"Missing {proc_zip}")
        parsed = stream_tsuboyama_csv(proc_zip)
        if proc_zip.exists():
            proc_zip.unlink()

        wt_info = parsed["wt_info"]
        wt_mean_dg = parsed["wt_mean_dg"]
        has_ddg_ml = parsed["has_ddg_ml"]
        single_ddg = parsed["single_ddg"]
        double_acc = parsed["double_acc"]
        dropped = parsed["dropped"]

        def drop_post(reason: str) -> None:
            dropped[reason] = dropped.get(reason, 0) + 1

        singles_p = find_input("megascale_ddg_pairs.parquet")
        parity_report = None
        if singles_p is not None:
            parity_report = verify_singles_parity(single_ddg, singles_p)
            print(f"[job8] Singles parity verified: {parity_report}", flush=True)
        else:
            record_error(
                "singles_parity_lookup",
                FileNotFoundError("megascale_ddg_pairs.parquet not found under /kaggle/input"),
            )

        available_pdbs, n_pdb_files = collect_megascale_pdbs()
        double_parents = sorted({k[0] for k in double_acc})
        aligned_coords, pdb_skip_reasons, mismatch_hist, pipe_stats = align_parent_pdbs(
            double_parents, wt_info, available_pdbs
        )
        print(
            f"[job8] PDB alignment: {len(aligned_coords)}/{len(double_parents)} double-mutant parents "
            f"aligned (indexed {n_pdb_files} PDBs; pipe/colon aligned "
            f"{pipe_stats['pipe_or_colon_parents_aligned']}/{pipe_stats['pipe_or_colon_parents_considered']})",
            flush=True,
        )

        base_records: list[dict[str, object]] = []
        for dkey in sorted(double_acc.keys()):
            parent, site_i, wt_i, mt_i, site_j, wt_j, mt_j = dkey
            rec = wt_info.get(parent)
            if rec is None:
                drop_post("no_wildtype_row_for_construct")
                continue
            wt_seq = rec["aa_seq"]
            if site_i < 0 or site_j >= len(wt_seq) or site_i >= site_j:
                drop_post("position_outside_sequence")
                continue
            if wt_seq[site_i] != wt_i or wt_seq[site_j] != wt_j:
                drop_post("wildtype_residue_mismatch")
                continue

            code_a = f"{wt_i}{site_i + 1}{mt_i}"
            code_b = f"{wt_j}{site_j + 1}{mt_j}"
            ddg_a = single_ddg.get((parent, code_a))
            ddg_b = single_ddg.get((parent, code_b))
            if ddg_a is None or ddg_b is None:
                drop_post("missing_single_mutant_ddg")
                continue

            dacc = double_acc[dkey]
            raw_double_mean = dacc[0] / dacc[1]
            if has_ddg_ml:
                ddg_ab = raw_double_mean
            else:
                if parent not in wt_mean_dg:
                    drop_post("no_wildtype_dg_for_construct")
                    continue
                ddg_ab = raw_double_mean - wt_mean_dg[parent]

            dddg = float(ddg_ab - ddg_a - ddg_b)
            coords = aligned_coords.get(parent)
            if coords is not None:
                dist_cb = float(np.linalg.norm(coords[site_i] - coords[site_j]))
            else:
                dist_cb = float("nan")

            base_records.append({
                "parent": parent,
                "parent_seq": wt_seq,
                "wt_dna": rec["cds"],
                "site_i": int(site_i),
                "wt_i": wt_i,
                "mt_i": mt_i,
                "site_j": int(site_j),
                "wt_j": wt_j,
                "mt_j": mt_j,
                "ddg_a": float(ddg_a),
                "ddg_b": float(ddg_b),
                "ddg_ab": float(ddg_ab),
                "dddg": dddg,
                "dist_cb": dist_cb,
                "split": "",
                "group_used": "",
            })

        if not base_records:
            raise RuntimeError("0 double-mutant records survived invariant filtering")

        base_dist = np.array([float(r["dist_cb"]) for r in base_records], dtype=np.float64)
        base_fin = np.isfinite(base_dist)
        base_parents = len({str(r["parent"]) for r in base_records})

        # Stage 3: Emit all 3 split-aligned double-mutant Parquets
        split_specs = (
            ("id25", "splits_megascale_id25.parquet", "megascale_double_mutants_id25.parquet"),
            ("fold_tm60", "splits_megascale_fold_tm60.parquet", "megascale_double_mutants_fold_tm60.parquet"),
            ("id30", "splits_megascale.parquet", "megascale_double_mutants_id30.parquet"),
        )
        split_outputs: dict[str, dict[str, object]] = {}
        for tier, in_split_name, out_pq_name in split_specs:
            try:
                sp_path = find_input(in_split_name)
                if sp_path is None:
                    raise FileNotFoundError(f"Split parquet not found: {in_split_name}")
                out_pq = WORK / out_pq_name
                sp_info = emit_split_parquet(base_records, sp_path, out_pq)
                split_outputs[tier] = sp_info
                summary["outputs"][out_pq.name] = {
                    "bytes": sp_info["bytes"],
                    "sha256": sp_info["sha256"],
                    "md5": sp_info["md5"],
                }
                print(
                    f"[job8] Emitted {out_pq.name}: {sp_info['rows_written']:,} rows, "
                    f"contact<8A={sp_info['contact_breakdown']['rows_contact_lt_8A']:,}, "
                    f"distant>=8A={sp_info['contact_breakdown']['rows_distant_ge_8A']:,}",
                    flush=True,
                )
            except Exception as exc:
                record_error(f"emit_split_{tier}", exc)

        summary["split_outputs"] = split_outputs

        etl_summary = {
            "source_csv": parsed["csv_member"],
            "raw_columns": parsed["raw_columns"],
            "n_raw_columns": len(parsed["raw_columns"]),
            "sign_convention": "stabilizing_positive (ddG = dG_mut - dG_wt)",
            "rows_in_csv": int(parsed["rows_in_csv"]),
            "wt_constructs_seen": int(len(wt_info)),
            "raw_singles_kept": int(parsed["raw_singles_kept"]),
            "unique_singles_after_dedup": int(len(single_ddg)),
            "n_duplicate_singles_averaged": int(parsed["raw_singles_kept"] - len(parsed["single_acc"])),
            "raw_doubles_kept": int(parsed["raw_doubles_kept"]),
            "unique_doubles_before_join": int(len(double_acc)),
            "n_duplicate_doubles_averaged": int(parsed["raw_doubles_kept"] - len(double_acc)),
            "validated_base_doubles": len(base_records),
            "validated_base_parents": base_parents,
            "base_contact_breakdown": {
                "rows_with_finite_dist_cb": int(base_fin.sum()),
                "rows_contact_lt_8A": int((base_fin & (base_dist < 8.0)).sum()),
                "rows_distant_ge_8A": int((base_fin & (base_dist >= 8.0)).sum()),
                "rows_missing_dist_cb": int((~base_fin).sum()),
            },
            "pdb_alignment": {
                "pdb_files_indexed": n_pdb_files,
                "parents_considered": int(len(double_parents)),
                "parents_with_aligned_pdb": int(len(aligned_coords)),
                "pipe_or_colon_normalization": pipe_stats,
                "skipped_parents_by_reason": dict(sorted(pdb_skip_reasons.items())),
                "mismatch_histogram": {str(k): int(v) for k, v in sorted(mismatch_hist.items())},
            },
            "sign_check_internal": parsed["sign_check_internal"],
            "sign_parity_vs_singles": parity_report,
            "dropped_by_reason": dict(sorted(dropped.items())),
            "split_outputs": split_outputs,
        }
        summary["epistasis_etl"] = etl_summary
        del base_records
        gc.collect()
    except Exception as exc:
        record_error("epistasis_etl", exc)
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)
        shutil.rmtree(TMP / "ms_pdbs", ignore_errors=True)

    etl_sum_p = WORK / "epistasis_etl_summary.json"
    etl_sum_p.write_text(
        json.dumps(etl_summary if etl_summary else {"errors": errors}, indent=2),
        encoding="utf-8",
    )
    summary["outputs"][etl_sum_p.name] = file_hashes(etl_sum_p)

    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["status"] = "ok" if not errors else "completed_with_errors"
    sum_p = WORK / "job8_summary.json"
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
