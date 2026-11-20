"""
scripts/build_epistasis_table.py
================================
Memory-safe streaming ETL that builds the MegaScale double-mutant epistasis
table (``megascale_double_mutants.parquet``) directly from the Tsuboyama et al.
(2023) Zenodo 7844779 processed CSV (``Tsuboyama2023_Dataset2_Dataset3_20230416.csv``)
and ``AlphaFold_model_PDBs/``.

Why streaming
-------------
The shipped CSV is ~665 MB (~1.8M rows, 37 columns). Reading only the columns
needed via ``csv.DictReader`` keeps peak memory tiny while extracting:
  * wild-type domain sequences + reading-frame-verified ``wt_dna`` CDSs
  * all finite single-substitution measurements (for matching ``ddg_a``, ``ddg_b``)
  * all reliable exact 2-point substitutions (no ins/del/wt, no synonymous
    halves like ``T9Q:N23N``, no same-site collisions, finite dG/ddG)

Invariants enforced loudly before writing
-----------------------------------------
1. **Sign convention:** ``ddG = dG_mut - dG_wt`` (strictly stabilizing-positive),
   verified against ``Stabilizing_mut``, mean single-mutant ddG (< 0), and
   (when supplied) exact parity against ``megascale_ddg_pairs.parquet``.
2. **Canonical pair ordering:** ``site_i < site_j`` (0-based indices into
   ``parent_seq``), swapping ``(i, a) <-> (j, b)`` whenever the CSV lists the
   higher-index site first, and matching ``ddg_a`` to ``site_i`` and ``ddg_b``
   to ``site_j`` from the SAME parent construct.
3. **Sequence identity check:** ``parent_seq[site_i] == wt_i`` and
   ``parent_seq[site_j] == wt_j`` on 100% of emitted rows.
4. **3D C-beta distance:** Exact ``dist_cb`` (Angstroms) computed from
   ``AlphaFold_model_PDBs/`` via ``src/geometry.py`` (``parse_pdb_cb`` +
   ``align_to_sequence`` + ``NAME_RE``) so ``scripts/train_epistasis.py`` can
   stratify contacting (``< 8 A``) vs. distant (``>= 8 A``) pairs.
5. **Split mapping:** Maps ``split`` and ``group_used`` from ``--splits``
   (e.g. ``splits_megascale_id25.parquet`` or ``splits_megascale_fold_tm60.parquet``),
   recording exact counts of parents absent from ``--splits`` in
   ``epistasis_etl_summary.json``.

Usage
-----
python -m scripts.build_epistasis_table \\
    --source DATASETS/DATASETS/dataset/mega_scale_cdna/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \\
    --pdb-dir DATASETS/DATASETS/dataset/mega_scale_cdna/AlphaFold_model_PDBs \\
    --splits ../extras/zenodo_staging/Holo-StabBench-v1.0/splits/splits_megascale_id25.parquet \\
    --singles DATASETS/DATASETS/data/megascale_ddg_pairs.parquet \\
    --out DATASETS/DATASETS/data/megascale_double_mutants.parquet
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.build_megascale_table import domain_cds
from src.geometry import (
    MAX_MISMATCH_FRACTION,
    NAME_RE,
    align_to_sequence,
    normalize_pdb_stem,
    parse_pdb_cb,
    resolve_pdb_path,
)
from src.substitution import AA_TO_IDX

SINGLE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

DEFAULT_SOURCE = os.path.join(
    "dataset", "mega_scale_cdna", "Processed_K50_dG_datasets",
    "Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
)
DEFAULT_PDB_DIR = os.path.join("dataset", "mega_scale_cdna", "AlphaFold_model_PDBs")

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


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


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
    if wt_aa not in AA_TO_IDX or mt_aa not in AA_TO_IDX:
        return None
    return pos1, wt_aa, mt_aa


def _parse_double_code(
    mut_type: str,
) -> tuple[tuple[tuple[int, str, str], tuple[int, str, str]] | None, str | None]:
    """
    Parse an exact two-point substitution string ``A12V:K34E``.

    Returns ``(((site_i, wt_i, mt_i), (site_j, wt_j, mt_j)), None)`` with
    0-based ``site_i < site_j``, or ``(None, drop_reason)``.
    """
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


def _load_splits_lookup(splits_path: Path) -> tuple[dict[str, tuple[str, str]], str]:
    """
    Load ``(key -> (split, group_used))`` from a splits Parquet file.

    Supports both ``parent``-keyed tables and ``parent_seq``-keyed MegaScale
    split tables (``splits_megascale_*.parquet``).
    """
    schema_cols = set(pq.ParquetFile(splits_path).schema_arrow.names)
    if "split" not in schema_cols:
        raise ValueError(f"Splits file missing 'split' column: {splits_path}")
    use_key = "parent" if "parent" in schema_cols else "parent_seq"
    if use_key not in schema_cols:
        raise ValueError(f"Splits file needs 'parent' or 'parent_seq' column: {splits_path}")

    cols = [use_key, "split"]
    has_group = "group_used" in schema_cols
    if has_group:
        cols.append("group_used")

    sp = pd.read_parquet(splits_path, columns=cols).drop_duplicates()
    if sp[use_key].duplicated().any():
        raise ValueError(
            f"Splits file {splits_path} maps a {use_key} to multiple splits/groups; "
            "splits must be parent-disjoint."
        )
    lookup: dict[str, tuple[str, str]] = {}
    for row in sp.itertuples(index=False):
        k = str(getattr(row, use_key))
        if use_key == "parent_seq":
            k = k.upper()
        sp_val = str(row.split)
        grp_val = str(row.group_used) if has_group and pd.notna(row.group_used) else ""
        lookup[k] = (sp_val, grp_val)
    return lookup, use_key


def _verify_singles_parity(
    single_ddg: dict[tuple[str, str], float],
    singles_path: Path,
) -> dict:
    """Verify exact sign & numerical parity against ``megascale_ddg_pairs.parquet``."""
    schema_cols = set(pq.ParquetFile(singles_path).schema_arrow.names)
    ddg_col = "ddG" if "ddG" in schema_cols else ("ddg" if "ddg" in schema_cols else None)
    if not {"parent", "mut_type"}.issubset(schema_cols) or ddg_col is None:
        raise ValueError(
            f"Singles parquet {singles_path} must contain parent, mut_type, ddG/ddg"
        )
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
        raise SystemExit(
            f"[epistasis_etl] REFUSING: 0 shared single mutants found between CSV "
            f"and {singles_path}; cannot verify sign parity."
        )
    a = np.asarray(csv_vals, dtype=np.float64)
    b = np.asarray(ref_vals, dtype=np.float64)
    nz = (np.abs(a) > 1e-6) & (np.abs(b) > 1e-6)
    sign_agree = float((np.sign(a[nz]) == np.sign(b[nz])).mean()) if nz.any() else 1.0
    max_diff = float(np.max(np.abs(a - b)))
    pearson_r = float(np.corrcoef(a, b)[0, 1]) if (n_shared >= 3 and np.std(a) > 0 and np.std(b) > 0) else float("nan")

    if sign_agree < 0.99 or (math.isfinite(pearson_r) and pearson_r < 0.99):
        raise SystemExit(
            f"[epistasis_etl] REFUSING: sign/value parity failure vs {singles_path}: "
            f"sign_agreement={sign_agree:.4f}, pearson_r={pearson_r:.4f}"
        )
    return {
        "n_shared_singles": n_shared,
        "sign_agreement": sign_agree,
        "pearson_r": pearson_r,
        "max_abs_diff": max_diff,
    }


def _align_parent_pdbs(
    parents_needed: list[str],
    wt_info: dict[str, dict],
    pdb_dir: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[int, int]]:
    """Parse and uniquely align PDB C-beta coordinates once per parent construct."""
    aligned: dict[str, np.ndarray] = {}
    skip_reasons: dict[str, int] = {}
    mismatch_hist: dict[int, int] = {}

    def note(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if pdb_dir is None or not pdb_dir.is_dir():
        for p in parents_needed:
            note("pdb_dir_missing")
        return aligned, skip_reasons, mismatch_hist

    available = {
        normalize_pdb_stem(f): f
        for f in os.listdir(pdb_dir)
        if f.lower().endswith(".pdb")
    }

    for parent in parents_needed:
        info = wt_info.get(parent)
        if info is None:
            note("no_wildtype_sequence")
            continue
        m = NAME_RE.match(parent)
        if not m:
            note("unparseable_construct_name")
            continue
        fname = resolve_pdb_path(parent, available)
        if fname is None:
            note("no_pdb_for_construct")
            continue

        pdb_seq, coords = parse_pdb_cb(str(pdb_dir / fname))
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
    return aligned, skip_reasons, mismatch_hist


def build_epistasis_table(
    source: str | Path,
    out_path: str | Path,
    *,
    pdb_dir: str | Path | None = None,
    splits_path: str | Path | None = None,
    singles_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    keep_unsplit: bool = False,
    chunk_size: int = 50_000,
) -> dict:
    """
    Stream ``source`` CSV and write the double-mutant epistasis Parquet table
    and companion ``epistasis_etl_summary.json``.
    """
    src = Path(source)
    out = Path(out_path)
    pdb_p = Path(pdb_dir) if pdb_dir is not None and str(pdb_dir).strip() else None
    splits_p = Path(splits_path) if splits_path is not None and str(splits_path).strip() else None
    singles_p = Path(singles_path) if singles_path is not None and str(singles_path).strip() else None
    sum_p = Path(summary_path) if summary_path is not None else out.with_name("epistasis_etl_summary.json")

    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    dropped: dict[str, int] = {}

    def drop(reason: str, n: int = 1) -> None:
        if n:
            dropped[reason] = dropped.get(reason, 0) + n

    wt_info: dict[str, dict] = {}
    # Single measurements: (parent, mut_code) -> [val_sum, count]
    single_acc: dict[tuple[str, str], list] = {}
    # Double measurements: (parent, site_i, wt_i, mt_i, site_j, wt_j, mt_j) -> [val_sum, count]
    double_acc: dict[tuple[str, int, str, str, int, str, str], list] = {}

    rows_in = 0
    raw_singles_kept = 0
    raw_doubles_kept = 0
    n_flagged = 0
    n_flagged_pos = 0
    min_flagged_ddg = float("inf")
    single_sum_all = 0.0
    single_n_all = 0

    print(f"[epistasis_etl] streaming {src}")
    with _open_text(src) as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit(f"[epistasis_etl] empty CSV header in {src}")
        fset = set(reader.fieldnames)
        if not {"WT_name", "mut_type", "aa_seq"}.issubset(fset):
            raise SystemExit(
                f"[epistasis_etl] CSV {src} missing required columns (WT_name, mut_type, aa_seq)"
            )
        has_ddg_ml = "ddG_ML" in fset
        dg_col = "dG_ML" if "dG_ML" in fset else ("deltaG_ML" if "deltaG_ML" in fset else None)
        if not has_ddg_ml and dg_col is None:
            raise SystemExit(
                f"[epistasis_etl] CSV {src} needs ddG_ML and/or dG_ML/deltaG_ML"
            )
        has_cds_cols = {"aa_seq_full", "dna_seq"}.issubset(fset)
        has_stab_col = "Stabilizing_mut" in fset

        for rows_in, row in enumerate(reader, 1):
            if chunk_size > 0 and rows_in % chunk_size == 0:
                print(
                    f"  [chunk] {rows_in:,} rows read | {len(wt_info)} wt | "
                    f"{len(single_acc):,} unique singles | {len(double_acc):,} unique doubles",
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
        raise SystemExit("[epistasis_etl] no valid double substitutions found in source CSV")

    # Resolve single-mutant ddG values (using WT dG subtraction only when ddG_ML is absent).
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
        else:
            if parent not in wt_mean_dg:
                continue
            single_ddg[skey] = mean_val - wt_mean_dg[parent]

    if not has_ddg_ml and single_ddg:
        vals = list(single_ddg.values())
        single_sum_all = float(sum(vals))
        single_n_all = len(vals)

    # Internal sign verification (positive = stabilizing).
    mean_single_ddg = (single_sum_all / single_n_all) if single_n_all else float("nan")
    implied_stab = (n_flagged_pos / n_flagged) if n_flagged else None
    if implied_stab is not None and implied_stab < 0.99:
        raise SystemExit(
            f"[epistasis_etl] REFUSING: only {implied_stab:.2%} of Stabilizing_mut rows "
            "have ddG > 0; sign convention mismatch."
        )
    if single_n_all >= 20 and math.isfinite(mean_single_ddg) and mean_single_ddg >= 0.0:
        raise SystemExit(
            f"[epistasis_etl] REFUSING: mean single-mutant ddG is {mean_single_ddg:+.3f}; "
            "expected negative under stabilizing_positive convention."
        )

    parity_report = None
    if singles_p is not None:
        parity_report = _verify_singles_parity(single_ddg, singles_p)

    # Load splits lookup if supplied.
    splits_lookup: dict[str, tuple[str, str]] | None = None
    splits_key_kind: str | None = None
    if splits_p is not None:
        splits_lookup, splits_key_kind = _load_splits_lookup(splits_p)

    # Align PDBs once per parent occurring in double_acc.
    double_parents = sorted({k[0] for k in double_acc})
    aligned_coords, pdb_skip_reasons, mismatch_hist = _align_parent_pdbs(
        double_parents, wt_info, pdb_p
    )

    # Build final double-mutant records.
    records: list[dict] = []
    absent_parents: dict[str, int] = {}
    for dkey in sorted(double_acc.keys()):
        parent, site_i, wt_i, mt_i, site_j, wt_j, mt_j = dkey
        rec = wt_info.get(parent)
        if rec is None:
            drop("no_wildtype_row_for_construct")
            continue
        wt_seq = rec["aa_seq"]
        if site_i < 0 or site_j >= len(wt_seq) or site_i >= site_j:
            drop("position_outside_sequence")
            continue
        if wt_seq[site_i] != wt_i or wt_seq[site_j] != wt_j:
            drop("wildtype_residue_mismatch")
            continue

        code_a = f"{wt_i}{site_i + 1}{mt_i}"
        code_b = f"{wt_j}{site_j + 1}{mt_j}"
        ddg_a = single_ddg.get((parent, code_a))
        ddg_b = single_ddg.get((parent, code_b))
        if ddg_a is None or ddg_b is None:
            drop("missing_single_mutant_ddg")
            continue

        dacc = double_acc[dkey]
        raw_double_mean = dacc[0] / dacc[1]
        if has_ddg_ml:
            ddg_ab = raw_double_mean
        else:
            if parent not in wt_mean_dg:
                drop("no_wildtype_dg_for_construct")
                continue
            ddg_ab = raw_double_mean - wt_mean_dg[parent]

        if splits_lookup is not None:
            lookup_key = parent if splits_key_kind == "parent" else wt_seq
            sp_entry = splits_lookup.get(lookup_key)
            if sp_entry is None:
                absent_parents[parent] = absent_parents.get(parent, 0) + 1
                if not keep_unsplit:
                    drop("absent_from_splits")
                    continue
                split_val, group_val = "unsplit", ""
            else:
                split_val, group_val = sp_entry
        else:
            split_val, group_val = "unsplit", parent

        dddg = float(ddg_ab - ddg_a - ddg_b)
        coords = aligned_coords.get(parent)
        if coords is not None:
            dist_cb = float(np.linalg.norm(coords[site_i] - coords[site_j]))
        else:
            dist_cb = float("nan")

        records.append({
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
            "split": split_val,
            "group_used": group_val,
        })

    if not records:
        raise SystemExit("[epistasis_etl] 0 double-mutant rows survived filtering")

    df = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    df["wt_dna"] = df["wt_dna"].astype("object")
    for col in ("site_i", "site_j"):
        df[col] = df[col].astype(np.int64)
    for col in ("ddg_a", "ddg_b", "ddg_ab", "dddg", "dist_cb"):
        df[col] = df[col].astype(np.float64)

    # Hard invariants prior to serialization.
    assert bool((df["site_i"] < df["site_j"]).all()), "canonical ordering site_i < site_j violated"
    assert all(
        s[i] == w for s, i, w in zip(df["parent_seq"], df["site_i"], df["wt_i"])
    ), "wt_i mismatch vs parent_seq"
    assert all(
        s[j] == w for s, j, w in zip(df["parent_seq"], df["site_j"], df["wt_j"])
    ), "wt_j mismatch vs parent_seq"

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    dist_vals = df["dist_cb"].to_numpy(dtype=np.float64)
    finite_dist = np.isfinite(dist_vals)
    split_counts = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    parents_per_split = {
        str(k): int(v) for k, v in df.groupby("split")["parent"].nunique().items()
    }
    absent_rows = int(sum(absent_parents.values()))

    summary = {
        "source": str(src),
        "out": str(out),
        "pdb_dir": str(pdb_p) if pdb_p is not None else None,
        "splits_file": str(splits_p) if splits_p is not None else None,
        "splits_key": splits_key_kind,
        "keep_unsplit": bool(keep_unsplit),
        "singles_file": str(singles_p) if singles_p is not None else None,
        "sign_convention": "stabilizing_positive (ddG = dG_mut - dG_wt)",
        "rows_in_csv": int(rows_in),
        "wt_constructs_seen": int(len(wt_info)),
        "raw_singles_kept": int(raw_singles_kept),
        "unique_singles_after_dedup": int(len(single_ddg)),
        "n_duplicate_singles_averaged": int(raw_singles_kept - len(single_acc)),
        "raw_doubles_kept": int(raw_doubles_kept),
        "unique_doubles_before_join": int(len(double_acc)),
        "n_duplicate_doubles_averaged": int(raw_doubles_kept - len(double_acc)),
        "rows_written": int(len(df)),
        "unique_parents_written": int(df["parent"].nunique()),
        "cds_coverage_fraction": float(df["wt_dna"].notna().mean()),
        "split_counts": split_counts,
        "parents_per_split": parents_per_split,
        "absent_from_splits": {
            "n_parents": int(len(absent_parents)),
            "n_rows": absent_rows,
            "action": "tagged_unsplit" if keep_unsplit else "dropped",
            "parents": sorted(absent_parents.keys()),
        },
        "pdb_alignment": {
            "parents_considered": int(len(double_parents)),
            "parents_with_aligned_pdb": int(len(aligned_coords)),
            "rows_with_finite_dist_cb": int(finite_dist.sum()),
            "rows_contact_lt_8A": int((finite_dist & (dist_vals < 8.0)).sum()),
            "rows_distant_ge_8A": int((finite_dist & (dist_vals >= 8.0)).sum()),
            "skipped_parents_by_reason": dict(sorted(pdb_skip_reasons.items())),
            "mismatch_histogram": {str(k): int(v) for k, v in sorted(mismatch_hist.items())},
        },
        "sign_check_internal": {
            "n_stabilizing_flagged": int(n_flagged),
            "implied_positive_fraction": implied_stab,
            "min_flagged_ddg": min_flagged_ddg if n_flagged else None,
            "mean_single_ddg": mean_single_ddg if math.isfinite(mean_single_ddg) else None,
        },
        "sign_parity_vs_singles": parity_report,
        "dddg_stats": {
            "mean": float(df["dddg"].mean()),
            "std": float(df["dddg"].std(ddof=1)) if len(df) > 1 else 0.0,
            "min": float(df["dddg"].min()),
            "max": float(df["dddg"].max()),
        },
        "dropped_by_reason": dict(sorted(dropped.items())),
    }

    sum_p.parent.mkdir(parents=True, exist_ok=True)
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[epistasis_etl] wrote {len(df):,} double mutants across "
        f"{summary['unique_parents_written']} parents -> {out}"
    )
    print(f"[epistasis_etl] summary -> {sum_p}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--splits", default=None, help="Split Parquet (e.g. splits_megascale_id25.parquet)")
    ap.add_argument("--singles", default=None, help="Optional megascale_ddg_pairs.parquet for sign parity check")
    ap.add_argument("--out", default=os.path.join("data", "megascale_double_mutants.parquet"))
    ap.add_argument("--summary", default=None, help="Optional path for epistasis_etl_summary.json")
    ap.add_argument("--keep-unsplit", action="store_true", help="Tag parents absent from --splits as split='unsplit' instead of dropping")
    ap.add_argument("--chunk", type=int, default=50_000, help="Streaming progress interval in rows")
    args = ap.parse_args(argv)

    build_epistasis_table(
        source=args.source,
        out_path=args.out,
        pdb_dir=args.pdb_dir,
        splits_path=args.splits,
        singles_path=args.singles,
        summary_path=args.summary,
        keep_unsplit=args.keep_unsplit,
        chunk_size=args.chunk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
