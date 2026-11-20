"""
remote/cpu_jobs/job1_tm_leakage_audit.py
========================================
Kaggle CPU Worker 1 (Account: mdahbab) — Structural Fold Leakage Audit.

Computes exact pairwise TM-align alignments between EVERY MegaScale test
parent PDB (37/37) and EVERY MegaScale structured train/val parent PDB (336)
using ``tmtools`` (C++ TM-align bindings), answering GEOMETRIC_PLAN.md §6.2:

  "Our split is sequence-identity based at 30%; fold similarity survives below
  30%, and the existing MinHash 8-mer audit cannot see fold similarity at all."

Outputs in /kaggle/working:
  * megascale_tm_leakage_matrix.parquet  (all test x train/val pairwise scores)
  * megascale_tm_test_max.parquet        (max TM-score per test protein -> buckets)
  * megascale_tm_leakage_summary.json    (summary stats, environment & SHA-256)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job1")
TMP = Path("/tmp/holo_job1") if Path("/tmp").is_dir() else WORK / "_tmp"
WORK.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)


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
    return None


def _install_tmtools() -> None:
    try:
        import tmtools  # noqa: F401
        return
    except ImportError:
        pass
    print("[job1] Installing tmtools + pyarrow ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "tmtools", "pyarrow", "pandas", "numpy"],
        check=True,
    )


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}


def parse_ca(pdb_path: Path):
    import numpy as np

    best = {}
    order = []
    chain0 = None
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            ch = line[21]
            if chain0 is None:
                chain0 = ch
            if ch != chain0:
                continue
            if line[12:16].strip() != "CA":
                continue
            res = line[17:20].strip().upper()
            if res not in THREE_TO_ONE:
                continue
            key = (ch, line[22:27])
            if key in best:
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            best[key] = (THREE_TO_ONE[res], xyz)
            order.append(key)
    seq = "".join(best[k][0] for k in order)
    coords = np.asarray([best[k][1] for k in order], dtype=np.float64)
    return seq, coords


def _align_one_test(args):
    from tmtools import tm_align

    test_parent, test_seq, test_coords, ref_list = args
    rows = []
    for ref_parent, ref_split, ref_seq, ref_coords in ref_list:
        res = tm_align(test_coords, ref_coords, test_seq, ref_seq)
        rows.append({
            "test_parent": test_parent,
            "ref_parent": ref_parent,
            "ref_split": ref_split,
            "tm_norm_test": float(res.tm_norm_chain1),
            "tm_norm_ref": float(res.tm_norm_chain2),
            "tm_max": float(max(res.tm_norm_chain1, res.tm_norm_chain2)),
            "rmsd": float(res.rmsd),
            "test_len": len(test_seq),
            "ref_len": len(ref_seq),
        })
    return rows


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    _install_tmtools()
    import re
    import numpy as np
    import pandas as pd

    pairs_p = find_input("megascale_ddg_pairs.parquet")
    splits_p = find_input("splits_megascale.parquet")
    if not pairs_p or not splits_p:
        raise SystemExit(f"Missing inputs: pairs={pairs_p}, splits={splits_p}")

    # Locate PDBs: either extracted directory under /kaggle/input or tar.gz archive
    pdb_map: dict[str, Path] = {}
    tar_p = find_input("megascale_pdbs.tar.gz")
    if tar_p and tar_p.is_file() and tarfile.is_tarfile(tar_p):
        ext_dir = TMP / "pdbs"
        ext_dir.mkdir(exist_ok=True)
        with tarfile.open(tar_p, "r:gz") as tf:
            tf.extractall(ext_dir)
        for p in ext_dir.rglob("*.pdb"):
            pdb_map[p.stem.upper()] = p
    if not pdb_map:
        for root in ("/kaggle/input", "DATASETS"):
            rp = Path(root)
            if rp.exists():
                for p in rp.rglob("*.pdb"):
                    pdb_map[p.stem.upper()] = p
    print(f"[job1] Found {len(pdb_map)} PDB files", flush=True)
    if not pdb_map:
        raise SystemExit("No PDB files found under /kaggle/input or megascale_pdbs.tar.gz")

    pairs = pd.read_parquet(pairs_p, columns=["parent", "wt_seq"]).drop_duplicates()
    splits = pd.read_parquet(splits_p, columns=["parent_seq", "split", "group_used"]).drop_duplicates()
    meta = splits.merge(pairs, left_on="parent_seq", right_on="wt_seq", how="left")
    meta = meta.drop_duplicates("parent_seq")

    name_re = re.compile(r"^(?P<base>[^.]+)\.pdb(?:_(?P<mut>[A-Z]\d+[A-Z]|pross\d+))?$")
    parsed: dict[str, tuple[str, str, np.ndarray]] = {}
    split_of: dict[str, str] = {}
    group_of: dict[str, str] = {}
    missing_pdb = []

    for row in meta.itertuples(index=False):
        parent = str(row.parent)
        m = name_re.match(parent)
        pfile = pdb_map.get(m.group("base").upper()) if m else None
        if pfile is None:
            missing_pdb.append((parent, str(row.split)))
            continue
        pseq, coords = parse_ca(pfile)
        if len(coords) < 5:
            continue
        parsed[parent] = (pseq, str(row.wt_seq), coords)
        split_of[parent] = str(row.split)
        group_of[parent] = str(row.group_used)

    test_items = [(p, parsed[p][0], parsed[p][2]) for p, sp in split_of.items() if sp == "test"]
    ref_items = [(p, split_of[p], parsed[p][0], parsed[p][2])
                 for p, sp in split_of.items() if sp in ("train", "val")]
    print(f"[job1] Aligning {len(test_items)} test PDBs x {len(ref_items)} train/val PDBs "
          f"= {len(test_items) * len(ref_items):,} TM-align pairs ...", flush=True)

    tasks = [(tp, tseq, tcoords, ref_items) for tp, tseq, tcoords in test_items]
    all_rows = []
    workers = max(1, min(os.cpu_count() or 4, 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk in ex.map(_align_one_test, tasks):
            all_rows.extend(chunk)

    df = pd.DataFrame(all_rows)
    mat_path = WORK / "megascale_tm_leakage_matrix.parquet"
    df.to_parquet(mat_path, index=False)

    train_only = df[df["ref_split"] == "train"]
    per_test = (
        train_only.sort_values("tm_max", ascending=False)
        .groupby("test_parent", as_index=False)
        .first()
        .rename(columns={
            "ref_parent": "nearest_train_parent",
            "tm_max": "max_tm_vs_train",
            "tm_norm_test": "max_tm_norm_test",
            "rmsd": "nearest_rmsd",
        })
    )
    per_test["test_group"] = per_test["test_parent"].map(group_of)
    per_test["tm_bucket"] = pd.cut(
        per_test["max_tm_vs_train"],
        bins=[-0.01, 0.5, 0.7, 1.01],
        labels=["novel_fold(<0.5)", "shared_fold(0.5-0.7)", "near_identical(>=0.7)"],
    ).astype(str)
    max_path = WORK / "megascale_tm_test_max.parquet"
    per_test.to_parquet(max_path, index=False)

    n_test = len(per_test)
    summary = {
        "job": "job1_tm_leakage_audit",
        "elapsed_s": round(time.time() - t0, 2),
        "n_test_structured": n_test,
        "n_train_structured": sum(1 for _, sp, _, _ in ref_items if sp == "train"),
        "n_val_structured": sum(1 for _, sp, _, _ in ref_items if sp == "val"),
        "n_missing_pdb": len(missing_pdb),
        "total_alignments": int(len(df)),
        "test_max_tm_vs_train": {
            "median": round(float(per_test["max_tm_vs_train"].median()), 4),
            "mean": round(float(per_test["max_tm_vs_train"].mean()), 4),
            "p90": round(float(per_test["max_tm_vs_train"].quantile(0.90)), 4),
            "max": round(float(per_test["max_tm_vs_train"].max()), 4),
            "count_tm_ge_0.5": int((per_test["max_tm_vs_train"] >= 0.5).sum()),
            "frac_tm_ge_0.5": round(float((per_test["max_tm_vs_train"] >= 0.5).mean()), 4),
            "count_tm_ge_0.6": int((per_test["max_tm_vs_train"] >= 0.6).sum()),
            "count_tm_ge_0.7": int((per_test["max_tm_vs_train"] >= 0.7).sum()),
            "bucket_counts": per_test["tm_bucket"].value_counts().to_dict(),
        },
        "outputs": {
            mat_path.name: {"bytes": mat_path.stat().st_size, "sha256": sha256_file(mat_path)},
            max_path.name: {"bytes": max_path.stat().st_size, "sha256": sha256_file(max_path)},
        },
    }
    sum_path = WORK / "megascale_tm_leakage_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
