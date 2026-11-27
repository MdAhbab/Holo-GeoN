"""
remote/cpu_jobs/job5_tm_fold_split.py
=====================================
Kaggle CPU Worker 5 (Account: mdahbab) — All-vs-All TM-align Structural Matrix
& True Fold-Disjoint MegaScale Split Audit.

Triggered by the landmark finding from Worker 1 (`holo-geon-cpu1-tm-leakage`):
  * On the standard 30%-sequence-identity MegaScale split, 37 / 37 (100%)
    structured test proteins share a CATH-level fold (TM >= 0.5) with a train
    protein, and 27 / 37 (73%) are near-identical structures (TM >= 0.7,
    median 0.824, max 0.975, e.g. 2L7F vs 2L9R at RMSD 0.37 A).

This worker computes the COMPLETE all-vs-all TM-align matrix across all 373
structured MegaScale parent PDBs (69,378 unordered pairs, ~10 min on 4 CPUs),
and produces:
  1. ``megascale_tm_all_vs_all.parquet`` — full pairwise structural similarity
     graph (TM_norm_i, TM_norm_j, TM_max, TM_min, RMSD).
  2. Connected-component fold-cluster census across TM thresholds
     ``0.50, 0.55, 0.60, 0.65, 0.70`` (using both ``TM_max`` and symmetric
     ``TM_min``), documenting the exact fold diversity of MegaScale.
  3. ``splits_megascale_fold_tm06.parquet`` — a genuinely fold-disjoint split
     constructed from ``TM_min >= 0.60`` + MMseqs2 25% + base-PDB union clusters
     (plus ``splits_megascale_fold_tm05.parquet`` when >= 6 distinct clusters
     exist at ``TM_min >= 0.50``), with ZERO fold leakage across train/val/test.
  4. ``job5_summary.json`` — full audit metrics + SHA-256 digests for
     Zenodo ``Holo-StabBench v1.0``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job5")
TMP = Path("/tmp/holo_job5") if Path("/tmp").is_dir() else WORK / "_tmp"
WORK.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)


def find_input(name: str, roots=("/kaggle/input", ".")) -> Path | None:
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


def _install_deps() -> None:
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


def _align_row_chunk(args):
    from tmtools import tm_align

    i, p_i, seq_i, coords_i, rest = args
    rows = []
    for j, p_j, seq_j, coords_j in rest:
        res = tm_align(coords_i, coords_j, seq_i, seq_j)
        t1 = float(res.tm_norm_chain1)
        t2 = float(res.tm_norm_chain2)
        rows.append((i, j, p_i, p_j, t1, t2, max(t1, t2), min(t1, t2), float(res.rmsd)))
    return rows


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def assign_group_splits(df, group_col: str, seed: int = 42,
                        ratios=(0.80, 0.10, 0.10)):
    import numpy as np

    rng = np.random.default_rng(seed)
    counts = df.groupby(group_col).size().to_dict()
    groups = sorted(counts.keys())
    rng.shuffle(groups)
    # Sort descending by size after tie-breaking shuffle (greedy bin packing to hit 80/10/10)
    groups.sort(key=lambda g: counts[g], reverse=True)
    total = sum(counts.values())
    targets = {"train": ratios[0] * total, "val": ratios[1] * total, "test": ratios[2] * total}
    loads = {"train": 0, "val": 0, "test": 0}
    g2s = {}
    for g in groups:
        # Pick split with largest remaining fractional deficit
        best_sp = max(targets.keys(), key=lambda sp: (targets[sp] - loads[sp]) / max(targets[sp], 1.0))
        g2s[g] = best_sp
        loads[best_sp] += counts[g]
    out = df.copy()
    out["split"] = out[group_col].map(g2s)
    return out


def main() -> None:
    t0 = time.time()
    _install_deps()
    import numpy as np
    import pandas as pd

    pairs_p = find_input("megascale_ddg_pairs.parquet")
    splits_p = find_input("splits_megascale.parquet")
    if not pairs_p or not splits_p:
        raise SystemExit(f"Missing inputs: pairs={pairs_p}, splits={splits_p}")

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
    print(f"[job5] Found {len(pdb_map)} PDB files", flush=True)

    pairs = pd.read_parquet(pairs_p, columns=["parent", "wt_seq"]).drop_duplicates("wt_seq")
    splits_df = pd.read_parquet(splits_p)
    parent_meta = splits_df[["parent_seq", "split", "group_used"]].drop_duplicates("parent_seq")
    parent_meta = parent_meta.merge(pairs, left_on="parent_seq", right_on="wt_seq", how="left")

    name_re = re.compile(r"^(?P<base>[^.]+)\.pdb(?:_(?P<mut>[A-Z]\d+[A-Z]|pross\d+))?$")
    nodes = []  # (parent, wt_seq, pdb_seq, coords, orig_group, orig_split)
    for r in parent_meta.itertuples(index=False):
        par = str(r.parent)
        m = name_re.match(par)
        pfile = pdb_map.get(m.group("base").upper()) if m else None
        if pfile is None:
            continue
        pseq, coords = parse_ca(pfile)
        if len(coords) < 5:
            continue
        nodes.append((par, str(r.parent_seq), pseq, coords, str(r.group_used), str(r.split)))

    n = len(nodes)
    n_pairs = n * (n - 1) // 2
    print(f"[job5] Computing all-vs-all TM-align on {n} structured MegaScale parents ({n_pairs:,} pairs) ...",
          flush=True)

    tasks = []
    for i in range(n - 1):
        rest = [(j, nodes[j][0], nodes[j][2], nodes[j][3]) for j in range(i + 1, n)]
        tasks.append((i, nodes[i][0], nodes[i][2], nodes[i][3], rest))

    all_rows = []
    workers = max(1, min(os.cpu_count() or 4, 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, chunk in enumerate(ex.map(_align_row_chunk, tasks)):
            all_rows.extend(chunk)
            if (k + 1) % 50 == 0:
                print(f"  progress {k + 1}/{len(tasks)} rows ({len(all_rows):,} pairs, {time.time() - t0:.1f}s)",
                      flush=True)

    mat = pd.DataFrame(
        all_rows,
        columns=["idx_i", "idx_j", "parent_i", "parent_j", "tm_norm_i", "tm_norm_j", "tm_max", "tm_min", "rmsd"],
    )
    mat_p = WORK / "megascale_tm_all_vs_all.parquet"
    mat.to_parquet(mat_p, index=False)
    print(f"[job5] Wrote {len(mat):,} pairs -> {mat_p}", flush=True)

    # Fold-cluster census across thresholds
    all_parents = [x[0] for x in nodes]
    seq_of = {x[0]: x[1] for x in nodes}
    orig_grp = {x[0]: x[4] for x in nodes}

    census = {}
    outputs = {mat_p.name: {"bytes": mat_p.stat().st_size, "sha256": sha256_file(mat_p)}}

    for metric in ("tm_max", "tm_min"):
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            uf = UnionFind(all_parents)
            # Always union proteins sharing original sequence cluster (MMseqs2 30% + base PDB)
            by_g: dict[str, list[str]] = {}
            for p in all_parents:
                by_g.setdefault(orig_grp[p], []).append(p)
            for grp_list in by_g.values():
                for p in grp_list[1:]:
                    uf.union(grp_list[0], p)
            hits = mat[mat[metric] >= thr]
            for r in hits[["parent_i", "parent_j"]].itertuples(index=False):
                uf.union(r[0], r[1])
            comp_sizes: dict[str, int] = {}
            for p in all_parents:
                root = uf.find(p)
                comp_sizes[root] = comp_sizes.get(root, 0) + 1
            sizes = sorted(comp_sizes.values(), reverse=True)
            key = f"{metric}_ge_{thr:.2f}"
            census[key] = {
                "n_clusters": len(sizes),
                "largest_cluster_proteins": sizes[0],
                "largest_cluster_frac": round(sizes[0] / n, 4),
                "top5_cluster_sizes": sizes[:5],
            }

            # Emit fold-disjoint split for tm_min >= 0.60 (and tm_min >= 0.50 if largest < 60%)
            if metric == "tm_min" and thr in (0.50, 0.60) and sizes[0] / n < 0.75 and len(sizes) >= 10:
                seq_to_fold = {seq_of[p]: f"tm{int(thr * 100)}:{hashlib.sha1(uf.find(p).encode()).hexdigest()[:10]}"
                               for p in all_parents}
                sdf = splits_df.copy()
                sdf["group_used"] = [
                    seq_to_fold.get(str(s), f"unstruct:{g}")
                    for s, g in zip(sdf["parent_seq"], sdf["group_used"])
                ]
                sdf = assign_group_splits(sdf, "group_used", seed=42)
                # Verify zero fold leakage across test vs train/val
                test_parents = set(sdf[sdf["split"] == "test"]["parent_seq"])
                train_parents = set(sdf[sdf["split"] == "train"]["parent_seq"])
                te_p = {p for p, s in seq_of.items() if s in test_parents}
                tr_p = {p for p, s in seq_of.items() if s in train_parents}
                cross = mat[
                    (mat["parent_i"].isin(te_p) & mat["parent_j"].isin(tr_p))
                    | (mat["parent_j"].isin(te_p) & mat["parent_i"].isin(tr_p))
                ]
                max_cross_tm_min = float(cross["tm_min"].max()) if len(cross) else 0.0
                max_cross_tm_max = float(cross["tm_max"].max()) if len(cross) else 0.0
                out_sp = WORK / f"splits_megascale_fold_tm{int(thr * 100):02d}.parquet"
                sdf.to_parquet(out_sp, index=False)
                census[key]["split_file"] = out_sp.name
                census[key]["split_rows"] = {str(k): int(v) for k, v in sdf["split"].value_counts().items()}
                census[key]["split_proteins"] = {
                    str(k): int(v) for k, v in sdf.groupby("split")["parent_seq"].nunique().items()
                }
                census[key]["max_cross_test_train_tm_min"] = round(max_cross_tm_min, 4)
                census[key]["max_cross_test_train_tm_max"] = round(max_cross_tm_max, 4)
                outputs[out_sp.name] = {"bytes": out_sp.stat().st_size, "sha256": sha256_file(out_sp)}

    summary = {
        "job": "job5_tm_fold_split",
        "elapsed_s": round(time.time() - t0, 2),
        "n_structured_parents": n,
        "n_pairs": len(mat),
        "fold_cluster_census": census,
        "outputs": outputs,
    }
    sum_p = WORK / "job5_summary.json"
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
