"""
remote/cpu_jobs/job2_mmseqs25_splits.py
=======================================
Kaggle CPU Worker 2 (Account: mdahbabhamidkhan) — MMseqs2 25% Homology Splits
& FireProtDB-HF (25%-Homology-Filtered External Holdout).

Resolves the 30% vs 25% sequence-identity split gap noted in BENCHMARK_PLAN.md
and PROJECT_STATUS.md §5:
  1. Installs the static MMseqs2 Linux AVX2 binary into /tmp.
  2. Builds field-standard 25%-identity (c=0.80) MegaScale train/val/test splits
     with 8-mer MinHash near-duplicate leakage verification.
  3. Runs MMseqs2 easy-search of all 176 FireProtDB holdout proteins against
     MegaScale (at 25% and 30% identity) to produce ``fireprotdb_hf25.parquet``
     (strictly homology-filtered external set matching ThermoMPNN FireProt-HF).

Outputs in /kaggle/working:
  * splits_megascale_id25.parquet
  * leakage_report_megascale_id25.json
  * fireprotdb_hf25.parquet
  * fireprotdb_homology_audit.parquet
  * job2_summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job2")
TMP = Path("/tmp/holo_job2") if Path("/tmp").is_dir() else WORK / "_tmp"
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


def install_mmseqs() -> str:
    mm_bin = TMP / "mmseqs" / "bin" / "mmseqs"
    if mm_bin.is_file():
        return str(mm_bin)
    url = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
    tgz = TMP / "mmseqs.tar.gz"
    print(f"[job2] Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(TMP)
    os.chmod(mm_bin, 0o755)
    out = subprocess.check_output([str(mm_bin), "version"], text=True).strip()
    print(f"[job2] MMseqs2 version: {out}", flush=True)
    return str(mm_bin)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mmseqs_cluster(mmseqs: str, seqs: list[str], min_seq_id: float = 0.25,
                   cov: float = 0.80) -> dict[str, str]:
    cdir = TMP / f"cluster_{int(min_seq_id * 100)}"
    cdir.mkdir(parents=True, exist_ok=True)
    fasta = cdir / "in.fasta"
    with open(fasta, "w", encoding="utf-8") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">S{i}\n{s}\n")
    pref = cdir / "res"
    tdir = cdir / "tmp"
    tdir.mkdir(exist_ok=True)
    subprocess.run(
        [mmseqs, "easy-cluster", str(fasta), str(pref), str(tdir),
         "--min-seq-id", str(min_seq_id), "-c", str(cov),
         "--cov-mode", "0", "-s", "7.5", "--threads", "4"],
        check=True,
    )
    seq_to_rep: dict[str, str] = {}
    with open(cdir / "res_cluster.tsv", encoding="utf-8") as fh:
        for line in fh:
            rep, mem = line.strip().split("\t")
            seq_to_rep[seqs[int(mem[1:])]] = rep
    return seq_to_rep


def mmseqs_search(mmseqs: str, queries: dict[str, str], targets: dict[str, str]) -> list[dict]:
    sdir = TMP / "search"
    sdir.mkdir(parents=True, exist_ok=True)
    q_fa, t_fa = sdir / "q.fasta", sdir / "t.fasta"
    with open(q_fa, "w", encoding="utf-8") as fh:
        for k, s in queries.items():
            fh.write(f">{k}\n{s}\n")
    with open(t_fa, "w", encoding="utf-8") as fh:
        for k, s in targets.items():
            fh.write(f">{k}\n{s}\n")
    out_tsv = sdir / "hits.tsv"
    tdir = sdir / "tmp"
    tdir.mkdir(exist_ok=True)
    subprocess.run(
        [mmseqs, "easy-search", str(q_fa), str(t_fa), str(out_tsv), str(tdir),
         "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
         "-s", "7.5", "-e", "10", "--min-seq-id", "0.15", "--threads", "4"],
        check=True,
    )
    rows = []
    if out_tsv.is_file():
        for line in out_tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q, t, fid, aln, qcov, tcov, ev, bits = line.split("\t")
            rows.append({
                "query": q, "target": t, "seq_identity": float(fid),
                "aln_len": int(aln), "qcov": float(qcov), "tcov": float(tcov),
                "evalue": float(ev), "bits": float(bits),
            })
    return rows


def main() -> None:
    t0 = time.time()
    import numpy as np
    import pandas as pd

    mmseqs = install_mmseqs()
    pairs_p = find_input("megascale_ddg_pairs.parquet")
    splits30_p = find_input("splits_megascale.parquet")
    fp_p = find_input("fireprotdb_holdout.parquet")
    if not pairs_p or not splits30_p or not fp_p:
        raise SystemExit(f"Missing inputs: pairs={pairs_p}, splits={splits30_p}, fp={fp_p}")

    # 1. MegaScale 25%-identity clustering & deterministic 80/10/10 cluster split
    splits30 = pd.read_parquet(splits30_p)
    pairs = pd.read_parquet(pairs_p, columns=["parent", "wt_seq"]).drop_duplicates()
    uniq_seqs = sorted(splits30["parent_seq"].unique())
    print(f"[job2] Clustering {len(uniq_seqs)} unique MegaScale parent sequences at 25% id ...", flush=True)
    seq_to_cluster = mmseqs_cluster(mmseqs, uniq_seqs, min_seq_id=0.25, cov=0.80)

    # Union with base PDB construct name so background mutants stay grouped
    import re
    name_re = re.compile(r"^(?P<base>[^.]+)\.pdb")
    seq_to_base = {}
    for row in pairs.itertuples(index=False):
        m = name_re.match(str(row.parent))
        seq_to_base[str(row.wt_seq)] = m.group("base").upper() if m else str(row.parent)

    # Union-find over (mmseqs_cluster, base_pdb_id)
    parent_uf: dict[str, str] = {}
    def find(x):
        parent_uf.setdefault(x, x)
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_uf[ra] = rb

    for s in uniq_seqs:
        c = "C:" + seq_to_cluster[s]
        b = "B:" + seq_to_base.get(s, s)
        union(c, b)
    seq_to_group25 = {s: find("C:" + seq_to_cluster[s]) for s in uniq_seqs}

    # Assign groups greedily/deterministically to 80/10/10 by row count
    df25 = splits30.copy()
    df25["group_used"] = df25["parent_seq"].map(seq_to_group25)
    g_counts = df25.groupby("group_used").size().to_dict()
    groups_sorted = sorted(g_counts.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(groups_sorted)
    total = len(df25)
    g_split = {}
    n_test = n_val = 0
    for g in groups_sorted:
        cnt = g_counts[g]
        if n_test < 0.10 * total:
            g_split[g] = "test"
            n_test += cnt
        elif n_val < 0.10 * total:
            g_split[g] = "val"
            n_val += cnt
        else:
            g_split[g] = "train"
    df25["split"] = df25["group_used"].map(g_split)

    # Verify zero group leakage and 8-mer overlap audit
    train_g = set(df25[df25["split"] == "train"]["group_used"])
    val_g = set(df25[df25["split"] == "val"]["group_used"])
    test_g = set(df25[df25["split"] == "test"]["group_used"])
    assert not (train_g & val_g) and not (train_g & test_g) and not (val_g & test_g)

    def kmers(s, k=8):
        return {s[i:i + k] for i in range(max(0, len(s) - k + 1))}

    train_seqs = sorted(df25[df25["split"] == "train"]["parent_seq"].unique())
    held_seqs = sorted(df25[df25["split"] != "train"]["parent_seq"].unique())
    train_k = [kmers(s) for s in train_seqs]
    near_dups = 0
    for hs in held_seqs:
        hk = kmers(hs)
        if not hk:
            continue
        best_j = max(len(hk & tk) / max(len(hk | tk), 1) for tk in train_k)
        if best_j >= 0.30:
            near_dups += 1

    out_s25 = WORK / "splits_megascale_id25.parquet"
    df25.to_parquet(out_s25, index=False)

    leak_rep = {
        "method": "mmseqs_id25_union_base_pdb",
        "identity": 0.25,
        "coverage": 0.80,
        "seed": 42,
        "n_rows": {sp: int((df25["split"] == sp).sum()) for sp in ("train", "val", "test")},
        "n_proteins": {sp: int(df25[df25["split"] == sp]["parent_seq"].nunique()) for sp in ("train", "val", "test")},
        "n_clusters": {sp: int(df25[df25["split"] == sp]["group_used"].nunique()) for sp in ("train", "val", "test")},
        "group_leakage": False,
        "heldout_8mer_jaccard_ge_0.30": near_dups,
    }
    out_l25 = WORK / "leakage_report_megascale_id25.json"
    out_l25.write_text(json.dumps(leak_rep, indent=2), encoding="utf-8")

    # 2. FireProtDB homology audit against MegaScale (25% and 30% identity)
    fp = pd.read_parquet(fp_p)
    fp_seq_col = "wt_seq" if "wt_seq" in fp.columns else "parent_seq"
    fp_par_col = "parent" if "parent" in fp.columns else "uniprot_id"
    fp_uniq = fp[[fp_par_col, fp_seq_col]].drop_duplicates()
    q_map = {str(r[0]): str(r[1]).upper() for r in fp_uniq.itertuples(index=False)}
    t_map = {f"MS_{i}": s for i, s in enumerate(uniq_seqs)}

    hits = mmseqs_search(mmseqs, q_map, t_map)
    hits_df = pd.DataFrame(hits) if hits else pd.DataFrame(
        columns=["query", "target", "seq_identity", "aln_len", "qcov", "tcov", "evalue", "bits"]
    )
    out_hits = WORK / "fireprotdb_homology_audit.parquet"
    hits_df.to_parquet(out_hits, index=False)

    max_id_by_q = hits_df.groupby("query")["seq_identity"].max().to_dict() if len(hits_df) else {}
    fp["max_megascale_seq_id"] = fp[fp_par_col].astype(str).map(lambda x: float(max_id_by_q.get(x, 0.0)))
    fp_hf25 = fp[fp["max_megascale_seq_id"] < 0.25].reset_index(drop=True)
    out_hf25 = WORK / "fireprotdb_hf25.parquet"
    fp_hf25.to_parquet(out_hf25, index=False)

    summary = {
        "job": "job2_mmseqs25_splits",
        "elapsed_s": round(time.time() - t0, 2),
        "megascale_id25": leak_rep,
        "fireprotdb_total_rows": int(len(fp)),
        "fireprotdb_total_proteins": int(fp[fp_par_col].nunique()),
        "fireprotdb_hf25_rows": int(len(fp_hf25)),
        "fireprotdb_hf25_proteins": int(fp_hf25[fp_par_col].nunique()),
        "fireprotdb_proteins_ge_25pct": int(sum(1 for v in max_id_by_q.values() if v >= 0.25)),
        "fireprotdb_proteins_ge_30pct": int(sum(1 for v in max_id_by_q.values() if v >= 0.30)),
        "outputs": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in (out_s25, out_l25, out_hits, out_hf25)
        },
    }
    (WORK / "job2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
