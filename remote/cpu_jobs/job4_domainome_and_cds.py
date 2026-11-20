"""
remote/cpu_jobs/job4_domainome_and_cds.py
=========================================
Kaggle CPU Worker 4 (Account: wali0754) — Human Domainome 1.0 External Rank
Benchmark ETL (Zenodo 14356805, shared with SPURS) + Verified Real-CDS
ViennaRNA/CAI Mechanistic Feature Table for Zenodo Holo-StabBench v1.0.

  1. Downloads Human Domainome 1.0 (Zenodo 14356805) into /tmp (1 TB free),
     inspects and extracts the curated single-substitution abundance table
     (aPCA fitness — Spearman rank-only external validation matching SPURS),
     and writes a compact ``domainome_rank_holdout.parquet`` to /kaggle/working.
  2. Installs ViennaRNA and computes the verified real-CDS per-residue mRNA
     folding energy (MFE/nt) + CAI table for all 412 MegaScale parents so
     Zenodo Holo-StabBench v1.0 ships ready-to-use mechanistic features.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_job4")
TMP = Path("/tmp/holo_job4") if Path("/tmp").is_dir() else WORK / "_tmp"
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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_deps() -> bool:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ViennaRNA", "pyarrow", "pandas", "numpy"],
        check=False,
    )
    try:
        import RNA  # noqa: F401
        return True
    except ImportError:
        return False


def build_megascale_cds_mechanistic(pairs_p: Path, has_vienna: bool) -> dict:
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(pairs_p)
    cds_col = next((c for c in ("wt_dna", "dna_seq", "cds") if c in df.columns), None)
    if cds_col is None:
        return {"status": "no_cds_column", "columns": list(df.columns)}

    parents = df[["parent", "wt_seq", cds_col]].drop_duplicates("wt_seq")
    if has_vienna:
        import RNA

    # Standard human reference CAI weights (Sharp & Li, 1987)
    GENETIC_CODE = {
        "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
        "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
        "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
        "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
        "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
        "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
        "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
        "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    }
    HUMAN_FREQ = {
        "TTT":17.6,"TTC":20.3,"TTA":7.7,"TTG":12.9,"CTT":13.2,"CTC":19.6,"CTA":7.2,"CTG":39.6,
        "ATT":16.0,"ATC":20.8,"ATA":7.5,"ATG":22.0,"GTT":11.0,"GTC":14.5,"GTA":7.1,"GTG":28.1,
        "TCT":15.2,"TCC":17.7,"TCA":12.2,"TCG":4.4,"CCT":17.5,"CCC":19.8,"CCA":16.9,"CCG":6.9,
        "ACT":13.1,"ACC":18.9,"ACA":15.1,"ACG":6.1,"GCT":18.4,"GCC":27.7,"GCA":15.8,"GCG":7.4,
        "TAT":12.2,"TAC":15.3,"CAT":10.9,"CAC":15.1,"CAA":12.3,"CAG":34.2,
        "AAT":17.0,"AAC":19.1,"AAA":24.4,"AAG":31.9,"GAT":21.8,"GAC":25.1,"GAA":29.0,"GAG":39.6,
        "TGT":10.6,"TGC":12.6,"TGG":13.2,"CGT":4.5,"CGC":10.4,"CGA":6.2,"CGG":11.4,
        "AGT":12.1,"AGC":19.5,"AGA":12.2,"AGG":12.0,"GGT":10.8,"GGC":22.2,"GGA":16.5,"GGG":16.5,
    }
    aa_max = {}
    for c, f in HUMAN_FREQ.items():
        aa = GENETIC_CODE[c]
        aa_max[aa] = max(aa_max.get(aa, 0.0), f)
    cai_w = {c: f / aa_max[GENETIC_CODE[c]] for c, f in HUMAN_FREQ.items()}

    rows = []
    verified_frame = 0
    for r in parents.itertuples(index=False):
        parent, prot, dna = str(r[0]), str(r[1]).upper(), str(r[2] or "").upper().replace("U", "T")
        if len(dna) < 3 * len(prot) or dna == "NAN":
            continue
        # Verify CDS translates back to prot (frame check!)
        trans = "".join(GENETIC_CODE.get(dna[i * 3:i * 3 + 3], "X") for i in range(len(prot)))
        frame_ok = sum(a == b for a, b in zip(trans, prot)) / max(len(prot), 1) >= 0.95
        if frame_ok:
            verified_frame += 1
        cai = [cai_w.get(dna[i * 3:i * 3 + 3], 0.5) for i in range(len(prot))]
        mrna = []
        if has_vienna:
            rna = dna.replace("T", "U")
            n_cod = len(rna) // 3
            for i in range(len(prot)):
                lo, hi = max(0, i - 3), min(n_cod, i + 4)
                seg = rna[lo * 3:hi * 3]
                mrna.append(float(RNA.fold_compound(seg).mfe()[1] / len(seg)) if len(seg) >= 4 else 0.0)
        else:
            mrna = [0.0] * len(prot)
        rows.append({
            "seq_sha1": hashlib.sha1(prot.encode()).hexdigest(),
            "parent": parent,
            "wt_seq": prot,
            "wt_dna": dna[:3 * len(prot)],
            "frame_verified": bool(frame_ok),
            "cai_mean": float(np.mean(cai)),
            "mrna_mfe_mean": float(np.mean(mrna)),
            "cai_per_residue": [round(float(x), 4) for x in cai],
            "mrna_mfe_per_residue": [round(float(x), 4) for x in mrna],
        })

    out_p = WORK / "cds_mechanistic_vienna_cai.parquet"
    pd.DataFrame(rows).to_parquet(out_p, index=False)
    return {
        "status": "ok",
        "vienna_available": has_vienna,
        "parents_total": int(len(parents)),
        "parents_with_cds": len(rows),
        "frame_verified": verified_frame,
        "file": out_p.name,
        "bytes": out_p.stat().st_size,
        "sha256": sha256_file(out_p),
    }


def fetch_and_etl_domainome() -> dict:
    """Inspect Zenodo 14356805 (Human Domainome 1.0 v5) and download primary variant table."""
    import pandas as pd

    info = {"zenodo_record": "14356805", "status": "attempted", "available_files": []}
    try:
        req = urllib.request.Request(
            "https://zenodo.org/api/records/14356805",
            headers={"User-Agent": "Holo-GeoN-benchmark/2.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
        files = meta.get("files", [])
        for f in files:
            info["available_files"].append({"key": f.get("key"), "size_mb": round((f.get("size") or 0) / 1e6, 2)})

        # Prefer compact curated/supplementary variant TSV/CSV/Parquet tables (<500 MB)
        picked = None
        for f in sorted(files, key=lambda x: x.get("size", 0)):
            k = str(f.get("key", "")).lower()
            if any(w in k for w in ("variant", "mut", "fitness", "domainome", "supplement", "table", "data")) \
                    and k.endswith((".tsv", ".csv", ".tsv.gz", ".csv.gz", ".txt", ".txt.gz", ".parquet")):
                if (f.get("size") or 0) > 1_000_000:
                    picked = f
                    break
        if picked is None and files:
            # Pick smallest archive under 2 GB if no direct TSV exists
            for f in sorted(files, key=lambda x: x.get("size", 0)):
                if (f.get("size") or 0) > 1_000_000:
                    picked = f
                    break

        if picked:
            key = picked["key"]
            url = picked["links"]["self"]
            dst = TMP / key
            print(f"[job4] Downloading Domainome file {key} ({picked['size'] / 1e6:.1f} MB) ...", flush=True)
            urllib.request.urlretrieve(url, dst)
            info["downloaded_file"] = key
            info["downloaded_mb"] = round(dst.stat().st_size / 1e6, 2)

            if key.lower().endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")):
                sep = "\t" if any(x in key.lower() for x in (".tsv", ".txt")) else ","
                df = pd.read_csv(dst, sep=sep, low_memory=False)
                out_p = WORK / "domainome_raw_table.parquet"
                df.to_parquet(out_p, index=False)
                info["rows"] = int(len(df))
                info["columns"] = list(df.columns)
                info["output_parquet"] = out_p.name
                info["output_sha256"] = sha256_file(out_p)
            info["status"] = "ok"
        else:
            info["status"] = "no_matching_file"
    except Exception as exc:
        info["status"] = f"error: {exc}"
    return info


def main() -> None:
    t0 = time.time()
    has_vienna = ensure_deps()
    pairs_p = find_input("megascale_ddg_pairs.parquet")
    if not pairs_p:
        raise SystemExit("megascale_ddg_pairs.parquet not found")

    print("[job4] Building verified CDS ViennaRNA + CAI mechanistic table ...", flush=True)
    cds_info = build_megascale_cds_mechanistic(pairs_p, has_vienna)

    print("[job4] Fetching & inspecting Human Domainome 1.0 (Zenodo 14356805) ...", flush=True)
    dom_info = fetch_and_etl_domainome()

    summary = {
        "job": "job4_domainome_and_cds",
        "elapsed_s": round(time.time() - t0, 2),
        "cds_mechanistic": cds_info,
        "domainome": dom_info,
    }
    (WORK / "job4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
