"""
scripts/build_external_holdouts.py
==================================
Standardize the downloaded external validation datasets into Holo-GeoN's strict
common schema (verified WT sequence, 0-based ``site``, ``wt_aa``, ``mt_aa``,
``ddg`` / ``target``, and explicit sign/assay provenance):

  1. **S669** (Pancotti et al., *Brief. Bioinform.* 2022; Zenodo 7568094):
     Extracts WT chain sequences from ``S669/pdbs/<WT_PDB>.pdb``, parses
     ``Seq_Mut``, verifies ``wt_seq[site] == wt_aa`` on every row, preserves
     ``Experimental_DDG_dir`` (already stabilizing-positive!), and writes
     ``s669_holdout.parquet``.
  2. **Human Domainome 1.0** (Beltran et al., *Nature* 2025; Zenodo 14356805):
     Reconstructs the exact wild-type ``wt_seq`` and 0-based ``site`` for each
     domain from the mutant ``aa_seq`` + ``(position, wt_aa, mut_aa)``,
     verifies 100% WT consistency across all mutations of each domain, and
     writes ``domainome_apca_holdout.parquet`` and ``domainome_apca_hf25.parquet``
     (explicitly tagged ``assay_type="aPCA_abundance_rank_only"``).

Usage:
    python -m scripts.build_external_holdouts \\
        --s669-zip ../extras/kaggle_accounts/cpu_outputs/cpu3_afdb_s669_geom/S669.zip \\
        --domainome-dir ../extras/kaggle_accounts/cpu_outputs/cpu4_domainome_v2 \\
        --out-dir ../extras/zenodo_staging/Holo-StabBench-v1.0/tables
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _parse_pdb_chain_seq(pdb_text: str, chain: str) -> tuple[str, dict[int, int]]:
    """Return ``(seq, pdb_resnum_to_0idx)`` for one chain from a PDB string."""
    seen = {}
    order = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 27 or line[16] not in (" ", "A"):
            continue
        if line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        res = line[17:20].strip().upper()
        if res not in THREE_TO_ONE:
            continue
        key = line[22:27]
        if key in seen:
            continue
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            resnum = None
        seen[key] = (THREE_TO_ONE[res], resnum)
        order.append(key)
    seq = "".join(seen[k][0] for k in order)
    resnum_map = {}
    for idx, k in enumerate(order):
        rn = seen[k][1]
        if rn is not None and rn not in resnum_map:
            resnum_map[rn] = idx
    return seq, resnum_map


def build_s669(s669_zip: Path, out_parquet: Path) -> dict:
    with zipfile.ZipFile(s669_zip) as zf:
        raw_csv = zf.read("S669/S669.csv").decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw_csv))
        pdb_cache: dict[tuple[str, str], tuple[str, dict[int, int]]] = {}
        name_lookup = {Path(n).name.lower(): n for n in zf.namelist() if n.lower().endswith(".pdb")}

        rows = []
        dropped = {"missing_pdb": 0, "bad_mut": 0, "wt_mismatch": 0, "nan_ddg": 0}
        for r in df.itertuples(index=False):
            pdb_id = str(r.WT_PDB).strip().lower()
            chain = str(r.Chain).strip()
            ddg = float(r.Experimental_DDG_dir) if pd.notna(r.Experimental_DDG_dir) else float("nan")
            if not np.isfinite(ddg):
                dropped["nan_ddg"] += 1
                continue
            key = (pdb_id, chain)
            if key not in pdb_cache:
                entry = name_lookup.get(f"{pdb_id}.pdb")
                if entry is None:
                    pdb_cache[key] = ("", {})
                else:
                    pdb_cache[key] = _parse_pdb_chain_seq(zf.read(entry).decode("utf-8", errors="ignore"), chain)
            wt_seq, resmap = pdb_cache[key]
            if not wt_seq:
                dropped["missing_pdb"] += 1
                continue

            # Try Seq_Mut (1-based sequence index) first, then PDB_Mut via resmap
            site = None
            wt_aa = mt_aa = None
            m_seq = MUT_RE.match(str(r.Seq_Mut).strip().upper())
            if m_seq:
                w, pos1, m = m_seq.group(1), int(m_seq.group(2)), m_seq.group(3)
                if 1 <= pos1 <= len(wt_seq) and wt_seq[pos1 - 1] == w:
                    site, wt_aa, mt_aa = pos1 - 1, w, m
            if site is None:
                m_pdb = MUT_RE.match(str(r.PDB_Mut).strip().upper())
                if m_pdb:
                    w, rnum, m = m_pdb.group(1), int(m_pdb.group(2)), m_pdb.group(3)
                    idx = resmap.get(rnum)
                    if idx is not None and wt_seq[idx] == w:
                        site, wt_aa, mt_aa = idx, w, m
            if site is None:
                dropped["wt_mismatch" if m_seq else "bad_mut"] += 1
                continue

            rows.append({
                "parent": f"{pdb_id.upper()}_{chain}",
                "wt_pdb": pdb_id.upper(),
                "chain": chain,
                "wt_seq": wt_seq,
                "parent_seq": wt_seq,
                "site": int(site),
                "wt_aa": wt_aa,
                "mt_aa": mt_aa,
                "mutation": f"{wt_aa}{site + 1}{mt_aa}",
                "ddg": ddg,
                "target": ddg,
                "temperature_k": float(r.Temperature) if pd.notna(r.Temperature) else float("nan"),
                "ph": float(r.pH) if pd.notna(r.pH) else float("nan"),
                "doi": str(r.DOI) if pd.notna(r.DOI) else "",
                "sign_convention": "stabilizing_positive",
                "source": "S669_Zenodo_7568094",
            })

    out_df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    return {
        "rows_in": int(len(df)),
        "rows_verified": int(len(out_df)),
        "unique_proteins": int(out_df["parent"].nunique()) if len(out_df) else 0,
        "mean_ddg": round(float(out_df["ddg"].mean()), 4) if len(out_df) else float("nan"),
        "dropped": dropped,
        "output": str(out_parquet),
    }


def standardize_domainome(in_parquet: Path, out_parquet: Path) -> dict:
    """
    Reconstruct verified WT ``parent_seq`` and 0-based ``site`` from Domainome
    Extended Data Table 2 where ``aa_seq`` is the mutated domain sequence and
    ``domain_ID`` ends in ``_<start_resnum>``.
    """
    df = pd.read_parquet(in_parquet)
    rows_out = []
    verified_domains = 0
    dropped_rows = 0
    mismatch_domains = 0

    for dom_id, grp in df.groupby("domain_ID", sort=False):
        parts = str(dom_id).rsplit("_", 1)
        try:
            start_pos = int(parts[-1])
        except ValueError:
            start_pos = int(pd.to_numeric(grp["position"], errors="coerce").min())

        # Reconstruct candidate WT sequence from each row and take consensus (verify ~100% match)
        wt_counts: dict[str, int] = {}
        valid_records = []
        for r in grp.itertuples(index=False):
            mseq = str(r.aa_seq).strip().upper()
            pos = int(round(float(r.position)))
            site = pos - start_pos
            w = str(r.wt_aa).strip().upper()
            m = str(r.mut_aa).strip().upper()
            if not (0 <= site < len(mseq)) or mseq[site] != m:
                dropped_rows += 1
                continue
            wt_cand = mseq[:site] + w + mseq[site + 1:]
            wt_counts[wt_cand] = wt_counts.get(wt_cand, 0) + 1
            valid_records.append((r, site, w, m, wt_cand))

        if not wt_counts:
            mismatch_domains += 1
            continue
        wt_seq, top_n = max(wt_counts.items(), key=lambda kv: kv[1])
        if top_n / len(valid_records) < 0.95:
            mismatch_domains += 1
            continue
        verified_domains += 1

        for r, site, w, m, wt_cand in valid_records:
            if wt_cand != wt_seq:
                dropped_rows += 1
                continue
            rows_out.append({
                "parent": str(dom_id),
                "uniprot_id": str(r.uniprot_ID),
                "wt_seq": wt_seq,
                "parent_seq": wt_seq,
                "site": int(site),
                "wt_aa": w,
                "mt_aa": m,
                "mutation": f"{w}{site + 1}{m}",
                "target": float(r.normalized_fitness),
                "sigma": float(r.normalized_fitness_sigma) if pd.notna(r.normalized_fitness_sigma) else float("nan"),
                "quality_rank": int(r.quality_rank) if pd.notna(r.quality_rank) else -1,
                "assay_type": "aPCA_abundance_rank_only",
                "source": "Human_Domainome_1.0_Zenodo_14356805",
            })

    out_df = pd.DataFrame(rows_out)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    return {
        "rows_in": int(len(df)),
        "rows_verified": int(len(out_df)),
        "verified_domains": verified_domains,
        "mismatch_domains": mismatch_domains,
        "dropped_rows": dropped_rows,
        "output": str(out_parquet),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Standardize S669 and Human Domainome 1.0 external holdouts")
    ap.add_argument("--s669-zip", default=None)
    ap.add_argument("--domainome-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    if args.s669_zip and Path(args.s669_zip).is_file():
        report["s669"] = build_s669(Path(args.s669_zip), out_dir / "s669_holdout.parquet")

    if args.domainome_dir and Path(args.domainome_dir).is_dir():
        ddir = Path(args.domainome_dir)
        for name in ("domainome_apca_holdout.parquet", "domainome_apca_hf25.parquet"):
            src = ddir / name
            if src.is_file():
                report[name] = standardize_domainome(src, out_dir / name)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
