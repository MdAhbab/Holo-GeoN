"""
scripts/etl_cds.py
==================
Attach the **real** coding sequence to a mutation table.

Why this exists
---------------
The mechanistic stream (CAI + local mRNA folding) needs a nucleotide sequence.
The processed MegaScale parquet shipped without one, so v1 ran with
``--back-translate``: a synthetic transcript sampled from a codon-usage table
and seeded from the protein. That makes CAI and folding energy deterministic
functions of the amino-acid sequence, so the mechanistic edges carried no
information the language-model embeddings did not already have. The measured
ablation agrees — single-track r = 0.5309 *beat* dual-track r = 0.5268.

Back-translation is gone. The mechanistic stream is only meaningful against a
real CDS, and the Tsuboyama source does carry one (a ``dna_seq`` column). This
script recovers it and joins it onto the processed table, keyed by the protein
sequence it translates to.

Rows with no recoverable CDS keep ``mech_valid = 0``: their mechanistic
channels stay zero and their E_mech is empty. That is an honest gap in the
data, not a mode of the model.

Example
-------
python -m scripts.etl_cds \
    --table data/megascale_ddg_pairs.parquet \
    --source data/mega_scale_cdna/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv \
    --out data/megascale_ddg_pairs_cds.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.codon_tables import GENETIC_CODE
from src.config import load_config
from src.data import detect_columns

_DNA_COLUMNS = ["dna_seq", "wt_dna", "cds", "coding_sequence", "nt_seq", "dna"]
_PROTEIN_COLUMNS = ["aa_seq", "wt_seq", "protein_seq", "sequence", "wildtype_seq"]


def translate(dna: str) -> str:
    """Translate a coding sequence to protein, stopping at the first stop codon."""
    dna = dna.upper().replace("U", "T")
    out: list[str] = []
    for i in range(len(dna) // 3):
        aa = GENETIC_CODE.get(dna[i * 3: i * 3 + 3], "X")
        if aa == "*":
            break
        out.append(aa)
    return "".join(out)


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    return next((lower[n] for n in names if n in lower), None)


def build_cds_map(source_paths: list[str]) -> dict[str, str]:
    """
    Map protein sequence -> coding sequence, from one or more source tables.

    A source is usable if it has a nucleotide column. The protein key is taken
    from an explicit protein column when present, and otherwise by translating
    the CDS — which also validates that the two agree when both exist.
    """
    cds_map: dict[str, str] = {}
    mismatches = 0
    for path in source_paths:
        if not os.path.exists(path):
            print(f"[etl] source not found, skipping: {path}")
            continue
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
        dna_col = _first_present(df, _DNA_COLUMNS)
        if dna_col is None:
            print(f"[etl] {path}: no nucleotide column "
                  f"(looked for {_DNA_COLUMNS}); skipping")
            continue
        prot_col = _first_present(df, _PROTEIN_COLUMNS)
        print(f"[etl] {path}: dna column {dna_col!r}, protein column {prot_col!r}, "
              f"{len(df)} rows")

        for i in range(len(df)):
            dna = str(df[dna_col].iat[i])
            if not dna or dna.lower() == "nan":
                continue
            translated = translate(dna)
            if not translated:
                continue
            if prot_col is not None:
                stated = str(df[prot_col].iat[i]).upper()
                if stated and stated.lower() != "nan" and stated != translated:
                    mismatches += 1
                    # Trust the translation: it is what the nucleotides encode,
                    # and a mismatch usually means the protein column is a
                    # trimmed construct rather than the full translated ORF.
            cds_map.setdefault(translated, dna)
    if mismatches:
        print(f"[etl] note: {mismatches} rows where the stated protein differed "
              f"from the translated CDS; keyed on the translation")
    return cds_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="processed mutation parquet")
    ap.add_argument("--source", nargs="+", required=True,
                    help="raw table(s) carrying a nucleotide column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/run.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_parquet(args.table)
    key = os.path.splitext(os.path.basename(args.table))[0]
    col = detect_columns(df, cfg.column_overrides(key))
    if col["wt_seq"] is None:
        raise SystemExit(f"{args.table}: no wild-type sequence column found")

    cds_map = build_cds_map(args.source)
    if not cds_map:
        raise SystemExit(
            "no coding sequences recovered. Without a real CDS the mechanistic "
            "stream stays inactive (mech_valid = 0) — which is the honest "
            "outcome, but this script has nothing to add."
        )

    wt = df[col["wt_seq"]].astype(str).str.upper()
    attached = wt.map(cds_map.get)

    # A processed protein may be a sub-construct of a source ORF; try a
    # containment match for the rows that missed on exact equality.
    missing = attached.isna()
    if bool(missing.any()):
        by_length = sorted(cds_map.items(), key=lambda kv: -len(kv[0]))
        for idx in df.index[missing]:
            target = wt.at[idx]
            for prot, dna in by_length:
                start = prot.find(target)
                if start >= 0:
                    attached.at[idx] = dna[start * 3: (start + len(target)) * 3]
                    break

    n_hit = int(attached.notna().sum())
    df["wt_dna"] = attached
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)

    stats = {
        "table": args.table,
        "sources": args.source,
        "rows": int(len(df)),
        "rows_with_cds": n_hit,
        "coverage": (n_hit / len(df)) if len(df) else 0.0,
        "unique_cds_recovered": len(cds_map),
        "out": args.out,
    }
    report = os.path.splitext(args.out)[0] + "_cds_report.json"
    with open(report, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"[etl] wrote {args.out}")
    print(f"[etl] CDS attached to {n_hit}/{len(df)} rows ({stats['coverage']:.1%})")
    if n_hit < len(df):
        print("[etl] rows without a CDS keep mech_valid = 0 and an empty E_mech; "
              "the mechanistic ablation should be read against this coverage.")
    print(f"[etl] report -> {report}")


if __name__ == "__main__":
    main()
