"""
scripts/inspect_data.py
=======================
Check a table before running anything heavy.

Reports which canonical columns were detected, how many rows survive
normalisation and why the rest do not, and — importantly — how the mutation
positions are distributed relative to the window.

That last part is the check v1 lacked. With head truncation at 256 residues and
a median ClinVar mutation position of 383, roughly 64% of variants had their
mutation cut away entirely, leaving wild-type and mutant inputs identical. That
was discovered after a full training run. This prints it in seconds.

Example
-------
python -m scripts.inspect_data --data data/clinvar_protein_pairs.parquet
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import detect_columns, load_mutation_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--kind", choices=["stability", "pathogenicity"], default=None)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    key = os.path.splitext(os.path.basename(args.data))[0]
    overrides = cfg.column_overrides(key)

    df = pd.read_parquet(args.data)
    print(f"[inspect] {args.data}: {len(df)} rows, {len(df.columns)} columns")
    print(f"[inspect] columns: {list(df.columns)}")
    if overrides:
        print(f"[inspect] overrides from configs/data.yaml [{key}]: {overrides}")

    detected = detect_columns(df, overrides)
    print("\n[inspect] canonical column mapping:")
    for canonical, actual in detected.items():
        flag = "" if actual else "   <- not found"
        print(f"    {canonical:<10} -> {actual}{flag}")

    kind = args.kind
    if kind is None:
        kind = "stability" if detected["target"] else "pathogenicity"
    print(f"\n[inspect] treating as: {kind}")

    try:
        norm, report = load_mutation_table(
            args.data, kind=kind, overrides=overrides,
            max_protein_length=cfg.max_protein_length,
        )
    except ValueError as exc:
        print(f"\n[inspect] cannot normalise: {exc}")
        return

    print(report.summary())
    if norm.empty:
        return

    sites = norm["site"].to_numpy()
    lengths = norm["parent_seq"].str.len().to_numpy()
    window = cfg.window

    print(f"\n[inspect] mutation position: median {int(np.median(sites)) + 1}, "
          f"mean {sites.mean() + 1:.0f}, max {int(sites.max()) + 1}")
    print(f"[inspect] parent length: median {int(np.median(lengths))}, "
          f"max {int(lengths.max())}")

    # What head truncation would have cost, versus what windowing costs.
    lost_to_truncation = float(np.mean(sites >= window))
    print(f"\n[inspect] with head truncation at {window}: "
          f"{lost_to_truncation:.1%} of mutations would fall outside the input "
          f"(wild-type and mutant would be identical for those rows)")
    print(f"[inspect] with mutation-centered windowing at {window}: 0.0% — the "
          f"window is placed around the mutation, so it is always present")

    beyond_context = float(np.mean(lengths > 1022))
    if beyond_context:
        print(f"[inspect] {beyond_context:.1%} of parents exceed the encoder's "
              f"1022-residue context and are embedded in overlapping chunks; "
              f"contacts spanning chunks are not recovered")

    has_cds = norm["cds"].notna().mean() if "cds" in norm.columns else 0.0
    print(f"[inspect] rows carrying a coding sequence: {has_cds:.1%} "
          f"({'E_mech active for those rows' if has_cds else 'E_mech will be empty'})")

    print(f"\n[inspect] first {args.rows} normalised rows:")
    cols = [c for c in ("site", "wt_aa", "mt_aa", "target", "label", "group")
            if c in norm.columns]
    print(norm[cols].head(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
