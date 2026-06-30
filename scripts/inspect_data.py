"""
scripts/inspect_data.py
=======================
Print the schema of a preprocessed parquet file and show how HoloGNN's column
auto-detection maps onto it. Run this FIRST on each dataset to confirm the
mapping; if anything is None, set it explicitly in configs/data.yaml.

Example
-------
python -m scripts.inspect_data --data data/megascale_ddg_pairs.parquet
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data import detect_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print(f"\nFILE: {args.data}")
    print(f"rows: {len(df):,}  cols: {len(df.columns)}")
    print("\ncolumns / dtypes:")
    for c in df.columns:
        print(f"  {c:30s} {str(df[c].dtype):12s} e.g. {repr(df[c].iloc[0])[:60]}")
    print("\nauto-detected mapping (None => set in configs/data.yaml):")
    for k, v in detect_columns(df).items():
        print(f"  {k:10s} -> {v}")
    print(f"\nhead({args.n}):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(args.n))


if __name__ == "__main__":
    main()
