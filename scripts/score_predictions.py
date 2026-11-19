"""
scripts/score_predictions.py
============================
One scorer for every model in the comparison table.

Each model supplies a parquet with ``row_id`` (position in the splits table) and
``y_pred``. Truth, protein and cluster come from the splits table, so every model
is scored on the SAME rows with the SAME code.

    python -m scripts.score_predictions \
        --splits data/splits_megascale.parquet --split test \
        --model holo_geon=runs/full_s42/pred_test.parquet:+1 \
        --model thermompnn=runs/competitor_thermompnn/pred_test.parquet:-1 \
        --subset native

Design points, each one a way this comparison can quietly go wrong:

* **Common rows.** Models are compared on the intersection of the rows they all
  scored (a structure-based model may be unable to score some), and the
  coverage of each is printed. A model that skipped hard rows would otherwise
  look better.
* **Sign.** ``name=path:+1`` is positive = stabilising (this repo); ``:-1`` is
  positive = destabilising (ThermoMPNN, FireProtDB raw). The scorer flips the
  second to the internal convention and REFUSES a model whose Pearson r is
  negative after the flip: that is a wrong sign, not a bad model.
* **Intervals.** A cluster bootstrap over ``group_used`` (the MMseqs identity
  clusters), not rows: rows within a protein, and proteins within a cluster, are
  not independent (OVERFITTING_FIX.md section 5.1 measured the row bootstrap
  3.4x too narrow). The paired difference to the first model uses the same
  resamples.
* **Subset.** ``--subset native`` keeps constructs whose parent is a plain PDB
  entry and drops the single-substitution background variants (``1A0N.pdb_L7S``),
  which a structure-based model has no structure for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "rmse": float("nan")}
    return {"pearson": float(np.corrcoef(y, p)[0, 1]),
            "spearman": float(stats.spearmanr(y, p)[0]),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def cluster_bootstrap(y, preds: dict, groups, reps: int, seed: int) -> dict:
    """Resample whole clusters; return per-model and paired-vs-first draws."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    idx = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    names = list(preds)
    draws = {n: {"pearson": [], "spearman": [], "rmse": []} for n in names}
    for _ in range(reps):
        pick = rng.integers(0, len(idx), len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        for n in names:
            m = _metrics(y[rows], preds[n][rows])
            for k, v in m.items():
                draws[n][k].append(v)
    return {n: {k: np.asarray(v) for k, v in d.items()} for n, d in draws.items()}


def _ci(a: np.ndarray) -> list:
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", default="data/splits_megascale.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", required=True,
                    help="name=path.parquet[:+1|:-1]  (sign of the model's own output)")
    ap.add_argument("--subset", choices=["all", "native"], default="all")
    ap.add_argument("--names", default=None,
                    help="megascale_ddg_pairs.parquet, to resolve parent names for --subset native")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.splits)
    df["row_id"] = np.arange(len(df))
    df = df[df["split"] == args.split]
    if args.subset == "native":
        names = pd.read_parquet(args.names or os.path.join(os.path.dirname(args.splits),
                                                           "megascale_ddg_pairs.parquet"),
                                columns=["parent", "wt_seq"]).drop_duplicates()
        df = df.merge(names, left_on="parent_seq", right_on="wt_seq", how="left")
        df = df[df["parent"].astype(str).str.endswith(".pdb")]

    scored: dict[str, pd.Series] = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        sign = 1.0
        if rest[-3:] in (":+1", ":-1"):
            sign, rest = float(rest[-2:]), rest[:-3]
        pred = pd.read_parquet(rest)[["row_id", "y_pred"]].drop_duplicates("row_id").set_index("row_id")["y_pred"]
        scored[name] = pred * sign

    common = set(df["row_id"])
    for s in scored.values():
        common &= set(s.index)
    if not common:
        raise SystemExit("no row is scored by every model")
    sub = df[df["row_id"].isin(common)].sort_values("row_id")
    y = sub["target"].to_numpy(float)
    groups = sub["group_used"].to_numpy()
    preds = {n: s.loc[sub["row_id"]].to_numpy(float) for n, s in scored.items()}

    print(f"rows scored by all models: {len(sub):,} of {len(df):,} in "
          f"{args.split}/{args.subset}; clusters: {pd.Series(groups).nunique()}; "
          f"proteins: {sub['parent_seq'].nunique()}")
    for n, s in scored.items():
        print(f"  coverage {n:<22}: {len(set(s.index) & set(df['row_id'])):,} / {len(df):,}")

    for n, p in preds.items():
        r = _metrics(y, p)["pearson"]
        if not np.isfinite(r) or r < 0:
            raise SystemExit(f"{n}: Pearson r = {r:+.3f} after applying the declared sign. "
                             f"That is a sign-convention error, not a model result. Re-check ':+1/:-1'.")

    boot = cluster_bootstrap(y, preds, groups, args.reps, args.seed)
    first = next(iter(preds))
    out: dict = {"split": args.split, "subset": args.subset, "n_rows": int(len(sub)),
                 "n_clusters": int(pd.Series(groups).nunique()), "reps": args.reps, "models": {}}
    print(f"\n{'model':<22}{'Pearson [95% CI]':<30}{'Spearman':<12}{'RMSE':<10}{'dPearson vs ' + first}")
    for n, p in preds.items():
        m = _metrics(y, p)
        lo, hi = _ci(boot[n]["pearson"])
        entry = {**m, "pearson_ci95": [lo, hi]}
        if n != first:
            d = boot[n]["pearson"] - boot[first]["pearson"]
            entry["pearson_delta_vs_first"] = float(m["pearson"] - _metrics(y, preds[first])["pearson"])
            entry["pearson_delta_ci95"] = _ci(d)
            dtxt = f"{entry['pearson_delta_vs_first']:+.4f} {entry['pearson_delta_ci95']}"
        else:
            dtxt = "-"
        out["models"][n] = entry
        print(f"{n:<22}{m['pearson']:.4f} [{lo:.3f},{hi:.3f}]".ljust(52)
              + f"{m['spearman']:<12.4f}{m['rmse']:<10.4f}{dtxt}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
