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


def _per_protein_medians(y: np.ndarray, p: np.ndarray, proteins: np.ndarray) -> dict:
    """
    Per-protein median Pearson and Spearman over proteins with >= 3 mutations.

    Kept strictly separate from pooled metrics (BENCHMARK_PLAN.md §2: SPURS
    reports per-protein median Spearman 0.83 while ThermoMPNN reports pooled
    0.725; both columns must be reported side by side for every model).
    """
    pears, spears = [], []
    _, inv = np.unique(proteins, return_inverse=True)
    for idx in range(int(inv.max()) + 1 if len(inv) else 0):
        m = inv == idx
        if int(m.sum()) < 3:
            continue
        yi, pi = y[m], p[m]
        if np.std(yi) == 0 or np.std(pi) == 0:
            continue
        pears.append(float(np.corrcoef(yi, pi)[0, 1]))
        spears.append(float(stats.spearmanr(yi, pi)[0]))
    return {
        "per_protein_median_pearson": float(np.median(pears)) if pears else float("nan"),
        "per_protein_median_spearman": float(np.median(spears)) if spears else float("nan"),
        "n_proteins_scored": len(spears),
    }


def cluster_bootstrap(y, preds: dict, groups, reps: int, seed: int, proteins=None) -> dict:
    """Resample whole clusters; return per-model pooled and per-protein draws."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    idx = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    names = list(preds)
    draws = {
        n: {"pearson": [], "spearman": [], "rmse": [], "med_spearman": [], "med_pearson": []}
        for n in names
    }
    for _ in range(reps):
        pick = rng.integers(0, len(idx), len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        for n in names:
            m = _metrics(y[rows], preds[n][rows])
            for k in ("pearson", "spearman", "rmse"):
                draws[n][k].append(m[k])
            if proteins is not None:
                pm = _per_protein_medians(y[rows], preds[n][rows], proteins[rows])
                draws[n]["med_spearman"].append(pm["per_protein_median_spearman"])
                draws[n]["med_pearson"].append(pm["per_protein_median_pearson"])
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

    proteins = sub["parent_seq"].to_numpy()
    boot = cluster_bootstrap(y, preds, groups, args.reps, args.seed, proteins=proteins)
    first = next(iter(preds))
    out: dict = {"split": args.split, "subset": args.subset, "n_rows": int(len(sub)),
                 "n_clusters": int(pd.Series(groups).nunique()),
                 "n_proteins": int(sub["parent_seq"].nunique()),
                 "reps": args.reps, "seed": args.seed,
                 "ci": "95% percentile interval over cluster-bootstrap resamples of group_used",
                 "models": {}}
    print(f"\n{'model':<22}{'Pearson [95% CI]':<28}{'PoolRho':<10}{'MedRho':<10}{'RMSE':<9}{'dPearson vs ' + first}")
    for n, p in preds.items():
        m = _metrics(y, p)
        pm = _per_protein_medians(y, p, proteins)
        lo, hi = _ci(boot[n]["pearson"])
        slo, shi = _ci(boot[n]["med_spearman"])
        # Every reported metric carries its own cluster-bootstrap interval: pooled
        # Pearson / Spearman / RMSE and the per-protein medians, in separate keys.
        entry = {
            **m,
            **pm,
            "pearson_ci95": [lo, hi],
            "spearman_ci95": _ci(boot[n]["spearman"]),
            "rmse_ci95": _ci(boot[n]["rmse"]),
            "per_protein_median_pearson_ci95": _ci(boot[n]["med_pearson"]),
            "per_protein_median_spearman_ci95": [slo, shi],
        }
        if n != first:
            d = boot[n]["pearson"] - boot[first]["pearson"]
            entry["pearson_delta_vs_first"] = float(m["pearson"] - _metrics(y, preds[first])["pearson"])
            entry["pearson_delta_ci95"] = _ci(d)
            dtxt = f"{entry['pearson_delta_vs_first']:+.4f} {entry['pearson_delta_ci95']}"
        else:
            dtxt = "-"
        out["models"][n] = entry
        ci_txt = f"[{lo:.3f},{hi:.3f}]" if lo is not None and hi is not None else "[n/a]"
        print(f"{n:<22}{m['pearson']:.4f} {ci_txt}".ljust(50)
              + f"{m['spearman']:<10.4f}{pm['per_protein_median_spearman']:<10.4f}"
              + f"{m['rmse']:<9.4f}{dtxt}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
