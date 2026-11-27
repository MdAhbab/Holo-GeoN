"""
scripts/error_decomp.py
=======================
Between-protein / between-site / within-site variance and error decomposition
(OVERFITTING_FIX.md §5.4).

Answers *where* a model's prediction error actually comes from on unseen
proteins:
  * **Between-protein mean error** — predicting a protein's overall baseline
    destabilization shift (``mean_p(y)`` vs ``mean_p(y_hat)``).
  * **Between-site mean error** — predicting which sites are buried/intolerant
    vs. solvent-exposed/tolerant (``mean_{p,i}(y) - mean_p(y)``).
  * **Within-site substitution error** — ranking the 19 substitutions at the
    SAME site (``y - mean_{p,i}(y)``).

Comparing Holo-GeoN v2 against v1 and competitors on these three orthogonal
components isolates whether gains come from better site-environment encoding
(E_geo / contact burial / multi-layer ScalarMix) or better substitution physics
(ESM-2 WT-marginal log-odds prior + AA biophysics).

Usage:
    python -m scripts.error_decomp --splits data/splits_megascale.parquet \\
        --split test --preds runs/v2_full/pred_test.parquet --out decomp.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats


def _safe_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _safe_rho(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    val = stats.spearmanr(a[mask], b[mask]).statistic
    return float(val) if np.isfinite(val) else float("nan")


def decompose_errors(df: pd.DataFrame, y_col: str = "target", pred_col: str = "y_pred",
                     prot_col: str = "parent_seq", site_col: str = "site") -> dict:
    """
    Orthogonal ANOVA-style decomposition of ``y`` and ``y_pred`` into:
      y_{p,i,a} = mu_global + b_p + s_{p,i} + w_{p,i,a}
    where:
      b_p       = mean_p(y) - mu_global            (protein offset)
      s_{p,i}   = mean_{p,i}(y) - mean_p(y)        (site sensitivity within protein)
      w_{p,i,a} = y_{p,i,a} - mean_{p,i}(y)        (substitution contrast within site)
    """
    work = df[[prot_col, site_col, y_col, pred_col]].dropna().copy()
    if work.empty:
        return {"n_rows": 0}

    y = work[y_col].to_numpy(dtype=np.float64)
    p = work[pred_col].to_numpy(dtype=np.float64)
    err = p - y

    # Group means
    work["_y_prot"] = work.groupby(prot_col)[y_col].transform("mean")
    work["_p_prot"] = work.groupby(prot_col)[pred_col].transform("mean")
    work["_y_site"] = work.groupby([prot_col, site_col])[y_col].transform("mean")
    work["_p_site"] = work.groupby([prot_col, site_col])[pred_col].transform("mean")

    y_prot = work["_y_prot"].to_numpy(dtype=np.float64)
    p_prot = work["_p_prot"].to_numpy(dtype=np.float64)
    y_site = work["_y_site"].to_numpy(dtype=np.float64)
    p_site = work["_p_site"].to_numpy(dtype=np.float64)

    # Orthogonal centered components (row-level sum == total centered signal)
    g_y, g_p = float(y.mean()), float(p.mean())
    b_y, b_p = y_prot - g_y, p_prot - g_p
    s_y, s_p = y_site - y_prot, p_site - p_prot
    w_y, w_p = y - y_site, p - p_site

    mse_total = float(np.mean(err ** 2))
    mse_global = float((g_p - g_y) ** 2)
    mse_prot = float(np.mean((b_p - b_y) ** 2))
    mse_site = float(np.mean((s_p - s_y) ** 2))
    mse_within = float(np.mean((w_p - w_y) ** 2))

    # Per-site and per-protein unique tables for unweighted correlations
    prot_df = work.groupby(prot_col, as_index=False)[[y_col, pred_col]].mean()
    site_df = work.groupby([prot_col, site_col], as_index=False)[[y_col, pred_col]].mean()
    # Keep sites with >= 3 mutations for within-site median r
    site_counts = work.groupby([prot_col, site_col]).size()
    multi_sites = site_counts[site_counts >= 3].index
    within_rs = []
    if len(multi_sites) > 0:
        for _, grp in work.set_index([prot_col, site_col]).loc[multi_sites].groupby(level=[0, 1]):
            r_s = _safe_r(grp[y_col].to_numpy(), grp[pred_col].to_numpy())
            if np.isfinite(r_s):
                within_rs.append(r_s)

    denom = max(mse_total, 1e-12)
    return {
        "n_rows": int(len(work)),
        "n_proteins": int(len(prot_df)),
        "n_sites": int(len(site_df)),
        "total": {
            "pearson": round(_safe_r(y, p), 4),
            "spearman": round(_safe_rho(y, p), 4),
            "rmse": round(float(np.sqrt(mse_total)), 4),
            "mse": round(mse_total, 5),
            "bias": round(float(g_p - g_y), 4),
        },
        "mse_decomposition": {
            "global_bias_mse": round(mse_global, 5),
            "between_protein_mse": round(mse_prot, 5),
            "between_site_mse": round(mse_site, 5),
            "within_site_mse": round(mse_within, 5),
            "sum_check_mse": round(mse_global + mse_prot + mse_site + mse_within, 5),
            "frac_global_bias": round(mse_global / denom, 4),
            "frac_between_protein": round(mse_prot / denom, 4),
            "frac_between_site": round(mse_site / denom, 4),
            "frac_within_site": round(mse_within / denom, 4),
        },
        "component_correlations": {
            "between_protein_pearson": round(_safe_r(prot_df[y_col].to_numpy(), prot_df[pred_col].to_numpy()), 4),
            "between_site_all_pearson": round(_safe_r(site_df[y_col].to_numpy(), site_df[pred_col].to_numpy()), 4),
            "between_site_within_protein_pearson": round(_safe_r(s_y, s_p), 4),
            "within_site_pooled_pearson": round(_safe_r(w_y, w_p), 4),
            "within_site_median_pearson": round(float(np.median(within_rs)), 4) if within_rs else float("nan"),
            "n_sites_ge3_muts": len(within_rs),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Between-protein / between-site / within-site error decomposition")
    ap.add_argument("--splits", required=True, help="Splits or holdout parquet")
    ap.add_argument("--split", default="test", help="Split name (ignored if parquet has no split column)")
    ap.add_argument("--preds", required=True, help="Prediction parquet with row_id and y_pred")
    ap.add_argument("--sign", type=float, default=1.0, help="Sign multiplier for y_pred (+1 or -1)")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args(argv)

    tbl = pd.read_parquet(args.splits).copy()
    tbl["row_id"] = np.arange(len(tbl), dtype=np.int64)
    if "split" in tbl.columns and args.split:
        tbl = tbl[tbl["split"] == args.split].copy()
    preds = pd.read_parquet(args.preds).copy()
    preds["y_pred"] = preds["y_pred"].astype(float) * float(args.sign)

    merged = tbl.merge(preds[["row_id", "y_pred"]], on="row_id", how="inner")
    y_col = "target" if "target" in merged.columns else "ddg"
    prot_col = "parent_seq" if "parent_seq" in merged.columns else "wt_seq"
    res = decompose_errors(merged, y_col=y_col, pred_col="y_pred", prot_col=prot_col, site_col="site")
    res["splits"] = args.splits
    res["preds"] = args.preds
    res["sign"] = args.sign

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
