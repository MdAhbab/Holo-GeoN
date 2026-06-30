"""
scripts/collect_results.py
==========================
Sort everything produced by a run into results/, organised by task. Reads
runs/*/*.json (+ split parquet) and writes:

    results/SUMMARY.md                      consolidated metric table
    results/stability/                      ΔΔG metrics (strict split)
    results/ablation/                       dual-track vs single-track (E_mech delta)
    results/pathogenicity/                  random vs gene-disjoint AUROC + CV
    results/baselines/                      ESM zero-shot ΔΔG
    results/idr/                            physics-prior Rg MAE
    results/splits/leakage_report.txt       leakage check + group counts

It also writes the component -> evidence map referenced by results/CLAIMS.md.
"""
import os, sys, json, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SUBDIRS = ["stability", "ablation", "pathogenicity", "baselines", "idr", "splits", "figures"]


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def copy(src, sub, name=None):
    if src and os.path.exists(src):
        os.makedirs(os.path.join(RESULTS, sub), exist_ok=True)
        shutil.copy(src, os.path.join(RESULTS, sub, name or os.path.basename(src)))


def leakage_report(runs_root):
    out = []
    try:
        import pandas as pd
        from src.splits import verify_no_leakage
        for f in glob.glob(os.path.join(ROOT, "data", "splits_*.parquet")):
            df = pd.read_parquet(f)
            gk = df["group_used"].tolist() if "group_used" in df else df["group"].tolist()
            ok = verify_no_leakage(gk, df["split"].tolist())
            counts = df["split"].value_counts().to_dict()
            out.append(f"{os.path.basename(f)}: no_leakage={ok} | splits={counts} | "
                       f"unique_groups={len(set(gk))} | rows={len(df)}")
    except Exception as e:
        out.append(f"(leakage report skipped: {e})")
    return "\n".join(out) if out else "(no split files found in data/)"


def main():
    runs = os.path.join(ROOT, "runs")
    os.makedirs(RESULTS, exist_ok=True)
    for d in SUBDIRS:
        os.makedirs(os.path.join(RESULTS, d), exist_ok=True)

    stab = jload(os.path.join(runs, "stability", "stability_metrics.json"))
    nomech = jload(os.path.join(runs, "stability_nomech", "stability_metrics.json"))
    path = jload(os.path.join(runs, "pathogenicity", "pathogenicity_metrics.json"))
    idr = jload(os.path.join(runs, "idr", "idr_metrics.json"))
    base = jload(os.path.join(runs, "baseline_esm_zeroshot.json"))

    copy(os.path.join(runs, "stability", "stability_metrics.json"), "stability")
    copy(os.path.join(runs, "stability_nomech", "stability_metrics.json"), "ablation", "stability_nomech_metrics.json")
    copy(os.path.join(runs, "pathogenicity", "pathogenicity_metrics.json"), "pathogenicity")
    copy(os.path.join(runs, "idr", "idr_metrics.json"), "idr")
    copy(os.path.join(runs, "baseline_esm_zeroshot.json"), "baselines")
    for png in glob.glob(os.path.join(runs, "figures", "*.png")):
        copy(png, "figures")

    # leakage report
    with open(os.path.join(RESULTS, "splits", "leakage_report.txt"), "w") as fh:
        fh.write(leakage_report(runs) + "\n")

    # SUMMARY.md
    L = ["# HoloGNN — Results Summary (auto-generated)\n",
         "Numbers below are produced by this run on YOUR data and splits.\n",
         "| Task | Metric | Value | Regime |", "|---|---|---|---|"]
    if stab and stab.get("best"):
        b = stab["best"]
        L += [f"| Stability | Pearson r | {b['pearson_r']:.4f} | 30% protein-disjoint |",
              f"| Stability | MAE | {b['mae']:.3f} | |",
              f"| Stability | RMSE | {b['rmse']:.3f} | |",
              f"| Stability | Mean antisymmetry violation | {b['mean_antisymmetry_violation']:.3f} | |"]
    if stab and nomech and stab.get("best") and nomech.get("best"):
        d = stab["best"]["pearson_r"] - nomech["best"]["pearson_r"]
        L += [f"| Ablation | Δr from E_mech track | {d:+.4f} | dual − single track (measured) |"]
    if path and path.get("results"):
        r = path["results"]
        if "random_split" in r:
            L += [f"| Pathogenicity | AUROC | {r['random_split']['auroc']:.3f} | random split (optimistic) |"]
        if "gene_disjoint_split" in r:
            L += [f"| Pathogenicity | AUROC | {r['gene_disjoint_split']['auroc']:.3f} | gene-disjoint |"]
        if "gene_disjoint_cv" in r:
            cv = r["gene_disjoint_cv"]
            L += [f"| Pathogenicity | AUROC (CV) | {cv['mean']:.3f} ± {cv['std']:.3f} | gene-disjoint GroupKFold |"]
    if base:
        L += [f"| Baseline | ESM zero-shot \\|r\\| | {base.get('abs_pearson_r', float('nan')):.3f} | same stability test split |"]
    if idr:
        L += [f"| IDR | Rg MAE vs physics prior | {idr.get('rg_mae_vs_prior', float('nan')):.2f} Å | disclosed prior (not experimental) |"]

    with open(os.path.join(RESULTS, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print("Wrote results/ (SUMMARY.md + sorted metric files). Key numbers:")
    print("\n".join(L[4:]) if len(L) > 4 else "  (no run outputs found yet - run the pipeline first)")


if __name__ == "__main__":
    main()
