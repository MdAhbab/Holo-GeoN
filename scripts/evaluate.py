"""
scripts/evaluate.py
===================
Collate metrics from run directories and render figures (ROC, summary). Reads the
JSON written by the train_* scripts — no recomputation needed.

Example
-------
python -m scripts.evaluate --runs runs/stability_150M runs/pathogenicity_150M --out runs/figures
"""
import os, sys, argparse, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsons(run_dirs):
    found = {}
    for d in run_dirs:
        for jp in glob.glob(os.path.join(d, "*.json")):
            found[jp] = json.load(open(jp))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="runs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load_jsons(args.runs)

    print("\n=== METRIC SUMMARY ===")
    for path, obj in data.items():
        print(f"\n{path}")
        if "best" in obj:               # stability
            b = obj["best"]
            print(f"  stability: r={b['pearson_r']:.4f}  MAE={b['mae']:.3f}  "
                  f"RMSE={b['rmse']:.3f}  antisym_viol={b['mean_antisymmetry_violation']:.3f}")
        if "results" in obj:            # pathogenicity
            res = obj["results"]
            for k, v in res.items():
                if isinstance(v, dict) and "auroc" in v:
                    print(f"  pathogenicity[{k}]: AUROC={v['auroc']:.3f} AUPRC={v.get('auprc',float('nan')):.3f}")
                elif k.endswith("cv"):
                    print(f"  pathogenicity[{k}]: AUROC={v['mean']:.3f} ± {v['std']:.3f}")

    # ROC plot (optional, needs matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for path, obj in data.items():
            roc = obj.get("roc_gene_disjoint")
            if roc and roc[0]:
                fpr, tpr = roc
                plt.figure(figsize=(4, 4))
                plt.plot(fpr, tpr, lw=2, label="gene-disjoint")
                plt.plot([0, 1], [0, 1], "k--", lw=1)
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title("Pathogenicity ROC (gene-disjoint)")
                plt.legend(); plt.tight_layout()
                fig = os.path.join(args.out, "roc_gene_disjoint.png")
                plt.savefig(fig, dpi=150); print(f"\nWrote {fig}")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
