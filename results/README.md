# results/

Sorted outputs of a run, organised by task. Populated by:

```bash
python run.py results        # (run.py all calls this automatically at the end)
```

Layout
------
```
results/
├── CLAIMS.md            # component -> code -> evidence map
├── SUMMARY.md           # auto-generated headline metrics table (after a run)
├── stability/           # ΔΔG: Pearson r, MAE, RMSE, antisymmetry violation
├── ablation/            # dual-track vs single-track  -> the E_mech effect
├── pathogenicity/       # AUROC: random vs gene-disjoint, + GroupKFold CV, ROC
├── baselines/           # ESM zero-shot ΔΔG on the same test split
├── idr/                 # Rg MAE vs the polymer-physics prior
├── splits/              # leakage_report.txt (no train/test group overlap)
└── figures/             # generated plots (ROC, etc.)
```

The **gene-/protein-disjoint** numbers are the headline, leakage-controlled
results. `CLAIMS.md` shows which file backs each component.
