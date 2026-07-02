# HoloGNN — Results Summary (auto-generated)

Numbers below are produced by this run on YOUR data and splits.

| Task | Metric | Value | Regime |
|---|---|---|---|
| Stability | Pearson r | 0.5268 | 30% protein-disjoint |
| Stability | MAE | 0.725 | |
| Stability | RMSE | 1.169 | |
| Stability | Mean antisymmetry violation | 0.029 | |
| Ablation | Δr from E_mech track | -0.0041 | dual − single track (measured) |
| Pathogenicity | AUROC | 0.918 | random split (optimistic) |
| Pathogenicity | AUROC | 0.669 | gene-disjoint |
| Pathogenicity | AUROC (CV) | 0.697 ± 0.017 | gene-disjoint GroupKFold |
| Baseline | ESM zero-shot \|r\| | 0.281 | same stability test split |
| IDR | Rg MAE vs physics prior | 26.37 Å | disclosed prior (not experimental) |
| Contacts (Q2) | Long-range fraction (top-L) | 0.252 vs 0.000 | ESM E_geo vs linear baseline |
