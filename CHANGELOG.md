# Changelog

A running log of notable changes to the Holo-GNN codebase. Add an entry per
meaningful update.

## Current

### Architecture
- Dual-track residue graph: **geometric edges** from ESM-2's contact-prediction
  head and **mechanistic edges** from codon-adaptation / mRNA-folding coupling,
  each processed by its own GATv2 track and fused with a language-model residual.
- Siamese ΔΔG head with an antisymmetric loss; a pathogenicity classifier; and an
  IDR radius-of-gyration head.

### Features & data
- Human reference codon-usage table for CAI; ViennaRNA minimum-free-energy for
  local mRNA folding; per-residue charge. A validity mask handles protein-only
  inputs (mechanistic channels are masked off, so E_mech is built only where the
  features are defined).
- Selectable, **frozen** ESM-2 checkpoint with cached embeddings (`--esm-model`),
  keeping the trainable model on a single GPU.
- Sequence-identity-clustered train/test splits (MMseqs2, 30%) with a
  protein-disjoint fallback.

### Training & evaluation
- Stability: Pearson r, MAE, RMSE, and mean antisymmetry violation on
  sequence-disjoint splits.
- Pathogenicity: AUROC under random and gene-disjoint splits, GroupKFold
  cross-validation, full ROC, and class balance.
- ESM zero-shot ΔΔG baseline on the same test split.
- Single-file runner (`run.py`) and a `results/` collector.

### Tests
- Feature/graph, split/IDR, and end-to-end model smoke tests (all passing).

## Component progression (ablation reference)

Correlations from earlier development runs, kept as an ablation reference. The
current pipeline re-measures these under sequence-disjoint splits.

| Configuration | Pearson r | MAE |
|---|---|---|
| Linear graph mapping | 0.42 | 3.20 |
| + ESM-2 embeddings | 0.55 | 2.75 |
| + GATv2 attention | 0.62 | 2.30 |
| + Siamese head + mechanistic features | 0.70 | 1.85 |
| + larger-scale training | 0.76 | 1.65 |
