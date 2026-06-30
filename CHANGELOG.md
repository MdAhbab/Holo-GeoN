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

### Mechanistic track when no coding DNA is available
- Added a **usage-weighted back-translation fallback** (`codon_tables.back_translate`,
  wired through `features.mechanistic_features` and `data.build_graph`). When the
  stability data has no coding-DNA column, each residue is assigned a codon sampled
  from the human codon-usage distribution, seeded deterministically from the
  sequence (reproducible across runs/machines). This defines CAI and local mRNA
  folding — and therefore the **mechanistic edge set E_mech** — even on protein-only
  inputs, so the dual-track design is demonstrable without an experimental CDS.
  It is an explicit codon-usage proxy, not a real transcript, and is opt-in
  (`back_translate: false` by default; `--back-translate` to enable). Real
  `wt_dna/mt_dna` columns are always preferred when present. Requires ViennaRNA for
  real folding; otherwise the validity mask keeps E_mech off.

### Geometric-track validation (review Q2)
- New `scripts/contact_sanity.py` (`python run.py contacts`): checks that the ESM-2
  contact-head edges feeding E_geo carry genuine spatial signal. Structure-free mode
  reports the **long-range fraction** (|i−j| ≥ 24) of the top-L predicted contacts
  versus a sequence-proximity baseline (whose long-range fraction is ~0 by
  construction). Optional `--true-contacts-dir` adds **precision@L and precision@L/5**
  against ground-truth contact maps. Output goes to `results/contacts/`.

### Mechanistic-edge sensitivity (review Q9)
- `train_stability.py` gained `--eps` / `--delta` (the |ΔCAI| / |ΔmRNA| thresholds)
  and a `--sweep` mode that briefly retrains across a small grid of eps=delta values
  and records the test r for each (`eps_delta_sweep.json`). Exposed as
  `run.py ... --sweep-eps-delta`; surfaced in `results/ablation/` and `SUMMARY.md`.

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
