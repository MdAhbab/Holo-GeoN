# Component → evidence map

Each model component, the code that implements it, and the test or results file
that backs it. Run `python run.py all` (or individual stages) then
`python run.py results` to populate the evidence files.

| Component | Implementation | Evidence |
|---|---|---|
| Dual-track graph (E_geo + E_mech) | `src/utils/graph_builder.py`, `src/backbone.py` (`DualTrackGNN`) | `tests/test_features_graph.py`; `results/ablation/` (E_mech Δr) |
| Geometric edges from ESM-2 contacts | `src/backbone.py` (`EsmEncoder.contacts`) | embedding cache |
| Mechanistic edges (CAI / mRNA coupling) | `build_mechanistic_edges` | `tests/test_features_graph.py` |
| Human CAI table | `src/codon_tables.py` | `tests/test_features_graph.py` |
| Real mRNA folding (ViennaRNA MFE) | `src/features.py` | `tests/test_features_graph.py` |
| Protein-only handling (validity mask) | `src/features.py` (`mech_valid`) | `tests/test_features_graph.py` |
| Frozen, selectable ESM-2 + cached embeddings | `src/backbone.py`, `scripts/precompute_embeddings.py` | run config in each `*_metrics.json` |
| Leakage-controlled splits (MMseqs2 30%) | `src/splits.py`, `scripts/make_splits.py` | `results/splits/leakage_report.txt` |
| Antisymmetric ΔΔG head + loss | `src/heads.py`, `src/loss.py` | `tests/test_model_smoke.py`; `results/stability/` |
| Pathogenicity (random + gene-disjoint + CV) | `scripts/train_pathogenicity.py` | `results/pathogenicity/` |
| IDR head vs polymer-physics prior | `src/idr_prior.py`, `scripts/train_idr.py` | `results/idr/` |
| ESM zero-shot ΔΔG baseline | `scripts/baselines.py` | `results/baselines/` |
| E_mech ε / δ hyperparameters | `configs/data.yaml`, `configs/run.yaml` | configs |
| Checkpoints + splits saved | `runs/*/*.pt`, `data/splits_*.parquet` | run outputs |

Headline numbers are reported under the **gene-/protein-disjoint** splits
(`results/SUMMARY.md`).
