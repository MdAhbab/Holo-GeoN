# Holo-GNN

Holo-GNN is a sequence-based (MSA-free) framework for predicting **protein
stability changes (ΔΔG)** and **variant pathogenicity**. It pairs frozen ESM-2
protein-language-model embeddings with mechanistic sequence features and a
**dual-track graph neural network**.

## Overview

Each protein is represented as a graph over its residues with two complementary
edge sets:

- **Geometric edges (E_geo)** — residue contacts from ESM-2's contact-prediction
  head.
- **Mechanistic edges (E_mech)** — links between residues with similar codon
  adaptation (CAI) and local mRNA folding energy (translational coupling).

A GATv2 track operates over each edge set; the two are fused with a
language-model residual and mean-pooled to a graph embedding. Three heads share
this backbone:

- a **Siamese antisymmetric** head for ΔΔG, enforcing
  ΔΔG(A→B) = −ΔΔG(B→A);
- a **pathogenicity** classifier;
- an **IDR** head predicting the radius of gyration of disordered regions.

ESM-2 is frozen and its embeddings are cached, so the trainable model fits a
single consumer GPU.

## Installation

```bash
python run.py setup                                   # create a venv + install deps
python run.py setup --torch-index-url https://download.pytorch.org/whl/cu121   # CUDA build
python run.py setup --venv /path/to/env               # custom venv location
```

Dependencies: PyTorch, PyTorch Geometric, Transformers, ViennaRNA, scikit-learn,
pandas/pyarrow, matplotlib (installed by `setup`). MMseqs2 (a separate binary) is
optional and enables sequence-identity-clustered splits.

## Quick start

```bash
python run.py smoke          # short sanity check (no data required)
# place the dataset parquet files under data/ (see guide.md), then:
python run.py all            # precompute -> splits -> train -> evaluate -> results
```

Individual stages: `python run.py {precompute,splits,stability,pathogenicity,idr,baseline,contacts,evaluate,results}`.

## Configuration

All paths, epochs, batch sizes, ESM checkpoint, and split settings live in
[`configs/run.yaml`](configs/run.yaml); any value can be overridden on the command
line. Parquet column names and graph hyperparameters (the E_mech thresholds
ε, δ) are in [`configs/data.yaml`](configs/data.yaml).

## Evaluation

The pipeline reports ΔΔG accuracy (Pearson r, MAE, RMSE) and thermodynamic
consistency (mean antisymmetry violation) on sequence-disjoint test splits;
pathogenicity AUROC under both random and gene-disjoint splits with GroupKFold
cross-validation; and an ESM zero-shot ΔΔG baseline on the same test split.
The dual-track vs single-track gap (`--no-mech-track`) isolates the contribution
of the mechanistic edges, and an optional `--sweep-eps-delta` probes their
threshold sensitivity. A `contacts` stage checks that the geometric edges carry
long-range (spatial) signal beyond sequence adjacency. Outputs are collected
under [`results/`](results/).

## Repository layout

```
run.py                 single entrypoint (setup / smoke / all / per-stage)
configs/               run.yaml (pipeline), data.yaml (columns + graph params)
src/                   model, features, graph construction, splits
scripts/               one module per stage (precompute, splits, train_*, baselines, ...)
tests/                 unit + smoke tests  (python run.py test)
results/               collected metrics and figures
guide.md               full setup + run instructions and time budget
```

## Method notes

- CAI is computed against a fixed human reference codon-usage table (a normalised
  codon-bias proxy). Local mRNA folding energy is the ViennaRNA minimum free
  energy. For protein-only inputs (no coding sequence) these channels are masked
  off, so E_mech is built only where the features are defined. When a coding
  sequence is unavailable but the mechanistic track is still wanted, an optional
  usage-weighted back-translation (`--back-translate`) supplies a deterministic
  proxy transcript; it is treated as a codon-usage proxy, not an experimental CDS.
- The IDR head is trained against a sequence-dependent polymer-physics target
  (Flory scaling); experimental ensemble datasets can be substituted as labels.
- Train/test splits are clustered at 30% sequence identity (MMseqs2), with whole
  clusters assigned to a single split.

See [`guide.md`](guide.md) for full instructions.
