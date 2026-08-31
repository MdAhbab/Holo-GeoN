# Holo-GeoN

A sequence-based (MSA-free, structure-free) graph neural network for **protein
stability change (ΔΔG)**, **variant pathogenicity**, and related per-variant
properties. Frozen ESM-2 embeddings supply residue representations and a
contact-probability map; a dual-stream GATv2 backbone refines them over a
spatial edge set and a translational-coupling edge set; a site-anchored readout
and a substitution-conditioned head turn that into a prediction for one
mutation.

> **Status: no results are reported here.** This branch is the rebuilt model,
> ready to train. Nothing in this repository has been trained or evaluated yet,
> so there are no numbers to quote. `results/` is populated only by an actual
> run — see [Reproducing](#reproducing).

---

## How it works

### Encoder (frozen)

ESM-2 (`t33_650M` by default) runs once per unique **parent** protein and its
output is cached: per-residue embeddings in fp16, plus a **sparse** top-k edge
list from ESM-2's *contact-prediction head*. The contact head is used rather
than raw attention because its output is supervised against real structural
contacts — `scripts/contact_sanity.py` measures the long-range fraction among
top-L predicted pairs against a sequence-proximity baseline, which is the check
that justifies building a graph rather than a chain.

The encoder never enters the training loop, so training holds only the ~20M
trainable parameters and fits a 16 GB card with room to spare.

### Windowing

Every input is a window **centred on the mutated residue**, never the sequence
head. This is not configurable, because the alternative is dropping the mutation:
with head truncation at 256 residues and a median ClinVar mutation position of
383, most variants would have wild-type and mutant inputs that are byte-identical.

### Two message-passing streams

| Stream | Edges | Source |
|---|---|---|
| `E_geo` | spatial contacts + backbone + self-loops | ESM-2 contact head |
| `E_mech` | translational coupling: similar codon adaptation and local mRNA folding | real coding sequence |

Both carry 3-channel edge features — strength, log sequence separation, backbone
flag — so the network can tell a local contact from a long-range one.

`E_mech` requires a **real** coding sequence. There is no back-translation
fallback: a proxy transcript generated from the protein is a deterministic
function of it, so features derived from that proxy carry no information the
embeddings do not already have. Where a dataset has no CDS, `mech_valid` is 0
and `E_mech` is empty for those rows.

### Site-anchored readout

`z = MLP([z_site ‖ z_neighborhood ‖ z_global])` — the mutated residue's own
embedding, an attention pool over its k-hop structural neighbourhood under
`E_geo`, and a mask-aware pool over the whole window. Pooling the whole graph
and subtracting two protein-level vectors dilutes a one-residue change to ~1/L
of the signal; anchoring at the site does not.

### Substitution-conditioned head

```
ΔΔG(wt → mt) = s(z, e_mt) − s(z, e_wt)
```

The parent is encoded once and the head is conditioned on residue identity. Two
properties follow from the form itself rather than from a training penalty:

- **Exact reversibility.** `ΔΔG(b→a) = −ΔΔG(a→b)` to floating point, asserted in
  the test suite in both train and eval mode.
- **Free deep mutational scans.** All 20 substitutions at a site share one
  encoder pass, so a full 20 × L scan is tractable in the interface.

The head is heteroscedastic: it predicts `(μ, log σ²)` and trains under Gaussian
NLL, so `μ ± 1.96σ` is a real interval — once `scripts/calibration.py` has
measured its coverage. Until then the interface says so explicitly.

### Heads

| Head | Task | Trained when |
|---|---|---|
| `SubstitutionScoreHead` | ΔΔG with calibrated σ | stability table present |
| `ThreeStateHead` | destabilising / neutral / stabilising, swap-equivariant | stability table present |
| `PathogenicityHead` | binary variant effect | ClinVar table present |
| `IdrEnsembleHead` | radius of gyration (μ, σ) | experimental Rg table present |
| `MfiHead` | mean fluorescence intensity | MFI table present |
| `ProteomicsHead` | retention time / expression | proteomics table present |

All six train jointly in one model with Kendall & Gal uncertainty weighting. A
head whose data is absent is left untrained, is recorded as untrained in the run
manifest, and its interface endpoint refuses rather than returning a forward pass
through random weights.

---

## Installation

```bash
python run.py setup
# For an NVIDIA GPU, install the matching CUDA wheel first:
python run.py setup --torch-index-url https://download.pytorch.org/whl/cu128
```

**RTX 50-series (Blackwell, sm_120 — e.g. 5070 Ti):** a plain `pip install torch`
may fetch a wheel without sm_120 kernels, so the GPU errors or silently runs on
CPU. Install the cu128 build first, as above. Every stage prints its detected
device and warns with the fix if the wheel cannot target the card.

`torch_geometric` is required, not optional — there is no non-graph fallback,
because a model that silently skips its GNN layers is not this model.

MMseqs2 is a separate binary and is needed for identity-clustered splits:
`conda install -c bioconda mmseqs2`. Without it, splits fall back to exact
parent-sequence grouping, which is still protein-disjoint but does not separate
homologs; the leakage report records which was used.

---

## Reproducing

```bash
python run.py test          # unit + invariant tests, no data or network needed
python run.py all           # etl → precompute → splits → train → baselines →
                            # ablations → calibration → contacts → results
```

Stages are individually runnable and resumable. Check your data first:

```bash
python -m scripts.inspect_data --data data/clinvar_protein_pairs.parquet
```

That reports the detected columns, what fraction of rows survive normalisation
and why the rest do not, and the mutation-position distribution against the
window — the check whose absence let a 64% truncation loss go unnoticed through
a full training run.

Configuration lives in `configs/run.yaml` and `configs/data.yaml`. **Both are
read.** Column overrides in `data.yaml` had no effect in the previous version
because no script imported it.

Hardware budget for a planned configuration:

```bash
python -m scripts.budget --batch-size 64 --window 256
```

---

## Evaluation

Every task is split with MMseqs2 identity clustering unioned with gene identity,
whole groups to one side, with a leakage report recording the grouping, the
per-split target spread, and every excluded row by reason.

Baselines run on the identical split:

- **`esm_mlp`** — a supervised MLP on the same cached embeddings, no graph. This
  is the control that matters: the gap between it and the full model is the
  message-passing contribution. It is the first thing a reviewer asks for.
- **`esm_zero_shot`** — wild-type-marginal masked-LM scoring (Meier et al., 2021).
- **`substitution_frequency`** — the amino-acid pair alone.

Ablations remove one component at a time and report the delta. Calibration
reports coverage at 50/80/90/95%, PIT-based ECE, sharpness, and a reversibility
check that re-encodes the mutant as its own parent.

`scripts/collect_results.py` writes `results/SUMMARY.md` from run artifacts
only. Every value is measured with a path to its artifact, or reads `not run`.

---

## Web interface

```bash
python runapp.py                 # http://127.0.0.1:8000
python runapp.py --weights runs/main/model_best.pt
```

ΔΔG prediction with the model's own confidence interval, a deep-mutational-scan
heatmap, IDR ensemble plots, AlphaFold2 pLDDT/PAE comparison, CSV/JSON/PDB
export, and a SQLite-backed history.

**There is no demo mode.** Without a trained checkpoint the interface starts,
reports that no model is loaded, and every prediction endpoint returns an error
explaining what is missing. It does not substitute an approximation.

---

## Repository layout

```
run.py            pipeline entrypoint (setup / test / all / per-stage)
runapp.py         web interface launcher
configs/          run.yaml, data.yaml — both read
src/              encoder, windowing, graphs, backbone, readout, heads, losses, data
scripts/          etl, precompute, splits, train, baselines, ablations,
                  calibration, contacts, budget, collect_results
tests/            invariants, end-to-end pipeline, live backend
app/              backend (FastAPI) + frontend (React/Vite)
results/          written only by a real run
```

---

## Authors

Md Ahbab Hamid Khan, Md Wali Ullah Khan, Jamiul Hasan, Shadhin Nandi,
Ahnaf Atique, Akib Bari, and Riasat Azim\*
Department of Computer Science & Engineering, United International University,
Dhaka 1212, Bangladesh. *\*Corresponding author.*

## License

Released for academic and research use. See `LICENSE`.
