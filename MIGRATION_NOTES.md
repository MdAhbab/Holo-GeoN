# Migration notes — v1 to v2

What moved, what was deliberately left behind, and what has **not** been run.

The previous submission was built under time pressure and a lot was skipped.
This branch reworks the model with those gaps closed. `version-1` preserves the
submitted state, including its results and configs.

---

## What has not been validated

**No part of this branch has been trained or evaluated on real data.**

The environment this was built in cannot reach HuggingFace (blocked by egress
policy, HTTP 403), so ESM-2 weights were never loaded, and it has no GPU. There
is therefore no encoder cache, no checkpoint, and no metric anywhere in this
repository. `results/` is empty by design.

What *was* run, and passes:

| Check | Status |
|---|---|
| Unit + invariant tests (`python run.py test`) | 120 checks, all passing |
| End-to-end pipeline on synthetic data through the real CLI stages | passing |
| Live web backend, with and without a checkpoint, every endpoint | passing |
| Frontend `tsc` + `vite build` | passing |
| Hardware budget from measured tensor shapes | ~20.6M trainable params, ~0.8 GB at batch 64 |

Everything measured in those runs used a **synthetic stub encoder** with seeded
noise for embeddings. Those numbers describe plumbing — shapes, batching,
checkpointing, metric computation — and say nothing whatever about accuracy.

Before quoting any number from this branch, run `python run.py all` on the real
data.

---

## Migrated from HoloGNN

The newer build had the better model machinery, and it came across:

- Per-sample graph construction with attention/contact weights as edge features,
  top-k neighbourhoods and guaranteed backbone edges.
- Mask-aware attention pooling.
- Antisymmetric-by-construction ΔΔG scoring and the heteroscedastic (μ, σ) head.
- Three-state, MFI and proteomics heads.
- Cross-attention fusion and the selective state-space mixer.
- The whole web interface (React + FastAPI + SQLite history).
- Blackwell/sm_120 device detection and the cu128 install guidance.

## Kept from Holo-GeoN v1

- MMseqs2 identity-clustered, leakage-controlled splits.
- ESM-2 **contact head** as the edge source, and its sanity check.
- Real ViennaRNA folding energy and the human CAI table.
- The `mech_valid` mask, which distinguishes "no codon context" from "value is 0".
- ESM zero-shot baseline, `run.py` staged runner, resumable stages, UTF-8 forcing.

---

## Reworked

### Substitution-conditioned instead of twin-tower

The parent is encoded once and ΔΔG is `s(z, e_mt) − s(z, e_wt)` at an anchored
site, replacing "encode both proteins, subtract global pools".

*Why:* a one-residue change contributed ~1/L of a global pool — under half a
percent at a 256-residue window — and then had to survive a LayerNorm and an MLP.
Reversibility also becomes exact rather than penalised (v1 measured mean
violations of 0.029–0.071 kcal/mol), and a full deep mutational scan drops from
20 × L × 2 encoder passes to one per position.

### Mutation-centered windowing, restored

Windowing was written and then deleted from main in `a18cdf1`. It is back and
unconditional.

*Why:* `Model_Changes.md` §7 records that ~64% of ClinVar variants have their
mutation past residue 256, with median position 383. Under head truncation, the
wild-type and mutant inputs for those rows were byte-identical — the model was
asked to classify a difference that was not in its input.

### Back-translation removed

`--back-translate` and the GC-content substitute for real folding are gone.
`scripts/etl_cds.py` recovers the real coding sequence instead.

*Why:* a back-translated transcript is a deterministic function of the protein
sequence, so CAI and mRNA folding derived from it are too — they carry no
information the language-model embeddings lack, and the mechanistic edge set can
only dilute. The v1 measurement agrees: single-track r = 0.5309 **beat**
dual-track r = 0.5268 on exactly that configuration. The mechanistic stream has
never actually been tested against a real CDS; now it can be.

### Leakage control on every task

v1 clustered stability only. The HoloGNN training path used `random_split`,
putting mutations of the same protein on both sides. Grouping is now the union
of identity clustering and gene identity, via union-find, for every task.

### One model, not three

v1 trained a separate model per task behind a "unified framework" description.
All heads now share one backbone and one readout, trained jointly with Kendall &
Gal uncertainty weighting.

### The missing baseline

Added `esm_mlp`: a supervised MLP on the same cached embeddings with no graph.
v1's only baseline was ESM zero-shot, which compares supervised against
unsupervised and cannot answer what the graph contributes. That question had no
answer in the repository.

### Uncertainty that means something

The interface's confidence interval now comes from the model's predicted σ,
trained under Gaussian NLL, with coverage and ECE measured by
`scripts/calibration.py`. Until a calibration run exists the interface says the
interval is uncalibrated.

*Why:* v1 computed its "95% CI" as `ddg ± (0.4 + 0.12·|ddg|)` — in **full** mode
as well as demo mode. A fixed-width formula wrapped around a real prediction.

---

## Deliberately left behind

### `make_figures.py` and the published figures

Not migrated. Its default DEMO path synthesised the paper's Results figures from
hardcoded constants — `_STABILITY_R = 0.7644`, a fixed confusion matrix, and
`_demo_pathogenicity_data()` iterating to find class means that make
`roc_auc_score` land on 0.84. The checked-in `images/*.png` behind the README's
results section were those reconstructions, and the repository contained no
metrics file, log or checkpoint that any of the numbers could be traced to.

Figures in v2 are generated from run artifacts only.

### `src/heuristics.py` and demo mode

Deleted. It stood in for the model whenever weights were absent, so the CLI and
the interface returned plausible numbers whether or not a model existed.

### HoloGNN's `ClinVarDataset`

Not migrated. It passed VCF **nucleotide** REF/ALT alleles to a protein language
model, and its docstring said so: *"We interpret the REF and ALT strings directly
as single-letter amino acid token sequences."* For an SNV that is a
one-character "protein", `A` versus `G`. Because A/C/G/T are valid amino-acid
letters the tokenizer accepted it silently — no error, no warning — and the task
could learn nothing beyond marginal base-substitution frequency. Holo-GeoN's
protein-pair ClinVar path is correct and is what survives.

### HoloGNN's `FireProtDataset` accession fallback

Not migrated. When it could not find real sequences it tokenized identifiers
(`"P00698"`, `"P00698_A23V"`) as proteins, reasoning that "the Siamese head only
needs the embedding difference, so relative token positions still carry the
mutation signal". That trains ΔΔG regression on identifier strings.

### v1 results and figures

`results/` is emptied. Those numbers came from a different architecture on
differently-constructed splits; reporting them beside v2 code would attribute
them to a model that never produced them. They remain on `version-1`.

### Version scaffolding

V1–V6 labels, `changelog.md`'s architecture history, and the "§8.2 future work"
framing are gone. This is one codebase, not a changelog.

---

## Other fixes

- **`configs/data.yaml` is now read.** It never was — the guide told users to set
  column names there when detection failed, and the edits did nothing.
- **Checkpoints are self-describing.** `model_kwargs_from_config()` is the single
  place a config becomes constructor arguments, and the result is stored in the
  checkpoint. Found by the new tests: `aa_dim` was a constructor argument nothing
  recorded, so a checkpoint built with a non-default value could not be loaded.
- **The primary metric key was wrong.** `train.py` looked for `pearson_r` where
  `src/metrics.py` emits `pearson`, so the validation score was NaN every epoch:
  nothing ever improved, no checkpoint was written, early stopping never fired.
  A full training run would have ended with no model. A missing key now raises.
- **`MutationParseError` was defined twice**, in two modules, so `except` on one
  did not catch the other.
- **The SSM ran an L-step Python loop.** Replaced with a parallel associative
  scan — 8 vectorised steps at L=256, verified equal to the sequential
  recurrence.
- **Dropout removed from the invariant-bearing heads.** An independent mask per
  call broke exact antisymmetry in train mode; regularisation for those heads
  lives in the shared readout, where it cancels out of the difference.
- **Exclusions are counted.** Every dropped row is attributed to a reason in the
  leakage report. v1 dropped 2.81% of ClinVar for having no cached embedding, and
  that never reached the results.
- **Pathogenicity persists a checkpoint.** v1 reported metrics but saved no
  classifier, so the interface could never serve that head at all.
- **Torch-geometric is a hard requirement.** v1 fell back to a linear projection
  that bypassed the GAT layers entirely when PyG was missing.

---

## Running it on the target machine

Built for an RTX 5070 Ti (16 GB, sm_120), 32 GB RAM, over 2–3 days.

Counted from the constructed model and computed from real tensor shapes
(`python -m scripts.budget`):

- Trainable parameters: 20,607,788 (~79 MiB fp32, ~236 MiB with AdamW states).
- One batch of 64 at window 256 with the 650M encoder: ~80 MiB of node features,
  ~1.23 GB of GATv2 edge tensors and ~2.50 GB of SSM scan buffers, giving an
  analytic **lower bound** of 4.12 GB.
- **There is no peak-memory estimate here.** The first version of this section
  carried one, derived from a `6×` constant multiplier applied to the node
  features; the round-1 review measured that it under-stated the real peak by an
  order of magnitude, and this hardware plan had been written on top of it.
  `scripts/budget.py` now refuses to print an activation figure that is not a
  `torch.cuda.max_memory_allocated()` measurement. Run it on the 5070 Ti to get
  the real number before committing to a batch size; 64 fits the lower bound
  with room, but the lower bound is not the peak.
- Encoder cache: `unique_residues × (2·d + 24·10)` bytes. At 12M residues with
  the 650M encoder that is ~31 GB. `precompute` prints the projection before
  writing and `--max-cache-gb` aborts rather than filling the disk. The 150M
  encoder roughly halves it.

The frozen encoder never enters the training loop, so its weights cost nothing
during training — only precompute and the interface load it.

Install the cu128 wheel first, or the card will error or silently fall back to
CPU.

---

## Adversarial review record

Phase 2 of this migration is a hostile-reviewer audit: a separate agent, given
no stake in the code, executes every claim the repository makes and writes up
what it can break. Each round is committed as `REVIEW_ROUND_<n>.md` so the
findings stay in the history rather than being quietly absorbed into a fix.

The bar for stopping is a round that raises no new BLOCKER or MAJOR — not a
fixed number of rounds.

### Round 1 — `REVIEW_ROUND_1.md` (commit `03d6145`)

**Verdict: reject in current form.** 27 findings: 5 BLOCKER, 15 MAJOR, 7 MINOR.

The five blockers, and why they mattered:

| ID | Finding | Consequence if shipped |
|---|---|---|
| R1-01 | Nothing in the training path was seeded | Two runs of one command differed by 0.030 test Pearson — larger than most ablation effects, so every single-run `delta_vs_full` was uninterpretable |
| R1-02 | Unreachable checkpoint guard | A run could select no checkpoint and still print test metrics, under a flag literally named `validated_by_real_run` |
| R1-03 | `itertools.cycle` over the loader | Retained a whole epoch of batches and re-served one frozen shuffle order |
| R1-04 | Substring matching on ClinVar CLNSIG | `Conflicting_interpretations_of_pathogenicity` and `non-pathogenic` both read as pathogenic |
| R1-05 | The CDS-joined table was written but never read downstream | `mech_valid` 0 everywhere, `E_mech` self-loops only — the `no_mech_stream` ablation would have measured a stream that was never fed, and a ~0 delta would have been written up as "the mechanistic stream does not contribute" |

The reviewer also confirmed what holds, and those parts were deliberately left
alone: exact head antisymmetry in both train and eval mode, swap-equivariance of
the three-state head, padding invariance across heterogeneous batch lengths
(2.98e-07), the parallel associative scan against the sequential recurrence
(≤4.8e-07), mutation windowing at both termini, `_cap_degree` correctness,
chunk-stitching, union-find grouping, the parameter count, and the absence of any
demo-mode path in the interface.

**Fixed in commit `102262c`** — all 27, each named in the commit message.
`tests/test_round1_fixes.py` reproduces the original defect for the findings that
can be exercised without data or a GPU, so none of them can return silently:
seeding determinism, the bounded multi-task objective (the reviewer's probe drove
the total to -1.9e19; it now converges), the CLNSIG vocabulary, full-length
whole-sequence coverage, ddG sign normalisation, unknown-residue handling, the
no-val-split refusal, and cache-manifest CDS coverage.

Three round-1 fixes changed what gets recorded rather than what gets computed,
and they are the ones that matter most for the paper:

- **`cds_coverage` is now a recorded number.** The encoder cache carries a
  manifest, `train.py` copies it into `metrics.json`, and training warns when the
  mechanistic stream is enabled against a cache with zero coverage. A claim about
  E_mech can now be checked against the cache the run actually read.
- **`scripts/budget.py` refuses to print an unmeasured estimate.** See above.
- **`validated_by_real_run` is deleted.** A flag that asserts validation is worse
  than no flag when nothing checks it.

### Round 2 — pending

