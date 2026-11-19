# BENCHMARK_PLAN.md - experiment and benchmark pipeline for Holo-GeoN

Prepared 2026-11-04 on branch `version-2` (`bbd620e`, identical to `origin/main`; nothing to pull).
Read `~/.claude/kaggle-colab-lessons.md` first; the design below follows it.
**Nothing here has been run on a GPU.** Every time and cost is an estimate from earlier
local measurements and is marked ASSUMED until the first remote run replaces it.

---

## 1. Data check (measured on the files in `DATASETS/DATASETS/`)

| file | rows | note |
|---|---|---|
| `data/megascale_ddg_pairs.parquet` (10 MB) | 389,068 | 412 parent constructs; columns `wt_seq, mt_seq, ddG, ddg_std, wt_dna, parent, mut_type`; real CDS present (`wt_dna`) |
| `data/splits_megascale.parquet` (7 MB) | 389,068 | train 305,995 / val 49,847 / test 33,226; 322 / 53 / 37 proteins; MMseqs2 30% identity; near-duplicate audit: 0 of 90 held-out proteins leak |
| `data/fireprotdb_holdout.parquet` (0.3 MB) | 4,233 | 176 proteins, no MegaScale rows; raw mean ddG +1.004, range -13.7 to +23.2 |
| `data/leakage_report_stability.json` | | splits made at commit `dbd1546` on the RTX 5070 Ti machine |
| `data/true_contacts/` | 373 maps | only for the contact-precision check |
| `dataset/mega_scale_cdna/AlphaFold_model_PDBs/` | 862 PDBs | the structures competitors need; all 35 test PDB ids present |

Findings that change how the runs are set up:

1. **The folder is nested**: the data is at `DATASETS/DATASETS/{data,dataset}`, and `run.yaml` reads `data/...`
   from the repo root. The runner finds inputs by name and links them into `data/`.
2. **Upload 3 files (17 MB), not 47 GB.** The ClinVar table (0.6 GB) plus its splits (0.4 GB) and the
   `dataset/` sources (UniRef50 12 GB, massive_kb 27 GB) are not needed for a ΔΔG benchmark.
3. **Stability-only.** `configs/benchmark.yaml` sets `clinvar_data: null` and the runner passes
   `--pathogenicity-splits none`; `scripts/train.py` already drops a task whose split file is absent. This
   skips ~15k extra parent proteins (25 GB cache, hours of encoding) and 41.6% of the training steps, and it is
   OVERFITTING_FIX.md B2's cheapest hypothesis test. The consequence: **these runs cannot be compared with the
   existing `runs/main` checkpoint** (r 0.7582 / 0.4518), which was trained with ClinVar.
4. **No encoder cache is shipped.** `run.yaml`'s `cache: C:/Users/User/...` is a Windows path on another
   machine. The stability-only cache covers 412 + 176 proteins (< 0.3 GB); encoding is the `precompute` item.
5. **Config precedence is DEFAULTS < run.yaml < data.yaml < CLI**, and `DEFAULTS` differ from the tuned values
   (`epochs 60`, `weight_decay 1e-4`, `patience 8`). `configs/benchmark.yaml` therefore restates every tuned
   value (A1 in OVERFITTING_FIX.md: 12 epochs, patience 4, weight decay 0.01).
6. **Sign convention is already declared** (`fireprotdb_holdout: ddg_sign: destabilizing_positive`); after
   normalisation the training mean is -0.76, as expected.
7. **`parent` includes background variants** (`1A0N.pdb_L7S` is a construct that already carries one
   substitution). Test split: 37 proteins, of which 33 are plain PDB entries (28,466 rows). Competitor models
   with structure input are scored on that **native subset** (`--subset native`).
8. Dependencies: `transformers>=4.40,<5` (ESM contact extraction), `torch_geometric`, `ViennaRNA` (without it
   `mech_valid = 0` and the `no_mech_stream` arm measures nothing). `mmseqs` is not needed: splits are shipped.

## 2. Benchmark models (late 2026)

### What this benchmark claims, and what it does not

Holo-GeoN takes **no 3D coordinates**. `E_geo` is built from ESM-2's contact head
(`src/graphs.py`), and its edge features are contact probability, log-scaled `|i - j|` and a
backbone flag: all three are functions of sequence position. The only equivariance in the code
is wild-type/mutant **swap** equivariance in `ThreeStateHead` (`src/heads.py`), a two-element
permutation symmetry. **Nothing here is SE(3)- or E(3)-equivariant, and no baseline below is
either**: ThermoMPNN is invariant message passing on ProteinMPNN features, SPURS is a frozen
ESM-2 with a ProteinMPNN adapter, RaSP is a 3D CNN, and PROSTATA, StabilityArc and the control
are sequence-only. The only equivariant ΔΔG model we found is Boyer, Money-Kyrle and Bent,
arXiv:2305.19801 (30 May 2023), E(3)-equivariant and trained on MegaScale - but it has no code, no
released weights and no stated venue, and its abstract reports only "promising initial results".
So no equivariant comparator can be run off the shelf.

**Coordinates are nonetheless available**, and `GEOMETRIC_PLAN.md` costs out using them: 862
AlphaFold backbones cover 37/37 test parents and 100% of test rows. Its recommendation is an
**invariant** coordinate path (ΔΔG is a scalar, so distances plus local frames are a complete
basis; the ΔΔG state of the art is invariant) added as a controlled ablation arm, with Holo-GeoN
still shipping as the structure-free model. Adding coordinates would not license the word
"equivariant" either, so the paragraph above stands whichever way that goes.

The claim this benchmark can support is therefore:

> a **structure-free** contact-graph GNN, given only a sequence, reaches the accuracy of
> predictors that are given backbone coordinates.

That is the stronger claim anyway, and the input asymmetry below is its evidence rather than its
weakness. "Geometric" in this repo names the contact-derived topology of `E_geo`, not Euclidean
geometry; say so once in the paper, and never write "SE(3)" or "geometric graph neural network"
about Holo-GeoN.

### Selection

Selection rule: open weights or a public training script, a ΔΔG scoring path, a published
MegaScale protocol. Two Sonnet agents surveyed papers, GitHub pages and API metadata, twice; the
second pass refuted three details of the first (PROSTATA's architecture, StabilityArc's standing,
SPURS's year). Facts below are **agent-reported** and several primary pages refused fetches, so
re-check every number against the paper before citing it.

**Pooled and per-protein Spearman are not comparable.** ThermoMPNN's 0.725 is pooled over 28,312
mutants; SPURS's 0.83 is a median over 28 proteins. Keep them in separate columns in the paper,
and report both for every model we run.

| # | model | repo; licence; weights | input | published number (their split, NOT ours) | use here |
|---|---|---|---|---|---|
| 1 | **ThermoMPNN** (PNAS 2024, `10.1073/pnas.2314853121`) | `Kuhlman-Lab/ThermoMPNN`; MIT; `models/thermoMPNN_default.pt` in repo | backbone PDB | MegaScale test, **pooled** (28 proteins, N=28,312): Pearson 0.754 / Spearman 0.725 / RMSE 0.708. FireProt-HF (N=2,578): 0.650 / 0.657 / 1.51 | field-standard structure-based reference; retrain on our train split; official weights on FireProtDB only |
| 2 | **SPURS** (Nat. Commun., `10.1038/s41467-025-67609-4`; **year unconfirmed** - the journal page would not open, so "2025" may be 2026) | `luo-group/SPURS`; MIT; HF `cyclization9/SPURS` (card empty), also `data/checkpoints` in a `data.tar.gz` | PDB + chain (AlphaFold ok) | **per-protein median** Spearman 0.83, vs 0.77 for a ThermoMPNN retrained on SPURS's 25%-filtered train set. No pooled Pearson/RMSE found | ESM-2 + structure reference on the same protocol; same backbone family as ours |
| 3 | **RaSP** (eLife 2023) | `KULL-Centre` RaSP release (confirm the exact repo before use); weights public | structure (3D CNN over a local environment) | MegaScale retrain by the ThermoMPNN authors: Pearson 0.71 / Spearman 0.67 / RMSE 1.08; FireProt-HF Pearson 0.47 | peer-reviewed, public weights, cheap to run. **Promoted into the table** (see below): a published structure-based method is worth more here than a workshop preprint |
| 4 | **PROSTATA** (Bioinformatics, Nov 2023; `PMC10651431`) | `AIRI-Institute/PROSTATA`; Apache-2.0; **weights not clearly downloadable** - the README offers training notebooks, a Colab tool and a web server, not checkpoints | sequence only: **one** ESM-2 650M backbone fine-tuned end to end with **five regression heads averaged** (not an ensemble of five models) | retrained on MegaScale by the ThermoMPNN authors: Pearson 0.64 / Spearman 0.59 / RMSE 0.83; FireProt-HF 0.59 / 0.55 / 1.68. Its own paper reports S669 Pearson 0.49 / RMSE 1.45 and **no Spearman** | closest input match to Holo-GeoN; retrain on our split. Its own training mix (S2648 / Q3214 / VariBench plus 5,251 MegaScale miniproteins) is not clean against our test split, so retraining is required, not optional |
| ctl | frozen ESM-2 650M + MLP | in this repo (`scripts/baselines.py`) | sequence | none published that we could verify | same-input control on the identical split and budget |
| opt | **StabilityArc** (arXiv 2610.00742) | `AaronFeller/StabilityArc`; MIT; HF `aaronfeller/StabilityArc`; **3 commits, 0 stars** | sequence only (frozen ESMC-600M + 4-layer RoPE transformer) | ProteinGym stability leave-one-protein-out Spearman 0.7134 vs 0.6526 for ProSST-2048 - from **held-out checkpoints that were never released** | **Demoted to optional, retrain only.** Reasons below |

**Why StabilityArc moved to optional and RaSP moved in.** StabilityArc is real (Feller, Ellington,
Wilke; verified on arXiv, GitHub and Hugging Face) but it is a **NeurIPS 2026 workshop poster**,
not a main-track or journal paper, on a repo with three commits. Its output is a ranking score and
the README states it is not kcal/mol, so **RMSE is undefined for it** and only Spearman is
meaningful. Its released checkpoint saw all 66 ProteinGym stability assays, 64 of which are
MegaScale, so that checkpoint cannot be scored anywhere near our test split. Citing it as the
"latest sequence-only benchmark" would not survive review, while excluding peer-reviewed RaSP on
the grounds that it is Rosetta-trained applied a stricter bar to a stronger paper. Include
StabilityArc only if its retrained head fits the budget after the four above, and if it appears,
report Spearman only.

Checked and left out: **Stability Oracle** (its cDNA117K training set is MegaScale-derived and
leaks; no MegaScale test numbers), **ProteinMPNN-ddG** (zero-shot, no Pearson/RMSE),
**JanusDDG** (arXiv 2504.03278; trained on S2450, ProTherm lineage, likely overlaps FireProtDB;
no MegaScale evaluation; fallback only), **SaProtΔG / ESM3ΔG** (structure input; 9 of 28
MegaScale test wild types have >30% identity to its training set; also capped at 60-80 aa),
**ThermoMPNN-D** (double mutants), **ESM-MSR** (multi-mutant focus) and **StaB-ddG**
(ICML 2025, arXiv 2507.05502; needs coordinates / binding). SaProt and ProtSSN stay zero-shot
ProteinGym context only (ProtSSN is CC-BY-NC-ND, no derivatives). A zero-shot ProteinMPNN or
ESM-IF1 log-odds column is the standard cheap control and should be added if any structure path
is built.

**Traps, and the protocol that avoids them**

* **Our 0.758 is not comparable to ThermoMPNN's 0.754.** Different split (30% vs 25% identity,
  37 vs 28 test proteins, our table has 412 constructs vs their 298 proteins). Quote published
  numbers only as context; the comparison table uses models run by us on our split.
* **Pooled vs per-protein.** Report both for every model we run, in separate columns, and never
  place SPURS's published 0.83 beside ThermoMPNN's published 0.725.
* **Leakage of official weights.** ThermoMPNN's checkpoint was trained on its own MegaScale split,
  and our test proteins were not chosen against that split. So:
  * MegaScale **test**: competitors are **retrained on our train split** (trained heads only,
    backbone frozen); official weights are not scored there.
  * **FireProtDB external**: all models train on MegaScale only, so official weights (ThermoMPNN,
    SPURS, RaSP) are a fair comparison. Caveats: our holdout removed MegaScale-sourced *rows* but
    is not homology-filtered against MegaScale (ThermoMPNN's FireProt-HF is, at 25%), and
    FireProtDB 2.0 also integrates Human Domainome and ProTherm rows. These apply equally to every
    model, but absolute numbers will differ from published ones. Structures for the 176 UniProt
    proteins are fetched by `scripts/fetch_alphafold.py`, the one network step in the pipeline.
  * The field's current convention is a 25% identity filter between train and every test set.
    Ours is 30%; state the difference rather than implying parity.
* **Input asymmetry.** Holo-GeoN uses sequence only; ThermoMPNN, SPURS and RaSP get a structure
  (the release's AlphaFold models; native subset only). This is the point of the paper, not a
  footnote: state it in the table caption and in the abstract.
* **ESMC-600M licence** (only if StabilityArc is included). The `esm` repo's LICENSE.md is MIT
  (Chan Zuckerberg Biohub, 2026) and the model cards carry both `license: mit` and
  `license: other` plus an Acceptable Use Policy whose text we could not retrieve. Read the
  weight-specific terms before relying on it.
* **Not written yet: the competitor adapters.** The scorer, the prediction format (`row_id`,
  `y_pred`) and the manifest slots exist. An adapter per model (clone, train head on our train
  rows, predict test + FireProtDB) must be written and validated on a GPU; its head-training cost
  is unmeasured. **This, not the GPU budget, is the critical path**: four third-party repos with
  their own pinned `transformers` versions is the single largest unestimated piece of work in this
  plan, and the margin in section 4 is a reservation against it, not a plan.

All models are scored by one script, `scripts/score_predictions.py`: common rows only, declared
sign per model (it refuses a negative r after applying it), cluster bootstrap over the MMseqs
clusters (`group_used`) for every interval and for the paired difference to Holo-GeoN.
ThermoMPNN/RaSP/FireProt report destabilising-positive ΔΔG (`:-1`), Holo-GeoN
stabilising-positive (`:+1`).

## 3. Scaffolding added

| file | purpose |
|---|---|
| `configs/benchmark.yaml` | stability-only run config |
| `scripts/remote_job.py` | resumable runner: items = precompute / train / external / baselines / cmd / aggregate; packs + DONE markers on persistent storage, heartbeat, live-log copy, `report.json` (torch/CUDA/transformers/GPU/commit), input discovery by name, fail-fast preflight, `--plan`, `--shard K/N`, `--max-hours`, `--aggregate-only` |
| `configs/experiments/colab_g4_benchmark.yaml` | headline: Holo-GeoN x 5 seeds + external + baselines (+ competitor slots) |
| `configs/experiments/kaggle_ablation.yaml` | ablation grid: full, no_mech_stream, global_pool_readout, no_geo_stream x 3 seeds, 3 shards |
| `scripts/export_predictions.py` | per-row Holo-GeoN predictions in the common format |
| `scripts/fetch_alphafold.py` | the one network step: 176 AlphaFold DB structures for the FireProtDB holdout, resumable, pinned by sha256, validated per file. Run it locally, not on a paid VM |
| `scripts/score_predictions.py` | the single scorer (above) |
| `remote/kaggle_kernel.py`, `remote/kernel-metadata.json`, `remote/colab_run.sh` | platform launchers |
| `tests/test_remote_job.py` | 52 checks in 6 test functions (item order, seed-blocked shards, resume, input discovery, plan, scorer); added to `run_all`. **Pure-Python plumbing: no GPU, no data, no subprocess** - passing them says the runner's bookkeeping is right, not that the pipeline trains |

Ablation metric family is declared before any run (`ablation_metrics: stability: pearson`) and the deltas use the
existing paired-t + Holm code (`scripts/ablations.py`), so the scaffolding adds no new statistics.

## 4. Sizing

### Colab G4, 100 units (8.9 CU/h from earlier sessions => 11.2 GPU-h; keep a 10 CU reserve)

One run = 4,782 steps/epoch x 12 epochs. **The per-step time is the one unmeasured number**: the local RTX 5070 Ti
measured 33.8 ms; 40 ms is assumed for G4 + a Colab CPU feeding 4 loader workers.

| s/step | 1 run | main matrix (precompute + 5 seeds + external + baselines) | usable margin of 90 CU |
|---|---|---|---|
| 0.030 | 0.53 h | 3.6 h = 32 CU | +58 CU |
| **0.040 (assumed)** | **0.70 h = 6.2 CU** | **4.5 h = 40 CU** | **+50 CU** |
| 0.060 | 1.05 h | 6.3 h = 56 CU | +34 CU |
| 0.100 | 1.76 h | 9.8 h = 87 CU | +3 CU |

The margin is what competitor retraining and evaluation must fit in (not yet costed: nobody has timed ThermoMPNN or
SPURS head training on this split; measure one before committing).

Waves, re-planned between each (`--plan` re-projects from the measured median):

1. **Wave 0 (~1.2 h, ~10 CU):** `MAX_HOURS=1.5 bash remote/colab_run.sh run` = precompute + `full_s42` + its external eval.
   Read `s/step`, `best_epoch`, test r. Stop here if r is far from ~0.75 (stability-only differs from the old run).
2. **Wave 1:** seeds 43-44 and baselines.
3. **Wave 2:** seeds 45-46, then competitors. Cut order if over budget: competitor retrains, `*_s46`, `*_s45`, baselines.

Stop the VM as soon as the job ends. The unit balance cannot be read from inside the VM, so the guard is
`--max-hours` (= usable CU / 8.9) plus the persisted packs.

### Kaggle T4 (free, ~30 GPU-h/week/account, 2 concurrent sessions)

**Deferred.** Both GPU notebook slots are in use on other work, so the grid waits until the user
says the weekly quota is free. The config is ready and re-budgeted; nothing is queued.

Ablation grid, **6 epochs** (was 8), strict determinism (x1.3), 12 runs, sharded by seed.

| s/step | per run | per shard (4 arms of one seed) | grid of 12 | of a 30 h week |
|---|---|---|---|---|
| 0.100 (ASSUMED) | 1.14 h | 5.6 h | 16.7 h | 56% |
| **0.117 (from the only T4:G4 ratio we have)** | **1.33 h** | **6.3 h** | **19.0 h** | **63%** |
| 0.150 | 1.71 h | 7.8 h | 23.5 h | 78% |

The 0.117 row applies the lessons file's 72.5 ms T4 vs 24.7 ms G4 (~2.9x) to the G4 assumption of
40 ms. **At the old 8 epochs the 0.150 case was 30.4 h - over the whole week's quota, and 10.1 h a
shard, past the 8.5 h safe session length.** Six epochs is justified independently
(OVERFITTING_FIX.md A1: val Pearson saturates by epoch ~6) and leaves >=6 h for one retry, which
the lessons file says to expect. Measure the step with a short probe before committing the grid.

### Not sized: ESM-2 weights download
~2.6 GB from the Hub on each fresh VM; needs internet on Kaggle.

## 5. Decisions and prerequisites for the user

1. **Commit and push** the scaffolding. Nothing is committed yet. Kaggle/Colab need the code from a
   branch or a code dataset. The 47 GB `DATASETS/` tree is already covered by `.gitignore`
   (`data/`, `dataset/`, `*.parquet`) and `git status` confirms none of it is staged, so a commit
   here cannot carry the data into GitHub.
2. **Measure the step time** before anything is sized for real: one short probe (a few hundred
   steps) on each GPU type, then `--plan` again. Every hour and unit in section 4 hangs off this
   one number, and the G4 and T4 values differ by ~3x.
3. Colab: sign in and Drive consent are yours (OAuth flow); copy the 3 parquet files and the repo
   to `MyDrive/holo_geon/`.
4. **Kaggle is deferred** until the user says the weekly quota is free. When it is: create the
   `holo-geon-data` dataset from the 3 parquet files and fill in the username in
   `remote/kernel-metadata.json`. Nothing else about the Kaggle path is outstanding.
5. Fetch the FireProtDB structures, locally and once, before any competitor runs on the external
   set. It needs no GPU:

       python -m scripts.fetch_alphafold --holdout data/fireprotdb_holdout.parquet \
           --out data/alphafold_fireprot

   176 accessions, resumable, writes `manifest.json` with a sha256 per file and refuses to finish
   quietly if any are missing. Copy `data/alphafold_fireprot/` to Drive beside the parquet files.
6. **Write the competitor adapters.** This is the critical path, not the GPU budget: four
   third-party repos, each with its own pinned `transformers`, each needing a head trained on our
   train rows. Approve the retrain budget once one of them has been timed.
7. **Decide the ClinVar question.** These runs are stability-only, so none of them is comparable
   with the existing `runs/main` checkpoint (r 0.7582 / 0.4518), which was trained with ClinVar.
   Wave 0's stop criterion ("stop if r is far from ~0.75") therefore compares against a
   differently-trained model and is a soft gate, not a decision rule.
