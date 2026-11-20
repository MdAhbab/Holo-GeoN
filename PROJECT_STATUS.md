# PROJECT_STATUS.md

## 0. Update — 2026-11-20 (`version-2` active branch — Blocks A/B/C Code, Figures, Block D Full-Stack v2 & Epistasis ETL Ready)

* **Workspace scaffolded** into `project/` (Git repository only + gitignored `DATASETS/`), `paper/`, and `extras/` (non-Git notes, Zenodo staging, Kaggle account outputs).
* **Git branches:** `main` == `origin/main` at `f083217`. Active branch **`version-2`** (`origin/version-2` at `3da9bc3`, plus local verified commits `bc8fd66` figures/CIs and `6ae4c82` Block D backend v2, NO push yet). Legacy v1 (`arch: "v1"`, 20.6M params) remains selectable alongside Holo-GeoN v2 (`arch: "v2"`, 2.04M params) for paired ablations. **No trained v2 weights or v2 accuracy results exist yet.**
* **Block A — Holo-GeoN v2 Core & Efficiency (DONE & VERIFIED):**
  * Single-pass `EsmForMaskedLM` encoder (`src/encoder.py`), compressed `HoloBackboneV2` (**2,038,417 params** vs. v1 **20,607,788 params** = **10.11x smaller**), `EvolutionarySubstitutionHead`, `ContactGatedEpistasisHead`, `beta-NLL` (`beta = 0.5`), `ProteinBatchMutationDataset` + `_ParentGraphCache`, `cds_dropout=0.5`, SE(3)-invariant 3D `src/geometry.py`, full cluster-bootstrap CIs in `scripts/score_predictions.py`, and PIT curves in `scripts/calibration.py`.
* **Block B — Cross-Platform CLI & 5 Parallel Kaggle CPU Audits (DONE, 0 GPU HOURS USED):**
  * Non-Linux-friendly CLI (`holo.py` / `holo.bat` / `holo.ps1` / `pyproject.toml`) + private `holo-geon-v2-gpu-seed` dataset live & byte-identical on all 4 Kaggle accounts.
  * **Landmark Audit Finding #1 (MegaScale Structural Fold Collapse — `cpu1` + `cpu5`, 69,378 exact TM-align pairs):** On standard `id30`, **37/37 (100%)** structured test domains share a fold (`TM_max >= 0.5`, median `0.82`) with train (`27/37` `TM_max >= 0.7`). Even on `id25`, **40/40 (100%)** reach `TM_max >= 0.5` (median `0.72`). At `TM_max >= 0.50`, **97.05% (362/373)** collapse into one connected component, so no split disjoint at `TM_max >= 0.5` exists. Uniting `TM_min >= 0.60` links with `id30` sequence clusters yields **76 fold clusters** (`TM_min` alone = 78) -> **`splits_megascale_fold_tm60.parquet`** (**fold-clustered, NOT fold-disjoint**: `max_cross_tm_min = 0.5956`, median `TM_max = 0.62`, 37/39 test domains still `TM_max >= 0.5`).
  * **Landmark Audit Finding #2 (Hidden External Homology Leakage — `cpu2` + `cpu4`):** Even after removing all 604,920 MegaScale rows from FireProtDB 2.0, **68/176 proteins (55.3% of rows)** still have `>= 25%` identity to MegaScale -> **`fireprotdb_hf25.parquet`** (1,894 rows, 108 proteins) + **`splits_megascale_id25.parquet`** (zero 8-mer overlap). On Human Domainome 1.0 (522 domains, 536,164 variants), **195/522 (37.4%)** have `>= 25%` identity -> 100%-WT-verified **`domainome_apca_hf25.parquet`** (332,854 rows, 327 domains).
  * **Precomputed Benchmark Assets (`cpu3` + `cpu4`):** 166/176 SHA-256-pinned AFDB FireProtDB PDBs, S669 (`669/669` WT-PDB verified, 94 proteins), 412/412 frame-verified real-CDS ViennaRNA/CAI features, and 538 aligned SE(3)-invariant backbone tensors (`invariant_backbone_geometry.npz`).
* **Block C Scaffolding, Figures, Epistasis ETL, PDB Recovery & Zenodo `Holo-StabBench v1.0` (READY FOR GPU RUNS):**
  * 5 competitor adapters (`scripts/competitors/`), `scripts/train_epistasis.py`, streaming `scripts/build_epistasis_table.py`, `scripts/error_decomp.py`, reproducible figure pipeline (`scripts/make_figures.py`; F2 & T1a/T1b built on real release), and upgraded `scripts/zenodo_pack.py --local-processed-dir` (covers ClinVar+MANE 190k real-CDS pairs/splits + `true_contacts/` + MegaScale doubles alongside rc1 artifacts).
  * **PDB `| / : -> _` Stem Normalization (`src/geometry.py`):** Added `normalize_pdb_stem` / `resolve_pdb_path` across `attach_coords.py`, `build_contact_maps.py`, `build_epistasis_table.py`, and `scripts/competitors/{thermompnn,spurs,rasp}.py`, recovering the 37 MegaScale constructs (`EA|...`, `GG|...`, `XX|...`) previously missed on Windows/Linux due to `|` vs `:` vs `_` naming (~410/412 coverage vs 373/412).
  * **Active Free Kaggle CPU ETL Wave 2 (`cpu6`/`cpu7`/`cpu8`/`cpu9`, 0 GPU hours):**
    * `cpu6` (`mdahbabhamidkhan/holo-geon-cpu6-mgnify-hf25`): Zenodo `19411306` Rocklin Lab MGnify Stability (2.23 GB) -> streaming clean + MMseqs2 `<25%` + exact/substring anti-leakage filter -> `mgnify_stability_clean_hf25.parquet` (strictly firewalled Regime-2 pretraining only).
    * `cpu7` (`mhmehedi26/holo-geon-cpu7-s1724-rocklin`): Zenodo `19411306` `benchmarks.zip` + `figures.zip` -> `s1724_thermomutdb_holdout.parquet`, `s1724_thermomutdb_hf25.parquet` (`<25%` to MegaScale), and Rocklin 2026 Fig3/Fig4 reference predictions.
    * `cpu8` (`wali0754/holo-geon-cpu8-epistasis-7844779`): Zenodo `7844779` complete 39-col `Processed_K50_dG_datasets.zip` + `Data_tables_for_figs.zip` -> `megascale_double_mutants_{id25,fold_tm60,id30}.parquet` with recovered `|/: -> _` PDB `dist_cb` distances + Fig4 cross-check.
    * `cpu9` (`mdahbab/holo-geon-cpu9-geom-contacts-v2`): Rebuilds `invariant_backbone_geometry.npz` and `true_contacts_megascale.tar.gz` with `|/: -> _` PDB normalization (~410/412 MegaScale + 165 FireProtDB) + TM-align audit of the recovered constructs across `id30`/`id25`/`fold_tm60`.
  * Hardened Wave 0 T4 probe (`3da9bc3`, `schema_version: 2`) staged at `../extras/kaggle_accounts/_gpu_kernels/wave0_t4_probe/` — waiting on `2026-11-21 00:00 UTC` Kaggle GPU quota refresh.
  * Confirmed primary split = **`id25`** paired with the 3-tier honesty ladder (`id30 -> id25 -> fold_tm60`) + external holdouts (`FireProtDB-HF25`, `S669`, `Domainome-HF25`, `S1724-HF25`).
* **Block D — Full-Stack Web App v2 (Backend + Frontend Source Done, `tsc --noEmit` 0 errors):**
  * Backend (`app/backend/`, `app/API_V2.md`) and React+TS frontend (`app/frontend/src/`: `EpistasisTab.tsx`, optional CDS side-by-side & `structure_free`/`invariant_3d` mode toggle in `DdgTab.tsx`, SSE streaming heatmap/progress in `ScanTab.tsx`, 3Dmol AlphaFold pLDDT bands in `CompareTab.tsx`, `epistasis` in `ExportTab.tsx`/`HistoryTab.tsx`). Build & live demo deferred to Kaggle CPU / G4.

---

State of the Holo-GeoN benchmark as of **2026-11-04**, on `main` (`bbd620e` plus uncommitted work — now committed as `f083217`).
Companions: `BENCHMARK_PLAN.md` (the benchmark), `GEOMETRIC_PLAN.md` (the coordinate question),
`RUN_STATE.md` (the previous machine's completed run), `OVERFITTING_FIX.md`.

**Nothing has been committed. No GPU hour has been spent on this machine. No per-step time has
been measured on either target GPU.** Every hour and compute-unit figure anywhere in this repo is
a projection off one unmeasured number.

---

## 1. The one finding that should drive every decision

From `RUN_STATE.md` section 4, **already measured** on the previous machine:

> `esm_mlp_matched_capacity` - same frozen embeddings, same parameter count, **no graph** - scores
> **r 0.7537 / RMSE 0.6415 / R2 0.5550** against the full model's **0.7582 / 0.6482 / 0.5457**.

The graph, the site-anchored readout, the SSM, the cross-attention fusion and the mechanistic
channels together contribute **about 0.0045 Pearson, and the MLP is better on both error
metrics**. Transfer is also weak: r falls 0.758 -> 0.452 on FireProtDB.

This is a measurement, not a worry, and it reframes the project:

* The ablation grid's job is to **attribute** a contribution that is already known to be ~zero.
  Expect small effects, and pre-commit to writing a null as "cannot tell at 3 seeds" (the
  detectable effect is ~0.02 Pearson).
* **More training data does not address this.** Neither does a bigger architecture. The honest
  paper is either (a) the structure-free input-requirement claim with the competitor table, or
  (b) a negative result about graph inductive bias on frozen PLM embeddings.
* It is the strongest argument for `GEOMETRIC_PLAN.md`'s **arm E**: if real contact edges do not
  beat predicted ones either, that closes the question cheaply.

---

## 2. Done today

### Reframing (the SE(3) claim was false)

`README.md` line 3 already said "structure-free"; the benchmark framing did not match it. The
`E_geo` stream is built from ESM-2's contact head, with edge features (contact probability,
log `|i-j|`, backbone flag) that are all functions of sequence position. The only equivariance in
the code is wild-type/mutant **swap** equivariance in `ThreeStateHead`, a two-element permutation
symmetry.

* `BENCHMARK_PLAN.md` section 2 now opens with what the benchmark can and cannot claim, states
  that neither the model nor any baseline is SE(3)- or E(3)-equivariant, and fixes the defensible
  claim: *a structure-free contact-graph GNN, given only a sequence, reaches the accuracy of
  predictors given backbone coordinates.*
* `README.md` gained a "No 3D coordinates anywhere" note; "spatial edge set" became
  "predicted-contact edge set".

### Baseline table corrected

| change | why |
|---|---|
| **RaSP promoted in** (eLife 2023, public weights, MegaScale retrain 0.71/0.67/1.08) | peer-reviewed; it had been excluded while a workshop preprint was included |
| **StabilityArc demoted to optional** | real (arXiv 2610.00742, Feller/Ellington/Wilke) but a **NeurIPS 2026 workshop poster**, repo at 3 commits / 0 stars, output is a ranking score not kcal/mol so **RMSE is undefined**, and its released checkpoint saw all 66 ProteinGym stability assays (64 MegaScale) |
| **PROSTATA corrected** | it is **one** ESM-2 650M backbone with **five regression heads averaged**, not an ensemble of five models. Weights are not clearly downloadable (README offers notebooks, a Colab tool and a web server) |
| **Pooled vs per-protein split into separate columns** | SPURS's 0.83 is a per-protein median over 28 proteins; ThermoMPNN's 0.725 is pooled over 28,312 mutants. They were about to share a column |
| SPURS year flagged unconfirmed | the Nat. Commun. DOI resolves in the Nature Index but the journal page would not open; "2025" may be 2026 |
| Equivariant-baseline claim weakened | **my own error.** I had written that no equivariant ΔΔG baseline exists. One does: Boyer, Money-Kyrle & Bent, arXiv:2305.19801, E(3)-equivariant on MegaScale. Verified on arXiv: no code, no weights, no stated venue, abstract claims only "promising initial results". Conclusion unchanged - nothing runnable off the shelf - but the wording was too strong |

### Sharding fixed (`scripts/remote_job.py`)

`take_shard` was round-robin over runs (`j % n == k`), so `--shard 0/3` held
`full_s42, no_geo_s42, global_pool_s43, no_mech_s44` - **no shard contained a complete seed**, and
one lost session cost the paired test. The config comment claimed the opposite, and the test only
checked the partition.

Now it blocks by seed: `--shard k/3` is seed 42+k with all four arms, so one finished shard is
already a complete paired block for `scripts/ablations.py`. More shards than seeds is refused with
a message pointing at epochs/seeds. Verified on all three shards; the test now asserts
one-seed-per-shard, all-arms-per-shard, and the refusal.

### Kaggle grid re-budgeted (`configs/experiments/kaggle_ablation.yaml`)

8 -> **6 epochs**, justified independently by `OVERFITTING_FIX.md` A1 (val Pearson saturates by
epoch ~6).

| s/step | per run | per shard | grid of 12 | of a 30 h week |
|---|---|---|---|---|
| 0.100 (ASSUMED) | 1.14 h | 5.6 h | 16.7 h | 56% |
| 0.117 (from the only T4:G4 ratio we have) | 1.33 h | 6.3 h | 19.0 h | 63% |
| 0.150 | 1.71 h | 7.8 h | 23.5 h | 78% |

At the old 8 epochs the 0.150 case was **30.4 h - over the whole weekly quota**, with 10.1 h
shards past the 8.5 h safe session length. The file is marked NOT READY TO LAUNCH with its two
prerequisites; nothing is queued.

### `scripts/fetch_alphafold.py` added

The pipeline's only network step, and the thing blocking any competitor's FireProtDB score.
Resolves all 176 holdout accessions through the AlphaFold DB API rather than a hard-coded
`-model_v4.pdb`, resumable, validates that each download is a PDB containing that accession, pins
sha256 in `manifest.json`, writes `fetch_log.jsonl`, exits non-zero on any miss.

Verified offline (validator accepts/rejects correctly; 176 accessions resolve; missing-input path
exits 2). **The real download has not been run** - one command, no GPU:

```bash
python -m scripts.fetch_alphafold --holdout data/fireprotdb_holdout.parquet --out data/alphafold_fireprot
```

### Verified, not assumed

Measured off the files and independently re-checked:

| fact | value |
|---|---|
| MegaScale parent length | 32-72 aa, mean 52.6, median 51 |
| `window: 256` | **never binding** - 0 of 412 parents exceed it |
| test-split structure coverage | **37/37 parents, 33,226/33,226 rows (100%)** |
| train / val structure coverage | 93.1% / 71.6% of rows |
| parents without structure | 39: `EA*` 10, `GG*` 13, `XX*` 14 (all 47 aa), 2 `v*` (66, 70 aa) - **23 train, 16 val, 0 test** |
| structured parents | 373 of 412, matching the 373 maps in `data/true_contacts/` |
| the 3 upload files | 17.4 MB total |
| AlphaFold PDBs on disk | 862 |
| `.gitignore` vs the 47 GB `DATASETS/` | covered by `data/`, `dataset/`, `*.parquet`; `git status` confirms nothing staged |

**The length finding cuts the other way on cost**: the 33.8 ms/step local measurement came from a
ClinVar-inclusive config whose proteins reach 256 residues, so 40 ms/step on the G4 is probably
**pessimistic**. Wave 0 may re-project the budget *downward*.

### Tests

`scripts/remote_job.py`: **52 checks in 6 functions, all passing.** They are pure-Python plumbing -
no GPU, no data, no subprocess - so they say the runner's bookkeeping is right, not that the
pipeline trains. `BENCHMARK_PLAN.md` now says so.

Full suite: **348 checks, 6 failed.** All 6 failures are in the web backend and **pre-existing**:
the engine loads a structurally valid synthetic checkpoint but reports not-ready with
`error: None`, so `/api/ddg` and `/api/scan` return 503. Reproducible in isolation with and
without `-I`; none of today's files are in `app/backend`'s import graph. Note that
`pytest tests/test_app.py` reports "2 passed" because pytest collects only 2 of that module's 10
functions - **`python tests/run_all.py` is the real gate in this repo.**

---

## 3. Planned, not started

### A. Competitor adapters - the critical path

Four adapters (ThermoMPNN, SPURS, RaSP, PROSTATA), each cloning a third-party repo with its own
pinned `transformers`, training a head on our train rows, and writing `row_id, y_pred`. The
scorer, the format and the manifest slots exist; **no adapter does.** Realistically 4-8 days with
the project's largest variance. The competitor table *is* the paper; the GPU budget is not the
constraint.

### B. The coordinate arm - see `GEOMETRIC_PLAN.md`

Recommendation: an **invariant** coordinate path, explicitly **not** EGNN/GVP/e3nn/Equiformer.
ΔΔG is a scalar, so distances plus local frames are a complete basis (Villar et al.,
arXiv:2106.06610); equivariance buys sample efficiency at low data, and we have 306k rows; the
ΔΔG state of the art is invariant (ThermoMPNN is ProteinMPNN's RBF distances in local frames).

New `src/geometry.py` (parser factored out of `build_contact_maps.py`), a third
`build_structure_graph` stream (16 RBF bins CB-CB, CA->CA unit vector in residue *i*'s frame,
quaternion of `R_i^T R_j`), coordinates attached to the existing cache by a separate CPU script
(a flag on `precompute_embeddings.py` would trigger a needless ESM-2 re-encode of 588 proteins).
Five arms: A contacts / B coordinates / C both / D coordinate-topology-only / E **real contact
edges**. ~3 focused days. **Do not start until one competitor adapter works end to end.**

Hard prerequisite it creates: a **Foldseek or TM-align leakage audit**. Our split is
sequence-identity based at 30%, fold similarity survives below that, and the existing MinHash
8-mer audit cannot see fold similarity at all - so coordinates open a leakage route the current
audit does not detect.

### C. Data expansion - see section 4

---

## 4. Data expansion: what is worth downloading

Answering the 100 GB question directly: **the budget is not the constraint. Everything genuinely
relevant totals about 10 GB**, plus one dataset I could not locate. And the two highest-value CPU
jobs need **no new download at all**.

| dataset | size | what it is | verdict |
|---|---|---|---|
| **MGnify Stability** (Rocklin Lab + OpenFold, 26 May 2026) | **UNKNOWN - could not locate the download** | **1.8M** domains by cDNA display proteolysis, >200,000 sequence families, domains 60-80 aa, resolved to ~5 kcal/mol. Preprint: *Accurate protein stability prediction for small domains using mega-scale experiments*, bioRxiv `10.64898/2026.05.19.726285` | **The only candidate that could use the 100 GB.** ~4.6x MegaScale. Two blockers: I could not find the URL, size or licence (press releases name only a dataset page; bioRxiv and AIwire both returned 403), and it is the training set for SaProtΔG/ESM3ΔG, which `BENCHMARK_PLAN.md` already excludes for >30% identity to 9 of 28 MegaScale test wild types - so the same leakage question lands on us. **Ask the Rocklin Lab / OpenFold page for the link before planning around it** |
| **Tsuboyama full release** | **1.9 GB** (Zenodo `7844779`) | the complete MegaScale release: 776,298 measurements vs our 389,068 curated pairs. Raw NGS counts, K50/dG tables, AlphaFold models | **Worth it.** Lets us state exactly what our curation dropped (multi-mutants, indels, "unreliable" rows) instead of asserting it, and a multi-mutant arm becomes possible. Asks you to register your use before downloading |
| **Human Domainome 1.0** | **~4.2-4.7 GB** (Zenodo `13629491` v4 / `14356805` v5; `github.com/lehner-lab/domainome`) | >500k mutations across ~500-522 human domains by aPCA | **Worth it as a rank-only external set.** This is the set SPURS validated on, so it buys a shared external protocol. **It is aPCA abundance, not kcal/mol**: Spearman only, RMSE meaningless - the same restriction StabilityArc carries |
| **ProteinGym substitutions** | **~1.0-1.1 GB** (Zenodo `14997691`, v1.2) | 217 assays over Activity/Expression/Fitness/Stability/Binding | **Only if StabilityArc stays in.** No stability-only download exists; filter the stability tasks from the full file. The zero-shot (4.4 GB) and supervised (3.3 GB) score sets and MSAs (5.2 GB) are **not** needed |
| **S669 + Ssym+ / ThermoMutDB** | **~50 MB** (S669 structures: Zenodo `7568094`, 44.1 MB) | 1,338 single-site mutations (direct + reverse) over 95 chains; ThermoMutDB ships as SQLite | **Cheap, take it.** S669 ships wild-type *and* mutant structures, and homology-filtered S669 is what a late-2026 paper is expected to report. Caveat: ProTherm lineage, so overlap with older predictors' training sets must be disclosed |
| FireProtDB full | already here (1.7 GB in `DATASETS/`) | | nothing to fetch |
| AlphaFold models for FireProtDB | ~0.2 GB | 176 structures | `scripts/fetch_alphafold.py`, one command, not yet run |
| UniRef50 / massive_kb | already here (12 GB / 27 GB) | | **unused by a ΔΔG benchmark.** Do not upload either |

**Recommended download: ~7 GB** (Tsuboyama full + Human Domainome + S669/Ssym/ThermoMutDB +
FireProtDB structures), plus ProteinGym's 1.1 GB only if StabilityArc stays. That leaves the
100 GB budget almost untouched, which is the correct outcome: section 1 says data volume is not
what is limiting this project.

---

## 5. What the 5 Kaggle CPU notebooks are actually for

CPU sessions are a separate pool from the GPU ones (5 concurrent vs 2), so this work can proceed
**while the GPU quota stays busy**. Ranked by value:

1. **The Foldseek / TM-align leakage audit.** 37 test structures against 299 structured train
   parents, ~11k pairwise alignments: CPU-bound, embarrassingly parallel, and **needs no
   download** - the 862 PDBs are already on disk. `GEOMETRIC_PLAN.md` calls this the most
   important new check the coordinate path creates, and it is also worth reporting beside the
   existing MinHash number regardless of whether the coordinate arm is ever built.
2. **MMseqs2 re-clustering at 25% identity.** The field's convention is 25%; ours is 30%. CPU
   only, our sequences only, **no download**. Lets the paper report both and stop explaining the
   difference.
3. **ETL of the full Tsuboyama release** (1.9 GB) - quantify what our curation dropped.
4. **Human Domainome ETL** (~4.5 GB) - parse, map domains to UniProt, build a rank-only holdout.
5. **AlphaFold fetch** - only if this machine's network is slower than a Kaggle session's
   (internet must be enabled on the notebook).

Constraints from `kaggle-colab-lessons.md` that apply to all of the above: uploaded archives are
**extracted** (`X.tsv.gz` becomes a folder `X.tsv/` holding `X.tsv`), so job code must find inputs
by `rglob` and flatten one-file folders; `/tmp` on a CPU session has ~1 TB but `/kaggle/working` is
20 GB and turns read-only when full; first reads of a large input dataset can take hours in a
fresh session; set `PYTHONUTF8=1` on the Windows CLI; a large `datasets create` may print an error
and exit 1 yet succeed, so check `kaggle datasets list --mine` before re-uploading.

---

## 6. Open decisions

1. **Commit?** 13 paths are untracked or modified. Kaggle and Colab both need the code from a
   branch.
2. **Measure the step time** before anything is sized for real - one short probe per GPU type,
   then `--plan` again.
3. **The MGnify Stability link.** Everything about the large-download option depends on a URL,
   size and licence I could not find.
4. **Sequencing.** Competitor adapters first, or the coordinate arm first? `GEOMETRIC_PLAN.md`
   argues adapters first, and section 1 argues the coordinate arm matters more than usual because
   the graph currently buys nothing.
5. **The web backend's 6 failing checks** - separate from the benchmark, unfixed, and the reason
   `/api/ddg` returns 503.
