# PROJECT_STATUS.md

## 0. Update — 2026-11-20 (`version-2` active branch, pre-v2 implementation checkpoint)

* **Workspace scaffolded** into `project/` (Git repository only + gitignored `DATASETS/`), `paper/`, and `extras/` (non-Git notes, Zenodo staging, Kaggle account templates).
* **Git synced:** Previous uncommitted benchmark/geometry/runner scaffolding reviewed, verified (`tests/test_remote_job.py` 52/52 passed), committed as `f083217`, pushed to `origin/main`, and fast-forwarded onto **`version-2`** (`origin/version-2`). All v2 implementation proceeds on `version-2` before merging back to `main`.
* **Expanded compute pool:** 4 Kaggle accounts (2 weeks -> ~240h T4 GPU + 20 concurrent CPU workers) + ~200 CU (~20h) 95 GB G4 GPU (keep 20 CU safety reserve).
* **Approved roadmap (details in `../extras/notes/HOLO_GEON_MASTER_PLAN.md` & `../extras/notes/HANDOFF.md`):**
  1. **Block A (next):** Holo-GeoN v2 core — single-pass `EsmForMaskedLM` (fixing the duplicate 650M forward pass in `src/encoder.py` + extracting free WT-marginal log-odds, Shannon entropy, and multi-layer taps), parent graph cache + Protein-Batch training, `20.6M -> ~2.4M` compressed backbone (FiLM + compact bidirectional mixer + DropEdge + CDS modality dropout), evolutionary-anchored antisymmetric head + `beta-NLL` (`beta=0.5`), Contact-Gated Double-Mutant Epistasis head, upgraded `score_predictions.py` (pooled + per-protein median), and fixing the 6 pre-existing `tests/test_app.py` failures + `runapp.py`.
  2. **Block B:** Cross-platform non-Linux CLI (`holo.py`/`hologe`), local `fetch_alphafold.py` (176 FireProtDB structures), and non-registration Kaggle CPU jobs (Foldseek/TM-align audit, MMseqs2 25% splits, S669/Domainome ETL, ViennaRNA/CAI, `src/geometry.py`).
  3. **Block C:** User completes Zenodo `7844779` & Rocklin/OpenFold MGnify Stability registrations -> 4-account Kaggle T4 ablations + competitor adapters (ThermoMPNN, SPURS, RaSP, PROSTATA, optional DDGemb) -> Unified same-GPU 95 GB G4 headline runs -> Zenodo `Holo-StabBench v1.0` dataset & weights release.
  4. **Block D (later):** Modern Web GUI redesign (`app/frontend/`) + live real-time CLI & tunneled GUI run on G4.

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
