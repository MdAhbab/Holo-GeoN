# Changelog — round-2 review fixes

Every change in this pass, keyed to a finding in `REVIEW_ROUND_2.md`. Written so
a session starting cold can pick up without re-deriving anything: what was
wrong, what the code does now, which files moved, and how it was checked.

**Status: all 17 round-2 findings addressed, plus the 4 round-1 items round 2
re-opened and the documentation contradictions.** Test suite 177 → 288 checks,
0 failed. `tests/test_round2_fixes.py` is new and every check in it fails
against the pre-fix code at `bbae5e7`.

A self-audit after the first fix commit (`934111c`) found three items I had
recorded as done that were not fully done. They are fixed in the follow-up
commit and listed in §4a below. Everything else in this document was verified
by re-reading the tree, not from memory.

**Environment this was done in:** Windows 11, Python 3.13.7, torch 2.10.0+cpu,
no NVIDIA GPU (the machine has an Intel Arc), no ESM-2 weights, no MMseqs2, no
real datasets. ViennaRNA **is** importable here, unlike in the reviewer's
environment. Every memory figure below is therefore analytic, not measured —
see "Still open" at the end.

---

## 1. Blockers

### R2-01 — `cds_coverage` counted rows in the input table, not records in the cache

The number introduced so an empty `E_mech` could not hide was computed as
`n_with_cds / len(parents)` over the **input table**, and reported `1.0` against
a cache in which every record had `mech_valid = 0`.

- `src/encoder.py` — new `EncodingStore.mech_coverage(seqs, sample=None)`. Opens
  the cached records and counts `mech[:, CH_VALID].max() > 0`. Bounded by
  `sample` and labels its own result `cds_coverage_is_a_sample`, so a 20k-parent
  cache does not cost 20k deserialisations to describe itself.
- `scripts/precompute_embeddings.py` — the manifest is built from that
  measurement (`MANIFEST_COVERAGE_SAMPLE = 512`). `parents_with_cds_column` is
  kept but is now explicitly named as a property of the table, not the cache. A
  parent whose table gained a CDS but whose cached record has none is
  **re-encoded** rather than skipped.
- `scripts/train.py` — the warning gate is **inverted**. A *missing*
  `manifest.json` is now the loud branch (it was the silent one, which is why the
  warning could never fire for any cache that existed). New
  `measure_mech_coverage()` additionally samples the parents *this run will
  read* and writes them to `metrics.json` as `encoder_cache_measured`.

Verified: a mech-free cache reads `0.0`, a fed cache reads `1.0`.

### R2-02 — whole-sequence batches were an 8× memory blowup

R1-07 grew whole-sequence graphs 256 → 2048 nodes while every task shared one
`batch_size: 64`.

- `src/data.py` — new `NodeBudgetBatchSampler`. Caps
  `len(batch) * max_nodes_in_batch` instead of `len(batch)`. Shuffles, buckets by
  length to limit padding, and derives its order from `(seed, epoch, pass)`.
- `scripts/train.py` — `_sequence_loader()` uses it; budget defaults to
  `batch_size * window` (16 384 nodes), i.e. **the same per-step node cost as the
  ΔΔG batch**, so no task can silently demand more than another.
- `configs/run.yaml`, `src/config.py` — new `sequence_node_budget` (null = the
  default above).
- `scripts/budget.py` — removed the spurious `/ 2` from the GAT term (edge lists
  are already directed) and added the whole-sequence path.

Measured with the repo's own budget script:

| | padded nodes | analytic lower bound |
|---|---|---|
| ΔΔG batch (64 × 256) | 16 384 | 5.35 GB (was reported 4.12 with the `/ 2` bug) |
| whole-sequence, uncapped at `batch_size: 64` | 131 072 | **48.1 GB** |
| whole-sequence, node budget | 16 384 | **6.29 GB** |

Note the uncapped figure is worse than the report's scaled 33 GB, because the
SSM scan-step count grows with node count too.

### R2-03 — the interface shipped a hardcoded biophysical classifier

- `app/frontend/src/tabs/IdrTab.tsx` — the `μ / L^0.33 < 2.5 → "Globular /
  compact"` chip is **deleted**.
- `app/backend/inference.py` — the IDR payload gains `value_source` and
  `ci_source`, matching how the ΔΔG payload already declares provenance.
- `app/frontend/src/tabs/DdgTab.tsx` — `isStabilizing = result.ddg > 0` is gone.
  Direction, colour and arrow now come from `result.verdict` (the three-state
  head), with `verdict_source` shown beneath it. A neutral verdict renders
  neutral instead of green "▲ Stabilizing".
- `app/frontend/src/types.ts`, `styles.css` — `verdict_source`, a
  `.verdict-neutral` class, a `.result-sub` provenance line.

**Correction to the report:** `app/frontend/dist/` is gitignored and appears in
**no commit on any branch** (`git rev-list --all` finds no tree containing it).
The source defect was entirely real; the "checked-in bundle" half describes a
locally built artifact. There is no stale bundle to rebuild and no CI check
needed for one.

---

## 2. Majors

### R2-04 — the whole-sequence leakage control could not fail

- `src/splits.py` — `build_groups()` moved here from `make_splits.py` so the
  whole-sequence path uses the **identical** relation as the primary tasks
  (clustering ∪ a declared parent/gene column). New `near_duplicate_audit()`:
  bottom-32 MinHash over 8-mers, comparing held-out sequences against training
  sequences **directly**, which is the check that can actually fail.
- `src/data.py` — `load_sequence_table()` reads a declared `parent` column
  (candidates: parent / construct / protein / uniprot / accession / …), falling
  back to gene, then to the sequence — and records which in
  `report.notes["group_source"]`.
- `scripts/train.py` — `build_sequence_loaders()` groups via `build_groups` and
  **refuses the run** above `NEAR_DUPLICATE_TOLERANCE = 0.0`, naming the overlap
  and how to fix it.
- `scripts/make_splits.py` — same audit; `no_group_spans_splits` now writes the
  check's **return value** instead of the literal `True` (this was R2-17's row).
- `configs/data.yaml` — documents `parent` and why a table may be refused.

Verified on the reviewer's exact fixture: `verify_no_leakage` still passes,
the audit reports **100%**, 0% on unrelated proteins, and a declared construct
column collapses 400 variants into 1 group.

### R2-05 — the deep mutational scan still ran one ESM-2 pass per position

- `app/backend/inference.py` — `mutation_scan()` calls `_encode_once()` once,
  then `_batch_for_sites()` in chunks of `SCAN_SITE_BATCH = 32`.
- `src/substitution.py`, `README.md` — the "one encoder pass" claim corrected to
  what the code now does.

Verified: 60 positions → **1** encoder call (was 60), and a batched scan cell
matches the single-site `predict_ddg` path to 1e-3.

### R2-06 — `separable_from_noise` had no null model and no multiplicity control

`scripts/ablations.py` rewritten:

- **Pre-registered family**: one metric per task from `ablation_metrics` in the
  config, not every numeric key. Split properties (`target_std`, `pos_rate`, `n`,
  …) are refused outright.
- **Paired t-test** across shared seeds, **Holm-Bonferroni** over the whole
  declared family, **95% CI** on every delta.
- `MIN_SEEDS_FOR_TEST = 3`; below it no p-value is emitted, with a stated reason.
- `--seeds` default 1 → `[42,43,44,45,46]`; `run.py` now passes them (it passed
  none, so the documented pipeline ran a single seed).
- Every arm gets `--strict-determinism` — `src/seeding.py` claimed this was on
  for ablations and no caller ever set it (R2-15).

| P(≥1 spurious flag over 20 null comparisons) | 3 seeds | 5 seeds |
|---|---|---|
| old criterion | 0.82 | 0.27 |
| new | 0.051 | **0.055** |

**And a result that came out of the fix — read this before the real run.**
Correct statistics say the experiment as designed is underpowered. New
`minimum_detectable_effect()` reports the smallest effect each comparison could
have found at 80% power; rows below it are marked `underpowered`:

| seeds | family of 8 | family of 20 |
|---|---|---|
| 5 (the new default) | 0.117 | 0.145 |
| 10 | 0.059 | 0.067 |
| 15 | 0.044 | 0.049 |
| 20 | 0.037 | 0.041 |

Ablation effects in this literature are commonly 0.01–0.05. At five seeds most
arms will read "not significant" for want of power, not for want of an effect.
**Decide deliberately**: narrow the declared family, budget more seeds, or state
the limit in the results. What is no longer possible is reporting a
non-significant delta as evidence a component does not matter.

### R2-07 — the loss floor was not gradient-neutral and the docstring said it was

- `src/losses.py` — one `GLOBAL_LOSS_FLOOR` for **every** task, replacing
  `-4.0811` for two tasks and `0.0` for the rest. Docstrings corrected: the shift
  is constant w.r.t. model *parameters* but not gradient-neutral, because
  `lv* = log(2(L − floor))` is learned. New `equilibrium_weights()` and
  `current_weights()` make the weighting inspectable.
- `scripts/train.py` — per-epoch `task_weights` in `history`;
  `MAX_SATURATED_EPOCHS = 10` **aborts** when a task sits at a log-variance
  clamp bound (it was recorded and acted on by nothing).

Weight imbalance at ΔΔG NLL 1.0 vs BCE 0.6: **8.5× → 1.09×**.

### R2-08 — the MFI head was anchored at residue 0

`src/model.py` — `HoloGeoN.mfi` uses `self.global_context`, matching `idr` and
`proteomics` and matching what `src/readout.py` says whole-sequence tasks do.

### R2-09 — the interface reported calibration for a σ it did not serve

- `app/backend/inference.py` — when `sigma_scale != 1.0`, `health()` reports
  `variance_scale.calibration_after_scaling` (which the calibration run already
  computed and nothing read) and gates `uncertainty_calibrated` on **that**
  verdict. A scale with no post-scaling measurement reports "NOT MEASURED"
  rather than the raw σ's numbers. New `describes` field names which σ.
- The **IDR head no longer inherits** the ΔΔG-fitted scale — that was a
  correction in kcal/mol applied to a radius of gyration in ångström.

Verified on the reviewer's constructed case: flag `True` → `False`, reported ECE
0.0053 → 0.0729 (the served band's real value, above the 0.05 tolerance).

---

## 3. Minors

| ID | Change |
|---|---|
| R2-10 | `steps_per_epoch` is now one expected pass over the **primary** task (`len(loader) / p`), not `max(len(loader))`. Printed with its derivation; small-table replay factors warned. **Budget note:** this makes an epoch ~1.6× longer in steps at MegaScale scale — 60 real passes instead of 37.7. Lower `epochs` to ~38 to match the old wall clock. |
| R2-11 | Neutral band promoted to config key `three_state_sigma`, recorded in `metrics.json` with a **sensitivity table** at 0.4×/1×/2×. `sigma <= 0` is treated as missing rather than used. `src/heads.py` docstring corrected. |
| R2-12 | `src/seeding.reseed_loaders_for_epoch()` makes batch order a pure function of `(seed, epoch, pass)`, so resume is exact by construction. A failed checkpoint save is now **fatal** rather than one printed line. |
| R2-13 | `scripts/contact_sanity.py` gains a **random-edge control** and `--require-structure`. Records `supports_structural_claim`. `run.py contacts` prints that the structure-free result cannot be cited as evidence for the graph. `src/encoder.py`'s forward reference corrected. |
| R2-14 | `src/readout.py` states the arm's per-dataset meaning; `scripts/train.py` measures `window_coverage` (fraction of rows shorter than the window) per task into `metrics.json`, so the delta travels with what it means. |
| R2-15 | New `src/provenance.py`: git commit + dirty flag, Python, platform, package versions, device, CUDA, MMseqs2, ViennaRNA — written into `metrics.json`, `baselines.json`, `ablations.json`, the leakage report and `calibration.json`. `requirements.txt` pins `transformers>=4.40,<5` (the contact head needs eager attention) and documents the `pip freeze` lockfile step. |
| R2-16 | `esm_mlp` gets the model's epoch budget, warmup+cosine schedule, gradient clipping and early-stopping patience. Reports `parameters` and `training_budget`; capacity is still unmatched and now **said so** rather than hidden. `run.py` runs baselines on the **pathogenicity** split too (it never did). Attribution sentence corrected in `baselines.py`, `collect_results.py` and `README.md`. |
| R2-17 | Antisymmetry claim (README + this file), `budget.py` "room to spare", 7→8 mechanistic channels (3 files), `§8.2` scaffolding headers, `tests/test_splits.py` reference, `no_group_spans_splits` literal, and PDB export — which now carries 8 `REMARK 999` lines stating the coordinates are a grid and not a prediction. |

---

## 4. Round-1 items round 2 re-opened

- **R1-07 (reporting half)** — `truncated` was set and read by nothing.
  `collate_graphs` now carries it into the batch, and `metrics.json` records
  `node_cap.rows_over_cap`.
- **R1-13** — see R2-05.
- **R1-17 (second claim)** — the commit said "built graphs cached"; no cache of
  any kind existed. The vectorised `_cap_degree` (the real half) already made
  graph building 2.81 ms. Added the cache that *is* worth having — an optional
  LRU on `EncodingStore.get`, since a mutation dataset reads one record per
  **row** from a cache keyed by **parent**. **Off by default** (`record_cache_size: 0`)
  with the RAM cost documented, rather than inherited silently.
- **R1-26** — the clamp was a provable no-op (max attainable margin is
  `(n-1)/2`, already below the `n/2` bound). Replaced with
  `chunk_context_score()`: the **fraction of a residue's real context** the chunk
  retains, worse side taken. Dominates the old rule at every residue and every
  length tested; the worst-served residue improves 0.143 → 0.250 at L=1023, and
  the reviewer's residue 894 now comes from the 1022-residue chunk instead of the
  257-residue tail.

---

## 4a. Gaps found in my own first pass (fixed in the follow-up commit)

Three items I had marked done and had not finished. Found by walking the report
against the tree rather than trusting the previous summary.

- **R2-17, row 2 — missed entirely.** `MIGRATION_NOTES.md:263` still restated
  "exact head antisymmetry in both train and eval mode" — the same over-claim
  R1-23 was raised for, in the document that records R1-23. I had fixed the
  README line and not this one. Now corrected, with the dropout explanation
  attached.
- **R1-11 residual — still undisclosed.** The reversibility check substituted
  `preferred_codon(mt_aa)`, whose CAI weight is **1.0 by definition** (it is the
  arg-max of a table normalised so the most-used codon scores 1). Measured here:
  a **+0.273** one-sided CAI shift at the one site the check is about. New
  `cai_matched_codon()` picks the synonymous codon nearest the parent codon's own
  weight, cutting the mean shift to **+0.089**, and the residual is now reported
  as `induced_cai_shift_at_mutated_site` rather than assumed to be zero. It
  cannot be driven to zero — the amino acid changed, so the reachable codon
  weights changed with it — which is exactly why it is reported.
- **R2-16, second clause — done weaker than asked.** The finding asked to
  "either match parameter count or report both a matched-capacity and a
  matched-budget variant". I had done neither, only disclosed the 14.1× gap.
  `scripts/baselines.py` now runs **both**: `esm_mlp` (matched budget, narrow)
  and `esm_mlp_matched_capacity`, sized by `mlp_width_for()` to the full model's
  own parameter count — 20,604,300 against the model's 20,607,788 (1.00×). If
  the model beats the narrow MLP but not the wide one, the gap was capacity.

---

## 5. Files

**New:** `src/provenance.py`, `tests/test_round2_fixes.py`, this file.

**Changed:** `README.md`, `MIGRATION_NOTES.md`, `requirements.txt`, `run.py`,
`configs/{run,data}.yaml`,
`src/{encoder,data,splits,losses,model,readout,seeding,config,heads,features,backbone,substitution,sequence_mixers}.py`,
`scripts/{train,ablations,baselines,budget,make_splits,precompute_embeddings,contact_sanity,calibration,collect_results}.py`,
`app/backend/{inference,app}.py`,
`app/frontend/src/{types.ts,styles.css,tabs/DdgTab.tsx,tabs/IdrTab.tsx}`,
`tests/run_all.py`.

## 6. New / changed config keys

| Key | Default | Why |
|---|---|---|
| `sequence_node_budget` | `null` → `batch_size * window` | R2-02 |
| `three_state_sigma` | `0.5` | R2-11 |
| `ablation_metrics` | one metric per task | R2-06 |
| `record_cache_size` | `0` (off) | R1-17 |
| `epochs` | `60` — but see the R2-10 budget note | R2-10 |

New CLI: `scripts/train --strict-determinism`; `scripts/contact_sanity
--require-structure --seed`; `run.py --seeds --true-contacts-dir`;
`scripts/budget --sequence-nodes`; `scripts/baselines --epochs` now defaults to
the config's `epochs` rather than a hardcoded 20.

---

## 7. Still open — needs the target machine

Unchanged from §5 of `REVIEW_ROUND_2.md`. Nothing below could be checked here.

1. **Peak VRAM.** The 48.1 GB / 6.29 GB figures are analytic lower bounds. Run
   `python -m scripts.budget --device cuda` and
   `--device cuda --sequence-nodes 2048` on the 5070 Ti before trusting either.
2. **GPU determinism (R1-01).** Verified bit-identical on CPU only. Re-run the
   two-invocation check with `num_workers: 4` on the card.
3. **`cds_coverage` against a real cache.** Coverage is measured correctly now,
   but no real cache exists here to measure. Check it before interpreting any
   `no_mech_stream` delta.
4. **Contact-head quality (R2-13).** Needs `--true-contacts-dir` over real
   PDB-derived maps. The structure-free number establishes nothing on its own.
5. **Split quality with MMseqs2 present (R2-04).** All runs here used the
   exact-sequence fallback. Confirm `grouping.clustering` reads `mmseqs2 @ 30%`.
6. **Ablation power (R2-06).** Decide the seed budget and the declared family
   before the run, using the MDE table above.
7. **ΔΔG sign on the real MegaScale table (R1-19).** The warning only fires at
   `len(out) >= 100`; confirm which branch the real table takes.
8. **Calibration (R2-09).** Needs a trained model to fit a real variance scale.
9. **Wall clock.** The R2-10 epoch redefinition changes the budget — re-time
   before committing to the 2–3 day plan.

## 8. How to re-check any of this

```bash
python -m tests.run_all          # 288 checks, 0 failed
python -m tests.test_round2_fixes # just the round-2 regressions
python -m scripts.budget --device cpu   # the memory table above
```

Every check in `tests/test_round2_fixes.py` names the finding it guards and was
written to fail against `bbae5e7`.
