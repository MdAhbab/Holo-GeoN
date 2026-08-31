# Review round 2

Repository: `Holo-GeoN`, branch `version-2`, HEAD `427f1fa`.
Prior round: `REVIEW_ROUND_1.md` (commit `03d6145`), fixes claimed in `102262c`.

**Environment.** Python 3.11.15, torch 2.13.0+cu130 (CPU only), torch_geometric
2.8.0, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.17.1. No GPU, no
HuggingFace access, no ESM-2 weights, no real datasets, no ViennaRNA, no MMseqs2.
`python3 -m tests.run_all` → **175 checks passed, 0 failed, 0 errors**, matching
the commit message. Every end-to-end number below was produced with
`tests/helpers.stub_encoding` (seeded noise, not protein representations);
**nothing measured through it is a statement about accuracy** and it is used here
only to exercise plumbing, memory and control flow. Probe scripts are in
`/tmp/p2/`.

---

## 1. Verdict

**Major revision.** The round-1 fixes are real work and most of them hold:
seeding now makes a run bit-reproducible on CPU, the no-val-split failure is
loud, the ClinVar vocabulary is exact, the multi-task objective is bounded, the
budget script refuses to invent a peak, `filter_to_cache` is wired into the
primary tasks, and the interface's calibration flag is no longer "a file
exists". But the commit message claims all 27 findings were "executed, not
reasoned about", and three of them were not fixed at all — R1-13 (the deep
mutational scan still runs one ESM-2 pass **per position**; my probe counts 60
calls for 60 positions), R1-26 (the chunk-selection clamp is a provable no-op:
128 residues per boundary still take their embedding from a shorter chunk,
before and after), and the R1-07 "reported" half (the `truncated` flag is set
and consumed by nothing). Two more moved the defect rather than removing it:
R1-25's sign-test verdict now lives in the React tab instead of the backend, and
R1-14's flag now describes a σ the interface does not serve.

Two new findings are blocking. `cds_coverage` — the number the fix commit
presents as the guarantee that "E_mech being empty can no longer be invisible" —
is computed from the input table rather than the cache, and I made it report
`1.0` against a cache in which every record has `mech_valid = 0` (R2-01). And
the R1-07 fix, by growing whole-sequence graphs from 256 to 2048 nodes while
leaving one shared `batch_size: 64`, puts the analytic lower bound for an
IDR/MFI/proteomics batch at **33 GB** on the documented 16 GB card, scaled from
the repository's own budget numbers (R2-02).

Separately, and most consequential for the manuscript: the interface still ships
a hardcoded biophysical classifier (`μ / L^0.33 < 2.5 → "Globular / compact"`)
presented to the user as a result, in both source and the checked-in `dist/`
bundle (R2-03). That is the predecessor's exact pathology, in the one component
the migration notes single out as cleaned.

The statistics also need work before any ablation table is written: the
`separable_from_noise` criterion has an 8.3% per-comparison false-positive rate
at the 3 seeds the code itself recommends, and the shipped grid writes 80
uncorrected comparisons, so the probability of at least one spurious "component
matters" flag under the global null is 0.999 (R2-06).

I do not think the architecture is wrong. I think the code is not yet in a state
where a two-day run produces a defensible table.

---

## 2. Round-1 fix verification

Every row was executed. "Probe" names the script under `/tmp/p2/`.

| ID | What the fix claims | What I observed | Status |
|---|---|---|---|
| R1-01 | `seed_everything`/`loader_kwargs`/RNG wired into train, baselines, calibration | Two invocations of `scripts.train` with the same config produced **bit-identical** test metrics (`pearson 0.816176` both times, `best_epoch 6` both times). See §5. Caveat: CPU, `num_workers: 0`, `strict_determinism: false`; GPU determinism unverified. | **verified** (CPU) |
| R1-02 | Fail at loader construction with no val split; non-finite score guards selection; `validated_by_real_run` deleted | `exit 1`, no `metrics.json` written, message names the missing split. Manifest keys contain no `validated_by_real_run`; `best_checkpoint` is a real path. `MAX_NAN_EPOCHS` value guard present at `train.py:662-674`. | **verified** |
| R1-03 | `endless()` replaces `itertools.cycle` | `itertools` is no longer imported by `scripts/train.py` (only named in a comment); `endless` is a `yield from` generator holding one batch. | **verified** |
| R1-04 | CLNSIG parsed against an exact vocabulary; legacy integers refused | `Conflicting_interpretations_of_pathogenicity` → `None`, `Conflicting_classifications_of_pathogenicity` → `None`, `non-pathogenic` → `None`, `Benign/Likely_benign` → 0, `Pathogenic,_low_penetrance` → 1; numeric `2,3,4,5` → `None`. Dropped rows counted in `rejected_label_terms`. | **verified** |
| R1-05 | CDS threaded through `run.py all`; cache manifest records `cds_coverage`; train warns at zero coverage | Threading in `run.py:228-231` is correct. But the warning is gated on `if mech_on and cache_manifest and …` — a cache **without** a manifest yields `{}`, no warning, and `metrics.json["encoder_cache"] == {}`. My end-to-end run trained with `ablation.mech_stream: true`, `E_mech = 48` edges on 48 nodes (self-loops only), `mech_valid mean 0.00`, and **no warning**. Worse, with ViennaRNA present the manifest reports `cds_coverage: 1.0` for a cache in which every record has `mech_valid = 0` (R2-01). | **not fixed** |
| R1-06 | `filter_to_cache` in `build_split_loaders`; report in `metrics.json`; precompute exits non-zero | Deleting one of 24 cache files: run exits 0, prints `kept 301/315 (4.44% dropped) \| reasons: no_cached_encoding=14`, and `metrics.json.data_reports.stability.dropped_by_reason == {"no_cached_encoding": 14}`. Precompute raises `SystemExit` on `failures > 0` (`precompute_embeddings.py:194`). | **verified** |
| R1-07 | Whole-sequence tasks span the full protein; node cap "applied AND reported"; true length travels | Span verified: a 900-residue protein gives 900 nodes, `length=900`, `seq_length` reaches the heads. But `truncated` is set in `build_sample_graph` and **read by nothing** — `collate_graphs` drops it, and a repo-wide grep finds no consumer. The regression test asserts only `capped["truncated"] is True` on the dict, which does not establish reporting. Two follow-on defects: R2-02 (memory) and R2-08 (`mfi` still anchored at residue 0). | **partially fixed** |
| R1-08 | Whole-sequence tables get "a real grouped three-way split" | Three-way split and `verify_no_leakage` are present, but the group key is the exact sequence (`train.py:199`). On a DMS-shaped MFI table (426 single mutants of one 200-residue protein) it produced 426 groups, `verify_no_leakage → True`, and **100% of test rows had a training row 1–2 residues away**. README's "MMseqs2 identity clustering unioned with gene identity … every task" remains false for three heads. See R2-04. | **partially fixed** |
| R1-09 | Objective bounded by per-task floors and clamped log-variances | Bounded: with the reviewer's divergence setup the total settles at −7.08 and the weight at 20.09 (`exp(3)`), versus −1.9e19 / 4.7e18 before. But the floor is not gradient-neutral as the docstring claims, and the 1/(2L) contribution degeneracy is unchanged. See R2-07. | **verified** (with R2-07) |
| R1-10 | Tasks sampled proportionally to loader length | Implemented (`train.py:521-532`), realised steps recorded in `history`. At the shipped temperature 0.5 the primary task now sees **63% of its own training batches per "epoch"** and a 300-row IDR table is still replayed 18×. See R2-10. | **verified** (with R2-10) |
| R1-11 | Reversibility carries the parent CDS through the substitution; mutants to a scratch store | `_substituted_cds` gives `mech_valid` 1.00 on both sides (was 1.00 vs 0.00); scratch store at `runs/main/reversibility_cache`. Residual: the mutant codon is `preferred_codon(mt_aa)`, whose CAI weight is 1.0 by definition versus a 0.785 mean over synonymous codons — a systematic +0.215 CAI shift at the one mutated site, undisclosed. | **verified** (minor residual) |
| R1-12 | Zero-shot scores the full sequence; raise rather than fall back to test | `esm_zero_shot_baseline` scores the full sequence and counts `scored_on_truncated_context`. With `val` emptied, `scripts.baselines` exits 1: *"no validation rows… selecting on test would inflate it."* | **verified** |
| R1-13 | "encoder.encode hoisted out of the deep-mutational-scan loop: one encoder pass per protein, not one per site" | **False.** `mutation_scan` still calls `_context_for(seq, site)` per position (`inference.py:292`), which calls `_encode_once` → `encoder.encode`. Probe: 60-position scan → **60 encoder calls, 1 distinct sequence, all identical to the input**. The `_batch_for_sites` helper the fix added is only ever called with a single site. `src/substitution.py:20-22` still says "a full 20 x L scan costs one encoder pass"; the fix commit did not touch that docstring. See R2-05. | **not fixed** |
| R1-14 | `uncertainty_calibrated` gated on measured coverage and PIT-ECE, with a variance-rescaling step | Gating is real (`bool(cal.get("well_calibrated"))`). But the flag and the coverage table describe the **unrescaled** σ while the interface serves `σ × sigma_scale`. Constructed case: flag `True`, reported 95% coverage 0.948, ECE 0.0053 — while the served band has true coverage 0.998 and ECE 0.070, above the repo's own 0.05 tolerance. See R2-09. | **partially fixed** |
| R1-15 | Three-state head evaluated (macro-F1, per-class recall, confusion); class weights wired | Real: `metrics.json.test.stability.three_state` carries `macro_f1`, `accuracy`, `recall_per_class`, `support_per_class`, `confusion_matrix`, `class_order`. `compute_class_weights` supplies both `three_state` weights and `pathogenicity` `pos_weight`. Caveat R2-11: the labels come from a hardcoded 0.5 kcal/mol band whenever the table has no `sigma`. | **verified** (with R2-11) |
| R1-16 | Budget uses analytic terms and refuses an estimate without a CUDA measurement | Confirmed: prints `MEASURED peak: UNAVAILABLE`, an `Analytic LOWER BOUND at 64: 4.12 GB`, and no "estimate". Two residuals: the GAT term carries a spurious `/ 2` (1.23 GB where 2.46 GB is correct, matching round 1's independent 2.38 GB), and the bound ignores whole-sequence batches entirely (R2-02). README's "fits a 16 GB card with room to spare" was not updated. | **partially fixed** |
| R1-17 | `_cap_degree` vectorised (34.2 ms → 1.18 ms) **and built graphs cached** | Vectorisation verified and sufficient: `build_sample_graph` on a 500-residue protein is now **2.81 ms** (was 35.3 ms); a MegaScale-scale epoch is 0.08 h of data loading on 4 workers. Correctness holds (§4). But **no graph cache exists** — repo-wide grep finds no `lru_cache` in `src/` or `scripts/`, and eight identical `(seq, site)` `__getitem__` calls cost the same each time. The commit message's second clause is false; the first makes it moot. | **partially fixed** |
| R1-18 | `patience`, RNG state and GradScaler checkpointed and restored; resume `except` narrowed | `patience`, `nan_epochs`, `scaler`, `rng` are saved and restored; the `except` now raises. But the DataLoader's shuffle order is **not** restored: the loader carries its own `torch.Generator` seeded from `cfg.seed` at construction, outside the checkpointed global RNG. A run resumed after epoch 1 replays epoch 1's exact batch order as epoch 2 — while printing `rng_restored=True`. See R2-12. | **partially fixed** |
| R1-19 | `ddg_sign` declared per dataset, normalised at load, recorded, warned on positive mean | Verified in the suite's own output: the warning fires, `report.notes["ddg_sign"]` records the applied key, and the two readings are exact negations. `configs/data.yaml` documents the key. | **verified** |
| R1-20 | The unprovenanced `0.252` is gone | Repo-wide grep: `0.252` appears only in `REVIEW_ROUND_1.md` and a `node_modules` changelog. `src/encoder.py:17-22` now forward-references the artifact path. (But see R2-13 — that artifact cannot distinguish the contact head from noise.) | **verified** |
| R1-21 | `logaddexp` aggregation and clamped log-variance inside the head | Forced a raw score log-var of 12.0: aggregated log-var is `10.0` and finite in both fp32 and fp16 (was `inf` in fp16). The clamp lives in `score()`, so training, calibration and serving see one σ. | **verified** |
| R1-22 | Dead `pooling.py`/`idr_prior.py` deleted; `sequence_mixers` docstring matches the shipped config | Both files gone; the docstring is corrected and names the real dimensions (1280 + 8 = 1288). Residual: `sequence_mixers.py:44` and `:91` still carry `# §8.2-i` / `# §8.2-ii` section headers, the scaffolding MIGRATION_NOTES says is gone. New doc rot from R1-24: `src/features.py:196` says "(length, 7)", `src/backbone.py:83` says "mechanistic channels (7)", `precompute_embeddings.py:7` says "seven mechanistic channels" — `N_MECH_CHANNELS` is 8. | **partially fixed** |
| R1-23 | The antisymmetry claim stated precisely and asserted under `eval()` | The test asserts it under eval. But `README.md:73-74` still says *"asserted in the test suite in both train and eval mode"*, and `MIGRATION_NOTES.md:263-265` restates round 1 as having *confirmed* "exact head antisymmetry in both train and eval mode". Probe: `[eval] independent passes 0.000e+00`, `[train] independent passes 8.149e-02`. | **not fixed (docs)** |
| R1-24 | `aa_known` channel; interface accepts X/B/Z/U/O; nucleotides refused with a reason | `clean_sequence("MKVXBZUO")` accepted; `"ACGTACGTACGT"` refused with the nucleotide message; unknown residues get mid-scale 0.5 volume plus `aa_known = 0`. | **verified** |
| R1-25 | Verdict from the three-state argmax; non-finite values serialise as `null` | Backend verified: `verdict` is `classes[argmax(three)]`, `verdict_source` names it, `_json_safe(-inf) → null`. But `app/frontend/src/tabs/DdgTab.tsx:46-48,167-168` re-derives `isStabilizing = result.ddg > 0`, renders a "Direction: ▲ Stabilizing / ▼ Destabilizing" chip from it, and colours the three-state verdict with the sign-derived class. The exact contradiction R1-25 named is still on screen, and the string is in the checked-in `dist/` bundle. See R2-03. | **defect moved** |
| R1-26 | "Chunk ties break on context length" | **No-op.** The fix adds `margin = torch.clamp(margin, max=n/2.0)`. Simulating the shipped selection with and without the clamp gives identical counts at every length: **128 residues per boundary** take their embedding from a strictly shorter chunk, before and after, for L ∈ {1023, 1200, 1600, 2000, 3000, 5000}. e.g. L=1023, residue 894: chunk `(766,1023)` (len 257) wins with margin 128 over chunk `(0,1022)` (len 1022) whose margin is 127; the cap at 128.5 never binds. | **not fixed** |
| R1-27 | `global_pool_readout` and `no_geo_stream` arms added, multi-seed with `separable_from_noise`; the `esm_mlp` gap no longer attributed to message passing alone | Both arms exist and work. But the attribution sentence is unchanged in all three places: `scripts/collect_results.py:149`, `scripts/baselines.py:230`, `README.md:164` all still say the gap "is the message-passing contribution". `run.py stage_ablations` passes no `--seeds`, so the documented one-command pipeline runs one seed and writes `separable_from_noise: null` deltas. Statistics: R2-06. Confound: R2-14. | **partially fixed** |

---

## 3. New findings

### R2-01 [BLOCKER] `cds_coverage` counts rows in the input table, not records in the cache — and I made it report 1.0 against a cache that is entirely mech-free

**Location:** `scripts/precompute_embeddings.py:114` (`todo` skips cached
parents), `:121` (`n_with_cds` over **all** `parents`), `:136-158`
(`record_manifest`); `scripts/train.py:437-449` (the warning gate);
`scripts/train.py:765` (copied into `metrics.json`).

**Problem.** `n_with_cds` is computed over every parent in the input table.
`todo` excludes parents already in the cache. If a cache was built before the
CDS was attached — the ordinary path, since `--cds-source` is optional and
`stage_etl` skips without it — re-running precompute against the CDS-joined
table encodes nothing, yet writes `cds_active = n_with_cds` and
`cds_coverage = 1.0`. Every cached record still has `mech_valid = 0` and
`E_mech` is self-loops only.

Separately, `train.py:439` gates the warning on `cache_manifest` being truthy.
A cache with no `manifest.json` — every cache that exists today, including the
one round 1 used — returns `{}`, so the warning is skipped and
`metrics.json["encoder_cache"]` is `{}`.

**Probe** (`/tmp/p2/p09b.py`, ViennaRNA simulated present since it is absent here):

```
pre-existing cache: mech_valid mean = 0.0
[precompute] unique parents=6  to encode=0  residues=0
[precompute] parents with a real coding sequence: 6/6
[precompute] cache already complete.

manifest: {"parents_requested": 6, "parents_encoded_this_pass": 0,
           "parents_with_cds_column": 6, "vienna_available": true,
           "cds_active": 6, "cds_coverage": 1.0}
ACTUAL mech_valid across the cached records: 0.0
```

and (`/tmp/p2/p08_mech.py`, the real end-to-end training run of §5):

```
cache manifest.json present? -> False
store.manifest() -> {}
metrics.json['encoder_cache'] -> {}
metrics.json ablation.mech_stream -> True
training log contained the R1-05 warning? -> False

actual graph for a training sample: n=48 nodes
  E_geo  edges = 394
  E_mech edges = 48   (self-loops alone would be 48)
  mech_valid mean = 0.00
```

**Consequence.** The fix commit presents this number as the guarantee that
"E_mech being empty can no longer be invisible, and a ~0 `no_mech_stream` delta
can no longer be written up as a scientific result when it is a plumbing one."
It is a derived constant that does not describe the cache it claims to describe.
A manuscript could carry `cds_coverage: 1.0` in its methods while the
mechanistic stream — the paper's distinguishing contribution — was inactive for
the entire run. This is the same shape as the predecessor's fabricated figures:
a number that reaches the paper with no measurement behind it.

**Fix.** Compute coverage from the cache: iterate the records that will be read
and report `mean(rec["mech"][:, CH_VALID].max() > 0)` over them, or at minimum
`sum(1 for s in parents if store.get(s).get("has_cds"))`. Refuse to skip a
cached parent whose `has_cds` disagrees with the table (re-encode it, or fail
with instructions). Invert the `train.py:439` gate: a **missing** manifest must
be the loud case, not the silent one — a cache that cannot describe itself is
not a cache a reported run may read.

---

### R2-02 [BLOCKER] The R1-07 fix makes any whole-sequence batch a 33 GB lower bound on the documented 16 GB card

**Location:** `src/data.py:498-501` (whole protein, `max_nodes`);
`configs/run.yaml:97` (`max_sequence_nodes: 2048`); `configs/run.yaml:80`
(one shared `batch_size: 64`); `scripts/budget.py` (models `window` only).

**Problem.** Before the fix, a whole-sequence sample was a 256-node window; now
it is up to 2048 nodes. Every task shares one `batch_size`. Edge count is linear
in node count by construction (`graph_top_k` per node + backbone + self-loops),
so a whole-sequence batch costs 8× the GATv2 message tensors and 8× the SSM scan
buffers of the ΔΔG batch the budget models.

**Probe** (`/tmp/p2/p26b_oom.py`, scaling the repository's own budget output —
no assumption about edge density beyond linearity in `n`):

```
scripts/budget.py, its own numbers at window=256, batch=64:
   GATv2 edge tensors 1.23 GB | SSM scan 2.50 GB | nodes 0.08 GB | stated LOWER BOUND 4.12 GB

Whole-sequence tasks share batch_size 64 at max_sequence_nodes = 2048, 8x the nodes:
   GATv2 edge tensors 9.8 GB | SSM scan 20.0 GB | nodes 0.62 GB
   scaled LOWER BOUND at batch 64: 33.0 GB   (target card: 16 GB)
   and the GAT term is itself under-counted 2x by budget.py -> 39.7 GB
```

The 2× under-count is independently confirmed (`/tmp/p2/p21_gat.py`):

```
directed edges in the batch                : 161,408
scripts/budget.py term (note the '/ 2')     : 1.23 GB
per-edge message tensors actually kept      : 2.46 GB
round-1 review's independent figure         : 2.38 GB
```

Whole-sequence graph construction is also expensive with a real CDS
(`/tmp/p2/p25_seqmem.py`): `build_mechanistic_graph(n=2048, mech_valid=1)` takes
118 ms and materialises five dense 2048×2048 matrices (80 MiB transient) per
sample — 1.9 s of CPU per batch on 4 workers, inside a DataLoader worker.

**Consequence.** The moment `idr_data`, `mfi_data` or `proteomics_data` is
configured, the documented run OOMs on the documented card, part-way through
epoch 1 — the same failure mode as R1-03, reintroduced by the R1-07 fix. Three
of the six heads in the README table cannot be trained at the shipped
configuration. Nothing in the budget script or `MIGRATION_NOTES.md` mentions it.

**Fix.** Give whole-sequence tasks their own `batch_size` (or bucket by node
count and cap `nodes_per_batch` rather than `samples_per_batch`), model that path
in `scripts/budget.py`, and remove the `/ 2` from the GAT term. Consider
`torch.utils.checkpoint` on the SSM scan, which is 60% of the term.

---

### R2-03 [BLOCKER] The interface ships a hardcoded biophysical classifier presented as a result

**Location:** `app/frontend/src/tabs/IdrTab.tsx:173`, present in the checked-in
bundle `app/frontend/dist/assets/index-B1yqkqeY.js`.

**Problem.** The IDR tab renders a "Character" chip:

```tsx
{result.mu / Math.pow(result.length, 0.33) < 2.5 ? 'Globular / compact' : 'Extended / disordered'}
```

Neither `0.33` nor `2.5` appears anywhere else in the repository. There is no
citation, no derivation, no configuration key, and no model involved: the label
is a threshold on the model's μ divided by `L^0.33`, computed in the browser.
The user is shown a categorical biophysical conclusion — *globular* versus
*disordered* — that the model never produced.

**Probe.** Present in source and in the shipped build:

```
$ grep -o ".\{110\}Globular.\{80\}" app/frontend/dist/assets/index-B1yqkqeY.js
-chip",children:["Character:"," ",jsxRuntimeExports.jsx("strong",{children:
Hr.mu/Math.pow(Hr.length,.33)<2.5?"Globular / compact":"Extended / disordered"})]})]})]})
```

**Consequence.** `MIGRATION_NOTES.md` ("`src/heuristics.py` and demo mode":
*"Deleted. It stood in for the model whenever weights were absent"*) and
`README.md` (*"It does not substitute an approximation"*) both assert that no
heuristic stands in for model output in the interface. This one does, on every
IDR request, whether or not the model is trained. It is a screenshot away from a
figure. A reviewer who finds this will not accept the rest of the
"we removed the fallbacks" argument on trust.

**Fix.** Delete the chip, or derive the label from a head that was trained to
produce it and label it with its provenance the way `ci_source` and
`verdict_source` already do in the ΔΔG payload. Then rebuild `dist/` — and add a
CI check that `dist/` is regenerated from `src/`, since the same bundle is what
a reader would run.

Related, same file family: `DdgTab.tsx:46-48,167-168` re-derives
`isStabilizing = result.ddg > 0` and displays a "Direction" chip and a colour
class from it, next to the backend's three-state `verdict`. R1-25 removed
exactly this sign test from the backend; it is still on screen. A +0.006 kcal/mol
prediction whose three-state argmax is `neutral` still renders as
"▲ Stabilizing", in green.

---

### R2-04 [MAJOR] The whole-sequence "leakage control" reports no leakage on a split where every test row has a near-identical training row

**Location:** `scripts/train.py:198-206` (`groups = df["parent_seq"].tolist()`);
claim at `README.md:156`; `MIGRATION_NOTES.md:97-101`.

**Problem.** The R1-08 fix groups by the exact parent sequence. For the three
tables this path serves, that is not a homology relation. An MFI table is a deep
mutational scan: thousands of single point mutants of one protein, every one a
distinct sequence and therefore its own group. Retention-time corpora are
peptides from the same proteome. `verify_no_leakage` passes trivially.

**Probe** (`/tmp/p2/p03_seqsplit.py`):

```
MFI-shaped table: 426 rows, all single mutants of ONE 200-residue protein
unique groups                : 426
verify_no_leakage passes     : True
split sizes                  : Counter({'train': 341, 'val': 43, 'test': 42})
min Hamming distance from a TEST row to the nearest TRAIN row (first 40 test rows):
    [1, 2]  -> identical parent, differing in this many residues
   fraction of test rows whose nearest train row differs by <=2 residues: 100%
```

**Consequence.** `README.md:156` — *"Every task is split with MMseqs2 identity
clustering unioned with gene identity, whole groups to one side"* — is false for
the IDR, MFI and proteomics heads: no clustering, no gene relation, and a check
that returns `True` on >99%-identity leakage. Any metric these heads report is
an interpolation score presented as generalisation. The heads are in the README
capability table and are served by the API.

**Fix.** Route these tables through `scripts/make_splits.py` like the other two
(they already carry `group`), so they get union-find over clustering ∪ gene, a
leakage report, and the same `no_group_spans_splits` verification. For a DMS
table the group must be the **parent** protein, not the variant — add an explicit
`parent`/`construct` column to `configs/data.yaml` and require it, rather than
defaulting to the row's own sequence. Until then, remove the three heads from the
paper's claims and say why.

---

### R2-05 [MAJOR] The deep mutational scan still runs one ESM-2 pass per position; R1-13 was not fixed

**Location:** `app/backend/inference.py:292` (`z, batch = self._context_for(seq, site)`
inside the position loop), `:202-208` (`_context_for` → `_encode_once` →
`encoder.encode`); `:190-200` (`_batch_for_sites`, never called with more than
one site); claims at `src/substitution.py:20-22`, `README.md:75-76`.

**Probe** (`/tmp/p2/p01_scan.py`, counting encoder with a stub):

```
scan over 60 positions
encoder.encode() calls        : 60
distinct sequences encoded    : 1
all identical to the input    : True

src/substitution.py claim     : ['same ``z``, so a full 20 x L scan costs one encoder pass rather than']
inference._encode_once claim  : ['a scan reuses it across every position.',
                                 'Round-1 review, R1-13: the scan called this per position on the']
predict_ddg encode() calls    : 1
```

The fix commit refactored `_context_for` into `_encode_once` + `_batch_for_sites`
— the two pieces needed to hoist the call — and then left `mutation_scan`
calling `_context_for` per position. `_encode_once`'s own docstring says the
result "does not depend on any mutation site, so a scan reuses it across every
position"; it does not.

**Consequence.** `MAX_SCAN_POSITIONS = 600`, so a heatmap is 600 identical ESM-2
650M forward passes. `/api/compare` (`app.py:234-236`) calls `mutation_scan` for
up to 400 positions on every AlphaFold comparison. Any wall-clock claim about the
scan endpoint is off by a factor of L. The commit message states this was fixed
and "executed, not reasoned about"; it was not executed.

**Fix.** Call `_encode_once` once, pass every site to `_batch_for_sites`, chunk
the resulting batch, and run the backbone in batches. Then correct
`src/substitution.py:20-22` and `README.md:75-76` — the honest claim is "one
encoder pass per protein", which is a genuine and large win over 20·L·2 and
worth stating accurately.

---

### R2-06 [MAJOR] `separable_from_noise` flags a null comparison 8.3% of the time, and the shipped grid makes 80 uncorrected comparisons

**Location:** `scripts/ablations.py:154-158` (the criterion),
`:141` (`for task, metric in sorted(tasks_metrics)` — every numeric metric),
`:94-97` (`--seeds` default `[42]`); `run.py:188-189` (never passes `--seeds`).

**Problem.** The criterion is `abs(delta_mean) > stdev(arm) + stdev(full)`. It is
not a test: it has no null model, no degrees of freedom, and no multiplicity
control, and it is computed for *every* numeric key in the test metrics.

**Probe A** — a real 3-arm × 2-seed sweep on synthetic data
(`/tmp/p2/w1/runs/abl/ablations.json`):

```
comparisons written to ablations.json in this 2-arm run: 32
metrics compared per (arm,task): ['accuracy','auprc','auroc','f1','mae','mcc',
  'mean_predicted_sigma','pearson','pos_rate','precision','r2','recall','rmse',
  'spearman','target_std']
  no_mech_stream/stability/target_std: delta=0.0, separable=False  <- a property of the DATA
comparisons flagged 'separable' at 2 seeds: 2
  no_mech_stream/pathogenicity/recall: delta=-0.5000 arm_sd=0.3536 full_sd=0.0000 n_seeds=2
with the shipped 5 arms this becomes 80 reported comparisons, uncorrected.
```

Note `full_sd = 0.0000`: when the two `full` runs happen to agree the threshold
collapses to `stdev(arm)` alone. `target_std` and `pos_rate` are properties of
the test split and get `delta_vs_full` rows.

**Probe B** — false-positive rate under the global null, 200,000 replicates
(`/tmp/p2/p17_sep.py`):

```
 n_seeds    P(flagged separable)    P(any of 80 comparisons flagged)
       2                  0.2156                              1.0000
       3                  0.0834                              0.9991
       5                  0.0155                              0.7129
      10                  0.0004                              0.0330
For reference, a paired two-sided t-test at alpha=0.05 would give 0.05 / 0.98.
```

**Consequence.** At the 3 seeds `ablations.py:95-97` itself recommends, the
probability that at least one arm × task × metric cell is flagged "separable"
when no component matters at all is **0.999**. The paper's ablation table would
be written from that flag. And `run.py all` runs one seed, so the documented
pipeline emits `delta_mean` to four decimals with `separable_from_noise: null` —
the exact situation R1-01 was raised about.

**Fix.** Restrict the comparison to a small pre-registered set (primary metric
per task, named in the config, not every key), use a paired test across seeds
with a Holm or BH correction over the declared family, report the delta with a
confidence interval rather than a boolean, and make `run.py stage_ablations`
pass at least 5 seeds. Drop `target_std`, `pos_rate` and `n` from the comparison
set — they are data properties.

---

### R2-07 [MAJOR] The multi-task loss floor is not gradient-neutral, and the docstring says it is

**Location:** `src/losses.py:103-113` (`LOSS_FLOORS`), `:126-130` (the claim),
`:164-165` (`shifted = value - floor`).

**Problem.** The docstring states: *"The shift is a constant, so it changes no
gradient with respect to the model — only the sign of what the weight
multiplies."* The shift is a constant with respect to the model **parameters**,
but the weight `exp(-lv)` is learned, and its equilibrium is
`lv* = log(2·(L − floor))`. Changing the floor changes `lv*`, hence changes
`exp(-lv)`, hence changes the gradient the model receives from that task.

**Probe** (`/tmp/p2/p19_floor.py`):

```
 task loss L     floor   weight w/ floor  weight w/o floor (K&G)   ratio
        2.00    -4.081            0.0822                  0.2500     3.0x
        1.00    -4.081            0.0984                  0.5000     5.1x
        0.50    -4.081            0.1091                  1.0000     9.2x

   stability      L=1.00 floor=-4.081 -> weight 0.0984, contribution 0.500
   pathogenicity  L=0.60 floor= 0.000 -> weight 0.8333, contribution 0.500
```

**Consequence.** At a typical heteroscedastic ΔΔG NLL of 1.0, the paper's primary
task ends up with weight 0.098 against pathogenicity's 0.833 — an 8.5×
imbalance created by an asymmetric constant (stability and IDR get −4.0811,
every other task gets 0.0), not by the data. The round-1 secondary degeneracy is
also unchanged and now exact: every task's contribution to the total is 0.5 at
equilibrium regardless of difficulty or dataset size, which is the opposite of
what `src/losses.py:19-23` claims Kendall & Gal buys ("Tasks with different units
and noise levels then balance themselves"). Separately, my probe drove both
log-variances to the clamp bound within 300 steps; `criterion.saturated()`
records that in the manifest but nothing acts on it.

**Fix.** Correct the docstring. Then apply the *same* floor to every task (a
uniform offset preserves the relative weighting), or better, drop the constant
`0.5·log 2π` from `gaussian_nll` and use a `softplus` on the task loss so no
floor is needed. Report `exp(-lv)` per task per epoch alongside `realised_steps`,
and abort when a task saturates for more than N epochs rather than only
recording it.

---

### R2-08 [MAJOR] The MFI head is anchored at residue 0 of the whole protein

**Location:** `src/model.py:175-177` (`mfi` uses `self.context`, while `idr` and
`proteomics` use `self.global_context`); `src/data.py:501` (`anchor = 0` when
`site is None`); `src/data.py:530` (`local_site = anchor - start = 0`).

**Probe** (`/tmp/p2/p24_mfi.py`):

```
  HoloGeoN.idr         -> self.global_context
  HoloGeoN.mfi         -> self.context (SITE-ANCHORED)
  HoloGeoN.proteomics  -> self.global_context

  SequenceDataset sample: n=900 nodes, length=900, site=0, neighborhood size=14
  site-anchored context and global context differ: True
```

**Consequence.** For MFI, `z_site` is the N-terminal residue's embedding and
`z_neighborhood` is an attention pool over the 14 nodes within 2 hops of residue
0 — two of the three components of the context vector are determined by the
protein's N-terminus. R1-07 removed the `[0, 256)` window for the whole-sequence
tasks but left the N-terminal *anchor* for this one head. `src/readout.py:27-28`
states that whole-sequence tasks "use `SiteAnchoredReadout.global_only`"; the MFI
path does not.

**Fix.** One character: `self.global_context(batch)` in `HoloGeoN.mfi`. If the
site-anchored path is intended for mutation-level MFI rows, then MFI needs a
`MutationDataset`, not a `SequenceDataset`, and the docstring should say which.

---

### R2-09 [MAJOR] The interface reports calibration numbers for a σ it does not serve

**Location:** `app/backend/inference.py:118-123` (reads `calibration` and
`variance_scale.scale`), `:154-162` (`health()` reports the pre-scaling verdict
and coverage), `:228` and `:330` (serves `σ × scale`);
`scripts/calibration.py:309` computes `calibration_after_scaling` and nothing
reads it.

**Probe** (`/tmp/p2/p10_calib.py`, synthetic residuals; a well-calibrated head on
test plus a val split whose residuals are 1.6× wider — an ordinary consequence of
a group-level split):

```
A. head sigma is well calibrated on TEST:
   well_calibrated = True  ECE = 0.0053  coverage = {'50%':0.495,'80%':0.797,'90%':0.899,'95%':0.948}
B. variance scale fitted on VAL: a = 1.588
C. calibration of the sigma the interface ACTUALLY serves (test, sigma*a):
   well_calibrated = False ECE = 0.0703  coverage = {'50%':0.712,'80%':0.955,'90%':0.99,'95%':0.998}
D. what /api/health tells the user:
   uncertainty_calibrated : True
   sigma_scale            : 1.588
   observed_coverage      : {'50%':0.495,'80%':0.797,'90%':0.899,'95%':0.948}
   expected_calibration_error: 0.0053
```

**Consequence.** The flag reads `True`, the surfaced ECE is 0.0053, and the
served 95% band has true coverage 0.998 and an ECE of 0.070 — above the
repository's own `ECE_TOLERANCE = 0.05`. The interface publishes a coverage table
measured on an interval it does not draw. This is a narrower version of R1-14,
introduced by R1-14's own fix.

Second defect in the same path: `inference.py:330` multiplies the **IDR** head's
σ by the same scale, which `scripts/calibration.py:148-156` fits from
`model.ddg()` alone. A scalar in kcal/mol units is applied to a radius of
gyration in ångström (`/tmp/p2/p16_idrsigma.py`: `sigma_scale=1.0 → IDR 95% CI
width 1.78 Å`; `sigma_scale=2.5 → 4.44 Å`, from the same model and sequence).

**Fix.** When `sigma_scale != 1.0`, report `variance_scale.calibration_after_scaling`
in `health()` and gate `uncertainty_calibrated` on *that* verdict. Fit and store
a per-head scale, or apply the scale only to the head it was fitted on and leave
the IDR σ untouched.

---

### R2-10 [MINOR] "Epoch", `epochs: 60` and `early_stop_patience: 8` no longer denote passes over any dataset

**Location:** `scripts/train.py:528` (`steps_per_epoch = max(task_sizes.values())`),
`:617-620` (proportional sampling), `:611` (`cycles` rebuilt each epoch).

**Probe** (`/tmp/p2/p04_sampling.py`, at MegaScale/ClinVar scale and the shipped
`task_sampling_temperature: 0.5`):

```
steps_per_epoch = max(len(loader)) = 4219
task              batches       p  expected steps/epoch  passes over its data
stability            4219   0.628                  2648                  0.63
pathogenicity         938   0.296                  1249                  1.33
idr                     5   0.022                    91                 18.23
mfi                    32   0.055                   231                  7.21

=> the PRIMARY task sees 63% of its training batches in an 'epoch'.
   total stability batches over 60 epochs = 37.7 full passes.
```

**Consequence.** The 844× replay R1-10 measured is down to 18×, which is the
real improvement. But `epochs: 60` now buys 37.7 passes over the ΔΔG data, and
`early_stop_patience: 8` counts units that depend on the *other* tasks' sizes —
adding a large proteomics corpus changes what an epoch means for stability
(probe: `p=0.339`, 0.63 passes, but a different `steps_per_epoch`). `history`
records `realised_steps`, which is the right mitigation; the config comments do
not mention it.

**Fix.** Define `steps_per_epoch` from the primary task's loader length divided
by its sampling probability, so an epoch is one pass over the primary task; or
rename the knob to `steps_per_checkpoint` and document that early stopping counts
those.

---

### R2-11 [MINOR] The three-state neutral band is a hardcoded 0.5 kcal/mol whenever the table has no `sigma`, under a docstring that denies it

**Location:** `scripts/train.py:84` (`DEFAULT_DDG_SIGMA = 0.5`), `:228-231`;
`src/heads.py:163-167` (the claim); `configs/data.yaml:28` (the `sigma:`
override is commented out for every shipped dataset).

**Probe** (`/tmp/p2/p11_band.py`):

`src/heads.py:164-167` states: *"The neutral band is not a hardcoded constant.
`labels_from_ddg` takes a per-measurement uncertainty, so 'neutral' means
'indistinguishable from zero given this experiment's error', which is a noise
model rather than an arbitrary threshold."* With no `sigma` column it is exactly
a hardcoded constant:

```
scripts/train.py DEFAULT_DDG_SIGMA = 0.5
sigma column detected? True | all NaN? True
labels for ddG = [-0.6, -0.4, 0.0, 0.4, 0.6, 3.0]  ->  [0, 1, 1, 1, 2, 2]
  band=+/-0.2: 8.6% of a N(-1.0, 1.5) ddG distribution is labelled 'neutral'
  band=+/-0.5: 21.2% of a N(-1.0, 1.5) ddG distribution is labelled 'neutral'
  band=+/-1.0: 40.9% of a N(-1.0, 1.5) ddG distribution is labelled 'neutral'
```

**Consequence.** The three-state macro-F1 and confusion matrix R1-15 added — the
only reported evidence that the swap-equivariant parameterisation buys anything —
are a function of an undeclared constant that decides the class of ~21% of a
realistic ΔΔG distribution, and moving it from 0.2 to 1.0 moves that to 8.6% and
40.9%. The class weights `compute_class_weights` derives inherit it. A separate
point worth stating in the paper: the three-state labels are a deterministic
function of the ΔΔG regression target, so the two heads are not independent
tasks and "multi-task" should not be claimed for that pair.

Also unguarded: `labels_from_ddg` accepts `sigma <= 0`. With `sigma = 0` only an
exactly-zero ΔΔG is neutral; with a negative `sigma` (a corrupt column) nothing
is ever neutral and no error is raised.

**Fix.** Promote the band to a required config key with its value recorded in
`metrics.json`, refuse to train the three-state head when the table reports no
per-measurement error (or report the band's effect with a sensitivity row), and
`clamp_min` the sigma.

---

### R2-12 [MINOR] Resume prints `rng_restored=True` while replaying epoch 1's data order

**Location:** `src/seeding.py:78-91` (`loader_kwargs` builds a fresh
`torch.Generator` per loader), `:94-125` (`rng_state` snapshots only the global
generators); `scripts/train.py:600-606`.

**Probe** (`/tmp/p2/p05_shuffle.py`):

```
uninterrupted epoch orders:
  epoch 1: [[6,5],[4,0],[8,9],[2,1],[3,7]]
  epoch 2: [[3,6],[5,8],[9,4],[2,0],[7,1]]
  epoch 3: [[7,4],[5,0],[9,1],[3,2],[8,6]]

resumed run (rng_restored=True):
  epoch 2: [[6,5],[4,0],[8,9],[2,1],[3,7]]      <- epoch 1's order
  epoch 3: [[3,6],[5,8],[9,4],[2,0],[7,1]]      <- epoch 2's order

epoch 2 identical to the uninterrupted run? -> False
epoch 2 of the resumed run equals epoch 1 (loader restarted)? -> True
```

**Consequence.** `src/seeding.py:19-20` says the snapshot exists "so a resumed
run continues the same trajectory rather than a fresh one (R1-18)". It does not:
each resume rewinds the data order to epoch 1 and shifts the whole schedule.
`train.py:603-606` only warns when `rng_ok` is `False`, which never happens here.
For a 2–3 day run where resume is the normal path, that is a reproducibility gap
with an affirmative indicator on it. Note this is only detectable because R1-01
made the loader deterministic — the fix is what makes the omission visible.

**Fix.** Put the loaders' generator states in the checkpoint
(`gen.get_state()` per loader, keyed by task and split) and restore them, or
re-seed each loader per epoch as `manual_seed(cfg.seed * 10000 + epoch)` so the
order is a pure function of `(seed, epoch)` and resume is exact by construction.
Related: `train.py:721-722` swallows a checkpoint-save failure with one printed
line and continues, so a disk-full at epoch 3 of 60 is discovered at the end.

---

### R2-13 [MINOR] The "check that justifies building a graph at all" cannot distinguish the contact head from noise

**Location:** `scripts/contact_sanity.py:12-18` (structure-free mode is the
default), `:19-21` (`--true-contacts-dir` is optional);
`run.py:212` (`stage_contacts` never passes it);
claims at `README.md:26-28`, `src/encoder.py:16-22`.

**Problem.** The default mode compares the contact head's long-range fraction
against a "sequence-proximity baseline" whose long-range fraction is, as the
docstring itself says, "exactly 0 by construction". Winning that comparison shows
the head emits long-range *predictions*, not that any of them are *correct*.

**Probe** (`/tmp/p2/p23_contacts.py`, 40 caches whose edges are drawn uniformly at
random with `|i-j| >= 3`):

```
Contact-head sanity (E_geo source vs sequence-proximity baseline):
  sequences sampled                 : 40
  contact head long-range fraction  : 0.880
  baseline     long-range fraction  : 0.000
```

**Consequence.** A uniformly random edge set scores 0.880 against a baseline
defined to score 0.000. The number `run.py all` writes into
`results/contacts/contact_sanity.json` and `SUMMARY.md` is not evidence that the
contact head carries structural signal. The mode that *would* be — precision@L
against real contact maps — exists but is never invoked by the pipeline. This is
the artifact `src/encoder.py:16-22` now forward-references in place of the
deleted `0.252`, so R1-20's fix points at a check that cannot support the claim.

**Fix.** Make the structure-free mode report a random-edge control alongside the
proximity one, and make `run.py contacts` require `--true-contacts-dir` (or fail
with instructions) for any run whose results will be reported. Precision@L
against even a few hundred PDB structures is the standard metric and is cheap.

---

### R2-14 [MINOR] The `global_pool_readout` arm still receives the mutation site

**Location:** `scripts/train.py:413-415` and `src/readout.py:80-84` (the arm);
`src/data.py:503-505` (the window is centred on the site regardless).

**Probe** (`/tmp/p2/p13_ablation.py`, same substitution at three sites):

```
arm: --global-pool-readout  (readout.site_anchored = False )
  site  50: window=[26,74)  ddG = -0.089287
  site 150: window=[126,174) ddG = -0.028687
  site 300: window=[276,324) ddG = -0.058390
  identical across sites? -> False
with window >= L (no site-centred cropping): [-0.025589, -0.025589, -0.025589] -> identical: True
```

**Consequence.** For proteins longer than `window` the arm is not site-blind: the
crop still encodes where the mutation is, so the arm measures "anchored readout"
versus "unanchored readout on a site-centred crop", not the 1/L dilution argument
`src/readout.py:6-13` makes. For proteins shorter than `window` — which is most
of MegaScale, whose domains are tens of residues — the arm *is* fully site-blind
and is therefore a weaker strawman than the v1 formulation it claims to
reproduce. The arm consequently means different things on the two primary
datasets, which the delta cannot separate.

**Fix.** State the arm's meaning per dataset in the results, or add a matched
arm that keeps the site-anchored readout but randomises the window centre, so the
crop is held constant across the comparison.

---

### R2-15 [MINOR] Nothing about the software environment is recorded or pinned

**Location:** `requirements.txt` (floors only: `torch>=2.7`,
`transformers>=4.40`, `torch_geometric>=2.5`, …); `scripts/train.py:747-777`
(the manifest).

**Observed.** The manifest keys written by a real run are:

```
['ablation','best_checkpoint','best_epoch','best_val_stability_pearson','config',
 'data_reports','encoder_cache','history','loss_weight_saturation','model_kwargs',
 'parameters','seed','tag','task_sampling_probabilities','task_sizes_batches',
 'test','trained_tasks','untrained_heads','wall_clock_hours']
```

No git commit, no package versions, no torch/CUDA version, no device, no
hostname, no MMseqs2 version or presence flag. `strict_determinism` defaults to
`false`, so a GPU run is not bit-reproducible even with the seed. And
`src/seeding.py:26` asserts determinism "is off by default and **on for the
ablation arms**, where reproducibility is the point" — a repo-wide grep finds
`strict_determinism` in exactly three places (`config.py:65`, `train.py:433`,
`run.yaml:51`) and no caller ever sets it to `True`, `scripts/ablations.py`
included.

**Consequence.** The encoder's contact extraction is version-sensitive by the
code's own admission (`src/encoder.py:140-142`: eager attention is required
because "the default sdpa implementation in transformers >= 5 does not emit"
attention maps), and `transformers` is pinned only at `>=4.40`. An independent
group installing from `requirements.txt` today gets a different stack from the
authors'. Nothing in `results/` would let them tell.

**Fix.** Ship a lockfile (`pip freeze` / `uv.lock`), record `git rev-parse HEAD`,
`torch.__version__`, `torch.version.cuda`, `transformers.__version__`,
`torch_geometric.__version__`, the device name and `mmseqs --version` into
`metrics.json`, and either set `strict_determinism: true` in
`scripts/ablations.py` or delete the sentence in `src/seeding.py:22-26`.

---

### R2-16 [MINOR] The `esm_mlp` control is documented as "same budget" and is not; and there is no control for the pathogenicity task

**Location:** `scripts/baselines.py:21-25` (the claim), `:244`
(`--epochs` default 20), `:186-198` (no schedule, no clipping, no early stop);
`run.py:170-179` (`stage_baselines` passes neither `--epochs` nor the ClinVar
split).

**Probe** (`/tmp/p2/p12_baseline.py`):

```
model (scripts/train.py, via run.py train):
   epochs = 60 | early stopping patience 8 | cosine LR with warmup | grad clipping
   parameters = 20,601,900
esm_mlp control (scripts/baselines.py, via run.py baselines):
   epochs = 20   <-- run.py stage_baselines passes no --epochs
   early stopping = none | LR schedule = none | grad clipping = none
   parameters = 1,463,297
docstring claim: "same budget. The gap between this and the full model is the graph's ..."
```

**Consequence.** The gap the paper's central empirical claim rests on conflates
the graph with a 3× training budget, a missing LR schedule and a 14× parameter
difference. And `stage_baselines` only ever passes the stability split, so a
reported ClinVar AUROC has no control at all.

**Fix.** Give the MLP the same epoch budget, schedule, clipping and early
stopping, and either match parameter count or report both a matched-capacity and
a matched-budget variant. Run `scripts.baselines` on the pathogenicity split too.
And correct the attribution sentence in the three places R1-27 asked for it
(`collect_results.py:149`, `baselines.py:230`, `README.md:164`) — with five arms
now in the grid, "the union of the graph, the site-anchored readout, the SSM,
the cross-attention fusion and the mechanistic channels" is both accurate and
easy to write.

---

### R2-17 [MINOR] Documentation contradicted by the code it describes

Each is a one-line edit; together they are what a method section gets written
from.

| Claim | Location | Reality |
|---|---|---|
| "asserted in the test suite in both train and eval mode" | `README.md:73-74` | train-mode independent passes violate by `8.149e-02` (probe `/tmp/p2/p20_batch2.py`); the test asserts eval only |
| round 1 "confirmed … exact head antisymmetry in both train and eval mode" | `MIGRATION_NOTES.md:263-265` | round 1 raised R1-23 for exactly this over-statement |
| "fits a 16 GB card with room to spare" | `README.md:30-31` | the repo's own `budget.py` refuses to state a peak and gives a 4.12 GB *lower bound* at window 256, 33 GB for whole-sequence batches (R2-02) |
| "Build the `(length, 7)` mechanistic feature tensor" / "mechanistic channels (7)" / "the seven mechanistic channels" | `src/features.py:196`, `src/backbone.py:83`, `scripts/precompute_embeddings.py:7` | `N_MECH_CHANNELS = 8` since the R1-24 fix |
| "V1–V6 labels … and the '§8.2 future work' framing are gone" | `MIGRATION_NOTES.md:169-172` | `src/sequence_mixers.py:44` `# §8.2-i`, `:91` `# §8.2-ii` |
| "The split-assignment logic is pure and unit-tested (`tests/test_splits.py`)" | `src/splits.py:16` | no such file exists |
| `"no_group_spans_splits": True` written as a literal into the leakage report | `scripts/make_splits.py:181` | guarded by the preceding `verify_no_leakage` raise, so it is true — but it is a constant asserting validation, the pattern R1-02 deleted `validated_by_real_run` for. Write the check's return value. |
| PDB export writes CA atoms on a boustrophedon grid with no `REMARK` disclaiming the geometry | `app/backend/app.py:344-368` | a downloadable `.pdb` with invented coordinates and no provenance line |

---

## 4. Found sound

These I tried to break and could not. They should not be churned by the next
round.

* **Determinism of a training run (CPU).** Two invocations of
  `python3 -m scripts.train` with the same config produced bit-identical
  results: `pearson 0.816176, spearman 0.796825, rmse 0.898390, mae 0.727379,
  r2 0.651581` both times, `best_epoch 6` both times,
  `a["test"] == b["test"] -> True`. R1-01's 0.030 spread is gone on this path.
  (`/tmp/p2/p2 run a/b`, logs in `/tmp/p2/w1/log_a.txt`, `log_b.txt`.)
* **`_cap_degree` correctness and its performance fix.** I reimplemented the
  pre-fix greedy loop from `03d6145` and compared at realistic densities. At the
  shipped pipeline's density (cache top-24, threshold 0.30, `graph_top_k = 16`)
  the vectorised version keeps 89–98% of the greedy version's edges, the max
  degree is exactly 16 in both, no node is isolated, and the long-range fraction
  is unchanged to three decimals (`0.850` vs `0.850`, `0.852` vs `0.851`). No
  long-range bias. Timing: 35.3 ms → **2.81 ms** per sample. The docstring's
  "gives the same answer whenever it is decidable without sequential state"
  over-states it by ≤11% of edges; the answer is not worse, only sparser.
  (`/tmp/p2/p06_cap.py`, `p06b_cap.py`, `p07_perf.py`.)
* **Exclusion accounting on the primary tasks.** Removing one of 24 cache files
  produced `kept 301/315 (4.44% dropped) | reasons: no_cached_encoding=14` on
  stdout *and* in `metrics.json.data_reports`, with no crash. R1-06 is genuinely
  closed. (`/tmp/p2/p20_batch2.py`.)
* **The no-val-split refusal, end to end.** `scripts.train` exits 1 and writes no
  `metrics.json`; `scripts.baselines` exits 1 rather than selecting on test. Both
  messages name the fix. (`/tmp/p2/p20_batch2.py`.)
* **CLNSIG vocabulary.** 15 real ClinVar terms mapped correctly, including the
  three conflicting variants, `non-pathogenic`, the `|` and `,` modifier forms,
  and `Benign/Likely_benign`. Legacy integers 2–5 all refused. Unrecognised
  terms return `None` and are counted in `rejected_label_terms`.
  (`/tmp/p2/p18_batch.py`.)
* **fp16 log-variance aggregation.** A raw score log-var of 12.0 gives a finite
  aggregated log-var of `10.0` in both fp32 and fp16, where the pre-fix form
  overflowed to `inf`. The clamp is inside `score()`, so the σ is the same in
  training, calibration and serving. (`/tmp/p2/p18_batch.py`.)
* **The bounded multi-task objective.** The round-1 divergence setup (perfect fit
  at the log-var floor, 3000 steps) now settles at total `−7.0803` with weight
  `20.09`, versus `−1.9e19` / `4.7e18` before. `saturated()` correctly reports
  both tasks at the clamp. (`/tmp/p2/p18_batch.py`.)
* **Degenerate-input robustness.** Single-class classification → `auroc: nan`
  with a warning and finite threshold metrics; constant-target regression →
  `pearson: nan`, finite RMSE; NaN ΔΔG rows dropped and counted as
  `non_finite_target`; a 1-residue protein runs end to end through
  `build_sample_graph` → `collate_graphs` → `model.ddg` giving a finite value
  with `n=1, E_geo=1`. No crash, no silent NaN in a reported metric.
  (`/tmp/p2/p15_robust.py`.)
* **The R1-11 mechanistic asymmetry.** `_substituted_cds` genuinely removes it:
  `mech_valid` is 1.00 on both sides where round 1 measured 1.00 vs 0.00, and
  every non-mutated site's CAI is identical between parent and mutant. The
  scratch store is separate from `cfg.cache`. (`/tmp/p2/p22_rev.py`.)
* **No demo path in the backend.** `tests/test_app.py` runs a live uvicorn in
  both states; all 175 checks pass, including 503 on every prediction endpoint
  without a checkpoint, absence of `demo_mode` from the API and schema, and the
  untrained-head refusal. `src/heuristics.py` is gone. I found no fallback in
  `app/backend/`. The heuristic I did find is in the **frontend** (R2-03).
* **`scripts/collect_results.py`.** Re-read in full: it emits only what an
  artifact contains and the literal `not run` otherwise. No default, no cached
  constant, no reconstruction. Its one problem remains the causal sentence at
  line 149.
* **Test-suite honesty.** 175 checks, 0 failed, 0 errors — the commit message's
  figure is exact. `tests/helpers.py` labels the stub encoder
  `TEST-STUB-NOT-A-REAL-ENCODER` and lives outside `src/`; no production path
  imports it. `tests/test_pipeline.py`'s docstring states plainly that nothing
  it measures says anything about accuracy.

---

## 5. Not verifiable in this environment

No GPU, no ESM-2 weights, no ViennaRNA, no MMseqs2, no MegaScale/ClinVar/PED
data. Everything below needs one of those and should be checked before the real
run, in this order.

1. **GPU determinism (R1-01).** My reproducibility result is CPU, `num_workers:
   0`, `strict_determinism: false`. Re-run the two-invocation check on the 5070
   Ti with `num_workers: 4` and confirm bit-identical `metrics.json`; if not,
   set `strict_determinism: true` and re-check. Until then R1-01 is verified only
   for the CPU path.
2. **Peak VRAM (R2-02, R1-16).** Run `python -m scripts.budget --device cuda`
   for `--window 256 --batch-size 64` *and* for a whole-sequence batch at
   `max_sequence_nodes 2048`. My 33 GB figure is a scaled lower bound, not a
   measurement. Fix the `/ 2` in the GAT term before trusting either.
3. **`cds_coverage` against a real cache (R2-01).** After precompute with a real
   CDS and ViennaRNA installed, load a sample of cache records and confirm
   `mech[:, CH_VALID].mean() > 0` and that `E_mech` has more than `n` edges.
   Cross-check against the manifest. Do this *before* interpreting any
   `no_mech_stream` delta.
4. **Contact-head quality (R2-13).** Run `scripts/contact_sanity.py` with
   `--true-contacts-dir` over a few hundred PDB-derived contact maps and report
   precision@L and precision@L/5 against the proximity baseline **and** against a
   random-edge control. The long-range fraction alone establishes nothing.
5. **Calibration (R2-09).** Whether `well_calibrated` and the served σ agree
   depends on the fitted `variance_scale`, which needs a trained model. Confirm
   the interface reports the post-scaling numbers once the fix lands.
6. **Reversibility magnitude (R1-11).** The check now compares like with like
   structurally, but the actual violation in kcal/mol needs a trained model and
   the mutant encoder pass. This is the bound on what the substitution-conditioned
   formulation gives up; it should be in the paper.
7. **Split quality with MMseqs2 present (R2-04).** With `mmseqs` installed,
   confirm `grouping.clustering` reads `mmseqs2 @ 30% identity` rather than the
   exact-sequence fallback, and check `n_groups` against `n_parent_sequences`. My
   whole runs used the fallback path.
8. **ΔΔG sign on the real MegaScale table (R1-19).** The warning fires only for
   `len(out) >= 100`. Confirm which branch it takes on the real table and that
   `mean_ddg_after_sign_normalisation` is negative, before anything downstream is
   read.
9. **Ablation deltas (R2-06).** Nothing about the actual effect sizes can be
   assessed here. Run ≥5 seeds per arm, apply a paired test with multiplicity
   control over a pre-registered metric set, and report intervals.
10. **Wall clock and throughput.** `wall_clock_hours` is recorded but the 2–3 day
    budget is unvalidated. The 118 ms/sample `build_mechanistic_graph` cost at
    n=2048 with a real CDS (R2-02) is a CPU measurement here and should be
    re-timed on the target machine.
