# Review round 1

## Verdict

**Reject in current form / major revision required.** The architecture is a real
improvement over v1 and several of its structural claims genuinely hold (exact
head antisymmetry, swap-equivariant three-state, padding invariance, the
associative scan, mutation-centered windowing for the mutation-conditioned
tasks). But this code **cannot produce the paper it is meant to back**: training
is not seeded at all, so two runs of the same command with `seed: 42` differ by
0.03 Pearson (R1-01) and every ablation delta is uninterpretable; the multi-task
loop retains ~354 GB of batches in host RAM at MegaScale scale and will OOM
before epoch 1 finishes (R1-03); the ClinVar label mapping silently calls every
`Conflicting_interpretations_of_pathogenicity` variant pathogenic (R1-04); and
the documented one-command pipeline never wires the recovered CDS into the run,
so the mechanistic stream — the paper's distinguishing contribution — is inactive
by default and its ablation would read ~0 for the wrong reason (R1-05).

Most seriously for a submission whose entire rebuttal is *"we removed the silent
fallbacks"*: the exact v1 failure the MIGRATION_NOTES claim to have fixed (NaN
primary metric → no checkpoint → results reported anyway) is fully reproducible,
and the guard written to catch it is provably unreachable (R1-02). Several
docstring numbers (`0.252` long-range fraction, `~0.8 GB` peak) are asserted as
measurements this repository produced, with no artifact and, for the first, no
possible provenance.

## Summary of findings

| ID | Severity | One-line description |
|---|---|---|
| R1-01 | BLOCKER | Nothing in the training path is seeded; `seed: 42` controls only one 90/10 split. Two identical runs differ by 0.03 test Pearson. |
| R1-02 | BLOCKER | Empty val split → NaN metric → no checkpoint ever written → test metrics reported from the last-epoch model, exit 0. The guard is unreachable. |
| R1-03 | BLOCKER | `itertools.cycle` over DataLoaders retains every batch: ~354 GB host RAM for a MegaScale-scale epoch on a 32 GB machine. |
| R1-04 | BLOCKER | `_binary_label` maps `Conflicting_*_of_pathogenicity` → pathogenic, and numeric CLNSIG Benign(2)/Likely-benign(3) → pathogenic. Counted as kept. |
| R1-05 | BLOCKER | `run.py all` writes the CDS-joined table and never uses it; the mechanistic stream is silently off in the documented pipeline. |
| R1-06 | MAJOR | `filter_to_cache` is never applied to the stability/pathogenicity split tables → `FileNotFoundError` mid-training, no exclusion accounting for the two primary tasks. |
| R1-07 | MAJOR | Whole-sequence tasks (IDR Rg, MFI, proteomics) are head-truncated to residues `[0,256)` — the exact failure the paper says it eliminated. |
| R1-08 | MAJOR | `build_sequence_loaders` splits by row permutation, not unique sequence; no clustering, no leakage report, and no test split for three of six heads. |
| R1-09 | MAJOR | The Kendall & Gal objective is unbounded below: with a negative Gaussian NLL, `log_var → −43` and total loss → −1.9e19. |
| R1-10 | MAJOR | `steps_per_epoch = max(...)` replays a 300-row table ~844×/epoch; K&G then gives every task the same 1/2 share of the shared gradient. |
| R1-11 | MAJOR | The reversibility check compares a dual-stream forward pass against a mech-free reverse pass (472 vs 64 E_mech edges), and poisons the encoder cache. |
| R1-12 | MAJOR | Baselines are handicapped/leaky: zero-shot is scored on a 256-residue window; with no val split `esm_mlp` is early-stopped on the test set it is scored on. |
| R1-13 | MAJOR | The deep mutational scan costs **L** encoder passes on the identical sequence, not "one". Claim contradicted in README and `src/substitution.py`. |
| R1-14 | MAJOR | `uncertainty_calibrated` flips to `true` on the mere existence of `calibration.json`, at 11% observed coverage for a 95% nominal band. |
| R1-15 | MAJOR | `ThreeStateHead` is trained, shipped in the README head table and served by the API, but is never evaluated anywhere; `pos_weight`/`class_weights` are dead. |
| R1-16 | MAJOR | `scripts/budget.py` estimates 0.78 GB at batch 64; the GATv2 edge tensors alone are 2.4 GB and a measured fwd+bwd extrapolates to ~9.2 GB. |
| R1-17 | MAJOR | Graphs are rebuilt from scratch every epoch; `_cap_degree` is a Python loop costing 34 ms/sample → ~0.7 h/epoch of data loading, ×60 epochs ×4 models. |
| R1-18 | MAJOR | Resume restores neither `patience`, nor RNG state, nor the `GradScaler`; failure to resume silently continues from a hybrid state. |
| R1-19 | MAJOR | The ΔΔG sign convention lives only in a docstring, is contradicted in `baselines.py`, and is invisible to Pearson r — an inverted table passes every metric. |
| R1-20 | MAJOR | `src/encoder.py:17` asserts a `0.252` long-range contact fraction "measured" by this repository's own script, with no artifact and no possible ESM-2 run. |
| R1-21 | MINOR | `logvar` is clamped to [−10,10] in the loss but used unclamped at eval/serve; `exp(lv_wt)+exp(lv_mt)` overflows in fp16 at lv > 11.09 with `amp: true`. |
| R1-22 | MINOR | Three modules carry stale v1 docstrings (`sequence_mixers.py` "default to OFF" / §8.2 / 320-3-323 dims; `pooling.py` V5/V6; `idr_prior.py`), two of them dead code. |
| R1-23 | MINOR | "Exact reversibility … in train mode" holds only for a shared context vector; two independent training-mode calls violate it by 4.7e-2 kcal/mol. |
| R1-24 | MINOR | Non-standard residues (B/Z/X/U/O): volume channel 0.0 (below glycine), no unknown flag, scan has no zero wild-type column, and the API rejects them outright. |
| R1-25 | MINOR | The interface `verdict` never says "neutral" and contradicts the three-state probabilities in the same payload; `metrics.json` writes invalid JSON (`-Infinity`). |
| R1-26 | MINOR | Encoder chunking takes 128 residues per boundary from the *shorter* chunk, contradicting "representations always come from the most context available". |
| R1-27 | MINOR | The ablation grid omits the two load-bearing components (site-anchored readout, E_geo), and the `esm_mlp` gap is attributed to "message passing" alone. |

---

## Findings

### R1-01 [BLOCKER] Nothing in the training path is seeded

**Location:** `scripts/train.py` (whole file — the only seed use is line 134);
`configs/run.yaml:44` (`seed: 42`); `scripts/baselines.py:157-161`;
`scripts/ablations.py:314-318`

**Problem:** There is no `torch.manual_seed`, no `numpy.random.seed`, no
`random.seed`, no DataLoader `generator=`, no `worker_init_fn` and no cuDNN
determinism setting anywhere in `src/`, `scripts/`, `run.py` or `app/`. The
config's `seed: 42` reaches exactly two places: `src/splits.assign_groups_to_splits`
(via `make_splits`) and the 90/10 row permutation at `scripts/train.py:134`.
Model initialisation, dropout, and batch order are all unseeded. `esm_mlp` in
`scripts/baselines.py` is likewise unseeded.

**Why it matters:** Every ablation arm in `scripts/ablations.py` is a single
unseeded run, and `delta_vs_full` is reported to four decimals with no error
bars. I measured a run-to-run spread of **0.030 test Pearson** on identical
data with the same config — larger than most published ablation effects. No
delta this pipeline reports can be distinguished from initialisation noise, and
no third party can reproduce a number from this repository even with the data.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p7_seed.py
run1: best_epoch=3 best_val=0.8260542498185235 test={'pearson': 0.867316, 'spearman': 0.858347, 'rmse': 0.84263, ...}
run2: best_epoch=3 best_val=0.9099813375012522 test={'pearson': 0.897504, 'spearman': 0.882978, 'rmse': 0.758258, ...}

SEED IS 42 IN THE CONFIG FOR BOTH RUNS.
identical test metrics? -> False
```
(two invocations of `python -m scripts.train --config <same> --stability-splits <same> --device cpu`)

```
$ grep -rn "manual_seed\|np.random.seed\|random.seed\|worker_init\|use_deterministic\|cudnn" --include=*.py src scripts run.py app
scripts/train.py:134:    rng = np.random.default_rng(cfg.seed)
```
(the remaining hits are the word "deterministic" in prose docstrings)

**Fix direction:** Add a `seed_everything(cfg.seed)` at the top of `train.py`,
`baselines.py` and `calibration.py` covering `random`, `numpy`, `torch`,
`torch.cuda`; pass `generator=torch.Generator().manual_seed(seed)` and a
`worker_init_fn` to every `DataLoader`; set
`torch.use_deterministic_algorithms(True)` where feasible. Then run each
ablation arm over ≥3 seeds and report mean ± sd, not a single delta.

---

### R1-02 [BLOCKER] NaN primary metric silently produces no checkpoint, and the test numbers are reported anyway

**Location:** `scripts/train.py:374-384` (the guard), `398` (`improved`),
`435-437` (best-checkpoint reload), `459` (`validated_by_real_run`);
`src/metrics.py:42-49`

**Problem:** Two separate defects compose.

1. The guard at line 375, `if primary_val and primary_metric not in primary_val:
   raise KeyError(...)`, **can never fire**. When a val loader exists,
   `regression_metrics`/`classification_metrics` always include the key —
   `src/metrics.py:44` initialises `"pearson": float("nan")` and returns it
   unconditionally. When no val loader exists, `primary_val` is `{}` and the
   `if primary_val` short-circuits. MIGRATION_NOTES ("Other fixes") states "A
   missing key now raises"; it cannot.

2. When the primary task has no val split (a group-level 80/10/10 on few groups,
   or any table where the val side is empty), `score` is NaN forever,
   `improved` is `False` forever, **`model_best.pt` is never written**, and the
   `if os.path.exists(ckpt_path)` reload at line 435 silently no-ops — so the
   reported test metrics come from whatever the last epoch left in memory, with
   nothing in the manifest saying so. The process exits 0, so `run.py all` and
   `scripts/ablations.py` treat it as a completed run, and the manifest still
   records `"validated_by_real_run": true`.

**Why it matters:** This is precisely the v1 failure the migration notes present
as fixed ("nothing ever improved, no checkpoint was written, early stopping
never fired… A full training run would have ended with no model"). Downstream,
`run.py calibration` prints "no checkpoint" and skips, `runapp.py` reports "no
model loaded", while `results/SUMMARY.md` carries a headline Pearson from a
model that no longer exists on disk.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p8_noval.py
original split counts: {'train': 251, 'test': 38, 'val': 26}
modified split counts: {'train': 277, 'test': 38}
[train] epoch 1/3  loss=2.7940  stability.pearson=nan  |
[train] epoch 2/3  loss=1.8315  stability.pearson=nan  |
[train] epoch 3/3  loss=1.1901  stability.pearson=nan  |
[train] best epoch -1  best val stability.pearson=-inf
[train] test stability: {'n': 38, 'pearson': 0.9738084261369842, ...}
exit: 0
model_best.pt written? -> False
manifest best_epoch: -1  best_val: -inf
manifest still claims validated_by_real_run: True
manifest TEST metrics (from which model?): {'pearson': 0.9738, ...}
```
and, for the guard's unreachability:
```
$ /home/user/.venv/bin/python /tmp/probe/p3_guard.py
constant-prediction regression metrics: {'n': 64.0, 'pearson': nan, ...}
'pearson' in metrics -> True  value: nan
train.py guard `primary_metric not in primary_val` fires? -> False
single-class classification 'auroc' present: True value nan
guard fires? -> False
```

**Fix direction:** Fail hard at loader-construction time if the primary task has
no val split. Replace the key-presence guard with a value guard: raise if
`not np.isfinite(score)` for more than one consecutive epoch. Refuse to write
`metrics.json` (or write `"best_checkpoint": null` and omit `test`) when
`best_epoch < 0`. Delete `validated_by_real_run` — a constant `True` in a file
written by the run itself carries no information.

---

### R1-03 [BLOCKER] `itertools.cycle` retains every batch: ~354 GB of host RAM per epoch

**Location:** `scripts/train.py:349`

**Problem:** `cycles = {t: itertools.cycle(v["train"]) for t, v in loaders.items()}`.
`itertools.cycle` saves a copy of every element it yields so it can replay them.
Each element here is a fully collated batch whose `esm` tensor alone is
`(64, 256, 1280)` fp32 = 83.9 MB. A single pass over the training loader
therefore accumulates the whole epoch's batches in RAM, for every task
simultaneously.

**Why it matters:** MegaScale is ~270k mutations → 4,219 batches → **354 GB**
retained, on a machine documented as having 32 GB (`MIGRATION_NOTES.md`, "Running
it on the target machine"). Training will be OOM-killed part-way through epoch 1.
This is invisible in `tests/test_pipeline.py`, which uses ~300 rows at
`hidden: 32`. A secondary consequence: within an epoch `cycle` replays the saved
batches in the *same order with the same composition*, so a small task's
"extra" passes are byte-identical repeats rather than reshuffled data.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p12_mem.py
shipped shape: esm batch = (64,256,1280) fp32 = 83.9 MB
RSS after ONE pass of 30 batches through itertools.cycle: +2.43 GB  (tensors alone = 2.52 GB)
extrapolated for a 270,000-row epoch (4218 batches): 354 GB retained. Target machine: 32 GB RAM.
```
and, for the no-reshuffle behaviour:
```
$ /home/user/.venv/bin/python /tmp/probe/p2_cycle.py
first  pass order: [[1, 8], [4, 7], [9, 0], [2, 5], [6, 3]]
second pass order: [[1, 8], [4, 7], [9, 0], [2, 5], [6, 3]]
identical (no reshuffle): True
```

**Fix direction:** Replace `itertools.cycle(loader)` with a helper that
re-creates the iterator on `StopIteration`:
```python
def endless(loader):
    while True:
        yield from loader
```
This holds one batch at a time and reshuffles on each pass.

---

### R1-04 [BLOCKER] ClinVar "conflicting interpretations" is labelled pathogenic

**Location:** `src/data.py:198-215` (`_binary_label`), reached from `:184`;
candidate column list at `src/data.py:54-55` explicitly includes `clnsig`

**Problem:** The string branch tests `if "benign" in s: return 0` then
`if "pathogenic" in s or "deleterious" in s: return 1`. ClinVar's
`Conflicting_interpretations_of_pathogenicity` (and its 2023+ rename
`Conflicting_classifications_of_pathogenicity`) contains the substring
`pathogenic`, so it is labelled **1 = pathogenic**. Separately, the numeric
branch at line 203 returns `int(value != 0)`, so the legacy integer CLNSIG
encoding (0 Uncertain, 2 Benign, 3 Likely benign, 4 Likely pathogenic,
5 Pathogenic) maps Benign and Likely-benign to **pathogenic**.

**Why it matters:** Conflicting-classification missense variants are one of the
largest CLNSIG classes and are explicitly *not* a pathogenic call — ACMG
guidance treats them as unresolved. They are the hardest, most ambiguous
variants, and assigning them all to the positive class both corrupts the labels
and inflates the apparent positive rate. Crucially these rows are **kept**, so
`LoadReport` records nothing: the leakage report will show 0 dropped for this
reason and the corruption is invisible in every artifact. This is the same class
of defect as the predecessor's nucleotides-as-amino-acids bug that
MIGRATION_NOTES ("HoloGNN's `ClinVarDataset`") holds up as the reason that repo's
ClinVar task was meaningless.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p20_clnsig.py
CLNSIG value                                               -> label
  Conflicting_interpretations_of_pathogenicity             -> 1   <== WRONG: conflicting -> 'pathogenic'
  Conflicting_classifications_of_pathogenicity             -> 1   <== WRONG: conflicting -> 'pathogenic'
  Conflicting_interpretations_of_pathogenicity|other       -> 1   <== WRONG: conflicting -> 'pathogenic'
  non-pathogenic                                           -> 1   <== WRONG
  Uncertain_significance                                   -> None
numeric CLNSIG (legacy integer encoding: 0=Uncertain 2=Benign 3=Likely benign 4=Likely path 5=Path):
  CLNSIG=2 -> label 1   <== WRONG: 'Benign' becomes pathogenic
  CLNSIG=3 -> label 1   <== WRONG: 'Benign' becomes pathogenic
```

**Fix direction:** Replace substring matching with an explicit allow-list keyed
on the exact CLNSIG vocabulary, dropping anything not on it and counting it in
`LoadReport` (`unmappable_label`). Reject `conflicting`, `uncertain`,
`not_provided`, `drug_response`, `risk_factor`, `association`, `protective`
before the pathogenic test. Delete the numeric branch or require an explicit
per-dataset `label_encoding` key in `configs/data.yaml`; silently interpreting
"any nonzero" as pathogenic is not recoverable from the artifacts.

---

### R1-05 [BLOCKER] `run.py all` never uses the CDS it recovers — the mechanistic stream is silently inactive

**Location:** `run.py:100-114` (`stage_etl`), `run.py:117-125`
(`stage_precompute`), `run.py:211-222` (`stage_all`)

**Problem:** `stage_etl` writes `<stability_data>_cds.parquet` and then **prints**
`"[etl] set stability_data: … in configs/run.yaml to use it."` It does not
update the config, and `stage_all` immediately calls `stage_precompute` with the
unmodified `cfg.stability_data`. Every downstream stage — precompute, splits,
train, baselines, ablations — reads the original table with no `wt_dna` column,
so `mech_valid` is 0 everywhere and `E_mech` contains only self-loops.

**Why it matters:** The dual-stream mechanistic graph is the paper's
distinguishing contribution, and `scripts/ablations.py:11-18` states explicitly
that `no_mech_stream` "has never been tested against a real coding sequence…
With a real CDS the question is open again, and this arm is how it gets
answered." Run as documented (`python run.py all --cds-source …`), that arm
measures a stream that was never fed. The delta will read ≈0 and would be
written up as "the mechanistic stream does not contribute" — the wrong
conclusion from a plumbing bug. This is a silent degradation of exactly the kind
`src/model.py:15-18` and `src/backbone.py:245-248` say has been removed.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p15_etl.py
$ ... -m scripts.etl_cds --table .../synthetic_stability.parquet --source .../cds_source.csv --out .../synthetic_stability_cds.parquet
[etl] CDS attached to 14/14 rows (100.0%)
[etl] set stability_data: /tmp/.../synthetic_stability_cds.parquet in configs/run.yaml to use it.
exit 0
configs/run.yaml modified by the etl stage? -> False
cds table produced: True
cfg.stability_data still points to: synthetic_stability.parquet
run.py all -> stage_precompute(cfg.stability_data) will read the NON-CDS table.
wt_dna column present in produced table: True | non-null: 14 / 14
```

**Fix direction:** Have `stage_etl` return the produced path and thread it
through `stage_precompute`/`stage_splits` in-memory (or rewrite the YAML). Then
make `precompute` refuse to proceed when `--cds-source` was supplied but
`parents with a real coding sequence: 0/N`, and record the CDS coverage fraction
in the run manifest so the mechanistic ablation cannot be read without it.

---

### R1-06 [MAJOR] `filter_to_cache` is never applied to the two primary tasks

**Location:** `src/data.py:276-290` (defined); `scripts/train.py:130` (the only
call site, inside `build_sequence_loaders`); `scripts/train.py:104-115`
(`build_split_loaders`, no call)

**Problem:** `build_split_loaders` constructs a `MutationDataset` directly from
the split table. `MutationDataset.__getitem__` calls `store.get(row["parent_seq"])`,
which `torch.load`s a path that may not exist. A parent whose encoding failed in
precompute (`scripts/precompute_embeddings.py:150-152` swallows every exception
and only increments a counter) therefore crashes training with an uncaught
`FileNotFoundError` from inside a DataLoader worker.

**Why it matters:** `src/data.py:16-22` and MIGRATION_NOTES ("Exclusions are
counted") make the loud-exclusion accounting a headline fix: *"v1 dropped 2.81%
of ClinVar for having no cached embedding, and that never reached the results."*
The mechanism to count it exists and is wired into the three optional
whole-sequence tasks only. For stability and pathogenicity — the two tasks the
paper reports — there is neither counting nor filtering, just a crash that can
land 20 epochs into a two-day run.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p9.py
=== B. missing cache entry in a stability split table ===
removed 1 of 24 cache files
exit code: 1
stderr tail: [..., "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/.../cache/4ac83f44….pt'"]

$ grep -rn "filter_to_cache" --include=*.py .
./src/data.py:276:def filter_to_cache(
./scripts/train.py:130:        df = filter_to_cache(df, store, report)
```

**Fix direction:** Call `filter_to_cache` in `build_split_loaders` for every
split, and write the resulting `LoadReport` into `metrics.json`. Better still,
run the check once in `make_splits` so the exclusion is recorded in the leakage
report alongside every other dropped row, and make `precompute` exit non-zero
when `failures > 0`.

---

### R1-07 [MAJOR] Whole-sequence tasks are head-truncated to the first 256 residues

**Location:** `src/data.py:311` (`anchor = 0 if site is None else …`),
`src/data.py:405` (`SequenceDataset.__getitem__`),
`app/backend/inference.py:281` (`self._context_for(seq, 0)`)

**Problem:** `SequenceDataset` calls `build_sample_graph(rec, None, window, cfg)`.
With `site is None` the anchor is 0, so `window_bounds(length, 0, 256)` returns
`(0, 256)` for any protein longer than the window. IDR radius of gyration, MFI
and retention time are therefore trained and served on residues `[0, 256)` only.

**Why it matters:** This is head truncation — the failure `src/windowing.py:8-18`
and the whole "Mutation-centered windowing, restored" section of MIGRATION_NOTES
are written to eliminate — reintroduced for three of the six heads. It is worst
for the IDR task: radius of gyration is dominated by chain length (`R_g = R_0
N^ν`, as the repository's own `src/idr_prior.py:15` states), and a 900-residue
IDR and a 300-residue IDR sharing a 256-residue N-terminus are indistinguishable
to this model. The mask covers only 256 positions, so `global_only` cannot even
recover length. Predicting Rg from a fixed-length prefix while the label is for
the whole chain is not a hard task, it is an ill-posed one.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p5b.py
SequenceDataset window for a 900-residue protein: n=256 nodes, site=0
 -> whole-sequence (IDR/Rg, retention time) tasks see only residues [0,256).
```

**Fix direction:** For whole-sequence tasks either (a) pool over the full-length
cached embedding (no windowing — there is no site to anchor on and the encoder
cache is already full length), or (b) tile the sequence into windows and pool
across them, and in both cases feed chain length to the head explicitly. If
neither is done, drop the IDR/proteomics heads from the paper's claims.

---

### R1-08 [MAJOR] The whole-sequence split is a row permutation, not a sequence-disjoint split

**Location:** `scripts/train.py:133-142`

**Problem:** The comment at line 133 reads *"Sequence-disjoint by construction:
the group key is the sequence itself."* The code below it permutes **row
indices** (`order = rng.permutation(len(df))`) and cuts at 90%. There is no
`groupby`, no dedup on `parent_seq`, and no use of `src/splits.py`. A sequence
appearing in two rows can land on both sides.

**Why it matters:** README ("Evaluation") states *"Every task is split with
MMseqs2 identity clustering unioned with gene identity, whole groups to one
side"*, and MIGRATION_NOTES ("Leakage control on every task") repeats it. Neither
is true for IDR, MFI or proteomics: no clustering, no union-find, no gene
relation, no leakage report, and **no test split at all** — only train/val, so
any number these heads produce is the split early stopping ran on. Repeated
sequences are the norm in retention-time and expression corpora (the same peptide
across runs) and common in curated Rg tables.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p14.py
=== B. build_sequence_loaders 'sequence-disjoint by construction' ===
 comment says: ['# Sequence-disjoint by construction: the group key is the sequence itself.']
 rows split by np.random permutation of ROW index (train 9, val 1)
 sequences appearing in BOTH train and val: ['S5']  -> leakage: True
```

**Fix direction:** Route these tables through `scripts/make_splits.py` like the
other two (they already carry a `group` column set to the sequence), produce a
three-way split with a leakage report, and either report test metrics for these
heads or state in the paper that they are unevaluated.

---

### R1-09 [MAJOR] The Kendall & Gal objective is unbounded below

**Location:** `src/losses.py:118` (`weighted = value * torch.exp(-lv) + 0.5 * lv`);
`src/losses.py:38-59` (`gaussian_nll`); `scripts/train.py:153`

**Problem:** Kendall & Gal's derivation assumes a non-negative task loss. The
primary task's loss here is a heteroscedastic Gaussian NLL, which is negative
whenever the predictive density exceeds 1 — with the `[-10, 10]` logvar clamp,
its minimum is **−4.081**. For a negative `value`, `value·exp(−lv)` is minimised
by driving `lv → −∞`, and the total is unbounded below. There is no lower clamp
on `log_var` and no floor on the per-task loss.

**Why it matters:** ΔΔG NLL goes negative once the typical predicted σ drops
below ≈0.24 kcal/mol, which is entirely reachable on MegaScale where a large
fraction of mutations are near-neutral and well predicted. Once it does, the
stability weight explodes (I measured `exp(−lv) = 4.7e18` after 300 steps), the
global `clip_grad_norm_` at `scripts/train.py:362` then allocates the entire
gradient budget to stability, and every other head stops learning — while the
printed "loss" goes hugely negative and looks like spectacular progress. The
separate, milder degeneracy the review brief asks about is also present and
confirmed: each weight converges to `1/(2·L_i)`, so a task's contribution to the
shared backbone is `L_i · 1/(2 L_i) = 1/2` regardless of its difficulty or size.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p6_kg.py
min achievable gaussian_nll (logvar clamped at -10, perfect fit): -4.081061840057373
  step    0  total=      -3.000 log_var(stab)=  -0.100 weight(stab)=1.000e+00
  step  100  total=-14056759.000 log_var(stab)= -15.272 weight(stab)=3.514e+06
  step  200  total=-92202021448318976.000 log_var(stab)= -37.923 weight(stab)=2.305e+16
  step  300  total=-18787537702738722816.000 log_var(stab)= -42.950 weight(stab)=4.697e+18
after 3000 steps with a fixed easy(0.05)/hard(8.0) pair:
  weight(easy)=9.9922  weight(hard)=0.0631  ratio=158.3
  analytic optimum weight_i = 1/(2*L_i): 10.0 0.0625
```

**Fix direction:** Either shift the NLL to be non-negative (drop the
`0.5·log 2π` constant and add a per-task offset, or use `softplus` on the loss),
or clamp `log_var` to a bounded range (e.g. `[-3, 3]`) and monitor saturation.
Log `exp(-lv)` per task every epoch and abort if any weight exceeds a threshold.

---

### R1-10 [MAJOR] The multi-task step budget oversamples small tasks by three orders of magnitude

**Location:** `scripts/train.py:305` (`steps_per_epoch = max(...)`),
`scripts/train.py:349` (`itertools.cycle`), `scripts/train.py:356-358`

**Problem:** Every task takes exactly `steps_per_epoch` optimizer contributions
per epoch, where `steps_per_epoch` is the **largest** loader's length. A small
task's loader is cycled to fill the gap. This is the reverse of the bias the
review brief anticipates: the loop is not biased toward the task with the most
batches, it forcibly equalises them, which massively over-represents the small
ones.

**Why it matters:** At the shipped `batch_size: 64`, a 300-row experimental Rg
table is replayed **~844 times per epoch** (and, per R1-03, replayed as the same
five batches in the same order). Multiplied by the R1-09 result that each task
converges to a `1/2` share of the shared-backbone gradient, a 300-row table
gets the same gradient budget as a 270,000-row ΔΔG table while being seen 844×
more often. The backbone will overfit the tiny tables and the paper's headline
ΔΔG numbers will be degraded by heads the paper barely reports. Nothing in
`configs/run.yaml` can change this ratio.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p21_multitask.py
  stability (MegaScale)       270000 rows =  4219 batches
  pathogenicity (ClinVar)      60000 rows =   938 batches
  idr (PED/SAXS Rg)              300 rows =     5 batches
  steps_per_epoch = max(...) = 4219
  -> stability     : its  4219 batches are replayed      1.0x per 'epoch'
  -> pathogenicity : its   938 batches are replayed      4.5x per 'epoch'
  -> idr           : its     5 batches are replayed    843.8x per 'epoch'
  -> mfi           : its    32 batches are replayed    131.8x per 'epoch'
```

**Fix direction:** Sample tasks per step in proportion to `len(loader)` (or to
`sqrt(len)`) rather than stepping all of them every iteration, and expose the
sampling temperature in the config. Report the realised per-task step counts in
the manifest.

---

### R1-11 [MAJOR] The reversibility check compares a dual-stream model against a mech-free one, and writes mutants into the encoder cache

**Location:** `scripts/calibration.py:129-150`, especially `:134` and `:135`

**Problem:** The check re-encodes the mutant as its own parent, then sets
`mut_rec["mech"] = mechanistic_features(mutant, None, len(mutant))` — a hardcoded
`dna_seq=None`, so `mech_valid = 0` and `E_mech` is self-loops only. The parent
record read from the precompute cache has `mech_valid = 1` and a fully populated
`E_mech` whenever a CDS was attached. The forward direction is therefore
evaluated through the dual-stream model and the reverse through an effectively
single-stream one. Separately, line 135 calls `store.put(mut_rec)`, permanently
inserting mutant sequences (with `mech_valid = 0`) into the shared encoder cache
from an evaluation script.

**Why it matters:** `MIGRATION_NOTES.md` ("Substitution-conditioned instead of
twin-tower") and `scripts/train.py:17-25` name this check as the honest
measurement that mitigates what the substitution-conditioned formulation loses
versus encoding the mutant protein. As written it cannot serve that role: any
measured violation is confounded with an architectural asymmetry the check
itself introduces, and the confound is largest exactly when the mechanistic
stream is doing the most work. The cache write also silently changes what
`store.has` returns for later runs.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p14.py
=== A. calibration.py reversibility: parent has a CDS, the re-encoded mutant does not ===
 parent : mech_valid mean=1.00  E_mech edges=472
 mutant : mech_valid mean=0.00  E_mech edges=64
 -> the forward direction runs the dual-stream model, the reverse a mech-free one.
```

**Fix direction:** Carry the parent's CDS through the substitution (a single
codon change at `site`) and recompute the mechanistic channels for the mutant
from it, or run the check with `use_mech_stream=False` on both sides and say so.
Write mutant encodings to a separate scratch store, never `cfg.cache`.

**On the underlying methodological question** (asked in the brief, and worth
stating plainly in the paper): encoding the parent once and conditioning on
residue identity gives exact algebraic antisymmetry and cheap scans, but it
means the model never sees the mutant's own representation — so it cannot
capture effects that change the fold, the local secondary structure or the
contact map (proline introduction, glycine in a turn, buried-charge burial,
disulfide loss). Those are a large share of the strongly destabilising tail. The
reversibility check is the right instrument for bounding that loss, which makes
fixing it a prerequisite for the claim, not an optional extra.

---

### R1-12 [MAJOR] The baselines are handicapped and, in one branch, selected on the test set

**Location:** `scripts/baselines.py:86-89` (zero-shot window), `:179`
(`loaders.get("val", loaders["test"])`), `:237-238` (`if val.empty: val = test`)

**Problem:** Two separate issues in the control arm.

1. `esm_zero_shot_baseline` scores the masked-LM marginal on the **256-residue
   window** (`sub = seq[start:end]`), with the comment "so the comparison is
   fair". It is not fair: the model's own inputs are per-residue embeddings
   computed by ESM-2 over the **whole protein** and cached at full length
   (`src/encoder.py:21-27`, `src/windowing.py:19-23` — "a residue representation
   computed with the whole protein in context is strictly more informative").
   The baseline is given strictly less context than the model.
2. When the split table has no val rows, `val = test` at line 237 and
   `loaders.get("val", loaders["test"])` at line 179 both fall through to the
   test set, so `esm_mlp` — described in README and `collect_results.py:147-149`
   as "the control that matters" — is early-stopped on the very split it is then
   scored on.

**Why it matters:** The paper's central empirical claim is *"the gap between
[esm_mlp] and the full model is the message-passing contribution."* Handicapping
one control and leaking into the other makes that gap unmeasurable in the
direction it is being claimed.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p21_multitask.py
=== esm_mlp baseline model selection when the split has no val rows ===
 scripts/baselines.py: ['if val.empty:']
 scripts/baselines.py: ['for x, y in loaders.get("val", loaders["test"]):']
 -> with no val split the MLP baseline is early-stopped on the TEST set it is then scored on.
```
and by reading `scripts/baselines.py:87-89` against `src/data.py:309-315`, which
slices the full-length cached embedding rather than re-encoding the window.

**Fix direction:** Score the zero-shot baseline on the full sequence (with the
encoder's own chunking for >1022) and take the marginal at the true position.
Raise rather than falling back to `test` when val is missing.

---

### R1-13 [MAJOR] The deep mutational scan costs L encoder passes, not one

**Location:** `app/backend/inference.py:244-250` (the per-position loop) and
`:153-176` (`_context_for`); claims at `README.md` ("Free deep mutational scans…
a full 20 × L scan is tractable"), `src/substitution.py:20-22` ("a full 20 x L
scan costs **one encoder pass**"), `app/backend/inference.py:221-225`

**Problem:** `mutation_scan` loops over positions and calls `_context_for(seq,
site)` for each. `_context_for` calls `self._encoder.encode(seq, …)` — on the
**identical, unchanged sequence** every time. The parent's ESM-2 embedding does
not depend on `site` at all; it is recomputed from scratch once per position.

**Why it matters:** `src/substitution.py:20-22` asserts one encoder pass; the
implementation does L. With `MAX_SCAN_POSITIONS = 600` and the 650M encoder,
that is 600 identical ESM-2 forward passes plus 600 backbone passes for a single
heatmap. The genuine 40× saving over the predecessor's `20·L·2` is real and
worth stating — but "one pass" is off by a factor of L and the endpoint's
practicality claim rests on the wrong number.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p5b.py
DMS scan over 60 positions -> encoder.encode() called 60 times
distinct sequences given to the encoder: 1
all identical to the input sequence: True
```

**Fix direction:** Hoist `encoder.encode(seq)` and `mechanistic_features(seq, …)`
out of the loop (they are position-independent), build the batch of per-position
graphs once, and run the backbone in batches. Then the scan really is one
encoder pass. Correct the claim in `src/substitution.py:20-22` and the README to
"one encoder pass per protein" once that holds.

---

### R1-14 [MAJOR] `uncertainty_calibrated` reports the existence of a file, not calibration

**Location:** `app/backend/inference.py:114-116`, `:144`, `:207`

**Problem:** `self._calibration = json.load(fh).get("calibration")` if
`calibration.json` exists, and both `health()["uncertainty_calibrated"]` and the
`/api/ddg` payload report `bool(self._calibration)`. No coverage value, ECE or
sharpness figure is ever compared against anything. A calibration run that
measures catastrophic under-coverage flips the interface to "calibrated".

**Why it matters:** README ("Web interface") and MIGRATION_NOTES ("Uncertainty
that means something") position this flag as the fix for v1's fabricated
`ddg ± (0.4 + 0.12·|ddg|)` band, and the test suite asserts on it ("health admits
uncertainty is uncalibrated without a calibration run"). But the flag encodes
"someone ran the script", not "the interval is honest", and the interface draws
`μ ± 1.96σ` unchanged either way. There is also no *re*-calibration anywhere in
the repository — `scripts/calibration.py` measures and reports; nothing rescales
σ — so a measured miscalibration has no remedy in this codebase.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p5b.py
health.uncertainty_calibrated with 11% observed coverage at 95% nominal: True
ddg payload 'uncertainty_calibrated': True | ci_source: model predicted sigma (Gaussian, 95%)
```
(`calibration.json` written with `coverage_observed = {"95%": 0.11, …}`,
`expected_calibration_error = 0.49`)

**Fix direction:** Gate the flag on measured quality — e.g. every nominal level
within a tolerance of its observed coverage and ECE below a threshold — and
surface the observed coverage numbers in the payload, not just in `health()`.
Add a variance-rescaling step (a single scalar `a` minimising NLL on the val
split, applied to `logvar`) so a measured miscalibration can actually be fixed.

---

### R1-15 [MAJOR] The three-state head is trained and served but never evaluated

**Location:** `scripts/train.py:157-158` (trained), `:183-220` (`evaluate_task`,
no `three_state` branch), `:74` (`ALL_TASKS` lists it), `app/backend/inference.py:193`
and `:208-212` (served); `src/losses.py:131,138` (`class_weights`/`pos_weight`
never supplied)

**Problem:** `three_state` is a loss term, not an entry in `loaders`, so it never
reaches `evaluate_task`, which in any case has no branch for a 3-class task. No
accuracy, macro-F1, per-class recall or confusion matrix is computed anywhere in
`train.py`, `calibration.py`, `baselines.py` or `collect_results.py`. Separately,
`three_state_loss` and `pathogenicity_loss` both accept class-balancing weights
and no caller ever passes them.

**Why it matters:** `README.md` lists `ThreeStateHead` as one of six heads with a
"Trained when" column, and `/api/ddg` returns its softmax on every request. A
head the paper describes and the interface displays has zero reported
performance, so no reviewer can tell whether the swap-equivariant
parameterisation buys anything — and the class distribution it is trained on is
severely skewed (most MegaScale substitutions are destabilising), with the
balancing hooks left unwired. `collect_results.py` will silently omit it from
`SUMMARY.md`, so its absence is not even visible in the results file.

**Verified by:**
```
$ grep -rn "three_state" --include=*.py . | grep -v node_modules
./scripts/train.py:158:        losses["three_state"] = three_state_loss(model.three_state(batch, z), labels)
./app/backend/inference.py:193:            three = torch.softmax(self._model.three_state(batch, z), dim=-1)[0]
(no occurrence in evaluate_task, calibration.py, baselines.py or collect_results.py)

$ grep -rn "pos_weight\|class_weights" --include=*.py . | grep -v node_modules
./src/losses.py:131,134,138,142   (definitions only — no caller supplies either)
```

**Fix direction:** Add a `three_state` branch to `evaluate_task` reporting
macro-F1, per-class recall and the confusion matrix on the same split as
stability, and surface it in `collect_results.py`. Compute inverse-frequency
class weights from the train split and pass them; likewise `pos_weight` for
ClinVar.

---

### R1-16 [MAJOR] The memory budget under-states peak activation by an order of magnitude

**Location:** `scripts/budget.py` (the "activations ~6x node features"
estimate); `MIGRATION_NOTES.md:213-215`

**Problem:** The budget script reports `estimate at 64: 0.78 GB (activations ~6x
node features…)` and MIGRATION_NOTES turns that into *"estimated ~0.8 GB peak.
There is substantial headroom — larger batches are available if throughput needs
them."* The `6×` multiplier is a constant with no relation to the actual graph.
Two components each exceed the whole estimate on their own:

* `GATv2Conv` materialises a per-edge message tensor of shape
  `(E, heads, out_channels)` and keeps it for backward. At the script's own
  `E_geo = 145,024`, `heads = 4`, `hidden = 256` that is **0.59 GB per tensor**,
  and there are four (conv1+conv2 × geo+mech streams) = **2.38 GB**.
* `_PureSelectiveSSM._associative_scan` (`src/sequence_mixers.py:110-134`)
  allocates `a_prev` and `x_prev` and rewrites `a_c`/`x_c` at each of 8 steps,
  each a `(64, 256, 1280)` tensor of 84 MB, all retained by autograd —
  **~2.68 GB**.

**Why it matters:** The hardware plan in MIGRATION_NOTES is built on this
number, and it drives the "2–3 days on a 16 GB 5070 Ti" claim. A measured
forward+backward extrapolates to ~9.2 GB at batch 64 in fp32; even halved by
AMP this leaves no headroom on a 16 GB card, and the advice to raise the batch
size is wrong.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p22_mem.py
B=16 L=256 D=1280  E_geo=36256  fwd+bwd took 7.5s on CPU
peak RSS delta for ONE fwd+bwd: 2.31 GB
linear extrapolation to the shipped batch_size=64: 9.24 GB
scripts/budget.py estimate at 64: 0.78 GB ('activations ~6x node features')
```
(RSS on CPU in fp32 is a proxy, not a CUDA peak — but the two analytic figures
above, 2.38 GB and 2.68 GB, are independent of that measurement and each already
exceed 0.78 GB.)

**Fix direction:** Replace the `6×` constant with the analytic edge-tensor and
scan-intermediate terms; make `scripts/budget.py` refuse to print an "estimate"
label without a `torch.cuda.max_memory_allocated()` measurement. Consider
`torch.utils.checkpoint` on the scan, which is where most of the second term
lives.

---

### R1-17 [MAJOR] Graphs are rebuilt every epoch and `_cap_degree` dominates the wall clock

**Location:** `src/graphs.py:139-154` (`_cap_degree`), `src/data.py:296-352`
(`build_sample_graph`), `src/data.py:379-387` (`MutationDataset.__getitem__`)

**Problem:** `build_sample_graph` is a pure function of `(parent_seq, site, cfg)`
and is recomputed on every `__getitem__` on every epoch. Its cost is dominated by
`_cap_degree`, a Python loop over sorted edges that does two scalar tensor reads
(`int(edges[0, idx])`) per iteration.

**Why it matters:** I measured 35.3 ms per sample for `build_sample_graph` on a
500-residue protein at the shipped `graph_top_k`/`window`, of which 34.2 ms is
`_cap_degree`. At MegaScale scale with `num_workers: 4` that is **~0.7 h of
data loading per epoch**, ×60 epochs, ×4 models (`run.py all` trains `main` plus
three ablation arms) — roughly 160 hours of pure CPU preprocessing against a
stated 2–3 day budget, before a single GPU step is counted. This is fully
avoidable: nothing about the graph changes between epochs.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p11_perf.py
protein L=500 dim=1280: cache file 1.36 MB, 6689 cached contact edges
per __getitem__: torch.load 0.8 ms + build_sample_graph 35.3 ms = 36.1 ms
-> one batch of 64: 2310 ms of CPU work, 87 MB re-read from disk per batch
-> MegaScale-scale epoch (270000 rows) on 4 workers: 0.68 h of data-loading per epoch (config: 60 epochs)
   of which _cap_degree (python loop over 1761 edges): 34.2 ms
```

**Fix direction:** Vectorise `_cap_degree` (sort by probability, then use
`torch.cumsum` on a scatter of per-node counts, or accept an approximate cap via
per-node `topk`), and cache the built graph per `(seq_hash, site)` on disk or in
an LRU — mutations of the same parent at the same site recur across epochs and
across the reverse direction.

---

### R1-18 [MAJOR] Resume restores less state than it saves, and fails silently

**Location:** `scripts/train.py:330-344` (resume), `:418-425` (save), `:324`
(`patience = 0`), `:317` (`scaler`)

**Problem:** `last.pt` stores `{model, criterion, opt, sched, epoch, best_score,
best_epoch, history}`. Not saved, and therefore reset on every resume:

* `patience` — re-initialised to 0 at line 324. A run resumed more often than
  `early_stop_patience: 8` epochs can never early-stop.
* All RNG state (torch, numpy, python, CUDA) and the DataLoader shuffle order —
  a resumed run sees a different data order and different dropout masks. With
  R1-01 there is no seed to fall back on either.
* The `GradScaler` state — the AMP loss scale restarts at its default and
  re-converges, so the first steps after every resume are effectively skipped
  or run at the wrong scale.

Additionally, `except Exception as exc: print(…); starting fresh` at lines
343-344 catches a failure *after* `model.load_state_dict` may already have
succeeded, leaving a loaded model with a fresh optimizer and `start_epoch = 1` —
one printed line in a 60-epoch log.

**Why it matters:** README advertises the stages as "individually runnable and
resumable" and the plan is a 2–3 day run on one machine, i.e. resume is the
normal path, not the exception. Early stopping silently not firing changes which
epoch is selected; a hybrid resume state changes the optimisation trajectory
without any record in `history`.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p16_resume.py
keys SAVED in last.pt      : ['best_epoch', 'best_score', 'criterion', 'epoch', 'history', 'model', 'opt', 'sched']
keys LOADED on resume      : ['model','criterion','opt','sched','epoch','best_score','best_epoch','history']

NOT saved anywhere, therefore reset on every resume:
   - patience (early-stop counter)
   - torch/numpy/python RNG state
   - DataLoader shuffle order / worker seeds
   - GradScaler (AMP) state
   - steps_per_epoch / total_steps used to build the LR lambda
```

**Fix direction:** Add `patience`, `scaler.state_dict()`, and
`torch.get_rng_state()` / `torch.cuda.get_rng_state_all()` /
`np.random.get_state()` / `random.getstate()` to the checkpoint and restore them
all. Narrow the `except` to the specific errors you intend to tolerate and exit
non-zero on anything else; a partially-loaded resume must not proceed.

---

### R1-19 [MAJOR] The ΔΔG sign convention is asserted in a docstring, contradicted elsewhere, and invisible to the reported metric

**Location:** `app/backend/inference.py:20` (the only statement of the
convention), `app/backend/inference.py:213`; `src/heads.py:186-189`
(`labels_from_ddg`); `scripts/baselines.py:19-21` and `:103-106` (the contrary
statement); `src/data.py:52` (`target` candidates); `configs/data.yaml:20`

**Problem:** The convention is stated once, in a comment:
`app/backend/inference.py:20` — *"Sign convention (dG_mut - dG_wt): ddG > 0
stabilising, ddG < 0 destabilising."* `ThreeStateHead.labels_from_ddg` agrees
(`ddg > band → class 2`, mapped to `"stabilizing"` in the payload).
`scripts/baselines.py:19-21` states the opposite expectation: *"Higher score
means more tolerated, so the expected correlation with a destabilising ddG is
negative"* — under the stated convention a more-tolerated substitution has a
*higher* ΔΔG, so the expected zero-shot correlation is **positive**.

There is no sign check, no normalisation, and no per-dataset sign field. The
`target` column is auto-detected from `["ddg","ddG","delta_delta_g","ddg_exp",
"DDG_mean","target","score"]` — names drawn from sources that do not share a
convention (`DDG_mean` in particular comes from tables where positive commonly
means *destabilising*). `configs/data.yaml:20` documents `target` as "numeric ddG
label" with no sign statement.

**Why it matters:** Pearson r and Spearman ρ — the primary and reported metrics —
are **invariant to a global sign flip of the labels**. An ingested table with the
opposite convention would produce an apparently excellent r while the three-state
class labels, the interface's "stabilizing"/"destabilizing" verdict, the
zero-shot baseline's expected direction, and every qualitative claim in the paper
are inverted, with nothing in any artifact revealing it.

**Verified by:** reading the four locations above against one another; and
`src/heads.py:186-189` executed:
```
$ /home/user/.venv/bin/python /tmp/probe/p9.py
labels_from_ddg(ddg=[-3.0,-0.4,0.0,0.5,4.2], sigma=0.5) -> [0, 1, 1, 1, 2]
```
i.e. ΔΔG = +4.2 → class 2 → rendered "stabilizing" by
`app/backend/inference.py:210`, which is the opposite of the ProTherm-style
convention several of the accepted column names come from.

**Fix direction:** Add a required `ddg_sign: stabilizing_positive |
destabilizing_positive` key per dataset in `configs/data.yaml`, normalise at load
time, and record the applied sign in the leakage report and the checkpoint. Add
a sanity assertion that the mean ΔΔG over all substitutions is negative under
the internal convention (mutations are destabilising on average) and fail loudly
if it is not. Fix the contradictory note in `scripts/baselines.py:19-21,103-106`.

---

### R1-20 [MAJOR] A "measured" contact statistic with no artifact and no possible provenance

**Location:** `src/encoder.py:14-19`

**Problem:** The module docstring reads: *"The repository's own
`scripts/contact_sanity.py` measures a 0.252 long-range fraction (|i-j| >= 24)
among top-L contact-head pairs versus 0.000 for a sequence-proximity baseline —
that non-local signal is the whole reason for building a graph instead of a
chain."* This is present tense and attributed to this repository's own script.
`MIGRATION_NOTES.md` states that ESM-2 weights were never loaded here and
`results/` is empty by design; there is no `results/contacts/contact_sanity.json`
in the tree. Running the script against the only cache this repository can
produce (the test stub) yields 0.000, not 0.252.

**Why it matters:** `MIGRATION_NOTES.md` ("Deliberately left behind") makes the
deletion of `_STABILITY_R = 0.7644` and the demo-figure path the centrepiece of
the credibility argument, and `results/README.md` says *"every claim about model
performance in this repository should read 'not yet measured', because it has not
been."* An unverifiable measurement asserted in a source docstring — the one that
justifies building a graph at all — is the same defect in a smaller package. Note
that the neighbouring v1 numbers (`r = 0.5309` / `0.5268`,
`0.029–0.071 kcal/mol`, `2.81%`, `64%`) are all explicitly labelled as v1
measurements; `0.252` is not.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p13_contacts.py
Contact-head sanity (E_geo source vs sequence-proximity baseline):
  sequences sampled                 : 40
  contact head long-range fraction  : 0.000
  baseline     long-range fraction  : 0.000

$ ls results/
README.md          # no contacts/ directory, no contact_sanity.json
```

**Fix direction:** Delete the number from `src/encoder.py:14-19` and replace it
with a forward reference ("run `scripts/contact_sanity.py` after precompute; the
result is written to `results/contacts/contact_sanity.json`"), or label it as a
v1 measurement with its branch and artifact path, as the other retained v1
numbers are.

---

### R1-21 [MINOR] `logvar` is clamped in training but not at evaluation or serving, and overflows in fp16

**Location:** `src/heads.py:101`; `src/losses.py:55`; `scripts/train.py:192`;
`scripts/calibration.py:102`; `app/backend/inference.py:196`;
`configs/run.yaml:89` (`amp: true`)

**Problem:** Two related numerics issues.

1. `gaussian_nll` clamps `logvar` to `[-10, 10]`, so the head is *trained* under
   a clamped σ (and receives zero gradient once saturated). Every consumer at
   evaluation time uses `torch.exp(0.5 * logvar)` **unclamped**. The head's
   aggregation `log(exp(lv_wt) + exp(lv_mt) + 1e-8)` is bounded below by
   `log(1e-8) = −18.42`, so the saturated region is reachable: at
   `logvar = −14` training behaves as if σ = 0.0067 while calibration and the
   web interface report σ = 0.0009 — a 7× narrower interval than the model was
   optimised to produce, which will show up as spurious under-coverage.
2. `torch.exp(lv_wt) + torch.exp(lv_mt)` overflows to `inf` in fp16 once a raw
   score log-variance exceeds ≈11.09. `amp: true` is the shipped default, so the
   head runs under `autocast`. The loss survives (`clamp(inf, max=10) == 10`) but
   the gradient path through `log(inf)` does not, and at inference
   `exp(0.5·inf) = inf` gives an infinite confidence band.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p17.py
  raw score log-var= 11.0 dtype=torch.float16    -> aggregated logvar=inf
  raw score log-var= 12.0 dtype=torch.float32    -> aggregated logvar=12.693147659301758
   logvar= -14.0: training sees sigma=   0.0067   inference reports sigma=   0.0009
   logvar=  14.0: training sees sigma= 148.4132   inference reports sigma=1096.6332
```

**Fix direction:** Do the aggregation in log space with
`torch.logaddexp(lv_wt, lv_mt)` (numerically stable, no overflow) and apply the
same clamp inside `SubstitutionScoreHead.forward` so training, calibration and
serving see identical σ. Add `@torch.autocast(enabled=False)` around the head if
fp32 is needed there.

---

### R1-22 [MINOR] Three modules carry stale v1 documentation; two are dead code

**Location:** `src/sequence_mixers.py:1-22`; `src/pooling.py` (whole file);
`src/idr_prior.py:1-24`

**Problem:**

* `src/sequence_mixers.py:20-21` states *"Both default to OFF in the backbone,
  so existing behaviour and any existing checkpoints are preserved unless
  explicitly enabled."* Both are on: `src/model.py:64` defaults `use_ssm=True`
  and `src/backbone.py:331` constructs `CrossAttentionFusion` unconditionally.
  The same docstring references "paper §8.2", `esm_dim = 320`, `mech_dim = 3`
  and "the 323-dim contract" — the shipped model is 1280 and 7 channels
  (`src/features.py:51`).
* `src/pooling.py` is imported by nothing (`grep` confirms) and its docstring is
  entirely v1 scaffolding: "Holo-GNN backbone (V6)", "The V5 backbone pooled…".
  It duplicates `MaskedAttentionPool` from `src/readout.py`.
* `src/idr_prior.py` is imported by nothing and states at line 8 that it is *"a
  citable physics prior used as the training target for IDREnsembleHead"* — in
  v2 that head trains on a real Rg table (`scripts/train.py:265-267`), so the
  docstring describes a training procedure that no longer exists.

**Why it matters:** `MIGRATION_NOTES.md` ("Version scaffolding") states that the
V1–V6 labels and the "§8.2 future work" framing are gone — they are not. Method
sections get written from module docstrings; a reviewer reading
`src/sequence_mixers.py` would conclude the SSM and cross-attention fusion are
optional additions that are off by default, which is the opposite of the shipped
model, and would size the fusion at 320 dimensions.

**Verified by:**
```
$ grep -rn "from src.pooling\|import pooling\|AttentionPooling\|masked_mean\|idr_prior" --include=*.py . | grep -v node_modules
./src/pooling.py:...   (definitions only)
./src/idr_prior.py:2:src/idr_prior.py
```
and `src/model.py:64` / `src/backbone.py:330-331` against
`src/sequence_mixers.py:20-21`.

**Fix direction:** Delete `src/pooling.py` and `src/idr_prior.py`, or wire them
in and document them accurately. Rewrite the `sequence_mixers.py` docstring for
the shipped configuration and dimensions.

---

### R1-23 [MINOR] "Exact reversibility in train mode" holds only for a shared context

**Location:** `tests/test_invariants.py:167-175`; `README.md` ("Substitution-
conditioned head"); `src/substitution.py:16-18`

**Problem:** The test computes `z = model.context(batch)` once and passes it to
both `model.ddg(batch, z)` and `model.ddg(reversed_batch, z)`. That correctly
verifies the *head's* antisymmetry. It does not verify the *model's*: with
`z=None` (the ordinary call path) the readout's dropout
(`src/readout.py:81-86`) draws an independent mask per call, so the two
directions see different contexts.

**Why it matters:** README states "Exact reversibility. `ΔΔG(b→a) = −ΔΔG(a→b)`
to floating point, asserted in the test suite in both train and eval mode." In
eval mode the model-level property does hold exactly. In train mode two
independent calls violate it by 4.7e-2 kcal/mol — the same order as the
0.029–0.071 kcal/mol that MIGRATION_NOTES cites as v1's failure. This matters
because `antisymmetry_violation` (`src/losses.py:74-82`) is documented as a
monitor that "should read as floating-point noise, and anything larger is a bug";
computed the natural way during training it will read ~0.05 and be misdiagnosed.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p23_anti.py
[eval] shared-z (as tested)        max|ddg(a->b)+ddg(b->a)| = 0.000e+00
[eval] independent passes (real)   max|ddg(a->b)+ddg(b->a)| = 0.000e+00
[train] shared-z (as tested)       max|ddg(a->b)+ddg(b->a)| = 0.000e+00
[train] independent passes (real)  max|ddg(a->b)+ddg(b->a)| = 4.664e-02
```

**Fix direction:** State the claim precisely — "antisymmetric for any fixed
context vector, and therefore exactly antisymmetric at inference" — and add the
independent-pass assertion under `model.eval()` to the test suite so the
distinction is visible.

---

### R1-24 [MINOR] Non-standard residues are encoded inconsistently and rejected at the interface

**Location:** `src/features.py:207-210`; `src/heads.py:104-121`
(`scan`, `n_aa=20`); `src/substitution.py:155-160`;
`app/backend/inference.py:305-310` (`clean_sequence`)

**Problem:** `AminoAcidEmbedding` has a dedicated `UNKNOWN_AA_IDX = 20` slot
"never a silent 0" — but the mechanistic channels have no such flag, and their
defaults are inconsistent: for B/Z/X/U/O, hydropathy falls back to mid-scale
(0.5), helix propensity to 1.0 (mid-scale), and **volume to 0.0** — smaller than
glycine (0.25), i.e. the network is told "this side chain is smaller than any
real residue" rather than "this residue is unknown". Separately,
`SubstitutionScoreHead.scan` uses `n_aa=20`, so when the parent residue is
non-standard (`aa_index → 20`) no column of the returned heatmap is zero,
contradicting the docstring's "the wild-type column is exactly 0". And
`clean_sequence` rejects any sequence containing B/Z/X/U/O outright, so the web
interface refuses UniProt sequences that the training path accepts.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p10_edge.py
 hydropathy channel: [0.7, 0.778, 0.111, 0.5, 0.5, 0.5, 0.5, 0.5, 0.711, 0.067]
 volume channel    : [0.369, 0.452, 0.463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.679, 0.702]

$ /home/user/.venv/bin/python /tmp/probe/p23_anti.py
 scan row min|value| over the 20 columns: 0.0005193054676055908  (docstring: 'the wild-type column is exactly 0')
```

**Fix direction:** Add an `aa_known` channel to the mechanistic features and set
the scalar channels to the dataset mean rather than 0 for unknown residues. Make
`scan` return 21 columns (or drop rows whose wild-type residue is non-standard,
counted). Let `clean_sequence` accept X/B/Z/U/O and mark them, rather than
rejecting the whole sequence.

---

### R1-25 [MINOR] The interface verdict contradicts its own three-state output, and `metrics.json` is not valid JSON

**Location:** `app/backend/inference.py:213`; `scripts/train.py:455`, `:461-462`

**Problem:** `"verdict": "stabilizing" if ddg > 0 else "destabilizing"` is a
strict sign test with no neutral case, in a payload that simultaneously reports
a three-class probability including `"neutral"`. A ΔΔG of +0.006 is rendered
"stabilizing" while the head's own argmax is "neutral". Separately, when
`best_score` stays at `-inf` (R1-02) `json.dump` writes the bare token
`-Infinity`, which is invalid per RFC 8259 and rejected by strict parsers.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p5b.py
verdict vs three-state argmax disagreement example: {'mutation': 'N1A', 'ddg': 0.0059,
 'three_state': {'destabilizing': 0.3078, 'neutral': 0.3641, 'stabilizing': 0.328}, 'verdict': 'stabilizing'}

$ grep -n "Infinity" /tmp/.../runNV/metrics.json
77:  "best_val_stability_pearson": -Infinity,
```

**Fix direction:** Derive `verdict` from the three-state argmax (which is the
head that has a defined neutral class) or from a `|ΔΔG| ≤ n·σ` band using the
model's own predicted σ, and say which. Serialise non-finite values as `null`.

---

### R1-26 [MINOR] Chunk selection takes 128 residues per boundary from the shorter chunk

**Location:** `src/encoder.py:203-214`, docstring at `:186-189`

**Problem:** The docstring states *"Each residue's embedding is taken from the
chunk where it sits furthest from a boundary, so representations always come
from the most context available."* Distance-to-boundary is not "most context
available": for a 1023-residue protein the spans are `[(0,1022), (766,1023)]`,
and residues 894–1021 have a larger boundary margin in the 257-residue tail
chunk than in the 1022-residue chunk, so they take their embedding from the
short fragment. Exactly 128 residues per boundary are affected, for every
protein above the context limit. The contact dedup at `:224-231` uses
`scatter_reduce_(reduce="amax")`, which likewise prefers whichever chunk assigned
the higher probability — biasing toward the lower-context fragment.

**Verified by:**
```
$ /home/user/.venv/bin/python /tmp/probe/p19_margin.py
L= 1023 spans=[(0, 1022), (766, 1023)]
   residues whose embedding is taken from a SHORTER chunk than one that also covered them: 128
     residue 894: taken from chunk (766, 1023) (len 257) although chunk (0, 1022) (len 1022) also covered it
L= 1600 spans=[(0, 1022), (766, 1600)]
   residues whose embedding is taken from a SHORTER chunk than one that also covered them: 128
```

**Fix direction:** Break ties on `(margin, chunk_length)` or use
`min(margin, chunk_length/2)` as the selection score, so a residue never moves to
a chunk with strictly less context. Consider averaging the overlap rather than
choosing.

---

### R1-27 [MINOR] The ablation grid omits the two components the paper's argument rests on

**Location:** `scripts/ablations.py:314-318` (`ARMS`);
`README.md` ("Evaluation"); `scripts/collect_results.py:147-149`

**Problem:** The arms are `full`, `no_mech_stream`, `no_ssm`. There is no arm for
the **site-anchored readout** (vs. the global-pool-and-subtract it replaces) and
none for **E_geo** (the contact graph itself). Those are the two changes the
paper's narrative rests on: `src/readout.py:6-13` argues the site anchoring is
the fix for the 1/L dilution problem, and `scripts/contact_sanity.py:5-9` calls
the contact graph "the check that justifies building a graph at all". Meanwhile
`collect_results.py:147-149` writes into `SUMMARY.md` that *"the gap between
[esm_mlp] and the model is the message-passing contribution"* — but `esm_mlp`
differs from the full model in five ways at once (graph, readout, SSM,
cross-attention fusion, mechanistic channels), so that gap cannot be attributed
to message passing.

**Verified by:** reading `scripts/ablations.py:314-318` (three arms, no readout
or geo arm) against `scripts/collect_results.py:147-149` and `src/model.py:73-83`
(the full model differs from a plain MLP on cached embeddings in all five
respects).

**Fix direction:** Add `global_pool_readout` (replace `SiteAnchoredReadout` with
mean/attention pooling and a difference, i.e. the v1 formulation) and
`no_geo_stream` arms — the second is a one-line change given
`use_mech_stream` already exists as an axis. Then attribute the `esm_mlp` gap to
the union of components, not to message passing alone. Run every arm over
multiple seeds (R1-01).

---

## What I checked and found sound

These held under direct testing. Do not churn them.

* **Exact head antisymmetry and swap-equivariant three-state.** `ddG(b→a) =
  −ddG(a→b)` is 0.0 to the bit in both train and eval mode for a fixed context,
  and `logits(mt,wt) == flip(logits(wt,mt))` is exact. The `_DETERMINISTIC = 0.0`
  dropout decision in `src/heads.py:36-43` is correct and the reasoning is right.
  (See R1-23 for the one over-stated part of the claim.)
* **Padding invariance and `collate_graphs` offset arithmetic.** I built a batch
  with heterogeneous window lengths `[7, 32, 32, 3]` and different sites and
  compared each sample's batched prediction against the same sample alone:
  max difference 2.98e-07. The `off = i * L` offsets are consistent with the
  `(B*L, C)` reshape in `src/backbone.py:378`, padding nodes have no edges and
  never contribute to a pool, and the causal SSM scan cannot leak right-padding
  into real positions.
* **The associative scan.** `_PureSelectiveSSM._associative_scan` matches the
  sequential recurrence to ≤4.8e-07 at L ∈ {1,2,3,5,8,17,256}, including all the
  non-power-of-two lengths. This is correct and the claimed 8-step complexity at
  L=256 holds.
* **Mutation-centered windowing for the mutation-conditioned tasks.**
  `window_bounds` guarantees `start ≤ site < end` at both termini and returns a
  full-length window for interior sites; verified at sites 0, 1, L−2, L−1 for
  L=1000, and the whole sequence for L ≤ window.
* **Edge cases.** Single-residue proteins, length 2 and 3, site 0 and site L−1,
  empty graphs, an all-padding pooling row, and an out-of-range k-hop site all
  produce finite outputs with no crash and no NaN.
* **`_cap_degree` correctness.** It really caps: a 1128-edge near-complete graph
  reduces to 99 edges with max degree exactly 4 at `top_k=4`. (Its cost, not its
  correctness, is the problem — R1-17.)
* **Encoder chunk stitching mechanics.** `emb[start:end][take] = c_emb[take]`
  does write through the view; `_chunk_spans` covers every residue for L ∈
  {1022, 1023, 1500, 2000, 5000}; the two-level `lo*L + hi` pair encoding is
  bijective in both the per-chunk and the whole-protein dedup. The cross-chunk
  contact limitation is disclosed honestly in the docstring. (The *selection
  rule* is the issue — R1-26.)
* **Union-find grouping in `make_splits.py`.** The `c:`/`s:`/`g:` node scheme
  does produce connected components under "same cluster OR same gene", and
  `verify_no_leakage` is a real check that raises rather than warning. I also
  checked the `df["group"].equals(df["parent_seq"])` guard at line 95 — pandas
  `Series.equals` compares values, not names, so `has_gene` is correctly `False`
  for stability tables. The remaining leakage risk in this path is MMseqs2's
  greedy clustering not being the transitive closure of the 30% relation, which
  is a documented property of the tool rather than a defect here — but it is
  worth a sentence in the paper, and the fallback message when `mmseqs` is
  absent is honest.
* **Parameter count.** `MIGRATION_NOTES.md`'s 20,601,900 trainable parameters is
  exactly right: `{'backbone': 19940608, 'readout': 329730, 'heads': 332906,
  'total': 20601900}`.
* **"No demo mode" in the interface.** All 120 checks pass, including the
  no-checkpoint 503 path on every prediction endpoint, the absence of
  `demo_mode` from the API and the SQLite schema, refusal for untrained heads
  (`/api/idr` 503), and the wild-type-mismatch 400. The confidence interval does
  come from the model's σ, not a formula. `src/heuristics.py` is genuinely gone.
* **`scripts/collect_results.py`.** It writes only what an artifact contains and
  the literal `not run` otherwise; I found no default, cached constant or
  reconstruction anywhere in it. (Its one problem is the causal claim it
  appends — R1-27.)
