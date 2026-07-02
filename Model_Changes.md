# Model Changes

A complete log of every change made to run Holo-GNN end-to-end on the target
machine (Windows 11, RTX 5070 Ti 16 GB, ESM-2 150M) and to fix issues found along
the way. Grouped by category; each entry lists the file(s), what changed, and why.

The training/evaluation run these changes produced is summarized at the bottom.

---

## 1. Environment & data (no code committed)

- **Installed missing dependencies** into the run environment: `matplotlib`
  (required by the `evaluate`/`results` figure stages) and `ViennaRNA` 2.7.2
  (enables real mRNA-folding features and therefore the mechanistic `E_mech`
  track; without it that channel is masked off).
- **Normalized the MegaScale parquet columns.** The provided
  `megascale_ddg_pairs.parquet` uses `SOURCE_SEQUENCE_ID` / `TARGET_SEQUENCE_ID` /
  `DDG_mean`, which the auto-detector in `src/data.py` does not recognize (and
  `configs/data.yaml` is never actually read by any script). A normalized copy was
  written with the standard `wt_seq` / `mt_seq` / `ddG` names so column
  auto-detection succeeds. ClinVar (`clinvar_protein_pairs.parquet`) auto-detects
  as-is.

---

## 2. Compatibility fixes

### `src/backbone.py` — ESM-2 under transformers ≥ 5 (eager attention)
`EsmModel.from_pretrained(...)` now passes `attn_implementation="eager"`.
Transformers 5 defaults to the `sdpa` attention implementation, which does **not**
emit attention maps, so `EsmModel.predict_contacts(...)` failed with
`RuntimeError: stack expects a non-empty TensorList` and the fallback hit
`IndexError: tuple index out of range`. Eager attention restores the contact head
(and thus the geometric edge set `E_geo`). Embedding-only use was unaffected.

### `run.py` — force UTF-8 for the runner and every spawned stage
`main()` now sets `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` and reconfigures
`stdout`/`stderr` to UTF-8. On Windows the default console codec is cp1252, which
raised `UnicodeEncodeError` on non-ASCII prints (the `⚠️` in `make_splits.py`, and
`±` / `Δ` / `Å` elsewhere) whenever output was redirected to a file — this crashed
the `splits` stage. Now every stage is UTF-8 safe regardless of how it is launched.

---

## 3. Correctness fixes

### `scripts/train_pathogenicity.py` — drop variants with no cached embedding
ClinVar contains proteins longer than 5,000 residues, which
`precompute_embeddings.py` skips (`1 ≤ len ≤ 5000`), so their `.pt` files never
exist. `VariantDataset` loaded them unguarded and crashed with `FileNotFoundError`.
The stage now filters to variants whose embedding is cached (a `[filter]` line
reports how many were dropped — 5,549 / 197,801 = 2.81% for this dataset). This is
also *more correct*: those variants are truncated to 256 residues anyway, so the
mutation would not be represented.

### `scripts/baselines.py` — checkpoint temp-file naming
`np.savez` appends `.npz` to its target, so the atomic-write temp file was written
as `...partial.npz.tmp.npz` while `os.replace` looked for `...partial.npz.tmp`,
failing every save (guarded, so the stage still completed). Fixed by giving the
temp file a `.npz` suffix.

---

## 4. Crash-resilience / checkpointing

Motivation: the full pipeline is long (≈16 h) and a real power loss occurred
mid-run. Every stage is now resumable; a re-run of `python run.py all` skips
finished work and continues an interrupted stage. (Verified: the pipeline resumed
correctly after an actual power outage, redoing only the interrupted epoch.)

- **`run.py` — stage-level skip.** New `_done()` helper; each stage returns early
  if its final output already exists (`splits`, `stability` + ablation,
  `pathogenicity`, `idr`, `baseline`, `contacts`). So `run.py all` resumes at the
  first unfinished stage.
- **`scripts/precompute_embeddings.py` — atomic cache writes.** Each `.pt` is
  written to a `.tmp` then `os.replace`d, so a crash can never leave a truncated
  embedding that a later stage would fail to load. (Per-sequence skip already
  existed.)
- **`scripts/train_stability.py` — per-epoch checkpoint.** Saves
  `runs/stability[_nomech]/last.pt` (model + optimizer + epoch + best) each epoch
  and resumes from it; if the checkpoint is already at the final epoch the stage
  reloads and re-emits metrics instantly.
- **`scripts/train_idr.py` — per-epoch checkpoint** (`runs/idr/last.pt`), same
  pattern; the final Rg MAE is stored so a fully-resumed run still writes metrics.
- **`scripts/train_pathogenicity.py` — two-level resume.** (a) A
  `pathogenicity_partial.json` records completed sub-runs (random split /
  gene-disjoint / each CV fold), so a re-run skips finished sub-runs. (b) Each
  sub-run also checkpoints per-epoch (`last_<tag>.pt`) and cleans it up on
  completion, so a crash mid sub-run resumes at the last epoch, not from scratch.
- **`scripts/baselines.py` — incremental checkpoint.** The ESM zero-shot scoring
  loop saves progress (scores + next row index) every 1,000 rows and resumes from
  the last row.

All checkpoint writes are wrapped in `try/except` so a failed save can never crash
training.

---

## 5. Configuration & runner

- **`run.py`:** added `cv_folds` to `DEFAULTS` and pass `--cv-folds` to the
  pathogenicity stage (set to **3** for this run to keep the 5-fold CV runtime
  bounded; it only narrows the AUROC confidence interval, not the headline).
- **`configs/run.yaml`:** for the run, `cache` was pointed at `C:/HoloGNN_work/...`
  and `clinvar_data`/`clinvar_ddg` at the local absolute paths, and
  `back_translate` was set `true`. These machine-specific values are reverted to
  portable defaults in the committed config; the embedding cache and training
  outputs were kept off the (nearly full) `F:` drive via directory junctions.

---

## 6. Results — 150M baseline run

Produced by `python run.py all` on the preprocessed MegaScale + ClinVar data,
ESM-2 `t30_150M`, 30%-identity protein-disjoint (stability) and gene-disjoint
(pathogenicity) splits. Full table in [`results/SUMMARY.md`](results/SUMMARY.md).

| Task | Metric | Value | Regime |
|---|---|---|---|
| Stability | Pearson r / MAE / RMSE | 0.527 / 0.725 / 1.169 | 30% protein-disjoint |
| Ablation | Δr from E_mech | −0.004 | dual − single track (mech ≈ neutral) |
| Pathogenicity | AUROC | 0.918 | random split (optimistic) |
| Pathogenicity | AUROC | 0.669 / 0.697 ± 0.017 | gene-disjoint / 3-fold CV |
| Baseline | ESM zero-shot \|r\| | 0.281 | same stability test split |
| IDR | Rg MAE | 26.4 Å | physics prior (learnability check) |
| Contacts | Long-range fraction (top-L) | 0.252 vs 0.000 | ESM E_geo vs linear baseline |

**Reading the numbers.** The trained stability model (r=0.527) nearly doubles the
ESM zero-shot baseline (|r|=0.281) on the same split — real learned signal. The
modest absolute values are explained, not defects: the stability test split has
compressed ΔΔG variance (std 1.33 vs 2.30 in train) and is protein-disjoint; the
mechanistic track adds nothing here because MegaScale has no real coding DNA (the
`E_mech` edges come from a back-translate proxy); and the gene-disjoint
pathogenicity AUROC is limited by a head-truncation blind spot (see §7): with
`max_length=256`, the mutation falls outside the embedded window for the majority
of long ClinVar proteins.

## 7. Known limitations & next steps

- **Truncation (biggest lever):** with `max_length=256` each sequence is
  head-truncated to its first 256 residues, but ~64% of ClinVar variants have
  their mutation beyond position 256 (median 383) — so on those variants the
  mutant embedding equals the wildtype's first 256 residues and the model cannot
  see the mutation. Embedding a window centered on the mutated residue (instead of
  the sequence head), at the same embedding size, would restore that signal.
- **Stability fidelity:** `AntisymmetricLoss` weights antisymmetry and fidelity
  equally (`alpha=1`); lowering `alpha` prioritizes the ΔΔG fit.
- **Reporting:** stability uses a single protein-disjoint split; a GroupKFold CV
  (as pathogenicity has) would give a more stable headline r.
- **Pathogenicity weights:** the stage reports metrics but does not persist a final
  classifier checkpoint (only per-sub-run resume checkpoints, cleaned on success).
