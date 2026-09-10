# Run state — where this stopped, and how to run the ablations

Everything below is on disk. Nothing needs rebuilding to run the ablation grid;
the splits, the encoder cache and the trained checkpoint are all persisted.

---

## 1. To run the ablations later — the only command you need

```bash
cd d:/F/Projects/Holo-GeoN/Holo-GeoN

# MMseqs2 lives outside the repo; a NEW terminal already has it on PATH.
# In this session's shell it had to be added explicitly:
export PATH="$PATH:/c/Users/User/tools/mmseqs/bin"      # git-bash
#   PowerShell: $env:Path += ';C:\Users\User\tools\mmseqs\bin'

# Stop the web interface first — it holds ~4 GB of VRAM (see §5).
python run.py ablations --seeds 42 43 44 --epochs 8
```

**Budget it before starting.** One training run took **5.5 h** (25 epochs,
early-stopped). The grid is `arms x seeds` full training runs:

| command | runs | approx wall-clock |
|---|---|---|
| `--seeds 42 43 44 --epochs 8` | 5 arms x 3 = 15 | **~13 h** |
| `--seeds 42 43 44 --epochs 8 --arms full no_mech_stream global_pool_readout` | 3 x 3 = 9 | ~5 h |
| default (`--seeds 42 43 44 45 46`, 33 epochs) | 5 x 5 = 25 | **~137 h — do not** |

Three seeds is the floor `scripts/ablations.py` itself sets: below that no
paired test is possible and every verdict reads null. Eight epochs is past where
both completed runs plateaued (best epochs were 8 and 15).

The stage is resumable and skips arms whose results already exist, so an
interrupted grid continues where it stopped.

---

## 2. What is already done

| stage | artifact | status |
|---|---|---|
| data tables | `data/*.parquet` | done |
| contact maps | `data/true_contacts/` (373 maps) | done |
| encoder cache | `C:/Users/User/AppData/Local/holo-geon/cache/esm2_650M` (24.8 GB, 15,733 records) | done |
| splits | `data/splits_megascale.parquet`, `data/splits_clinvar.parquet` | done |
| **training** | `runs/main/model_best.pt`, `last.pt`, `metrics.json` | **done** — early stop at ep 25, best ep 15 |
| calibration | `runs/main/calibration.json` | done |
| external eval | `runs/main/external_fireprotdb.json` | done |
| contacts | `results/contacts/contact_sanity.json` | done |
| baselines | `runs/baselines/{stability,pathogenicity}/baselines.json` | done |
| **ablations** | `runs/ablations/ablations.json` | **NOT RUN** |
| results summary | `results/SUMMARY.md` | run `python run.py results` |

`runs/baseline_wd1e-4/` is the earlier weight-decay-1e-4 run, kept for
comparison. It is not the shipped model.

---

## 3. Final held-out scores

Print them any time:

```bash
python -m scripts.final_scores                  # final test scores
python -m scripts.show_progress                 # per-epoch validation history
python -m scripts.show_progress --compare runs/baseline_wd1e-4
```

| | test | external (FireProtDB) |
|---|---|---|
| ddG Pearson r | 0.7582 | **0.4518** |
| ddG RMSE | 0.6482 | 1.6639 |
| ddG R^2 | 0.5457 | 0.1901 |
| three-state macro-F1 | 0.5180 | — |
| ClinVar AUROC / AUPRC | 0.8971 / 0.8378 | — |

Contact edges: precision@L **0.729** vs 0.077 random (n=10 structures).
Calibration: ECE 0.117 -> **0.034** after the fitted variance scale of 2.385.

---

## 4. Three findings to carry into any write-up

**The graph does not beat a plain MLP.** `esm_mlp_matched_capacity` — same
embeddings, same parameter count, no graph — scores r 0.7537 / RMSE 0.6415 /
R^2 0.5550 against the full model's 0.7582 / 0.6482 / 0.5457. It is within
0.0045 Pearson and **better** on both error metrics. The README describes this
gap as the joint contribution of the graph, the site-anchored readout, the SSM,
the cross-attention fusion and the mechanistic channels; measured, that
contribution is approximately zero. The ablation grid is what would attribute
this per component — and it should be expected to find small effects.

**Transfer is much weaker than the test split suggests.** r drops 0.758 -> 0.452
and RMSE nearly triples on FireProtDB proteins outside the training
distribution. Quote both numbers together; the test figure alone overstates the
model.

**The ClinVar AUROC is mostly gene-level discrimination.** An oracle scoring
each variant by its own gene's positive rate reaches AUROC 0.952 on val, and
99.8% of the AUROC pair mass is cross-gene. The model has essentially saturated
that ceiling. It is not evidence of ranking variants *within* a protein, which
is the clinical question.

---

## 5. Environment notes that will bite otherwise

- **The encoder cache is on the SSD**, at
  `C:/Users/User/AppData/Local/holo-geon/cache/esm2_650M` (set in
  `configs/run.yaml`). The repo lives on a spinning HDD, and with the cache
  there a ClinVar batch took 1.8-2.3 s against 0.07 s on the SSD — a 31x
  difference that left the GPU idle at 0-5%. Do not move it back. The stale copy
  under `cache/esm2_650M` on D: can be deleted to reclaim 24.8 GB.
- **Stop the web interface before the ablations.** It holds ESM-2 650M plus the
  model in VRAM (~4 GB of 16 GB). Find it with
  `tasklist | findstr python` or just close its terminal.
- **MMseqs2** is at `C:\Users\User\tools\mmseqs\bin`, appended to the *user*
  PATH. A new terminal picks it up; an already-open one will not.
- **Restart the web interface** (note: `runapp.py` builds its own venv without
  torch, so the model cannot load there — use the main venv):

```bash
cd app/backend
PYTHONPATH=d:/F/Projects/Holo-GeoN/Holo-GeoN \
HOLOGEON_WEIGHTS=d:/F/Projects/Holo-GeoN/Holo-GeoN/runs/main/model_best.pt \
../../.venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

- **Nothing has been committed.** `git status` shows the full diff: 8 new
  scripts, fixes in `src/data.py`, `src/splits.py`, `src/encoder.py`,
  `scripts/train.py`, `scripts/baselines.py`, `scripts/calibration.py`, new
  tests, and config changes. The test suite passes at 312 checks.
