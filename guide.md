# HoloGNN — Train / Test / Eval Guide (single GPU, ~2–3 hours)

Written for the machine that has the **GPU and the preprocessed parquet datasets**.
You do **not** download any datasets — they're already on the box. You install
software, point the runner at the parquet files, and run **one command**.

The heavy step (ESM embeddings) runs **once and is cached**; everything after is light.

---

## TL;DR

```bash
python run.py setup                 # make a venv + install deps (one time)
#   activate it (printed at the end of setup), then:
python run.py smoke                 # 30-sec sanity check (no data needed)
# put the parquet files in data/ (see §2), then:
python run.py all                   # precompute -> splits -> train -> eval -> results
```
Everything is controlled by **`configs/run.yaml`** (paths, epochs, ESM size). Any
value can be overridden on the command line.

---

## 1. Setup the environment (with a venv)

`run.py setup` creates a virtual environment and installs `requirements.txt` into it.

```bash
# default: creates ./.venv in the repo root
python run.py setup

# choose a custom location if you prefer:
python run.py setup --venv D:\envs\holognn

# for an NVIDIA GPU, install the matching CUDA torch wheel (recommended):
python run.py setup --torch-index-url https://download.pytorch.org/whl/cu121
```

If `.venv` (or your `--venv` path) already exists, it is reused, not recreated.
After setup, **activate** it:

```bash
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

Optional but recommended for strict splits — **MMseqs2** (a binary, not pip):
```bash
conda install -c bioconda mmseqs2 -y
```
Without it, splitting falls back to protein-disjoint grouping (still no leakage).

ESM-2 weights download automatically from HuggingFace on first use (≈130 MB for
150M, ≈2.5 GB for 650M).

Sanity check: `python run.py test` runs the unit tests; `python run.py smoke`
exercises the real GATv2 dual-track forward/backward on synthetic data.

---

## 2. Put the preprocessed files where the runner expects them

```
data/
├── megascale_ddg_pairs.parquet               # 108,901 ΔΔG pairs (stability)
├── clinvar_protein_pairs.parquet             # 197,801 variants  (pathogenicity)
└── clinvar_protein_pairs.parquet.ddg_w49.npy # cached ΔΔG (optional hybrid feature)
```
(Other paths are fine — set them in `configs/run.yaml` or via `--stability-data` /
`--clinvar-data`.)

**Confirm column names once** (the one thing that can differ on your box):
```bash
python -m scripts.inspect_data --data data/megascale_ddg_pairs.parquet
python -m scripts.inspect_data --data data/clinvar_protein_pairs.parquet
```
If any of `wt_seq / mt_seq / target / label / gene` shows `-> None`, set the real
name in **`configs/data.yaml`**. If MegaScale has a coding-DNA column, set
`wt_dna/mt_dna` — that switches on the **real CAI/mRNA features and the E_mech track**.

---

## 3. Choose an ESM size for your time budget

| Checkpoint | dim | speed | when |
|---|---|---|---|
| `facebook/esm2_t30_150M_UR50D` | 640 | **fast** | the 2–3 h run (default) |
| `facebook/esm2_t33_650M_UR50D` | 1280 | ~4× slower | paper-grade / matches 1280-d |

Set it once in `configs/run.yaml` (`esm_model:` and `cache:`), or pass
`--esm-model ... --cache cache/emb_650M` on the CLI.

---

## 4. Run it

**One command (uses `configs/run.yaml`):**
```bash
python run.py all
```
This runs, in order: `precompute → splits → stability (+ ablation) → pathogenicity
→ idr → baseline → evaluate → results`.

**Override anything on the CLI (CLI wins over the config):**
```bash
python run.py all --esm-model facebook/esm2_t33_650M_UR50D --cache cache/emb_650M \
    --epochs 10 --batch 32 --device cuda
python run.py all --max-seqs 60000 --max-rows 40000   # faster first pass
```

**Run a single stage** (same overrides apply):
```bash
python run.py precompute
python run.py splits
python run.py stability            # add --no-ablation to skip the single-track run
python run.py pathogenicity
python run.py idr
python run.py baseline
python run.py evaluate
python run.py results
```

### Rough time budget (12–16 GB GPU, 150M)
precompute ~40–80 min (dominates) · splits a few min · stability ×2 ~20–40 min ·
pathogenicity+CV ~20–40 min · idr/baseline/eval ~10–20 min → inside 2–3 h.
Tight on time? `--max-seqs` on precompute, and run only `stability` + `pathogenicity`.

---

## 5. Results (sorted, claim-by-claim)

After `run.py all` (or `run.py results`), look in **`results/`**:

```
results/
├── CLAIMS.md         # each model component -> the code + file that backs it
├── SUMMARY.md        # headline metrics table (auto-generated)
├── stability/        # r, MAE, RMSE, antisymmetry violation (strict split)
├── ablation/         # dual-track vs single-track  = the measured E_mech effect
├── pathogenicity/    # random vs gene-disjoint AUROC + GroupKFold CV + ROC
├── baselines/        # ESM zero-shot ΔΔG on the same test split
├── idr/              # Rg MAE vs the disclosed physics prior
└── splits/           # leakage_report.txt (no train/test group overlap)
```
The **gene-/protein-disjoint** numbers are the headline, leakage-controlled figures.
See `results/CLAIMS.md` for the component → evidence map.

---

## 6. What the numbers mean

- **Stability:** metrics on the 30%-identity, protein-disjoint test split. The gap
  between the dual-track and `--no-mech-track` runs is the measured E_mech ablation.
- **Pathogenicity:** gene-disjoint AUROC (± CV std) is the headline; the random-split
  number is the optimistic reference — both are reported (e.g. 0.956 / 0.786).
- **IDR:** Rg MAE vs a polymer-physics target — a learnability check against the prior,
  not an experimental validation (upgrade path: PED / ALBATROSS labels).
- **Baseline:** ESM zero-shot ΔΔG on the same split, for a fair comparison.

---

## 7. Troubleshooting

- **CUDA OOM (precompute):** use `esm2_t30_150M` (or `t12_35M`), lower `--max-length`.
  Training itself is light (cached features).
- **`-> None` in inspect_data:** set the column in `configs/data.yaml`.
- **`mmseqs not found`:** install it or set `split_method: group` (still no leakage).
- **ViennaRNA missing:** mRNA falls back to a GC proxy and `mech_valid=0` (E_mech off);
  install ViennaRNA to enable the real feature + second track.
- **Cache/model mismatch:** `--esm-model` for training must match the one used to build
  `--cache` (same embedding dim).
- **No venv yet:** `python run.py setup --venv <path>` makes one; reused if it exists.

---

## 8. After the run
The numbers to report are in `results/SUMMARY.md`, under the gene-/protein-disjoint
splits. `results/CLAIMS.md` maps each model component to the file that backs it.
