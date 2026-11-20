"""
remote/gpu_jobs/wave0_t4_probe.py
=================================
Wave 0 Kaggle T4 Timing, VRAM & 3-Epoch Smoke Probe (Holo-GeoN v2 vs. v1).

Run this ONCE (~12-18 min on 1x T4) BEFORE launching the full 4-account 5-seed
ablation suites (`kaggle_v2_ablations.yaml`, `kaggle_v2_geometry.yaml`), per
`extras/kaggle-colab-lessons.md`:
  "Measure the step time before anything is sized for real - one short probe
  per GPU type, then --plan again."

What it measures and writes to `/kaggle/working/wave0_probe_report.json`:
  1. Environment & GPU hardware facts (driver, torch/CUDA, transformers, GPU name, VRAM).
  2. Exact single-pass `EsmForMaskedLM` encoding wall-clock time for all 412 MegaScale
     + 176 FireProtDB + 94 S669 proteins (and saves the reusable v2 cache to
     `/kaggle/working/esm2_650M_v2_cache.tar` or keeps it ready for downstream runs).
  3. Measured `sec_per_step`, `peak_vram_gb`, and 3-epoch `val_pearson` / `test_pearson`
     / `fireprotdb_hf25_pearson` / `s669_pearson` for:
       - `v2_full` (2.04M params, Protein-Batch, beta-NLL, CDS dropout 0.5)
       - `v1_baseline` (20.6M params, legacy batching, standard NLL)
  4. Re-projected `--plan` hour estimates for `kaggle_v2_ablations.yaml` and
     `kaggle_v2_geometry.yaml` using the MEASURED `sec_per_step`.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

REPO_URL = os.environ.get("HOLO_REPO_URL", "https://github.com/MdAhbab/Holo-GeoN.git")
REPO_BRANCH = os.environ.get("HOLO_REPO_BRANCH", "version-2")
PROBE_EPOCHS = int(os.environ.get("HOLO_PROBE_EPOCHS", "3"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_wave0")
REPO = WORK / "Holo-GeoN"
CACHE = Path("/tmp/holo/cache/esm2_650M_v2") if Path("/tmp").is_dir() else WORK / "cache_v2"
WORK.mkdir(parents=True, exist_ok=True)


def sh(cmd: list[str] | str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), check=check)


def get_code() -> None:
    if (REPO / "scripts" / "train.py").is_file():
        return
    hits = (
        glob.glob("/kaggle/input/*/**/scripts/remote_job.py", recursive=True)
        + glob.glob("/kaggle/input/*/scripts/remote_job.py")
    )
    if hits:
        src = Path(hits[0]).parent.parent
        shutil.copytree(src, REPO)
        return
    sh(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO)])


def find_input(name: str, roots=("/kaggle/input", ".")) -> Path | None:
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for hit in rp.rglob(name):
            if hit.is_file():
                return hit
            if hit.is_dir():
                inner = hit / name
                if inner.is_file():
                    return inner
    return None


def stage_data() -> dict[str, str]:
    data_dir = REPO / "data"
    data_dir.mkdir(exist_ok=True)
    names = (
        "megascale_ddg_pairs.parquet",
        "splits_megascale.parquet",
        "splits_megascale_id25.parquet",
        "splits_megascale_fold_tm60.parquet",
        "fireprotdb_holdout.parquet",
        "fireprotdb_hf25.parquet",
        "s669_holdout.parquet",
        "invariant_backbone_geometry.npz",
    )
    staged: dict[str, str] = {}
    for n in names:
        p = find_input(n)
        if p is not None:
            dst = data_dir / n
            if not dst.exists():
                try:
                    os.symlink(p, dst)
                except OSError:
                    shutil.copy2(p, dst)
            staged[n] = str(dst)
    print(f"[wave0] Staged {len(staged)} input(s): {sorted(staged.keys())}", flush=True)
    return staged


def main() -> None:
    t0 = time.time()
    get_code()
    sh("pip install -q 'transformers>=4.40,<5' torch_geometric ViennaRNA pyyaml "
       "pyarrow scikit-learn scipy pandas tqdm")
    staged = stage_data()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[wave0] CUDA GPU required for timing probe")
    cap = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    print(f"[wave0] GPU: {gpu_name} (sm_{cap[0]}{cap[1]}, {vram_gb} GB VRAM)", flush=True)
    if cap[0] < 7:
        raise SystemExit(f"[wave0] Refusing pre-sm_70 GPU ({gpu_name})")

    # 1. Single-pass EsmForMaskedLM precompute
    enc_tables = [p for k, p in staged.items() if k in (
        "megascale_ddg_pairs.parquet", "fireprotdb_holdout.parquet", "s669_holdout.parquet"
    )]
    t_enc = time.time()
    sh([
        sys.executable, "-m", "scripts.precompute_embeddings",
        "--data", *enc_tables,
        "--config", "configs/benchmark_v2.yaml",
        "--cache", str(CACHE),
        "--device", "cuda",
    ])
    enc_s = round(time.time() - t_enc, 2)

    # Attach precomputed SE(3)-invariant backbone coordinates if staged
    if "invariant_backbone_geometry.npz" in staged:
        sh([
            sys.executable, "-m", "scripts.attach_coords",
            "--cache", str(CACHE),
            "--geom-npz", staged["invariant_backbone_geometry.npz"],
        ], check=False)

    splits_arg = staged.get("splits_megascale_id25.parquet") or staged["splits_megascale.parquet"]
    results = {}

    for tag, extra_flags in (
        ("v2_full", ["--config", "configs/benchmark_v2.yaml"]),
        ("v1_baseline", [
            "--config", "configs/benchmark_v2.yaml",
            "--arch", "v1", "--nll-beta", "0.0", "--cds-dropout", "0.0", "--no-protein-batch",
        ]),
    ):
        out_dir = WORK / f"probe_{tag}"
        t_tr = time.time()
        sh([
            sys.executable, "-m", "scripts.train",
            *extra_flags,
            "--stability-splits", splits_arg,
            "--pathogenicity-splits", "none",
            "--cache", str(CACHE),
            "--device", "cuda",
            "--epochs", str(PROBE_EPOCHS),
            "--seed", "42",
            "--tag", tag,
            "--out", str(out_dir),
        ])
        tr_s = round(time.time() - t_tr, 2)
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

        # External evaluations (FireProtDB-HF25 and S669 when present)
        ext_scores = {}
        for ext_key in ("fireprotdb_hf25.parquet", "fireprotdb_holdout.parquet", "s669_holdout.parquet"):
            if ext_key in staged:
                sh([
                    sys.executable, "-m", "scripts.eval_external",
                    "--checkpoint", str(out_dir / "model_best.pt"),
                    "--data", staged[ext_key],
                    "--config", "configs/benchmark_v2.yaml",
                    "--cache", str(CACHE),
                    "--device", "cuda",
                    "--out", str(out_dir),
                ], check=False)
                ef = out_dir / "external_fireprotdb.json"
                if ef.is_file():
                    ext_scores[ext_key] = json.loads(ef.read_text(encoding="utf-8")).get("metrics")

        results[tag] = {
            "wall_clock_s": tr_s,
            "epochs": PROBE_EPOCHS,
            "sec_per_epoch": round(tr_s / max(PROBE_EPOCHS, 1), 2),
            "test_stability": (metrics.get("test") or {}).get("stability"),
            "param_counts": metrics.get("param_counts"),
            "external": ext_scores,
        }

    # Re-project full ablation plan using measured v2 sec_per_epoch
    v2_spe = results["v2_full"]["sec_per_epoch"]
    v2_sps = round(v2_spe / 600.0, 4)
    sh([
        sys.executable, "-m", "scripts.remote_job",
        "--manifest", "configs/experiments/kaggle_v2_ablations.yaml",
        "--plan", "--sec-per-step", str(v2_sps),
    ], check=False)

    report = {
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "capability": list(cap),
        "splits_used": splits_arg,
        "precompute_s": enc_s,
        "measured_v2_sec_per_step_est": v2_sps,
        "total_probe_s": round(time.time() - t0, 2),
        "arms": results,
    }
    (WORK / "wave0_probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
