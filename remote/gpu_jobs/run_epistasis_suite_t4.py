"""
remote/gpu_jobs/run_epistasis_suite_t4.py
=========================================
Resumable Kaggle T4 GPU runner for the Holo-GeoN v2 Double-Mutant Epistasis
4-Arm Controlled Suite across 5 seeds on ``id25`` and ``fold_tm60``.

Arms (novelty pillar #1 — Contact-Gated Potts Bilinear Epistasis Head):
  1. ``additive_only``       — zero epistasis baseline (ddG_ab = ddG_a + ddG_b)
  2. ``potts_no_gate``       — Potts bilinear head with constant scalar gate
  3. ``contact_gated_egeo``  — Potts bilinear head gated by structure-free ESM-2 E_geo
  4. ``true_contact_oracle`` — Potts bilinear head gated by true 3D C-beta contacts

Outputs are written directly in the layout consumed by ``scripts/make_figures.py``
(Figure F8):
    /kaggle/working/epistasis_<split>/<arm>_s<seed>/epistasis_metrics.json
    /kaggle/working/epistasis_<split>/<arm>_s<seed>/pred_epistasis_test.parquet
    /kaggle/working/epistasis_suite_report.json

Always catches per-item exceptions and exits 0 so the Kaggle kernel version ends
COMPLETE with full logs and every finished seed/arm intact.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["GIT_TERMINAL_PROMPT"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_URL = os.environ.get("HOLO_REPO_URL", "https://github.com/MdAhbab/Holo-GeoN.git")
REPO_BRANCH = os.environ.get("HOLO_REPO_BRANCH", "version-2")
SPLITS = [s.strip() for s in os.environ.get("HOLO_EPI_SPLITS", "id25,fold_tm60").split(",") if s.strip()]
SEEDS = [int(s.strip()) for s in os.environ.get("HOLO_EPI_SEEDS", "42,43,44,45,46").split(",") if s.strip()]
ARMS = [
    a.strip() for a in os.environ.get(
        "HOLO_EPI_ARMS",
        "additive_only,potts_no_gate,contact_gated_egeo,true_contact_oracle",
    ).split(",") if a.strip()
]
EPOCHS = int(os.environ.get("HOLO_EPI_EPOCHS", "25"))
UNFREEZE = os.environ.get("HOLO_EPI_UNFREEZE", "1").strip() not in ("0", "false", "False")
MAX_HOURS = float(os.environ.get("HOLO_MAX_HOURS", "8.0"))

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_epistasis")
TMP_ROOT = Path("/tmp/holo_epi") if Path("/tmp").is_dir() else WORK / "_tmp"
REPO = TMP_ROOT / "Holo-GeoN"
CACHE = TMP_ROOT / "cache" / "esm2_650M_v2"

PIP_PACKAGES = [
    "transformers>=4.40,<5", "torch_geometric", "ViennaRNA", "pyyaml",
    "pyarrow", "scikit-learn", "scipy", "pandas", "tqdm",
]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[epi-suite] {msg}", flush=True)


def find_input(name: str) -> Path | None:
    for root in ("/kaggle/input", "project/DATASETS", "extras"):
        rp = Path(root)
        if not rp.is_dir():
            continue
        for hit in rp.rglob(name):
            if hit.is_file():
                return hit
            if hit.is_dir() and (hit / name).is_file():
                return hit / name
    return None


def find_pretrained_ckpt(split: str, seed: int) -> Path | None:
    """Locate a matching split-specific pretrained single-mutant Holo-GeoN v2 checkpoint."""
    inp = Path("/kaggle/input")
    if not inp.is_dir():
        return None
    # Strictly match split-specific paths FIRST so fold_tm60 never leaks into id25!
    patterns = (
        f"*{split}*full_s{seed}*.pt",
        f"*{split}*/full_s{seed}/model_best.pt",
        f"{split}_full_s{seed}.pt",
        f"{split}/full_s{seed}/model_best.pt",
    )
    for pat in patterns:
        hits = sorted(inp.rglob(pat))
        if hits:
            return hits[0]
    return None


def get_code(report: dict) -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.parent.mkdir(parents=True, exist_ok=True)
    inp = Path("/kaggle/input")
    src = None
    if inp.is_dir():
        for hit in sorted(inp.rglob("scripts/train_epistasis.py")):
            if hit.is_file():
                src = hit.parent.parent
                break
    if src is not None:
        shutil.copytree(src, REPO)
        report["code_source"] = f"dataset:{src}"
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO)],
            check=True,
        )
        report["code_source"] = f"git:{REPO_URL}@{REPO_BRANCH}"
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    report["commit_sha"] = sha.stdout.strip() if sha.returncode == 0 else "unknown"


def main() -> int:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(TMP_ROOT / "hf")
    os.environ["TORCH_HOME"] = str(TMP_ROOT / "torch")

    report: dict = {
        "started_utc": utc(),
        "splits": SPLITS,
        "seeds": SEEDS,
        "arms": ARMS,
        "epochs": EPOCHS,
        "unfreeze_backbone": UNFREEZE,
        "runs": {},
        "errors": [],
    }
    report_path = WORK / "epistasis_suite_report.json"

    def save_report() -> None:
        report["updated_utc"] = utc()
        report["elapsed_s"] = round(time.time() - t0, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        get_code(report)
        save_report()

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES], check=True)

        # Verify CUDA sm_70+
        chk = subprocess.run(
            [sys.executable, "-c",
             "import json, torch; "
             "print(json.dumps({'cuda': torch.cuda.is_available(), "
             "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
             "'cap': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}))"],
            capture_output=True, text=True, check=True,
        )
        env_info = json.loads(chk.stdout.strip().splitlines()[-1])
        report["environment"] = env_info
        if not env_info.get("cuda") or (env_info.get("cap") or [0])[0] < 7:
            raise RuntimeError(f"Unsupported GPU environment: {env_info}")

        # Stage double-mutant tables and precompute ESM-2 cache ONCE for all parents
        data_dir = REPO / "data"
        data_dir.mkdir(exist_ok=True)
        staged_tables: dict[str, Path] = {}
        precompute_inputs: list[str] = []
        ms_pairs = find_input("megascale_ddg_pairs.parquet")
        if ms_pairs is not None:
            dst = data_dir / "megascale_ddg_pairs.parquet"
            if not dst.exists():
                shutil.copy2(ms_pairs, dst)
            precompute_inputs.append(str(dst))

        for sp in SPLITS:
            fname = f"megascale_double_mutants_{sp}.parquet"
            src_p = find_input(fname)
            if src_p is None:
                raise FileNotFoundError(f"Missing {fname} under /kaggle/input")
            dst_p = data_dir / fname
            if not dst_p.exists():
                shutil.copy2(src_p, dst_p)
            staged_tables[sp] = dst_p
            if str(dst_p) not in precompute_inputs:
                precompute_inputs.append(str(dst_p))

        # Precompute parent embeddings ONCE (single-pass EsmForMaskedLM)
        log("precomputing single-pass ESM-2 650M cache...")
        t_pre = time.time()
        subprocess.run(
            [
                sys.executable, "-m", "scripts.precompute_embeddings",
                "--data", *precompute_inputs,
                "--config", "configs/benchmark_v2.yaml",
                "--cache", str(CACHE),
                "--device", "cuda",
            ],
            cwd=str(REPO),
            check=True,
        )
        report["precompute_s"] = round(time.time() - t_pre, 1)
        save_report()

        # Run seed-major across splits so every completed seed is a full 4-arm paired block
        for sp in SPLITS:
            table_p = staged_tables[sp]
            sp_out_root = WORK / f"epistasis_{sp}"
            sp_out_root.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                for arm in ARMS:
                    if (time.time() - t0) / 3600.0 > MAX_HOURS:
                        log(f"reached MAX_HOURS={MAX_HOURS}; stopping cleanly")
                        report["stopped_by_budget"] = True
                        save_report()
                        return 0
                    tag = f"{sp}/{arm}_s{seed}"
                    run_out = sp_out_root / f"{arm}_s{seed}"
                    metrics_file = run_out / "epistasis_metrics.json"
                    if metrics_file.is_file():
                        log(f"skipping {tag} (already done)")
                        continue
                    ckpt = find_pretrained_ckpt(sp, seed)
                    cmd = [
                        sys.executable, "-m", "scripts.train_epistasis",
                        "--data", str(table_p),
                        "--config", "configs/benchmark_v2.yaml",
                        "--cache", str(CACHE),
                        "--out", str(run_out),
                        "--arm", arm,
                        "--epochs", str(EPOCHS),
                        "--seed", str(seed),
                        "--device", "cuda",
                    ]
                    if UNFREEZE:
                        cmd.append("--unfreeze-backbone")
                    if ckpt is not None:
                        cmd += ["--checkpoint", str(ckpt)]

                    log(f"=== {tag} ===")
                    t_item = time.time()
                    try:
                        proc = subprocess.run(cmd, cwd=str(REPO))
                        dt_s = round(time.time() - t_item, 1)
                        rec = {
                            "returncode": proc.returncode,
                            "seconds": dt_s,
                            "checkpoint_used": str(ckpt) if ckpt else None,
                        }
                        if metrics_file.is_file():
                            m = json.loads(metrics_file.read_text(encoding="utf-8"))
                            rec["test_dddg_pearson"] = (m.get("test") or {}).get("dddg_epistasis", {}).get("pearson")
                            rec["test_ddg_ab_pearson"] = (m.get("test") or {}).get("ddg_ab", {}).get("pearson")
                            rec["contact_lt8_dddg_pearson"] = (
                                (m.get("test") or {}).get("dddg_contact_lt8A", {}).get("pearson")
                            )
                            rec["distant_ge8_dddg_pearson"] = (
                                (m.get("test") or {}).get("dddg_distant_ge8A", {}).get("pearson")
                            )
                        report["runs"][tag] = rec
                        if proc.returncode != 0:
                            report["errors"].append({"stage": tag, "returncode": proc.returncode})
                    except Exception as exc:
                        report["errors"].append({
                            "stage": tag,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        })
                    save_report()

        report["status"] = "ok" if not report["errors"] else "completed_with_errors"
    except Exception as exc:
        log(f"ERROR: {exc!r}")
        report["errors"].append({"stage": "top_level", "error": repr(exc), "traceback": traceback.format_exc()})
        report["status"] = "error_caught"
    finally:
        save_report()
        log(f"done status={report.get('status')} elapsed_s={report['elapsed_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
