"""
remote/kaggle_kernel.py - hardened entry point for a Kaggle GPU *script* kernel
running one shard of a Holo-GeoN experiment manifest.

Design rules (extras/kaggle-colab-lessons.md):
  * Repo clone, HF_HOME, TORCH_HOME, ESM-2 cache, and temporary run outputs ALL
    live under ``/tmp/holo_run`` so ``/kaggle/working`` stays small and clean.
  * Only finished run packs, logs of failures, heartbeat, report.json, and
    ``kernel_bootstrap_report.json`` are written to ``/kaggle/working``.
  * ``--no-persist-cache`` keeps the multi-GB ESM-2 cache in ``/tmp`` rather
    than copying it into ``/kaggle/working``.
  * ``torch`` is NEVER reinstalled; its version + CUDA build are logged before
    and after pip so silent image changes are recorded.
  * Every stage is wrapped in try/except and the script ALWAYS exits 0 after
    writing ``kernel_bootstrap_report.json`` so the Kaggle kernel version ends
    COMPLETE (returning full logs and all finished packs rather than hiding
    diagnostics behind an ERROR state).
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

SHARD = os.environ.get("HOLO_SHARD", "0/5")
MANIFEST = os.environ.get("HOLO_MANIFEST", "configs/experiments/kaggle_v2_ablations.yaml")
MAX_HOURS = os.environ.get("HOLO_MAX_HOURS", "10.0")
ONLY = os.environ.get("HOLO_ONLY", "").strip()
REPO_URL = os.environ.get("HOLO_REPO_URL", "https://github.com/MdAhbab/Holo-GeoN.git")
REPO_BRANCH = os.environ.get("HOLO_REPO_BRANCH", "version-2")

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_kaggle_shard")
TMP_ROOT = Path("/tmp/holo_run") if Path("/tmp").is_dir() else WORK / "_tmp"
REPO = TMP_ROOT / "Holo-GeoN"
CACHE = TMP_ROOT / "cache" / "esm2_650M_v2"
OUT_ROOT = TMP_ROOT / "runs" / "shard"
SHARD_TAG = SHARD.replace("/", "_of_")
PERSIST = WORK / f"packs_shard_{SHARD_TAG}"

PIP_PACKAGES = [
    "transformers>=4.40,<5", "torch_geometric", "ViennaRNA", "pyyaml",
    "pyarrow", "scikit-learn", "scipy", "pandas", "tqdm",
]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[kaggle-kernel] {msg}", flush=True)


def torch_build() -> str:
    r = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else "unavailable: " + r.stderr.strip()[-200:]


def find_code_dataset() -> Path | None:
    inp = Path("/kaggle/input")
    if not inp.is_dir():
        return None
    for hit in sorted(inp.rglob("scripts/remote_job.py")):
        if hit.is_file():
            return hit.parent.parent
    return None


def get_code(report: dict) -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.parent.mkdir(parents=True, exist_ok=True)
    src = find_code_dataset()
    if src is not None:
        log(f"copying code from attached dataset: {src}")
        shutil.copytree(src, REPO)
        report["code_source"] = f"dataset:{src}"
    elif REPO_URL:
        log(f"cloning {REPO_URL}@{REPO_BRANCH} -> {REPO}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO)],
            check=True,
        )
        report["code_source"] = f"git:{REPO_URL}@{REPO_BRANCH}"
    else:
        raise RuntimeError("No code dataset attached and HOLO_REPO_URL is empty")

    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    report["commit_sha"] = sha.stdout.strip() if sha.returncode == 0 else "unknown"


def main() -> int:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(TMP_ROOT / "hf")
    os.environ["TORCH_HOME"] = str(TMP_ROOT / "torch")

    report: dict = {
        "started_utc": utc(),
        "manifest": MANIFEST,
        "shard": SHARD,
        "max_hours": MAX_HOURS,
        "only": ONLY or None,
        "persist": str(PERSIST),
        "errors": [],
    }
    report_path = WORK / "kernel_bootstrap_report.json"

    def write_report() -> None:
        report["updated_utc"] = utc()
        report["elapsed_s"] = round(time.time() - t0, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        get_code(report)
        write_report()

        report["torch_before_pip"] = torch_build()
        log("installing required packages (leaving torch intact)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES],
            check=True,
        )
        report["torch_after_pip"] = torch_build()
        write_report()

        # Print plan first for transparent logging
        plan_cmd = [
            sys.executable, "-m", "scripts.remote_job",
            "--manifest", MANIFEST, "--shard", SHARD, "--plan",
        ]
        log("$ " + " ".join(plan_cmd))
        subprocess.run(plan_cmd, cwd=str(REPO), check=False)

        run_cmd = [
            sys.executable, "-m", "scripts.remote_job",
            "--manifest", MANIFEST,
            "--shard", SHARD,
            "--persist", str(PERSIST),
            "--out-root", str(OUT_ROOT),
            "--cache", str(CACHE),
            "--no-persist-cache",
            "--max-hours", str(MAX_HOURS),
            "--data-root", "/kaggle/input",
        ]
        if ONLY:
            run_cmd += ["--only", *ONLY.split()]
        log("$ " + " ".join(run_cmd))
        rc = subprocess.run(run_cmd, cwd=str(REPO)).returncode
        report["runner_returncode"] = rc
        report["status"] = "ok" if rc == 0 else "completed_with_failures"
    except Exception as exc:
        log(f"ERROR: {exc!r}")
        report["errors"].append({
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        report["status"] = "error_caught"
    finally:
        # Mirror remote_job's report.json to top-level /kaggle/working if present
        rj_rep = OUT_ROOT / "report.json"
        if rj_rep.is_file():
            shutil.copy2(rj_rep, WORK / "remote_job_report.json")
        write_report()
        log(f"finished status={report.get('status')} elapsed_s={report['elapsed_s']}")

    # Always return 0 so Kaggle finishes COMPLETE and preserves all outputs/logs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
