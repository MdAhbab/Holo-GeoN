"""
remote/kaggle_kernel.py - entry point of a Kaggle *script* kernel for one ablation shard.

Attach, in the kernel's settings:
  * the data dataset    (megascale_ddg_pairs.parquet, splits_megascale.parquet,
                         fireprotdb_holdout.parquet - 17 MB in total; NOT the 47 GB
                         DATASETS folder),
  * the code dataset    (a zip of this repository), unless the repo is public and
                         HOLO_REPO_URL is set,
  * GPU = T4 x1, Internet = on (pip and the ESM-2 weights from the Hub).

Choose the shard with the HOLO_SHARD environment variable ("0/3", "1/3", "2/3").
The runner refuses a pre-sm_70 GPU (P100) before starting anything.

Kaggle extracts uploaded archives and turns X.parquet into a folder X.parquet/, so
the runner finds inputs by NAME (scripts/remote_job.find_inputs).
"""
import glob
import os
import shutil
import subprocess
import sys

SHARD = os.environ.get("HOLO_SHARD", "0/3")
MANIFEST = os.environ.get("HOLO_MANIFEST", "configs/experiments/kaggle_ablation.yaml")
REPO_URL = os.environ.get("HOLO_REPO_URL", "")
REPO_BRANCH = os.environ.get("HOLO_REPO_BRANCH", "version-2")
WORK = "/kaggle/working"
REPO = os.path.join(WORK, "Holo-GeoN")
PERSIST = os.path.join(WORK, f"packs_shard{SHARD.split('/')[0]}")

os.environ.setdefault("PYTHONUTF8", "1")


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def get_code() -> None:
    if os.path.isfile(os.path.join(REPO, "scripts", "remote_job.py")):
        return
    hits = glob.glob("/kaggle/input/*/**/scripts/remote_job.py", recursive=True) + \
        glob.glob("/kaggle/input/*/scripts/remote_job.py")
    if hits:
        src = os.path.dirname(os.path.dirname(hits[0]))
        shutil.copytree(src, REPO)
        return
    if REPO_URL:
        sh(f"git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL} {REPO}")
        return
    raise SystemExit("no code: attach the code dataset or set HOLO_REPO_URL")


def main() -> None:
    get_code()
    # Pin what the code depends on; do not reinstall torch (the image's build is
    # matched to the GPU driver, and its version changes without notice, so the
    # runner records it in report.json instead).
    sh("pip install -q 'transformers>=4.40,<5' torch_geometric ViennaRNA pyyaml "
       "pyarrow scikit-learn scipy pandas tqdm")
    os.chdir(REPO)
    sh(f"{sys.executable} -m scripts.remote_job --manifest {MANIFEST} --plan")
    rc = subprocess.run(
        f"{sys.executable} -m scripts.remote_job --manifest {MANIFEST} "
        f"--shard {SHARD} --persist {PERSIST} --out-root {WORK}/runs/shard "
        f"--max-hours 8.5 --data-root /kaggle/input", shell=True).returncode
    # Only finished packs and the report belong in the kernel output.
    print(f"[kernel] runner exit code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
