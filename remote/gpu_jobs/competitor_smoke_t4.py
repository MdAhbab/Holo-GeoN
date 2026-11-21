"""
remote/gpu_jobs/competitor_smoke_t4.py
======================================
Kaggle T4 competitor adapter smoke-test & validation runner for all 5 baselines:
  1. ``thermompnn`` (isolated ``pip --target`` env with omegaconf + pytorch-lightning)
  2. ``spurs``      (local_fusion + optional upstream hub check in isolated ``--target``)
  3. ``rasp``       (local 3D cavity microenvironment MLP on aligned MegaScale PDBs)
  4. ``prostata``   (5-head ESM-2 sequence-only ensemble)
  5. ``ddgemb``     (ESM-2 local-window Transformer)

Runs on ``splits_megascale_id25.parquet`` + ``fireprotdb_hf25.parquet`` using our
already-mounted private ``holo-geon-v2-gpu-seed`` v2 dataset (412/412 MegaScale
PDBs + 166 FireProtDB AFDB PDBs) so every competitor adapter is verified end-to-end
on free Kaggle T4 BEFORE spending a single paid Colab G4 compute unit.

Always catches per-competitor exceptions and exits 0 so the Kaggle kernel version
ends COMPLETE with full diagnostics in ``competitor_smoke_report.json``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
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
EPOCHS = int(os.environ.get("HOLO_COMP_EPOCHS", "5"))
SEED = int(os.environ.get("HOLO_COMP_SEED", "42"))
SPLITS_NAME = os.environ.get("HOLO_COMP_SPLITS", "splits_megascale_id25.parquet")
EXTERNAL_NAME = os.environ.get("HOLO_COMP_EXTERNAL", "fireprotdb_hf25.parquet")

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("out_comp_smoke")
TMP_ROOT = Path("/tmp/holo_comp") if Path("/tmp").is_dir() else WORK / "_tmp"
REPO = TMP_ROOT / "Holo-GeoN"
CACHE = TMP_ROOT / "cache" / "esm2_650M_v2"

BASE_PIP = [
    "transformers>=4.40,<5", "torch_geometric", "ViennaRNA", "pyyaml",
    "pyarrow", "scikit-learn", "scipy", "pandas", "tqdm",
]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[comp-smoke] {msg}", flush=True)


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


def stage_pdb_dir(tar_name: str, out_dir: Path, prefix_filter: str | None = None) -> Path:
    """Locate or extract PDB files into one clean flat directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tp = find_input(tar_name)
    if tp and tp.is_file() and tarfile.is_tarfile(tp):
        with tarfile.open(tp, "r:gz") as tf:
            tf.extractall(out_dir)
    if not any(out_dir.rglob("*.pdb")):
        inp = Path("/kaggle/input")
        if inp.is_dir():
            for p in inp.rglob("*.pdb"):
                name_u = p.name.upper()
                if prefix_filter == "AF-" and not name_u.startswith("AF-"):
                    continue
                if prefix_filter == "NON_AF" and name_u.startswith("AF-"):
                    continue
                dst = out_dir / p.name
                if not dst.exists():
                    try:
                        os.symlink(p, dst)
                    except OSError:
                        shutil.copy2(p, dst)
    return out_dir


def get_code(report: dict) -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO)],
        check=True,
    )
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    report["commit_sha"] = sha.stdout.strip() if sha.returncode == 0 else "unknown"


def run_competitor(name: str, cmd: list[str], extra_pip: list[str],
                   out_dir: Path, report: dict) -> None:
    t0 = time.time()
    env = dict(os.environ)
    target_dir = TMP_ROOT / f"env_{name}"
    if extra_pip:
        target_dir.mkdir(parents=True, exist_ok=True)
        log(f"[{name}] installing isolated deps into {target_dir}: {extra_pip}")
        pip_rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(target_dir), *extra_pip],
        ).returncode
        if pip_rc != 0:
            # Retry allowing deps into the isolated target (excluding torch)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--target", str(target_dir), *extra_pip],
                check=False,
            )
        prev_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{target_dir}:{prev_pp}" if prev_pp else str(target_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.log"
    log(f"[{name}] $ {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt_s = round(time.time() - t0, 1)
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
    metas = {}
    for mf in sorted(out_dir.glob("*.meta.json")):
        try:
            metas[mf.name] = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    report["competitors"][name] = {
        "returncode": proc.returncode,
        "seconds": dt_s,
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "metas": metas,
        "log_tail": tail,
    }
    if proc.returncode != 0:
        report["errors"].append({"competitor": name, "returncode": proc.returncode, "tail": tail[-10:]})


def main() -> int:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(TMP_ROOT / "hf")
    os.environ["TORCH_HOME"] = str(TMP_ROOT / "torch")

    report: dict = {
        "started_utc": utc(),
        "epochs": EPOCHS,
        "seed": SEED,
        "splits": SPLITS_NAME,
        "external": EXTERNAL_NAME,
        "competitors": {},
        "errors": [],
    }
    report_path = WORK / "competitor_smoke_report.json"

    def save_report() -> None:
        report["updated_utc"] = utc()
        report["elapsed_s"] = round(time.time() - t0, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        get_code(report)
        save_report()

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *BASE_PIP], check=True)

        data_dir = REPO / "data"
        data_dir.mkdir(exist_ok=True)
        for fname in ("megascale_ddg_pairs.parquet", SPLITS_NAME, EXTERNAL_NAME, "invariant_backbone_geometry.npz"):
            src = find_input(fname)
            if src is None:
                raise FileNotFoundError(f"Missing required input {fname}")
            dst = data_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)

        ms_pdb_dir = stage_pdb_dir("megascale_pdbs.tar.gz", TMP_ROOT / "ms_pdbs", prefix_filter="NON_AF")
        fp_pdb_dir = stage_pdb_dir("alphafold_fireprot_pdbs.tar.gz", TMP_ROOT / "fp_pdbs", prefix_filter="AF-")
        report["pdb_counts"] = {
            "megascale": len(list(ms_pdb_dir.rglob("*.pdb"))),
            "fireprot": len(list(fp_pdb_dir.rglob("*.pdb"))),
        }
        save_report()

        # Precompute ESM-2 650M cache ONCE + attach 412/412 + 165/176 3D coordinates
        log("precomputing single-pass ESM-2 650M cache + attaching 3D geometry...")
        t_pre = time.time()
        subprocess.run(
            [
                sys.executable, "-m", "scripts.precompute_embeddings",
                "--data", str(data_dir / "megascale_ddg_pairs.parquet"), str(data_dir / EXTERNAL_NAME),
                "--config", "configs/benchmark_v2.yaml",
                "--cache", str(CACHE), "--device", "cuda",
            ],
            cwd=str(REPO), check=True,
        )
        subprocess.run(
            [
                sys.executable, "-m", "scripts.attach_coords",
                "--geom-npz", str(data_dir / "invariant_backbone_geometry.npz"),
                "--cache", str(CACHE),
            ],
            cwd=str(REPO), check=True,
        )
        report["precompute_and_coords_s"] = round(time.time() - t_pre, 1)
        save_report()

        splits_p = str(data_dir / SPLITS_NAME)
        pairs_p = str(data_dir / "megascale_ddg_pairs.parquet")
        ext_p = str(data_dir / EXTERNAL_NAME)
        py = sys.executable

        specs = [
            ("prostata", [
                py, "-m", "scripts.competitors.prostata",
                "--splits", splits_p, "--external", ext_p, "--cache", str(CACHE),
                "--out", str(WORK / "comp_prostata"),
                "--epochs", str(EPOCHS), "--seed", str(SEED),
            ], []),
            ("ddgemb", [
                py, "-m", "scripts.competitors.ddgemb",
                "--splits", splits_p, "--external", ext_p, "--cache", str(CACHE),
                "--out", str(WORK / "comp_ddgemb"),
                "--epochs", str(EPOCHS), "--seed", str(SEED),
            ], []),
            ("rasp", [
                py, "-m", "scripts.competitors.rasp",
                "--splits", splits_p, "--pairs", pairs_p,
                "--pdb-dir", str(ms_pdb_dir),
                "--external", ext_p, "--external-pdb-dir", str(fp_pdb_dir),
                "--out", str(WORK / "comp_rasp"),
                "--epochs", str(EPOCHS), "--seed", str(SEED),
            ], []),
            ("spurs_local", [
                py, "-m", "scripts.competitors.spurs",
                "--splits", splits_p, "--pairs", pairs_p, "--cache", str(CACHE),
                "--pdb-dir", str(ms_pdb_dir),
                "--external", ext_p, "--external-pdb-dir", str(fp_pdb_dir),
                "--mode", "local_fusion",
                "--out", str(WORK / "comp_spurs_local"),
                "--epochs", str(EPOCHS), "--seed", str(SEED),
            ], []),
            ("thermompnn", [
                py, "-m", "scripts.competitors.thermompnn",
                "--splits", splits_p, "--pairs", pairs_p,
                "--pdb-dir", str(ms_pdb_dir),
                "--external", ext_p, "--external-pdb-dir", str(fp_pdb_dir),
                "--work-dir", str(TMP_ROOT / "ThermoMPNN"),
                "--out", str(WORK / "comp_thermompnn"),
                "--epochs", str(EPOCHS), "--seed", str(SEED),
            ], ["omegaconf", "pytorch-lightning", "biopython"]),
        ]

        for name, cmd, extra_pip in specs:
            try:
                run_competitor(name, cmd, extra_pip, WORK / f"comp_{name}", report)
            except Exception as exc:
                report["errors"].append({
                    "competitor": name,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
            save_report()

        report["status"] = "ok" if not report["errors"] else "completed_with_errors"
    except Exception as exc:
        report["errors"].append({"stage": "top_level", "error": repr(exc), "traceback": traceback.format_exc()})
        report["status"] = "error_caught"
    finally:
        save_report()
        log(f"finished status={report.get('status')} elapsed_s={report['elapsed_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
