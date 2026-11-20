"""
remote/gpu_jobs/wave0_t4_probe.py
=================================
Wave 0 Kaggle T4 timing / VRAM / smoke probe (Holo-GeoN v2 vs v1).

Run this ONCE, as a Kaggle *script* kernel (GPU T4 x1, Internet on, the seed
dataset attached), BEFORE the 4-account 5-seed suites
(`configs/experiments/kaggle_v2_ablations.yaml`, `kaggle_v2_geometry.yaml`) are
launched. Its only job is to replace the ASSUMED numbers in those manifests with
MEASURED ones (~/.claude/kaggle-colab-lessons.md: measure the time per epoch on the
target GPU with the real model before quoting a cost, and size long runs from a
steady-state measurement, not from a short run that includes start-up cost).
Expect tens of minutes, not a fixed figure: the precompute and the `v1_baseline`
arm (row-level batching, several times more steps per epoch than the Protein-Batch
v2 arms) dominate. `total_probe_s` in the report is the real number.

The probe trains ONLY on the MegaScale split `splits_megascale_id25.parquet`
(`--pathogenicity-splits none`). FireProtDB and S669 are evaluation-only holdouts:
they are encoded and scored, never trained on (a guard refuses otherwise).

Stages, each wrapped in try/except so the kernel version ends COMPLETE
-----------------------------------------------------------------------
get_code, pip_required, pip_optional, stage_data, environment (refuses a pre-sm_70
GPU), esm_prewarm, precompute, attach_coords, then the arms in this order:
`v2_full` (the headline rate), `v2_both_streams` (v2_full + `--use-str-stream` =
geometry arm C; skipped, with a recorded reason, if the geometry NPZ was not staged
or attach_coords failed), `v1_baseline` (slowest, so last). After each arm: the
external evaluations (`fireprotdb_hf25`, `fireprotdb_holdout`, `s669_holdout`) and a
small-file copy into /kaggle/working. Finally: `--plan` re-projection of BOTH
manifests plus an in-report per-arm-class projection.

Why the kernel must never end in ERROR (~/.claude/kaggle-colab-lessons.md, Kaggle,
seen 2026-11-08): for a notebook whose newest version ends in ERROR, `kaggle kernels
output` and `kernels logs` return the PREVIOUS successful version's files and an empty
log. Every stage therefore catches its exception, appends
{stage, exc_type, message, traceback} to `errors`, and the probe goes on. The report is
rewritten after EVERY stage. Exit status is 0 whenever the report reached disk (this
includes the refusal of a P100 and a budget cut-off); it is non-zero only if the report
could not be written anywhere.

Disk: /kaggle/working is the kernel's output, so it stays small. The repo clone, the
Hugging Face and Torch caches (HF_HOME / TORCH_HOME), the ESM cache and every run
directory live under /tmp/holo_wave0. Only `wave0_probe_report.json` and, per arm, a
`probe_<arm>/` folder holding `metrics.json`, `external_*.json` and prediction parquets
up to 25 MB are copied to /kaggle/working. The ESM cache is NOT saved or reusable.
Free disk is printed and recorded before and after the ESM-2 download.

Environment knobs
-----------------
HOLO_REPO_DIR                     an existing checkout (must contain scripts/train.py) to use
                                  as-is; the Kaggle bootstrap clones to /tmp/holo_wave0/Holo-GeoN
                                  and exports this. Unset or invalid: the probe clones to
                                  /tmp/holo_wave0/Holo-GeoN itself (never into /kaggle/working)
HOLO_REPO_URL, HOLO_REPO_BRANCH   repo to clone (default MdAhbab/Holo-GeoN, version-2)
HOLO_PROBE_EPOCHS                 epochs per arm, default 3 (epoch 1 is excluded from
                                  the steady rate, so epochs 2..N must exist: N >= 2)
HOLO_PROBE_ARM_TIMEOUT_H          per-arm wall-clock limit, default 2.5
HOLO_PROBE_TOTAL_BUDGET_H         whole-probe limit, default 6 (the Kaggle batch limit is
                                  11.5 h and sessions have overrun budgets by ~0.3 h)
HOLO_PROBE_MIN_FREE_GB            refuse the ESM download below this much free /tmp (8)

Report: /kaggle/working/wave0_probe_report.json (off Kaggle: out_wave0/)
-----------------------------------------------------------------------
Top-level keys (schema_version 2):
  schema_version, status ("running" | "complete" | "complete_with_errors" | "aborted" |
  "interrupted"), abort_reason, started_utc, finished_utc, updated_utc, total_probe_s,
  report_path, knobs, paths, environment, provenance, disk, stages, esm_download,
  precompute, attach_coords, arms, measured_rates, projections, timing_semantics,
  warnings, errors.

  environment   gpu_name, compute_capability, vram_total_gb, driver_version,
                driver_cuda_version, torch_version, torch_cuda_version,
                transformers_version, python_version, packages, torch_arch_list, ...
  provenance    repo_url, repo_branch_requested, repo_commit (+ note), code_source,
                inputs{name: {source_path, staged_path, size_bytes, sha256}},
                seed_manifest, splits_used, train_tables, evaluation_only_tables.
  arms.<tag>    status, skipped_reason, steps_per_epoch (parsed from train.py stdout, or
                null), epoch_seconds, first_epoch_s, steady_epoch_s, steady_sec_per_step
                (THE headline rate; null + rate_status="unavailable" when it cannot be
                measured), naive_* (all-epoch means; never size from them), startup_s,
                peak_vram_gb + vram{...}, val_pearson, test_pearson, test_rmse,
                parameters, external{<table stem>: {rows_scored, distinct_proteins,
                pooled_metrics, ...}}, epoch_lines, stdout_tail.
  projections   <manifest name>: {plan: {plan_sec_per_step, plan_stdout, ...},
                per_arm_class: {by_class, per_arm, total_h_per_class_rates,
                total_h_single_rate, ...}}.

Everything in `external.*.pooled_metrics` and in `test_pearson` is POOLED over all rows.
Per-protein medians and cluster-bootstrap CIs come only from
`scripts/score_predictions.py` run on the full runs.
"""
from __future__ import annotations

import codecs
import collections
import contextlib
import datetime as dt
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

T0 = time.time()
ENV_PROBLEMS: list[str] = []


def _env_num(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw)
    except ValueError:
        ENV_PROBLEMS.append(f"{name}={raw!r} is not a valid {cast.__name__}; using {default}")
        return default


REPO_URL = os.environ.get("HOLO_REPO_URL", "https://github.com/MdAhbab/Holo-GeoN.git")
REPO_BRANCH = os.environ.get("HOLO_REPO_BRANCH", "version-2")
PROBE_EPOCHS = max(1, _env_num("HOLO_PROBE_EPOCHS", 3, int))
ARM_TIMEOUT_H = _env_num("HOLO_PROBE_ARM_TIMEOUT_H", 2.5, float)
TOTAL_BUDGET_H = _env_num("HOLO_PROBE_TOTAL_BUDGET_H", 6.0, float)
MIN_FREE_GB = _env_num("HOLO_PROBE_MIN_FREE_GB", 8.0, float)
ARM_TIMEOUT_S = ARM_TIMEOUT_H * 3600.0
TOTAL_BUDGET_S = TOTAL_BUDGET_H * 3600.0
SEED = 42
GPU_SAMPLE_INTERVAL_S = 2.0
MIN_ARM_START_S = 120.0           # do not start an arm with less budget than this
TAIL_LINES = 40                   # stdout lines kept per stage in the report
MAX_COPY_BYTES = 25 * 1024 * 1024  # "small" prediction parquets copied to /kaggle/working

# Non-arm stage time limits (seconds); each is also capped by the remaining total budget.
STAGE_TIMEOUT_S = {
    "clone": 600, "pip": 1200, "environment": 180, "prewarm": 1800,
    "precompute": 5400, "attach_coords": 900, "external": 1200, "plan": 120,
}

# Paths. /kaggle/working is the kernel OUTPUT: keep it small. Everything big is in /tmp.
KAGGLE = Path("/kaggle/working").is_dir()
WORK = (Path("/kaggle/working") if KAGGLE else Path("out_wave0")).resolve()
TMP = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
SCRATCH = TMP / "holo_wave0"


def _resolve_repo() -> Path:
    """HOLO_REPO_DIR if it is set and holds scripts/train.py, else <scratch>/Holo-GeoN (cloned
    on demand by get_code). Never under /kaggle/working, which is the kernel output."""
    raw = os.environ.get("HOLO_REPO_DIR", "").strip()
    if raw:
        cand = Path(raw)
        if (cand / "scripts" / "train.py").is_file():
            return cand.resolve()
        ENV_PROBLEMS.append(f"HOLO_REPO_DIR={raw!r} has no scripts/train.py; ignoring it and "
                            f"cloning to {SCRATCH / 'Holo-GeoN'} instead")
    return SCRATCH / "Holo-GeoN"


REPO = _resolve_repo()
RUNS = SCRATCH / "runs"
CACHE = SCRATCH / "cache" / "esm2_650M_v2"     # probe-private, so it can be wiped safely
HF_HOME = SCRATCH / "hf_home"
TORCH_HOME = SCRATCH / "torch_home"
REPORT_NAME = "wave0_probe_report.json"

CONFIG = "configs/benchmark_v2.yaml"
DEFAULT_ESM = "facebook/esm2_t33_650M_UR50D"
GEOM_NPZ = "invariant_backbone_geometry.npz"
EXT_LEGACY_JSON = "external_fireprotdb.json"

# The seed dataset holds exactly these tables/bundles (+ SEED_MANIFEST.json and the
# extracted alphafold_fireprot_pdbs/ and megascale_pdbs/ folders, which the probe does
# not need: geometry comes from the NPZ).
INPUT_NAMES = (
    "megascale_ddg_pairs.parquet",
    "splits_megascale.parquet",
    "splits_megascale_id25.parquet",
    "splits_megascale_fold_tm60.parquet",
    "fireprotdb_holdout.parquet",
    "fireprotdb_hf25.parquet",
    "s669_holdout.parquet",
    GEOM_NPZ,
)
TRAIN_SPLITS_PREFERRED = "splits_megascale_id25.parquet"
TRAIN_SPLITS_FALLBACK = "splits_megascale.parquet"
ENCODE_TABLES = ("megascale_ddg_pairs.parquet", "fireprotdb_holdout.parquet", "s669_holdout.parquet")
EXTERNAL_TABLES = ("fireprotdb_hf25.parquet", "fireprotdb_holdout.parquet", "s669_holdout.parquet")
# Evaluation-only: encoded and scored, NEVER passed to scripts.train.
EVAL_ONLY_TABLES = frozenset(EXTERNAL_TABLES) | {"fireprotdb_holdout.parquet", "s669_holdout.parquet"}
MANIFESTS = (
    # (manifest, arm class whose measured rate feeds `--plan`)
    ("configs/experiments/kaggle_v2_ablations.yaml", "v2"),
    ("configs/experiments/kaggle_v2_geometry.yaml", "str"),
)

# Arms, cheapest first so an out-of-time session still yields the headline v2 rate.
ARMS = (
    {"tag": "v2_full", "arm_class": "v2", "flags": [], "needs_geometry": False,
     "description": "headline Holo-GeoN v2 (Protein-Batch, beta-NLL, CDS dropout 0.5)"},
    {"tag": "v2_both_streams", "arm_class": "str", "flags": ["--use-str-stream"],
     "needs_geometry": True,
     "description": "v2_full + invariant 3D coordinate stream = geometry arm C (both_streams)"},
    {"tag": "v1_baseline", "arm_class": "v1",
     "flags": ["--arch", "v1", "--nll-beta", "0.0", "--cds-dropout", "0.0", "--no-protein-batch"],
     "needs_geometry": False,
     "description": "legacy v1 (20.6M params, row-level batching, standard NLL); slowest"},
)
ARM_CLASS_SOURCE = {"v2": "v2_full", "str": "v2_both_streams", "v1": "v1_baseline"}

PIP_REQUIRED = ["transformers>=4.40,<5", "torch_geometric", "pyyaml", "pyarrow",
                "scikit-learn", "scipy", "pandas", "tqdm"]
PIP_OPTIONAL = ["ViennaRNA"]      # without it E_mech stays masked; do not fail the probe on it

TIMING_SEMANTICS = {
    "esm_download.est_download_s": (
        "first EsmForMaskedLM.from_pretrained() in a CPU-only pre-warm subprocess "
        "(Hub download + deserialise) MINUS a second, cache-hot from_pretrained(): an "
        "estimate of the one-off Hub transfer a fresh session pays. Not contained in "
        "precompute.wall_s when the pre-warm succeeded (precompute.includes_download)."
    ),
    "precompute.wall_s": (
        "wall-clock of scripts.precompute_embeddings AFTER the pre-warm: imports + weight "
        "load from the local cache + GPU transfer + single-pass encoding + ViennaRNA "
        "mechanistic features + cache writes. Per-session cost of the manifests' precompute item."
    ),
    "arms.*.steady_sec_per_step": (
        "HEADLINE rate = steady_epoch_s / steps_per_epoch. steady_epoch_s is the median "
        "wall-clock between consecutive '[train] epoch i/n' lines for epochs 2..n (epoch 1 "
        "excluded: cuDNN autotune, dataloader warm-up). One epoch = steps_per_epoch optimiser "
        "steps PLUS the per-epoch validation, train-subsample evaluation and checkpoint "
        "writes, so this rate already contains the eval overhead that remote_job.est_hours "
        "adds again as (1 + sizing.eval_overhead). steps_per_epoch is parsed from train.py "
        "stdout, never assumed."
    ),
    "arms.*.naive_*": (
        "Averages that include epoch 1 and/or process start-up. Kept only so they can never "
        "be mistaken for the steady rate; do not size from them."
    ),
    "arms.*.peak_vram_gb": (
        "peak of nvidia-smi memory.used on GPU 0, sampled every ~2 s for the duration of the "
        "arm. Whole-device (includes the CUDA context and PyTorch's cached-but-unallocated "
        "blocks, so >= torch.cuda.max_memory_allocated) and can miss spikes shorter than the "
        "sampling interval. vram.peak_above_baseline_mib subtracts memory in use before launch."
    ),
    "arms.*.external.*.pooled_metrics": (
        "POOLED smoke numbers over all scored rows; large proteins dominate. NOT per-protein. "
        "Per-protein medians and cluster-bootstrap CIs come from scripts/score_predictions.py "
        "on the full runs."
    ),
    "projections.*.plan": (
        "Captured stdout of `scripts.remote_job --plan --sec-per-step <plan_sec_per_step>`. "
        "That plan prices EVERY train arm at one rate and one steps_per_epoch (the manifest's), "
        "so it understates the v1 arm."
    ),
    "projections.*.per_arm_class": (
        "Same est_hours formula, but each arm class uses its own measured rate and measured "
        "steps_per_epoch: v2 arms at v2_full, str-stream arms at v2_both_streams, the v1 arm at "
        "v1_baseline. sizing.strict_factor (1.3) is still an ASSUMPTION: the probe runs without "
        "--strict-determinism whereas the manifests' arms use it."
    ),
}


# ---------------------------------------------------------------------------
# Report plumbing
# ---------------------------------------------------------------------------
def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_report() -> dict:
    return {
        "schema_version": 2,
        "status": "running",
        "abort_reason": None,
        "started_utc": utc(),
        "finished_utc": None,
        "updated_utc": None,
        "total_probe_s": None,
        "report_path": None,
        "knobs": {
            "repo_url": REPO_URL, "repo_branch": REPO_BRANCH, "probe_epochs": PROBE_EPOCHS,
            "seed": SEED, "arm_timeout_h": ARM_TIMEOUT_H, "total_budget_h": TOTAL_BUDGET_H,
            "min_free_gb_for_esm_download": MIN_FREE_GB,
            "gpu_sample_interval_s": GPU_SAMPLE_INTERVAL_S, "stage_timeout_s": dict(STAGE_TIMEOUT_S),
            "config": CONFIG, "strict_determinism": False,
        },
        "paths": {
            "work_output": str(WORK), "scratch": str(SCRATCH), "repo": str(REPO), "cache": str(CACHE),
            "runs": str(RUNS), "hf_home": str(HF_HOME), "torch_home": str(TORCH_HOME),
        },
        "environment": {},
        "provenance": {},
        "disk": [],
        "stages": {},
        "esm_download": {},
        "precompute": {},
        "attach_coords": {},
        "arms": {},
        "measured_rates": {},
        "projections": {},
        "timing_semantics": TIMING_SEMANTICS,
        "warnings": list(ENV_PROBLEMS),
        "errors": [],
    }


REPORT: dict = new_report()
REPORT_STATE = {"written": False}
CHILD: dict = {"proc": None}


def _clean(obj, _depth: int = 0):
    """JSON-safe copy: non-finite floats -> null, Paths -> str, odd objects -> str."""
    if _depth > 40:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset, collections.deque)):
        return [_clean(v, _depth + 1) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if hasattr(obj, "item") and callable(obj.item):      # numpy scalar
        try:
            return _clean(obj.item(), _depth + 1)
        except Exception:
            pass
    return str(obj)


def write_report() -> bool:
    """Atomically rewrite the report. Called after EVERY stage. True if it reached disk."""
    REPORT["updated_utc"] = utc()
    REPORT["total_probe_s"] = round(time.time() - T0, 2)
    try:
        payload = json.dumps(_clean(REPORT), indent=2)
    except Exception as exc:                              # never lose the whole report to one bad value
        payload = json.dumps({"schema_version": 2, "status": "report_serialisation_failed",
                              "error": f"{type(exc).__name__}: {exc}",
                              "errors": _clean(REPORT.get("errors"))}, indent=2)
    candidates = [WORK, Path.cwd(), Path(tempfile.gettempdir())]
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            tmp = base / (REPORT_NAME + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, base / REPORT_NAME)
            if REPORT["report_path"] is None:
                REPORT["report_path"] = str(base / REPORT_NAME)
            REPORT_STATE["written"] = True
            if base != WORK:
                print(f"[wave0] WARNING: could not write the report to {WORK}; wrote {base}", flush=True)
            return True
        except Exception as exc:
            print(f"[wave0] could not write report to {base}: {exc}", flush=True)
    return False


def record_error(stage_name: str, exc: BaseException) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    REPORT["errors"].append({
        "stage": stage_name,
        "exc_type": type(exc).__name__,
        "message": str(exc)[:4000],
        "traceback": tb[-6000:],
    })
    print(f"[wave0] ERROR in {stage_name}: {type(exc).__name__}: {str(exc)[:600]}", flush=True)


class Abort(Exception):
    """Stop the probe (after recording it in the report); the process still exits 0."""


class StageTimeout(RuntimeError):
    pass


class SubprocessFailed(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class TotalBudgetExceeded(BudgetExceeded):
    pass


class ArmTimeout(BudgetExceeded):
    pass


@contextlib.contextmanager
def stage(name: str):
    """Run one stage; swallow (and record) any Exception so the kernel ends COMPLETE."""
    rec = {"status": "running", "started_utc": utc()}
    REPORT["stages"][name] = rec
    t = time.time()
    print(f"\n[wave0] === stage {name} ===", flush=True)
    try:
        yield rec
        if rec["status"] == "running":
            rec["status"] = "ok"
    except Abort:
        rec["status"] = "aborted"
        raise
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        record_error(name, exc)
    finally:
        rec["seconds"] = round(time.time() - t, 2)
        rec["finished_utc"] = utc()
        write_report()


def remaining_budget_s() -> float:
    return TOTAL_BUDGET_S - (time.time() - T0)


def budgeted(default_s: float) -> float:
    """A stage time limit capped by what is left of the total budget."""
    rem = remaining_budget_s()
    if rem <= 0:
        raise TotalBudgetExceeded(
            f"total probe budget of {TOTAL_BUDGET_H} h is used up "
            f"({(time.time() - T0) / 3600:.2f} h elapsed)")
    return min(float(default_s), rem)


def require(what: str, paths) -> None:
    """Fail fast (lessons file): every input a stage needs must exist BEFORE it starts."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{what}: missing input(s): {missing}")


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------
def _existing_ancestor(p: Path) -> Path:
    p = Path(p)
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def dir_size_bytes(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def disk_snapshot(label: str) -> dict:
    """Print and record free disk (a full /tmp once turned read-only and killed a session)."""
    snap: dict = {"label": label, "utc": utc()}
    for key, base in (("tmp", TMP), ("work", WORK)):
        try:
            u = shutil.disk_usage(_existing_ancestor(base))
            snap[f"{key}_free_gb"] = round(u.free / 2 ** 30, 2)
            snap[f"{key}_total_gb"] = round(u.total / 2 ** 30, 2)
        except Exception as exc:
            snap[f"{key}_error"] = repr(exc)
    for key, d in (("hf_home", HF_HOME), ("esm_cache", CACHE), ("runs", RUNS), ("work_output", WORK)):
        if Path(d).exists():
            snap[f"{key}_used_gb"] = round(dir_size_bytes(d) / 2 ** 30, 3)
    REPORT["disk"].append(snap)
    print("[wave0] disk[" + label + "]: " + ", ".join(
        f"{k}={v}" for k, v in snap.items() if k not in ("label", "utc")), flush=True)
    return snap


# ---------------------------------------------------------------------------
# Subprocess streaming
# ---------------------------------------------------------------------------
_LINE_SPLIT = re.compile(r"\r\n|\r|\n")


def child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               HF_HOME=str(HF_HOME), HF_HUB_CACHE=str(HF_HOME / "hub"),
               TORCH_HOME=str(TORCH_HOME), TQDM_MININTERVAL="10")
    env.pop("TRANSFORMERS_CACHE", None)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def kill_tree(proc, grace_s: float = 10.0) -> None:
    """Kill a child and everything it spawned (dataloader workers)."""
    if proc is None:
        return
    try:
        if os.name == "posix":
            pgid = proc.pid                       # start_new_session=True => pgid == pid
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.time() + grace_s
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.25)
            try:
                os.killpg(pgid, signal.SIGKILL)   # also reaps workers that outlive the parent
            except ProcessLookupError:
                pass
        elif proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _describe(cmd) -> str:
    parts = [str(c) for c in cmd]
    if "-c" in parts:
        i = parts.index("-c")
        if i + 1 < len(parts):
            parts[i + 1] = "<inline code>"
    return " ".join(parts)


def _echo(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode(sys.stdout.encoding or "utf-8", "replace")
                         .decode(sys.stdout.encoding or "utf-8", "replace"))
    sys.stdout.flush()


def run_stream(cmd, *, label: str, cwd=None, timeout_s: float | None = None, on_line=None,
               capture: bool = False, tail_n: int = TAIL_LINES, extra_env: dict | None = None,
               echo: bool = True) -> dict:
    """
    Run `cmd`, stream its merged stdout+stderr to the console AS IT ARRIVES (so the Kaggle
    log still shows progress), timestamp every line, and kill the whole process tree after
    `timeout_s`. `on_line(ts, line)` sees each completed line (split on \\n and \\r).
    Never raises for a non-zero exit: the caller decides (see check_run).
    """
    cmd = [str(c) for c in cmd]
    print(f"$ {_describe(cmd)}", flush=True)
    kw: dict = {}
    if os.name == "posix":
        kw["start_new_session"] = True
    t_launch = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, env=child_env(extra_env),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        bufsize=0, **kw)
    CHILD["proc"] = proc
    state = {"timed_out": False}
    timer = None
    if timeout_s is not None:
        def _expire() -> None:
            state["timed_out"] = True
            print(f"\n[wave0] TIMEOUT: {label} exceeded {timeout_s:.0f} s; killing its process tree",
                  flush=True)
            kill_tree(proc)
        timer = threading.Timer(max(float(timeout_s), 0.5), _expire)
        timer.daemon = True
        timer.start()

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail: collections.deque = collections.deque(maxlen=max(tail_n, 1))
    kept: list[str] | None = [] if capture else None
    counters = {"n": 0}

    def handle(ts: float, text: str) -> None:
        if not text.strip():
            return
        counters["n"] += 1
        tail.append(text[:600])
        if kept is not None:
            kept.append(text)
        if on_line is not None:
            try:
                on_line(ts, text)
            except Exception:                 # a parser bug must never kill the stream
                pass

    pending = ""
    try:
        while True:
            chunk = proc.stdout.read(8192)    # raw FileIO: returns as soon as bytes arrive
            if not chunk:
                break
            ts = time.time()
            text = decoder.decode(chunk)
            if echo:
                _echo(text)
            pending += text
            parts = _LINE_SPLIT.split(pending)
            pending = parts.pop()
            for part in parts:
                handle(ts, part)
        rest = pending + decoder.decode(b"", final=True)
        if rest:
            handle(time.time(), rest)
    except BaseException:
        kill_tree(proc, grace_s=3.0)      # never leave a child running (or block in wait() below)
        raise
    finally:
        if timer is not None:
            timer.cancel()
        try:
            proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
        CHILD["proc"] = None
    t_end = time.time()
    return {"returncode": rc, "timed_out": state["timed_out"], "timeout_s": timeout_s,
            "seconds": round(t_end - t_launch, 2), "t_launch": t_launch, "t_end": t_end,
            "n_lines": counters["n"], "tail": list(tail), "lines": kept}


def check_run(res: dict, label: str) -> None:
    if res["timed_out"]:
        raise StageTimeout(f"{label} timed out after {res['timeout_s']:.0f} s and was killed")
    if res["returncode"] != 0:
        tail = "\n".join(res["tail"][-15:])
        raise SubprocessFailed(f"{label} exited with code {res['returncode']}; last output:\n{tail}")


def marker_json(lines: list[str] | None, marker: str) -> dict:
    for line in reversed(lines or []):
        if marker in line:
            return json.loads(line.split(marker, 1)[1])
    raise ValueError(f"no {marker!r} line in the subprocess output")


# ---------------------------------------------------------------------------
# GPU sampling (nvidia-smi from a daemon thread; the training runs in a subprocess,
# so torch.cuda.max_memory_allocated() in this process would see nothing)
# ---------------------------------------------------------------------------
class GpuSampler:
    QUERY = ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]

    def __init__(self, interval_s: float = GPU_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.mem: list[float] = []
        self.util: list[float] = []
        self.baseline_mib: float | None = None
        self.available: bool | None = None
        self.note: str | None = None
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self):
        r = subprocess.run(self.QUERY, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or f"exit {r.returncode}").strip()[:200])
        first = r.stdout.strip().splitlines()[0]
        fields = [x.strip() for x in first.split(",")]
        mem = float(fields[0])
        try:
            util = float(fields[1])
        except (ValueError, IndexError):
            util = None
        return mem, util

    def _loop(self) -> None:
        fails = 0
        while not self._halt.is_set():
            try:
                mem, util = self._query()
                self.mem.append(mem)
                if util is not None:
                    self.util.append(util)
                fails = 0
            except FileNotFoundError:
                self.available, self.note = False, "nvidia-smi not found"
                return
            except Exception as exc:
                fails += 1
                if fails >= 5:
                    self.available, self.note = False, f"nvidia-smi failed repeatedly: {exc}"
                    return
            self._halt.wait(self.interval_s)

    def start(self) -> "GpuSampler":
        try:
            self.baseline_mib, _ = self._query()      # memory already in use BEFORE the arm starts
            self.available = True
        except FileNotFoundError:
            self.available, self.note = False, "nvidia-smi not found"
            return self
        except Exception as exc:
            self.note = f"baseline query failed: {exc}"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        ok = bool(self.mem)
        peak = max(self.mem) if ok else None
        return {
            "available": ok,
            "note": self.note if not ok else None,
            "interval_s": self.interval_s,
            "n_samples": len(self.mem),
            "baseline_mib": self.baseline_mib,
            "peak_mib": peak,
            "median_mib": round(statistics.median(self.mem), 1) if ok else None,
            "peak_gb": round(peak / 1024.0, 2) if ok else None,
            "peak_above_baseline_mib": (round(peak - self.baseline_mib, 1)
                                        if ok and self.baseline_mib is not None else None),
            "gpu_util_peak_pct": max(self.util) if self.util else None,
            "gpu_util_median_pct": round(statistics.median(self.util), 1) if self.util else None,
            "semantics": "nvidia-smi memory.used, whole GPU 0; see timing_semantics.arms.*.peak_vram_gb",
        }

    def __enter__(self) -> "GpuSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.summary = self.stop()


# ---------------------------------------------------------------------------
# train.py stdout parsing: steps_per_epoch (measured, never assumed) and epoch timing
# ---------------------------------------------------------------------------
class TrainStream:
    """
    Consumes train.py's merged stdout line by line (timestamped on arrival).

    train.py prints, in this order:
      [train] steps_per_epoch=<N> = one expected pass over 'stability' (...)   once, just
                                                                  before the epoch loop
      [train] epoch <i>/<n>  loss=... stability.pearson=... | ...   once per epoch, AFTER
                                                       that epoch's steps and evaluation
    So epoch 1's duration is (its line) - (the steps_per_epoch line) and epoch i>1 is the delta to
    the previous epoch line.
    """

    STEPS_RE = re.compile(r"\[train\]\s*steps_per_epoch\s*=\s*(\d+)")
    EPOCH_RE = re.compile(r"\[train\]\s*epoch\s+(\d+)\s*/\s*(\d+)")

    def __init__(self) -> None:
        self.steps_per_epoch: int | None = None
        self.t_loop_start: float | None = None
        self.epochs: list[dict] = []

    def __call__(self, ts: float, line: str) -> None:
        if self.steps_per_epoch is None:
            m = self.STEPS_RE.search(line)
            if m:
                self.steps_per_epoch = int(m.group(1))
                self.t_loop_start = ts
                return
        m = self.EPOCH_RE.search(line)
        if m:
            self.epochs.append({"epoch": int(m.group(1)), "of": int(m.group(2)), "ts": ts,
                                "line": line.strip()[:300]})

    def summary(self, t_launch: float, t_end: float) -> dict:
        secs: list[float | None] = []
        prev = self.t_loop_start
        for ep in self.epochs:
            secs.append(None if prev is None else round(ep["ts"] - prev, 3))
            prev = ep["ts"]
        steady_vals = [s for s in secs[1:] if s is not None]      # epoch 1 excluded
        known = [s for s in secs if s is not None]
        first = secs[0] if secs else None
        steady = round(statistics.median(steady_vals), 3) if steady_vals else None
        naive_mean = round(sum(known) / len(known), 3) if known else None
        spe = self.steps_per_epoch
        if spe is None:
            reason = ("no '[train] steps_per_epoch=' line in the train.py output, so there is no "
                      "measured divisor; the per-step rate is left unavailable rather than assumed")
        elif steady is None:
            reason = "fewer than 2 timed epochs: epoch 1 is excluded and no steady-state epoch exists"
        else:
            reason = None
        return {
            "steps_per_epoch": spe,
            "steps_per_epoch_source": ("parsed from train.py stdout '[train] steps_per_epoch=N'"
                                       if spe is not None else "absent from train.py stdout"),
            "epochs_logged": len(self.epochs),
            "epoch_seconds": secs,
            "first_epoch_s": first,
            "steady_epoch_s": steady,
            "steady_sec_per_step": round(steady / spe, 6) if (steady is not None and spe) else None,
            "rate_status": "measured_steady" if reason is None else "unavailable",
            "rate_unavailable_reason": reason,
            "naive_mean_epoch_s_incl_first": naive_mean,
            "naive_sec_per_step_incl_first_epoch": (round(naive_mean / spe, 6)
                                                    if (naive_mean is not None and spe) else None),
            "startup_s": (round(self.t_loop_start - t_launch, 2)
                          if self.t_loop_start is not None else None),
            "post_loop_s": round(t_end - self.epochs[-1]["ts"], 2) if self.epochs else None,
            "epoch_lines": [{"epoch": e["epoch"], "t_since_launch_s": round(e["ts"] - t_launch, 2),
                             "line": e["line"]} for e in self.epochs],
        }


def read_train_metrics(path: Path) -> dict:
    """What train.py's metrics.json really holds (note: 'parameters', not 'param_counts')."""
    m = json.loads(path.read_text(encoding="utf-8"))
    history = m.get("history") or []
    test_stab = (m.get("test") or {}).get("stability") or {}
    wall_h = m.get("wall_clock_hours")
    return {
        "epochs_run": len(history),
        "best_epoch": m.get("best_epoch"),
        "best_checkpoint_selected": bool(m.get("best_checkpoint")),
        "trained_tasks": m.get("trained_tasks"),
        "val_pearson": m.get("best_val_stability_pearson"),        # at the selected epoch
        "val_pearson_by_epoch": [((h.get("val") or {}).get("stability") or {}).get("pearson")
                                 for h in history],
        # POOLED over every test row; not a per-protein statistic.
        "test_pearson": test_stab.get("pearson"),
        "test_rmse": test_stab.get("rmse"),
        "test_stability": test_stab,
        "parameters": m.get("parameters"),
        "task_sizes_batches": m.get("task_sizes_batches"),
        "metrics_wall_clock_s": round(wall_h * 3600.0, 2) if isinstance(wall_h, (int, float)) else None,
    }


# ---------------------------------------------------------------------------
# Inputs and code
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_input(name: str, roots=("/kaggle/input", ".")) -> Path | None:
    """Find an input BY NAME. Kaggle extracts uploads and turns X.tar.gz / X.parquet into a
    folder X/ holding X, so a directory hit is flattened to the file inside it."""
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


def git_commit(repo: Path) -> tuple[str | None, str | None]:
    """(sha, note). Only trusts a git root that IS the repo, not a parent checkout."""
    try:
        top = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if not top or Path(top).resolve() != Path(repo).resolve():
            return None, "no .git in the repo copy (code was not cloned), so the commit is unknown"
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return (sha or None), None
    except Exception as exc:
        return None, f"git unavailable: {exc}"


def get_code() -> dict:
    info: dict = {"repo_url": REPO_URL, "repo_branch_requested": REPO_BRANCH, "repo_path": str(REPO),
                  "holo_repo_dir_env": os.environ.get("HOLO_REPO_DIR")}
    if (REPO / "scripts" / "train.py").is_file():
        info["code_source"] = ("HOLO_REPO_DIR checkout used as-is" if os.environ.get("HOLO_REPO_DIR")
                               else "pre-existing checkout at repo_path")
    else:
        if REPO.exists():
            shutil.rmtree(REPO, ignore_errors=True)
        REPO.parent.mkdir(parents=True, exist_ok=True)
        hits = (glob.glob("/kaggle/input/*/**/scripts/remote_job.py", recursive=True)
                + glob.glob("/kaggle/input/*/scripts/remote_job.py"))
        if hits:
            src = Path(hits[0]).parent.parent
            # DATASETS/ can be tens of GB; never copy it into /tmp.
            shutil.copytree(src, REPO, ignore=shutil.ignore_patterns("DATASETS", "__pycache__", "runs"))
            info["code_source"] = f"copied from the attached dataset {src}"
        else:
            res = run_stream(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO)],
                             label="git clone", timeout_s=budgeted(STAGE_TIMEOUT_S["clone"]), capture=True)
            check_run(res, "git clone")
            info["code_source"] = f"git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL}"
    sha, note = git_commit(REPO)
    info["repo_commit"], info["repo_commit_note"] = sha, note
    if not (REPO / "scripts" / "train.py").is_file():
        raise FileNotFoundError(f"{REPO}/scripts/train.py is missing after get_code")
    print(f"[wave0] code: {info['code_source']}; commit={sha}", flush=True)
    return info


def stage_data() -> dict[str, str]:
    """Symlink (or copy) each named input into REPO/data and record where it really came from."""
    data_dir = REPO / "data"
    data_dir.mkdir(exist_ok=True)
    staged: dict[str, str] = {}
    inputs: dict[str, dict] = {}
    for n in INPUT_NAMES:
        p = find_input(n)
        if p is None:
            inputs[n] = {"found": False}
            continue
        src = p.resolve()
        dst = data_dir / n
        if dst.is_symlink() or dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        how = "symlink"
        try:
            os.symlink(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
            how = "copy"
        staged[n] = str(dst)
        inputs[n] = {"found": True, "source_path": str(src), "staged_path": str(dst), "how": how,
                     "size_bytes": src.stat().st_size, "sha256": sha256_file(src)}
    REPORT["provenance"]["inputs"] = inputs

    sm = find_input("SEED_MANIFEST.json")
    if sm is not None:
        info = {"source_path": str(sm.resolve()), "size_bytes": sm.stat().st_size,
                "sha256": sha256_file(sm)}
        try:
            if info["size_bytes"] <= 200_000:
                info["content"] = json.loads(sm.read_text(encoding="utf-8"))
        except Exception as exc:
            info["parse_error"] = f"{type(exc).__name__}: {exc}"
        REPORT["provenance"]["seed_manifest"] = info
    else:
        REPORT["provenance"]["seed_manifest"] = None
        REPORT["warnings"].append("SEED_MANIFEST.json not found under /kaggle/input")

    missing = [n for n in INPUT_NAMES if n not in staged]
    print(f"[wave0] Staged {len(staged)} input(s): {sorted(staged)}; not found: {missing}", flush=True)
    return staged


def choose_train_splits(staged: dict[str, str]) -> str:
    """id25 split for the probe; a fallback to splits_megascale.parquet is recorded, never silent."""
    prov = REPORT["provenance"]
    if TRAIN_SPLITS_PREFERRED in staged:
        name, fallback = TRAIN_SPLITS_PREFERRED, False
    elif TRAIN_SPLITS_FALLBACK in staged:
        name, fallback = TRAIN_SPLITS_FALLBACK, True
        REPORT["warnings"].append(
            f"{TRAIN_SPLITS_PREFERRED} not staged; FELL BACK to {TRAIN_SPLITS_FALLBACK}. "
            "steps_per_epoch (and hence every projection) refers to that split instead.")
    else:
        raise Abort(f"neither {TRAIN_SPLITS_PREFERRED} nor {TRAIN_SPLITS_FALLBACK} was found; nothing to train on")
    if name in EVAL_ONLY_TABLES or not name.startswith("splits_megascale"):
        raise Abort(f"refusing to train on {name}: not a MegaScale split table")
    prov["splits_used"] = name
    prov["splits_fallback"] = fallback
    prov["train_tables"] = [name]
    prov["evaluation_only_tables"] = sorted(EVAL_ONLY_TABLES)
    return staged[name]


# ---------------------------------------------------------------------------
# Environment, ESM-2 pre-warm
# ---------------------------------------------------------------------------
ENV_PROBE_CODE = r'''
import json, platform, sys
o = {"python_version": sys.version.split()[0], "python_full": sys.version.replace("\n", " "),
     "platform": platform.platform()}
def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return "MISSING: " + type(e).__name__
o["packages"] = {m: ver(m) for m in ("torch", "torch_geometric", "transformers", "numpy", "pandas",
                                     "scipy", "sklearn", "pyarrow", "yaml", "tqdm", "RNA")}
try:
    import torch
    o["torch_version"] = torch.__version__
    o["torch_cuda_version"] = torch.version.cuda
    o["cudnn_version"] = torch.backends.cudnn.version()
    o["cuda_available"] = bool(torch.cuda.is_available())
    if o["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        o["gpu_name"] = p.name
        o["compute_capability"] = [p.major, p.minor]
        o["vram_total_gb"] = round(p.total_memory / 2 ** 30, 2)
        o["device_count"] = torch.cuda.device_count()
        o["torch_arch_list"] = list(torch.cuda.get_arch_list())
except Exception as e:
    o["torch_error"] = type(e).__name__ + ": " + str(e)
try:
    import transformers
    o["transformers_version"] = transformers.__version__
except Exception as e:
    o["transformers_error"] = type(e).__name__ + ": " + str(e)
print("ENV_PROBE_JSON=" + json.dumps(o))
'''

ESM_PREWARM_CODE = r'''
import gc, json, os, sys, time
t0 = time.time()
from transformers import AutoTokenizer, EsmForMaskedLM
import_s = time.time() - t0
name = sys.argv[1]
t = time.time(); AutoTokenizer.from_pretrained(name); tok_s = time.time() - t
t = time.time(); m = EsmForMaskedLM.from_pretrained(name, attn_implementation="eager"); first_s = time.time() - t
n_params = sum(p.numel() for p in m.parameters())
del m; gc.collect()
t = time.time(); m = EsmForMaskedLM.from_pretrained(name, attn_implementation="eager"); reload_s = time.time() - t
print("ESM_PREWARM_JSON=" + json.dumps({"model": name, "import_s": import_s, "tokenizer_s": tok_s,
      "first_load_s": first_s, "cached_reload_s": reload_s, "n_params": n_params,
      "hf_home": os.environ.get("HF_HOME")}))
'''


def nvidia_smi_facts() -> dict:
    out: dict = {"nvidia_smi_available": False}
    try:
        r = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        out["note"] = "nvidia-smi not found"
        return out
    except Exception as exc:
        out["note"] = f"nvidia-smi failed: {exc}"
        return out
    if r.returncode == 0 and r.stdout.strip():
        f = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        out.update(nvidia_smi_available=True, smi_gpu_name=f[0],
                   driver_version=f[1] if len(f) > 1 else None,
                   smi_memory_total_mib=f[2] if len(f) > 2 else None)
        try:
            full = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
            m = re.search(r"CUDA Version:\s*([\d.]+)", full)
            out["driver_cuda_version"] = m.group(1) if m else None
        except Exception:
            out["driver_cuda_version"] = None
    else:
        out["note"] = (r.stderr or r.stdout or f"exit {r.returncode}").strip()[:200]
    return out


def probe_environment() -> dict:
    """GPU/torch/transformers facts, gathered in a SUBPROCESS so this process never holds a CUDA
    context (which would also show up in the nvidia-smi VRAM samples)."""
    res = run_stream([sys.executable, "-u", "-c", ENV_PROBE_CODE], label="environment probe",
                     cwd=REPO, timeout_s=budgeted(STAGE_TIMEOUT_S["environment"]), capture=True)
    check_run(res, "environment probe")
    env = marker_json(res["lines"], "ENV_PROBE_JSON=")
    env.update(nvidia_smi_facts())
    env["platform_kaggle"] = KAGGLE
    return env


def esm_model_name() -> str:
    try:
        text = (REPO / CONFIG).read_text(encoding="utf-8")
        m = re.search(r"^esm_model:\s*(\S+)", text, re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return DEFAULT_ESM


def do_esm_prewarm() -> dict:
    model = esm_model_name()
    before = disk_snapshot("before_esm_download")
    free = before.get("tmp_free_gb")
    if free is not None and free < MIN_FREE_GB:
        raise RuntimeError(
            f"only {free} GB free under {TMP}; the ESM-2 weights plus the cache need >= {MIN_FREE_GB} GB "
            "(a full /tmp turned read-only and killed an earlier session). Override with "
            "HOLO_PROBE_MIN_FREE_GB if you are sure.")
    res = run_stream([sys.executable, "-u", "-c", ESM_PREWARM_CODE, model], label="ESM-2 pre-warm",
                     cwd=REPO, timeout_s=budgeted(STAGE_TIMEOUT_S["prewarm"]), capture=True,
                     extra_env={"CUDA_VISIBLE_DEVICES": "", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    check_run(res, "ESM-2 pre-warm")
    d = marker_json(res["lines"], "ESM_PREWARM_JSON=")
    after = disk_snapshot("after_esm_download")
    first, reload_s = float(d["first_load_s"]), float(d["cached_reload_s"])
    return {
        "status": "ok",
        "model": d["model"],
        "hf_home": d.get("hf_home"),
        "first_load_s": round(first, 2),
        "cached_reload_s": round(reload_s, 2),
        "est_download_s": round(max(first - reload_s, 0.0), 2),
        "tokenizer_s": round(float(d["tokenizer_s"]), 2),
        "import_s": round(float(d["import_s"]), 2),
        "n_params": d.get("n_params"),
        "subprocess_wall_s": res["seconds"],
        "hf_home_used_gb_after": after.get("hf_home_used_gb"),
        "tmp_free_gb_before": before.get("tmp_free_gb"),
        "tmp_free_gb_after": after.get("tmp_free_gb"),
        "what_is_what": "est_download_s = first_load_s - cached_reload_s (Hub transfer); precompute.wall_s is measured afterwards on a warm local cache",
    }


# ---------------------------------------------------------------------------
# Precompute, attach_coords
# ---------------------------------------------------------------------------
PRE_PARENTS_RE = re.compile(r"unique parents=(\d+)\s+to encode=(\d+)\s+residues=([\d,]+)")
PRE_FAIL_RE = re.compile(r"\[precompute\] done\..*failures=(\d+)")
PRE_CACHE_GB_RE = re.compile(r"projected cache size:\s*([\d.]+)\s*GB")


def do_precompute(staged: dict[str, str], prewarm_ok: bool) -> dict:
    tables = [staged[k] for k in ENCODE_TABLES if k in staged]
    require("precompute", [staged.get("megascale_ddg_pairs.parquet", REPO / "data" / "megascale_ddg_pairs.parquet")])
    if CACHE.exists():
        shutil.rmtree(CACHE, ignore_errors=True)       # a pre-existing cache would make precompute a no-op
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    facts: dict = {}

    def on_line(_ts: float, line: str) -> None:
        m = PRE_PARENTS_RE.search(line)
        if m:
            facts["n_parents"], facts["n_encoded"] = int(m.group(1)), int(m.group(2))
            facts["residues"] = int(m.group(3).replace(",", ""))
        m = PRE_FAIL_RE.search(line)
        if m:
            facts["failures"] = int(m.group(1))
        m = PRE_CACHE_GB_RE.search(line)
        if m:
            facts["projected_cache_gb"] = float(m.group(1))

    cmd = [sys.executable, "-u", "-m", "scripts.precompute_embeddings", "--data", *tables,
           "--config", CONFIG, "--cache", str(CACHE), "--device", "cuda"]
    with GpuSampler() as gpu:
        res = run_stream(cmd, label="precompute_embeddings", cwd=REPO, on_line=on_line,
                         timeout_s=budgeted(STAGE_TIMEOUT_S["precompute"]))
    out = {
        "status": "incomplete",
        "wall_s": res["seconds"],
        "returncode": res["returncode"],
        "includes_download": (not prewarm_ok),
        "wall_s_meaning": ("imports + weight load from the warm local cache + GPU transfer + single-pass "
                           "encoding + mechanistic features + cache writes; NO Hub download"
                           if prewarm_ok else
                           "INCLUDES the Hub download: the pre-warm stage failed, so it is not separable"),
        "tables_encoded": [Path(t).name for t in tables],
        **facts,
        "vram": gpu.summary,
        "cache_used_gb": round(dir_size_bytes(CACHE) / 2 ** 30, 3) if CACHE.exists() else None,
        "stdout_tail": res["tail"],
    }
    if out.get("residues"):
        out["encode_wall_s_per_1k_residues"] = round(res["seconds"] / (out["residues"] / 1000.0), 3)
    REPORT["precompute"] = out
    disk_snapshot("after_precompute")
    check_run(res, "precompute_embeddings")
    out["status"] = "ok"
    return out


def do_attach_coords(staged: dict[str, str]) -> dict:
    npz = staged.get(GEOM_NPZ)
    if not npz:
        raise FileNotFoundError(f"{GEOM_NPZ} was not staged")
    require("attach_coords", [npz, CACHE])
    if not any(CACHE.glob("*.pt")):
        raise FileNotFoundError(f"encoder cache {CACHE} holds no records: precompute did not produce a cache")
    report_json = SCRATCH / "attach_coords_report.json"
    report_json.unlink(missing_ok=True)
    cmd = [sys.executable, "-u", "-m", "scripts.attach_coords", "--cache", str(CACHE),
           "--geom-npz", npz, "--report", str(report_json)]
    res = run_stream(cmd, label="attach_coords", cwd=REPO,
                     timeout_s=budgeted(STAGE_TIMEOUT_S["attach_coords"]))
    out: dict = {"status": "incomplete", "seconds": res["seconds"], "returncode": res["returncode"],
                 "stdout_tail": res["tail"]}
    if report_json.is_file():
        out["summary"] = json.loads(report_json.read_text(encoding="utf-8"))
    REPORT["attach_coords"] = out
    check_run(res, "attach_coords")
    attached = (out.get("summary") or {}).get("attached")
    # attach_coords exits 0 even when nothing was attached; that is a failure for the str arm.
    if not attached:
        raise RuntimeError(f"attach_coords exit 0 but attached={attached!r} record(s): "
                           f"{out.get('summary')}")
    out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def copy_small_outputs(tag: str, out_dir: Path) -> list[dict]:
    """metrics.json, external_*.json and prediction parquets <= 25 MB -> /kaggle/working/probe_<tag>/."""
    dst_dir = WORK / f"probe_{tag}"
    copied: list[dict] = []
    if not out_dir.is_dir():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    wanted = [out_dir / "metrics.json"] + sorted(out_dir.glob("external_*.json")) \
        + sorted(out_dir.glob("pred_*.parquet"))
    for src in wanted:
        if not src.is_file():
            continue
        size = src.stat().st_size
        if src.suffix == ".parquet" and size > MAX_COPY_BYTES:
            copied.append({"file": src.name, "size_bytes": size, "copied": False,
                           "reason": f"larger than {MAX_COPY_BYTES} bytes"})
            continue
        shutil.copy2(src, dst_dir / src.name)
        copied.append({"file": src.name, "size_bytes": size, "copied": True})
    return copied


def run_arm(spec: dict, splits_arg: str, state: dict) -> dict:
    tag = spec["tag"]
    arm: dict = {
        "tag": tag, "arm_class": spec["arm_class"], "description": spec["description"],
        "flags": list(spec["flags"]), "status": "incomplete", "skipped_reason": None,
        "epochs_requested": PROBE_EPOCHS, "seed": SEED, "strict_determinism": False,
        "stability_splits": Path(splits_arg).name,
    }
    REPORT["arms"][tag] = arm

    reason = None
    if state["stop_reason"]:
        reason = f"skipped: {state['stop_reason']}"
    elif spec["needs_geometry"] and not state["geometry_ok"]:
        reason = f"skipped: {state['geometry_reason']}"
    if reason is None and remaining_budget_s() < MIN_ARM_START_S:
        exc = TotalBudgetExceeded(
            f"only {remaining_budget_s():.0f} s of the {TOTAL_BUDGET_H} h total budget is left; "
            f"arm {tag} and the remaining arms were not started")
        record_error(f"arm:{tag}", exc)
        state["stop_reason"] = f"TotalBudgetExceeded: {exc}"
        reason = f"skipped: {state['stop_reason']}"
    if reason:
        arm.update(status="skipped", skipped_reason=reason)
        REPORT["warnings"].append(f"{tag}: {reason}")
        write_report()
        return arm

    arm_timeout = min(ARM_TIMEOUT_S, remaining_budget_s())
    limit_name = "per_arm_timeout" if ARM_TIMEOUT_S <= remaining_budget_s() else "total_budget"
    out_dir = RUNS / f"probe_{tag}"
    arm.update(out_dir=str(out_dir), timeout_s_applied=round(arm_timeout, 1), timeout_limit=limit_name)

    with stage(f"arm:{tag}"):
        ok = False
        try:
            require(f"arm {tag}", [splits_arg, REPO / CONFIG])
            if not CACHE.is_dir() or not any(CACHE.glob("*.pt")):
                raise FileNotFoundError(f"encoder cache {CACHE} is empty: precompute did not complete")
            if out_dir.exists():
                shutil.rmtree(out_dir)       # a leftover last.pt would make train.py RESUME and skew timing
            cmd = [sys.executable, "-u", "-m", "scripts.train", "--config", CONFIG, *spec["flags"],
                   "--stability-splits", splits_arg,
                   "--pathogenicity-splits", "none",     # FireProtDB / S669 are never trained on
                   "--cache", str(CACHE), "--device", "cuda", "--epochs", str(PROBE_EPOCHS),
                   "--seed", str(SEED), "--tag", tag, "--out", str(out_dir)]
            arm["cmd"] = cmd
            arm["started_utc"] = utc()
            parser = TrainStream()
            with GpuSampler() as gpu:
                res = run_stream(cmd, label=f"train {tag}", cwd=REPO, timeout_s=arm_timeout,
                                 on_line=parser)
            arm["finished_utc"] = utc()
            arm["returncode"] = res["returncode"]
            arm["timed_out"] = res["timed_out"]
            arm["wall_clock_s"] = res["seconds"]
            arm.update(parser.summary(res["t_launch"], res["t_end"]))
            if parser.epochs:
                arm["naive_wall_sec_per_epoch_incl_startup"] = round(res["seconds"] / len(parser.epochs), 2)
            arm["vram"] = gpu.summary
            arm["peak_vram_gb"] = gpu.summary.get("peak_gb")
            arm["stdout_tail"] = res["tail"]
            disk_snapshot(f"after_arm_{tag}")

            if res["timed_out"]:
                exc_cls = ArmTimeout if limit_name == "per_arm_timeout" else TotalBudgetExceeded
                msg = f"arm {tag} killed after {arm_timeout:.0f} s ({limit_name}); remaining arms skipped"
                state["stop_reason"] = f"{exc_cls.__name__}: {msg}"
                arm["status"] = "timeout"
                raise exc_cls(msg)
            if res["returncode"] != 0:
                arm["status"] = "error"
                check_run(res, f"train {tag}")
            metrics_path = out_dir / "metrics.json"
            require(f"arm {tag} metrics", [metrics_path])
            arm.update(read_train_metrics(metrics_path))
            if arm.get("trained_tasks") not in (["stability"], None):
                REPORT["warnings"].append(f"{tag}: trained_tasks={arm['trained_tasks']}, expected ['stability']")
            ok = True
            arm["status"] = "ok"
        finally:
            if not ok and arm["status"] == "incomplete":
                arm["status"] = "error"
    return arm


def read_external_result(out_dir: Path, key: str, data_file: str):
    """Read the result of ONE external table: external_<key>.json (key = the data file's stem).

    The legacy external_fireprotdb.json is a fallback ONLY for a `fireprotdb*` key and ONLY when the
    per-key file is absent (an older eval_external). The current eval_external never writes the
    legacy name for S669, so reading it for any other key would file a FireProtDB result under that
    key. Whatever is read must also say it belongs to this table: `table_key` (new checkouts) must
    equal `key`, and `data` (all checkouts) must name this table's file; a mismatch raises."""
    candidates = [(out_dir / f"external_{key}.json", "keyed")]
    if key.startswith("fireprotdb"):
        candidates.append((out_dir / EXT_LEGACY_JSON, "legacy_fallback"))
    for path, fmt in candidates:
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        table_key = d.get("table_key")
        if table_key is not None and table_key != key:
            raise RuntimeError(f"{path.name} has table_key={table_key!r} but table {key!r} was requested; "
                               "refusing to attribute it to this table")
        wrote_for = Path(str(d.get("data", ""))).name
        if wrote_for and wrote_for != data_file:
            raise RuntimeError(f"{path.name} was written for {wrote_for!r}, not {data_file!r}; "
                               "refusing to attribute it to this table")
        if table_key is None and not wrote_for:
            raise RuntimeError(f"{path.name} carries neither table_key nor data, so it cannot be "
                               f"verified as belonging to {key!r}")
        d["_read_from"] = fmt
        return d, path.name, fmt
    return None, None, None


def run_external_evals(tag: str, staged: dict[str, str], state: dict) -> None:
    arm = REPORT["arms"][tag]
    out_dir = RUNS / f"probe_{tag}"
    ext: dict = arm.setdefault("external", {})
    # POOLED smoke numbers only. Every metric recorded below (Pearson, Spearman, RMSE, ...) is
    # pooled over all scored rows, so long proteins dominate and rows within a protein are not
    # independent. The project reports TWO columns separately - pooled AND per-protein median -
    # each with cluster-bootstrap CIs. Per-protein medians and CIs come ONLY from
    # scripts/score_predictions.py run on the full runs' pred_*.parquet. Never present a
    # number from this block as if it were per-protein.
    for data_file in EXTERNAL_TABLES:
        key = Path(data_file).stem
        if data_file not in staged:
            ext[key] = {"status": "skipped", "reason": f"{data_file} was not staged"}
            continue
        if state["stop_reason"]:
            ext[key] = {"status": "skipped", "reason": state["stop_reason"]}
            continue
        rec: dict = {"status": "incomplete", "data_file": data_file, "pooled_only": True}
        ext[key] = rec
        ckpt = out_dir / "model_best.pt"
        with stage(f"external:{tag}:{key}"):
            require(f"eval_external {tag}/{key}", [ckpt, CACHE])   # fail fast, before the subprocess
            # Anything found after the call must have been written BY this call: delete the keyed
            # file and the legacy file first, so a failed eval can never be read back stale.
            for stale in (out_dir / f"external_{key}.json", out_dir / EXT_LEGACY_JSON):
                stale.unlink(missing_ok=True)
            cmd = [sys.executable, "-u", "-m", "scripts.eval_external", "--checkpoint", str(ckpt),
                   "--data", staged[data_file], "--config", CONFIG, "--cache", str(CACHE),
                   "--device", "cuda", "--out", str(out_dir)]
            res = run_stream(cmd, label=f"eval_external {tag}/{key}", cwd=REPO,
                             timeout_s=budgeted(STAGE_TIMEOUT_S["external"]))
            rec["seconds"] = res["seconds"]
            check_run(res, f"eval_external {tag}/{key}")
            d, fname, fmt = read_external_result(out_dir, key, data_file)
            if d is None:
                raise FileNotFoundError(
                    f"eval_external exited 0 but wrote no usable result for {key!r}: expected "
                    f"external_{key}.json" + (f" or the legacy {EXT_LEGACY_JSON}" if key.startswith("fireprotdb")
                                              else f" (the legacy {EXT_LEGACY_JSON} is not read for this table)"))
            rec.update({
                "result_file": fname, "result_format": fmt,
                "rows_scored": d.get("rows_scored"),
                "rows_without_cached_encoding": d.get("rows_without_cached_encoding"),
                "distinct_proteins": d.get("distinct_proteins"),
                "ddg_sign_applied": d.get("ddg_sign_applied"),
                "pooled_metrics": d.get("metrics"),
                "predictions_parquet": Path(str(d.get("predictions_parquet", ""))).name or None,
            })
            rec["status"] = "ok"
        if rec["status"] == "incomplete":       # the stage swallowed an exception (see errors)
            rec["status"] = "error"


# ---------------------------------------------------------------------------
# Re-projection
# ---------------------------------------------------------------------------
def arm_class_of(flags) -> str:
    f = [str(x) for x in flags]
    if "--arch" in f:
        i = f.index("--arch")
        if i + 1 < len(f) and f[i + 1] == "v1":
            return "v1"
    if "--use-str-stream" in f:
        return "str"
    return "v2"


def build_measured_rates() -> dict:
    out = {}
    for cls, tag in ARM_CLASS_SOURCE.items():
        a = REPORT["arms"].get(tag) or {}
        out[tag] = {
            "arm_class": cls,
            "arm_status": a.get("status", "not_run"),
            "steady_sec_per_step": a.get("steady_sec_per_step"),
            "steps_per_epoch": a.get("steps_per_epoch"),
            "rate_status": a.get("rate_status", "unavailable"),
            "rate_unavailable_reason": a.get("rate_unavailable_reason") or a.get("skipped_reason"),
            "naive_sec_per_step_incl_first_epoch": a.get("naive_sec_per_step_incl_first_epoch"),
            "peak_vram_gb": a.get("peak_vram_gb"),
        }
    REPORT["measured_rates"] = out
    return out


def load_remote_job():
    path = REPO / "scripts" / "remote_job.py"
    spec = importlib.util.spec_from_file_location("holo_remote_job_probe", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_sizing(manifest_rel: str) -> dict:
    import yaml
    with open(REPO / manifest_rel, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def plan_rate_for(plan_class: str, sz: dict) -> dict:
    """The --sec-per-step to hand to remote_job for one manifest.

    remote_job.est_hours multiplies steps*rate by (1 + sizing.eval_overhead). The steady epoch time
    already contains the per-epoch evaluation and checkpoint writes, so the rate passed in is
    steady / (1 + eval_overhead); otherwise that overhead would be counted twice."""
    rates = REPORT["measured_rates"]
    eo = float(sz.get("eval_overhead", 0.10))
    src_tag = ARM_CLASS_SOURCE[plan_class]
    steady = rates[src_tag]["steady_sec_per_step"]
    info: dict = {"plan_class": plan_class, "eval_overhead": eo, "measured_from": src_tag}
    if steady is None and plan_class != "v2":
        v2 = rates["v2_full"]["steady_sec_per_step"]
        if v2 is not None:
            steady = v2
            info["measured_from"] = "v2_full"
            info["note"] = (f"FALLBACK: {src_tag} has no measured rate "
                            f"({rates[src_tag]['rate_unavailable_reason']}); the v2_full rate is used and "
                            "UNDERSTATES the str-stream arms")
    if steady is None:
        info.update(steady_sec_per_step=None, plan_sec_per_step=None,
                    rate_source="ASSUMED (manifest default): the probe measured no usable rate")
    else:
        info.update(steady_sec_per_step=steady, plan_sec_per_step=round(steady / (1.0 + eo), 6),
                    rate_source="MEASURED steady state",
                    derivation=f"{steady} / (1 + eval_overhead {eo}) = {round(steady / (1.0 + eo), 6)}")
    return info


def do_plan(manifest_rel: str, plan_class: str) -> dict:
    manifest = read_sizing(manifest_rel)
    sz = manifest["sizing"]
    pr = plan_rate_for(plan_class, sz)
    cmd = [sys.executable, "-m", "scripts.remote_job", "--manifest", manifest_rel, "--plan"]
    if pr.get("plan_sec_per_step"):
        cmd += ["--sec-per-step", f"{pr['plan_sec_per_step']:.6f}"]
    res = run_stream(cmd, label=f"remote_job --plan {manifest_rel}", cwd=REPO,
                     timeout_s=STAGE_TIMEOUT_S["plan"], capture=True)
    out = {**pr, "command": cmd[2:], "returncode": res["returncode"],
           "plan_stdout": "\n".join(res["lines"] or [])}
    REPORT["projections"].setdefault(manifest["name"], {})["plan"] = out
    check_run(res, f"remote_job --plan {manifest_rel}")
    return out


def do_per_arm_class(manifest_rel: str, plan_class: str) -> dict:
    """Same est_hours formula as `--plan`, but every arm class gets its own measured rate and
    measured steps_per_epoch (the manifests price every arm at ONE rate and ONE steps_per_epoch)."""
    rj = load_remote_job()
    manifest = read_sizing(manifest_rel)
    sz = dict(manifest["sizing"])
    eo = float(sz.get("eval_overhead", 0.10))
    rates = REPORT["measured_rates"]
    items = rj.expand_items(manifest)
    notes: list[str] = []

    pre = REPORT.get("precompute", {})
    dl = (REPORT.get("esm_download") or {}).get("est_download_s")
    pre_h = None
    if pre.get("status") == "ok" and pre.get("wall_s"):
        pre_h = (pre["wall_s"] + (dl or 0.0)) / 3600.0
        notes.append("precompute item priced at the MEASURED per-session (ESM download + precompute) time")

    classes: dict[str, dict] = {}
    for cls, tag in ARM_CLASS_SOURCE.items():
        own = rates[tag]
        rate, rate_src = own["steady_sec_per_step"], f"measured ({tag})"
        spe, spe_src = own["steps_per_epoch"], f"measured ({tag})"
        if rate is None:
            if cls != "v2" and rates["v2_full"]["steady_sec_per_step"] is not None:
                rate, rate_src = rates["v2_full"]["steady_sec_per_step"], f"FALLBACK to v2_full rate ({tag} not measured)"
        if rate is not None:
            plan_rate = rate / (1.0 + eo)
        else:
            plan_rate, rate_src = float(sz["sec_per_step"]), "ASSUMED (manifest default)"
        if spe is None:
            if cls == "v1":
                spe_src = "UNAVAILABLE: v1 uses row-level batching; the manifest's steps_per_epoch is a Protein-Batch figure"
            elif cls == "str" and rates["v2_full"]["steps_per_epoch"] is not None:
                spe, spe_src = rates["v2_full"]["steps_per_epoch"], "FALLBACK to v2_full measured steps_per_epoch"
            else:
                spe, spe_src = sz.get("steps_per_epoch"), "ASSUMED (manifest sizing.steps_per_epoch)"
        classes[cls] = {
            "measured_arm": tag, "steady_sec_per_step": rate, "rate_source": rate_src,
            "plan_sec_per_step": round(plan_rate, 6), "steps_per_epoch_used": spe,
            "steps_per_epoch_source": spe_src,
            "steps_per_epoch_manifest_assumed": sz.get("steps_per_epoch"),
            "steps_per_epoch_vs_manifest_pct": (round(100.0 * (spe - sz["steps_per_epoch"]) / sz["steps_per_epoch"], 1)
                                                if spe and sz.get("steps_per_epoch") else None),
            "n_runs": 0, "hours": 0.0 if spe else None,
        }

    per_arm: dict[str, dict] = {}
    non_train_h = 0.0
    for it in items:
        if it["kind"] == "train":
            cls = arm_class_of(it.get("flags", []))
            c = classes[cls]
            row = per_arm.setdefault(it["arm"], {"class": cls, "n_runs": 0, "hours": 0.0})
            row["n_runs"] += 1
            c["n_runs"] += 1
            if c["steps_per_epoch_used"]:
                h = rj.est_hours(it, dict(sz, steps_per_epoch=c["steps_per_epoch_used"]), c["plan_sec_per_step"])
                row["hours"] += h
                c["hours"] += h
            else:
                row["hours"] = None
        else:
            it2 = dict(it)
            if it["kind"] == "precompute" and pre_h is not None:
                it2["est_hours"] = pre_h
            non_train_h += rj.est_hours(it2, sz, 0.0)

    vm_h = float(sz.get("vm_overhead_h", 0.0))
    unavailable = [k for k, c in classes.items() if c["n_runs"] and c["hours"] is None]
    total = vm_h + non_train_h + sum(c["hours"] for c in classes.values() if c["hours"] is not None)

    # What `--plan` prints: one rate and the manifest's steps_per_epoch for EVERY train arm.
    pr = plan_rate_for(plan_class, sz)
    single_rate = pr["plan_sec_per_step"] if pr.get("plan_sec_per_step") else float(sz["sec_per_step"])
    single = vm_h + sum(rj.est_hours(i, sz, single_rate) if i["kind"] == "train"
                        else rj.est_hours(i, sz, 0.0) for i in items)

    notes += [
        "sizing.strict_factor is ASSUMED (the probe runs without --strict-determinism; the manifests' arms use it)",
        "ablation arms that REMOVE a component (no_ssm, no_geo_stream, ...) are priced at the full v2 rate: an upper bound",
        "true_contact_edges (loads real contact maps) is priced at the v2 rate: unmeasured",
        "str_only / str_topology_only drop the geo stream and are priced at the both_streams rate: an upper bound",
        "vm_overhead_h and the export/external/aggregate items stay at the manifest's ASSUMED values",
    ]
    out = {
        "manifest": manifest_rel, "plan_class": plan_class,
        "sizing_assumed": {k: sz.get(k) for k in ("steps_per_epoch", "sec_per_step", "eval_overhead",
                                                  "strict_factor", "epochs")},
        "by_class": classes, "per_arm": {k: {**v, "hours": (round(v["hours"], 3) if v["hours"] is not None else None)}
                                         for k, v in per_arm.items()},
        "non_train_h": round(non_train_h, 3), "vm_overhead_h": vm_h,
        "total_h_per_class_rates": round(total, 2),
        "complete": not unavailable, "unavailable_classes": unavailable,
        "total_h_single_rate": round(single, 2),
        "per_class_minus_single_rate_h": round(total - single, 2),
        "notes": notes,
    }
    for c in classes.values():
        c["hours"] = round(c["hours"], 3) if c["hours"] is not None else None
    REPORT["projections"].setdefault(manifest["name"], {})["per_arm_class"] = out
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_probe() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    for d in (SCRATCH, RUNS, HF_HOME, TORCH_HOME):
        d.mkdir(parents=True, exist_ok=True)
    write_report()                                  # a report exists from the very first second
    disk_snapshot("start")

    with stage("get_code") as st:
        REPORT["provenance"].update(get_code())
    if not (REPO / "scripts" / "train.py").is_file():
        raise Abort("no repo code available; nothing can run")

    with stage("pip_required"):
        res = run_stream([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                          *PIP_REQUIRED], label="pip install (required)", cwd=REPO,
                         timeout_s=budgeted(STAGE_TIMEOUT_S["pip"]))
        check_run(res, "pip install (required)")
    with stage("pip_optional"):
        res = run_stream([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                          *PIP_OPTIONAL], label="pip install (ViennaRNA)", cwd=REPO,
                         timeout_s=budgeted(STAGE_TIMEOUT_S["pip"]))
        check_run(res, "pip install (ViennaRNA)")

    staged: dict[str, str] = {}
    splits_arg = ""
    with stage("stage_data"):
        staged = stage_data()
        if "megascale_ddg_pairs.parquet" not in staged:
            raise Abort("megascale_ddg_pairs.parquet was not found under /kaggle/input; nothing can run")
        splits_arg = choose_train_splits(staged)

    with stage("environment"):
        env = probe_environment()
        REPORT["environment"].update(env)
        print(f"[wave0] GPU: {env.get('gpu_name')} sm_{''.join(str(x) for x in env.get('compute_capability', []))} "
              f"{env.get('vram_total_gb')} GB; driver {env.get('driver_version')}; torch {env.get('torch_version')} "
              f"(CUDA {env.get('torch_cuda_version')}); transformers {env.get('transformers_version')}", flush=True)
        if not env.get("cuda_available"):
            raise Abort("CUDA GPU required for the timing probe (torch.cuda.is_available() is False)")
        cap = env.get("compute_capability") or [0, 0]
        if cap[0] < 7:
            raise Abort(f"Refusing pre-sm_70 GPU ({env.get('gpu_name')}, sm_{cap[0]}{cap[1]}): newer torch builds lack sm_60")
        archs = env.get("torch_arch_list") or []
        if archs and f"sm_{cap[0]}{cap[1]}" not in archs and not any(a.startswith("compute_") for a in archs):
            REPORT["warnings"].append(f"this torch build lists no sm_{cap[0]}{cap[1]} kernels: {archs}")
        if str(env.get("packages", {}).get("RNA", "")).startswith("MISSING"):
            REPORT["warnings"].append("ViennaRNA (module RNA) is missing: E_mech stays masked, "
                                      "so the arms' step time omits the mechanistic stream's real cost")

    prewarm_ok = False
    with stage("esm_prewarm"):
        REPORT["esm_download"] = {"status": "incomplete"}
        REPORT["esm_download"] = do_esm_prewarm()
        prewarm_ok = True
    if not prewarm_ok and REPORT["esm_download"].get("status") == "incomplete":
        REPORT["esm_download"]["status"] = "error"

    with stage("precompute"):
        do_precompute(staged, prewarm_ok)

    state = {"stop_reason": None, "geometry_ok": False, "geometry_reason": None}
    if GEOM_NPZ not in staged:
        state["geometry_reason"] = f"{GEOM_NPZ} was not staged (not found under /kaggle/input)"
        REPORT["attach_coords"] = {"status": "skipped", "reason": state["geometry_reason"]}
    else:
        state["geometry_reason"] = "attach_coords failed (see errors / attach_coords)"
        REPORT["attach_coords"] = {"status": "incomplete"}
        with stage("attach_coords"):
            do_attach_coords(staged)
            state["geometry_ok"] = True
        if REPORT["attach_coords"].get("status") == "incomplete":
            REPORT["attach_coords"]["status"] = "error"

    for spec in ARMS:
        arm = run_arm(spec, splits_arg, state)
        if arm["status"] == "ok":          # a failed/killed arm has no trustworthy checkpoint to score
            run_external_evals(spec["tag"], staged, state)
        if arm["status"] != "skipped":
            with stage(f"copy_outputs:{spec['tag']}"):
                arm["copied_files"] = copy_small_outputs(spec["tag"], RUNS / f"probe_{spec['tag']}")
    build_measured_rates()
    write_report()

    for manifest_rel, plan_class in MANIFESTS:
        name = Path(manifest_rel).stem
        with stage(f"reproject:{name}:plan"):
            do_plan(manifest_rel, plan_class)
        with stage(f"reproject:{name}:per_arm_class"):
            do_per_arm_class(manifest_rel, plan_class)


def finalize() -> None:
    build_measured_rates()
    disk_snapshot("end")
    REPORT["finished_utc"] = utc()
    if REPORT["status"] == "running":
        REPORT["status"] = "complete_with_errors" if REPORT["errors"] else "complete"
    write_report()
    brief = {
        "status": REPORT["status"], "abort_reason": REPORT["abort_reason"],
        "total_probe_s": REPORT["total_probe_s"], "report_path": REPORT["report_path"],
        "gpu": REPORT["environment"].get("gpu_name"),
        "repo_commit": REPORT["provenance"].get("repo_commit"),
        "esm_est_download_s": (REPORT.get("esm_download") or {}).get("est_download_s"),
        "precompute_wall_s": (REPORT.get("precompute") or {}).get("wall_s"),
        "measured_rates": REPORT["measured_rates"],
        "projected_total_h": {k: {"per_class": (v.get("per_arm_class") or {}).get("total_h_per_class_rates"),
                                  "single_rate": (v.get("per_arm_class") or {}).get("total_h_single_rate")}
                              for k, v in REPORT["projections"].items()},
        "errors": [{"stage": e["stage"], "exc_type": e["exc_type"], "message": e["message"][:200]}
                   for e in REPORT["errors"]],
    }
    print("\n[wave0] SUMMARY\n" + json.dumps(_clean(brief), indent=2), flush=True)


def _on_signal(signum, _frame) -> None:
    kill_tree(CHILD.get("proc"), grace_s=3.0)
    REPORT["status"] = "interrupted"
    REPORT["abort_reason"] = f"received signal {signum}"
    write_report()
    sys.exit(0)


def main() -> int:
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass
    try:
        run_probe()
    except Abort as exc:
        REPORT["status"] = "aborted"
        REPORT["abort_reason"] = str(exc)
        record_error("abort", exc)
    except BaseException as exc:                      # even SystemExit/KeyboardInterrupt: still write the report
        kill_tree(CHILD.get("proc"), grace_s=3.0)
        record_error("probe", exc)
        if REPORT["status"] == "running":
            REPORT["status"] = "aborted"
            REPORT["abort_reason"] = f"unhandled {type(exc).__name__}: {str(exc)[:300]}"
    finally:
        try:
            finalize()
        except BaseException as exc:
            print(f"[wave0] finalize failed: {exc!r}", flush=True)
            write_report()
    # Exit 0 whenever the report reached disk, so the kernel version ends COMPLETE and its
    # output/log are visible. Non-zero ONLY if the report could not be written anywhere.
    return 0 if REPORT_STATE["written"] else 2


if __name__ == "__main__":
    sys.exit(main())
