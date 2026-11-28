"""
remote/gpu_jobs/l4_ui_recapture.py
==================================
Recapture the seven publication screenshots of the Holo-GeoN web interface on a
Colab L4 (or any CUDA VM) from the ALREADY-TRAINED A100 release bundle. No training,
no new predictions beyond what the interface itself requests.

    python -m remote.gpu_jobs.l4_ui_recapture \
        --bundle /content/drive/MyDrive/holo_geon_g4/runs/a100_suite/block3_demo/release_bundle \
        --out    /content/drive/MyDrive/holo_geon_g4/runs/a100_suite/ui_recapture_l4

What it does, failing fast where an input is missing
----------------------------------------------------
1. Preflight: the bundle holds ``hologe_v2_lean_best.pt`` and ``calibration.json``
   (required), ``hologe_v2_3d_best.pt`` (for ui_07) and ``hologe_v2_epistasis_best.pt``
   (for ui_06). Records GPU, torch, transformers and node versions.
2. Copies the checkpoints to local disk (/tmp), rebuilds the frontend if dist/ is
   missing or older than its sources, and makes sure Playwright + Chromium exist.
3. Finds the MegaScale ``1UBQ.pdb`` (``--ubq-pdb``, else extracted from
   ``<drive_root>/pdbs/megascale_pdbs.tar.gz``) for ui_07; without it ui_07 is skipped.
4. Server A (lean + 3D + calibration) -> ui_01..05 and ui_07.
   Server B (epistasis checkpoint, its own uncalibrated sigma) -> ui_06.
   Only one server holds the GPU at a time.
5. Every PNG is mirrored to ``--out`` the moment it is written, followed by
   ``ui_recapture_report.json`` (versions, health snapshots, per-shot checks, errors).

Cost (L4 at about 1.5 compute units/hour): roughly 6-9 minutes wall clock on a fresh VM
(pip + Playwright/Chromium + npm build + ESM-2 650M download dominate; the captures
themselves take about 2 minutes), i.e. about 0.15-0.25 units. A reused VM is faster.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))  # append, never insert at 0 (shadowing lesson)

from remote.gpu_jobs.a100_ui_capture import (  # noqa: E402
    DEFAULT_GB1_EPISTASIS_PAIRS,
    SHOT_NAMES,
    _ensure_playwright,
    capture_ui,
    ensure_frontend_built,
)

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/holo_geon_g4"
DEFAULT_BUNDLE = f"{DEFAULT_DRIVE_ROOT}/runs/a100_suite/block3_demo/release_bundle"
DEFAULT_OUT = f"{DEFAULT_DRIVE_ROOT}/runs/a100_suite/ui_recapture_l4"

SEQS = {
    "ubq": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "gb1": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    # Human TP53 (UniProt P04637) DNA-binding domain construct, residues 94-292
    "p53_dbd": (
        "SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPP"
        "GTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFR"
        "HSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVR"
        "VCACPGRDRRTEEENLRKK"
    ),
}

LEAN = "hologe_v2_lean_best.pt"
CKPT_3D = "hologe_v2_3d_best.pt"
CKPT_EPI = "hologe_v2_epistasis_best.pt"
CALIB = "calibration.json"


def log(msg: str) -> None:
    print(f"[l4-ui {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cmd_version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else None
    except Exception:
        return None


def environment() -> dict:
    env: dict = {"python": platform.python_version(), "platform": platform.platform()}
    env["gpu"] = _cmd_version(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    env["node"] = _cmd_version(["node", "--version"])
    env["npm"] = _cmd_version(["npm", "--version"])
    for mod in ("torch", "transformers", "torch_geometric", "fastapi", "uvicorn", "playwright"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "present")
        except Exception as exc:
            env[mod] = f"missing ({type(exc).__name__})"
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda"] = torch.version.cuda
    except Exception:
        env["cuda_available"] = False
    try:
        import RNA  # noqa: F401
        env["viennarna"] = "present"
    except Exception:
        env["viennarna"] = "missing (CDS capability will report unavailable)"
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                           text=True, timeout=10).stdout.strip() or None
        env["git_branch"] = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT,
                                           capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        pass
    return env


def preflight(bundle: Path) -> dict:
    """Fail fast on what every shot needs; record what the optional shots need."""
    if not bundle.is_dir():
        raise FileNotFoundError(f"release bundle not found: {bundle} (is Drive mounted?)")
    found = {name: (bundle / name).is_file() for name in (LEAN, CALIB, CKPT_3D, CKPT_EPI)}
    missing_required = [n for n in (LEAN, CALIB) if not found[n]]
    if missing_required:
        raise FileNotFoundError(f"release bundle {bundle} lacks required file(s): {missing_required}")
    sizes = {n: (bundle / n).stat().st_size for n, ok in found.items() if ok}
    log(f"Bundle OK: {json.dumps({n: f'{s / 1e6:.1f} MB' for n, s in sizes.items()})}")
    if not found[CKPT_3D]:
        log(f"NOTE: {CKPT_3D} missing -> ui_07 (3D comparison) will be skipped")
    if not found[CKPT_EPI]:
        log(f"NOTE: {CKPT_EPI} missing -> ui_06 (epistasis) will be skipped")
    return {"present": found, "bytes": sizes}


def stage_bundle(bundle: Path, work: Path) -> Path:
    """Copy checkpoints to local disk: torch.load over Drive FUSE is slow and can stall."""
    dst = work / "bundle"
    dst.mkdir(parents=True, exist_ok=True)
    for name in (LEAN, CALIB, CKPT_3D, CKPT_EPI):
        src = bundle / name
        if src.is_file() and (not (dst / name).is_file() or (dst / name).stat().st_size != src.stat().st_size):
            t0 = time.time()
            shutil.copy2(src, dst / name)
            log(f"Copied {name} ({src.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    return dst


def find_ubq_pdb(explicit: str | None, drive_root: Path, work: Path) -> tuple[str | None, str]:
    """The MegaScale AlphaFold model of 1UBQ (B-factor = pLDDT, as the 3D checkpoint expects)."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"--ubq-pdb {p} does not exist")
        return p.read_text(encoding="utf-8", errors="ignore"), str(p)
    for cand in (work / "1UBQ.pdb", Path("/tmp/holo_a100/pdbs/megascale"), Path("/tmp/holo_g4/pdbs/megascale")):
        if cand.is_file():
            return cand.read_text(encoding="utf-8", errors="ignore"), str(cand)
        if cand.is_dir():
            for p in cand.rglob("*.pdb"):
                if "1UBQ" in p.name.upper():
                    return p.read_text(encoding="utf-8", errors="ignore"), str(p)
    tar_path = drive_root / "pdbs" / "megascale_pdbs.tar.gz"
    manifest = drive_root / "MANIFEST.json"
    if not tar_path.is_file() and manifest.is_file():
        try:
            for entry in json.loads(manifest.read_text(encoding="utf-8")).get("files", []):
                if entry.get("rel", "").endswith("megascale_pdbs.tar.gz"):
                    tar_path = drive_root / entry["rel"]
                    break
        except Exception:
            pass
    if tar_path.is_file():
        with tarfile.open(tar_path, "r:*") as tf:
            for member in tf:
                base = os.path.basename(member.name).upper()
                if member.isfile() and base.endswith(".PDB") and "1UBQ" in base:
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    text = fh.read().decode("utf-8", errors="ignore")
                    (work / "1UBQ.pdb").write_text(text, encoding="utf-8")
                    return text, f"{tar_path}::{member.name}"
    return None, "not found (pass --ubq-pdb /path/to/1UBQ.pdb)"


class Server:
    """One uvicorn process for app/backend/app.py with the given checkpoint environment."""

    def __init__(self, port: int, env_extra: dict, logfile: Path):
        self.port = port
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([p for p in (env.get("PYTHONPATH"), str(REPO_ROOT)) if p])
        env["PYTHONUTF8"] = "1"
        env.update(env_extra)
        self.logfile = logfile
        self.fh = open(logfile, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(REPO_ROOT / "app" / "backend"), env=env, stdout=self.fh, stderr=subprocess.STDOUT, text=True,
        )
        self.url = f"http://127.0.0.1:{port}"

    def wait_health(self, timeout_s: int = 600) -> dict:
        """/api/health loads the model (and downloads ESM-2 650M on a fresh VM)."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server on {self.port} exited (rc={self.proc.returncode}); see {self.logfile}")
            try:
                with urllib.request.urlopen(f"{self.url}/api/health", timeout=600) as r:
                    health = json.loads(r.read().decode())
                if not health.get("ready"):
                    raise RuntimeError(f"model not ready on {self.port}: {health.get('error')}")
                log(f"Server {self.port} ready in {time.time() - t0:.0f}s")
                return health
            except RuntimeError:
                raise
            except Exception:
                time.sleep(2.0)
        raise TimeoutError(f"server on {self.port} not healthy after {timeout_s}s; see {self.logfile}")

    def post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        req = urllib.request.Request(f"{self.url}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
        self.fh.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recapture the Holo-GeoN UI screenshots from the A100 release bundle.")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE, help="Drive folder with the trained checkpoints")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where PNGs + report go (Drive or local)")
    ap.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT, help="used to find pdbs/megascale_pdbs.tar.gz")
    ap.add_argument("--ubq-pdb", default=None, help="explicit path to the MegaScale 1UBQ.pdb (for ui_07)")
    ap.add_argument("--work", default="/tmp/holo_l4_ui", help="local scratch (checkpoints, logs, PNGs)")
    ap.add_argument("--scale", type=int, default=2, help="device scale factor (2 = 2x DPI)")
    ap.add_argument("--panels", action="store_true", help="also save tight figure-panel crops")
    args = ap.parse_args(argv)

    t_start = time.time()
    bundle, out, work = Path(args.bundle), Path(args.out), Path(args.work)
    shots_local = work / "shots"
    for d in (work, shots_local, out):
        d.mkdir(parents=True, exist_ok=True)
    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "bundle": str(bundle), "out": str(out)}
    report_path = out / "ui_recapture_report.json"

    def save_report() -> None:
        report["elapsed_s"] = round(time.time() - t_start, 1)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def mirror(path: Path) -> None:
        rel = path.relative_to(shots_local) if shots_local in path.parents else Path(path.name)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)

    rc = 0
    try:
        report["environment"] = environment()
        log(f"Environment: {json.dumps(report['environment'])}")
        report["preflight"] = preflight(bundle)
        save_report()

        local_bundle = stage_bundle(bundle, work)
        report["frontend_build"] = ensure_frontend_built(REPO_ROOT, log=log, rebuild_if_stale=True)
        sync_pw, pw_err = _ensure_playwright(log)
        if sync_pw is None:
            raise RuntimeError(f"Playwright unavailable: {pw_err}")
        ubq_pdb_text, pdb_source = (None, "skipped: no 3D checkpoint")
        if report["preflight"]["present"][CKPT_3D]:
            ubq_pdb_text, pdb_source = find_ubq_pdb(args.ubq_pdb, Path(args.drive_root), work)
        report["ubq_pdb_source"] = pdb_source
        log(f"1UBQ.pdb: {pdb_source}")
        save_report()

        captures: list[dict] = []

        # ---- Server A: lean checkpoint + SE(3)-invariant 3D checkpoint + calibration -> ui_01..05, ui_07
        env_a = {
            "HOLOGEON_WEIGHTS": str(local_bundle / LEAN),
            "HOLOGEON_CALIBRATION": str(local_bundle / CALIB),
            "HOLOGEON_DB": str(work / "history_main.db"),
        }
        if report["preflight"]["present"][CKPT_3D]:
            env_a["HOLOGEON_WEIGHTS_3D"] = str(local_bundle / CKPT_3D)
        srv = Server(8011, env_a, work / "uvicorn_8011.log")
        try:
            report["health_main"] = srv.wait_health()
            t0 = time.time()
            srv.post("/api/ddg", {"wt_sequence": SEQS["ubq"], "mutation": "L8P"})  # warm-up (CUDA kernels)
            report["warmup_ms"] = round((time.time() - t0) * 1000, 1)
            if ubq_pdb_text:
                try:
                    cm = srv.post("/api/ddg", {"wt_sequence": SEQS["ubq"], "mutation": "L8P",
                                               "compare_modes": True, "structure_pdb": ubq_pdb_text})
                    report["compare_modes_api"] = {k: cm.get(k) for k in ("mutation", "ddg", "comparison")}
                    log("compare_modes on 1UBQ: HTTP 200")
                except Exception as exc:
                    report["compare_modes_api_error"] = f"{type(exc).__name__}: {exc}"
                    log(f"WARNING: compare_modes API check failed: {exc}")
            save_report()
            captures.append(capture_ui(
                srv.url, None, shots_local, SEQS, log=log,
                af_uniprot_candidates=("P04637", "P69905"), mut_p53_dbd="R82H",
                ubq_pdb_text=ubq_pdb_text, device_scale_factor=args.scale,
                panel_dir=(shots_local / "panels") if args.panels else None, on_shot=mirror,
            ))
        finally:
            srv.stop()
            shutil.copy2(work / "uvicorn_8011.log", out / "uvicorn_8011.log")

        # ---- Server B: epistasis checkpoint (no ddG calibration applies to it) -> ui_06
        if report["preflight"]["present"][CKPT_EPI]:
            env_b = {
                "HOLOGEON_WEIGHTS": str(local_bundle / CKPT_EPI),
                "HOLOGEON_DB": str(work / "history_epi.db"),
                # the lean checkpoint's calibration does not describe this model's sigma
                "HOLOGEON_CALIBRATION": str(work / "no_calibration_for_epistasis_checkpoint.json"),
            }
            srv2 = Server(8012, env_b, work / "uvicorn_8012.log")
            try:
                report["health_epistasis"] = srv2.wait_health()
                captures.append(capture_ui(
                    None, srv2.url, shots_local, SEQS, log=log, gb1_pairs=DEFAULT_GB1_EPISTASIS_PAIRS,
                    device_scale_factor=args.scale,
                    panel_dir=(shots_local / "panels") if args.panels else None, on_shot=mirror,
                ))
            finally:
                srv2.stop()
                shutil.copy2(work / "uvicorn_8012.log", out / "uvicorn_8012.log")
        else:
            report.setdefault("skipped", {})[SHOT_NAMES["epistasis"]] = f"{CKPT_EPI} not in bundle"

        merged: dict = {"screenshots": [], "panels": [], "shot_details": {}, "failures": {}, "skipped": {},
                        "console_errors": [], "page_errors": []}
        for c in captures:
            for k in ("screenshots", "panels", "console_errors", "page_errors"):
                merged[k].extend(c.get(k, []))
            for k in ("shot_details", "failures", "skipped"):
                merged[k].update(c.get(k, {}))
        merged["skipped"].update(report.get("skipped", {}))
        report["ui_capture"] = merged
        expected = sorted(SHOT_NAMES.values())
        got = sorted(merged["screenshots"])
        report["missing_shots"] = [s for s in expected if s not in got]
        log(f"Captured {len(got)}/{len(expected)}: {got}")
        if report["missing_shots"]:
            log(f"Missing: {report['missing_shots']} | failures={merged['failures']} | skipped={merged['skipped']}")
            rc = 1
    except Exception as exc:
        report["fatal"] = f"{type(exc).__name__}: {exc}"
        log(f"FATAL: {report['fatal']}")
        rc = 2
    finally:
        save_report()
        log(f"Report: {report_path}  (elapsed {time.time() - t_start:.0f}s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
