#!/usr/bin/env python3
"""
runapp.py — launcher for the Holo-GeoN web interface
====================================================

    python runapp.py                          # start the interface
    python runapp.py --weights path/to/model_best.pt
    python runapp.py --port 8080 --no-browser
    python runapp.py --dev                    # backend + Vite hot reload

What it does
------------
1. Creates ``app/backend/.venv`` and installs the HTTP-layer dependencies.
2. Builds the frontend if ``dist/`` is missing (needs Node/npm).
3. Starts uvicorn, waits for ``/api/health``, and opens a browser.

On dependencies
---------------
The interface runs the real model, so it needs the model stack (torch,
transformers, torch_geometric) and a trained checkpoint. There is no lightweight
mode: the predecessor's ``--full`` flag existed because the interface could fall
back to a biophysical heuristic when the model stack was absent, which meant it
always showed numbers whether or not a model existed. Without a checkpoint this
launcher still starts, but every prediction endpoint returns an error explaining
what is missing.

GPU note (RTX 50-series / Blackwell, sm_120): install the CUDA 12.8 wheel first,
or the card errors or silently runs on CPU::

    pip install torch --index-url https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
BACKEND = APP / "backend"
FRONTEND = APP / "frontend"
VENV = BACKEND / ".venv"

DEFAULT_WEIGHTS = ROOT / "runs" / "main" / "model_best.pt"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd, cwd=None, env=None, check=True):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=check)


def _current_env_ready() -> bool:
    """True if the active interpreter already has FastAPI, uvicorn, torch and PyG."""
    for mod in ("fastapi", "uvicorn", "torch", "torch_geometric"):
        try:
            __import__(mod)
        except Exception:
            return False
    return True


def ensure_venv(use_current: bool = False) -> Path:
    if use_current or _current_env_ready():
        print(f"[runapp] using active Python environment: {sys.executable}")
        return Path(sys.executable)
    if not venv_python().exists():
        print(f"[runapp] creating virtualenv at {VENV}")
        run([sys.executable, "-m", "venv", VENV])
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "-q", "pip"])
    run([py, "-m", "pip", "install", "-q", "-r", BACKEND / "requirements.txt"])
    return py


def build_frontend(force: bool) -> None:
    dist = FRONTEND / "dist"
    if dist.exists() and not force:
        print(f"[runapp] frontend already built ({dist})")
        return
    npm = shutil.which("npm")
    if not npm:
        print("[runapp] npm not found — skipping the frontend build. The API will "
              "serve, but the browser interface will not. Install Node.js and "
              "re-run with --rebuild-frontend.")
        return
    if not (FRONTEND / "node_modules").exists():
        run([npm, "install"], cwd=FRONTEND)
    run([npm, "run", "build"], cwd=FRONTEND)


def wait_for_health(port: int, timeout: float = 120.0) -> dict | None:
    """Poll /api/health until the server answers, returning its payload."""
    import json

    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(1.0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch the Holo-GeoN web interface.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 on remote VMs)")
    ap.add_argument("--weights", default="", help="path to a trained checkpoint")
    ap.add_argument("--use-current-env", action="store_true",
                    help="run uvicorn in the current Python environment (recommended on GPU/G4 VMs)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--rebuild-frontend", action="store_true")
    ap.add_argument("--dev", action="store_true", help="also run the Vite dev server")
    args = ap.parse_args()

    py = ensure_venv(use_current=args.use_current_env)
    build_frontend(args.rebuild_frontend)

    env = dict(os.environ)
    weights = Path(args.weights).resolve() if args.weights else DEFAULT_WEIGHTS
    env["HOLOGEON_WEIGHTS"] = str(weights)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH", "")]))

    if not weights.exists():
        print(f"\n[runapp] NOTE: no checkpoint at {weights}.")
        print("[runapp] The interface will start, but every prediction endpoint "
              "will return an error until a model exists. It does not substitute "
              "approximate results. Train one with `python run.py train`.\n")

    server = subprocess.Popen(
        [str(py), "-m", "uvicorn", "app:app", "--host", args.host,
         "--port", str(args.port)],
        cwd=BACKEND, env=env,
    )

    vite = None
    if args.dev:
        npm = shutil.which("npm")
        if npm:
            vite = subprocess.Popen([npm, "run", "dev"], cwd=FRONTEND)

    try:
        health = wait_for_health(args.port)
        if health is None:
            print("[runapp] server did not become healthy in time")
            server.terminate()
            return 1

        url = f"http://127.0.0.1:{args.port}"
        print(f"\n[runapp] listening on {url}")
        if health.get("ready"):
            print(f"[runapp] model loaded from {health.get('weights_path')}")
            if not health.get("uncertainty_calibrated"):
                print("[runapp] uncertainty is NOT calibrated — run "
                      "`python run.py calibration` before treating an interval "
                      "as a 95% interval")
            if health.get("untrained_heads"):
                print(f"[runapp] untrained heads (disabled): "
                      f"{health['untrained_heads']}")
        else:
            print(f"[runapp] no model: {health.get('error')}")

        if not args.no_browser:
            webbrowser.open(url)
        server.wait()
    except KeyboardInterrupt:
        print("\n[runapp] shutting down")
    finally:
        server.terminate()
        if vite:
            vite.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
