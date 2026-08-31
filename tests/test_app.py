"""
tests/test_app.py
=================
Live exercise of the web backend, in both states it can be in.

**Without a checkpoint** every prediction endpoint must return 503 with a
reason. This is the behaviour that replaced demo mode: the predecessor answered
these same requests with a biophysical heuristic and a fabricated confidence
interval, so the interface looked identical whether or not a model existed.

**With a checkpoint** the endpoints must return well-formed payloads, no field
may be named ``demo_mode``, and the confidence interval must be derived from the
model's predicted sigma rather than a fixed-width formula.

The checkpoint used here holds randomly-initialised weights and a synthetic
encoder, so the *values* are meaningless — only the contract is under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tests.helpers import check, random_protein, section

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
PORT = 8931


def _request(path: str, payload: dict | None = None, timeout: float = 120.0):
    """Return (status, body). Errors carry a status too, not an exception."""
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:
            return exc.code, {}


def _wait_ready(proc, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            status, _ = _request("/api/health", timeout=3)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _serve(env_extra: dict):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), env.get("PYTHONPATH", "")])
    )
    env.update(env_extra)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=BACKEND, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def test_backend_without_model() -> None:
    section("web backend — no checkpoint present")

    with tempfile.TemporaryDirectory(prefix="holo_geon_app_") as tmp:
        proc = _serve({
            "HOLOGEON_WEIGHTS": os.path.join(tmp, "does_not_exist.pt"),
            "HOLOGEON_DB": os.path.join(tmp, "history.db"),
        })
        try:
            if not _wait_ready(proc):
                check(False, "backend started")
                print((proc.stdout.read() if proc.stdout else "")[-2000:])
                return
            check(True, "backend started")

            status, health = _request("/api/health")
            check(status == 200, "health responds")
            check(health.get("ready") is False, "health reports the model is not ready")
            check(bool(health.get("error")), "health explains why it is not ready")
            check("demo_mode" not in health, "no demo_mode field in health")

            for path, payload in [
                ("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P"}),
                ("/api/scan", {"sequence": SEQ[:20], "start": 1, "end": 5}),
                ("/api/idr", {"sequence": SEQ}),
            ]:
                status, body = _request(path, payload)
                check(status == 503,
                      f"{path} refuses with 503 rather than approximating "
                      f"(got {status})")
                check("detail" in body, f"{path} explains the refusal")
        finally:
            proc.terminate()
            proc.wait(timeout=20)


def _make_checkpoint(path: str) -> None:
    """A structurally valid checkpoint with random weights and a stub encoder."""
    from src.model import HoloGeoN

    torch.manual_seed(0)
    kwargs = {
        "esm_dim": 320, "hidden": 32, "out_dim": 32, "aa_dim": 8,
        "gat_heads": 4, "dropout": 0.1, "use_ssm": True,
        "use_mech_stream": True, "num_species": 0,
    }
    model = HoloGeoN(**kwargs)
    torch.save(
        {
            "model": model.state_dict(),
            # Self-describing: the loader rebuilds from this, so a constructor
            # argument that is not recorded cannot make a checkpoint unloadable.
            "model_kwargs": kwargs,
            "esm_dim": 320,
            "config": {
                "esm_model": "facebook/esm2_t6_8M_UR50D",
                "window": 48, "contact_top_k": 8, "contact_min_sep": 3,
                "graph_top_k": 8, "geo_threshold": 0.30,
                "eps": 0.10, "delta": 0.10, "mech_max_degree": 8,
                "neighborhood_hops": 2,
            },
            "untrained_heads": ["idr", "mfi", "proteomics"],
        },
        path,
    )


def test_backend_with_model() -> None:
    section("web backend — checkpoint loaded (random weights, stub encoder)")

    with tempfile.TemporaryDirectory(prefix="holo_geon_app2_") as tmp:
        ckpt = os.path.join(tmp, "model_best.pt")
        _make_checkpoint(ckpt)

        # The real encoder needs ESM-2 weights, which are not reachable here, so
        # the engine's encoder is replaced with the deterministic test stub via
        # a sitecustomize shim. This tests the HTTP and inference contract, not
        # protein representations.
        shim = os.path.join(tmp, "sitecustomize.py")
        Path(shim).write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from tests.helpers import stub_encoding\n"
            "import src.encoder as _enc\n"
            "class _StubEncoder:\n"
            "    def __init__(self, *a, **k):\n"
            "        self.name = 'TEST-STUB'\n"
            "        self.dim = 320\n"
            "    def encode(self, seq, top_k=24, min_sep=3):\n"
            "        return stub_encoding(seq, dim=320, top_k=8)\n"
            "_enc.EsmEncoder = _StubEncoder\n"
        )
        proc = _serve({
            "HOLOGEON_WEIGHTS": ckpt,
            "HOLOGEON_DB": os.path.join(tmp, "history.db"),
            "PYTHONPATH": os.pathsep.join([tmp, str(ROOT)]),
        })
        try:
            if not _wait_ready(proc):
                check(False, "backend started with a checkpoint")
                print((proc.stdout.read() if proc.stdout else "")[-3000:])
                return

            status, health = _request("/api/health")
            check(health.get("ready") is True,
                  f"health reports ready (error: {health.get('error')})")
            check(health.get("uncertainty_calibrated") is False,
                  "health admits uncertainty is uncalibrated without a calibration run")
            check(health.get("untrained_heads") == ["idr", "mfi", "proteomics"],
                  "health lists heads that were never trained")

            status, ddg = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P"})
            check(status == 200, f"/api/ddg succeeds (got {status}: {ddg})")
            if status == 200:
                check("demo_mode" not in ddg, "ddg payload has no demo_mode field")
                check("sigma" in ddg and ddg["sigma"] > 0,
                      "ddg payload carries the model's predicted sigma")
                width = ddg["ci_high"] - ddg["ci_low"]
                expected = 2 * 1.959964 * ddg["sigma"]
                check(abs(width - expected) < 1e-3,
                      "confidence interval is derived from sigma, not a fixed formula")
                check(ddg["ci_source"].startswith("model"),
                      "payload states where the interval came from")
                check(set(ddg["three_state"]) == {"destabilizing", "neutral", "stabilizing"},
                      "three-state probabilities are returned")

            status, scan = _request("/api/scan",
                                    {"sequence": SEQ[:30], "start": 1, "end": 6})
            check(status == 200, f"/api/scan succeeds (got {status})")
            if status == 200:
                check(len(scan["matrix"]) == 20 and len(scan["matrix"][0]) == 6,
                      "scan returns a 20 x N matrix")
                # The wild-type cell must be exactly zero, since the head is a
                # difference of scores at the same site.
                zeros = [
                    scan["matrix"][scan["aa_order"].index(wt)][col]
                    for col, wt in enumerate(scan["wt_residues"])
                ]
                check(all(abs(v) < 1e-4 for v in zeros),
                      "wild-type cells in the scan are zero")

            status, idr = _request("/api/idr", {"sequence": SEQ})
            check(status == 503,
                  f"/api/idr refuses because that head was never trained (got {status})")

            status, body = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "A8P"})
            check(status == 400, "a wild-type mismatch is rejected with 400")
            status, body = _request("/api/ddg",
                                    {"wt_sequence": "ACGT" * 5, "mutation": "A1C"})
            check(status in (200, 400), "nucleotide-looking input is handled")

            status, hist = _request("/api/history")
            check(status == 200 and "items" in hist, "history endpoint works")
            if status == 200 and hist["items"]:
                check(all("demo_mode" not in item for item in hist["items"]),
                      "history records carry no demo_mode column")
        finally:
            proc.terminate()
            proc.wait(timeout=20)


TESTS = [test_backend_without_model, test_backend_with_model]
