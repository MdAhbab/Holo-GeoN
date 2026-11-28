"""
tests/test_app_v2.py
====================
The v2 web-backend contract, exercised live over HTTP (a uvicorn subprocess, as in
tests/test_app.py): double-mutant epistasis, the optional coding sequence with both paths
side by side, the structure-free vs SE(3)-invariant 3D mode, streamed progress, and the
refusals that keep each of them honest.

Weights are random and the encoder is a deterministic stub, so every VALUE is meaningless;
what is tested is the contract and the properties that must hold for any weights:

* the epistasis head is refused unless it was trained, and a pair typed in either order
  gives the identical answer (the head is exactly symmetric);
* ddg_total = ddg_additive + epistasis_dddg, and ddg_additive equals the sum of the two
  single-mutant /api/ddg answers (the two endpoints cannot disagree);
* a coding sequence must encode the protein; with ViennaRNA missing it is refused, never
  silently ignored; the without-CDS path equals the plain request;
* the default mode reads no coordinates (supplying a structure does not move its answer),
  and the 3D mode is SE(3)-invariant (a rigidly moved structure gives the same answer);
* a stream ends in the same numbers as the blocking endpoint.
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

import numpy as np
import torch

from tests.helpers import check, section

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
PORT = 8932
SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU", "Q": "GLN", "G": "GLY",
         "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
         "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


# --- HTTP helpers ---------------------------------------------------------------------
def _request(path: str, payload: dict | None = None, timeout: float = 120.0):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:
            return exc.code, {}


def _stream(path: str, payload: dict, timeout: float = 120.0):
    """(status, content-type, [(event, data), ...]) of a server-sent-event response."""
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    events, name, data = [], None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif line == "" and name is not None:
                    events.append((name, json.loads("".join(data))))
                    name, data = None, []
            return resp.status, ctype, events
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, "application/json", [("http_error", json.loads(exc.read().decode()))]
        except Exception:
            return exc.code, "", []


def _wait_ready(proc, timeout: float = 120.0) -> bool:
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
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH", "")]))
    env.update(env_extra)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT),
         "--log-level", "warning"],
        cwd=BACKEND, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


# --- fixtures -------------------------------------------------------------------------
def _checkpoint(path: Path, *, epistasis_trained: bool, str_stream: bool = False, seed: int = 0) -> None:
    from src.model import HoloGeoN

    torch.manual_seed(seed)
    kwargs = {"arch": "v2", "esm_dim": 32, "hidden": 32, "out_dim": 32, "aa_dim": 8, "gat_heads": 4,
              "dropout": 0.1}
    cfg = {"arch": "v2", "esm_model": "stub", "window": 64, "contact_top_k": 8, "contact_min_sep": 3,
           "graph_top_k": 8, "geo_threshold": 0.3, "eps": 0.1, "delta": 0.1, "mech_max_degree": 8,
           "neighborhood_hops": 2}
    if str_stream:
        kwargs["use_str_stream"] = True
        cfg["use_str_stream"] = True
    torch.save({"model": HoloGeoN(**kwargs).state_dict(), "model_kwargs": kwargs, "esm_dim": 32, "config": cfg,
                "untrained_heads": ["idr", "mfi", "proteomics"],
                "trained_tasks": ["stability"] + (["epistasis"] if epistasis_trained else [])}, path)


def _shim(tmp: Path, *, vienna: bool) -> None:
    """sitecustomize: a deterministic stub encoder, and ViennaRNA either stubbed in or absent."""
    (tmp / "sitecustomize.py").write_text(
        "import sys, hashlib\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import torch\n"
        "from tests.helpers import stub_encoding\n"
        "import src.encoder as _enc\n"
        "import src.features as _feat\n"
        "class _StubEncoder:\n"
        "    def __init__(self, *a, **k):\n"
        "        self.name = 'TEST-STUB'\n"
        "        self.dim = 32\n"
        "    def encode(self, seq, top_k=24, min_sep=3):\n"
        "        rec = stub_encoding(seq, dim=32, top_k=8)\n"
        "        g = torch.Generator().manual_seed(int(hashlib.sha1(seq.encode()).hexdigest()[:8], 16))\n"
        "        L = len(seq)\n"
        "        rec['layer_embs'] = torch.randn(4, L, 32, generator=g).half()\n"
        "        rec['wt_logprobs'] = torch.log_softmax(torch.randn(L, 20, generator=g), dim=-1).half()\n"
        "        rec['site_entropy'] = torch.rand(L, generator=g).half()\n"
        "        return rec\n"
        "_enc.EsmEncoder = _StubEncoder\n"
        + ("_feat.VIENNA_AVAILABLE = True\n"
           "_feat.mrna_fold_per_residue = lambda dna_seq, length, codon_window=7: "
           "[-0.2 + 0.001 * i for i in range(length)]\n" if vienna else "_feat.VIENNA_AVAILABLE = False\n"),
        encoding="utf-8")


def _cds_for(seq: str, stop: bool = True) -> str:
    from src.codon_tables import GENETIC_CODE

    first: dict[str, str] = {}
    for codon, aa in GENETIC_CODE.items():
        if aa != "*" and aa not in first:
            first[aa] = codon
    return "".join(first[a] for a in seq) + ("TAA" if stop else "")


def _backbone(seq: str, seed: int = 5) -> np.ndarray:
    """(L, 4, 3) N, CA, C, CB of a helix-like chain with non-degenerate frames."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(seq), 4, 3))
    for i in range(len(seq)):
        ca = np.array([2.3 * np.cos(np.deg2rad(100 * i)), 2.3 * np.sin(np.deg2rad(100 * i)), 1.5 * i])
        ca = ca + 0.2 * rng.normal(size=3)

        def off(r):
            v = rng.normal(size=3)
            return ca + r * v / np.linalg.norm(v)
        out[i] = [off(1.46), ca, off(1.52), off(1.53)]
    return out


def _rigid(coords: np.ndarray, seed: int = 9) -> np.ndarray:
    """A proper rotation (no reflection) plus a translation."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return coords @ q.T + rng.normal(size=3) * 20.0


def _pdb(seq: str, coords: np.ndarray, bfactor: float = 90.0) -> str:
    lines, serial = [], 1
    for i, aa in enumerate(seq):
        for name, xyz in zip(("N", "CA", "C", "CB"), coords[i]):
            lines.append("ATOM  " + f"{serial:5d}" + " " + f" {name:<3s}" + " " + f"{THREE[aa]:>3s}" + " A"
                         + f"{i + 1:4d}" + " " + "   " + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                         + f"{1.0:6.2f}{bfactor:6.2f}")
            serial += 1
    return "\n".join(lines) + "\nEND\n"


# --- tests ----------------------------------------------------------------------------
def test_backend_v2_capabilities() -> None:
    section("web backend v2 - epistasis, CDS, 3D mode, streaming (trained epistasis head, ViennaRNA stubbed in)")
    short = SEQ[:30]
    coords = _backbone(short)
    pdb = _pdb(short, coords)
    with tempfile.TemporaryDirectory(prefix="holo_geon_v2_", ignore_cleanup_errors=True) as tmpd:
        tmp = Path(tmpd)
        _checkpoint(tmp / "main.pt", epistasis_trained=True, seed=0)
        _checkpoint(tmp / "str3d.pt", epistasis_trained=False, str_stream=True, seed=1)
        _shim(tmp, vienna=True)
        proc = _serve({
            "HOLOGEON_WEIGHTS": str(tmp / "main.pt"), "HOLOGEON_WEIGHTS_3D": str(tmp / "str3d.pt"),
            "HOLOGEON_DB": str(tmp / "history.db"), "HOLOGEON_CALIBRATION": str(tmp / "none.json"),
            "HOLOGEON_SCAN_BATCH": "8", "PYTHONPATH": os.pathsep.join([str(tmp), str(ROOT)])})
        try:
            if not _wait_ready(proc):
                check(False, "v2 backend started")
                print((proc.stdout.read() if proc.stdout else "")[-3000:])
                return
            status, health = _request("/api/health")
            caps = health.get("capabilities", {})
            check(health.get("ready") is True, f"health reports ready (error: {health.get('error')})")
            check(caps.get("epistasis", {}).get("available") is True, "health: epistasis is available (head trained)")
            check(caps.get("cds", {}).get("available") is True, "health: CDS is available when ViennaRNA is")
            check(caps.get("invariant_3d", {}).get("available") is True, f"health: the 3D mode is available "
                  f"({caps.get('invariant_3d', {}).get('reason')})")
            check("/api/scan/stream" in caps.get("streaming", {}).get("endpoints", []), "health lists the stream endpoints")
            modes = caps.get("modes", {})
            check("NOT SE(3)" in modes.get("structure_free", "") and "reads no coordinates" in modes.get("structure_free", ""),
                  "health states the default model is structure-free and NOT SE(3)-equivariant")
            check("SE(3)-invariant" in modes.get("invariant_3d", ""), "health calls the 3D mode SE(3)-invariant")

            # --- default path: structure-free, no CDS
            status, plain = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P"})
            check(status == 200 and plain.get("mode") == "structure_free" and plain.get("uses_coordinates") is False,
                  f"/api/ddg defaults to the structure-free mode (got {status})")
            check(plain.get("cds") == {"supplied": False} and "NOT SE(3)" in plain.get("symmetry", ""),
                  "the default payload says no CDS was used and that the model is not SE(3)-equivariant")

            # --- optional coding sequence, both paths side by side
            cds = _cds_for(SEQ)
            status, withc = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P", "cds": cds})
            c = withc.get("cds", {})
            check(status == 200 and c.get("supplied") and c.get("frame_verified") and c.get("n_codons") == len(SEQ)
                  and c.get("stop_codon_stripped") is True, f"a valid CDS is accepted and frame-verified (got {status})")
            check("with_cds" in c and "without_cds" in c, "both CDS paths are returned side by side")
            if "without_cds" in c:
                check(abs(c["without_cds"]["ddg"] - plain["ddg"]) < 1e-4,
                      "the without-CDS path equals the plain request (CDS dropout path)")
                check(c["with_cds"]["ddg"] != c["without_cds"]["ddg"] and abs(
                    c["delta_ddg"] - (c["with_cds"]["ddg"] - c["without_cds"]["ddg"])) < 2e-4,
                      "the CDS reaches the model, and delta_ddg is with minus without")
                check(withc["ddg"] == c["with_cds"]["ddg"], "the headline ddG uses the CDS when one is supplied")
            # replace codon 11 by a codon for a DIFFERENT amino acid (a synonymous swap would be valid)
            bad = cds[:30] + ("TGG" if SEQ[10] != "W" else "GGG") + cds[33:]
            status, body = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P", "cds": bad})
            check(status == 400 and "does not encode" in body.get("detail", ""),
                  f"a CDS that does not encode the protein is refused with where (got {status}: {body.get('detail')})")
            status, body = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P", "cds": cds[:-1]})
            check(status == 400 and "multiple of 3" in body.get("detail", ""), "a CDS with no reading frame is refused")

            # --- epistasis
            pairs = ["V5A+L8P", "L8P+V5A", "K6A+T7A"]
            status, epi = _request("/api/epistasis", {"wt_sequence": SEQ, "pairs": pairs})
            check(status == 200 and epi.get("n_pairs") == 3, f"/api/epistasis scores three pairs (got {status}: {epi})")
            if status == 200:
                a, b, c3 = epi["pairs"]
                check(a["pair"] == b["pair"] == "V5A+L8P", "pairs are reported in ascending site order")
                check(a["ddg_total"] == b["ddg_total"] and a["epistasis_dddg"] == b["epistasis_dddg"],
                      "a pair typed in either order gives the identical answer (the head is exactly symmetric)")
                check(all(abs(r["ddg_total"] - (r["ddg_additive"] + r["epistasis_dddg"])) < 2e-4 for r in epi["pairs"]),
                      "ddg_total = ddg_additive + epistasis_dddg")
                check(all(0.0 <= r["contact_prob"] <= 1.0 for r in epi["pairs"]) and a["seq_separation"] == 3,
                      "contact probability is a probability and the sequence separation is reported")
                s1 = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "V5A"})[1]["ddg"]
                s2 = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P"})[1]["ddg"]
                check(abs(a["ddg_additive"] - (s1 + s2)) < 2e-3,
                      f"ddg_additive equals the sum of the two single-mutant answers ({a['ddg_additive']} vs {s1 + s2:.4f})")
                check(epi["uncertainty_calibrated"] is False and "epistasis term itself has no uncertainty" in
                      epi["uncertainty_note"], "the interval is not presented as a calibrated claim about the epistasis term")
                check(epi["epistasis_head"]["trained"] and epi["epistasis_head"]["contact_gated"] is True
                      and "no coordinates" in epi["epistasis_head"]["gate_input"] and epi["uses_coordinates"] is False,
                      "the gate is the predicted contact graph, with no coordinates")
            status, body = _request("/api/epistasis", {"wt_sequence": SEQ, "pairs": ["L8P+L8A"]})
            check(status == 400, f"two substitutions at one site are refused (got {status})")
            status, body = _request("/api/epistasis", {"wt_sequence": SEQ, "pairs": ["L8P"]})
            check(status == 400 and "two substitutions" in body.get("detail", ""), "a single substitution is not a pair")
            status, withc_epi = _request("/api/epistasis", {"wt_sequence": SEQ, "pairs": ["V5A+L8P"], "cds": cds})
            check(status == 200 and withc_epi["cds"]["supplied"], "epistasis accepts an optional CDS")

            # --- streaming: progress events, and the same numbers as the blocking endpoint
            status, ctype, events = _stream("/api/scan/stream", {"sequence": SEQ[:30], "start": 1, "end": 30})
            names = [e[0] for e in events]
            check(status == 200 and ctype.startswith("text/event-stream"), "/api/scan/stream is a server-sent-event stream")
            check(names[0] == "start" and names[-1] == "result" and names.count("chunk") == 4,
                  f"stream = start, a chunk per batch of 8 sites (4), result (got {names})")
            dones = [e[1]["done"] for e in events if e[0] == "chunk"]
            check(dones == sorted(dones) and dones[-1] == 30 and all("eta_s" in e[1] for e in events if e[0] == "chunk"),
                  "chunks carry increasing done / total and an ETA")
            blocking = _request("/api/scan", {"sequence": SEQ[:30], "start": 1, "end": 30})[1]
            check(events[-1][1]["matrix"] == blocking["matrix"], "the streamed matrix equals the blocking /api/scan matrix")
            check(events[-1][1]["timing"]["total_mutations"] == 600 and events[-1][1]["timing"]["muts_per_sec"] > 0,
                  "the scan reports its own timing (mutations and throughput)")
            status, ctype, events = _stream("/api/epistasis/stream", {"wt_sequence": SEQ, "pairs": pairs})
            check(status == 200 and [e[0] for e in events][0] == "start" and events[-1][0] == "result"
                  and events[-1][1]["n_pairs"] == 3, "/api/epistasis/stream streams the same request")
            status, ctype, events = _stream("/api/scan/stream", {"sequence": SEQ[:30], "start": 1, "end": 5, "cds": bad})
            check(status == 400, f"a refusal arrives as an HTTP error before any stream starts (got {status})")
            status, hist = _request("/api/history?kind=epistasis")
            check(status == 200 and len(hist.get("items", [])) >= 2, "epistasis requests are recorded in history")

            # --- structure-free vs SE(3)-invariant 3D
            short_free = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P"})[1]
            status, with_struct = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "structure_pdb": pdb})
            check(status == 200 and with_struct["ddg"] == short_free["ddg"] and with_struct["uses_coordinates"] is False,
                  "the default mode ignores a supplied structure: structure-free means it reads no coordinates")
            status, body = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "invariant_3d"})
            check(status == 400 and "structure_pdb" in body.get("detail", ""), "the 3D mode without a structure is refused")
            status, d3 = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "invariant_3d",
                                               "structure_pdb": pdb})
            check(status == 200 and d3["mode"] == "invariant_3d" and d3["uses_coordinates"] is True
                  and "SE(3)-invariant" in d3["symmetry"], f"the 3D mode answers from the 3D checkpoint (got {status}: {d3})")
            if status == 200:
                check(d3["structure"]["aligned_offset"] == 0 and abs(d3["structure"]["mean_plddt"] - 90.0) < 0.1,
                      "the structure is aligned to the sequence and its pLDDT read")
                check(d3["uncertainty_calibrated"] is False and "not calibrated for the 3D checkpoint" in d3["ci_source"],
                      "the 3D interval is not presented as calibrated")
                moved = _pdb(short, _rigid(np.round(coords, 3)))
                d3m = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "invariant_3d",
                                            "structure_pdb": moved})[1]
                shuffled = _pdb(short, np.roll(coords, 7, axis=0))
                d3s = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "invariant_3d",
                                            "structure_pdb": shuffled})[1]
                dm, ds = abs(d3m["ddg"] - d3["ddg"]), abs(d3s["ddg"] - d3["ddg"])
                check(dm < 5e-4, f"the 3D mode is SE(3)-invariant: a rigidly moved structure moves ddG by {dm:.5f}")
                check(ds > 2e-3 and ds > 4 * dm,
                      f"...while a different structure does move it ({ds:.4f} vs {dm:.5f}), so the test has power")
            status, cmp_ = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "structure_free",
                                                 "structure_pdb": pdb, "compare_modes": True})
            comp = cmp_.get("comparison", {})
            check(status == 200 and set(["structure_free", "invariant_3d"]) <= set(comp)
                  and abs(comp["delta_ddg_3d_minus_structure_free"] - (comp["invariant_3d"]["ddg"] - comp["structure_free"]["ddg"])) < 2e-4,
                  "compare_modes returns both modes and the delta")
            # A structure missing a terminal tail (the MegaScale 1UBQ model has 72 of ubiquitin's
            # 76 residues) used to be refused with HTTP 400 "structure_shorter_than_sequence".
            tail_cut = _pdb(short[:26], coords[:26])
            status, part = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "structure_free",
                                                 "structure_pdb": tail_cut, "compare_modes": True})
            pst = part.get("comparison", {}).get("structure", {})
            check(status == 200 and pst.get("sequence_coverage", {}).get("covered") == 26
                  and pst.get("residues_without_coordinates") == [27, 28, 29, 30]
                  and pst.get("mutation_site_masked") is False,
                  f"a structure covering residues 1-26 of 30 is aligned, its tail masked (got {status}: {part.get('detail')})")
            status, part3 = _request("/api/ddg", {"wt_sequence": short, "mutation": "A28G", "mode": "invariant_3d",
                                                  "structure_pdb": tail_cut})
            check(status == 200 and part3.get("structure", {}).get("mutation_site_masked") is True,
                  "a mutation outside the covered span is reported as having no coordinates")
            status, body = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "invariant_3d",
                                                 "structure_pdb": _pdb(short[:20], coords[:20])})
            check(status == 400 and "covers only" in body.get("detail", ""),
                  f"a structure covering under 80% of the sequence is refused (got {status})")
            status, body = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "invariant_3d",
                                                 "structure_pdb": _pdb("A" * 30, coords)})
            check(status == 400 and "aligned" in body.get("detail", ""), "a structure of another protein is refused, not used")
            status, scan3 = _request("/api/scan", {"sequence": short, "start": 1, "end": 6, "mode": "invariant_3d",
                                                   "structure_pdb": pdb})
            check(status == 200 and scan3["mode"] == "invariant_3d" and len(scan3["matrix"]) == 20
                  and len(scan3["matrix"][0]) == 6, "a scan can run in the 3D mode")
            status, body = _request("/api/ddg", {"wt_sequence": short, "mutation": "L8P", "mode": "teleport"})
            check(status == 400, "an unknown mode is rejected")
        finally:
            proc.terminate()
            proc.wait(timeout=30)


def test_backend_v2_refusals() -> None:
    section("web backend v2 - refuses rather than approximates (untrained epistasis head, no ViennaRNA, no 3D checkpoint)")
    with tempfile.TemporaryDirectory(prefix="holo_geon_v2b_", ignore_cleanup_errors=True) as tmpd:
        tmp = Path(tmpd)
        _checkpoint(tmp / "main.pt", epistasis_trained=False, seed=0)
        _shim(tmp, vienna=False)
        proc = _serve({
            "HOLOGEON_WEIGHTS": str(tmp / "main.pt"), "HOLOGEON_WEIGHTS_3D": "",
            "HOLOGEON_DB": str(tmp / "history.db"), "HOLOGEON_CALIBRATION": str(tmp / "none.json"),
            "PYTHONPATH": os.pathsep.join([str(tmp), str(ROOT)])})
        try:
            if not _wait_ready(proc):
                check(False, "v2 backend (refusal configuration) started")
                print((proc.stdout.read() if proc.stdout else "")[-3000:])
                return
            health = _request("/api/health")[1]
            caps = health.get("capabilities", {})
            check(caps["epistasis"]["available"] is False and "never trained" in (caps["epistasis"]["reason"] or ""),
                  "health: epistasis unavailable, and says the head was never trained")
            check(caps["cds"]["available"] is False and "ViennaRNA" in (caps["cds"]["reason"] or ""),
                  "health: CDS unavailable, and says ViennaRNA is missing")
            check(caps["invariant_3d"]["available"] is False and "HOLOGEON_WEIGHTS_3D" in (caps["invariant_3d"]["reason"] or ""),
                  "health: the 3D mode is unavailable, and says which checkpoint is missing")
            status, body = _request("/api/epistasis", {"wt_sequence": SEQ, "pairs": ["V5A+L8P"]})
            check(status == 503 and "never trained" in body.get("detail", ""),
                  f"/api/epistasis refuses with the reason instead of returning a noise-level number (got {status})")
            status, ctype, events = _stream("/api/epistasis/stream", {"wt_sequence": SEQ, "pairs": ["V5A+L8P"]})
            check(status == 503, "the stream endpoint refuses the same way, before streaming")
            status, body = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P", "cds": _cds_for(SEQ)})
            check(status == 503 and "ViennaRNA" in body.get("detail", ""),
                  "a CDS is refused when ViennaRNA is missing, not silently ignored")
            status, body = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P", "mode": "invariant_3d",
                                                 "structure_pdb": "ATOM"})
            check(status == 503, "the 3D mode is refused without a 3D checkpoint, not answered by the structure-free model")
            status, plain = _request("/api/ddg", {"wt_sequence": SEQ, "mutation": "L8P"})
            check(status == 200 and plain["mode"] == "structure_free", "the plain request is unaffected")
        finally:
            proc.terminate()
            proc.wait(timeout=30)


TESTS = [test_backend_v2_capabilities, test_backend_v2_refusals]
