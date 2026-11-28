"""
tests/test_a100_ui_capture.py
=============================
Static + contract verification for ``remote/gpu_jobs/a100_ui_capture.py``:
  1. Verifies every DOM id, CSS class, and button/label text referenced in
     ``UI_SELECTORS`` exists in ``app/frontend/src/**/*.tsx``.
  2. Verifies the FastAPI Pydantic request schemas in ``app/backend/app.py``
     match both the frontend TSX payloads and ``a100_unified_suite.py``.
  3. Verifies ``ensure_frontend_built`` behavior on synthetic directories
     (already built vs missing package.json) without invoking npm locally.

    python -m tests.test_a100_ui_capture
"""
from __future__ import annotations

import os
import py_compile
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.helpers import check, section


def test_module_compiles_and_exports() -> None:
    section("a100_ui_capture compiles and exposes public API")
    mod_path = ROOT / "remote" / "gpu_jobs" / "a100_ui_capture.py"
    py_compile.compile(str(mod_path), doraise=True)
    from remote.gpu_jobs import a100_ui_capture as ui

    check(callable(ui.ensure_frontend_built), "ensure_frontend_built is callable")
    check(callable(ui.capture_ui), "capture_ui is callable")
    check("--enable-webgl" in ui.CHROMIUM_WEBGL_ARGS and "--use-angle=swiftshader" in ui.CHROMIUM_WEBGL_ARGS,
          "SwiftShader WebGL launch flags configured for 3Dmol")


def test_selectors_exist_in_tsx() -> None:
    section("every DOM selector/id/text in UI_SELECTORS exists in app/frontend/src")
    from remote.gpu_jobs.a100_ui_capture import UI_SELECTORS

    fe_src = ROOT / "app" / "frontend" / "src"
    app_tsx = (fe_src / "App.tsx").read_text(encoding="utf-8")
    epi_tsx = (fe_src / "tabs" / "EpistasisTab.tsx").read_text(encoding="utf-8")
    # every component, including src/components/*.tsx (the status strip lives there)
    all_tsx = "\n".join(p.read_text(encoding="utf-8") for p in sorted(fe_src.rglob("*.tsx")))

    # Dynamic tab ids in App.tsx: id={`tab-${tab.id}`} for id in TABS
    tab_ids_in_app = set(re.findall(r"\{\s*id:\s*'([a-z]+)'", app_tsx))
    check("id={`tab-${tab.id}`}" in app_tsx and "id={`panel-${activeTab}`}" in app_tsx,
          "App.tsx renders #tab-<id> and #panel-<id>")

    for key, sel in UI_SELECTORS.items():
        if key.startswith("tab_"):
            tid = sel.removeprefix("#tab-")
            check(tid in tab_ids_in_app, f"{key} ({sel}) registered in App.tsx TABS")
        elif sel.startswith("#"):
            raw_id = sel[1:]
            found = f'id="{raw_id}"' in all_tsx or f"id='{raw_id}'" in all_tsx
            check(found, f"{key} ({sel}) present as id=\"{raw_id}\" in TSX")
        elif sel.startswith("."):
            cls = sel[1:]
            found = re.search(rf'className="[^"]*\b{re.escape(cls)}\b', all_tsx) is not None or (
                re.search(rf"className=\{{`[^`]*\b{re.escape(cls)}\b", all_tsx) is not None
            )
            check(found, f"{key} ({sel}) present as CSS class in TSX")
        elif sel.startswith("table."):
            cls = sel.removeprefix("table.")
            found = f"<table" in epi_tsx and cls in epi_tsx
            check(found, f"{key} ({sel}) present in EpistasisTab.tsx")
        else:
            # Literal UI text (button label or header substring)
            check(sel in all_tsx, f"{key} (text {sel!r}) present in TSX")


def load_backend_app():
    """Load app/backend/app.py under a private module name.

    ``import app`` is ambiguous inside ``tests.run_all``: ``project/app`` is also a
    namespace package on sys.path, which can shadow ``app/backend/app.py``.
    """
    import importlib.util

    backend_dir = str(ROOT / "app" / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    mod = sys.modules.get("holo_backend_app")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("holo_backend_app", ROOT / "app" / "backend" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["holo_backend_app"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_backend_schemas_match_ui_and_suite() -> None:
    section("FastAPI request schemas match frontend & a100_unified_suite payloads")
    backend_app = load_backend_app()
    ddg_fields = set(backend_app.DdgRequest.model_fields.keys())
    scan_fields = set(backend_app.ScanRequest.model_fields.keys())
    epi_fields = set(backend_app.EpistasisRequest.model_fields.keys())
    cmp_fields = set(backend_app.CompareRequest.model_fields.keys())

    check({"wt_sequence", "mutation", "compare_modes", "structure_pdb"} <= ddg_fields,
          f"DdgRequest has wt_sequence/mutation/compare_modes/structure_pdb ({sorted(ddg_fields)})")
    check("sequence" not in ddg_fields,
          "DdgRequest uses wt_sequence (NOT sequence) — catches Bug A")
    check({"sequence", "start", "end"} <= scan_fields, "ScanRequest has sequence/start/end")
    check({"wt_sequence", "pairs"} <= epi_fields, "EpistasisRequest has wt_sequence/pairs")
    check({"uniprot_id"} <= cmp_fields, "CompareRequest has uniprot_id")

    # Verify a100_unified_suite.py sends wt_sequence to /api/ddg, never {"sequence": ...}
    suite_src = (ROOT / "remote" / "gpu_jobs" / "a100_unified_suite.py").read_text(encoding="utf-8")
    check('"/api/ddg", {"sequence":' not in suite_src,
          "a100_unified_suite.py does NOT send wrong key `sequence` to /api/ddg")
    check('"/api/ddg", {"wt_sequence":' in suite_src,
          "a100_unified_suite.py sends `wt_sequence` to /api/ddg")


def test_ensure_frontend_built_unit() -> None:
    section("ensure_frontend_built handles already-built and missing-package cleanly")
    from remote.gpu_jobs.a100_ui_capture import ensure_frontend_built

    with tempfile.TemporaryDirectory(prefix="holo_ui_test_") as td:
        root = Path(td)
        # Case 1: dist/index.html already exists -> returns immediately without invoking npm
        idx = root / "app" / "frontend" / "dist" / "index.html"
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text("<html><body>ok</body></html>", encoding="utf-8")
        res = ensure_frontend_built(root, log=lambda _: None)
        check(res["built"] is False and res["status"] == "already_built",
              "already-built dist/index.html returns without running npm")

        # Case 2: a dist OLDER than a frontend source is stale -> rebuild is attempted; with no
        # package.json that attempt fails loudly instead of screenshotting the old bundle
        src_file = root / "app" / "frontend" / "src" / "App.tsx"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text("export {}", encoding="utf-8")
        old = time.time() - 3600
        os.utime(idx, (old, old))
        raised = False
        try:
            ensure_frontend_built(root, log=lambda _: None)
        except RuntimeError:
            raised = True
        check(raised, "a dist older than its sources is treated as stale (rebuild attempted)")
        res = ensure_frontend_built(root, log=lambda _: None, rebuild_if_stale=False)
        check(res["status"] == "already_built", "rebuild_if_stale=False keeps an existing dist")

        # Case 3: missing dist and missing package.json -> raises RuntimeError clearly
        idx.unlink()
        raised = False
        try:
            ensure_frontend_built(root, log=lambda _: None)
        except RuntimeError:
            raised = True
        check(raised, "missing dist + package.json raises RuntimeError instead of silently succeeding")


def test_capture_covers_seven_shots() -> None:
    section("capture_ui covers ui_01..ui_07 (native p53 numbering, 3D comparison shot)")
    import inspect

    from remote.gpu_jobs import a100_ui_capture as ui

    names = sorted(ui.SHOT_NAMES.values())
    check(len(names) == 7 and all(n.startswith(f"ui_0{i + 1}_") for i, n in enumerate(names)),
          f"seven screenshots ui_01..ui_07 ({names})")
    check("ui_07_compare_modes_3d.png" in names, "ui_07 is the structure-free vs SE(3)-invariant 3D comparison")
    params = inspect.signature(ui.capture_ui).parameters
    for p in ("ubq_pdb_text", "p53_first_residue", "mut_p53_native", "device_scale_factor", "on_shot", "panel_dir"):
        check(p in params, f"capture_ui accepts {p}")
    check(params["device_scale_factor"].default == 2, "screenshots default to 2x DPI")
    # R175H in native numbering is R82H in the 94-292 construct: the API contract is unchanged
    p53 = __import__("remote.gpu_jobs.l4_ui_recapture", fromlist=["SEQS"]).SEQS["p53_dbd"]
    idx = int(ui.P53_NATIVE_MUTATION[1:-1]) - (ui.P53_DBD_FIRST_RESIDUE - 1)
    check(idx == 82 and p53[idx - 1] == ui.P53_NATIVE_MUTATION[0],
          f"R175H maps to construct index 82 and the wild-type residue there is R (got {p53[idx - 1]})")
    suite_src = (ROOT / "remote" / "gpu_jobs" / "a100_unified_suite.py").read_text(encoding="utf-8")
    check("ubq_pdb_text=ubq_pdb_text" in suite_src, "the A100 suite passes the 1UBQ PDB through for ui_07")


def test_frontend_honesty_and_plot_titles() -> None:
    section("frontend: Plotly 3 title syntax, legacy name, equivariance wording, calibration from the server")
    fe = ROOT / "app" / "frontend"
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted((fe / "src").rglob("*.ts*")))
    tsx_only = "\n".join(p.read_text(encoding="utf-8") for p in sorted((fe / "src").rglob("*.tsx")))
    # plotly.js 3 silently drops these: every axis/colorbar title vanished in the A100 screenshots
    check(re.search(r"\btitle:\s*['`\"]", tsx_only) is None,
          "no Plotly string-shorthand `title: '...'` (plotly.js 3 renders no title for it)")
    check("titlefont" not in tsx_only and "titleside" not in tsx_only,
          "no removed Plotly keys titlefont / titleside")
    html = (fe / "index.html").read_text(encoding="utf-8")
    check("Holo-GNN" not in src and "Holo-GNN" not in html, "the legacy name Holo-GNN is gone from the interface")
    # every mention of SE(3)/E(3) equivariance must be a negation
    bad = []
    for m in re.finditer(r"(?:SE\(3\)|E\(3\))(?:/E\(3\))?[- ]?equivariant", src, flags=re.IGNORECASE):
        before = src[max(0, m.start() - 40):m.start()].lower()
        if "not" not in before:
            bad.append(src[max(0, m.start() - 40):m.end()])
    check(not bad, f"SE(3)/E(3)-equivariance is only ever negated in the UI ({bad[:2]})")
    check("well_calibrated" in src and "tolerances" in src,
          "the calibration status is the server's own verdict and tolerances, not a client-side rule")
    app_tsx = (fe / "src" / "App.tsx").read_text(encoding="utf-8")
    check("warn-banner" not in app_tsx and "EpistemicStrip" in app_tsx,
          "the two global warning banners are replaced by the epistemic status strip")


def test_l4_recapture_driver() -> None:
    section("l4_ui_recapture: compiles, fails fast without the release bundle, finds 1UBQ in the tarball")
    import io
    import tarfile

    mod_path = ROOT / "remote" / "gpu_jobs" / "l4_ui_recapture.py"
    py_compile.compile(str(mod_path), doraise=True)
    from remote.gpu_jobs import l4_ui_recapture as l4

    with tempfile.TemporaryDirectory(prefix="holo_l4_test_") as td:
        root = Path(td)
        raised = False
        try:
            l4.preflight(root / "missing_bundle")
        except FileNotFoundError:
            raised = True
        check(raised, "a missing bundle directory fails fast")
        bundle = root / "bundle"
        bundle.mkdir()
        (bundle / l4.LEAN).write_bytes(b"x")
        raised = False
        try:
            l4.preflight(bundle)
        except FileNotFoundError:
            raised = True
        check(raised, "a bundle without calibration.json fails fast")
        (bundle / l4.CALIB).write_text("{}", encoding="utf-8")
        pre = l4.preflight(bundle)
        check(pre["present"][l4.LEAN] and not pre["present"][l4.CKPT_3D],
              "preflight records which optional checkpoints exist")

        drive = root / "drive"
        (drive / "pdbs").mkdir(parents=True)
        payload = b"ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 96.00           N\nEND\n"
        with tarfile.open(drive / "pdbs" / "megascale_pdbs.tar.gz", "w:gz") as tf:
            info = tarfile.TarInfo("megascale/AlphaFold_model_PDBs/1UBQ.pdb")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        work = root / "work"
        work.mkdir()
        text, source = l4.find_ubq_pdb(None, drive, work)
        check(text is not None and "MET A   1" in text and "1UBQ.pdb" in source,
              f"1UBQ.pdb is extracted from the MegaScale tarball on Drive ({source})")


TESTS = [
    test_module_compiles_and_exports,
    test_selectors_exist_in_tsx,
    test_backend_schemas_match_ui_and_suite,
    test_ensure_frontend_built_unit,
    test_capture_covers_seven_shots,
    test_frontend_honesty_and_plot_titles,
    test_l4_recapture_driver,
]

if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
