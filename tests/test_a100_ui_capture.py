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
    ddg_tsx = (fe_src / "tabs" / "DdgTab.tsx").read_text(encoding="utf-8")
    scan_tsx = (fe_src / "tabs" / "ScanTab.tsx").read_text(encoding="utf-8")
    cmp_tsx = (fe_src / "tabs" / "CompareTab.tsx").read_text(encoding="utf-8")
    epi_tsx = (fe_src / "tabs" / "EpistasisTab.tsx").read_text(encoding="utf-8")
    all_tsx = "\n".join([app_tsx, ddg_tsx, scan_tsx, cmp_tsx, epi_tsx])

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

        # Case 2: missing dist and missing package.json -> raises RuntimeError clearly
        idx.unlink()
        raised = False
        try:
            ensure_frontend_built(root, log=lambda _: None)
        except RuntimeError:
            raised = True
        check(raised, "missing dist + package.json raises RuntimeError instead of silently succeeding")


TESTS = [
    test_module_compiles_and_exports,
    test_selectors_exist_in_tsx,
    test_backend_schemas_match_ui_and_suite,
    test_ensure_frontend_built_unit,
]

if __name__ == "__main__":
    from tests import helpers

    for fn in TESTS:
        fn()
    passed, failed = helpers.results()
    print(f"{passed} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)
