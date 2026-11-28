"""
remote/gpu_jobs/a100_ui_capture.py
==================================
Frontend build helper and high-DPI headless-Chromium Playwright capture for the
Holo-GeoN v2 React Web GUI (used by ``stage_block3_demo`` in
``remote/gpu_jobs/a100_unified_suite.py`` and by ``remote/gpu_jobs/l4_ui_recapture.py``).

Why this lives in its own module
--------------------------------
Keeping the Playwright capture logic and verified DOM selectors isolated makes
it directly testable via static TSX selector checks (``tests/test_a100_ui_capture.py``)
without pulling in GPU dependencies.

Screenshots
-----------
* ``ui_01_ddg_ubq_L8P.png``            ubiquitin L8P, structure-free
* ``ui_02_ddg_p53_R175H.png``          TP53 DBD construct typed in native numbering (first residue 94)
* ``ui_03_scan_landscape_ubq.png``     full 1-76 ubiquitin landscape, fitted to the card
* ``ui_04_alphafold_compare_plddt.png`` AlphaFold compare, ribbon by pLDDT
* ``ui_05_alphafold_compare_ddg.png``   AlphaFold compare, ribbon by min ddG
* ``ui_06_epistasis_gb1.png``          GB1 double mutants (epistasis checkpoint)
* ``ui_07_compare_modes_3d.png``       structure-free vs SE(3)-invariant 3D side by side (needs a PDB)

Public API
----------
* ``ensure_frontend_built(repo_root: Path, log=..., rebuild_if_stale=True) -> dict``
* ``capture_ui(base_url, epi_base_url, out_dir, seqs, log=..., ...) -> dict``
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Verified DOM selectors (cross-checked against app/frontend/src/**/*.tsx by
# tests/test_a100_ui_capture.py so any frontend rename breaks a unit test first)
# ---------------------------------------------------------------------------
UI_SELECTORS: dict[str, str] = {
    # App.tsx navigation tabs, header capability pills & epistemic status strip
    "tab_ddg": "#tab-ddg",
    "tab_scan": "#tab-scan",
    "tab_epistasis": "#tab-epistasis",
    "tab_compare": "#tab-compare",
    "header_caps": ".header-cap-pills",
    "status_strip": ".epistemic-strip",
    # DdgTab.tsx
    "ddg_seq": "#ddg-seq",
    "ddg_mut": "#ddg-mut",
    "ddg_numbering": "#ddg-numbering",
    "ddg_compare_modes": "#ddg-compare-modes",
    "ddg_pdb": "#ddg-pdb",
    "ddg_result_value": ".result-value",
    "ddg_result_block": ".result-block",
    "ddg_mode_comparison": ".mode-comparison",
    "ddg_export_json_text": "Export JSON",
    # ScanTab.tsx
    "scan_seq": "#scan-seq",
    "scan_start": "#scan-start",
    "scan_end": "#scan-end",
    "scan_numbering": "#scan-numbering",
    "scan_heatmap_box": ".heatmap-box",
    "scan_submit_text": "Run Scan",
    "scan_export_csv_text": "Export CSV",
    # CompareTab.tsx
    "af_uniprot": "#af-uniprot",
    "af_submit_text": "Compare",
    "af_viewer_container": ".viewer-container",
    "af_plddt_header": "colored by pLDDT",
    "af_ddg_toggle_text": "ΔΔG stability",
    "af_ddg_header": "colored by min ΔΔG stability",
    # EpistasisTab.tsx
    "epi_seq": "#epi-seq",
    "epi_pairs": "#epi-pairs",
    "epi_table": "table.epistasis-table",
    "epi_export_json_text": "Export JSON",
    "epi_refused": ".capability-refused",
    # Common Plotly container / error message
    "plot_wrap": ".plot-wrap",
    "error_msg": ".error-msg",
}

SHOT_NAMES: dict[str, str] = {
    "ddg_ubq": "ui_01_ddg_ubq_L8P.png",
    "ddg_p53": "ui_02_ddg_p53_R175H.png",
    "scan_ubq": "ui_03_scan_landscape_ubq.png",
    "af_plddt": "ui_04_alphafold_compare_plddt.png",
    "af_ddg": "ui_05_alphafold_compare_ddg.png",
    "epistasis": "ui_06_epistasis_gb1.png",
    "compare_modes": "ui_07_compare_modes_3d.png",
}

CHROMIUM_WEBGL_ARGS: list[str] = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--font-render-hinting=none",
]

DEFAULT_GB1_EPISTASIS_PAIRS: list[str] = [
    "V39A+D40A",
    "V39A+G41A",
    "D40A+G41A",
    "G41A+V54A",
    "V39A+V54A",
    "L5A+V54A",
]

#: TP53 DNA-binding-domain construct (UniProt P04637 residues 94-292): R175H is the
#: canonical hotspot, i.e. sequence index 175 - 93 = 82 of the construct.
P53_DBD_FIRST_RESIDUE = 94
P53_NATIVE_MUTATION = "R175H"


def _default_log(msg: str) -> None:
    print(f"[ui {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_shell(cmd: list[str], cwd: Path, timeout: int = 300) -> tuple[int, float, str]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        rc = proc.returncode
        tail = (proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-1500:]
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        tail = f"TIMEOUT after {timeout}s\n{out[-1000:]}\n{err[-1000:]}"
    return rc, round(time.time() - t0, 2), tail


def _newest_source_mtime(fe_dir: Path) -> float:
    """Newest modification time among the inputs of `npm run build` (0.0 when there are none)."""
    newest = 0.0
    candidates = [fe_dir / "index.html", fe_dir / "package.json", fe_dir / "package-lock.json",
                  fe_dir / "vite.config.ts", fe_dir / "tsconfig.json"]
    src = fe_dir / "src"
    if src.is_dir():
        candidates.extend(p for p in src.rglob("*") if p.is_file())
    for p in candidates:
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
    return newest


def ensure_frontend_built(
    repo_root: Path,
    log: Callable[[str], None] = _default_log,
    timeout: int = 300,
    rebuild_if_stale: bool = True,
) -> dict:
    """
    Ensure ``<repo_root>/app/frontend/dist/index.html`` exists and is current before starting uvicorn.

    On a fresh Colab VM ``dist/`` is gitignored and does not exist yet. On a reused VM it
    may predate a ``git pull``; with ``rebuild_if_stale`` a ``dist/index.html`` older than any
    frontend source is rebuilt, so a recapture never screenshots a stale bundle. Runs
    ``npm ci --no-audit --no-fund`` (falling back to ``npm install``) only when
    ``node_modules`` is missing, then ``npm run build``; raises ``RuntimeError`` if
    ``dist/index.html`` is still absent.
    """
    repo_root = Path(repo_root)
    fe_dir = repo_root / "app" / "frontend"
    dist_index = fe_dir / "dist" / "index.html"
    if dist_index.is_file():
        stale = rebuild_if_stale and _newest_source_mtime(fe_dir) > dist_index.stat().st_mtime
        if not stale:
            return {"built": False, "dist_index": str(dist_index), "status": "already_built"}
        log(f"Frontend dist is older than its sources; rebuilding {dist_index} ...")

    if not (fe_dir / "package.json").is_file():
        raise RuntimeError(f"Frontend package.json missing at {fe_dir / 'package.json'}")
    npm_bin = shutil.which("npm")
    if not npm_bin:
        raise RuntimeError(f"{dist_index} is missing or stale and `npm` was not found in PATH")

    rc_i, dt_i, tail_i, install_cmd = 0, 0.0, "", "skipped (node_modules present)"
    if not (fe_dir / "node_modules").is_dir():
        log("Installing frontend dependencies (npm ci) ...")
        rc_i, dt_i, tail_i = _run_shell([npm_bin, "ci", "--no-audit", "--no-fund"], cwd=fe_dir, timeout=timeout)
        install_cmd = "npm ci"
        if rc_i != 0:
            log(f"npm ci failed (rc={rc_i}, {dt_i}s); falling back to npm install ...")
            rc_i, dt_i, tail_i = _run_shell([npm_bin, "install", "--no-audit", "--no-fund"], cwd=fe_dir, timeout=timeout)
            install_cmd = "npm install"

    rc_b, dt_b, tail_b = _run_shell([npm_bin, "run", "build"], cwd=fe_dir, timeout=timeout)
    info = {
        "built": True,
        "install_cmd": install_cmd,
        "install_rc": rc_i,
        "install_s": dt_i,
        "build_rc": rc_b,
        "build_s": dt_b,
        "dist_index": str(dist_index),
    }
    if rc_b != 0 or not dist_index.is_file():
        raise RuntimeError(
            f"Frontend build failed to produce {dist_index} "
            f"(install_rc={rc_i}, build_rc={rc_b}):\n{(tail_b or tail_i)[-1000:]}"
        )
    log(f"Frontend built successfully in {dt_i + dt_b:.1f}s -> {dist_index}")
    return info


def _ensure_playwright(log: Callable[[str], None]) -> tuple[object | None, str | None]:
    """Lazily import playwright.sync_api; install package + Chromium only if missing."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        return sync_playwright, None
    except Exception:
        pass

    log("Playwright not installed; attempting lazy install + chromium download ...")
    if shutil.which("uv"):
        rc_p, _, tail_p = _run_shell(
            ["uv", "pip", "install", "--system", "--quiet", "playwright"], cwd=Path.cwd(), timeout=180
        )
    else:
        rc_p, _, tail_p = _run_shell(
            [sys.executable, "-m", "pip", "install", "-q", "playwright"], cwd=Path.cwd(), timeout=180
        )
    if rc_p != 0:
        return None, f"playwright pip install failed (rc={rc_p}): {tail_p[-500:]}"

    rc_b, _, tail_b = _run_shell(
        [sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"],
        cwd=Path.cwd(), timeout=300,
    )
    if rc_b != 0:
        rc_b, _, tail_b = _run_shell(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            cwd=Path.cwd(), timeout=300,
        )
    if rc_b != 0:
        return None, f"playwright chromium install failed (rc={rc_b}): {tail_b[-500:]}"

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        return sync_playwright, None
    except Exception as exc:
        return None, f"playwright import failed after install: {type(exc).__name__}: {exc}"


def _wait_plotly_ready(page, min_plots: int = 1, timeout_ms: int = 30_000) -> None:
    """Wait until at least ``min_plots`` Plotly SVG canvases are rendered in the DOM."""
    page.wait_for_function(
        f"() => document.querySelectorAll('.js-plotly-plot .main-svg').length >= {int(min_plots)}",
        timeout=timeout_ms,
    )


def _settle(page, ms: int = 350) -> None:
    """Park the pointer (no hover modebar / tooltips in the shot) and let layout settle."""
    try:
        page.mouse.move(2, 2)
    except Exception:
        pass
    try:  # bundled webfonts must be in before Plotly measures and the shot is taken
        page.evaluate("() => document.fonts ? document.fonts.ready.then(() => true) : true")
    except Exception:
        pass
    page.wait_for_timeout(ms)


def _attach_error_listeners(page, console_errors: list[dict], page_errors: list[str], tag: str) -> None:
    def _on_console(msg) -> None:
        try:
            mtype = msg.type
            if mtype in ("error", "warning"):
                console_errors.append({"page": tag, "type": mtype, "text": str(msg.text)[:600]})
        except Exception:
            pass

    def _on_pageerror(exc) -> None:
        page_errors.append(f"[{tag}] {exc}")

    page.on("console", _on_console)
    page.on("pageerror", _on_pageerror)


#: JS probe for ui_03: does the landscape fit its card, and which residues are labelled?
_HEATMAP_PROBE_JS = """() => {
  const box = document.querySelector('.heatmap-box');
  if (!box) return null;
  const ticks = Array.from(box.querySelectorAll('.xtick text')).map(t => (t.textContent || '').trim()).filter(Boolean);
  return {
    box_client_width: box.clientWidth,
    box_scroll_width: box.scrollWidth,
    fits_without_scroll: box.scrollWidth <= box.clientWidth + 1,
    x_tick_count: ticks.length,
    first_tick: ticks[0] || null,
    last_tick: ticks[ticks.length - 1] || null,
  };
}"""


def capture_ui(
    base_url: str | None,
    epi_base_url: str | None,
    out_dir: Path,
    seqs: dict[str, str],
    log: Callable[[str], None] = _default_log,
    af_uniprot_candidates: tuple[str, ...] = ("P04637", "P69905"),
    mut_p53_dbd: str = "R82H",
    gb1_pairs: list[str] | None = None,
    p53_first_residue: int = P53_DBD_FIRST_RESIDUE,
    mut_p53_native: str = P53_NATIVE_MUTATION,
    ubq_pdb_text: str | None = None,
    device_scale_factor: int = 2,
    viewport: tuple[int, int] = (1440, 960),
    panel_dir: Path | None = None,
    on_shot: Callable[[Path], None] | None = None,
) -> dict:
    """
    Capture high-DPI (``device_scale_factor`` x, 1440x960 viewport, full_page=True) headless
    Chromium screenshots of the live Holo-GeoN v2 React GUI with SwiftShader WebGL for 3Dmol.

    Pass both ``base_url`` and ``epi_base_url`` to capture all shots in one browser session,
    or only one of them (the other ``None``) when the main (8011) and epistasis (8012)
    uvicorn servers are started one at a time to keep GPU memory minimal.

    ``ui_02`` types ``mut_p53_native`` (R175H) with the first-residue field set to
    ``p53_first_residue`` (94); the interface sends sequence index ``mut_p53_dbd`` (R82H) to
    the API, whose contract is unchanged. ``ui_07`` needs ``ubq_pdb_text`` and the 3D mode;
    without either it is recorded as skipped with the reason.

    Every screenshot is independently guarded so one failure never aborts the rest.
    ``panel_dir`` additionally saves tight element crops of the key figure panels;
    ``on_shot(path)`` is called after every saved file (e.g. to mirror it to Drive at once).
    Returns a structured report suitable for embedding in ``live_api_verification.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if panel_dir is not None:
        panel_dir = Path(panel_dir)
        panel_dir.mkdir(parents=True, exist_ok=True)
    main_url = base_url.rstrip("/") if base_url else None
    epi_url = epi_base_url.rstrip("/") if epi_base_url else None

    ubq = seqs["ubq"]
    p53 = seqs["p53_dbd"]
    gb1 = seqs["gb1"]
    pairs = gb1_pairs or DEFAULT_GB1_EPISTASIS_PAIRS

    result: dict = {
        "screenshots": [],
        "panels": [],
        "shot_details": {},
        "failures": {},
        "skipped": {},
        "console_errors": [],
        "page_errors": [],
        "device_scale_factor": device_scale_factor,
        "viewport": list(viewport),
    }
    if not main_url and not epi_url:
        return result

    sync_playwright, pw_err = _ensure_playwright(log)
    if sync_playwright is None:
        result["playwright_unavailable"] = pw_err or "playwright could not be imported"
        log(f"WARNING: UI capture unavailable: {result['playwright_unavailable']}")
        return result

    def _save(page, name: str, details: dict) -> None:
        path = out_dir / name
        page.screenshot(path=str(path), full_page=True, animations="disabled", caret="hide")
        result["screenshots"].append(name)
        result["shot_details"][name] = {"status": "ok", **details}
        log(f"Captured {name}")
        if on_shot is not None:
            try:
                on_shot(path)
            except Exception as exc:  # mirroring is best-effort; the local file exists
                log(f"WARNING: on_shot({name}) failed: {exc}")

    def _panel(page, selector: str, name: str) -> None:
        if panel_dir is None:
            return
        try:
            loc = page.locator(selector).first
            loc.scroll_into_view_if_needed(timeout=5_000)
            path = panel_dir / name
            loc.screenshot(path=str(path), animations="disabled", caret="hide")
            result["panels"].append(name)
            if on_shot is not None:
                on_shot(path)
        except Exception as exc:
            result["failures"][f"panel:{name}"] = f"{type(exc).__name__}: {exc}"

    def _new_context(browser):
        return browser.new_context(
            viewport={"width": int(viewport[0]), "height": int(viewport[1])},
            device_scale_factor=device_scale_factor,
            color_scheme="light",
            reduced_motion="reduce",
        )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=CHROMIUM_WEBGL_ARGS)
            try:
                if main_url:
                    ctx = _new_context(browser)
                    try:
                        page = ctx.new_page()
                        _attach_error_listeners(page, result["console_errors"], result["page_errors"], "main")

                        page.goto(f"{main_url}/", wait_until="networkidle", timeout=60_000)
                        page.wait_for_selector(UI_SELECTORS["ddg_seq"], state="visible", timeout=30_000)
                        try:
                            page.wait_for_selector(UI_SELECTORS["header_caps"], state="visible", timeout=15_000)
                            page.wait_for_selector(UI_SELECTORS["status_strip"], state="visible", timeout=15_000)
                        except Exception:
                            pass

                        # ----------------------------------------------------------
                        # 1. ui_01 — Ubiquitin L8P live prediction + interval
                        # ----------------------------------------------------------
                        shot1 = SHOT_NAMES["ddg_ubq"]
                        try:
                            page.locator(UI_SELECTORS["ddg_seq"]).fill(ubq)
                            page.locator(UI_SELECTORS["ddg_numbering"]).fill("1")
                            page.locator(UI_SELECTORS["ddg_mut"]).fill("L8P")
                            with page.expect_response(
                                lambda r: "/api/ddg" in r.url and r.request.method == "POST",
                                timeout=120_000,
                            ) as resp_info:
                                page.locator('#panel-ddg button[type="submit"]').click()
                            status1 = resp_info.value.status
                            if status1 != 200:
                                raise RuntimeError(f"/api/ddg returned HTTP {status1}")
                            page.wait_for_selector(UI_SELECTORS["ddg_result_value"], state="visible", timeout=30_000)
                            page.locator(f"#panel-ddg {UI_SELECTORS['ddg_result_block']}").get_by_text("L8P").first.wait_for(
                                state="visible", timeout=15_000)
                            _wait_plotly_ready(page, min_plots=1, timeout_ms=20_000)
                            _settle(page)
                            _save(page, shot1, {"http": status1})
                        except Exception as exc:
                            result["failures"][shot1] = f"{type(exc).__name__}: {exc}"
                            log(f"WARNING: {shot1} failed: {exc}")

                        # ----------------------------------------------------------
                        # 2. ui_02 — TP53 DBD construct, typed as R175H (first residue 94)
                        # ----------------------------------------------------------
                        shot2 = SHOT_NAMES["ddg_p53"]
                        try:
                            page.locator(UI_SELECTORS["tab_ddg"]).click()
                            page.wait_for_selector(UI_SELECTORS["ddg_seq"], state="visible", timeout=15_000)
                            page.locator(UI_SELECTORS["ddg_seq"]).fill(p53)
                            page.locator(UI_SELECTORS["ddg_numbering"]).fill(str(int(p53_first_residue)))
                            page.locator(UI_SELECTORS["ddg_mut"]).fill(mut_p53_native)
                            with page.expect_response(
                                lambda r: "/api/ddg" in r.url and r.request.method == "POST",
                                timeout=120_000,
                            ) as resp_info:
                                page.locator('#panel-ddg button[type="submit"]').click()
                            resp2 = resp_info.value
                            status2 = resp2.status
                            if status2 != 200:
                                raise RuntimeError(f"/api/ddg (p53 {mut_p53_native}) returned HTTP {status2}")
                            sent = None
                            try:
                                sent = (resp2.request.post_data_json or {}).get("mutation")
                            except Exception:
                                pass
                            page.locator(f"#panel-ddg {UI_SELECTORS['ddg_result_block']}").get_by_text(mut_p53_native).first.wait_for(
                                state="visible", timeout=20_000)
                            _wait_plotly_ready(page, min_plots=1, timeout_ms=20_000)
                            _settle(page)
                            _save(page, shot2, {"http": status2, "mutation_native": mut_p53_native,
                                                "first_residue": int(p53_first_residue),
                                                "mutation_sent_to_api": sent,
                                                "expected_api_mutation": mut_p53_dbd})
                        except Exception as exc:
                            result["failures"][shot2] = f"{type(exc).__name__}: {exc}"
                            log(f"WARNING: {shot2} failed: {exc}")

                        # ----------------------------------------------------------
                        # 3. ui_03 — Full ubiquitin 1-76 landscape, fitted to the card
                        # ----------------------------------------------------------
                        shot3 = SHOT_NAMES["scan_ubq"]
                        try:
                            page.locator(UI_SELECTORS["tab_scan"]).click()
                            page.wait_for_selector(UI_SELECTORS["scan_seq"], state="visible", timeout=15_000)
                            # filling #scan-seq resets #scan-end to min(len, 60), so set start/end AFTER it
                            page.locator(UI_SELECTORS["scan_seq"]).fill(ubq)
                            page.locator(UI_SELECTORS["scan_start"]).fill("1")
                            page.locator(UI_SELECTORS["scan_end"]).fill(str(len(ubq)))
                            page.locator(UI_SELECTORS["scan_numbering"]).fill("1")
                            page.locator('#panel-scan button[type="submit"]').click()
                            # Export CSV renders only when the final SSE `result` event lands
                            page.locator("#panel-scan button", has_text=UI_SELECTORS["scan_export_csv_text"]).wait_for(
                                state="visible", timeout=180_000)
                            _wait_plotly_ready(page, min_plots=1, timeout_ms=20_000)
                            _settle(page, 600)
                            probe = page.evaluate(_HEATMAP_PROBE_JS) or {}
                            if probe and not probe.get("fits_without_scroll", True):
                                log(f"WARNING: landscape overflows its card: {probe}")
                            _save(page, shot3, {"positions": f"1-{len(ubq)}", "heatmap": probe})
                            _panel(page, "#panel-scan .card:nth-of-type(2)", "ui_03_landscape_card.png")
                        except Exception as exc:
                            result["failures"][shot3] = f"{type(exc).__name__}: {exc}"
                            log(f"WARNING: {shot3} failed: {exc}")

                        # ----------------------------------------------------------
                        # 4 & 5. AlphaFold compare: ribbon by pLDDT, then by min ddG
                        # ----------------------------------------------------------
                        shot4 = SHOT_NAMES["af_plddt"]
                        shot5 = SHOT_NAMES["af_ddg"]
                        try:
                            page.locator(UI_SELECTORS["tab_compare"]).click()
                            page.wait_for_selector(UI_SELECTORS["af_uniprot"], state="visible", timeout=15_000)
                            used_uid = None
                            last_err = None
                            for uid in af_uniprot_candidates:
                                page.locator(UI_SELECTORS["af_uniprot"]).fill(uid)
                                with page.expect_response(
                                    lambda r: "/api/compare" in r.url and r.request.method == "POST",
                                    timeout=180_000,
                                ) as resp_info:
                                    page.locator('#panel-compare button[type="submit"]').click()
                                st = resp_info.value.status
                                if st == 200:
                                    used_uid = uid
                                    break
                                last_err = f"{uid}: HTTP {st}"
                                log(f"AlphaFold compare failed for {uid} (HTTP {st}); trying next candidate ...")
                            if used_uid is None:
                                raise RuntimeError(f"All AlphaFold candidates failed ({last_err})")

                            page.wait_for_selector(f"{UI_SELECTORS['af_viewer_container']} canvas", state="visible",
                                                   timeout=30_000)
                            page.locator("#panel-compare").get_by_text(UI_SELECTORS["af_plddt_header"]).first.wait_for(
                                state="visible", timeout=20_000)
                            _wait_plotly_ready(page, min_plots=2, timeout_ms=30_000)
                            _settle(page, 900)  # SwiftShader WebGL needs a moment for the first frame
                            _save(page, shot4, {"uniprot_id": used_uid, "color_mode": "plddt"})
                            _panel(page, "#panel-compare .compare-layout", "ui_04_structure_and_pae.png")

                            try:
                                ddg_btn = page.locator("#panel-compare button", has_text=UI_SELECTORS["af_ddg_toggle_text"]).first
                                ddg_btn.wait_for(state="visible", timeout=10_000)
                                ddg_btn.click()
                                page.locator("#panel-compare").get_by_text(UI_SELECTORS["af_ddg_header"]).first.wait_for(
                                    state="visible", timeout=15_000)
                                _settle(page, 900)
                                _save(page, shot5, {"uniprot_id": used_uid, "color_mode": "ddg"})
                                _panel(page, "#panel-compare .compare-layout", "ui_05_structure_and_pae.png")
                            except Exception as exc5:
                                result["failures"][shot5] = f"{type(exc5).__name__}: {exc5}"
                                log(f"WARNING: {shot5} failed: {exc5}")
                        except Exception as exc4:
                            result["failures"][shot4] = f"{type(exc4).__name__}: {exc4}"
                            result["failures"].setdefault(shot5, f"skipped because {shot4} failed: {exc4}")
                            log(f"WARNING: {shot4}/{shot5} failed: {exc4}")

                        # ----------------------------------------------------------
                        # 7. ui_07 — structure-free vs SE(3)-invariant 3D side by side
                        # ----------------------------------------------------------
                        shot7 = SHOT_NAMES["compare_modes"]
                        if not ubq_pdb_text:
                            result["skipped"][shot7] = "no ubiquitin PDB text supplied (ubq_pdb_text)"
                            log(f"Skipping {shot7}: {result['skipped'][shot7]}")
                        else:
                            try:
                                page.locator(UI_SELECTORS["tab_ddg"]).click()
                                page.wait_for_selector(UI_SELECTORS["ddg_seq"], state="visible", timeout=15_000)
                                box = page.locator(UI_SELECTORS["ddg_compare_modes"])
                                if box.is_disabled():
                                    result["skipped"][shot7] = "the server reports the SE(3)-invariant 3D mode unavailable"
                                    log(f"Skipping {shot7}: {result['skipped'][shot7]}")
                                else:
                                    page.locator(UI_SELECTORS["ddg_seq"]).fill(ubq)
                                    page.locator(UI_SELECTORS["ddg_numbering"]).fill("1")
                                    page.locator(UI_SELECTORS["ddg_mut"]).fill("L8P")
                                    box.check()
                                    page.wait_for_selector(UI_SELECTORS["ddg_pdb"], state="visible", timeout=10_000)
                                    page.locator(UI_SELECTORS["ddg_pdb"]).fill(ubq_pdb_text)
                                    with page.expect_response(
                                        lambda r: "/api/ddg" in r.url and r.request.method == "POST",
                                        timeout=180_000,
                                    ) as resp_info:
                                        page.locator('#panel-ddg button[type="submit"]').click()
                                    status7 = resp_info.value.status
                                    if status7 != 200:
                                        detail = ""
                                        try:
                                            detail = str((resp_info.value.json() or {}).get("detail", ""))[:300]
                                        except Exception:
                                            pass
                                        raise RuntimeError(f"/api/ddg compare_modes returned HTTP {status7}: {detail}")
                                    page.wait_for_selector(UI_SELECTORS["ddg_mode_comparison"], state="visible",
                                                           timeout=30_000)
                                    _wait_plotly_ready(page, min_plots=1, timeout_ms=20_000)
                                    _settle(page)
                                    _save(page, shot7, {"http": status7, "pdb_chars": len(ubq_pdb_text)})
                                    _panel(page, f"#panel-ddg {UI_SELECTORS['ddg_mode_comparison']}",
                                           "ui_07_mode_comparison.png")
                            except Exception as exc7:
                                result["failures"][shot7] = f"{type(exc7).__name__}: {exc7}"
                                log(f"WARNING: {shot7} failed: {exc7}")
                    finally:
                        ctx.close()

                # --------------------------------------------------------------
                # 6. ui_06 — GB1 epistasis (epistasis checkpoint, port 8012)
                # --------------------------------------------------------------
                shot6 = SHOT_NAMES["epistasis"]
                if epi_url:
                    ctx2 = _new_context(browser)
                    try:
                        page2 = ctx2.new_page()
                        _attach_error_listeners(page2, result["console_errors"], result["page_errors"], "epistasis")
                        page2.goto(f"{epi_url}/", wait_until="networkidle", timeout=60_000)
                        page2.wait_for_selector(UI_SELECTORS["tab_epistasis"], state="visible", timeout=30_000)
                        try:
                            page2.wait_for_selector(UI_SELECTORS["header_caps"], state="visible", timeout=15_000)
                        except Exception:
                            pass
                        page2.locator(UI_SELECTORS["tab_epistasis"]).click()
                        page2.wait_for_selector(UI_SELECTORS["epi_seq"], state="visible", timeout=15_000)
                        if page2.locator(UI_SELECTORS["epi_refused"]).count() > 0:
                            raise RuntimeError("Epistasis head reported unavailable (HTTP 503 Refusal) on epistasis server")
                        page2.locator(UI_SELECTORS["epi_seq"]).fill(gb1)
                        page2.locator(UI_SELECTORS["epi_pairs"]).fill("\n".join(pairs))
                        page2.locator('#panel-epistasis button[type="submit"]').click()
                        page2.wait_for_selector(f"{UI_SELECTORS['epi_table']} tbody tr", state="visible", timeout=180_000)
                        page2.wait_for_function(
                            "() => { const b = Array.from(document.querySelectorAll('#panel-epistasis button'))"
                            ".find(el => (el.textContent || '').includes('Export JSON')); return Boolean(b && !b.disabled); }",
                            timeout=180_000,
                        )
                        _wait_plotly_ready(page2, min_plots=2, timeout_ms=30_000)
                        _settle(page2, 500)
                        _save(page2, shot6, {"n_pairs": len(pairs)})
                        _panel(page2, "#panel-epistasis .compare-grid", "ui_06_epistasis_plots.png")
                    except Exception as exc6:
                        result["failures"][shot6] = f"{type(exc6).__name__}: {exc6}"
                        log(f"WARNING: {shot6} failed: {exc6}")
                    finally:
                        ctx2.close()
            finally:
                browser.close()
    except Exception as exc:
        result["failures"]["browser_launch"] = f"{type(exc).__name__}: {exc}"
        log(f"WARNING: Playwright browser error: {exc}")

    return result
