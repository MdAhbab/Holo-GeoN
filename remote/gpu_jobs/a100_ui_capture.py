"""
remote/gpu_jobs/a100_ui_capture.py
==================================
Frontend build helper and high-DPI headless-Chromium Playwright capture for the
Holo-GeoN v2 React Web GUI (used by ``stage_block3_demo`` in
``remote/gpu_jobs/a100_unified_suite.py``).

Why this lives in its own module
--------------------------------
Keeping the Playwright capture logic and verified DOM selectors isolated makes
it directly testable via static TSX selector checks (``tests/test_a100_ui_capture.py``)
without pulling in GPU dependencies.

Public API
----------
* ``ensure_frontend_built(repo_root: Path, log=...) -> dict``
* ``capture_ui(base_url, epi_base_url, out_dir, seqs, log=..., ...) -> dict``
"""
from __future__ import annotations

import os
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
    # App.tsx navigation tabs & header capability pills
    "tab_ddg": "#tab-ddg",
    "tab_scan": "#tab-scan",
    "tab_epistasis": "#tab-epistasis",
    "tab_compare": "#tab-compare",
    "header_caps": ".header-cap-pills",
    # DdgTab.tsx
    "ddg_seq": "#ddg-seq",
    "ddg_mut": "#ddg-mut",
    "ddg_result_value": ".result-value",
    "ddg_export_json_text": "Export JSON",
    # ScanTab.tsx
    "scan_seq": "#scan-seq",
    "scan_start": "#scan-start",
    "scan_end": "#scan-end",
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

CHROMIUM_WEBGL_ARGS: list[str] = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
]

DEFAULT_GB1_EPISTASIS_PAIRS: list[str] = [
    "V39A+D40A",
    "V39A+G41A",
    "D40A+G41A",
    "G41A+V54A",
    "V39A+V54A",
    "L5A+V54A",
]


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


def ensure_frontend_built(
    repo_root: Path,
    log: Callable[[str], None] = _default_log,
    timeout: int = 300,
) -> dict:
    """
    Ensure ``<repo_root>/app/frontend/dist/index.html`` exists before starting uvicorn.

    On a fresh Colab/A100 VM, ``dist/`` is gitignored and does not exist yet.
    When missing, runs ``npm ci --no-audit --no-fund`` (falling back to
    ``npm install --no-audit --no-fund``) followed by ``npm run build``, and
    raises ``RuntimeError`` if ``dist/index.html`` is still absent.
    """
    repo_root = Path(repo_root)
    fe_dir = repo_root / "app" / "frontend"
    dist_index = fe_dir / "dist" / "index.html"
    if dist_index.is_file():
        return {"built": False, "dist_index": str(dist_index), "status": "already_built"}

    if not (fe_dir / "package.json").is_file():
        raise RuntimeError(f"Frontend package.json missing at {fe_dir / 'package.json'}")
    npm_bin = shutil.which("npm")
    if not npm_bin:
        raise RuntimeError(f"{dist_index} is missing and `npm` was not found in PATH")

    log(f"Frontend dist missing ({dist_index}); running npm ci && npm run build ...")
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
    if not dist_index.is_file():
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
    # Prefer uv if available (fast on Colab), else pip
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

    # Try installing system deps first (Ubuntu/Colab), fall back to browser-only
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


def capture_ui(
    base_url: str | None,
    epi_base_url: str | None,
    out_dir: Path,
    seqs: dict[str, str],
    log: Callable[[str], None] = _default_log,
    af_uniprot_candidates: tuple[str, ...] = ("P04637", "P69905"),
    mut_p53_dbd: str = "R82H",
    gb1_pairs: list[str] | None = None,
) -> dict:
    """
    Capture high-DPI (2x, 1440x960, full_page=True) headless Chromium screenshots
    of the live Holo-GeoN v2 React GUI with SwiftShader WebGL enabled for 3Dmol.

    Pass both ``base_url`` and ``epi_base_url`` to capture all 6 screenshots in
    one browser session, or pass only one of them (the other ``None``) to run
    sequentially when the main (8011) and epistasis (8012) uvicorn servers are
    started one at a time to keep GPU VRAM minimal.

    Every screenshot is independently guarded so a single timeout/failure never
    aborts the remaining captures. Returns a structured report suitable for
    embedding directly in ``live_api_verification.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_url = base_url.rstrip("/") if base_url else None
    epi_url = epi_base_url.rstrip("/") if epi_base_url else None

    ubq = seqs["ubq"]
    p53 = seqs["p53_dbd"]
    gb1 = seqs["gb1"]
    pairs = gb1_pairs or DEFAULT_GB1_EPISTASIS_PAIRS

    result: dict = {
        "screenshots": [],
        "shot_details": {},
        "failures": {},
        "console_errors": [],
        "page_errors": [],
    }
    if not main_url and not epi_url:
        return result

    sync_playwright, pw_err = _ensure_playwright(log)
    if sync_playwright is None:
        result["playwright_unavailable"] = pw_err or "playwright could not be imported"
        log(f"WARNING: UI capture unavailable: {result['playwright_unavailable']}")
        return result

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=CHROMIUM_WEBGL_ARGS)
            try:
                if main_url:
                    ctx = browser.new_context(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
                    try:
                        page = ctx.new_page()
                        _attach_error_listeners(page, result["console_errors"], result["page_errors"], "main")

                        page.goto(f"{main_url}/", wait_until="networkidle", timeout=60_000)
                        # Wait for React root & /api/health capability pills
                        page.wait_for_selector(UI_SELECTORS["ddg_seq"], state="visible", timeout=30_000)
                        try:
                            page.wait_for_selector(UI_SELECTORS["header_caps"], state="visible", timeout=15_000)
                        except Exception:
                            pass

                        # --------------------------------------------------------------
                        # 1. ui_01_ddg_ubq_L8P.png — Ubiquitin L8P live prediction + CI
                        # --------------------------------------------------------------
                        shot1 = "ui_01_ddg_ubq_L8P.png"
                        try:
                            page.locator(UI_SELECTORS["ddg_seq"]).fill(ubq)
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
                            page.locator("#panel-ddg").get_by_text("L8P").first.wait_for(state="visible", timeout=15_000)
                            _wait_plotly_ready(page, min_plots=1, timeout_ms=20_000)
                            page.wait_for_timeout(300)
                            page.screenshot(path=str(out_dir / shot1), full_page=True)
                            result["screenshots"].append(shot1)
                            result["shot_details"][shot1] = {"status": "ok", "http": status1}
                            log(f"Captured {shot1}")
                        except Exception as exc:
                            result["failures"][shot1] = f"{type(exc).__name__}: {exc}"
                            log(f"WARNING: {shot1} failed: {exc}")

                        # --------------------------------------------------------------
                        # 2. ui_02_ddg_p53_R175H.png — Human TP53 DBD R82H (= R175H)
                        # --------------------------------------------------------------
                        shot2 = "ui_02_ddg_p53_R175H.png"
                        try:
                            page.locator(UI_SELECTORS["tab_ddg"]).click()
                            page.wait_for_selector(UI_SELECTORS["ddg_seq"], state="visible", timeout=15_000)
                            page.locator(UI_SELECTORS["ddg_seq"]).fill(p53)
                            page.locator(UI_SELECTORS["ddg_mut"]).fill(mut_p53_dbd)
                            with page.expect_response(
                                lambda r: "/api/ddg" in r.url and r.request.method == "POST",
                                timeout=120_000,
                            ) as resp_info:
                                page.locator('#panel-ddg button[type="submit"]').click()
                            status2 = resp_info.value.status
                            if status2 != 200:
                                raise RuntimeError(f"/api/ddg (p53 {mut_p53_dbd}) returned HTTP {status2}")
                            page.locator("#panel-ddg .result-item").get_by_text(mut_p53_dbd).first.wait_for(
                                state="visible", timeout=20_000
                            )
                            _wait_plotly_ready(page, min_plots=1, timeout_ms=20_000)
                            page.wait_for_timeout(300)
                            page.screenshot(path=str(out_dir / shot2), full_page=True)
                            result["screenshots"].append(shot2)
                            result["shot_details"][shot2] = {"status": "ok", "http": status2, "mutation": mut_p53_dbd}
                            log(f"Captured {shot2}")
                        except Exception as exc:
                            result["failures"][shot2] = f"{type(exc).__name__}: {exc}"
                            log(f"WARNING: {shot2} failed: {exc}")

                        # --------------------------------------------------------------
                        # 3. ui_03_scan_landscape_ubq.png — Full Ubiquitin 1-76 scan
                        # --------------------------------------------------------------
                        shot3 = "ui_03_scan_landscape_ubq.png"
                        try:
                            page.locator(UI_SELECTORS["tab_scan"]).click()
                            page.wait_for_selector(UI_SELECTORS["scan_seq"], state="visible", timeout=15_000)
                            # Note: filling #scan-seq resets #scan-end to min(len, 60) via onSeqChange,
                            # so fill #scan-start and #scan-end AFTER #scan-seq to capture full 1-76!
                            page.locator(UI_SELECTORS["scan_seq"]).fill(ubq)
                            page.locator(UI_SELECTORS["scan_start"]).fill("1")
                            page.locator(UI_SELECTORS["scan_end"]).fill(str(len(ubq)))
                            page.locator('#panel-scan button[type="submit"]').click()
                            # Export CSV button renders only when the final SSE `result` event lands
                            page.locator("#panel-scan button", has_text=UI_SELECTORS["scan_export_csv_text"]).wait_for(
                                state="visible", timeout=180_000
                            )
                            _wait_plotly_ready(page, min_plots=1, timeout_ms=20_000)
                            page.wait_for_timeout(400)
                            page.screenshot(path=str(out_dir / shot3), full_page=True)
                            result["screenshots"].append(shot3)
                            result["shot_details"][shot3] = {"status": "ok", "positions": f"1-{len(ubq)}"}
                            log(f"Captured {shot3}")
                        except Exception as exc:
                            result["failures"][shot3] = f"{type(exc).__name__}: {exc}"
                            log(f"WARNING: {shot3} failed: {exc}")

                        # --------------------------------------------------------------
                        # 4 & 5. ui_04_alphafold_compare_plddt.png + ui_05_alphafold_compare_ddg.png
                        # --------------------------------------------------------------
                        shot4 = "ui_04_alphafold_compare_plddt.png"
                        shot5 = "ui_05_alphafold_compare_ddg.png"
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

                            # Wait for 3Dmol WebGL canvas + both Plotly charts (PAE + overlay)
                            page.wait_for_selector(f"{UI_SELECTORS['af_viewer_container']} canvas", state="visible", timeout=30_000)
                            page.locator("#panel-compare").get_by_text(UI_SELECTORS["af_plddt_header"]).first.wait_for(
                                state="visible", timeout=20_000
                            )
                            _wait_plotly_ready(page, min_plots=2, timeout_ms=30_000)
                            page.wait_for_timeout(500)
                            page.screenshot(path=str(out_dir / shot4), full_page=True)
                            result["screenshots"].append(shot4)
                            result["shot_details"][shot4] = {"status": "ok", "uniprot_id": used_uid, "color_mode": "plddt"}
                            log(f"Captured {shot4} ({used_uid})")

                            # Toggle 3Dmol ribbon coloring from pLDDT -> ΔΔG stability
                            try:
                                ddg_btn = page.locator("#panel-compare button", has_text=UI_SELECTORS["af_ddg_toggle_text"]).first
                                ddg_btn.wait_for(state="visible", timeout=10_000)
                                ddg_btn.click()
                                page.locator("#panel-compare").get_by_text(UI_SELECTORS["af_ddg_header"]).first.wait_for(
                                    state="visible", timeout=15_000
                                )
                                page.wait_for_timeout(500)
                                page.screenshot(path=str(out_dir / shot5), full_page=True)
                                result["screenshots"].append(shot5)
                                result["shot_details"][shot5] = {"status": "ok", "uniprot_id": used_uid, "color_mode": "ddg"}
                                log(f"Captured {shot5} ({used_uid})")
                            except Exception as exc5:
                                result["failures"][shot5] = f"{type(exc5).__name__}: {exc5}"
                                log(f"WARNING: {shot5} failed: {exc5}")
                        except Exception as exc4:
                            result["failures"][shot4] = f"{type(exc4).__name__}: {exc4}"
                            result["failures"].setdefault(shot5, f"skipped because {shot4} failed: {exc4}")
                            log(f"WARNING: {shot4}/{shot5} failed: {exc4}")
                    finally:
                        ctx.close()

                # --------------------------------------------------------------
                # 6. ui_06_epistasis_gb1.png (on epistasis server, port 8012)
                # --------------------------------------------------------------
                shot6 = "ui_06_epistasis_gb1.png"
                if epi_url:
                    ctx2 = browser.new_context(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
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
                        # Wait for table rows, enabled Export JSON (only enabled when final result arrives), and both Plotly charts
                        page2.wait_for_selector(f"{UI_SELECTORS['epi_table']} tbody tr", state="visible", timeout=180_000)
                        page2.wait_for_function(
                            "() => { const b = Array.from(document.querySelectorAll('#panel-epistasis button'))"
                            ".find(el => (el.textContent || '').includes('Export JSON')); return Boolean(b && !b.disabled); }",
                            timeout=180_000,
                        )
                        _wait_plotly_ready(page2, min_plots=2, timeout_ms=30_000)
                        page2.wait_for_timeout(400)
                        page2.screenshot(path=str(out_dir / shot6), full_page=True)
                        result["screenshots"].append(shot6)
                        result["shot_details"][shot6] = {"status": "ok", "n_pairs": len(pairs)}
                        log(f"Captured {shot6}")
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
