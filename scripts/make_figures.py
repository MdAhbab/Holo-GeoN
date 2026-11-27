"""
scripts/make_figures.py
=======================
Build every paper figure and table from released artifacts only.

    python -m scripts.make_figures \
        --release ../extras/zenodo_staging/Holo-StabBench-v1.0 \
        --results <results dir>  --out ../paper  [--only F2,T1]  [--primary-split id30]

Figures go to <out>/figures (PDF, PNG, a `_data.csv` table twin and a draft
`_caption.txt` carrying the numbers), tables to <out>/tables (CSV + booktabs .tex), and
<out>/figure_manifest.json records, for every output, the SHA-256 of each input file
plus the code commit and plotting stack. Nothing is read except the release and the
results folder, so every number in the paper traces to a released file and rebuilds
without a GPU.

A builder whose inputs do not exist yet is SKIPPED with a message naming the missing
file; it never draws from stand-in data. ``--demo`` is for the test-suite's synthetic
fixtures only: every output is watermarked and suffixed `_DEMO`.

Ids: F2 fold leakage . T1 dataset/split inventory . F3 accuracy vs split . F4 competitors
. T2 headline table . F5 calibration . F6 (+T3) ablations . F7 (+T4) geometry arms
. F8 epistasis . F9 (+T5) efficiency. F1 (architecture schematic) and F10 (GUI
screenshot) are not generated here.

Exit status: 1 if any builder raised or any consistency check failed, else 0
(``--strict`` also fails on a skipped builder).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.figures import style as S  # noqa: E402
from scripts.figures.loaders import SPLIT_VARIANTS, Ctx, MissingArtifact  # noqa: E402

DEFAULT_RELEASE = "../extras/zenodo_staging/Holo-StabBench-v1.0"
ALIASES = {"T3": "F6", "T4": "F7", "T5": "F9"}


def _lazy(module: str, name: str):
    def run(ctx):
        import importlib
        return getattr(importlib.import_module(f"scripts.figures.{module}"), name)(ctx)
    return run


BUILDERS = {
    "F2": _lazy("audit_figs", "build_f2"),
    "T1": _lazy("audit_figs", "build_t1"),
    "F3": _lazy("result_figs", "build_f3"),
    "F4": _lazy("result_figs", "build_f4"),
    "T2": _lazy("result_figs", "build_t2"),
    "F5": _lazy("result_figs", "build_f5"),
    "F6": _lazy("result_figs", "build_f6"),
    "F7": _lazy("result_figs", "build_f7"),
    "F8": _lazy("result_figs", "build_f8"),
    "F9": _lazy("result_figs", "build_f9"),
}


def _portable(inputs: dict, release: Path, results: Path) -> dict:
    out = {}
    for k, v in inputs.items():
        p = Path(k)
        for base, tag in ((results, "results"), (release, "release")):
            try:
                out[f"{tag}/{p.relative_to(base).as_posix()}"] = v
                break
            except ValueError:
                continue
        else:
            out[p.name] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", default=DEFAULT_RELEASE, help="Holo-StabBench release folder")
    ap.add_argument("--results", default=None, help="results folder (default: <release>/results)")
    ap.add_argument("--out", default="../paper", help="paper folder; figures/ and tables/ are created in it")
    ap.add_argument("--only", default=None, help="comma-separated ids, e.g. F2,T1")
    ap.add_argument("--primary-split", default="id30", choices=SPLIT_VARIANTS,
                    help="split used by the single-split figures and tables (F4, T2, F9)")
    ap.add_argument("--demo", action="store_true",
                    help="inputs are synthetic fixtures: watermark and suffix every output")
    ap.add_argument("--strict", action="store_true", help="also exit 1 when a builder is skipped")
    args = ap.parse_args(argv)

    release = Path(args.release).resolve()
    results = Path(args.results).resolve() if args.results else release / "results"
    out = Path(args.out).resolve()
    ctx = Ctx(release=release, results=results, figures=out / "figures", tables=out / "tables",
              demo=args.demo, primary_split=args.primary_split)

    wanted = None
    if args.only:
        wanted = {ALIASES.get(x.strip().upper(), x.strip().upper()) for x in args.only.split(",") if x.strip()}
        unknown = wanted - set(BUILDERS)
        if unknown:
            ap.error(f"unknown id(s) {sorted(unknown)}; choose from {sorted(BUILDERS)} (T3/T4/T5 ride with F6/F7/F9)")

    records: list[dict] = []
    skipped: dict[str, str] = {}
    failed: list[str] = []
    bad_checks: list[str] = []
    for bid, fn in BUILDERS.items():
        if wanted is not None and bid not in wanted:
            continue
        t0 = time.time()
        try:
            recs = fn(ctx)
        except MissingArtifact as exc:
            skipped[bid] = str(exc)
            print(f"SKIP {bid}: {exc}")
            continue
        except Exception:
            failed.append(bid)
            print(f"FAIL {bid}:")
            traceback.print_exc()
            continue
        for rec in recs:
            rec["inputs"] = _portable(rec.get("inputs", {}), release, results)
            records.append(rec)
            print(f"ok   {rec['id']:<18} {', '.join(rec['files'])}  ({time.time() - t0:.1f}s)")
            for c in rec.get("checks", []):
                if not c["ok"]:
                    msg = f"{rec['id']}: {c['check']} expected {c['expected']} got {c['got']}"
                    bad_checks.append(msg)
                    print(f"CHECK FAILED {msg}")

    manifest_path = out / "figure_manifest.json"
    prior: dict = {}
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8")).get("records", {})
        except (OSError, ValueError):
            prior = {}
    for rec in records:
        prior[rec["id"] + ("_DEMO" if rec["demo"] else "")] = rec
    rm = release / "release_manifest.json"
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": S.figure_provenance(),
        "release": release.name,
        "release_manifest_sha256": S.sha256_file(rm) if rm.is_file() else None,
        "primary_split": args.primary_split,
        "records": prior,
        "skipped": skipped,
    }
    out.mkdir(parents=True, exist_ok=True)
    S.write_text_lf(manifest_path, json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\n{len(records)} output(s), {len(skipped)} skipped, {len(failed)} failed; manifest: {manifest_path}")
    if failed or bad_checks or (args.strict and skipped):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
