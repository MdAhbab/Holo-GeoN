#!/usr/bin/env python
"""
run.py — single entrypoint for Holo-GeoN
========================================

    python run.py setup            # create a venv and install dependencies
    python run.py test             # unit + invariant tests (no data needed)
    python run.py all              # the whole pipeline, end to end

Individual stages, in dependency order::

    python run.py etl              # attach real coding sequences (enables E_mech)
    python run.py precompute       # frozen ESM-2 pass -> encoding cache
    python run.py splits           # identity-clustered, leakage-controlled splits
    python run.py train            # the multi-task model
    python run.py baselines        # substitution frequency, ESM-only MLP, zero-shot
    python run.py ablations        # one component removed at a time
    python run.py calibration      # sigma coverage/ECE + reversibility
    python run.py contacts         # geometric-edge sanity
    python run.py results          # collect everything into results/

Configuration lives in ``configs/run.yaml`` and ``configs/data.yaml``; both are
read, and any value can be overridden on the command line (CLI wins).

Stages are resumable: each checks for its own output and skips work that is
already complete, so re-running after an interruption continues rather than
restarting.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.config import load_config  # noqa: E402


def sh(cmd: list) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}")
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def py(module: str, *a) -> None:
    sh([sys.executable, "-m", module, *a])


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def done(path: str) -> bool:
    return os.path.exists(_abs(path))


def venv_python(venv: str) -> str:
    return os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv, "bin", "python")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_setup(args, cfg) -> None:
    venv = args.venv or os.path.join(ROOT, ".venv")
    if not os.path.exists(venv_python(venv)):
        sh([sys.executable, "-m", "venv", venv])
    vpy = venv_python(venv)
    sh([vpy, "-m", "pip", "install", "--upgrade", "pip"])
    if args.torch_index_url:
        sh([vpy, "-m", "pip", "install", "torch", "--index-url", args.torch_index_url])
    sh([vpy, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])
    activate = (f"{venv}\\Scripts\\activate" if os.name == "nt"
                else f"source {venv}/bin/activate")
    print("\n" + "=" * 70)
    print("Setup complete. Activate the environment, then run the pipeline:")
    print(f"    {activate}")
    print("    python run.py test")
    print("    python run.py all")
    print()
    print("RTX 50-series (Blackwell, sm_120 — e.g. 5070 Ti): install the CUDA 12.8")
    print("wheel first, or the GPU will error or silently fall back to CPU:")
    print("    python run.py setup --torch-index-url https://download.pytorch.org/whl/cu128")
    print()
    print("MMseqs2 is a separate binary and is needed for identity-clustered")
    print("splits:  conda install -c bioconda mmseqs2")
    print("=" * 70)


def stage_inspect(args, cfg) -> None:
    for path in (cfg.stability_data, cfg.clinvar_data):
        if done(path):
            py("scripts.inspect_data", "--data", path)


def stage_etl(args, cfg) -> None:
    if not args.cds_source:
        print("[etl] no --cds-source given. Skipping.\n"
              "      Without a real coding sequence the mechanistic channels stay "
              "masked (mech_valid = 0) and E_mech is empty. That is the honest\n"
              "      state, not a failure — but the mechanistic stream then has "
              "nothing to contribute and its ablation will read ~0.")
        return
    out = cfg.stability_data.replace(".parquet", "_cds.parquet")
    if done(out):
        print(f"[skip] etl already produced {out}")
        return
    py("scripts.etl_cds", "--table", cfg.stability_data,
       "--source", *args.cds_source, "--out", out)
    print(f"[etl] set stability_data: {out} in configs/run.yaml to use it.")


def stage_precompute(args, cfg) -> None:
    a = ["--data", cfg.stability_data, cfg.clinvar_data,
         "--esm-model", cfg.esm_model, "--cache", cfg.cache, "--device", cfg.device]
    if args.max_seqs:
        a += ["--max-seqs", args.max_seqs]
    if args.max_cache_gb:
        a += ["--max-cache-gb", args.max_cache_gb]
    py("scripts.precompute_embeddings", *a)


def _split_paths() -> tuple[str, str]:
    return (os.path.join("data", "splits_megascale.parquet"),
            os.path.join("data", "splits_clinvar.parquet"))


def stage_splits(args, cfg) -> None:
    stability, clinvar = _split_paths()
    if not done(stability):
        py("scripts.make_splits", "--data", cfg.stability_data,
           "--kind", "stability", "--out", stability)
    else:
        print(f"[skip] {stability} exists")
    if done(cfg.clinvar_data) and not done(clinvar):
        py("scripts.make_splits", "--data", cfg.clinvar_data,
           "--kind", "pathogenicity", "--out", clinvar)
    elif done(clinvar):
        print(f"[skip] {clinvar} exists")


def stage_train(args, cfg) -> None:
    out = os.path.join(cfg.out_root, "main")
    if done(os.path.join(out, "metrics.json")):
        print(f"[skip] training already complete ({out}/metrics.json)")
        return
    stability, clinvar = _split_paths()
    py("scripts.train", "--stability-splits", stability,
       "--pathogenicity-splits", clinvar, "--out", out)


def stage_baselines(args, cfg) -> None:
    out = os.path.join(cfg.out_root, "baselines")
    if done(os.path.join(out, "baselines.json")):
        print("[skip] baselines already complete")
        return
    stability, _ = _split_paths()
    a = ["--splits", stability, "--out", out]
    if args.skip_zero_shot:
        a.append("--skip-zero-shot")
    py("scripts.baselines", *a)


def stage_ablations(args, cfg) -> None:
    out = os.path.join(cfg.out_root, "ablations")
    if done(os.path.join(out, "ablations.json")):
        print("[skip] ablations already complete")
        return
    stability, clinvar = _split_paths()
    py("scripts.ablations", "--stability-splits", stability,
       "--pathogenicity-splits", clinvar, "--out", out)


def stage_calibration(args, cfg) -> None:
    run_dir = os.path.join(cfg.out_root, "main")
    ckpt = os.path.join(run_dir, "model_best.pt")
    if not done(ckpt):
        print(f"[calibration] no checkpoint at {ckpt}; run `python run.py train` first")
        return
    if done(os.path.join(run_dir, "calibration.json")):
        print("[skip] calibration already complete")
        return
    stability, _ = _split_paths()
    a = ["--checkpoint", ckpt, "--splits", stability, "--out", run_dir]
    if args.encode_mutants:
        a.append("--encode-mutants")
    py("scripts.calibration", *a)


def stage_contacts(args, cfg) -> None:
    if done(os.path.join("results", "contacts", "contact_sanity.json")):
        print("[skip] contact sanity already complete")
        return
    py("scripts.contact_sanity", "--cache", cfg.cache, "--max-seqs", 300)


def stage_results(args, cfg) -> None:
    py("scripts.collect_results", "--runs", os.path.join(cfg.out_root, "main"),
       "--baselines", os.path.join(cfg.out_root, "baselines", "baselines.json"),
       "--ablations", os.path.join(cfg.out_root, "ablations", "ablations.json"))


def stage_test(args, cfg) -> None:
    py("tests.run_all")


def stage_all(args, cfg) -> None:
    stage_etl(args, cfg)
    stage_precompute(args, cfg)
    stage_splits(args, cfg)
    stage_train(args, cfg)
    stage_baselines(args, cfg)
    stage_ablations(args, cfg)
    stage_calibration(args, cfg)
    stage_contacts(args, cfg)
    stage_results(args, cfg)
    print("\n[done] pipeline complete. See results/SUMMARY.md")


STAGES = {
    "setup": stage_setup, "test": stage_test, "inspect": stage_inspect,
    "etl": stage_etl, "precompute": stage_precompute, "splits": stage_splits,
    "train": stage_train, "baselines": stage_baselines, "ablations": stage_ablations,
    "calibration": stage_calibration, "contacts": stage_contacts,
    "results": stage_results, "all": stage_all,
}


def main() -> None:
    # Force UTF-8 everywhere: the Windows console default (cp1252) raises on the
    # non-ASCII characters these stages print, which used to crash a stage
    # outright when its output was redirected to a file.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Holo-GeoN pipeline runner.")
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--venv", default=None)
    ap.add_argument("--torch-index-url", default=None)
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--max-cache-gb", type=float, default=None)
    ap.add_argument("--cds-source", nargs="+", default=None,
                    help="raw table(s) carrying a nucleotide column, for the etl stage")
    ap.add_argument("--encode-mutants", action="store_true",
                    help="calibration: embed mutants for the reversibility check")
    ap.add_argument("--skip-zero-shot", action="store_true",
                    help="baselines: skip the slow masked-LM baseline")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={
        "esm_model": args.esm_model, "cache": args.cache, "device": args.device,
        "out_root": args.out_root, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_rows": args.max_rows,
    })
    if args.stage != "setup":
        print(f"[run] stage={args.stage}  esm={cfg.esm_model}  device={cfg.device}  "
              f"cache={cfg.cache}")
    STAGES[args.stage](args, cfg)


if __name__ == "__main__":
    main()
