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


def stage_etl(args, cfg) -> str | None:
    """
    Attach the real coding sequence and RETURN the path downstream stages read.

    Round-1 review, R1-05: this used to write the CDS-joined table and then
    merely print "set stability_data ... to use it". ``stage_all`` immediately
    called precompute with the unmodified path, so every downstream stage read
    the table with no ``wt_dna`` column: ``mech_valid`` was 0 everywhere,
    ``E_mech`` held only self-loops, and the ``no_mech_stream`` ablation - the
    arm that exists to test the paper's distinguishing contribution - measured a
    stream that had never been fed. The delta would have read ~0 and been
    written up as "the mechanistic stream does not contribute", which is the
    wrong conclusion from a plumbing bug.
    """
    if not args.cds_source:
        print("[etl] no --cds-source given. Skipping.\n"
              "      Without a real coding sequence the mechanistic channels stay "
              "masked (mech_valid = 0) and E_mech is empty. That is the honest\n"
              "      state, not a failure — but the mechanistic stream then has "
              "nothing to contribute and its ablation will read ~0.")
        return None
    out = cfg.stability_data.replace(".parquet", "_cds.parquet")
    if done(out):
        print(f"[etl] reusing {out}")
        return out
    py("scripts.etl_cds", "--table", cfg.stability_data,
       "--source", *args.cds_source, "--out", out)
    print(f"[etl] downstream stages will read {out}")
    return out


def stage_precompute(args, cfg) -> None:
    # The external table's proteins are encoded here too: scripts/eval_external
    # only scores rows that have a cached encoding, so leaving them out would
    # turn "the model transfers this well" into a statement about whichever
    # proteins happened to be in the cache.
    tables = [cfg.stability_data, cfg.clinvar_data]
    if cfg.values.get("external_data"):
        tables.append(cfg.values["external_data"])
    a = ["--data", *tables,
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
    """
    Controls for BOTH reported tasks.

    Round-2 review, R2-16: this only ever passed the stability split, so a
    reported ClinVar AUROC had no control at all - nothing said what an MLP on
    the same embeddings would score, which is the comparison a reviewer asks for
    first.
    """
    out = os.path.join(cfg.out_root, "baselines")
    stability, clinvar = _split_paths()
    for kind, splits in (("stability", stability), ("pathogenicity", clinvar)):
        target = os.path.join(out, kind)
        if done(os.path.join(target, "baselines.json")):
            print(f"[skip] {kind} baselines already complete")
            continue
        if not done(splits):
            print(f"[baselines] no {kind} split table at {splits}; skipping. Its "
                  f"head will have no control, so do not report it against one.")
            continue
        a = ["--splits", splits, "--out", target]
        if args.skip_zero_shot:
            a.append("--skip-zero-shot")
        py("scripts.baselines", *a)


def stage_ablations(args, cfg) -> None:
    out = os.path.join(cfg.out_root, "ablations")
    if done(os.path.join(out, "ablations.json")):
        print("[skip] ablations already complete")
        return
    stability, clinvar = _split_paths()
    # Round-2 review, R2-06: this passed no --seeds, so the documented
    # one-command pipeline ran a single seed per arm and wrote deltas to four
    # decimals with no test behind them. Five seeds is the minimum at which the
    # paired test has usable power against the 0.030 test-Pearson run-to-run
    # spread round 1 measured.
    py("scripts.ablations", "--stability-splits", stability,
       "--pathogenicity-splits", clinvar, "--out", out,
       "--seeds", *[str(s) for s in args.seeds])


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
    a = ["--cache", cfg.cache, "--max-seqs", 300]
    # Default to the maps built from the release's own AlphaFold models. Without
    # ground truth this stage can only report a long-range FRACTION, which
    # uniformly random edges also score well on; with it, precision@L - the
    # number the graph claim actually rests on.
    if not args.true_contacts_dir:
        built = os.path.join("data", "true_contacts")
        if os.path.isdir(_abs(built)) and os.listdir(_abs(built)):
            args.true_contacts_dir = built
            print(f"[contacts] using ground-truth contact maps in {built}")
    if args.true_contacts_dir:
        a += ["--true-contacts-dir", args.true_contacts_dir, "--require-structure"]
    else:
        # Round-2 review, R2-13: without ground-truth maps this stage reports a
        # long-range FRACTION, which uniformly random edges also score highly
        # on. It runs - the number is a useful smoke test - but the pipeline says
        # plainly that it cannot support the claim it is cited for.
        print("[contacts] NOTE: no --true-contacts-dir. This produces a "
              "long-range fraction against a proximity baseline and a random "
              "control, which can show the contact edges are NON-LOCAL but not "
              "that they are CORRECT. Do not cite it as evidence for building a "
              "graph; pass --true-contacts-dir <dir of C-beta maps> for "
              "precision@L.")
    py("scripts.contact_sanity", *a)


def stage_results(args, cfg) -> None:
    py("scripts.collect_results", "--runs", os.path.join(cfg.out_root, "main"),
       "--baselines", os.path.join(cfg.out_root, "baselines", "stability",
                                   "baselines.json"),
       "--ablations", os.path.join(cfg.out_root, "ablations", "ablations.json"))


def stage_external(args, cfg) -> None:
    """Score the trained checkpoint on a table it never saw."""
    run_dir = os.path.join(cfg.out_root, "main")
    ckpt = os.path.join(run_dir, "model_best.pt")
    external = cfg.values.get("external_data")
    if not external:
        print("[external] no external_data configured; skipping.")
        return
    if not done(external):
        print(f"[external] no table at {external}. Build it with "
              f"`python -m scripts.build_fireprot_table`; skipping.")
        return
    if not done(ckpt):
        print(f"[external] no checkpoint at {ckpt}; run `python run.py train` first")
        return
    # eval_external writes external_{key}.json, key = the data file's stem. Runs
    # made before that wrote only external_fireprotdb.json, which still counts
    # as complete for a fireprotdb* table.
    key = os.path.splitext(os.path.basename(external))[0]
    if (done(os.path.join(run_dir, f"external_{key}.json"))
            or (key.startswith("fireprotdb")
                and done(os.path.join(run_dir, "external_fireprotdb.json")))):
        print("[skip] external evaluation already complete")
        return
    py("scripts.eval_external", "--checkpoint", ckpt, "--data", external,
       "--out", run_dir)


def stage_test(args, cfg) -> None:
    py("tests.run_all")


def stage_all(args, cfg) -> None:
    # The CDS-joined table produced by etl is threaded through the rest of the
    # run in memory, so the mechanistic stream is actually fed (R1-05).
    cds_table = stage_etl(args, cfg)
    if cds_table:
        cfg.values["stability_data"] = cds_table
    stage_precompute(args, cfg)
    stage_splits(args, cfg)
    stage_train(args, cfg)
    stage_baselines(args, cfg)
    stage_ablations(args, cfg)
    stage_calibration(args, cfg)
    stage_external(args, cfg)
    stage_contacts(args, cfg)
    stage_results(args, cfg)
    print("\n[done] pipeline complete. See results/SUMMARY.md")


STAGES = {
    "setup": stage_setup, "test": stage_test, "inspect": stage_inspect,
    "etl": stage_etl, "precompute": stage_precompute, "splits": stage_splits,
    "train": stage_train, "baselines": stage_baselines, "ablations": stage_ablations,
    "calibration": stage_calibration, "contacts": stage_contacts,
    "external": stage_external,
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
    ap.add_argument("--true-contacts-dir", default=None,
                    help="contacts: directory of <cache-key>.npy C-beta contact "
                         "maps. Without it the contact stage cannot show the "
                         "edges are correct, only that they are non-local (R2-13)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="seeds for the ablation grid. Deltas are paired across "
                         "them and Holm-corrected; below 3 no test is possible "
                         "and the verdicts read null (R2-06)")
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
