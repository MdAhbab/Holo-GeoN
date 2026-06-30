#!/usr/bin/env python
"""
run.py — single entrypoint for HoloGNN
======================================
One file your friend runs to set up the environment and to train / test / eval
the whole model. Configure everything in configs/run.yaml (or override on the
command line; CLI always wins).

Quick start
-----------
    python run.py setup                 # create a venv + install deps
    #   (activate it, then:)
    python run.py smoke                 # 30-sec sanity check on synthetic data
    python run.py all                   # full pipeline (precompute -> ... -> eval)

Individual stages
-----------------
    python run.py precompute
    python run.py splits
    python run.py stability        [--no-ablation] [--sweep-eps-delta] [--back-translate]
    python run.py pathogenicity
    python run.py idr
    python run.py baseline
    python run.py contacts              # geometric-track sanity (Q2)
    python run.py evaluate
    python run.py test                  # run unit tests

Common overrides (work on any stage)
------------------------------------
    --config configs/run.yaml
    --esm-model facebook/esm2_t33_650M_UR50D --cache cache/emb_650M
    --device cuda|cpu
    --stability-data PATH --clinvar-data PATH --clinvar-ddg PATH
    --epochs N            (sets all epochs)  | --epochs-stability N etc.
    --batch N             | --split-method mmseqs|group --identity 0.30
    --max-seqs N --max-rows N --out-root runs/
    --back-translate      (proxy transcript when stability data has no coding DNA)
    --sweep-eps-delta     (probe E_mech threshold sensitivity)
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "esm_model": "facebook/esm2_t30_150M_UR50D",
    "cache": "cache/emb_150M",
    "device": "cuda",
    "max_length": 256,
    "out_root": "runs",
    "stability_data": "data/megascale_ddg_pairs.parquet",
    "clinvar_data": "data/clinvar_protein_pairs.parquet",
    "clinvar_ddg": "data/clinvar_protein_pairs.parquet.ddg_w49.npy",
    "split_method": "mmseqs",
    "identity": 0.30,
    "epochs_stability": 8,
    "epochs_pathogenicity": 6,
    "epochs_idr": 5,
    "batch_stability": 16,
    "batch_pathogenicity": 32,
    "lr": 3e-4,
    "run_ablation": True,
    "run_idr": True,
    "run_baseline": True,
    "run_contacts": True,
    "back_translate": False,
    "sweep_eps_delta": False,
    "max_seqs": None,
    "max_rows": None,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_yaml(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        print("! PyYAML not installed; ignoring config file. (pip install pyyaml)")
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def effective_config(args):
    cfg = dict(DEFAULTS)
    cfg.update(load_yaml(args.config))
    # CLI overrides (only those the user actually set)
    o = {
        "esm_model": args.esm_model, "cache": args.cache, "device": args.device,
        "max_length": args.max_length, "out_root": args.out_root,
        "stability_data": args.stability_data, "clinvar_data": args.clinvar_data,
        "clinvar_ddg": args.clinvar_ddg, "split_method": args.split_method,
        "identity": args.identity, "batch_stability": args.batch,
        "batch_pathogenicity": args.batch, "lr": args.lr,
        "max_seqs": args.max_seqs, "max_rows": args.max_rows,
    }
    for k, v in o.items():
        if v is not None:
            cfg[k] = v
    if args.epochs is not None:
        cfg["epochs_stability"] = cfg["epochs_pathogenicity"] = cfg["epochs_idr"] = args.epochs
    for k in ("epochs_stability", "epochs_pathogenicity", "epochs_idr"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if args.no_ablation:
        cfg["run_ablation"] = False
    if args.no_idr:
        cfg["run_idr"] = False
    if args.back_translate:
        cfg["back_translate"] = True
    if args.sweep_eps_delta:
        cfg["sweep_eps_delta"] = True
    return cfg


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def py(module, *a):
    sh([sys.executable, "-m", module, *[str(x) for x in a]])


def venv_python(venv):
    if os.name == "nt":
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_setup(args, cfg):
    venv = args.venv or os.path.join(ROOT, ".venv")
    if not os.path.exists(venv_python(venv)):
        print(f"Creating venv at: {venv}")
        sh([sys.executable, "-m", "venv", venv])
    else:
        print(f"Using existing venv: {venv}")
    vpy = venv_python(venv)
    sh([vpy, "-m", "pip", "install", "--upgrade", "pip"])
    if args.torch_index_url:
        sh([vpy, "-m", "pip", "install", "torch", "--index-url", args.torch_index_url])
    sh([vpy, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])
    act = (f"{venv}\\Scripts\\activate" if os.name == "nt" else f"source {venv}/bin/activate")
    print("\n" + "=" * 64)
    print("Setup done. Activate the venv, then run the pipeline:")
    print(f"    {act}")
    print("    python run.py smoke      # quick sanity check")
    print("    python run.py all        # full train/test/eval")
    print("Note: for an NVIDIA GPU, install the CUDA torch wheel, e.g.:")
    print("    python run.py setup --torch-index-url https://download.pytorch.org/whl/cu121")
    print("Optional: install MMseqs2 for strict splits (conda install -c bioconda mmseqs2).")
    print("=" * 64)


def stage_precompute(args, cfg):
    extra = []
    if cfg["max_seqs"]:
        extra += ["--max-seqs", cfg["max_seqs"]]
    py("scripts.precompute_embeddings", "--data", cfg["stability_data"], cfg["clinvar_data"],
       "--esm-model", cfg["esm_model"], "--cache", cfg["cache"],
       "--max-length", cfg["max_length"], "--device", cfg["device"], *extra)


def _splits_paths(cfg):
    sm = os.path.join("data", "splits_megascale.parquet")
    sc = os.path.join("data", "splits_clinvar.parquet")
    return sm, sc


def stage_splits(args, cfg):
    sm, sc = _splits_paths(cfg)
    py("scripts.make_splits", "--data", cfg["stability_data"], "--kind", "stability",
       "--method", cfg["split_method"], "--identity", cfg["identity"], "--out", sm)
    py("scripts.make_splits", "--data", cfg["clinvar_data"], "--kind", "pathogenicity",
       "--method", "group", "--out", sc)


def stage_stability(args, cfg):
    sm, _ = _splits_paths(cfg)
    out = os.path.join(cfg["out_root"], "stability")
    common = ["--splits", sm, "--cache", cfg["cache"], "--esm-model", cfg["esm_model"],
              "--epochs", cfg["epochs_stability"], "--batch", cfg["batch_stability"],
              "--lr", cfg["lr"], "--device", cfg["device"]]
    if cfg["max_rows"]:
        common += ["--max-rows", cfg["max_rows"]]
    if cfg.get("back_translate"):
        common += ["--back-translate"]
    main_extra = ["--sweep"] if cfg.get("sweep_eps_delta") else []
    py("scripts.train_stability", *common, *main_extra, "--out", out)
    if cfg["run_ablation"]:
        py("scripts.train_stability", *common, "--no-mech-track",
           "--out", out + "_nomech")


def stage_pathogenicity(args, cfg):
    out = os.path.join(cfg["out_root"], "pathogenicity")
    a = ["--data", cfg["clinvar_data"], "--cache", cfg["cache"],
         "--esm-model", cfg["esm_model"], "--epochs", cfg["epochs_pathogenicity"],
         "--batch", cfg["batch_pathogenicity"], "--lr", cfg["lr"],
         "--device", cfg["device"], "--out", out]
    if cfg["clinvar_ddg"] and os.path.exists(os.path.join(ROOT, cfg["clinvar_ddg"])):
        a += ["--ddg", cfg["clinvar_ddg"]]
    if cfg["max_rows"]:
        a += ["--max-rows", cfg["max_rows"]]
    py("scripts.train_pathogenicity", *a)


def stage_idr(args, cfg):
    py("scripts.train_idr", "--data", cfg["clinvar_data"], "--cache", cfg["cache"],
       "--esm-model", cfg["esm_model"], "--epochs", cfg["epochs_idr"],
       "--device", cfg["device"], "--out", os.path.join(cfg["out_root"], "idr"))


def stage_baseline(args, cfg):
    sm, _ = _splits_paths(cfg)
    py("scripts.baselines", "--splits", sm, "--esm-model", cfg["esm_model"],
       "--split", "test", "--device", cfg["device"])


def stage_contacts(args, cfg):
    a = ["--cache", cfg["cache"], "--max-seqs", 300]
    tcd = cfg.get("true_contacts_dir")
    if tcd:
        a += ["--true-contacts-dir", tcd]
    py("scripts.contact_sanity", *a)


def stage_evaluate(args, cfg):
    runs = [os.path.join(cfg["out_root"], "stability"),
            os.path.join(cfg["out_root"], "pathogenicity")]
    if cfg["run_ablation"]:
        runs.append(os.path.join(cfg["out_root"], "stability_nomech"))
    py("scripts.evaluate", "--runs", *runs, "--out", os.path.join(cfg["out_root"], "figures"))


def stage_results(args, cfg):
    py("scripts.collect_results")


def stage_smoke(args, cfg):
    py("tests.test_model_smoke")


def stage_test(args, cfg):
    py("tests.test_features_graph")
    py("tests.test_splits_idr")
    py("tests.test_model_smoke")


def stage_all(args, cfg):
    stage_precompute(args, cfg)
    stage_splits(args, cfg)
    stage_stability(args, cfg)
    stage_pathogenicity(args, cfg)
    if cfg["run_idr"]:
        stage_idr(args, cfg)
    if cfg["run_baseline"]:
        stage_baseline(args, cfg)
    if cfg.get("run_contacts", True):
        stage_contacts(args, cfg)
    stage_evaluate(args, cfg)
    stage_results(args, cfg)
    print("\n[done] Pipeline complete. See results/SUMMARY.md and results/CLAIMS.md.")


STAGES = {
    "setup": stage_setup, "smoke": stage_smoke, "test": stage_test,
    "precompute": stage_precompute, "splits": stage_splits,
    "stability": stage_stability, "pathogenicity": stage_pathogenicity,
    "idr": stage_idr, "baseline": stage_baseline, "contacts": stage_contacts,
    "evaluate": stage_evaluate, "results": stage_results, "all": stage_all,
}


def main():
    ap = argparse.ArgumentParser(description="HoloGNN single-file runner.")
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--config", default="configs/run.yaml")
    # env / venv
    ap.add_argument("--venv", default=None, help="venv location (default: ./.venv)")
    ap.add_argument("--torch-index-url", default=None, help="e.g. CUDA wheel index")
    # shared overrides
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--stability-data", default=None)
    ap.add_argument("--clinvar-data", default=None)
    ap.add_argument("--clinvar-ddg", default=None)
    ap.add_argument("--split-method", default=None, choices=[None, "mmseqs", "group"])
    ap.add_argument("--identity", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="set all epochs")
    ap.add_argument("--epochs-stability", type=int, default=None)
    ap.add_argument("--epochs-pathogenicity", type=int, default=None)
    ap.add_argument("--epochs-idr", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-idr", action="store_true")
    ap.add_argument("--back-translate", action="store_true",
                    help="usage-weighted proxy transcript when stability data has no coding DNA")
    ap.add_argument("--sweep-eps-delta", action="store_true",
                    help="probe E_mech eps/delta sensitivity in the stability stage (Q9)")
    args = ap.parse_args()

    cfg = effective_config(args)
    if args.stage != "setup":
        print(f"[run.py] stage={args.stage} esm={cfg['esm_model']} device={cfg['device']} "
              f"cache={cfg['cache']}")
    STAGES[args.stage](args, cfg)


if __name__ == "__main__":
    main()
