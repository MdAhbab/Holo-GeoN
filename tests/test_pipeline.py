"""
tests/test_pipeline.py
======================
End-to-end pipeline exercise on synthetic data.

This runs the real command-line stages — ``make_splits``, ``train``,
``baselines``, ``collect_results`` — against a synthetic encoder cache, so the
wiring is tested rather than mocked: column normalisation, exclusion counting,
identity grouping, windowing, batching, the multi-task loop, checkpointing,
early stopping, metric computation and the results collector.

**Nothing measured here says anything about model accuracy.** The embeddings
are seeded noise from ``tests/helpers.stub_encoding`` and the targets are a
synthetic function of the substitution. The only claims made are structural:
the stages run, the artifacts appear, the loss on a learnable synthetic signal
goes down, and the collector reports what exists and marks what does not.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from src.encoder import EncodingStore
from src.features import KD_HYDROPATHY
from tests.helpers import check, random_protein, section, stub_encoding

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB_DIM = 320   # matches facebook/esm2_t6_8M_UR50D so ESM_DIMS lookup agrees

CONFIG_TEMPLATE = """
esm_model: facebook/esm2_t6_8M_UR50D
cache: {cache}
device: cpu
window: 48
contact_top_k: 8
contact_min_sep: 3
out_root: {runs}
stability_data: {stability}
clinvar_data: {clinvar}
idr_data: null
mfi_data: null
proteomics_data: null
max_protein_length: 5000
split_method: mmseqs
identity: 0.30
coverage: 0.80
cv_folds: 2
seed: 42
graph_top_k: 8
geo_threshold: 0.30
eps: 0.10
delta: 0.10
mech_max_degree: 8
neighborhood_hops: 2
hidden: 32
out_dim: 32
gat_heads: 2
dropout: 0.1
use_ssm: true
use_mech_stream: true
epochs: {epochs}
batch_size: 16
lr: 0.003
weight_decay: 0.0001
warmup_frac: 0.1
early_stop_patience: 20
grad_clip: 1.0
amp: false
num_workers: 0
max_rows: null
"""


def _run(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    --- {label} stdout ---\n{proc.stdout[-3000:]}")
        print(f"    --- {label} stderr ---\n{proc.stderr[-3000:]}")
    return proc


def _synthesise(workdir: str, n_parents: int = 24, per_parent: int = 14) -> dict:
    """Synthetic parents, mutations with a learnable target, and a stub cache."""
    data_dir = os.path.join(workdir, "data")
    cache_dir = os.path.join(workdir, "cache")
    os.makedirs(data_dir, exist_ok=True)
    store = EncodingStore(cache_dir)

    g = torch.Generator().manual_seed(11)
    stability_rows, clinvar_rows = [], []

    for p in range(n_parents):
        # Long enough that some mutations fall outside a head-truncated window,
        # which is the situation windowing exists to handle.
        parent = random_protein(120, seed=100 + p)
        store.put(stub_encoding(parent, dim=STUB_DIM, top_k=8))

        for k in range(per_parent):
            site = int(torch.randint(0, len(parent), (1,), generator=g).item())
            wt_aa = parent[site]
            mt_aa = "ACDEFGHIKLMNPQRSTVWY"[
                int(torch.randint(0, 20, (1,), generator=g).item())
            ]
            if mt_aa == wt_aa:
                continue
            mutant = parent[:site] + mt_aa + parent[site + 1:]
            # A signal the model can actually fit: hydropathy change at the site.
            ddg = 0.4 * (KD_HYDROPATHY[mt_aa] - KD_HYDROPATHY[wt_aa])
            stability_rows.append({
                "wt_seq": parent, "mt_seq": mutant,
                "ddg": ddg, "ddg_std": 0.3,
            })
            if k % 3 == 0:
                clinvar_rows.append({
                    "wt_seq": parent, "mt_seq": mutant,
                    "label": int(ddg < 0), "gene": f"GENE{p}",
                })

    stability_path = os.path.join(data_dir, "synthetic_stability.parquet")
    clinvar_path = os.path.join(data_dir, "synthetic_clinvar.parquet")
    pd.DataFrame(stability_rows).to_parquet(stability_path)
    pd.DataFrame(clinvar_rows).to_parquet(clinvar_path)

    runs_dir = os.path.join(workdir, "runs")
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(CONFIG_TEMPLATE.format(
            cache=cache_dir, runs=runs_dir, stability=stability_path,
            clinvar=clinvar_path, epochs=6,
        ))

    return {
        "workdir": workdir, "data": data_dir, "cache": cache_dir, "runs": runs_dir,
        "config": config_path, "stability": stability_path, "clinvar": clinvar_path,
        "n_stability": len(stability_rows), "n_clinvar": len(clinvar_rows),
    }


def test_end_to_end() -> None:
    section("end-to-end pipeline (synthetic encoder — plumbing only)")

    with tempfile.TemporaryDirectory(prefix="holo_geon_test_") as workdir:
        env = _synthesise(workdir)
        print(f"    {env['n_stability']} stability rows, {env['n_clinvar']} clinvar rows")

        # --- splits --------------------------------------------------------
        split_stability = os.path.join(env["data"], "splits_stability.parquet")
        split_clinvar = os.path.join(env["data"], "splits_clinvar.parquet")
        ok = True
        for path, kind, out in (
            (env["stability"], "stability", split_stability),
            (env["clinvar"], "pathogenicity", split_clinvar),
        ):
            proc = _run([sys.executable, "-m", "scripts.make_splits",
                         "--data", path, "--kind", kind, "--out", out,
                         "--config", env["config"]], f"make_splits {kind}")
            ok = ok and proc.returncode == 0
        check(ok, "make_splits completed for both tasks")
        check(os.path.exists(split_stability) and os.path.exists(split_clinvar),
              "split tables written")

        if os.path.exists(split_stability):
            df = pd.read_parquet(split_stability)
            check(set(df["split"]) <= {"train", "val", "test"}, "split labels are valid")
            spans = df.groupby("group_used")["split"].nunique().max()
            check(int(spans) == 1, "no group appears in more than one split")
            report = os.path.join(env["data"], "leakage_report_stability.json")
            check(os.path.exists(report), "leakage report written")
            if os.path.exists(report):
                with open(report, encoding="utf-8") as fh:
                    rep = json.load(fh)
                check("per_split" in rep and rep["per_split"],
                      "leakage report records per-split target spread")
                check("load_report" in rep,
                      "leakage report records exclusion accounting")

        # --- train ---------------------------------------------------------
        run_dir = os.path.join(env["runs"], "main")
        proc = _run([sys.executable, "-m", "scripts.train",
                     "--config", env["config"],
                     "--stability-splits", split_stability,
                     "--pathogenicity-splits", split_clinvar,
                     "--out", run_dir], "train")
        check(proc.returncode == 0, "training completed")

        metrics_path = os.path.join(run_dir, "metrics.json")
        check(os.path.exists(metrics_path), "metrics.json written")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as fh:
                metrics = json.load(fh)
            check("stability" in metrics["trained_tasks"], "stability task trained")
            check("pathogenicity" in metrics["trained_tasks"], "pathogenicity task trained")
            check(set(metrics["untrained_heads"]) == {"idr", "mfi", "proteomics"},
                  "heads without data are recorded as untrained")

            history = metrics.get("history", [])
            check(len(history) >= 2, "training ran multiple epochs")
            if len(history) >= 2:
                first = history[0]["train"].get("stability", float("inf"))
                best = min(h["train"].get("stability", float("inf")) for h in history)
                check(best < first,
                      f"stability loss decreased on a learnable signal "
                      f"({first:.3f} -> {best:.3f})")
            check(os.path.exists(os.path.join(run_dir, "model_best.pt")),
                  "best checkpoint saved")
            check("test" in metrics and metrics["test"],
                  "test-split metrics computed")
            for task, m in (metrics.get("test") or {}).items():
                check("target_std" in m,
                      f"{task}: target spread reported alongside the metric")

        # --- resume --------------------------------------------------------
        proc = _run([sys.executable, "-m", "scripts.train",
                     "--config", env["config"],
                     "--stability-splits", split_stability,
                     "--pathogenicity-splits", split_clinvar,
                     "--out", run_dir], "train-resume")
        check(proc.returncode == 0 and "resumed from epoch" in proc.stdout,
              "an interrupted run resumes from its checkpoint")

        # --- baselines -----------------------------------------------------
        baselines_dir = os.path.join(env["runs"], "baselines")
        proc = _run([sys.executable, "-m", "scripts.baselines",
                     "--config", env["config"], "--splits", split_stability,
                     "--out", baselines_dir, "--epochs", "3",
                     "--skip-zero-shot"], "baselines")
        check(proc.returncode == 0, "baselines completed")
        baselines_path = os.path.join(baselines_dir, "baselines.json")
        if os.path.exists(baselines_path):
            with open(baselines_path, encoding="utf-8") as fh:
                b = json.load(fh)
            check("esm_mlp" in b["baselines"],
                  "supervised embedding-only control present (the graph-vs-no-graph test)")
            check("substitution_frequency" in b["baselines"],
                  "substitution-frequency control present")

        # --- collect -------------------------------------------------------
        results_dir = os.path.join(workdir, "results")
        proc = _run([sys.executable, "-m", "scripts.collect_results",
                     "--runs", run_dir,
                     "--baselines", baselines_path,
                     "--ablations", os.path.join(env["runs"], "ablations", "ablations.json"),
                     "--contacts", os.path.join(workdir, "nonexistent.json"),
                     "--splits-dir", env["data"],
                     "--out", results_dir], "collect_results")
        check(proc.returncode == 0, "collect_results completed")

        summary_path = os.path.join(results_dir, "SUMMARY.md")
        check(os.path.exists(summary_path), "SUMMARY.md written")
        if os.path.exists(summary_path):
            summary = open(summary_path, encoding="utf-8").read()
            check("not run" in summary,
                  "stages that did not run are reported as 'not run', not omitted")
            check("Ablations" in summary and "Uncertainty calibration" in summary,
                  "summary covers every stage, run or not")


TESTS = [test_end_to_end]
