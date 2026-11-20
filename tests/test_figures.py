"""
tests/test_figures.py
=====================
Paper figure / table builders (scripts/figures + scripts/make_figures), CPU only.

The fixtures are written through the pipeline's REAL writers wherever one can run
without a GPU - scripts.score_predictions.main for the scorer JSON, calibration_report
for calibration, remote_job.aggregate for the suite summaries - so the builders are
tested against the formats the pipeline produces, not against a hand-made guess.
Everything built from them is watermarked and suffixed _DEMO: these tests prove the
plumbing and the claims discipline, never a result.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tests.helpers import check, section

AA = list("ACDEFGHIKLMNPQRSTVWY")


# --- fixtures -------------------------------------------------------------------------
def _make_release(root: Path) -> Path:
    """A 24-domain stand-in for the Holo-StabBench release (6 fold blocks of 4)."""
    from scripts.figures.audit_figs import fold_clusters, tm_matrices

    rng = np.random.default_rng(3)
    rel = root / "release"
    (rel / "splits").mkdir(parents=True)
    (rel / "tables").mkdir()
    n = 24
    parents = [f"P{i:02d}.pdb" for i in range(n)]
    seqs = ["".join(rng.choice(AA, 40)) for _ in parents]
    pd.DataFrame({"parent": parents, "wt_seq": seqs}).to_parquet(rel / "tables/megascale_ddg_pairs.parquet", index=False)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 4 == j // 4:
                lo = 0.65 + 0.2 * float(rng.random())
                hi = max(lo, 0.9)
            else:
                lo = float(rng.uniform(0.2, 0.5))
                hi = lo + float(rng.uniform(0.0, 0.15))
            rows.append((parents[i], parents[j], hi, lo))
    pd.DataFrame(rows, columns=["parent_i", "parent_j", "tm_max", "tm_min"]).to_parquet(
        rel / "splits/megascale_tm_all_vs_all.parquet", index=False)

    assign = {"id30": {4: "test", 5: "test", 3: "val"}, "id25": {0: "test", 5: "test", 3: "val"},
              "fold_tm60": {2: "test", 5: "test", 3: "val"}}
    for v, m in assign.items():
        recs = []
        for i, (p, s) in enumerate(zip(parents, seqs)):
            split = m.get(i // 4, "train")
            for k in range(10):
                recs.append({"parent_seq": s, "site": k, "wt_aa": s[k], "mt_aa": "A", "target": float(rng.normal()),
                             "sigma": 0.2, "group": s, "cds": None, "split": split, "group_used": f"{v}:p{i}"})
        pd.DataFrame(recs).to_parquet(rel / f"splits/splits_megascale_{v}.parquet", index=False)

    _, mx, mn = tm_matrices(rel)
    census = {}
    for kind, mat in (("tm_max", mx), ("tm_min", mn)):
        for t in (0.50, 0.55, 0.60, 0.65, 0.70):
            lab = fold_clusters(mat, t)
            sizes = np.bincount(lab)
            census[f"{kind}_ge_{t:.2f}"] = {"n_clusters": int(len(sizes)), "largest_cluster_proteins": int(sizes.max()),
                                            "largest_cluster_frac": float(sizes.max() / n)}
    (rel / "splits/megascale_tm_fold_audit_summary.json").write_text(
        json.dumps({"fold_cluster_census": census}), encoding="utf-8")
    (rel / "splits/leakage_report_megascale_id25.json").write_text(
        json.dumps({"heldout_8mer_jaccard_ge_0.30": 0}), encoding="utf-8")
    meta = {"tables/fireprotdb_holdout.parquet": (400, 20), "tables/fireprotdb_hf25.parquet": (180, 12),
            "tables/s669_holdout.parquet": (66, 9), "tables/domainome_apca_holdout.parquet": (5000, 50),
            "tables/domainome_apca_hf25.parquet": (3200, 33)}
    (rel / "release_manifest.json").write_text(json.dumps({"files": [
        {"path": p, "parquet": {"rows": r, "unique_parent": u}} for p, (r, u) in meta.items()]}), encoding="utf-8")
    return rel


def _score(sp, argv: list[str]) -> None:
    old, sys.argv = sys.argv, argv
    try:
        with redirect_stdout(io.StringIO()):
            sp.main()
    finally:
        sys.argv = old


def _make_results(root: Path, rel: Path) -> Path:
    """Scorer JSON, calibration, suite summaries, epistasis and a speed benchmark."""
    from scripts import score_predictions as sp
    from scripts.calibration import calibration_report
    from scripts.remote_job import aggregate
    from src.metrics import regression_metrics

    rng = np.random.default_rng(11)
    res = root / "results"
    models = {"holo_geon": 0.5, "v1_baseline": 0.8, "esm_mlp_matched_capacity": 0.6, "thermompnn": 0.9}
    for v in ("id30", "id25", "fold_tm60"):
        df = pd.read_parquet(rel / f"splits/splits_megascale_{v}.parquet")
        y = df["target"].to_numpy()
        argv = ["score", "--splits", str(rel / f"splits/splits_megascale_{v}.parquet"), "--reps", "40",
                "--out", str(res / "scores" / f"{v}.json")]
        for name, noise in models.items():
            p = root / f"pred_{v}_{name}.parquet"
            pd.DataFrame({"row_id": np.arange(len(df)), "y_pred": y + rng.normal(0, noise, len(y))}).to_parquet(p)
            argv += ["--model", f"{name}={p}"]
        _score(sp, argv)

    (res / "calibration").mkdir(parents=True)
    for arm, sig in (("full", 0.55), ("standard_nll", 0.35), ("v1_baseline", 0.25)):
        for seed in (42, 43):
            y = rng.normal(0, 0.8, 500)
            mu = y + rng.normal(0, 0.5, 500)
            (res / "calibration" / f"{arm}_s{seed}.json").write_text(
                json.dumps({"calibration": calibration_report(mu, np.full(500, sig), y)}), encoding="utf-8")

    def suite(name: str, arms: dict[str, float]) -> None:
        runs = root / f"runs_{name}"
        items = []
        for arm, shift in arms.items():
            for seed in range(42, 47):
                pack = f"{arm}_s{seed}"
                d = runs / pack
                d.mkdir(parents=True)
                base = 0.75 + 0.01 * (seed - 44) + shift + float(rng.normal(0, 0.004))
                (d / "metrics.json").write_text(json.dumps({
                    "test": {"stability": {"pearson": base, "rmse": 0.65 - shift}}, "best_epoch": 6}), encoding="utf-8")
                (d / "external_fireprotdb_holdout.json").write_text(json.dumps({
                    "metrics": {"pearson": base - 0.3, "rmse": 1.5}}), encoding="utf-8")
                items.append({"kind": "train", "arm": arm, "seed": seed, "pack": pack})
        summary = aggregate({}, items, {"out_root": str(runs), "external_file": "data/fireprotdb_holdout.parquet"})
        (res / name).mkdir(parents=True, exist_ok=True)
        (res / name / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    suite("ablations", {"full": 0.0, "v1_baseline": -0.01, "no_ssm": 0.0005, "no_cds_dropout": -0.03,
                        "standard_nll": -0.002})
    suite("geometry", {"full": 0.0, "str_only": -0.02, "both_streams": 0.001, "str_topology_only": -0.04,
                       "true_contact_edges": 0.004})

    for arm, skill in (("additive_only", 0.0), ("potts_no_gate", 0.3), ("contact_gated_egeo", 0.6),
                       ("true_contact_oracle", 0.7)):
        for seed in (42, 43):
            d = res / "epistasis" / f"{arm}_s{seed}"
            d.mkdir(parents=True)
            y = rng.normal(0, 0.5, 400)
            dist = rng.uniform(3, 14, 400)
            pred = skill * y + rng.normal(0, 0.5, 400)
            near = dist < 8
            (d / "epistasis_metrics.json").write_text(json.dumps({"arm": arm, "seed": seed, "test": {
                "n_rows": 400, "dddg_epistasis": regression_metrics(y, pred),
                "dddg_contact_lt8A": {"n": int(near.sum()), **regression_metrics(y[near], pred[near])},
                "dddg_distant_ge8A": {"n": int((~near).sum()), **regression_metrics(y[~near], pred[~near])}}}),
                encoding="utf-8")
            if arm == "contact_gated_egeo":
                pd.DataFrame({"row_id": np.arange(400), "y_ab": y, "pred_ab": pred, "y_epi": y, "pred_epi": pred,
                              "dist_cb": dist}).to_parquet(d / "pred_epistasis_test.parquet", index=False)

    (res / "efficiency").mkdir()
    (res / "efficiency" / "benchmark_speed.json").write_text(json.dumps({
        "gpu": "TEST GPU", "dtype": "fp32", "repeats": 5, "scan_lengths": [100, 300, 500],
        "models": {"holo_geon": {"trainable_params": 2038417, "scan_seconds": {"100": 0.1, "300": 0.4, "500": 0.9},
                                 "peak_vram_gb": 3.2},
                   "v1_baseline": {"trainable_params": 20607788, "scan_seconds": {"100": 0.4, "300": 1.5, "500": 3.1},
                                   "peak_vram_gb": 5.0},
                   "thermompnn": {"trainable_params": 1700000, "scan_seconds": {"100": 0.2, "300": 0.8, "500": 1.7},
                                  "peak_vram_gb": 2.1}}}), encoding="utf-8")
    return res


# --- tests ----------------------------------------------------------------------------
def test_figures_unit_logic() -> None:
    section("figures - split clustering, LaTeX escaping, wording of a null")
    from scripts.figures import style as S
    from scripts.figures.audit_figs import fold_clusters
    from scripts.figures.result_figs import _reading
    from scripts.calibration import calibration_report

    tm = np.full((5, 5), 0.2)
    tm[0, 1] = tm[1, 0] = 0.7          # one linked pair
    tm[2, 3] = tm[3, 2] = 0.7          # another
    lab = fold_clusters(tm, 0.6)
    check(len(set(lab.tolist())) == 3, "TM links form components (5 domains, 2 linked pairs -> 3 clusters)")
    lab = fold_clusters(tm, 0.6, groups=["a", "a", "b", "b", "a"])
    check(len(set(lab.tolist())) == 2, "a shared sequence-cluster id also links domains (the released clustering rule)")

    t = S.tex("TM_min < 0.60 & ρ ≥ 0.5")
    check("\\_" in t and "$<$" in t and "\\&" in t and "$\\geq$" in t and "$\\rho$" in t,
          "LaTeX escaping handles underscores, < and Greek/inequality glyphs")

    base = {"n_seeds": 5, "p_holm": 0.4, "significant": False, "delta": 0.002, "mde": 0.02}
    check(_reading(base) == "cannot tell at 5 seeds",
          "a non-significant effect smaller than the detectable one is worded 'cannot tell at 5 seeds'")
    check(_reading({**base, "p_holm": 0.001, "significant": True, "delta": -0.03}) == "lower than full",
          "a significant negative delta is 'lower than full'")
    check(_reading({**base, "p_holm": float("nan"), "n_seeds": 2}).startswith("untested"),
          "fewer than 3 seeds is 'untested', not 'no effect'")

    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    rep = calibration_report(y + rng.normal(0, 0.5, 400), np.full(400, 0.5), y)
    rel = rep["reliability"]
    check(len(rel["nominal"]) == 99 and len(rel["empirical"]) == 99,
          "calibration_report now carries the 99-point PIT reliability curve")
    check(all(b >= a - 1e-9 for a, b in zip(rel["empirical"], rel["empirical"][1:])),
          "the reliability curve is monotone (it is an empirical CDF)")


def test_figures_pipeline_on_fixtures() -> None:
    section("figures - every builder runs on fixtures written by the real writers")
    import matplotlib

    from scripts import make_figures

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rel = _make_release(root)
        res = _make_results(root, rel)

        doc = json.loads((res / "scores" / "id30.json").read_text(encoding="utf-8"))
        e = doc["models"]["holo_geon"]
        want = ("pearson_ci95", "spearman_ci95", "rmse_ci95", "per_protein_median_pearson_ci95",
                "per_protein_median_spearman_ci95")
        check(all(k in e and len(e[k]) == 2 and e[k][0] is not None and e[k][0] <= e[k][1] for k in want),
              "the scorer emits a cluster-bootstrap interval for every pooled and per-protein-median metric")
        check(doc.get("ci", "").startswith("95%") and doc.get("seed") == 0, "the scorer records its CI method and seed")

        out = root / "paper"
        with redirect_stdout(io.StringIO()) as buf:
            rc = make_figures.main(["--release", str(rel), "--results", str(res), "--out", str(out), "--demo"])
        check(rc == 0, f"make_figures --demo exits 0 on the fixtures (log: {buf.getvalue()[-300:]!r})")
        figs, tabs = out / "figures", out / "tables"
        expect_f = ["F2_fold_leakage", "F3_accuracy_vs_split", "F4_competitors", "F5_calibration",
                    "F6_ablation_forest", "F7_geometry_arms", "F8_epistasis", "F9_efficiency"]
        for stem in expect_f:
            png = figs / f"{stem}_DEMO.png"
            check(png.is_file() and png.stat().st_size > 5_000 and (figs / f"{stem}_DEMO.pdf").is_file()
                  and (figs / f"{stem}_DEMO_data.csv").is_file() and (figs / f"{stem}_DEMO_caption.txt").is_file(),
                  f"{stem}: PNG, PDF, table twin and caption written with the _DEMO suffix")
        expect_t = ["T1a_splits", "T1b_external", "T2_headline", "T3_ablations", "T4_geometry", "T5_efficiency"]
        for stem in expect_t:
            tex = tabs / f"{stem}_DEMO.tex"
            check(tex.is_file() and "\\toprule" in tex.read_text(encoding="utf-8") and (tabs / f"{stem}_DEMO.csv").is_file(),
                  f"{stem}: booktabs LaTeX and CSV twin written")
        check(not any(p.name.endswith(".png") and "_DEMO" not in p.name for p in figs.iterdir()),
              "nothing built from fixtures is written without the _DEMO suffix")

        man = json.loads((out / "figure_manifest.json").read_text(encoding="utf-8"))
        recs = man["records"]
        check(len(recs) >= len(expect_f) + len(expect_t), "the manifest records every output")
        check(all(r["demo"] for r in recs.values()), "every manifest record is flagged demo")
        check(all(len(h) == 64 for r in recs.values() for h in r["inputs"].values()),
              "every input of every output is pinned by SHA-256")
        check(any(k.startswith("results/scores/") for r in recs.values() for k in r["inputs"]),
              "input paths are recorded relative to the release / results folders (portable)")
        check(man["provenance"]["matplotlib"] == matplotlib.__version__, "the plotting stack is recorded")

        t2 = pd.read_csv(tabs / "T2_headline_DEMO.csv")
        check(any(c.startswith("pooled_") for c in t2.columns) and any(c.startswith("per_protein_median_") for c in t2.columns),
              "T2 keeps pooled and per-protein-median metrics in separate columns")
        tex2 = (tabs / "T2_headline_DEMO.tex").read_text(encoding="utf-8")
        check("Pooled Spearman" in tex2 and "Per-protein median Spearman" in tex2 and "[" in tex2,
              "T2 LaTeX has both column groups, each with a bracketed 95% interval")
        t3 = pd.read_csv(tabs / "T3_ablations_DEMO.csv")
        check("full" in set(t3["arm"]) and any("reading" in c for c in t3.columns),
              "T3 carries a reference row and a reading column")
        text = " ".join(p.read_text(encoding="utf-8") for p in list(figs.glob("*.txt")) + list(tabs.glob("*.tex")))
        low = text.lower()
        check("equivariant" not in low, "no caption or table claims equivariance (the default model is structure-free)")
        check("invariant" in low and "structure-free" in low or "SE(3)-invariant" in text,
              "where 3D arms are mentioned they are called SE(3)-invariant")
        check("fold-disjoint" not in low.replace("not fold-disjoint", ""),
              "the fold split is never called fold-disjoint without the qualifier")

        f2 = recs["F2_fold_leakage_DEMO"]
        check(all(c["ok"] for c in f2.get("checks", [])), "F2's derived numbers agree with the release's audit summaries")

        # nothing available -> skipped, never faked
        empty = root / "empty_results"
        empty.mkdir()
        with redirect_stdout(io.StringIO()) as buf:
            rc_skip = make_figures.main(["--release", str(rel), "--results", str(empty), "--out", str(root / "p2"),
                                         "--demo", "--only", "F3,F5,F6,F8,F9"])
            rc_strict = make_figures.main(["--release", str(rel), "--results", str(empty), "--out", str(root / "p3"),
                                           "--demo", "--only", "F3", "--strict"])
        check(rc_skip == 0 and buf.getvalue().count("SKIP") >= 6,
              "builders with no inputs are SKIPPED with a message, not drawn from stand-in data")
        check(rc_strict == 1, "--strict turns a skipped builder into a failing exit status")
        check(not list((root / "p2").glob("figures/*.png")), "skipped builders leave no figure behind")


TESTS = [test_figures_unit_logic, test_figures_pipeline_on_fixtures]
