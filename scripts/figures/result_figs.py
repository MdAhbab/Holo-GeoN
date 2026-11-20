"""
scripts/figures/result_figs.py
==============================
F3-F9 and T2-T5, built from the outputs of the pipeline's own writers:

    scores/<split>.json                  scripts.score_predictions --out
    ablations|geometry/summary.json      scripts.remote_job aggregate (or scripts.ablations' ablations.json)
    calibration/<arm>_s<seed>.json       scripts.calibration
    epistasis/<arm>_s<seed>/...          scripts.train_epistasis
    efficiency/benchmark_speed.json      see build_f9 for the schema

Claims discipline enforced here, not left to the caller:
  * pooled and per-protein-median metrics are always separate panels / column groups;
  * every interval is the one the writer computed (cluster bootstrap over group_used for
    the scorer; paired t-interval across seeds for ablations), and its method is stated;
  * a non-significant ablation smaller than the detectable effect is worded
    "cannot tell at N seeds", never as evidence of no effect;
  * nothing says "SE(3)-equivariant": the default model is a structure-free contact-graph
    model, and only the 3D coordinate arms are SE(3)-invariant.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from . import style as S
from .audit_figs import SPLIT_SHORT, split_fold_leakage
from .loaders import (SPLIT_VARIANTS, Ctx, MissingArtifact, effect_table, find_runs, load_suite,
                      need, read_json, read_parquet, score_frame, split_docs)

_RANK = {"v2": 0, "v1": 1, "nll": 2, "mlp": 2, "other": 3}


def ordered(models) -> list[str]:
    """Our models first (v2, v1, controls), then everything else alphabetically."""
    return sorted(models, key=lambda m: (_RANK[S.entity(m)], str(m)))


def _handles(models, markers) -> list:
    return [Line2D([0], [0], color=S.color_of(m), lw=1.4, marker=markers[m], ms=5,
                   mfc=S.color_of(m), mec=S.SURFACE, mew=1.0, label=S.display(m)) for m in models]


def _span(*arrays, pad: float = 0.06):
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    d = (hi - lo) or 1.0
    return lo - pad * d, hi + pad * d


def _si(v: float, _=None) -> str:
    """1.5e6 -> '1.5M' for log-axis parameter counts."""
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suffix}"
    return f"{v:g}"


def _ci_note(doc: dict) -> str:
    return (f"95% intervals are cluster-bootstrap percentile intervals resampled on group_used "
            f"({doc.get('reps', '?')} resamples)")


# --- F3: accuracy vs split ------------------------------------------------------------
def build_f3(ctx: Ctx) -> list[dict]:
    docs = split_docs(ctx)
    present = [v for v in SPLIT_VARIANTS if v in docs]
    if len(present) < 2:
        raise MissingArtifact(f"F3 needs scorer outputs for at least two splits under "
                              f"{ctx.results / 'scores'} (found {present})")
    frames = pd.concat([score_frame(docs[v], v) for v in present], ignore_index=True)
    models = [m for m in ordered(frames["model"].unique())
              if all(((frames["split"] == v) & (frames["model"] == m)).any() for v in present)]
    if not models:
        raise MissingArtifact("F3: no model is scored on every available split")
    sources = [ctx.results / "scores" / f"{v}.json" for v in present]

    leak = None
    try:
        leak, _ = split_fold_leakage(ctx.release)
    except (MissingArtifact, ValueError, OSError):
        pass

    S.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True)
    xs = np.arange(len(present))
    markers = {m: S.MARKERS[i % len(S.MARKERS)] for i, m in enumerate(models)}
    panels = (("pooled", "Pooled over all test mutants"), ("per-protein median", "Median over test proteins"))
    for ax, (group, title) in zip(axes, panels):
        sub = frames[(frames["group"] == group) & (frames["metric"] == "spearman")]
        for k, m in enumerate(models):
            off = (k - (len(models) - 1) / 2) * 0.07
            d = sub[sub["model"] == m].set_index("split").reindex(present)
            color = S.color_of(m)
            ax.plot(xs + off, d["value"].to_numpy(), color=color, lw=1.4, zorder=3)
            for x, (_, r) in zip(xs + off, d.iterrows()):
                S.interval(ax, x, r["lo"], r["hi"], color, horizontal=False)
                S.dot(ax, x, r["value"], color, marker=markers[m])
        lo, hi = _span(sub["lo"].fillna(sub["value"]), sub["hi"].fillna(sub["value"]), sub["value"])
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.45, len(present) - 0.55)
        ax.set_ylabel("Spearman ρ")
        ax.set_title(title)
    labels = []
    for v in present:
        lab = SPLIT_SHORT[v]
        if leak is not None and (leak["split"] == v).any():
            med = float(leak.loc[leak["split"] == v, "median_nearest_train_tm_max"].iloc[0])
            lab += f"\nnearest-train\nTM {med:.2f}"
        labels.append(lab)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("split (more strict →)")
    S.panel_label(axes[0], "a")
    S.panel_label(axes[1], "b")
    fig.legend(handles=_handles(models, markers), loc="lower center", ncol=min(4, len(models)),
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.33, wspace=0.28, top=0.88, left=0.08, right=0.98)

    meta = pd.DataFrame([{"split": v, "n_rows": docs[v].get("n_rows"), "n_proteins": docs[v].get("n_proteins"),
                          "n_clusters": docs[v].get("n_clusters"), "reps": docs[v].get("reps")} for v in present])
    data = frames.merge(meta, on="split")
    if leak is not None:
        data = data.merge(leak[["split", "median_nearest_train_tm_max"]], on="split", how="left")
    names_txt = ", ".join(SPLIT_SHORT[v] for v in present)
    rows_txt = ", ".join(f"{docs[v].get('n_rows', 0):,} rows" for v in present)
    caption = (
        f"Accuracy of each model on splits of increasing strictness ({names_txt}). "
        f"(a) Spearman correlation pooled over all test mutants; (b) median over test proteins of the per-protein "
        f"Spearman correlation. The two are separate metrics and are not comparable. {_ci_note(docs[present[0]])}. "
        f"Each split retrains the model and has its own test set ({rows_txt}), so differences mix model and test-set "
        f"composition. The x-axis annotation is the median over structured test domains of the highest TM-score "
        f"to any training domain.")
    return [S.save_figure(fig, ctx.figures, "F3_accuracy_vs_split", data=data, sources=sources,
                          demo=ctx.demo, caption=caption)]


# --- F4: competitor comparison --------------------------------------------------------
def _primary(ctx: Ctx):
    docs = split_docs(ctx)
    if not docs:
        raise MissingArtifact(f"no scorer output under {ctx.results / 'scores'}")
    split = ctx.primary_split if ctx.primary_split in docs else next(iter(docs))
    return split, docs[split], ctx.results / "scores" / f"{split}.json"


def build_f4(ctx: Ctx) -> list[dict]:
    split, doc, src = _primary(ctx)
    fr = score_frame(doc, split)
    pooled = fr[(fr["group"] == "pooled") & (fr["metric"] == "spearman")].set_index("model")
    per = fr[(fr["group"] == "per-protein median") & (fr["metric"] == "spearman")].set_index("model")
    if pooled.empty:
        raise MissingArtifact(f"{src} lists no models")
    models = list(pooled["value"].sort_values(ascending=True, na_position="first").index)

    S.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.34 * len(models) + 1.5), sharey=True)
    best_other = max((m for m in models if S.entity(m) == "other"), key=lambda m: pooled.loc[m, "value"], default=None)
    for ax, grp, title in zip(axes, (pooled, per),
                              (f"Pooled over {doc.get('n_rows', 0):,} test mutants",
                               f"Median over {int(per['n_proteins'].max()) if per['n_proteins'].notna().any() else '?'} test proteins")):
        for yi, m in enumerate(models):
            r = grp.loc[m]
            color = S.color_of(m)
            S.interval(ax, yi, r["lo"], r["hi"], color)
            S.dot(ax, r["value"], yi, color)
            if S.entity(m) == "v2" or m == best_other:
                x_text = r["hi"] if np.isfinite(r["hi"]) else r["value"]
                ax.text(x_text + 0.014, yi, f"{r['value']:.3f}", va="center", ha="left", fontsize=6.6, color=S.INK2)
        lo, hi = _span(grp["lo"].fillna(grp["value"]), grp["hi"].fillna(grp["value"]), pad=0.1)
        ax.set_xlim(lo, hi + 0.06 * (hi - lo))
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Spearman ρ  [95% CI]")
        ax.set_title(title)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([S.display(m) for m in models])
    S.panel_label(axes[0], "a")
    S.panel_label(axes[1], "b")
    fig.subplots_adjust(wspace=0.12, left=0.22, right=0.98, bottom=0.2)

    data = fr[fr["metric"] == "spearman"].assign(n_rows=doc.get("n_rows"), n_clusters=doc.get("n_clusters"))
    caption = (
        f"Competitor comparison on the {SPLIT_SHORT.get(split, split)} split ({doc.get('n_rows', 0):,} test mutants, "
        f"{doc.get('n_proteins', '?')} proteins, {doc.get('n_clusters', '?')} clusters), all models scored on the same rows "
        f"by one scorer. (a) Spearman correlation pooled over all mutants; (b) median over proteins of the per-protein "
        f"Spearman correlation. Pooled and per-protein-median values are different quantities and must not be compared "
        f"with each other or with published figures of the other kind. {_ci_note(doc)}.")
    return [S.save_figure(fig, ctx.figures, "F4_competitors", data=data, sources=[src],
                          demo=ctx.demo, caption=caption)]


# --- T2: headline table ---------------------------------------------------------------
def build_t2(ctx: Ctx) -> list[dict]:
    split, doc, src = _primary(ctx)
    models = ordered((doc.get("models") or {}).keys())
    if not models:
        raise MissingArtifact(f"{src} lists no models")

    def ci(e, key):
        v = e.get(f"{key}_ci95") or [None, None]
        return [np.nan if x is None else float(x) for x in v]

    num, tex_rows = [], []
    first = models[0]
    for m in models:
        e = doc["models"][m]
        row = {"model": m}
        cells = {"Model": S.display(m)}
        for key, lab in (("pearson", "Pooled Pearson r"), ("spearman", "Pooled Spearman ρ"), ("rmse", "Pooled RMSE")):
            lo, hi = ci(e, key)
            row.update({f"pooled_{key}": e.get(key), f"pooled_{key}_lo": lo, f"pooled_{key}_hi": hi})
            cells[lab] = S.fmt_ci(e.get(key), lo, hi)
        for key, lab in (("pearson", "Per-protein median Pearson r"), ("spearman", "Per-protein median Spearman ρ")):
            lo, hi = ci(e, f"per_protein_median_{key}")
            row.update({f"per_protein_median_{key}": e.get(f"per_protein_median_{key}"),
                        f"per_protein_median_{key}_lo": lo, f"per_protein_median_{key}_hi": hi})
            cells[lab] = S.fmt_ci(e.get(f"per_protein_median_{key}"), lo, hi)
        row["n_proteins_scored"] = e.get("n_proteins_scored")
        d = e.get("pearson_delta_vs_first")
        dlo, dhi = (e.get("pearson_delta_ci95") or [None, None])
        row.update({"pooled_pearson_delta_vs_first": d, "delta_lo": dlo, "delta_hi": dhi})
        cells[f"Δ pooled Pearson vs {S.display(first)}"] = ("—" if m == first or d is None
                                                            else S.fmt_ci(d, dlo, dhi))
        num.append(row)
        tex_rows.append(cells)
    caption = (f"Headline comparison on the {SPLIT_SHORT.get(split, split)} split ({doc.get('n_rows', 0):,} test mutants, "
               f"{doc.get('n_proteins', '?')} proteins). Pooled metrics and per-protein-median metrics are separate "
               f"column groups.")
    note = (f"{_ci_note(doc)}. Per-protein medians use proteins with at least 3 mutations. The last column is the paired "
            f"difference to the first model on the same resamples.")
    return [S.save_table(pd.DataFrame(num), ctx.tables, "T2_headline", caption=caption, label="tab:headline",
                         tex_df=pd.DataFrame(tex_rows), sources=[src], demo=ctx.demo, note=note,
                         col_format="l" + "r" * 6)]


# --- F5: calibration ------------------------------------------------------------------
CAL_LABEL = {"full": "Holo-GeoN v2 (beta-NLL)", "standard_nll": "Holo-GeoN v2, Gaussian NLL",
             "v1_baseline": "Holo-GeoN v1 (Gaussian NLL)"}
CAL_ORDER = ("full", "standard_nll", "v1_baseline")


def build_f5(ctx: Ctx) -> list[dict]:
    runs = find_runs(ctx.results / "calibration", "calibration.json")
    by_arm: dict[str, list[tuple[int, dict]]] = {}
    sources = []
    for arm, seed, p in runs:
        cal = read_json(p).get("calibration")
        if cal:
            by_arm.setdefault(arm, []).append((seed, cal))
            sources.append(p)
    arms = [a for a in CAL_ORDER if a in by_arm] or sorted(by_arm)[:3]
    if not arms:
        raise MissingArtifact(f"no calibration.json under {ctx.results / 'calibration'}")

    rows = []
    for arm in arms:
        for seed, c in sorted(by_arm[arm]):
            row = {"arm": arm, "seed": seed, "ece": c.get("expected_calibration_error"),
                   "max_coverage_gap": c.get("max_coverage_gap"), "mean_sigma": c.get("mean_predicted_sigma"),
                   "residual_std": c.get("residual_std"), "gaussian_nll": c.get("gaussian_nll"),
                   "well_calibrated": c.get("well_calibrated"), "n": c.get("n")}
            row.update({f"coverage_{k}": v for k, v in (c.get("coverage_observed") or {}).items()})
            rows.append(row)
    data = pd.DataFrame(rows)

    S.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3))
    has_curve = False
    for arm in arms:
        color = S.color_of(arm)
        label = f"{CAL_LABEL.get(arm, S.display(arm))}  (ECE {data.loc[data['arm'] == arm, 'ece'].mean():.3f})"
        curves = [c["reliability"] for _, c in by_arm[arm] if c.get("reliability")]
        if curves:
            has_curve = True
            grid = np.asarray(curves[0]["nominal"], float)
            emp = np.vstack([np.asarray(c["empirical"], float) for c in curves])
            ax1.fill_between(grid, emp.min(0), emp.max(0), color=color, alpha=0.12, linewidth=0, zorder=2)
            ax1.plot(grid, emp.mean(0), color=color, lw=1.5, zorder=3, label=label)
        levels = sorted((c.get("coverage_observed") or {}).keys(), key=lambda k: float(k.strip("%")))
        nominal = [float(k.strip("%")) / 100 for k in levels]
        obs = np.array([[c["coverage_observed"][k] for k in levels] for _, c in by_arm[arm]], float)
        for r in obs:
            ax2.plot(nominal, r, linestyle="none", marker="o", ms=3, color=color, alpha=0.35, zorder=2)
        ax2.plot(nominal, obs.mean(0), color=color, lw=1.5, marker="o", ms=5, mfc=color, mec=S.SURFACE,
                 mew=1.0, zorder=3, label=None if has_curve else label)
    for ax in (ax1, ax2):
        ax.plot([0, 1], [0, 1], color=S.MUTED, lw=0.8, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("nominal probability level")
    ax1.set_ylabel("empirical frequency of PIT ≤ level")
    ax1.set_title("PIT reliability, all test mutants")
    ax2.set_xlabel("nominal central-interval level")
    ax2.set_ylabel("observed coverage")
    ax2.set_title("Interval coverage (seeds as dots)")
    if not has_curve:
        ax1.text(0.5, 0.5, "no PIT curve in these\ncalibration.json files", ha="center", va="center",
                 color=S.MUTED, fontsize=7)
    S.panel_label(ax1, "a")
    S.panel_label(ax2, "b")
    h, l = (ax1 if has_curve else ax2).get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.3, wspace=0.3, left=0.09, right=0.98, top=0.9)
    caption = ("Calibration of the predictive intervals. (a) Empirical frequency of the probability-integral-transform "
               "value at each nominal level (mean over seeds, band = min to max); a calibrated model lies on the gray "
               "diagonal. (b) Observed coverage of central intervals at 50, 80, 90 and 95%. ECE is the mean absolute "
               "gap of curve (a). Intervals are the model's own Gaussian sigma, before any variance rescaling.")
    return [S.save_figure(fig, ctx.figures, "F5_calibration", data=data, sources=sources,
                          demo=ctx.demo, caption=caption)]


# --- F6/F7 and T3/T4: paired effects --------------------------------------------------
def _is_true(v) -> bool:
    return isinstance(v, (bool, np.bool_)) and bool(v)


def _reading(r) -> str:
    n = int(r["n_seeds"]) if np.isfinite(r["n_seeds"]) else 0
    if not np.isfinite(r["p_holm"]):
        return f"untested ({n} seeds)"
    if _is_true(r["significant"]):
        return "lower than full" if r["delta"] < 0 else "higher than full"
    if np.isfinite(r["mde"]) and abs(r["delta"]) < r["mde"]:
        return f"cannot tell at {n} seeds"
    return "not significant"


def _effects(ctx: Ctx, suite: str, fig_stem: str, tab_stem: str, *, baseline_name: str, order=None,
             labels: dict | None = None, tab_label: str, tab_caption: str) -> list[dict]:
    doc = load_suite(ctx, suite)
    sources = [p for p in (ctx.results / suite / "summary.json", ctx.results / suite / "ablations.json") if p.is_file()]
    defs = (("test_pearson", "stability.pearson", "MegaScale test split"),
            ("ext_pearson", None, "FireProtDB external set"))
    panels = []
    for key, alt, title in defs:
        t = effect_table(doc, key)
        if t.empty and alt:
            t = effect_table(doc, alt)
        if not t.empty:
            t = t.assign(reading=t.apply(_reading, axis=1))
            panels.append((key, title, t))
    if not panels:
        raise MissingArtifact(f"{suite}: no paired comparison against 'full' in {sources[0] if sources else suite}")
    lab = {**S.ARM_LABEL, **(labels or {})}

    def rows_of(t):
        if order:
            t = t.set_index("arm").reindex([a for a in order if a in set(t["arm"])]).reset_index()
            return t
        return t.sort_values("delta").reset_index(drop=True)

    S.apply_style()
    n_rows = max(len(t) for _, _, t in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels) + 1.9, 0.34 * n_rows + 1.9),
                             sharey=True, squeeze=False)
    for ax, (key, title, t) in zip(axes[0], panels):
        t = rows_of(t)
        n_seeds = int(np.nanmax(t["n_seeds"].astype(float))) if len(t) else 0
        for i, r in t.reset_index(drop=True).iterrows():
            yi = len(t) - 1 - i
            if np.isfinite(r["mde"]):
                ax.plot([-r["mde"], r["mde"]], [yi, yi], color=S.GRID, lw=6.5, solid_capstyle="butt", zorder=1)
            S.interval(ax, yi, r["lo"], r["hi"], S.INK2)
            tested = np.isfinite(r["p_holm"])
            S.dot(ax, r["delta"], yi, S.SERIES[0] if tested else S.MUTED, hollow=not _is_true(r["significant"]))
        ax.axvline(0, color=S.MUTED, lw=0.8, zorder=2)
        ax.set_ylim(-0.7, len(t) - 0.3)
        ax.grid(axis="y", visible=False)
        lo, hi = _span(t["lo"].fillna(t["delta"]), t["hi"].fillna(t["delta"]), -t["mde"].fillna(0), t["mde"].fillna(0))
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"Δ pooled Pearson r (arm − {baseline_name})")
        ax.set_title(f"{title}\npaired over {n_seeds} seeds")
    names = []
    for r in rows_of(panels[0][2]).itertuples():
        names.append(lab.get(r.arm, r.arm) + (" †" if not np.isfinite(r.p_holm) else ""))
    axes[0][0].set_yticks(range(len(names)))
    axes[0][0].set_yticklabels(names[::-1])
    for k, ax in enumerate(axes[0]):
        S.panel_label(ax, "abc"[k])
    handles = [
        Line2D([0], [0], marker="o", ms=5, mfc=S.SERIES[0], mec=S.SURFACE, mew=1, lw=0, label="significant after Holm"),
        Line2D([0], [0], marker="o", ms=5, mfc=S.SURFACE, mec=S.SERIES[0], mew=1.3, lw=0, label="not significant"),
        Line2D([0], [0], color=S.GRID, lw=6.5, label="smaller than the detectable effect (80% power)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.24, wspace=0.08, left=0.3, right=0.98, top=0.82)

    # table
    first = panels[0][2]
    ref_mean, ref_sd = first["full_mean"].iloc[0], first["full_sd"].iloc[0]
    recs, tex_rows = [], []
    ref = {"arm": "full", "label": f"{baseline_name} (reference)", "n_seeds": int(first["n_seeds"].max())}
    for key, title, t in panels:
        ref[f"{key}_mean"], ref[f"{key}_sd"] = t["full_mean"].iloc[0], t["full_sd"].iloc[0]
    recs.append(ref)
    for r in rows_of(first).itertuples():
        row = {"arm": r.arm, "label": lab.get(r.arm, r.arm), "n_seeds": r.n_seeds}
        for key, title, t in panels:
            m = t[t["arm"] == r.arm]
            if m.empty:
                continue
            m = m.iloc[0]
            row.update({f"{key}_mean": m["arm_mean"], f"{key}_sd": m["arm_sd"], f"{key}_delta": m["delta"],
                        f"{key}_ci_lo": m["lo"], f"{key}_ci_hi": m["hi"], f"{key}_p_holm": m["p_holm"],
                        f"{key}_mde": m["mde"], f"{key}_significant": m["significant"], f"{key}_reading": m["reading"]})
        recs.append(row)
    tab = pd.DataFrame(recs)

    def pm(m, s):
        return "n/a" if pd.isna(m) else (f"{m:.4f} ± {s:.4f}" if not pd.isna(s) else f"{m:.4f}")

    for r in recs:
        cells = {"Arm": r["label"], "Seeds": str(int(r["n_seeds"]))}
        for key, title, _ in panels:
            tag = "test" if key == "test_pearson" else "external"
            cells[f"{tag.capitalize()} Pearson r"] = pm(r.get(f"{key}_mean"), r.get(f"{key}_sd"))
            if r["arm"] == "full":
                cells[f"Δ {tag} [95% CI]"] = "—"
                cells[f"p {tag} (Holm)"] = "—"
            else:
                cells[f"Δ {tag} [95% CI]"] = S.fmt_ci(r.get(f"{key}_delta"), r.get(f"{key}_ci_lo"), r.get(f"{key}_ci_hi"), 4)
                p = r.get(f"{key}_p_holm")
                cells[f"p {tag} (Holm)"] = "—" if pd.isna(p) else f"{p:.3f}"
        first_key = panels[0][0]
        cells["Reading (test)"] = "reference" if r["arm"] == "full" else r.get(f"{first_key}_reading", "n/a")
        tex_rows.append(cells)
    fam = first["family_size"].dropna()
    caption_fig = (
        f"Effect of each change relative to {baseline_name}, paired over seeds (same seeds in every arm). Dots are the mean "
        f"paired difference in pooled Pearson r, whiskers the 95% t-interval across seeds; filled dots are significant after "
        f"Holm-Bonferroni correction over the {int(fam.iloc[0]) if len(fam) else '?'} comparisons of the declared family. "
        f"The gray bar spans ± the smallest effect this design could detect at 80% power: an estimate inside it cannot be "
        f"told from zero at this many seeds, which is not evidence of no effect. † = fewer than 3 seeds, untested.")
    note = ("Δ = arm − reference on the test split, paired by seed; interval = paired t-interval over seeds; p-values are "
            "two-sided paired t-tests with Holm-Bonferroni correction over the declared family. 'cannot tell' = not "
            "significant and |Δ| below the detectable effect at 80% power.")
    rec_f = S.save_figure(fig, ctx.figures, fig_stem, data=tab, sources=sources, demo=ctx.demo, caption=caption_fig)
    rec_t = S.save_table(tab, ctx.tables, tab_stem, caption=tab_caption, label=tab_label, tex_df=pd.DataFrame(tex_rows),
                         sources=sources, demo=ctx.demo, note=note, col_format="l" + "r" * (len(tex_rows[0]) - 1))
    return [rec_f, rec_t]


def build_f6(ctx: Ctx) -> list[dict]:
    return _effects(ctx, "ablations", "F6_ablation_forest", "T3_ablations", baseline_name="Holo-GeoN v2",
                    tab_label="tab:ablations",
                    tab_caption="Architecture and loss ablations, paired over seeds against the full Holo-GeoN v2.")


def build_f7(ctx: Ctx) -> list[dict]:
    order = ["str_only", "both_streams", "str_topology_only", "true_contact_edges"]
    return _effects(ctx, "geometry", "F7_geometry_arms", "T4_geometry", baseline_name="A  predicted contacts",
                    order=order, tab_label="tab:geometry",
                    tab_caption=("Geometry arms B-E against arm A (predicted contacts, structure-free), identical split, "
                                 "readout, loss and seeds. Arms B-D use SE(3)-invariant 3D features."))


# --- F8: epistasis --------------------------------------------------------------------
EPI_ARMS = ("additive_only", "potts_no_gate", "contact_gated_egeo", "true_contact_oracle")
EPI_LABEL = {"additive_only": "additive", "potts_no_gate": "pairwise\n(ungated)",
             "contact_gated_egeo": "contact-\ngated E_geo", "true_contact_oracle": "true-contact\noracle"}
EPI_COLOR = {"additive_only": S.MUTED, "potts_no_gate": S.MUTED, "contact_gated_egeo": S.SERIES[0],
             "true_contact_oracle": S.SERIES[2]}


def build_f8(ctx: Ctx) -> list[dict]:
    runs = [(a, s, p) for a, s, p in find_runs(ctx.results / "epistasis", "epistasis_metrics.json") if a in EPI_ARMS]
    if not runs:
        raise MissingArtifact(f"no epistasis_metrics.json under {ctx.results / 'epistasis'}")
    rows, sources = [], [p for _, _, p in runs]
    for arm, seed, p in runs:
        test = read_json(p).get("test") or {}
        for key, strat in (("dddg_contact_lt8A", "< 8 Å"), ("dddg_distant_ge8A", "≥ 8 Å"), ("dddg_epistasis", "all")):
            e = test.get(key)
            if e:
                rows.append({"arm": arm, "seed": seed, "stratum": strat, "n": e.get("n"),
                             "pearson": e.get("pearson"), "spearman": e.get("spearman"), "rmse": e.get("rmse")})
    data = pd.DataFrame(rows)
    if data.empty:
        raise MissingArtifact("epistasis_metrics.json files carry no dddG metrics (need >= 3 double mutants per stratum)")

    scatter = None
    for arm, seed, p in sorted(runs, key=lambda r: (r[0] != "contact_gated_egeo", r[1])):
        pp = p.parent / "pred_epistasis_test.parquet"
        if arm == "contact_gated_egeo" and pp.is_file():
            scatter = (seed, read_parquet(pp))
            sources.append(pp)
            break

    S.apply_style()
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.1, 2.3], wspace=0.5, left=0.07, right=0.98, bottom=0.2, top=0.82)
    if scatter is not None:
        seed, pr = scatter
        rng = np.random.default_rng(0)
        ok = np.isfinite(pr["y_epi"]) & np.isfinite(pr["pred_epi"])
        lo, hi = _span(pr.loc[ok, "y_epi"], pr.loc[ok, "pred_epi"])   # one scale for both strata
        for k, (name, sel) in enumerate((("< 8 Å", pr["dist_cb"] < 8.0), ("≥ 8 Å", pr["dist_cb"] >= 8.0))):
            ax = fig.add_subplot(gs[0, k])
            d = pr[sel & ok]
            if len(d) > 15000:
                d = d.iloc[rng.choice(len(d), 15000, replace=False)]
            ax.scatter(d["y_epi"], d["pred_epi"], s=2.5, color=S.SERIES[0], alpha=0.25, linewidths=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=S.MUTED, lw=0.8, zorder=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"contact-gated E_geo, {name}\n(n = {int(sel.sum()):,}, seed {seed})", fontsize=7.4)
            ax.set_xlabel("observed ΔΔΔG")
            if k == 0:
                ax.set_ylabel("predicted ΔΔΔG")
                S.panel_label(ax, "a")
    else:
        ax = fig.add_subplot(gs[0, 0:2])
        ax.text(0.5, 0.5, "no pred_epistasis_test.parquet\nfor contact_gated_egeo", ha="center", va="center",
                color=S.MUTED, fontsize=7)
        ax.set_axis_off()
    ax = fig.add_subplot(gs[0, 3])
    arms = [a for a in EPI_ARMS if a in set(data["arm"])]
    for i, arm in enumerate(arms):
        for off, strat, shape, hollow in ((-0.17, "< 8 Å", "o", False), (0.17, "≥ 8 Å", "s", True)):
            v = data[(data["arm"] == arm) & (data["stratum"] == strat)]["pearson"].dropna().to_numpy(float)
            if not len(v):
                continue
            x = i + off
            ax.plot([x] * len(v), v, linestyle="none", marker=shape, ms=3, color=EPI_COLOR[arm], alpha=0.4, zorder=2)
            S.dot(ax, x, float(v.mean()), EPI_COLOR[arm], hollow=hollow, size=6, marker=shape)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([EPI_LABEL[a] for a in arms], fontsize=6.6)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Pearson r of ΔΔΔG")
    ax.set_title("Epistasis term by arm and contact stratum\n(mean of seeds as large marker)")
    S.panel_label(ax, "b")
    ax.legend(handles=[Line2D([0], [0], marker="o", ms=5, mfc=S.INK2, mec=S.SURFACE, lw=0, label="< 8 Å"),
                       Line2D([0], [0], marker="s", ms=5, mfc=S.SURFACE, mec=S.INK2, mew=1.3, lw=0, label="≥ 8 Å")],
              loc="best", title="C-beta distance", title_fontsize=6.5, alignment="left")
    caption = ("Double-mutant epistasis (ΔΔΔG = ΔΔG_AB − ΔΔG_A − ΔΔG_B). (a) Predicted vs observed for the contact-gated "
               "E_geo arm, split by the C-beta distance of the two mutated sites. (b) Pearson r of the epistasis term for "
               "the four arms in each stratum; small markers are single seeds, the large marker their mean. The "
               "contact-gated arm uses the structure-free predicted contact graph; the oracle arm uses true contacts.")
    return [S.save_figure(fig, ctx.figures, "F8_epistasis", data=data, sources=sources, demo=ctx.demo, caption=caption)]


# --- F9 / T5: efficiency --------------------------------------------------------------
def build_f9(ctx: Ctx) -> list[dict]:
    """
    Needs <results>/efficiency/benchmark_speed.json::

        {"gpu": str, "torch": str, "dtype": "fp32", "repeats": int, "scan_lengths": [100, 300, 500],
         "models": {name: {"trainable_params": int,
                           "scan_seconds": {"100": s, "300": s, "500": s}, "peak_vram_gb": float}}}

    where ``name`` matches a model name in the scorer output. Latency is the full 20 x L
    scan of a protein of length L, on ONE GPU type for every model.
    """
    bsrc = need(ctx.results / "efficiency" / "benchmark_speed.json", "speed / VRAM benchmark")
    bench = read_json(bsrc)
    split, doc, ssrc = _primary(ctx)
    fr = score_frame(doc, split)
    acc = fr[(fr["group"] == "pooled") & (fr["metric"] == "spearman")].set_index("model")
    lengths = [str(x) for x in bench.get("scan_lengths", [500])]
    key_len = "500" if "500" in lengths else lengths[-1]
    models = [m for m in ordered(bench.get("models", {})) if m in acc.index]
    if not models:
        raise MissingArtifact("no model appears in both the speed benchmark and the scorer output")
    rows = []
    for m in models:
        b = bench["models"][m]
        rows.append({"model": m, "trainable_params": b.get("trainable_params"),
                     "scan_seconds": (b.get("scan_seconds") or {}).get(key_len), "scan_length": int(key_len),
                     "peak_vram_gb": b.get("peak_vram_gb"), "pooled_spearman": acc.loc[m, "value"],
                     "pooled_spearman_lo": acc.loc[m, "lo"], "pooled_spearman_hi": acc.loc[m, "hi"]})
    df = pd.DataFrame(rows)

    S.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, col, xl, fmt in ((axes[0], "trainable_params", "trainable parameters", _si),
                             (axes[1], "scan_seconds", f"full 20 × {key_len} scan time (s)", lambda v, _: f"{v:g}")):
        xv = df[col].astype(float)
        xv = xv[np.isfinite(xv) & (xv > 0)]
        for r in df.itertuples():
            x = getattr(r, col)
            if x is None or not np.isfinite(x):
                continue
            color = S.color_of(r.model)
            S.interval(ax, x, r.pooled_spearman_lo, r.pooled_spearman_hi, color, horizontal=False, lw=1.0)
            S.dot(ax, x, r.pooled_spearman, color, size=6)
            ax.annotate(S.display(r.model), (x, r.pooled_spearman), xytext=(5, 4), textcoords="offset points",
                        fontsize=6.4, color=S.INK2)
        ax.set_xscale("log")
        if len(xv):
            ax.set_xlim(float(xv.min()) / 2.5, float(xv.max()) * 6)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel(xl)
    axes[0].set_ylabel("pooled Spearman ρ")
    axes[0].set_title("Accuracy vs size")
    axes[1].set_title(f"Accuracy vs speed ({bench.get('gpu', 'GPU')})")
    S.panel_label(axes[0], "a")
    S.panel_label(axes[1], "b")
    fig.subplots_adjust(wspace=0.12, bottom=0.2, left=0.08, right=0.98, top=0.86)
    caption = (f"Efficiency. Pooled Spearman correlation on the {SPLIT_SHORT.get(split, split)} split against (a) trainable "
               f"parameters and (b) wall-clock time of a full 20 × {key_len} saturation scan, every model on the same "
               f"{bench.get('gpu', 'GPU')} ({bench.get('dtype', '?')}, median of {bench.get('repeats', '?')} repeats). "
               f"Whiskers are cluster-bootstrap 95% intervals of the accuracy.")
    rec_f = S.save_figure(fig, ctx.figures, "F9_efficiency", data=df, sources=[bsrc, ssrc], demo=ctx.demo, caption=caption)

    tex = pd.DataFrame({
        "Model": [S.display(m) for m in df["model"]],
        "Trainable params": [f"{int(x):,}" if pd.notna(x) else "n/a" for x in df["trainable_params"]],
        **{f"Scan L={L} (s)": [("n/a" if (bench['models'][m].get('scan_seconds') or {}).get(L) is None
                                else f"{bench['models'][m]['scan_seconds'][L]:.2f}") for m in df["model"]] for L in lengths},
        "Peak VRAM (GB)": [f"{x:.1f}" if pd.notna(x) else "n/a" for x in df["peak_vram_gb"]],
        "Pooled Spearman ρ": [S.fmt_ci(v, lo, hi) for v, lo, hi in
                              zip(df["pooled_spearman"], df["pooled_spearman_lo"], df["pooled_spearman_hi"])],
    })
    rec_t = S.save_table(df, ctx.tables, "T5_efficiency",
                         caption=f"Model size, saturation-scan latency and memory on {bench.get('gpu', 'one GPU')}.",
                         label="tab:efficiency", tex_df=tex, sources=[bsrc, ssrc], demo=ctx.demo,
                         note="All models on the same GPU type; latency is the median of repeated full scans.",
                         col_format="l" + "r" * (len(tex.columns) - 1))
    return [rec_f, rec_t]
