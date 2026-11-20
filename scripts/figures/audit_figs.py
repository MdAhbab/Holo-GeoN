"""
scripts/figures/audit_figs.py
=============================
F2 (fold leakage), the per-split fold-leakage table and T1 (dataset / split inventory),
built from the released audit files only: no model output is involved.

The per-split table is DERIVED here, from the released all-vs-all TM-align matrix and
the three split files, so the same measurement is applied to every split. Its id30 and
fold_tm60 rows are checked against the audit summaries that ship with the release
(`checks` in the returned records); a mismatch is reported, never hidden.

Reading the numbers (claims discipline)
---------------------------------------
* TM_max / TM_min are the larger / smaller of a pair's two length normalisations.
* At the conventional same-fold threshold (TM >= 0.5) the 373 structured MegaScale
  domains form ONE connected component holding 97% of them, so no split disjoint at
  that threshold exists. `fold_tm60` is disjoint only in the weaker sense that no
  test-train pair has TM_min >= 0.60; its test domains still reach TM_max up to 0.69.
  It is therefore called a fold-CLUSTERED split, never a fold-disjoint one.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import style as S
from .loaders import SPLIT_VARIANTS, Ctx, MissingArtifact, need, read_json, read_parquet

SPLIT_NAME = {"id30": "30% identity", "id25": "25% identity", "fold_tm60": "fold-clustered (TM_min < 0.60)"}
SPLIT_SHORT = {"id30": "30% identity", "id25": "25% identity", "fold_tm60": "fold-clustered"}


# --- TM-align matrix ------------------------------------------------------------------
def tm_matrices(release: Path):
    """(parents, TM_max matrix, TM_min matrix) over the structured MegaScale domains."""
    av = read_parquet(need(Path(release) / "splits/megascale_tm_all_vs_all.parquet",
                           "all-vs-all TM-align table"),
                      columns=["parent_i", "parent_j", "tm_max", "tm_min"])
    parents = sorted(set(av.parent_i) | set(av.parent_j))
    idx = {p: k for k, p in enumerate(parents)}
    n = len(parents)
    mx = np.full((n, n), np.nan)
    mn = np.full((n, n), np.nan)
    i = av.parent_i.map(idx).to_numpy()
    j = av.parent_j.map(idx).to_numpy()
    mx[i, j] = mx[j, i] = av.tm_max.to_numpy()
    mn[i, j] = mn[j, i] = av.tm_min.to_numpy()
    return parents, mx, mn


def fold_clusters(tm: np.ndarray, thr: float, groups=None) -> np.ndarray:
    """Connected-component labels when two domains are linked at tm >= thr.

    ``groups`` (one id per domain, matrix order) additionally links domains that share
    an id. The released fold clusters are TM_min >= 0.60 links united with the
    30%-identity sequence clusters: that reproduces the published 76 clusters and
    their sizes, whereas TM_min >= 0.60 alone gives 78.
    """
    adj = np.nan_to_num(tm, nan=0.0) >= thr
    if groups is not None:
        g = np.asarray(groups)
        adj = adj | (g[:, None] == g[None, :])
    np.fill_diagonal(adj, False)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def parent_table(release: Path, variant: str) -> pd.DataFrame:
    """parent name -> split and cluster id under one split variant (parents are the unit)."""
    release = Path(release)
    names = read_parquet(need(release / "tables/megascale_ddg_pairs.parquet", "MegaScale pair table"),
                         columns=["parent", "wt_seq"]).drop_duplicates()
    sp = read_parquet(need(release / f"splits/splits_megascale_{variant}.parquet", f"{variant} split"),
                      columns=["parent_seq", "split", "group_used"]).drop_duplicates()
    if sp["parent_seq"].duplicated().any():
        raise ValueError(f"{variant}: a parent sequence appears in more than one split or cluster, "
                         f"so the split is not parent-disjoint")
    return names.merge(sp, left_on="wt_seq", right_on="parent_seq", how="inner")[
        ["parent", "split", "group_used"]]


def parent_splits(release: Path, variant: str) -> dict:
    """{parent name: split} under one split variant."""
    t = parent_table(release, variant)
    return dict(zip(t["parent"], t["split"]))


def split_fold_leakage(release: Path):
    """Per split: how close is each structured test domain to its nearest training domain?"""
    parents, mx, mn = tm_matrices(release)
    idx = {p: k for k, p in enumerate(parents)}
    rows, details = [], {}
    for v in SPLIT_VARIANTS:
        if not (Path(release) / f"splits/splits_megascale_{v}.parquet").is_file():
            continue
        sof = parent_splits(release, v)
        test = [p for p in parents if sof.get(p) == "test"]
        train = [p for p in parents if sof.get(p) == "train"]
        if not test or not train:
            continue
        ti = np.array([idx[p] for p in test])
        ri = np.array([idx[p] for p in train])
        sub_max, sub_min = mx[np.ix_(ti, ri)], mn[np.ix_(ti, ri)]
        best_max, best_min = np.nanmax(sub_max, axis=1), np.nanmax(sub_min, axis=1)
        nearest = [train[k] for k in np.nanargmax(sub_max, axis=1)]
        details[v] = pd.DataFrame({"test_parent": test, "nearest_train_parent": nearest,
                                   "best_tm_max": best_max, "best_tm_min": best_min})
        n = len(test)
        rows.append({
            "split": v, "structured_test_domains": n, "structured_train_domains": len(train),
            "test_tm_max_ge_0.5": int((best_max >= 0.5).sum()),
            "test_tm_max_ge_0.7": int((best_max >= 0.7).sum()),
            "test_tm_min_ge_0.5": int((best_min >= 0.5).sum()),
            "test_tm_min_ge_0.6": int((best_min >= 0.6).sum()),
            "median_nearest_train_tm_max": float(np.median(best_max)),
            "max_nearest_train_tm_max": float(best_max.max()),
            "max_nearest_train_tm_min": float(best_min.max()),
        })
    if not rows:
        raise MissingArtifact(f"no split file under {Path(release) / 'splits'}")
    return pd.DataFrame(rows), details


def audit_checks(release: Path, leak: pd.DataFrame, n_clusters_060: int) -> list[dict]:
    """Compare the derived numbers with the audit summaries shipped in the release."""
    release = Path(release)
    checks: list[dict] = []

    def add(name, expected, got, tol=1e-3):
        ok = (abs(float(expected) - float(got)) <= tol)
        checks.append({"check": name, "expected": expected, "got": got, "ok": bool(ok)})

    p = release / "splits/megascale_tm_leakage_summary.json"
    row = leak[leak["split"] == "id30"]
    if p.is_file() and len(row):
        ref, r = read_json(p)["test_max_tm_vs_train"], row.iloc[0]
        add("id30 test domains with TM_max >= 0.5", ref["count_tm_ge_0.5"], r["test_tm_max_ge_0.5"], 0)
        add("id30 test domains with TM_max >= 0.7", ref["count_tm_ge_0.7"], r["test_tm_max_ge_0.7"], 0)
        add("id30 median nearest-train TM_max", ref["median"], r["median_nearest_train_tm_max"], 1e-3)
    p = release / "splits/megascale_tm_fold_audit_summary.json"
    if p.is_file():
        census = read_json(p)["fold_cluster_census"]
        c60 = census.get("tm_min_ge_0.60", {})
        if "n_clusters" in c60:
            add("clusters at TM_min >= 0.60", c60["n_clusters"], n_clusters_060, 0)
        row = leak[leak["split"] == "fold_tm60"]
        if "max_cross_test_train_tm_min" in c60 and len(row):
            add("fold_tm60 max cross test-train TM_min", c60["max_cross_test_train_tm_min"],
                row.iloc[0]["max_nearest_train_tm_min"], 1e-3)
    return checks


# --- F2 -------------------------------------------------------------------------------
def build_f2(ctx: Ctx) -> list[dict]:
    rel = ctx.release
    parents, mx, mn = tm_matrices(rel)
    leak, details = split_fold_leakage(rel)
    if "id30" not in details:
        raise MissingArtifact(f"{rel / 'splits/splits_megascale_id30.parquet'} is needed for panel (a)")
    src_summary = need(rel / "splits/megascale_tm_fold_audit_summary.json", "fold audit summary")
    census = read_json(src_summary)["fold_cluster_census"]
    d30 = details["id30"]
    pt = parent_table(rel, "id30")
    gmap = dict(zip(pt["parent"], pt["group_used"]))
    labels = fold_clusters(mn, 0.60, [gmap.get(p, p) for p in parents])
    n_cl = int(len(set(labels.tolist())))
    checks = audit_checks(rel, leak, n_cl)

    S.apply_style()
    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.86], wspace=0.5, left=0.07, right=0.975,
                          bottom=0.18, top=0.82)

    # (a) nearest training domain of each structured test domain, 30% identity split
    ax = fig.add_subplot(gs[0, 0])
    n = len(d30)
    for col, label, color in (("best_tm_max", "TM_max", S.SERIES[0]), ("best_tm_min", "TM_min", S.SERIES[1])):
        v = np.sort(d30[col].to_numpy())
        y = np.arange(1, n + 1) / n
        ax.step(np.r_[0.0, v, 1.0], np.r_[0.0, y, 1.0], where="post", color=color, lw=1.5,
                label=label, zorder=3)
    for thr in (0.5, 0.7):
        ax.axvline(thr, color=S.AXIS, lw=0.8, zorder=1)
    ax.text(0.31, 0.62, "gray lines: TM = 0.5\n(same fold) and 0.7\n(near-identical)", fontsize=6.3,
            color=S.MUTED, ha="left", va="top", linespacing=1.25)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("highest TM-score to any training domain")
    ax.set_ylabel("share of test domains (cumulative)")
    ax.set_title(f"Nearest training domain of {n}\nstructured test domains (30% identity)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    S.panel_label(ax, "a")

    # (b) how fast does the fold graph fragment as the linkage threshold rises?
    ax = fig.add_subplot(gs[0, 1])
    thr = [0.50, 0.55, 0.60, 0.65, 0.70]
    rows_b = []
    for kind, label, color in (("tm_max", "TM_max ≥ x", S.SERIES[0]),
                               ("tm_min", "TM_min ≥ x", S.SERIES[1])):
        xs, ys = [], []
        for t in thr:
            c = census.get(f"{kind}_ge_{t:.2f}")
            if not c:
                continue
            xs.append(t)
            ys.append(c["largest_cluster_frac"])
            rows_b.append({"panel": "b_collapse", "linkage": kind, "threshold": t,
                           "n_clusters": c["n_clusters"], "largest_cluster_frac": c["largest_cluster_frac"]})
        ax.plot(xs, ys, color=color, lw=1.5, marker="o", ms=5, mfc=color, mec=S.SURFACE, mew=1.0,
                label=label, zorder=3)
    c_max = census.get("tm_max_ge_0.50") or {}
    c_min = census.get("tm_min_ge_0.60") or {}
    if c_max:
        v = c_max["largest_cluster_frac"]
        ax.annotate(f"{v:.0%} of domains\nin one cluster", xy=(0.50, v), xytext=(0.525, v - 0.17),
                    fontsize=6.5, color=S.INK2, ha="left", va="center",
                    arrowprops={"arrowstyle": "-", "color": S.AXIS, "lw": 0.7, "shrinkA": 0, "shrinkB": 3})
    if c_min:
        v = c_min["largest_cluster_frac"]
        ax.annotate(f"TM_min ≥ 0.60:\n{c_min['n_clusters']} clusters, largest {v:.0%}\n(basis of the fold split)",
                    xy=(0.60, v), xytext=(0.585, v + 0.42), fontsize=6.5, color=S.INK2, ha="left",
                    va="center",
                    arrowprops={"arrowstyle": "-", "color": S.AXIS, "lw": 0.7, "shrinkA": 0, "shrinkB": 3})
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0.48, 0.72)
    ax.set_xticks(thr)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("linkage threshold x")
    ax.set_ylabel(f"largest fold cluster\n(share of {len(parents)} domains)")
    ax.set_title("Fold clusters fragment only at\nstricter thresholds")
    ax.legend(loc="upper right", title="link domains if", title_fontsize=6.5, alignment="left")
    S.panel_label(ax, "b")

    # (c) the 373 x 373 TM_max matrix, ordered by the TM_min >= 0.60 fold clusters
    ax = fig.add_subplot(gs[0, 2])
    sizes = pd.Series(labels).value_counts()
    order = sorted(range(len(parents)), key=lambda k: (-int(sizes[labels[k]]), int(labels[k]), k))
    mat = np.where(np.isnan(mx), 1.0, mx)[np.ix_(order, order)]
    im = ax.imshow(mat, cmap=S.BLUES, vmin=0.15, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{len(parents)} domains by fold cluster\n({n_cl} clusters at TM_min ≥ 0.60)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, shrink=0.78)
    cb.set_label("TM_max", fontsize=7)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, width=0.4, length=2)
    S.panel_label(ax, "c")

    a = d30.assign(panel="a_ecdf")[["panel", "test_parent", "nearest_train_parent", "best_tm_max", "best_tm_min"]]
    data = pd.concat([a, pd.DataFrame(rows_b)], ignore_index=True)

    r30 = leak[leak["split"] == "id30"].iloc[0]
    parts = [
        f"Fold leakage in MegaScale. (a) For each of the {int(r30['structured_test_domains'])} structured test "
        f"domains of the 30%-identity split, the highest TM-score to any training domain "
        f"(TM_max and TM_min are the larger and smaller of a pair's two length normalisations). "
        f"{int(r30['test_tm_max_ge_0.5'])}/{int(r30['structured_test_domains'])} test domains reach "
        f"TM_max >= 0.5 and {int(r30['test_tm_max_ge_0.7'])}/{int(r30['structured_test_domains'])} reach >= 0.7 "
        f"(median {r30['median_nearest_train_tm_max']:.2f}, maximum {r30['max_nearest_train_tm_max']:.2f}).",
        f"(b) Share of the {len(parents)} structured domains in the largest connected fold cluster as the linkage "
        f"threshold rises; at TM_max >= 0.50 one cluster holds {c_max.get('largest_cluster_frac', float('nan')):.1%}.",
        f"Clusters in (b) and (c) link two domains when their TM-score reaches the threshold or when they share a "
        f"30%-identity sequence cluster (TM_min >= 0.60 alone gives more, smaller clusters).",
        f"(c) All-vs-all TM_max, ordered by the {n_cl} clusters at TM_min >= 0.60, which define the fold-clustered split.",
    ]
    if (leak["split"] == "fold_tm60").any():
        rf = leak[leak["split"] == "fold_tm60"].iloc[0]
        parts.append(
            f"On that split no test-train pair reaches TM_min = 0.60 (maximum {rf['max_nearest_train_tm_min']:.4f}), "
            f"but test domains still reach TM_max up to {rf['max_nearest_train_tm_max']:.2f}; it is fold-clustered, "
            f"not fold-disjoint at the conventional TM >= 0.5.")
    rec = S.save_figure(
        fig, ctx.figures, "F2_fold_leakage", data=data, demo=ctx.demo, caption=" ".join(parts),
        sources=[rel / "splits/megascale_tm_all_vs_all.parquet", src_summary,
                 rel / "tables/megascale_ddg_pairs.parquet",
                 rel / "splits/splits_megascale_id30.parquet", rel / "splits/splits_megascale_fold_tm60.parquet"])
    rec["checks"] = checks
    return [rec]


# --- T1 -------------------------------------------------------------------------------
def build_t1(ctx: Ctx) -> list[dict]:
    rel = ctx.release
    man = read_json(need(rel / "release_manifest.json", "release manifest"))
    meta = {f["path"]: f["parquet"] for f in man["files"] if "parquet" in f}
    leak, _ = split_fold_leakage(rel)

    rows_a = []
    srcs = [rel / "release_manifest.json", rel / "splits/megascale_tm_all_vs_all.parquet",
            rel / "tables/megascale_ddg_pairs.parquet"]
    for v in SPLIT_VARIANTS:
        path = rel / f"splits/splits_megascale_{v}.parquet"
        if not path.is_file() or not (leak["split"] == v).any():
            continue
        srcs.append(path)
        sp = read_parquet(path, columns=["parent_seq", "split", "group_used"])
        cnt = sp.groupby("split").size()
        test = sp[sp["split"] == "test"]
        lk = leak[leak["split"] == v].iloc[0]
        kmer = np.nan
        rp = rel / "splits/leakage_report_megascale_id25.json"
        if v == "id25" and rp.is_file():
            kmer = float(read_json(rp).get("heldout_8mer_jaccard_ge_0.30", np.nan))
            srcs.append(rp)
        rows_a.append({
            "split": v, "rows_train": int(cnt.get("train", 0)), "rows_val": int(cnt.get("val", 0)),
            "rows_test": int(cnt.get("test", 0)), "test_proteins": int(test["parent_seq"].nunique()),
            "test_clusters": int(test["group_used"].nunique()),
            "structured_test_domains": int(lk["structured_test_domains"]),
            "test_tm_max_ge_0.5": int(lk["test_tm_max_ge_0.5"]), "test_tm_max_ge_0.7": int(lk["test_tm_max_ge_0.7"]),
            "test_tm_min_ge_0.5": int(lk["test_tm_min_ge_0.5"]),
            "median_nearest_train_tm_max": round(float(lk["median_nearest_train_tm_max"]), 4),
            "max_nearest_train_tm_min": round(float(lk["max_nearest_train_tm_min"]), 4),
            "test_pairs_8mer_jaccard_ge_0.30": kmer,
        })
    a = pd.DataFrame(rows_a)
    tex_a = pd.DataFrame({
        "Split": [SPLIT_NAME[v] for v in a["split"]],
        "Rows train / val / test": [f"{r.rows_train:,} / {r.rows_val:,} / {r.rows_test:,}" for r in a.itertuples()],
        "Test proteins (clusters)": [f"{r.test_proteins} ({r.test_clusters})" for r in a.itertuples()],
        "TM_max ≥ 0.5": [f"{k}/{t}" for k, t in zip(a["test_tm_max_ge_0.5"], a["structured_test_domains"])],
        "TM_max ≥ 0.7": [f"{k}/{t}" for k, t in zip(a["test_tm_max_ge_0.7"], a["structured_test_domains"])],
        "Median nearest TM_max": a["median_nearest_train_tm_max"].map(lambda x: f"{x:.2f}"),
        "Max TM_min": a["max_nearest_train_tm_min"].map(lambda x: f"{x:.4f}"),
    })

    def m(path):
        return meta.get(path) or {}

    fp, fh = m("tables/fireprotdb_holdout.parquet"), m("tables/fireprotdb_hf25.parquet")
    s6 = m("tables/s669_holdout.parquet")
    dh, dd = m("tables/domainome_apca_holdout.parquet"), m("tables/domainome_apca_hf25.parquet")

    def n(d, k):
        return d.get(k, np.nan)

    rows_b = [
        {"set": "FireProtDB 2.0 (MegaScale rows removed)", "rows": n(fp, "rows"), "proteins": n(fp, "unique_parent"),
         "removed_proteins": np.nan, "removed_rows": np.nan, "metrics": "pooled and per-protein median"},
        {"set": "FireProt-HF25 (≤ 25% identity to MegaScale)", "rows": n(fh, "rows"), "proteins": n(fh, "unique_parent"),
         "removed_proteins": n(fp, "unique_parent") - n(fh, "unique_parent"),
         "removed_rows": n(fp, "rows") - n(fh, "rows"), "metrics": "pooled and per-protein median"},
        {"set": "S669 / Ssym+", "rows": n(s6, "rows"), "proteins": n(s6, "unique_parent"),
         "removed_proteins": np.nan, "removed_rows": np.nan, "metrics": "pooled and per-protein median"},
        {"set": "Human Domainome 1.0 (aPCA)", "rows": n(dh, "rows"), "proteins": n(dh, "unique_parent"),
         "removed_proteins": np.nan, "removed_rows": np.nan, "metrics": "Spearman only (not kcal/mol)"},
        {"set": "Domainome-HF25 (≤ 25% identity to MegaScale)", "rows": n(dd, "rows"), "proteins": n(dd, "unique_parent"),
         "removed_proteins": n(dh, "unique_parent") - n(dd, "unique_parent"),
         "removed_rows": n(dh, "rows") - n(dd, "rows"), "metrics": "Spearman only (not kcal/mol)"},
    ]
    b = pd.DataFrame(rows_b)

    def num(x):
        return "n/a" if pd.isna(x) else f"{int(x):,}"

    def pct(removed, base):
        return "—" if pd.isna(removed) or not base else f"{int(removed):,} ({removed / base:.1%})"

    base_p = [np.nan, n(fp, "unique_parent"), np.nan, np.nan, n(dh, "unique_parent")]
    base_r = [np.nan, n(fp, "rows"), np.nan, np.nan, n(dh, "rows")]
    tex_b = pd.DataFrame({
        "Evaluation set": b["set"], "Rows": b["rows"].map(num), "Proteins": b["proteins"].map(num),
        "Proteins removed": [pct(x, y) for x, y in zip(b["removed_proteins"], base_p)],
        "Rows removed": [pct(x, y) for x, y in zip(b["removed_rows"], base_r)],
        "Metrics": b["metrics"],
    })

    checks: list[dict] = []
    ap = rel / "splits/fireprotdb_homology_audit.parquet"
    if ap.is_file():
        aud = read_parquet(ap)
        got = int(aud.loc[aud["seq_identity"] >= 0.25, "query"].nunique())
        exp = int(b.loc[1, "removed_proteins"])
        checks.append({"check": "FireProtDB proteins >= 25% identity: audit table vs table difference",
                       "expected": exp, "got": got, "ok": got == exp})
        srcs.append(ap)
    dp = rel / "splits/domainome_homology_audit.parquet"
    if dp.is_file():
        da = read_parquet(dp)
        got = int((da["max_id_vs_megascale"] >= 0.25).sum())
        exp = int(b.loc[4, "removed_proteins"])
        checks.append({"check": "Domainome domains >= 25% identity: audit table vs table difference",
                       "expected": exp, "got": got, "ok": got == exp})
        srcs.append(dp)

    note_a = ("Every test protein has a structure. TM_max and TM_min are the larger and smaller of a pair's two length "
              "normalisations; the TM columns count test proteins whose most similar structured training protein reaches "
              "the stated TM_max, out of all structured test proteins, and the last column is the highest TM_min of any "
              "test-train pair. Clusters are the ids the split was grouped by (sequence clusters, or fold clusters for "
              "the fold-clustered split). The fold-clustered split is disjoint only in that no test-train pair reaches "
              "TM_min = 0.60; at the conventional TM >= 0.5 most test proteins still have a training relative.")
    note_b = "Sign convention: positive = stabilising (ddG = dG_mut - dG_wt). aPCA abundance is rank-only."
    cap_a = "MegaScale single-mutant splits and their sequence- and fold-level leakage audit."
    cap_b = "External evaluation sets and the homology filter against MegaScale."
    ra = S.save_table(a, ctx.tables, "T1a_splits", caption=cap_a, label="tab:splits", tex_df=tex_a,
                      sources=srcs, demo=ctx.demo, note=note_a, col_format="lrrrrrr")
    rb = S.save_table(b, ctx.tables, "T1b_external", caption=cap_b, label="tab:external", tex_df=tex_b,
                      sources=srcs, demo=ctx.demo, note=note_b, col_format="lrrrrl")
    ra["checks"] = checks
    return [ra, rb]
