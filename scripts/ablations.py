"""
scripts/ablations.py
====================
One axis at a time, everything else held fixed.

Each arm retrains the same architecture with a single component removed, on the
same splits and the same budget, and reports the delta against the full model.
The arms:

  * ``no_mech_stream``  — drop the E_mech message-passing stream. This is the
    paper's distinguishing claim and it has never been tested against a real
    coding sequence: v1 measured it with a back-translated proxy transcript,
    which is a deterministic function of the protein and therefore carries no
    information, and duly found the stream slightly *harmful*
    (single-track r = 0.5309 vs dual-track r = 0.5268). With a real CDS the
    question is open again, and this arm is how it gets answered.
  * ``global_pool_readout`` — replace the site-anchored readout with the v1
    formulation: pool the whole graph and take a difference. This is the arm
    that tests whether anchoring at the mutated residue actually recovers the
    signal that a global pool dilutes to ~1/L.
  * ``no_geo_stream`` — drop the contact-graph stream, leaving the mechanistic
    one. Tests whether the spatial edges earn their place.
  * ``no_ssm`` — drop the selective state-space residue mixer.

A negative delta is a result, not a failure. The point of running these is to
be able to say which components earn their place; an ablation that only ever
confirms the architecture is decoration.

Statistics
----------
Round-2 review, R2-06: the previous criterion was
``abs(delta_mean) > stdev(arm) + stdev(full)``, applied to *every* numeric key
in the test metrics. It has no null model, no degrees of freedom and no
multiplicity control, and the reviewer measured an 8.3% per-comparison
false-positive rate at the 3 seeds this script itself recommended. With the
5-arm grid writing 80 uncorrected comparisons, the probability of at least one
spurious "component matters" flag under the global null was 0.999 - and the
paper's ablation table would have been written from that flag. It also compared
``target_std`` and ``pos_rate``, which are properties of the test split and
identical across arms by construction.

What replaces it:

  * **One pre-registered metric per task** (``ablation_metrics`` in
    ``configs/run.yaml``), not every key that happens to be numeric.
  * **A paired test across seeds.** Arms share seeds, so the comparison is
    paired: the per-seed difference removes the seed's own effect.
  * **Holm-Bonferroni** over the declared family of comparisons, which controls
    the family-wise error rate without assuming independence.
  * **A confidence interval on the delta**, reported whether or not the test
    rejects. "Not significant" and "no effect" are different statements and the
    interval is what separates them.

Example
-------
python -m scripts.ablations --stability-splits data/splits_megascale.parquet \
    --cache cache/esm2_650M --out runs/ablations
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.provenance import provenance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS: dict[str, list[str]] = {
    "full": [],
    # The paper's distinguishing contribution.
    "no_mech_stream": ["--no-mech-stream"],
    # The two arms the architecture's narrative actually rests on. Round-1
    # review, R1-27: neither existed. src/readout.py argues the site anchoring
    # is the fix for the 1/L dilution problem, and scripts/contact_sanity.py
    # calls the contact graph "the check that justifies building a graph at
    # all" - but the grid measured neither, so the two load-bearing claims had
    # no ablation behind them.
    "global_pool_readout": ["--global-pool-readout"],
    "no_geo_stream": ["--no-geo-stream"],
    "no_ssm": ["--no-ssm"],
}


#: Metrics that describe the DATA rather than the model. They are identical
#: across arms by construction (same split, same rows), so a "delta" in them is
#: either exactly 0 or a bug - never a result. R2-06 found them being written
#: into delta_vs_full rows.
DATA_PROPERTY_METRICS = {
    "n", "target_std", "pos_rate", "positive_fraction", "support_per_class",
}

#: Seeds below this cannot support a paired test: with 2 the t distribution has
#: 1 degree of freedom and the interval is uninformative. The script reports the
#: delta and refuses to attach a p-value.
MIN_SEEDS_FOR_TEST = 3


def paired_stats(arm: list[float], full: list[float]) -> dict:
    """
    Paired comparison of one arm against the full model across shared seeds.

    Arms are run at the same seeds, so seed ``s`` contributes the pair
    ``(arm_s, full_s)`` and the per-seed difference removes whatever that seed
    did to both. Returns the mean difference, a t-based 95% interval, and a
    two-sided p-value - or a null p-value with a stated reason when there are
    too few seeds to compute one.
    """
    n = min(len(arm), len(full))
    diffs = [a - f for a, f in zip(arm[:n], full[:n])]
    out: dict = {
        "delta_mean": statistics.mean(diffs) if diffs else None,
        "n_seeds": n,
        "arm_mean": statistics.mean(arm) if arm else None,
        "full_mean": statistics.mean(full) if full else None,
        "arm_sd": statistics.stdev(arm) if len(arm) > 1 else None,
        "full_sd": statistics.stdev(full) if len(full) > 1 else None,
    }
    if n < MIN_SEEDS_FOR_TEST:
        out.update({
            "p_value": None, "ci95_low": None, "ci95_high": None,
            "note": f"{n} seed(s): below the {MIN_SEEDS_FOR_TEST} needed for a "
                    f"paired test. Run more seeds before reading this row.",
        })
        return out

    sd = statistics.stdev(diffs)
    if sd == 0.0:
        # Identical every seed. A p-value is undefined; say so rather than
        # emitting 0 and letting it be read as overwhelming evidence.
        out.update({
            "p_value": None, "ci95_low": out["delta_mean"],
            "ci95_high": out["delta_mean"],
            "note": "the difference is identical at every seed, so its variance "
                    "is 0 and no t statistic is defined",
        })
        return out

    from scipy import stats

    t_stat, p = stats.ttest_rel(arm[:n], full[:n])
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    out.update({
        "p_value": float(p),
        "t_statistic": float(t_stat),
        "paired_diff_sd": sd,
        "ci95_low": out["delta_mean"] - half,
        "ci95_high": out["delta_mean"] + half,
    })
    return out


def minimum_detectable_effect(sd_diff: float, n: int, family: int,
                              alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Smallest true effect this comparison could have found, at ``power``.

    Without it, "not significant" is unreadable: it may mean the component
    does not matter, or it may mean the run had no chance of showing that it
    does. Uses a Bonferroni alpha (``alpha / family``) as the conservative
    stand-in for Holm's worst case, so the number never flatters the design.
    """
    from scipy import stats

    if n < 2 or sd_diff <= 0 or family < 1:
        return float("nan")
    t_alpha = stats.t.ppf(1.0 - alpha / (2 * family), n - 1)
    t_beta = stats.t.ppf(power, n - 1)
    return float((t_alpha + t_beta) * sd_diff / math.sqrt(n))


def holm_bonferroni(p_values: dict) -> dict:
    """
    Holm-Bonferroni step-down adjustment over a family of comparisons.

    Controls the family-wise error rate at the nominal alpha without assuming
    the comparisons are independent - which they are not here, since every arm
    is compared against the same full-model runs. Entries whose p-value is None
    are not part of the family and are not counted in ``m``.
    """
    live = {k: v for k, v in p_values.items() if v is not None}
    m = len(live)
    adjusted: dict = {k: None for k in p_values}
    running = 0.0
    for rank, (key, p) in enumerate(sorted(live.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def run_arm(name: str, flags: list[str], common: list[str], out_root: str,
            seed: int) -> dict | None:
    """Train one arm at one seed unless its metrics already exist."""
    out_dir = os.path.join(out_root, f"{name}_seed{seed}")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[ablations] {name}: already complete, reusing {metrics_path}")
    else:
        # src/seeding.py says determinism is "on for the ablation arms, where
        # reproducibility is the point". Round-2 review, R2-15: no caller ever
        # set it. It is set here, so the sentence is true.
        cmd = [sys.executable, "-m", "scripts.train", *common, *flags,
               "--out", out_dir, "--tag", f"{name}_seed{seed}", "--seed", str(seed),
               "--strict-determinism"]
        print(f"\n[ablations] === {name} (seed {seed}) ===\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"[ablations] {name}: FAILED (exit {result.returncode})")
            return None
    with open(metrics_path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--stability-splits", default="data/splits_megascale.parquet")
    ap.add_argument("--pathogenicity-splits", default="data/splits_clinvar.parquet")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="runs/ablations")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="run every arm at each seed; deltas are paired across "
                         "them. Round-2 review, R2-06: the default was a single "
                         "seed, which cannot support any test - the shipped "
                         "pipeline emitted deltas to four decimals with a null "
                         "verdict attached")
    args = ap.parse_args()

    cfg = load_config(args.config)

    common = ["--config", args.config,
              "--stability-splits", args.stability_splits,
              "--pathogenicity-splits", args.pathogenicity_splits]
    for flag, value in (("--cache", args.cache), ("--device", args.device),
                        ("--epochs", args.epochs), ("--max-rows", args.max_rows)):
        if value is not None:
            common += [flag, str(value)]

    os.makedirs(args.out, exist_ok=True)
    # arm -> seed -> metrics
    results: dict[str, dict[int, dict]] = {}
    for name in args.arms:
        results[name] = {}
        for seed in args.seeds:
            res = run_arm(name, ARMS[name], common, args.out, seed)
            if res:
                results[name][seed] = res

    def series(arm: str, task: str, metric: str) -> list[float]:
        out = []
        for res in results.get(arm, {}).values():
            v = ((res.get("test") or {}).get(task) or {}).get(metric)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    # Deltas against the full model, paired across seeds and corrected for
    # multiplicity (R2-06). The comparison family is declared BEFORE the numbers
    # are seen: one metric per task, from the config.
    declared = dict(cfg.get("ablation_metrics") or {})
    tasks_present = sorted({
        task for res in results.get("full", {}).values()
        for task in (res.get("test") or {})
    })
    family: list[tuple] = []
    skipped_tasks: list[str] = []
    for task in tasks_present:
        metric = declared.get(task)
        if not metric:
            skipped_tasks.append(task)
            continue
        if metric in DATA_PROPERTY_METRICS:
            raise SystemExit(
                f"ablation_metrics[{task!r}] = {metric!r} is a property of the "
                f"test split, not of the model. Every arm sees the same rows, so "
                f"its delta is 0 by construction. Choose a model metric."
            )
        family.append((task, metric))
    if skipped_tasks:
        print(f"[ablations] no pre-registered metric for {skipped_tasks}; "
              f"those tasks are not compared. Add them to `ablation_metrics` in "
              f"{args.config} if they should be.")
    if not family:
        raise SystemExit(
            "no comparison could be formed: `ablation_metrics` in the config "
            "names no metric for any task present in the results."
        )
    print(f"[ablations] pre-registered comparison family "
          f"({len(family)} metric(s) x {len(results) - 1} arm(s)): "
          + ", ".join(f"{t}.{m}" for t, m in family))

    deltas: dict[str, dict] = {}
    p_values: dict[str, float | None] = {}
    for name in results:
        if name == "full":
            continue
        arm_delta: dict[str, dict] = {}
        for task, metric in family:
            base = series("full", task, metric)
            arm = series(name, task, metric)
            if not base or not arm:
                continue
            stats_row = paired_stats(arm, base)
            arm_delta.setdefault(task, {})[metric] = stats_row
            p_values[f"{name}/{task}/{metric}"] = stats_row.get("p_value")
        deltas[name] = arm_delta

    # One Holm correction over the whole declared family, not per arm.
    adjusted = holm_bonferroni(p_values)
    n_tested = sum(1 for v in p_values.values() if v is not None)
    for name, tasks in deltas.items():
        for task, metrics in tasks.items():
            for metric, row in metrics.items():
                key = f"{name}/{task}/{metric}"
                row["p_value_holm"] = adjusted.get(key)
                row["family_size"] = n_tested
                row["significant_after_holm"] = (
                    None if adjusted.get(key) is None else adjusted[key] < 0.05
                )
                if row.get("paired_diff_sd") is not None and n_tested:
                    mde = minimum_detectable_effect(
                        row["paired_diff_sd"], row["n_seeds"], n_tested)
                    row["minimum_detectable_effect_80pct_power"] = mde
                    row["underpowered"] = bool(
                        row["significant_after_holm"] is False
                        and abs(row["delta_mean"]) < mde
                    )
                    if row["underpowered"]:
                        row["note"] = (
                            f"not significant, but this comparison could only "
                            f"have detected an effect of {mde:.4f} or larger at "
                            f"80% power. |delta| is {abs(row['delta_mean']):.4f}, "
                            f"so this row is not evidence that the component "
                            f"does not matter - it is evidence the run cannot "
                            f"tell. Add seeds, or narrow the comparison family."
                        )

    payload = {
        "provenance": provenance(),
        "seeds": args.seeds,
        "arms": {
            name: {str(seed): res for seed, res in runs.items()} or {"status": "failed"}
            for name, runs in results.items()
        },
        "delta_vs_full": deltas,
        "comparison_family": [f"{t}.{m}" for t, m in family],
        "family_size_tested": n_tested,
        "statistics": {
            "test": "two-sided paired t-test across shared seeds",
            "correction": "Holm-Bonferroni over the whole declared family",
            "alpha": 0.05,
            "min_seeds_for_test": MIN_SEEDS_FOR_TEST,
        },
        "reading": (
            "delta_mean is (ablated - full) on the test split, paired by seed. A "
            "component earns its place when removing it makes the metric worse. "
            "Read ci95_low/ci95_high first: they say how large the effect could "
            "be. significant_after_holm is the family-wise-corrected verdict at "
            "alpha=0.05; null means too few seeds to test, not 'no effect'. A "
            "non-significant delta with a tight interval around zero is evidence "
            "the component does not matter; a non-significant delta with a wide "
            "interval is evidence of nothing and needs more seeds. "
            "minimum_detectable_effect_80pct_power is the smallest true "
            "effect this many seeds could have found over this family; a "
            "non-significant delta smaller than it is marked underpowered "
            "and says nothing either way."
        ),
    }
    out_path = os.path.join(args.out, "ablations.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n[ablations] deltas vs full (test split, paired over "
          f"{len(args.seeds)} seed(s), Holm-corrected over {n_tested} "
          f"comparison(s)):")
    for name, tasks in deltas.items():
        for task, metrics in tasks.items():
            for metric, d in metrics.items():
                ci = (f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]"
                      if d.get("ci95_low") is not None else "[no interval]")
                verdict = {True: "SIGNIFICANT", False: "not significant",
                           None: "not tested"}[d["significant_after_holm"]]
                if d.get("underpowered"):
                    verdict += ", UNDERPOWERED"
                print(f"  {name:<22} {task:<14} {metric:<10} "
                      f"{d['delta_mean']:+.4f}  95% CI {ci}  ({verdict})")
                if d.get("underpowered"):
                    print(f"  {'':<22} -> could only have detected "
                          f"{d['minimum_detectable_effect_80pct_power']:.4f} or "
                          f"larger at 80% power over {n_tested} comparison(s)")
    if len(args.seeds) < MIN_SEEDS_FOR_TEST:
        print(f"[ablations] NOTE: {len(args.seeds)} seed(s) per arm, below the "
              f"{MIN_SEEDS_FOR_TEST} needed for a paired test, so every "
              f"p-value is null. Run with --seeds 42 43 44 45 46 before "
              f"reporting any delta - round 1 measured 0.030 test Pearson "
              f"between identical runs, which is larger than most of these "
              f"effects.")
    print(f"[ablations] wrote {out_path}")


if __name__ == "__main__":
    main()
