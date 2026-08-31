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

Example
-------
python -m scripts.ablations --stability-splits data/splits_megascale.parquet \
    --cache cache/esm2_650M --out runs/ablations
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def run_arm(name: str, flags: list[str], common: list[str], out_root: str,
            seed: int) -> dict | None:
    """Train one arm at one seed unless its metrics already exist."""
    out_dir = os.path.join(out_root, f"{name}_seed{seed}")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[ablations] {name}: already complete, reusing {metrics_path}")
    else:
        cmd = [sys.executable, "-m", "scripts.train", *common, *flags,
               "--out", out_dir, "--tag", f"{name}_seed{seed}", "--seed", str(seed)]
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="run every arm at each seed and report mean +/- sd; "
                         "a single-seed delta is not separable from run-to-run "
                         "noise, which round-1 measured at 0.030 test Pearson")
    args = ap.parse_args()

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

    # Deltas against the full model, aggregated over seeds. A single-seed delta
    # is not separable from run-to-run noise (round-1 measured 0.030 test
    # Pearson between identical runs), so the spread travels with the number.
    deltas: dict[str, dict] = {}
    tasks_metrics: set = set()
    for res in results.get("full", {}).values():
        for task, metrics in (res.get("test") or {}).items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in {"n"}:
                    tasks_metrics.add((task, k))

    for name in results:
        if name == "full":
            continue
        arm_delta: dict[str, dict] = {}
        for task, metric in sorted(tasks_metrics):
            base = series("full", task, metric)
            arm = series(name, task, metric)
            if not base or not arm:
                continue
            d = statistics.mean(arm) - statistics.mean(base)
            arm_delta.setdefault(task, {})[metric] = {
                "delta_mean": d,
                "arm_mean": statistics.mean(arm),
                "arm_sd": statistics.stdev(arm) if len(arm) > 1 else None,
                "full_mean": statistics.mean(base),
                "full_sd": statistics.stdev(base) if len(base) > 1 else None,
                "n_seeds": len(arm),
                "separable_from_noise": (
                    None if len(arm) < 2 or len(base) < 2
                    else abs(d) > (statistics.stdev(arm) + statistics.stdev(base))
                ),
            }
        deltas[name] = arm_delta

    payload = {
        "seeds": args.seeds,
        "arms": {
            name: {str(seed): res for seed, res in runs.items()} or {"status": "failed"}
            for name, runs in results.items()
        },
        "delta_vs_full": deltas,
        "reading": (
            "delta_mean is (ablated - full) on the test split, averaged over "
            "seeds. A component earns its place when removing it makes the "
            "metric worse. separable_from_noise compares |delta| against the "
            "combined run-to-run spread; with one seed it is null, and a delta "
            "smaller than the spread is not evidence either way."
        ),
    }
    out_path = os.path.join(args.out, "ablations.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n[ablations] deltas vs full (test split, mean over "
          f"{len(args.seeds)} seed(s)):")
    for name, tasks in deltas.items():
        for task, metrics in tasks.items():
            for metric, d in list(metrics.items())[:3]:
                sd = d["arm_sd"]
                spread = f" +/- {sd:.4f}" if sd is not None else ""
                sep = {True: "separable", False: "WITHIN NOISE",
                       None: "single seed"}[d["separable_from_noise"]]
                print(f"  {name:<22} {task:<14} {metric:<10} "
                      f"{d['delta_mean']:+.4f}{spread}  ({sep})")
    if len(args.seeds) < 2:
        print("[ablations] NOTE: one seed per arm. Run with --seeds 42 43 44 "
              "before reporting any delta - a single run cannot be told apart "
              "from run-to-run variation.")
    print(f"[ablations] wrote {out_path}")


if __name__ == "__main__":
    main()
