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
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS: dict[str, list[str]] = {
    "full": [],
    "no_mech_stream": ["--no-mech-stream"],
    "no_ssm": ["--no-ssm"],
}


def run_arm(name: str, flags: list[str], common: list[str], out_root: str) -> dict | None:
    """Train one arm unless its metrics already exist, then return them."""
    out_dir = os.path.join(out_root, name)
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[ablations] {name}: already complete, reusing {metrics_path}")
    else:
        cmd = [sys.executable, "-m", "scripts.train", *common, *flags,
               "--out", out_dir, "--tag", name]
        print(f"\n[ablations] === {name} ===\n$ {' '.join(cmd)}")
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
    args = ap.parse_args()

    common = ["--config", args.config,
              "--stability-splits", args.stability_splits,
              "--pathogenicity-splits", args.pathogenicity_splits]
    for flag, value in (("--cache", args.cache), ("--device", args.device),
                        ("--epochs", args.epochs), ("--max-rows", args.max_rows)):
        if value is not None:
            common += [flag, str(value)]

    os.makedirs(args.out, exist_ok=True)
    results = {name: run_arm(name, ARMS[name], common, args.out) for name in args.arms}

    # Deltas against the full model, per task and metric.
    full = results.get("full")
    deltas: dict[str, dict] = {}
    if full:
        for name, res in results.items():
            if name == "full" or not res:
                continue
            arm_delta: dict[str, dict] = {}
            for task, metrics in (res.get("test") or {}).items():
                base = (full.get("test") or {}).get(task, {})
                arm_delta[task] = {
                    k: float(metrics[k] - base[k])
                    for k in metrics
                    if isinstance(metrics.get(k), (int, float))
                    and isinstance(base.get(k), (int, float))
                    and k not in {"n"}
                }
            deltas[name] = arm_delta

    payload = {
        "arms": {k: (v or {"status": "failed"}) for k, v in results.items()},
        "delta_vs_full": deltas,
        "reading": (
            "delta is (ablated - full) on the test split. A component earns its "
            "place when removing it makes the metric worse."
        ),
    }
    out_path = os.path.join(args.out, "ablations.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n[ablations] deltas vs full (test split):")
    for name, tasks in deltas.items():
        for task, metrics in tasks.items():
            head = {k: round(v, 4) for k, v in list(metrics.items())[:4]}
            print(f"  {name:<16} {task:<14} {head}")
    print(f"[ablations] wrote {out_path}")


if __name__ == "__main__":
    main()
