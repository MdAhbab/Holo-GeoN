"""
scripts/show_progress.py
========================
Print every validation metric for a run, per epoch, at any time — including
while the run is still going.

The per-epoch training log prints one headline metric per task, because a line
carrying twelve numbers per task is unreadable in a scrolling log. Everything
else is computed and stored regardless: it goes into ``history`` inside
``last.pt`` after every epoch, and into ``metrics.json`` when the run finishes.
This reads the checkpoint, so the full picture is available mid-run rather than
only at the end.

Usage
-----
python -m scripts.show_progress                      # runs/main
python -m scripts.show_progress --run runs/baseline_wd1e-4
python -m scripts.show_progress --compare runs/baseline_wd1e-4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def load(run: str) -> tuple[list, dict]:
    path = os.path.join(run, "last.pt")
    if not os.path.exists(path):
        raise SystemExit(f"no checkpoint at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("history", []), state


def nan(value, default=float("nan")) -> float:
    return default if value is None else value


def table(history: list, state: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"epochs={len(history)}  best_epoch={state.get('best_epoch')}  "
          f"best_score={nan(state.get('best_score')):.4f}  "
          f"patience={state.get('patience')}")
    head = (f"{'ep':>3} | {'Pearson':>7} {'Spear':>7} {'RMSE':>7} {'MAE':>7} "
            f"{'R2':>6} {'sig/err':>7} {'gap':>6} | {'3st F1':>7} | "
            f"{'AUROC':>7} {'AUPRC':>7} {'F1':>6}")
    print(head)
    print("-" * len(head))
    for r in history:
        s = r["val"].get("stability", {})
        p = r["val"].get("pathogenicity", {})
        t = s.get("three_state", {})
        ratio = s.get("rmse_over_predicted_sigma")
        if ratio is None and s.get("mean_predicted_sigma"):
            ratio = s["rmse"] / s["mean_predicted_sigma"]
        gap = r.get("train", {}).get("generalisation_gap_rmse")
        star = " *" if r["epoch"] == state.get("best_epoch") else ""
        print(f"{r['epoch']:3d} | {nan(s.get('pearson')):7.4f} "
              f"{nan(s.get('spearman')):7.4f} {nan(s.get('rmse')):7.4f} "
              f"{nan(s.get('mae')):7.4f} {nan(s.get('r2')):6.3f} "
              f"{nan(ratio):6.2f}x {nan(gap):6.3f} | "
              f"{nan(t.get('macro_f1')):7.4f} | {nan(p.get('auroc')):7.4f} "
              f"{nan(p.get('auprc')):7.4f} {nan(p.get('f1')):6.4f}{star}")
    print("\n  sig/err = val RMSE / mean predicted sigma. 1.0 means the head's")
    print("            uncertainty is honest; growth means it is overconfident.")
    print("  gap     = val RMSE / train RMSE on a fixed 20k training subsample.")
    print("            1.0 means no train/val gap; growth means memorisation.")
    print("  *       = the epoch model_best.pt currently holds.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=os.path.join("runs", "main"))
    ap.add_argument("--compare", default=None,
                    help="a second run directory to print alongside")
    ap.add_argument("--epoch", type=int, default=None,
                    help="dump every recorded metric for one epoch")
    args = ap.parse_args()

    history, state = load(args.run)
    if args.epoch is not None:
        match = [r for r in history if r["epoch"] == args.epoch]
        if not match:
            raise SystemExit(f"epoch {args.epoch} is not in {args.run}")
        import json
        print(json.dumps(match[0], indent=2, default=str))
        return

    table(history, state, args.run)
    if args.compare:
        h2, s2 = load(args.compare)
        table(h2, s2, args.compare)


if __name__ == "__main__":
    main()
