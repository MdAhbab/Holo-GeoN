"""
scripts/final_scores.py
=======================
Print the model's final held-out scores, in one place.

These are TEST-split numbers, computed once on the selected checkpoint at the end
of training — not the validation numbers that early stopping chose on, and not a
maximum over epochs. Use ``scripts/show_progress.py`` for the per-epoch
validation history; use this for the numbers that go in a report.

Every value is read from a run artifact and each is labelled with the file it
came from. A stage that has not run prints ``not run`` rather than being omitted,
so a missing control is visible instead of silently absent.

Usage
-----
python -m scripts.final_scores
python -m scripts.final_scores --run runs/main
python -m scripts.final_scores --json          # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NA = "not run"


def read(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def fmt(value, spec: str = ".4f") -> str:
    if value is None:
        return "     --"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=os.path.join("runs", "main"))
    ap.add_argument("--json", action="store_true", help="emit JSON instead")
    args = ap.parse_args()

    metrics = read(os.path.join(args.run, "metrics.json"))
    calib = read(os.path.join(args.run, "calibration.json"))
    ext = read(os.path.join(args.run, "external_fireprotdb.json"))
    contacts = read(os.path.join("results", "contacts", "contact_sanity.json"))
    base_s = read(os.path.join("runs", "baselines", "stability", "baselines.json"))
    base_p = read(os.path.join("runs", "baselines", "pathogenicity", "baselines.json"))
    abl = read(os.path.join("runs", "ablations", "ablations.json"))

    if metrics is None:
        raise SystemExit(f"no metrics.json under {args.run}; train first")

    test = metrics.get("test", {})
    stab = test.get("stability", {})
    path = test.get("pathogenicity", {})
    three = stab.get("three_state", {})

    out = {
        "run": args.run,
        "best_epoch": metrics.get("best_epoch"),
        "trained_tasks": metrics.get("trained_tasks"),
        "untrained_heads": metrics.get("untrained_heads"),
        "test_stability": stab,
        "test_pathogenicity": path,
        "external": (ext or {}).get("metrics"),
        "calibration": (calib or {}).get("calibration"),
        "contacts": (contacts or {}).get("precision_vs_structure"),
    }
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return

    print(f"\n{'=' * 66}")
    print(f"  HOLO-GeoN — final held-out scores   ({args.run})")
    print(f"{'=' * 66}")
    print(f"  selected checkpoint : epoch {metrics.get('best_epoch')} of "
          f"{metrics.get('epochs_configured', '?')}")
    print(f"  trained tasks       : {metrics.get('trained_tasks')}")
    print(f"  untrained heads     : {metrics.get('untrained_heads')}")

    print(f"\n  STABILITY (ddG) — test split, n={stab.get('n', '?')}")
    print(f"     Pearson r          {fmt(stab.get('pearson'))}")
    print(f"     Spearman rho       {fmt(stab.get('spearman'))}")
    print(f"     RMSE (kcal/mol)    {fmt(stab.get('rmse'))}")
    print(f"     MAE                {fmt(stab.get('mae'))}")
    print(f"     R^2                {fmt(stab.get('r2'))}")
    print(f"     target sd          {fmt(stab.get('target_std'))}")

    print(f"\n  THREE-STATE — test split")
    print(f"     macro-F1           {fmt(three.get('macro_f1'))}")
    print(f"     accuracy           {fmt(three.get('accuracy'))}")
    if three.get("recall_per_class"):
        for name, value in three["recall_per_class"].items():
            print(f"       recall {name:<14} {fmt(value)}")

    print(f"\n  PATHOGENICITY (ClinVar) — test split, n={path.get('n', '?')}")
    print(f"     AUROC              {fmt(path.get('auroc'))}")
    print(f"     AUPRC              {fmt(path.get('auprc'))}")
    print(f"     accuracy           {fmt(path.get('accuracy'))}")
    print(f"     F1                 {fmt(path.get('f1'))}")
    print(f"     MCC                {fmt(path.get('mcc'))}")
    print(f"     positive rate      {fmt(path.get('pos_rate'))}")

    print(f"\n  EXTERNAL TRANSFER (FireProtDB, no MegaScale rows or sequences)")
    if ext:
        m = ext["metrics"]
        print(f"     rows / proteins    {ext.get('rows_scored')} / "
              f"{ext.get('distinct_proteins')}")
        print(f"     Pearson r          {fmt(m.get('pearson'))}")
        print(f"     Spearman rho       {fmt(m.get('spearman'))}")
        print(f"     RMSE               {fmt(m.get('rmse'))}")
        print(f"     R^2                {fmt(m.get('r2'))}")
    else:
        print(f"     {NA}")

    print(f"\n  UNCERTAINTY CALIBRATION")
    if calib:
        c = calib.get("calibration", {})
        vs = calib.get("variance_scale", {})
        print(f"     well calibrated    {c.get('well_calibrated')}")
        print(f"     ECE                {fmt(c.get('expected_calibration_error'))}")
        print(f"     variance scale     {fmt(vs.get('scale'))}  (fitted on val)")
        after = vs.get("calibration_after_scaling", {})
        if after:
            print(f"     ECE after scaling  "
                  f"{fmt(after.get('expected_calibration_error'))}")
    else:
        print(f"     {NA}")

    print(f"\n  CONTACT EDGES (precision@L vs ground-truth C-beta maps)")
    pvs = (contacts or {}).get("precision_vs_structure")
    if pvs:
        print(f"     proteins with a structure   {pvs.get('n_with_structure')}")
        print(f"     contact head       P@L {fmt(pvs.get('contact_head_precision_at_L'))}"
              f"   P@L/5 {fmt(pvs.get('contact_head_precision_at_L5'))}")
        print(f"     proximity baseline P@L {fmt(pvs.get('baseline_precision_at_L'))}"
              f"   P@L/5 {fmt(pvs.get('baseline_precision_at_L5'))}")
        print(f"     random edges       P@L {fmt(pvs.get('random_precision_at_L'))}"
              f"   P@L/5 {fmt(pvs.get('random_precision_at_L5'))}")
    else:
        print(f"     {NA}")

    print(f"\n  CONTROLS (baselines)")
    for label, payload, key in (("stability", base_s, "pearson"),
                                ("pathogenicity", base_p, "auroc")):
        if not payload:
            print(f"     {label:<16} {NA}")
            continue
        for name, m in sorted((payload.get("baselines") or payload).items()):
            if isinstance(m, dict) and key in m:
                print(f"     {label}/{name:<22} {key}={fmt(m.get(key))}")

    print(f"\n  ABLATIONS           {'present' if abl else NA}")
    print(f"\n{'=' * 66}")
    print("  Read the EXTERNAL row next to the test row: the gap between them is")
    print("  what the model loses off its training distribution.")
    print(f"{'=' * 66}\n")


if __name__ == "__main__":
    main()
