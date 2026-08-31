"""
scripts/calibration.py
======================
Does the predicted sigma mean anything, and is reversibility real?

**Calibration.** The interface draws a confidence interval. v1 drew one too —
from ``heuristic_confidence(ddg) = ddg +/- (0.4 + 0.12*|ddg|)``, a formula with
no relationship to the model, applied in *full* mode as well as demo mode. This
model predicts its own variance and trains it under Gaussian NLL, so the band is
only honest if it is measured. Reported here:

  * **Coverage** at nominal 50 / 80 / 90 / 95%: the fraction of test targets
    inside ``mu +/- z sigma``. A well-calibrated model matches the nominal rate;
    over-coverage means timid intervals, under-coverage means overconfident ones.
  * **PIT / ECE.** The probability integral transform ``Phi((y - mu) / sigma)``
    is uniform on [0, 1] exactly when the predictive distribution is correct.
    Expected calibration error is the mean absolute gap between the empirical
    and nominal quantiles.
  * **Sharpness.** Mean predicted sigma. Calibration alone is trivial to game by
    predicting enormous intervals, so it is only meaningful beside sharpness.

**Reversibility.** The head makes ``ddG(a->b) = -ddG(b->a)`` exact *for a fixed
context*. That is a structural identity, and it is checked as one. The
substantive question is different: does the model agree with itself when the
mutant protein is encoded as its own parent? That requires a second encoder pass
and is the honest test of thermodynamic consistency, so it is measured
separately here rather than being claimed from the algebra.

Example
-------
python -m scripts.calibration --checkpoint runs/main/model_best.pt \
    --splits data/splits_megascale.parquet --cache cache/esm2_650M \
    --out runs/main --encode-mutants
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data import MutationDataset, build_sample_graph, collate_graphs, move_batch
from src.device import describe_device
from src.encoder import ESM_DIMS, EncodingStore, EsmEncoder
from src.features import mechanistic_features
from src.model import HoloGeoN
from src.substitution import aa_index

Z_FOR_NOMINAL = {0.50: 0.674490, 0.80: 1.281552, 0.90: 1.644854, 0.95: 1.959964}


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def calibration_report(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> dict:
    """Coverage, PIT-based ECE and sharpness for a Gaussian predictive model."""
    sigma = np.clip(sigma, 1e-6, None)
    coverage = {
        f"{int(p * 100)}%": float(np.mean(np.abs(y - mu) <= z * sigma))
        for p, z in Z_FOR_NOMINAL.items()
    }
    pit = _normal_cdf((y - mu) / sigma)
    grid = np.linspace(0.01, 0.99, 99)
    empirical = np.array([np.mean(pit <= q) for q in grid])
    ece = float(np.mean(np.abs(empirical - grid)))
    nll = float(np.mean(0.5 * (np.log(2 * np.pi * sigma ** 2) + ((y - mu) / sigma) ** 2)))
    return {
        "n": int(len(y)),
        "coverage_observed": coverage,
        "coverage_nominal": {f"{int(p * 100)}%": p for p in Z_FOR_NOMINAL},
        "expected_calibration_error": ece,
        "max_calibration_error": float(np.max(np.abs(empirical - grid))),
        "mean_predicted_sigma": float(np.mean(sigma)),
        "median_predicted_sigma": float(np.median(sigma)),
        "residual_std": float(np.std(y - mu)),
        "gaussian_nll": nll,
        "interpretation": (
            "observed coverage should track nominal; mean_predicted_sigma close "
            "to residual_std indicates the spread is right as well as the rate"
        ),
    }


@torch.no_grad()
def predict_split(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus, sigmas, ys = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        ddg, logvar = model.ddg(batch)
        mus.append(ddg.reshape(-1).float().cpu().numpy())
        sigmas.append(torch.exp(0.5 * logvar).reshape(-1).float().cpu().numpy())
        ys.append(batch["y"].reshape(-1).float().cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigmas), np.concatenate(ys)


@torch.no_grad()
def reversibility_check(
    model, df: pd.DataFrame, store: EncodingStore, cfg, device,
    encoder: EsmEncoder | None, limit: int,
) -> dict:
    """
    Encode each mutant as its own parent and compare ddG(mt->wt) with
    -ddG(wt->mt).

    Rows whose mutant protein is not cached are skipped unless ``encoder`` is
    supplied to embed them on the fly. Skipped rows are counted, never ignored.
    """
    rows = df.iloc[:limit]
    forward, reverse = [], []
    skipped = 0
    cfg_d = cfg.as_dict()

    for _, row in tqdm(list(rows.iterrows()), desc="reversibility"):
        parent = row["parent_seq"]
        site = int(row["site"])
        mutant = parent[:site] + row["mt_aa"] + parent[site + 1:]

        if store.has(mutant):
            mut_rec = store.get(mutant)
        elif encoder is not None:
            mut_rec = encoder.encode(mutant, top_k=cfg.contact_top_k,
                                     min_sep=cfg.contact_min_sep)
            mut_rec["mech"] = mechanistic_features(mutant, None, len(mutant)).half()
            store.put(mut_rec)
        else:
            skipped += 1
            continue

        fwd_g = build_sample_graph(store.get(parent), site, cfg.window, cfg_d)
        fwd_g.update({"wt_idx": aa_index(row["wt_aa"]), "mt_idx": aa_index(row["mt_aa"]),
                      "y": 0.0, "sigma": float("nan")})
        rev_g = build_sample_graph(mut_rec, site, cfg.window, cfg_d)
        rev_g.update({"wt_idx": aa_index(row["mt_aa"]), "mt_idx": aa_index(row["wt_aa"]),
                      "y": 0.0, "sigma": float("nan")})

        f_batch = move_batch(collate_graphs([fwd_g]), device)
        r_batch = move_batch(collate_graphs([rev_g]), device)
        forward.append(float(model.ddg(f_batch)[0].item()))
        reverse.append(float(model.ddg(r_batch)[0].item()))

    if not forward:
        return {"n": 0, "skipped_no_cached_mutant": skipped,
                "note": "no mutant encodings available; rerun with --encode-mutants"}

    f = np.asarray(forward)
    r = np.asarray(reverse)
    violation = np.abs(f + r)
    return {
        "n": int(len(f)),
        "skipped_no_cached_mutant": skipped,
        "mean_violation_kcal_per_mol": float(violation.mean()),
        "median_violation_kcal_per_mol": float(np.median(violation)),
        "max_violation_kcal_per_mol": float(violation.max()),
        "pearson_forward_vs_negated_reverse": float(np.corrcoef(f, -r)[0, 1])
        if len(f) > 1 else float("nan"),
        "note": (
            "this re-encodes the mutant as its own parent, so it measures "
            "representational consistency; the head's algebraic antisymmetry at "
            "a fixed context is exact by construction and is asserted in tests"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--encode-mutants", action="store_true",
                    help="embed mutant proteins on the fly for the reversibility check")
    ap.add_argument("--reversibility-limit", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={"cache": args.cache, "device": args.device})
    os.makedirs(args.out, exist_ok=True)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.esm_model, 1280))
    saved = state.get("config", {})
    model = HoloGeoN(
        esm_dim=esm_dim,
        hidden=saved.get("hidden", cfg.hidden),
        out_dim=saved.get("out_dim", cfg.out_dim),
        gat_heads=saved.get("gat_heads", cfg.gat_heads),
        dropout=saved.get("dropout", cfg.dropout),
        use_ssm=saved.get("use_ssm", cfg.use_ssm),
        use_mech_stream=saved.get("use_mech_stream", cfg.use_mech_stream),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    df = pd.read_parquet(args.splits)
    part = df[df.split == args.split]
    if part.empty:
        raise SystemExit(f"{args.splits}: split {args.split!r} is empty")

    loader = DataLoader(
        MutationDataset(part, store, cfg, target="target"),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_graphs,
    )
    mu, sigma, y = predict_split(model, loader, device)
    report = {"checkpoint": args.checkpoint, "splits": args.splits, "split": args.split,
              "calibration": calibration_report(mu, sigma, y)}

    encoder = EsmEncoder(cfg.esm_model, device=str(device)) if args.encode_mutants else None
    report["reversibility"] = reversibility_check(
        model, part, store, cfg, device, encoder, args.reversibility_limit)

    out_path = os.path.join(args.out, "calibration.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    cal = report["calibration"]
    print("[calibration] observed vs nominal coverage:")
    for k in cal["coverage_nominal"]:
        print(f"    {k:>4}: observed {cal['coverage_observed'][k]:.3f}")
    print(f"[calibration] ECE={cal['expected_calibration_error']:.4f}  "
          f"mean sigma={cal['mean_predicted_sigma']:.3f}  "
          f"residual std={cal['residual_std']:.3f}")
    print(f"[calibration] reversibility: {report['reversibility']}")
    print(f"[calibration] wrote {out_path}")


if __name__ == "__main__":
    main()
