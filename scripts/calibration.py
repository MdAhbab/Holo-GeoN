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
from src.codon_tables import cai_matched_codon, get_cai_weights
from src.features import mechanistic_features
from src.model import HoloGeoN, model_kwargs_from_config
from src.seeding import seed_everything
from src.substitution import aa_index

Z_FOR_NOMINAL = {0.50: 0.674490, 0.80: 1.281552, 0.90: 1.644854, 0.95: 1.959964}

#: Thresholds for calling the predictive intervals calibrated. A 5-point gap at
#: any nominal level, or an ECE above 0.05, is enough to withhold the claim.
ECE_TOLERANCE = 0.05
COVERAGE_TOLERANCE = 0.05


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _substituted_cds(
    parent_rec: dict, site: int, mt_aa: str
) -> tuple[str | None, float]:
    """
    The parent's coding sequence with the codon at ``site`` replaced.

    The replacement is the synonymous codon whose CAI weight is closest to the
    codon it replaces (:func:`cai_matched_codon`). Round-2 review, R1-11
    residual: this used ``preferred_codon``, whose weight is 1.0 by definition,
    so every mutant gained a systematically better codon than a real variant
    would carry - a one-sided ~+0.215 CAI shift at the one site the
    reversibility check is about.

    Returns ``(cds, induced_cai_shift)``. The shift is returned rather than
    assumed to be zero, because it cannot be driven to zero in general: the
    amino acid changed, so the reachable codon weights changed with it. The
    caller reports the residual.

    ``cds`` is None when the parent carries no CDS, in which case both
    directions are mech-free and the comparison stays symmetric anyway.
    """
    cds = parent_rec.get("cds")
    if not cds or len(cds) < (site + 1) * 3:
        return None, 0.0
    parent_codon = cds[site * 3: (site + 1) * 3].upper()
    weights = get_cai_weights("human")
    parent_weight = weights.get(parent_codon, 0.0)
    codon, weight = cai_matched_codon(mt_aa, parent_weight)
    if codon is None:
        return None, 0.0
    return cds[: site * 3] + codon + cds[(site + 1) * 3:], weight - parent_weight


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
    max_gap = float(np.max([abs(coverage[k] - p) for p, k in
                            zip(Z_FOR_NOMINAL, coverage)]))
    # Round-1 review, R1-14: the interface flipped "calibrated" on the mere
    # existence of this file, so a run measuring 11% coverage for a 95% band
    # reported calibrated intervals. The verdict is computed here, from the
    # numbers, and the interface reads it rather than inferring one.
    well_calibrated = bool(
        ece <= ECE_TOLERANCE and max_gap <= COVERAGE_TOLERANCE
    )
    return {
        "n": int(len(y)),
        "coverage_observed": coverage,
        "coverage_nominal": {f"{int(p * 100)}%": p for p in Z_FOR_NOMINAL},
        "expected_calibration_error": ece,
        "max_calibration_error": float(np.max(np.abs(empirical - grid))),
        "max_coverage_gap": max_gap,
        "mean_predicted_sigma": float(np.mean(sigma)),
        "median_predicted_sigma": float(np.median(sigma)),
        "residual_std": float(np.std(y - mu)),
        "gaussian_nll": nll,
        "well_calibrated": well_calibrated,
        "tolerances": {"ece": ECE_TOLERANCE, "coverage_gap": COVERAGE_TOLERANCE},
        "interpretation": (
            "observed coverage should track nominal; mean_predicted_sigma close "
            "to residual_std indicates the spread is right as well as the rate. "
            "well_calibrated is the verdict the interface reads before it may "
            "present an interval as calibrated"
        ),
    }


def fit_variance_scale(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """
    Scalar ``a`` minimising Gaussian NLL for ``sigma -> a * sigma``.

    Round-1 review, R1-14: nothing in the repository could act on a measured
    miscalibration - the script reported and stopped. For a Gaussian the optimal
    scale has a closed form, ``a^2 = mean( ((y - mu) / sigma)^2 )``, so a single
    number fitted on the VALIDATION split repairs systematic over- or
    under-confidence. It is fitted on val and applied at serving time; fitting
    it on the split it is then evaluated on would be circular.
    """
    z = (y - mu) / np.clip(sigma, 1e-6, None)
    return float(np.sqrt(np.mean(z ** 2)))


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
    encoder: EsmEncoder | None, limit: int, scratch: EncodingStore | None = None,
) -> dict:
    """
    Encode each mutant as its own parent and compare ddG(mt->wt) with
    -ddG(wt->mt).

    Two corrections from round-1 review, R1-11.

    **The mutant's mechanistic channels are derived from the parent's coding
    sequence**, with the one codon at the mutated site substituted, rather than
    being rebuilt with ``dna_seq=None``. Passing None set ``mech_valid = 0`` and
    left E_mech with self-loops only, so the forward direction ran through the
    dual-stream model and the reverse through an effectively single-stream one
    (the reviewer measured 472 versus 64 E_mech edges). Any violation that
    measured was confounded with an asymmetry the check itself introduced -
    largest exactly when the mechanistic stream was doing the most work.

    **Mutant encodings go to a scratch store**, never ``cfg.cache``. Writing
    them into the shared cache from an evaluation script permanently changed
    what ``store.has`` returned for later runs.

    Rows whose mutant protein is not available are skipped unless ``encoder`` is
    supplied to embed them on the fly. Skipped rows are counted, never ignored.
    """
    rows = df.iloc[:limit]
    forward, reverse = [], []
    skipped = 0
    mech_matched = 0
    cai_shifts: list[float] = []
    cfg_d = cfg.as_dict()
    scratch = scratch or store

    for _, row in tqdm(list(rows.iterrows()), desc="reversibility"):
        parent = row["parent_seq"]
        site = int(row["site"])
        mutant = parent[:site] + row["mt_aa"] + parent[site + 1:]
        parent_rec = store.get(parent)

        if scratch is not store and scratch.has(mutant):
            mut_rec = scratch.get(mutant)
        elif store.has(mutant):
            mut_rec = store.get(mutant)
        elif encoder is not None:
            mut_rec = encoder.encode(mutant, top_k=cfg.contact_top_k,
                                     min_sep=cfg.contact_min_sep)
            mut_cds, cai_shift = _substituted_cds(parent_rec, site, row["mt_aa"])
            if mut_cds is not None:
                mech_matched += 1
                cai_shifts.append(cai_shift)
            mut_rec["mech"] = mechanistic_features(
                mutant, mut_cds, len(mutant)
            ).half()
            scratch.put(mut_rec)
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
        "mutants_with_matched_cds": mech_matched,
        # The CAI change the codon substitution itself introduces at the mutated
        # site, which is a property of this check rather than of the model. It
        # was ~+0.215 and one-sided while the substituted codon was always the
        # most-preferred one; a CAI-matched codon should leave it near zero and
        # unbiased, and reporting it is what makes that checkable (R1-11).
        "induced_cai_shift_at_mutated_site": {
            "mean": float(np.mean(cai_shifts)) if cai_shifts else 0.0,
            "mean_abs": float(np.mean(np.abs(cai_shifts))) if cai_shifts else 0.0,
            "n": len(cai_shifts),
        },
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
    seed_everything(cfg.seed)
    device = describe_device(torch.device(cfg.device) if cfg.device else None)
    store = EncodingStore(cfg.cache)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    esm_dim = state.get("esm_dim", ESM_DIMS.get(cfg.esm_model, 1280))
    kwargs = state.get("model_kwargs") or model_kwargs_from_config(
        state.get("config", {}), esm_dim
    )
    model = HoloGeoN(**kwargs).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    df = pd.read_parquet(args.splits)
    part = df[df.split == args.split]
    if part.empty:
        raise SystemExit(f"{args.splits}: split {args.split!r} is empty")

    def _loader_for(frame):
        return DataLoader(
            MutationDataset(frame, store, cfg, target="target"),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_graphs,
        )

    mu, sigma, y = predict_split(model, _loader_for(part), device)
    report = {"provenance": provenance(),
              "checkpoint": args.checkpoint, "splits": args.splits,
              "split": args.split,
              "calibration": calibration_report(mu, sigma, y)}

    # Fit the variance scale on the VALIDATION split and report what it would do
    # to the evaluated split. Fitting on the split being scored would be
    # circular, so if there is no val split the scale is not fitted at all.
    val = df[df.split == "val"]
    if len(val):
        v_mu, v_sigma, v_y = predict_split(model, _loader_for(val), device)
        scale = fit_variance_scale(v_mu, v_sigma, v_y)
        report["variance_scale"] = {
            "scale": scale,
            "fitted_on": "val",
            "n_fit": int(len(v_y)),
            "calibration_after_scaling": calibration_report(mu, sigma * scale, y),
            "note": (
                "multiply the head's sigma by this scale at serving time to "
                "correct systematic over- or under-confidence; it is stored in "
                "the checkpoint directory and read by the interface"
            ),
        }
        print(f"[calibration] variance scale fitted on val: {scale:.4f}")
    else:
        report["variance_scale"] = {
            "scale": None,
            "note": "no validation split, so no scale was fitted (fitting on the "
                    "evaluated split would be circular)",
        }

    encoder = EsmEncoder(cfg.esm_model, device=str(device)) if args.encode_mutants else None
    # Mutant encodings are evaluation scratch, not training cache.
    scratch = EncodingStore(os.path.join(args.out, "reversibility_cache"))
    report["reversibility"] = reversibility_check(
        model, part, store, cfg, device, encoder, args.reversibility_limit, scratch)

    out_path = os.path.join(args.out, "calibration.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    cal = report["calibration"]
    print("[calibration] observed vs nominal coverage:")
    for k in cal["coverage_nominal"]:
        print(f"    {k:>4}: observed {cal['coverage_observed'][k]:.3f}")
    print(f"[calibration] ECE={cal['expected_calibration_error']:.4f}  "
          f"max coverage gap={cal['max_coverage_gap']:.4f}  "
          f"mean sigma={cal['mean_predicted_sigma']:.3f}  "
          f"residual std={cal['residual_std']:.3f}")
    print(f"[calibration] well_calibrated={cal['well_calibrated']} "
          f"(ECE <= {cal['tolerances']['ece']}, "
          f"coverage gap <= {cal['tolerances']['coverage_gap']})")
    print(f"[calibration] reversibility: {report['reversibility']}")
    print(f"[calibration] wrote {out_path}")


if __name__ == "__main__":
    main()
