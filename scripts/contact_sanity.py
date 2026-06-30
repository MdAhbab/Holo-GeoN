"""
scripts/contact_sanity.py
=========================
Sanity check for the geometric track (review Q2): do the ESM-2 contact-head
edges that feed E_geo carry genuine spatial (tertiary) signal, or just sequence
adjacency?

Two modes, both reading the cached contact maps written by precompute_embeddings
(cache/<md5>.pt holds {emb, contacts (L,L), seq}):

1. STRUCTURE-FREE (default, no downloads): for the top-L predicted contacts
   (separation >= min_sep), report the long-range fraction (|i-j| >= 24). A real
   structure-aware predictor places substantial mass on long-range pairs; the
   "linear-distance baseline" — which ranks pairs by sequence proximity — has a
   long-range fraction of 0 by construction. The gap is the non-local signal that
   a sequence-adjacency graph cannot produce.

2. STRUCTURE-BASED (optional --true-contacts-dir): if per-sequence ground-truth
   contact maps are available (npy named <md5>.npy, 1 = Cb-Cb < 8A, sep >= 6),
   also report precision@L and precision@L/5 for the ESM head vs the linear
   baseline — the standard contact-prediction metric.

Writes results/contacts/contact_sanity.json + a printed summary.

Example
-------
python -m scripts.contact_sanity --cache cache/emb_150M --max-seqs 300
python -m scripts.contact_sanity --cache cache/emb_150M --true-contacts-dir data/true_contacts
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG_RANGE = 24          # |i-j| >= 24 is "long-range" (standard contact convention)
MED_RANGE = 12           # 12 <= |i-j| < 24 is "medium-range"


def _upper_pairs(L, min_sep):
    iu = np.triu_indices(L, k=min_sep)
    return iu[0], iu[1]


def _topL_predicted(cm, L_top, min_sep):
    """Indices of the top-`L_top` predicted contacts (by probability), sep>=min_sep."""
    L = cm.shape[0]
    i, j = _upper_pairs(L, min_sep)
    if len(i) == 0:
        return np.array([], int), np.array([], int)
    scores = cm[i, j]
    k = min(L_top, len(i))
    sel = np.argsort(scores)[::-1][:k]
    return i[sel], j[sel]


def _topL_linear(L, L_top, min_sep):
    """Linear-distance baseline: rank pairs by sequence proximity (closest first)."""
    i, j = _upper_pairs(L, min_sep)
    if len(i) == 0:
        return np.array([], int), np.array([], int)
    sep = np.abs(i - j)
    k = min(L_top, len(i))
    sel = np.argsort(sep)[:k]          # smallest separation first
    return i[sel], j[sel]


def _long_range_fraction(i, j):
    if len(i) == 0:
        return float("nan"), float("nan")
    sep = np.abs(i - j)
    lr = float(np.mean(sep >= LONG_RANGE))
    mr = float(np.mean((sep >= MED_RANGE) & (sep < LONG_RANGE)))
    return lr, mr


def _precision(pi, pj, true_cm):
    if len(pi) == 0:
        return float("nan")
    return float(np.mean(true_cm[pi, pj] > 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="embedding/contact cache dir")
    ap.add_argument("--max-seqs", type=int, default=300, help="cap cached sequences sampled")
    ap.add_argument("--min-sep", type=int, default=6, help="ignore pairs closer than this")
    ap.add_argument("--true-contacts-dir", default=None,
                    help="optional dir of <md5>.npy ground-truth contact maps")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "contacts"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.cache, "*.pt")))
    if not files:
        print(f"! no cached *.pt in {args.cache}; run precompute first.")
        return
    files = files[: args.max_seqs]

    esm_lr, lin_lr, esm_mr = [], [], []
    esm_pL, esm_pL5, lin_pL, lin_pL5 = [], [], [], []
    sym_err = []
    n_struct = 0

    for f in files:
        rec = torch.load(f, map_location="cpu")
        cm = rec["contacts"].float().numpy()
        L = cm.shape[0]
        if L < args.min_sep + 2:
            continue
        sym_err.append(float(np.mean(np.abs(cm - cm.T))))

        ei, ej = _topL_predicted(cm, L, args.min_sep)
        li, lj = _topL_linear(L, L, args.min_sep)
        e_lr, e_mr = _long_range_fraction(ei, ej)
        l_lr, _ = _long_range_fraction(li, lj)
        esm_lr.append(e_lr); esm_mr.append(e_mr); lin_lr.append(l_lr)

        if args.true_contacts_dir:
            md5 = os.path.splitext(os.path.basename(f))[0]
            tp = os.path.join(args.true_contacts_dir, md5 + ".npy")
            if os.path.exists(tp):
                tcm = np.load(tp)
                if tcm.shape[0] >= L:
                    tcm = tcm[:L, :L]
                    n_struct += 1
                    e5i, e5j = _topL_predicted(cm, max(L // 5, 1), args.min_sep)
                    l5i, l5j = _topL_linear(L, max(L // 5, 1), args.min_sep)
                    esm_pL.append(_precision(ei, ej, tcm)); esm_pL5.append(_precision(e5i, e5j, tcm))
                    lin_pL.append(_precision(li, lj, tcm)); lin_pL5.append(_precision(l5i, l5j, tcm))

    def m(x):
        x = [v for v in x if v == v]   # drop NaN
        return float(np.mean(x)) if x else float("nan")

    summary = {
        "n_sequences": len(esm_lr),
        "min_separation": args.min_sep,
        "esm_contacts": {
            "long_range_fraction_topL": m(esm_lr),
            "medium_range_fraction_topL": m(esm_mr),
            "symmetry_error": m(sym_err),
        },
        "linear_baseline": {
            "long_range_fraction_topL": m(lin_lr),
        },
        "interpretation": (
            "ESM top-L contacts contain long-range (|i-j|>=24) pairs that the "
            "sequence-proximity baseline cannot produce (its long-range fraction "
            "is ~0); this is the non-local/spatial signal feeding E_geo."
        ),
    }
    if args.true_contacts_dir and n_struct:
        summary["precision_vs_structure"] = {
            "n_with_structure": n_struct,
            "esm_precision_at_L": m(esm_pL),
            "esm_precision_at_L5": m(esm_pL5),
            "linear_precision_at_L": m(lin_pL),
            "linear_precision_at_L5": m(lin_pL5),
        }

    out_path = os.path.join(args.out, "contact_sanity.json")
    json.dump(summary, open(out_path, "w"), indent=2)

    print("Contact-head sanity (ESM E_geo vs linear-distance baseline):")
    print(f"  sequences sampled : {summary['n_sequences']}")
    print(f"  ESM    long-range fraction (top-L): {summary['esm_contacts']['long_range_fraction_topL']:.3f}")
    print(f"  linear long-range fraction (top-L): {summary['linear_baseline']['long_range_fraction_topL']:.3f}")
    print(f"  ESM contact-map symmetry error    : {summary['esm_contacts']['symmetry_error']:.4f}")
    if "precision_vs_structure" in summary:
        p = summary["precision_vs_structure"]
        print(f"  precision@L   ESM={p['esm_precision_at_L']:.3f} vs linear={p['linear_precision_at_L']:.3f}")
        print(f"  precision@L/5 ESM={p['esm_precision_at_L5']:.3f} vs linear={p['linear_precision_at_L5']:.3f}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
