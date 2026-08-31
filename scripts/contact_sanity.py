"""
scripts/contact_sanity.py
=========================
Do the edges feeding E_geo carry spatial signal, or just sequence adjacency?

This is the check that has to justify building a graph at all. If the
contact-head edges were merely "residues near each other in sequence", the graph
would be a chain with extra steps and every claim about capturing tertiary
structure without MSAs or crystal structures would be empty.

Ruling that out is necessary, not sufficient: only the structure-based mode can
show the edges are *right* rather than merely non-local.

Two modes:

1. **Structure-free (default).** Among the top-L predicted contacts with
   separation >= ``--min-sep``, report the fraction that are long-range
   (|i - j| >= 24), against TWO controls: a sequence-proximity baseline (whose
   long-range fraction is 0 by construction) and a **random-edge control**
   drawn uniformly from eligible pairs.

   The random control is what makes this mode readable. Round-2 review, R2-13:
   with only the proximity baseline, the comparison was won by construction -
   the reviewer fed in 40 caches whose edges were drawn uniformly at random and
   scored 0.880 against a baseline defined to score 0.000. Beating the proximity
   baseline shows the head emits long-range *predictions*; it says nothing about
   whether any of them are *correct*. A head that cannot beat the random control
   has no spatial signal at all, and no structure-free number can establish more
   than that.

2. **Structure-based (optional, and the only mode that can support the claim).**
   With ground-truth contact maps (``--true-contacts-dir``, one
   ``<cache-key>.npy`` per sequence, 1 = C-beta contact), report precision@L and
   precision@L/5 against both controls — the standard contact-prediction metric.
   ``--require-structure`` makes its absence an error, for any run whose numbers
   will be reported.

Note on the sparse cache: encodings store the top-k contacts per residue
(k = ``contact_top_k``, default 24) rather than a dense L x L map. Selecting the
global top-L from that set is equivalent for L <= k*L/2, which holds for every
realistic sequence. Contact-map symmetry is no longer measured because edges are
canonicalised to i < j at extraction, so symmetry holds by construction rather
than as an empirical property.

Example
-------
python -m scripts.contact_sanity --cache cache/esm2_650M --max-seqs 300
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG_RANGE = 24          # |i-j| >= 24 is long-range, by convention
MED_RANGE = 12


def _top_predicted(edges: np.ndarray, probs: np.ndarray, k: int, min_sep: int):
    """Top-k cached contacts by probability, restricted to sep >= min_sep."""
    if edges.size == 0:
        return np.array([], int), np.array([], int)
    sep = np.abs(edges[0] - edges[1])
    keep = sep >= min_sep
    i, j, p = edges[0][keep], edges[1][keep], probs[keep]
    if i.size == 0:
        return np.array([], int), np.array([], int)
    order = np.argsort(p)[::-1][: min(k, i.size)]
    return i[order], j[order]


def _top_linear(length: int, k: int, min_sep: int):
    """Sequence-proximity baseline: closest eligible pairs first."""
    i, j = np.triu_indices(length, k=min_sep)
    if i.size == 0:
        return np.array([], int), np.array([], int)
    order = np.argsort(np.abs(i - j))[: min(k, i.size)]
    return i[order], j[order]


def _top_random(length: int, k: int, min_sep: int, seed: int):
    """
    Random-edge control: uniformly drawn eligible pairs.

    The floor any spatial claim has to clear. A uniformly random edge set has a
    high long-range fraction simply because most pairs at ``|i-j| >= min_sep``
    are far apart, so "beats the sequence-proximity baseline" is not evidence of
    structure - it is arithmetic (R2-13).
    """
    i, j = np.triu_indices(length, k=min_sep)
    if i.size == 0:
        return np.array([], int), np.array([], int)
    rng = np.random.default_rng(seed)
    pick = rng.choice(i.size, size=min(k, i.size), replace=False)
    return i[pick], j[pick]


def _range_fractions(i, j):
    if len(i) == 0:
        return float("nan"), float("nan")
    sep = np.abs(i - j)
    return float(np.mean(sep >= LONG_RANGE)), float(
        np.mean((sep >= MED_RANGE) & (sep < LONG_RANGE))
    )


def _precision(i, j, true_map) -> float:
    if len(i) == 0:
        return float("nan")
    return float(np.mean(true_map[i, j] > 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-seqs", type=int, default=300)
    ap.add_argument("--min-sep", type=int, default=6)
    ap.add_argument("--true-contacts-dir", default=None)
    ap.add_argument("--require-structure", action="store_true",
                    help="fail unless ground-truth contact maps are supplied and "
                         "matched. Use for any run whose contact numbers will be "
                         "reported: the structure-free mode cannot distinguish "
                         "the contact head from random edges (R2-13)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the random-edge control")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "contacts"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.require_structure and not args.true_contacts_dir:
        raise SystemExit(
            "--require-structure was given without --true-contacts-dir. The "
            "structure-free mode reports a long-range FRACTION, which a "
            "uniformly random edge set also scores highly on; it cannot "
            "establish that the contact head carries structural signal. Supply "
            "one <cache-key>.npy C-beta contact map per sequence, or drop "
            "--require-structure and do not cite the result as evidence for the "
            "graph (R2-13)."
        )

    files = sorted(glob.glob(os.path.join(args.cache, "*.pt")))[: args.max_seqs]
    if not files:
        raise SystemExit(f"no cached *.pt in {args.cache}; run scripts.precompute_embeddings first")

    esm_lr, esm_mr, lin_lr, rnd_lr = [], [], [], []
    esm_pL, esm_pL5, lin_pL, lin_pL5, rnd_pL, rnd_pL5 = [], [], [], [], [], []
    n_struct = 0

    for seq_index, path in enumerate(files):
        rec = torch.load(path, map_location="cpu", weights_only=False)
        edges = rec["contact_edges"].numpy().astype(int)
        probs = rec["contact_probs"].float().numpy()
        length = int(rec["emb"].shape[0])
        if length < args.min_sep + 2:
            continue

        ei, ej = _top_predicted(edges, probs, length, args.min_sep)
        li, lj = _top_linear(length, length, args.min_sep)
        ri, rj = _top_random(length, length, args.min_sep, args.seed + seq_index)
        lr, mr = _range_fractions(ei, ej)
        esm_lr.append(lr)
        esm_mr.append(mr)
        lin_lr.append(_range_fractions(li, lj)[0])
        rnd_lr.append(_range_fractions(ri, rj)[0])

        if args.true_contacts_dir:
            key = os.path.splitext(os.path.basename(path))[0]
            tp = os.path.join(args.true_contacts_dir, key + ".npy")
            if os.path.exists(tp):
                true_map = np.load(tp)
                if true_map.shape[0] >= length:
                    true_map = true_map[:length, :length]
                    n_struct += 1
                    k5 = max(length // 5, 1)
                    e5i, e5j = _top_predicted(edges, probs, k5, args.min_sep)
                    l5i, l5j = _top_linear(length, k5, args.min_sep)
                    r5i, r5j = _top_random(length, k5, args.min_sep,
                                           args.seed + seq_index)
                    esm_pL.append(_precision(ei, ej, true_map))
                    esm_pL5.append(_precision(e5i, e5j, true_map))
                    lin_pL.append(_precision(li, lj, true_map))
                    lin_pL5.append(_precision(l5i, l5j, true_map))
                    rnd_pL.append(_precision(ri, rj, true_map))
                    rnd_pL5.append(_precision(r5i, r5j, true_map))

    def mean(xs):
        xs = [v for v in xs if v == v]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "n_sequences": len(esm_lr),
        "min_separation": args.min_sep,
        "contact_head": {
            "long_range_fraction_topL": mean(esm_lr),
            "medium_range_fraction_topL": mean(esm_mr),
        },
        "sequence_proximity_baseline": {"long_range_fraction_topL": mean(lin_lr)},
        "random_edge_control": {"long_range_fraction_topL": mean(rnd_lr)},
        "interpretation": (
            "The long-range fraction is a NECESSARY check, not a sufficient one. "
            "Beating the sequence-proximity baseline is arithmetic - its "
            "long-range fraction is 0 by construction - and a uniformly random "
            "edge set scores near the random_edge_control figure here without "
            "carrying any structure. Read the head against the RANDOM control, "
            "and treat even that as evidence only that the edges are non-local. "
            "Whether they are CORRECT needs precision_vs_structure, which "
            "requires --true-contacts-dir (R2-13)."
        ),
        "supports_structural_claim": bool(args.true_contacts_dir and n_struct),
    }
    if args.true_contacts_dir and n_struct:
        summary["precision_vs_structure"] = {
            "n_with_structure": n_struct,
            "contact_head_precision_at_L": mean(esm_pL),
            "contact_head_precision_at_L5": mean(esm_pL5),
            "baseline_precision_at_L": mean(lin_pL),
            "baseline_precision_at_L5": mean(lin_pL5),
            "random_precision_at_L": mean(rnd_pL),
            "random_precision_at_L5": mean(rnd_pL5),
        }

    out_path = os.path.join(args.out, "contact_sanity.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Contact-head sanity (E_geo source vs sequence-proximity baseline):")
    print(f"  sequences sampled                 : {summary['n_sequences']}")
    print(f"  contact head long-range fraction  : {summary['contact_head']['long_range_fraction_topL']:.3f}")
    print(f"  proximity    long-range fraction  : {summary['sequence_proximity_baseline']['long_range_fraction_topL']:.3f}"
          f"   (0 by construction)")
    print(f"  RANDOM EDGES long-range fraction  : {summary['random_edge_control']['long_range_fraction_topL']:.3f}"
          f"   <- the floor that matters")
    if "precision_vs_structure" in summary:
        p = summary["precision_vs_structure"]
        print(f"  precision@L    head={p['contact_head_precision_at_L']:.3f} "
              f"vs baseline={p['baseline_precision_at_L']:.3f} "
              f"vs random={p['random_precision_at_L']:.3f}")
        print(f"  precision@L/5  head={p['contact_head_precision_at_L5']:.3f} "
              f"vs baseline={p['baseline_precision_at_L5']:.3f} "
              f"vs random={p['random_precision_at_L5']:.3f}")
    else:
        print("  precision vs structure            : NOT MEASURED. Without "
              "--true-contacts-dir this run cannot show the edges are correct, "
              "only that they are non-local (R2-13).")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
