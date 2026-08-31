"""
tests/test_invariants.py
========================
Properties the architecture claims, asserted as tests.

Each of these corresponds to a failure that actually shipped in a predecessor
build, so they are regression tests rather than decoration:

  * The mutation is always inside the encoded window. Head truncation at 256
    residues removed the mutation for ~64% of ClinVar variants, leaving
    wild-type and mutant inputs byte-identical.
  * ddG is exactly antisymmetric, in train mode as well as eval. A soft penalty
    left measured violations of 0.029-0.071 kcal/mol; dropout inside the score
    head would leave the identity holding only under `.eval()`.
  * A deep mutational scan agrees with per-mutation calls, so the interface's
    fast path cannot drift from the slow one.
  * ``mech_valid`` is 0 without a coding sequence and E_mech is empty, so the
    mechanistic stream cannot be fed a proxy that carries no information.
  * Splits never place a group on both sides.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src import splits as S
from src.features import CH_CAI, CH_MRNA, CH_VALID, mechanistic_features
from src.graphs import build_geometric_graph, build_mechanistic_graph, khop_neighborhood
from src.model import HoloGeoN
from src.substitution import MutationParseError, aa_index, parse_mutation
from src.windowing import mutation_position, window_bounds, window_for_pair
from tests.helpers import check, make_batch, random_protein, section


# ---------------------------------------------------------------------------
def test_windowing() -> None:
    section("windowing")

    # The mutated residue is inside the window for every position in a long
    # protein — including both termini, where the window has to slide.
    length, window = 1000, 256
    for site in [0, 1, 127, 128, 383, 500, 871, 998, 999]:
        start, end = window_bounds(length, site, window)
        check(start <= site < end,
              f"site {site} inside window [{start}, {end})")
        check(end - start == window, f"site {site}: full window length")

    # Short sequences yield the whole sequence, not a padded window.
    start, end = window_bounds(50, 10, 256)
    check((start, end) == (0, 50), "short sequence returns the whole sequence")

    # A paired window keeps both sequences aligned and differing at the site.
    wt = random_protein(600, seed=1)
    site = 400
    mt = wt[:site] + ("A" if wt[site] != "A" else "C") + wt[site + 1:]
    w_win, m_win, local, start = window_for_pair(wt, mt, 256)
    check(w_win[local] == wt[site] and m_win[local] == mt[site],
          "paired window preserves the substitution")
    check(sum(a != b for a, b in zip(w_win, m_win)) == 1,
          "paired windows differ at exactly one position")

    # The regression this replaces: head truncation would have dropped it.
    check(site >= 256, "site 400 would have been cut by head truncation at 256")

    # Indels and multi-substitutions are rejected loudly, not silently mapped
    # to position 0.
    for bad_wt, bad_mt, why in [
        ("ACDEF", "ACDE", "length mismatch"),
        ("ACDEF", "ACDEF", "no difference"),
        ("ACDEF", "AGDEG", "two differences"),
    ]:
        try:
            mutation_position(bad_wt, bad_mt)
            check(False, f"rejects {why}")
        except MutationParseError:
            check(True, f"rejects {why}")


# ---------------------------------------------------------------------------
def test_mechanistic_features() -> None:
    section("mechanistic features")

    seq = random_protein(60, seed=2)
    protein_only = mechanistic_features(seq, None, len(seq))
    check(float(protein_only[:, CH_VALID].max()) == 0.0,
          "no coding sequence -> mech_valid is 0 everywhere")
    check(float(protein_only[:, CH_MRNA].abs().max()) == 0.0
          and float(protein_only[:, CH_CAI].abs().max()) == 0.0,
          "no coding sequence -> mRNA and CAI channels are zero")
    check(float(protein_only[:, 2:6].abs().max()) > 0.0,
          "protein-only channels (charge/hydropathy/volume/helix) are populated")

    edges, attr = build_mechanistic_graph(
        protein_only[:, CH_CAI], protein_only[:, CH_MRNA], protein_only[:, CH_VALID]
    )
    non_loop = edges[:, edges[0] != edges[1]]
    check(non_loop.numel() == 0,
          "no coding sequence -> E_mech has no edges beyond self-loops")

    # With a real coding sequence the channels populate (given ViennaRNA).
    from src.features import VIENNA_AVAILABLE
    if VIENNA_AVAILABLE:
        from src.codon_tables import GENETIC_CODE
        rev = {}
        for codon, aa in GENETIC_CODE.items():
            rev.setdefault(aa, codon)
        cds = "".join(rev.get(a, "GCT") for a in seq)
        with_cds = mechanistic_features(seq, cds, len(seq))
        check(float(with_cds[:, CH_VALID].min()) == 1.0,
              "real coding sequence -> mech_valid is 1")
        check(float(with_cds[:, CH_CAI].abs().max()) > 0.0,
              "real coding sequence -> CAI channel is populated")
    else:
        print("    (ViennaRNA absent: skipping the with-CDS half)")


# ---------------------------------------------------------------------------
def test_graphs() -> None:
    section("graph construction")

    n = 40
    edges = torch.tensor([[2, 5, 9], [20, 30, 35]], dtype=torch.int32)
    probs = torch.tensor([0.9, 0.8, 0.7])
    ei, ea = build_geometric_graph(n, edges, probs, top_k=8, threshold=0.3)

    check(ea.shape[1] == 3, "edge features are 3-dimensional")
    check(int(ei.max()) < n and int(ei.min()) >= 0, "edge indices stay in range")

    degree = torch.bincount(ei[0], minlength=n)
    check(int(degree.min()) > 0, "every node has at least one outgoing edge")

    backbone = ea[:, 2] > 0.5
    check(int(backbone.sum()) == 2 * (n - 1), "backbone edges present in both directions")

    sep_of_backbone = (ei[0] - ei[1]).abs()[backbone]
    check(bool((sep_of_backbone == 1).all()), "backbone edges connect adjacent residues")

    # Thresholding drops weak contacts.
    ei2, ea2 = build_geometric_graph(
        n, edges, torch.tensor([0.9, 0.1, 0.1]), top_k=8, threshold=0.3
    )
    contacts = int((ea2[:, 2] < 0.5).sum()) - n   # minus self-loops
    check(contacts == 2, "sub-threshold contacts are dropped (1 pair, 2 directions)")

    # k-hop neighbourhood reaches along the backbone and across a contact.
    mask = khop_neighborhood(ei, 2, n, hops=1)
    check(bool(mask[2]), "neighbourhood contains the site itself")
    check(bool(mask[20]), "neighbourhood reaches across a contact edge")
    check(bool(mask[1]) and bool(mask[3]), "neighbourhood reaches along the backbone")


# ---------------------------------------------------------------------------
def test_model_invariants() -> None:
    section("model invariants")

    torch.manual_seed(0)
    dim = 32
    model = HoloGeoN(esm_dim=dim, hidden=24, out_dim=24, aa_dim=8)
    batch = make_batch(dim=dim, batch_size=4, length=30, seed=3)
    reversed_batch = dict(batch)
    reversed_batch["wt_idx"], reversed_batch["mt_idx"] = batch["mt_idx"], batch["wt_idx"]

    for mode in ("train", "eval"):
        model.train(mode == "train")
        with torch.no_grad():
            z = model.context(batch)
            ddg, logvar = model.ddg(batch, z)
            ddg_rev, _ = model.ddg(reversed_batch, z)
            violation = float((ddg + ddg_rev).abs().max())
            check(violation == 0.0,
                  f"[{mode}] ddG is exactly antisymmetric (max violation {violation:.2e})")

            ts = model.three_state(batch, z)
            ts_rev = model.three_state(reversed_batch, z)
            equiv = float((ts_rev - ts.flip(-1)).abs().max())
            check(equiv == 0.0,
                  f"[{mode}] three-state is swap-equivariant (max deviation {equiv:.2e})")

            scan = model.scan(batch, z)
            picked = scan.gather(1, batch["mt_idx"].view(-1, 1))
            check(float((picked - ddg).abs().max()) < 1e-5,
                  f"[{mode}] scan agrees with the per-mutation call")
            wt_col = scan.gather(1, batch["wt_idx"].view(-1, 1))
            check(float(wt_col.abs().max()) < 1e-5,
                  f"[{mode}] scan's wild-type column is zero")

            check(bool(torch.isfinite(logvar).all()), f"[{mode}] logvar is finite")
            mu, sigma = model.idr(batch)
            check(bool((sigma > 0).all()), f"[{mode}] IDR sigma is positive")

    # Padding must not change a real sample's prediction.
    model.eval()
    with torch.no_grad():
        small = make_batch(dim=dim, batch_size=1, length=20, seed=7)
        padded = make_batch(dim=dim, batch_size=1, length=20, seed=7, pad_to=40)
        a, _ = model.ddg(small)
        b, _ = model.ddg(padded)
        check(float((a - b).abs().max()) < 1e-4,
              f"padding does not change the prediction (delta {float((a - b).abs().max()):.2e})")


# ---------------------------------------------------------------------------
def test_substitution_parsing() -> None:
    section("substitution parsing")

    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ"
    mt, idx, wt_aa, mut_aa = parse_mutation(seq, "L8P")
    check((idx, wt_aa, mut_aa) == (7, "L", "P"), "parses a well-formed code")
    check(mt[7] == "P" and len(mt) == len(seq), "applies the substitution")

    for bad, why in [
        ("A8P", "wild-type mismatch"),
        ("L800P", "position out of range"),
        ("L8Z", "non-standard residue"),
        ("LP", "malformed code"),
        ("LxP", "unparseable position"),
    ]:
        try:
            parse_mutation(seq, bad)
            check(False, f"rejects {why}")
        except MutationParseError:
            check(True, f"rejects {why}")

    check(aa_index("A") == 0 and aa_index("Y") == 19, "alphabet indices are stable")
    check(aa_index("X") == 20, "unknown residues get a dedicated slot, not index 0")


# ---------------------------------------------------------------------------
def test_splits() -> None:
    section("split integrity")

    groups = [f"g{i % 17}" for i in range(500)]
    row_splits = S.row_splits_from_groups(groups, ratios=(0.8, 0.1, 0.1), seed=42)
    check(S.verify_no_leakage(groups, row_splits), "no group spans two splits")

    again = S.row_splits_from_groups(groups, ratios=(0.8, 0.1, 0.1), seed=42)
    check(row_splits == again, "assignment is deterministic for a fixed seed")

    other = S.row_splits_from_groups(groups, ratios=(0.8, 0.1, 0.1), seed=7)
    check(row_splits != other, "a different seed gives a different assignment")

    leaky = list(row_splits)
    first = groups[0]
    for i, g in enumerate(groups):
        if g == first:
            leaky[i] = "train" if leaky[i] != "train" else "test"
            break
    check(not S.verify_no_leakage(groups, leaky), "leakage is actually detected")


TESTS = [
    test_windowing,
    test_mechanistic_features,
    test_graphs,
    test_model_invariants,
    test_substitution_parsing,
    test_splits,
]
