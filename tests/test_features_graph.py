"""Unit tests for the rebuilt feature + dual-track graph modules.
Run from repo root:  python -m tests.test_features_graph
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src import features as F
from src.codon_tables import HUMAN_CAI_WEIGHT
from src.utils import graph_builder as G


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_codon_weights():
    assert approx(HUMAN_CAI_WEIGHT["ATG"], 1.0)   # Met, single codon
    assert approx(HUMAN_CAI_WEIGHT["TGG"], 1.0)   # Trp, single codon
    # CTG is the most-used Leu codon (frac 0.41) -> w == 1.0
    assert approx(HUMAN_CAI_WEIGHT["CTG"], 1.0)
    # TTA is rare Leu (0.07) -> small w
    assert HUMAN_CAI_WEIGHT["TTA"] < 0.2
    print("ok  codon weights (human relative adaptiveness)")


def test_cai():
    dna = "ATGCTGGCCAAA"  # M L A K
    per = F.cai_per_residue(dna, max_length=10)
    assert approx(per[0], 1.0) and approx(per[1], 1.0)   # ATG, CTG -> 1.0
    assert 0.0 < F.cai_sequence(dna) <= 1.0
    assert len(per) == 10  # padded
    print(f"ok  CAI per-residue + sequence ({F.cai_sequence(dna):.3f})")


def test_charge():
    assert F.charge_per_residue("KKKKKKK", 7)[3] > 0.9      # poly-Lys ~ +1
    assert F.charge_per_residue("EEEEEEE", 7)[3] < -0.9     # poly-Glu ~ -1
    assert approx(F.charge_per_residue("AAAAAAA", 7)[3], 0.0)
    print("ok  windowed charge")


def test_mrna():
    print(f"    ViennaRNA available: {F.VIENNA_AVAILABLE}")
    structured = "GGGGGGCCCCCCGGGGGGCCCCCC"  # should fold (negative MFE)
    vals = F.mrna_fold_per_residue(structured, max_length=8)
    assert len(vals) == 8
    if F.VIENNA_AVAILABLE:
        assert min(vals) < 0.0, "expected negative local MFE for a structured seq"
    print(f"ok  mRNA fold per-residue (min={min(vals):.3f})")


def test_mech_features_mask():
    prot = "MLAKDE"
    dna = "ATGCTGGCCAAAGATGAA"
    f_dna = F.mechanistic_features(prot, dna, max_length=8)
    f_prot = F.mechanistic_features(prot, None, max_length=8)
    assert f_dna.shape == (8, 4) and f_prot.shape == (8, 4)
    # DNA path: valid mask set where ViennaRNA real; charge always present
    assert f_prot[:, 3].sum() == 0.0, "protein-only must have mech_valid all zero"
    assert f_prot[0, 1] == 0.0, "protein-only CAI must be zeroed"
    assert f_prot[:, 2].abs().sum() > 0.0, "charge must still be computed"
    print(f"ok  mechanistic_features shape+mask (dna valid sum={f_dna[:,3].sum():.0f})")


def test_geometric_edges():
    L = 6
    cm = torch.zeros(L, L)
    cm[0, 4] = cm[4, 0] = 0.9
    cm[1, 5] = cm[5, 1] = 0.8
    e = G.build_geometric_edges(cm, threshold=0.5)
    assert e.shape[0] == 2 and e.shape[1] == 4  # 2 pairs * 2 directions
    # bidirectional check
    pairs = set(map(tuple, e.t().tolist()))
    assert (0, 4) in pairs and (4, 0) in pairs
    print(f"ok  E_geo from contact map ({e.shape[1]} directed edges)")


def test_mechanistic_edges():
    L = 12
    cai = torch.linspace(0.2, 0.9, L)
    mrna = torch.zeros(L)
    cai[2], cai[9] = 0.50, 0.50   # a far-apart matched pair
    mrna[2], mrna[9] = -0.3, -0.3
    valid = torch.ones(L)
    e = G.build_mechanistic_edges(cai, mrna, valid, eps=0.05, delta=0.05,
                                  max_degree=4, min_separation=3)
    pairs = set(map(tuple, e.t().tolist()))
    assert (2, 9) in pairs and (9, 2) in pairs, "expected long-range E_mech 2<->9"
    # protein-only (mech_valid all zero) -> no mechanistic edges
    e0 = G.build_mechanistic_edges(cai, mrna, torch.zeros(L), 0.05, 0.05)
    assert e0.shape[1] == 0, "protein-only must yield empty E_mech"
    print(f"ok  E_mech long-range coupling + empty-when-invalid ({e.shape[1]} edges)")


if __name__ == "__main__":
    test_codon_weights()
    test_cai()
    test_charge()
    test_mrna()
    test_mech_features_mask()
    test_geometric_edges()
    test_mechanistic_edges()
    print("\nALL FEATURE+GRAPH TESTS PASSED")
