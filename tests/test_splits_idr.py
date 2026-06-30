"""Unit tests for leakage-controlled splits and the IDR physics prior.
Run:  python -m tests.test_splits_idr
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import splits as S
from src import idr_prior as P


def test_no_leakage():
    # 5 parent proteins, each mutated several times -> rows
    parents = ["P0", "P1", "P2", "P3", "P4"]
    group_keys = []
    for p in parents:
        group_keys += [p] * 20  # 20 mutations each
    row_splits = S.row_splits_from_groups(group_keys, ratios=(0.6, 0.2, 0.2), seed=1)
    assert S.verify_no_leakage(group_keys, row_splits), "parent leaked across splits!"
    # every split present
    assert set(row_splits) <= {"train", "val", "test"}
    print(f"ok  protein-disjoint split, no leakage ({len(set(row_splits))} splits used)")


def test_group_key_stable():
    s = "MKTAYIAKQR"
    assert S.group_key_for_sequence(s) == S.group_key_for_sequence(s)
    assert S.group_key_for_sequence(s) != S.group_key_for_sequence(s + "A")
    print("ok  stable parent-sequence group key")


def test_assignment_deterministic():
    keys = [f"g{i}" for i in range(100)]
    a = S.assign_groups_to_splits(keys, seed=7)
    b = S.assign_groups_to_splits(keys, seed=7)
    assert a == b, "assignment must be deterministic for a fixed seed"
    print("ok  deterministic cluster->split assignment")


def test_idr_prior():
    assert P.rg_prior("A" * 100) > P.rg_prior("A" * 20)   # grows with N
    charged = "EKEKEKEKEKEKEKEKEKEK"
    hydrophobic = "LIVLIVLIVLIVLIVLIVLI"
    assert P.estimate_nu(charged) > P.estimate_nu(hydrophobic), \
        "charged IDR should be more expanded (higher nu) than hydrophobic"
    assert 0.33 <= P.estimate_nu(charged) <= 0.60
    print(f"ok  IDR physics prior (nu_charged={P.estimate_nu(charged):.2f} > "
          f"nu_hydrophobic={P.estimate_nu(hydrophobic):.2f})")


if __name__ == "__main__":
    test_no_leakage()
    test_group_key_stable()
    test_assignment_deterministic()
    test_idr_prior()
    print("\nALL SPLIT+IDR TESTS PASSED")
