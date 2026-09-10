"""
tests/test_dataset_etl.py
=========================
Regressions for the dataset-ingestion layer.

These cover failures found while building the real tables from
``dataset/``: each one produced wrong numbers rather than an error, which is the
class of bug the rest of this suite exists to catch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tests.helpers import check, section


def test_missing_cds_reads_as_absent() -> None:
    """
    A row with no coding sequence must report mech_valid = 0.

    ``str(value)`` on a missing cell yields the literal "nan", which is TRUTHY,
    so ``mechanistic_features`` took it for a real coding sequence: it stamped
    mech_valid = 1 and filled the CAI and mRNA-folding channels from a
    three-character non-sequence. The run then reports mechanistic coverage it
    does not have, and the ``no_mech_stream`` ablation - the arm that tests the
    paper's distinguishing contribution - is measured against partly fabricated
    channels.
    """
    section("dataset ETL — a missing CDS must read as absent, not as 'nan'")

    from src.data import clean_cds
    from src.features import CH_CAI, CH_MRNA, CH_VALID, mechanistic_features

    for missing in (None, np.nan, float("nan"), pd.NA, "nan", "NaN", "None", "", "   "):
        check(clean_cds(missing) is None,
              f"clean_cds({missing!r}) is None")
    check(clean_cds("ATGGCTAAA") == "ATGGCTAAA", "a real coding sequence survives")
    check(clean_cds("  ATGGCT  ") == "ATGGCT", "and is stripped")

    protein = "MKTAYIAKQ"
    # The exact value the old code produced for a missing cell.
    stale = mechanistic_features(protein, dna_seq="nan", length=len(protein))
    check(float(stale[:, CH_VALID].max()) == 1.0,
          "the literal string 'nan' WOULD have been taken for a real CDS "
          "(this is the bug, reproduced)")

    fixed = mechanistic_features(protein, dna_seq=clean_cds(np.nan),
                                 length=len(protein))
    check(float(fixed[:, CH_VALID].max()) == 0.0,
          "with clean_cds the same row reports mech_valid = 0")
    check(float(fixed[:, CH_CAI].abs().max()) == 0.0
          and float(fixed[:, CH_MRNA].abs().max()) == 0.0,
          "and its CAI and mRNA channels stay empty rather than fabricated")

    real = mechanistic_features(protein, dna_seq="ATGAAAACCGCTTATATCGCTCAA",
                                length=len(protein))
    check(float(real[:, CH_VALID].max()) == 1.0,
          "a genuine coding sequence still activates the stream")


def test_partially_populated_cds_column_loads() -> None:
    """
    A table whose CDS column is only partly filled must load, and only the
    populated rows may carry a coding sequence.

    Under pandas 3 ``astype(str)`` keeps missing values as NA instead of
    stringifying them, so the previous ``v.lower()`` raised AttributeError on
    any such column; under pandas 2 the same code silently produced a "nan"
    coding sequence. Both are covered here.
    """
    section("dataset ETL — a partly-populated CDS column")

    import tempfile

    from src.data import load_mutation_table

    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    rows = []
    for i, aa in enumerate("CDEF"):
        rows.append({
            "wt_seq": seq,
            "mt_seq": seq[:5] + aa + seq[6:],
            "ddg": -0.5 - i,
            # Only the first two rows have a coding sequence.
            "wt_dna": ("ATG" * len(seq)) if i < 2 else None,
        })

    with tempfile.TemporaryDirectory(prefix="holo_geon_cds_") as tmp:
        path = os.path.join(tmp, "partial_cds.parquet")
        pd.DataFrame(rows).to_parquet(path)

        norm, report = load_mutation_table(path, kind="stability")
        check(len(norm) == 4, f"all 4 rows load (got {len(norm)})")

        present = norm["cds"].notna()
        check(int(present.sum()) == 2,
              f"exactly the 2 rows with a real CDS carry one "
              f"(got {int(present.sum())})")
        # The bug was a missing cell arriving as the STRING "nan". A genuinely
        # missing value (None/NaN) is the correct representation, so it must be
        # excluded from this check with pd.isna rather than an `is not None`
        # test, which NaN passes.
        populated = [v for v in norm["cds"] if not pd.isna(v)]
        check(all(isinstance(v, str) for v in populated),
              "every populated coding sequence is a real string")
        check(not any(v.strip().lower() in ("nan", "none") for v in populated),
              "no row carries the literal string 'nan' as its coding sequence")


TESTS = [
    test_missing_cds_reads_as_absent,
    test_partially_populated_cds_column_loads,
]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
