"""
tests/run_all.py
================
Run every test module and exit non-zero on any failure.

    python -m tests.run_all      (or: python run.py test)

No data and no network are required: the model tests build batches directly,
and the pipeline test uses the synthetic encoder in ``tests/helpers.py``.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import helpers


def main() -> int:
    from tests import (
        test_app,
        test_cli,
        test_dataset_etl,
        test_invariants,
        test_pipeline,
        test_remote_job,
        test_round1_fixes,
        test_round2_fixes,
        test_v2,
    )

    modules = [
        ("invariants", test_invariants.TESTS),
        ("pipeline", test_pipeline.TESTS),
        ("web backend", test_app.TESTS),
        ("cross-platform CLI", test_cli.TESTS),
        ("round-1 review regressions", test_round1_fixes.TESTS),
        ("round-2 review regressions", test_round2_fixes.TESTS),
        ("dataset ETL regressions", test_dataset_etl.TESTS),
        ("remote job runner", test_remote_job.TESTS),
        ("Holo-GeoN v2 & geometry invariants", test_v2.TESTS),
    ]

    errors: list[str] = []
    for name, tests in modules:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        for fn in tests:
            try:
                fn()
            except Exception:
                errors.append(f"{name}.{fn.__name__} raised:\n{traceback.format_exc()}")
                print(f"  ERROR {fn.__name__} raised an exception")

    passed, failed = helpers.results()
    print(f"\n{'=' * 70}")
    print(f"{passed} checks passed, {len(failed)} failed, {len(errors)} errors")
    for f in failed:
        print(f"  FAILED: {f}")
    for e in errors:
        print(f"\n{e}")

    if failed or errors:
        print("\nTESTS FAILED")
        return 1
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
