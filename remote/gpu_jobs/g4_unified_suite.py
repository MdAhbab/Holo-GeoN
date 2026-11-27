#!/usr/bin/env python3
"""
remote/gpu_jobs/g4_unified_suite.py
===================================
Backwards-compatibility shim forwarding to ``remote/gpu_jobs/a100_unified_suite.py``.

Kept so any existing CLI invocation (``python remote/gpu_jobs/g4_unified_suite.py ...``)
or import (``from remote.gpu_jobs.g4_unified_suite import G4Suite``) continues to
work identically on A100 / Blackwell / any CUDA GPU without hardcoding a card name.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remote.gpu_jobs.a100_unified_suite import (  # noqa: E402,F401
    A100Suite,
    G4Suite,
    _eval_groups,
    _group_by_parent,
    _measure_contact_drift,
    _safe_pearson,
    _sha256_file,
    log,
    main,
    run_cmd,
    sync_to_drive,
    utc_now,
)

if __name__ == "__main__":
    raise SystemExit(main())
