"""
src/provenance.py
=================
What produced this artifact.

Round-2 review, R2-15: a run manifest carried the config, the seed and the
metrics, and nothing about the machine or the software. No git commit, no
package versions, no CUDA version, no device, no MMseqs2 presence flag. That
matters more here than in most projects because the encoder's contact extraction
is version-sensitive by the code's own admission - ``src/encoder.py`` requires
eager attention because "the default sdpa implementation in transformers >= 5
does not emit" attention maps - while ``requirements.txt`` pinned only
``transformers>=4.40``. An independent group installing from it today gets a
different stack, and nothing in ``results/`` would let them tell.

Everything here is best-effort: a missing package or a git-less checkout records
``None`` for that field rather than failing a training run.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Packages whose version can change results. Recorded by name so adding one is
#: a one-line change rather than a new code path.
TRACKED_PACKAGES = (
    "torch", "torch_geometric", "transformers", "numpy", "pandas",
    "scipy", "sklearn", "yaml",
)


def _package_versions() -> dict:
    versions: dict = {}
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def _git_commit() -> dict:
    """The commit this ran from, and whether the tree was dirty."""
    def run(*args):
        try:
            return subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip() or None
        except Exception:
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit does not identify what ran.
        "dirty": None if status is None else bool(status),
    }


def _device_info() -> dict:
    try:
        import torch
    except Exception:
        return {}
    info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{major}.{minor}"
            info["device_count"] = torch.cuda.device_count()
        except Exception:
            pass
    return info


def _mmseqs_version() -> str | None:
    import shutil

    if not shutil.which("mmseqs"):
        return None
    try:
        out = subprocess.run(["mmseqs", "version"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return out or "present, version unknown"
    except Exception:
        return "present, version unknown"


def provenance() -> dict:
    """
    A record of what produced an artifact, for its manifest.

    Cheap enough to call once per script; never raises.
    """
    try:
        import ViennaRNA  # noqa: F401
        vienna = True
    except Exception:
        vienna = False

    return {
        "git": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "device": _device_info(),
        "mmseqs": _mmseqs_version(),
        "vienna_rna_importable": vienna,
        "note": (
            "recorded so a result can be reproduced, or at least attributed to a "
            "stack. `git.dirty` true means the commit does not identify the code "
            "that ran (R2-15)."
        ),
    }


__all__ = ["provenance", "TRACKED_PACKAGES"]
