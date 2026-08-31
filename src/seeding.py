"""
src/seeding.py
==============
Determinism control.

Round-1 review, R1-01: the configuration carried ``seed: 42`` and it reached
exactly two places — the split assignment and one row permutation. Model
initialisation, dropout and batch order were all unseeded, and two runs of the
same command on identical data differed by **0.030 test Pearson**. That is
larger than most published ablation effects, which made every single-run
``delta_vs_full`` in the ablation table uninterpretable.

Everything stochastic now goes through here:

  * ``seed_everything`` covers ``random``, ``numpy``, ``torch`` (CPU and all
    CUDA devices), and optionally cuDNN/algorithm determinism.
  * ``loader_kwargs`` supplies the generator and ``worker_init_fn`` a DataLoader
    needs so shuffling and per-worker RNG are reproducible.
  * ``rng_state`` / ``restore_rng_state`` snapshot every generator so a resumed
    run continues the same trajectory rather than a fresh one (R1-18).

Determinism is not free. ``strict=True`` sets
``torch.use_deterministic_algorithms``, which makes some scatter/gather kernels
fall back to slower deterministic implementations, and PyTorch requires
``CUBLAS_WORKSPACE_CONFIG`` to be set for deterministic cuBLAS. It is off by
default and on for the ablation arms, where reproducibility is the point.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, strict: bool = False) -> int:
    """
    Seed every generator this project draws from.

    Args:
        seed   : the seed.
        strict : also request deterministic algorithms and disable cuDNN
                 benchmarking. Slower, but makes a rerun bit-reproducible.

    Returns the seed, so callers can record it in a run manifest.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict:
        # Required by PyTorch for deterministic cuBLAS GEMMs; must be set
        # before the first CUDA context is created to have any effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - torch version dependent
            print(f"[seed] deterministic algorithms unavailable: {exc}")
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible RNG stream."""
    base = torch.initial_seed() % (2 ** 31 - 1)
    seed = (base + worker_id) % (2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)


def loader_kwargs(seed: int, num_workers: int) -> dict[str, Any]:
    """
    Keyword arguments that make a DataLoader's shuffling reproducible.

    Passed to every DataLoader in the project; without the generator, shuffle
    order is drawn from the global RNG and changes with any unrelated call that
    consumes randomness.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {"generator": generator}
    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn
    return kwargs


def rng_state() -> dict[str, Any]:
    """Snapshot every generator, for a checkpoint."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    """
    Restore a snapshot from :func:`rng_state`. Returns True if it was applied.

    A resume that cannot restore RNG state is not the same run continued, so
    the caller is told rather than left to assume.
    """
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"])
                        else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as exc:  # pragma: no cover - device-count dependent
            print(f"[seed] could not restore CUDA RNG state: {exc}")
            return False
    return True


__all__ = [
    "seed_everything",
    "worker_init_fn",
    "loader_kwargs",
    "rng_state",
    "restore_rng_state",
]
