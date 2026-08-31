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
  * ``rng_state`` / ``restore_rng_state`` snapshot the global generators, so a
    resumed run continues the same trajectory for initialisation and dropout
    (R1-18).
  * ``reseed_loaders_for_epoch`` makes each loader's batch order a pure
    function of ``(seed, epoch)``. Round-2 review, R2-12: a DataLoader carries
    its OWN ``torch.Generator``, seeded once at construction and living outside
    the checkpointed global state, so a run resumed after epoch 1 replayed
    epoch 1's exact batch order as epoch 2 - while printing
    ``rng_restored=True``. Deriving the order from ``(seed, epoch)`` instead of
    from accumulated generator state makes resume exact by construction rather
    than by bookkeeping.

Determinism is not free. ``strict=True`` sets
``torch.use_deterministic_algorithms``, which makes some scatter/gather kernels
fall back to slower deterministic implementations, and PyTorch requires
``CUBLAS_WORKSPACE_CONFIG`` to be set for deterministic cuBLAS. It is off by
default and on for the ablation arms, where comparing across seeds makes
run-to-run noise the confound - ``scripts/ablations.py`` passes
``--strict-determinism`` to every arm. (Round-2 review, R2-15: this paragraph
claimed that before any caller did it.)
"""

from __future__ import annotations

import os
import random
import zlib
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


def reseed_loaders_for_epoch(loaders, seed: int, epoch: int) -> int:
    """
    Make every loader's batch order a pure function of ``(seed, epoch)``.

    Call at the top of each epoch. Two kinds of loader are handled:

      * shuffling loaders, whose ``torch.Generator`` is re-seeded;
      * loaders driven by a batch sampler with ``set_epoch`` (the
        node-budget sampler), which is told the epoch directly.

    Round-2 review, R2-12: without this, resume rewound the data order to
    epoch 1 and shifted the whole schedule, under an affirmative
    ``rng_restored=True``. Note this was only detectable because R1-01 made
    the loader deterministic in the first place - the fix is what made the
    omission visible.

    Returns the number of loaders re-seeded, so a caller can assert that the
    call did something rather than silently matching nothing.
    """
    touched = 0
    for task in sorted(loaders):
        splits = loaders[task]
        for split in sorted(splits):
            loader = splits[split]
            # A distinct stream per (task, split) so adding a task cannot
            # change another task's order. CRC32, not hash(): str hashing is
            # salted per process, so hash() here would make the batch order
            # differ between a fresh run and a resumed one - the very defect
            # this function exists to remove.
            stream = zlib.crc32(f"{task}/{split}".encode()) % 100_003
            derived = (seed * 1_000_003 + epoch * 10_007 + stream) % (2 ** 31 - 1)
            sampler = getattr(loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
                touched += 1
                continue
            gen = getattr(loader, "generator", None)
            if gen is not None:
                gen.manual_seed(derived)
                touched += 1
    return touched


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
    "reseed_loaders_for_epoch",
    "worker_init_fn",
    "loader_kwargs",
    "rng_state",
    "restore_rng_state",
]
