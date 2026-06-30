"""
src/splits.py
=============
Leakage-controlled splitting for mutation-effect data.

To avoid train/test leakage, this module assigns WHOLE sequence clusters (or, in
fallback, whole parent proteins) to a single split, rather than splitting at the
row level where the same parent protein could appear on both sides.

Primary path: MMseqs2 clustering at a strict identity (default 30%). MMseqs2 is
a binary (not pip-installable), so it runs on YOUR machine; this module shells
out to it and parses the cluster TSV. If MMseqs2 is unavailable, we fall back to
exact parent-sequence grouping, which is still protein-disjoint (no leakage),
just less aggressive than family-level clustering.

The split-assignment logic is pure and unit-tested (tests/test_splits.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import random
import hashlib


# ---------------------------------------------------------------------------
# Pure assignment logic (unit-tested, no external deps)
# ---------------------------------------------------------------------------
def assign_groups_to_splits(
    group_keys: list,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    """
    Map each unique group key -> {'train','val','test'} so that all members of a
    group land in the same split. Deterministic given (group set, ratios, seed).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    uniq = sorted(set(group_keys))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    split_of = {}
    for i, g in enumerate(uniq):
        if i < n_train:
            split_of[g] = "train"
        elif i < n_train + n_val:
            split_of[g] = "val"
        else:
            split_of[g] = "test"
    return split_of


def row_splits_from_groups(group_keys: list, **kw) -> list:
    """Return per-row split labels given a per-row list of group keys."""
    split_of = assign_groups_to_splits(group_keys, **kw)
    return [split_of[g] for g in group_keys]


def verify_no_leakage(group_keys: list, row_splits: list) -> bool:
    """True iff no group key appears in more than one split."""
    seen = {}
    for g, s in zip(group_keys, row_splits):
        if g in seen and seen[g] != s:
            return False
        seen[g] = s
    return True


# ---------------------------------------------------------------------------
# MMseqs2 clustering (runs on the user's machine)
# ---------------------------------------------------------------------------
def mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def cluster_with_mmseqs(
    id_to_seq: dict[str, str],
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    tmp_dir: str = ".mmseqs_tmp",
    threads: int = 8,
) -> dict[str, str]:
    """
    Cluster sequences with MMseqs2 easy-cluster at ``min_seq_id`` identity.
    Returns {sequence_id: cluster_representative_id}. Requires the mmseqs binary.
    """
    if not mmseqs_available():
        raise RuntimeError(
            "mmseqs not found on PATH. Install MMseqs2 (conda install -c bioconda "
            "mmseqs2) or use the parent-sequence fallback (group_key_for_sequence)."
        )
    os.makedirs(tmp_dir, exist_ok=True)
    fasta = os.path.join(tmp_dir, "input.fasta")
    with open(fasta, "w") as fh:
        for sid, seq in id_to_seq.items():
            fh.write(f">{sid}\n{seq}\n")
    out_prefix = os.path.join(tmp_dir, "clu")
    subprocess.run(
        ["mmseqs", "easy-cluster", fasta, out_prefix, os.path.join(tmp_dir, "work"),
         "--min-seq-id", str(min_seq_id), "-c", str(coverage),
         "--cov-mode", "0", "--threads", str(threads)],
        check=True,
    )
    clusters: dict[str, str] = {}
    with open(out_prefix + "_cluster.tsv") as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            clusters[member] = rep
    return clusters


def group_key_for_sequence(seq: str) -> str:
    """Fallback grouping key: hash of the exact parent sequence (protein-disjoint)."""
    return hashlib.md5(seq.encode()).hexdigest()[:12]
