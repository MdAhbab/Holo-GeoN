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

The split-assignment logic is pure and unit-tested
(``tests/test_round2_fixes.py``, and end to end in ``tests/test_pipeline.py``).

A split is only as good as its grouping relation, and ``verify_no_leakage``
can only see the relation it is handed: group by the exact sequence and it
returns True on a deep mutational scan whose test rows differ from training
rows by one residue (round-2 review, R2-04). :func:`near_duplicate_audit`
is the check that can actually fail - it measures sequence similarity across
the split boundary rather than trusting the key.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import random
import hashlib
import zlib
from collections import defaultdict


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
    Cluster sequences with MMseqs2 at ``min_seq_id`` identity.
    Returns {sequence_id: cluster_representative_id}. Requires the mmseqs binary.

    This runs ``createdb`` -> ``cluster`` -> ``createtsv`` rather than the
    single ``easy-cluster`` convenience wrapper. Those are the same three steps
    easy-cluster performs to produce the cluster assignment, and the assignment
    is the only thing this function returns.

    The two steps that are NOT run are ``result2repseq`` and ``result2flat``,
    which build the ``_rep_seq.fasta`` and ``_all_seqs.fasta`` files. Nothing
    here reads those, and ``result2flat`` segfaults intermittently in the
    official Windows (cygwin) build - after the cluster TSV has already been
    written, so easy-cluster would fail on a result that was complete and
    correct. Not asking for the artifact is better than tolerating a crash while
    producing it, because a real clustering failure must still be an error.
    """
    if not mmseqs_available():
        raise RuntimeError(
            "mmseqs not found on PATH. Install MMseqs2 (conda install -c bioconda "
            "mmseqs2, or the win64 build from the MMseqs2 releases page) or use "
            "the parent-sequence fallback (group_key_for_sequence)."
        )
    # mmseqs refuses to overwrite an existing output database ("clu.dbtype
    # exists already!"), so a second call with the same tmp_dir would fail on
    # the leftovers of the first. easy-cluster hid this by generating a fresh
    # hashed subdirectory per run; this scratch space belongs to this function,
    # so clearing it is the equivalent.
    work = os.path.join(tmp_dir, "work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    fasta = os.path.join(tmp_dir, "input.fasta")
    with open(fasta, "w") as fh:
        for sid, seq in id_to_seq.items():
            fh.write(f">{sid}\n{seq}\n")

    db = os.path.join(work, "db")
    clu = os.path.join(work, "clu")
    tsv = os.path.join(tmp_dir, "clu_cluster.tsv")

    def mmseqs(*argv: str) -> None:
        # mmseqs reports its errors on stdout, not stderr, so a bare exit code
        # would discard the one line that says what went wrong.
        proc = subprocess.run(["mmseqs", *argv], capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stdout or "").strip().splitlines()[-5:]
            raise RuntimeError(
                f"mmseqs {argv[0]} failed (exit {proc.returncode}): "
                + " | ".join(tail)
            )

    mmseqs("createdb", fasta, db)
    mmseqs("cluster", db, clu, os.path.join(work, "tmp"),
           "--min-seq-id", str(min_seq_id), "-c", str(coverage),
           "--cov-mode", "0", "--threads", str(threads))
    mmseqs("createtsv", db, db, clu, tsv, "--threads", str(threads))

    clusters: dict[str, str] = {}
    with open(tsv) as fh:
        for line in fh:
            if not line.strip():
                continue
            rep, member = line.rstrip("\n").split("\t")
            clusters[member] = rep
    if len(clusters) != len(id_to_seq):
        raise RuntimeError(
            f"mmseqs clustered {len(clusters)} of {len(id_to_seq)} sequences. "
            f"A partial clustering would silently drop proteins from the split, "
            f"so this is an error rather than a warning."
        )
    return clusters


def group_key_for_sequence(seq: str) -> str:
    """Fallback grouping key: hash of the exact parent sequence (protein-disjoint)."""
    return hashlib.md5(seq.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Grouping: connected components under "same cluster OR same declared group"
# ---------------------------------------------------------------------------
class UnionFind:
    """Minimal union-find over string keys."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(
    parent_seqs: list,
    declared: list | None = None,
    identity: float = 0.30,
    coverage: float = 0.80,
) -> tuple:
    """
    Per-row group keys: connected components under "same identity cluster OR
    same declared group".

    ``parent_seqs`` is the per-row parent protein. ``declared`` is an optional
    per-row grouping key that the data itself supplies - a gene symbol for
    ClinVar, a parent/construct id for a deep mutational scan. It is what makes
    a DMS splittable: every variant is a distinct sequence and therefore its own
    cluster, so without it the grouping relation collapses to the identity
    relation (round-2 review, R2-04).

    Returns ``(groups, info)``; ``info`` records which clustering relation was
    actually used, so an artifact can never imply MMseqs2 ran when it did not.
    """
    parents = sorted(set(parent_seqs))
    info: dict = {"n_parent_sequences": len(parents), "n_rows": len(parent_seqs)}

    seq_to_cluster: dict[str, str] = {}
    if mmseqs_available():
        keyed = {group_key_for_sequence(s): s for s in parents}
        clusters = cluster_with_mmseqs(keyed, min_seq_id=identity, coverage=coverage)
        for s in parents:
            k = group_key_for_sequence(s)
            seq_to_cluster[s] = clusters.get(k, k)
        info["clustering"] = f"mmseqs2 @ {identity:.0%} identity, {coverage:.0%} coverage"
    else:
        print(
            "[splits] mmseqs not found on PATH - falling back to exact "
            "parent-sequence grouping. This is still protein-disjoint, but a "
            "close homolog can appear on both sides. Install MMseqs2 "
            "(conda install -c bioconda mmseqs2) for identity-clustered splits."
        )
        for s in parents:
            seq_to_cluster[s] = group_key_for_sequence(s)
        info["clustering"] = "exact parent-sequence identity (MMseqs2 unavailable)"

    uf = UnionFind()
    for s in parents:
        uf.union(f"c:{seq_to_cluster[s]}", f"s:{s}")

    has_declared = bool(declared) and any(
        str(g) != str(s) for g, s in zip(declared, parent_seqs)
    )
    if has_declared:
        for g, seq in zip(declared, parent_seqs):
            uf.union(f"g:{g}", f"s:{seq}")
    info["declared_group_relation_used"] = bool(has_declared)

    groups = [uf.find(f"s:{s}") for s in parent_seqs]
    info["n_groups"] = len(set(groups))
    return groups, info


# ---------------------------------------------------------------------------
# Near-duplicate audit: the check that can actually fail
# ---------------------------------------------------------------------------
KMER = 8
SKETCH_SIZE = 32
#: Estimated k-mer Jaccard at or above which two proteins count as
#: near-duplicates. One point mutation in a 200-residue protein removes ~2*KMER
#: of ~L k-mers, so DMS variants of one parent sit near 0.9; unrelated proteins
#: sit near 0. The threshold is nowhere near either, so it is not a tuned knob.
NEAR_DUPLICATE_JACCARD = 0.5
#: Sequences kept per sketch bucket. Without a cap a DMS puts every variant in
#: every bucket and the audit becomes quadratic; a near neighbour that exists at
#: all is overwhelmingly likely to survive the cap.
_BUCKET_CAP = 32


def _kmer_sketch(seq: str, k: int = KMER, size: int = SKETCH_SIZE) -> frozenset:
    """
    Bottom-``size`` hashed k-mers of a sequence: a cheap MinHash sketch.

    The fraction of sketch values two sequences share estimates the Jaccard
    similarity of their k-mer sets. CRC32 rather than ``hash()`` because the
    latter is salted per process, which would make the audit unreproducible.
    """
    if len(seq) <= k:
        return frozenset({zlib.crc32(seq.encode())})
    values = {zlib.crc32(seq[i:i + k].encode()) for i in range(len(seq) - k + 1)}
    return frozenset(sorted(values)[:size])


def near_duplicate_audit(
    train_seqs: list,
    held_seqs: list,
    threshold: float = NEAR_DUPLICATE_JACCARD,
    max_held: int = 2000,
) -> dict:
    """
    Measure how many held-out proteins have a near-identical training protein.

    Round-2 review, R2-04: ``verify_no_leakage`` checks that no *group key*
    spans two splits, which is vacuous when the key is the row's own sequence.
    The reviewer built an MFI-shaped table - 426 single mutants of one
    200-residue protein - and got 426 groups, ``verify_no_leakage -> True``, and
    100% of test rows within 1-2 residues of a training row. A split check that
    cannot fail on that is not a check.

    This compares the sequences themselves across the boundary, so it fails
    exactly where the group-key check is blind. Deterministic, and roughly
    linear in the number of sequences.

    Returns the measured fraction, the threshold used, and up to three example
    pairs, so a failure is actionable rather than a bare number.
    """
    train_uniq = sorted(set(train_seqs))
    held_uniq = sorted(set(held_seqs))
    if not train_uniq or not held_uniq:
        return {
            "n_train": len(train_uniq),
            "n_held_checked": 0,
            "fraction_with_near_train_neighbour": 0.0,
            "jaccard_threshold": threshold,
            "examples": [],
            "note": "one side of the split is empty; nothing to compare",
        }

    # A RANDOM sample, not an alphabetical prefix.
    #
    # This was `held_uniq[:max_held]`, and `held_uniq` is `sorted(set(...))` -
    # so the "sample" was the alphabetically-first max_held amino-acid strings.
    # On the shipped ClinVar split that silently skipped 984 of 2984 held-out
    # proteins (33%), and the two proteins that DO have a near-identical
    # training neighbour both sit in the unchecked tail. The audit reported
    # 0.0 leakage, `NEAR_DUPLICATE_TOLERANCE` is 0.0, and the gate in
    # scripts/make_splits.py raises on any positive value - so the split was
    # written only because the check never looked at the rows that would have
    # stopped it. A guard whose blind spot is a deterministic function of the
    # data is not a guard.
    #
    # Seeded, so the audit stays reproducible.
    if len(held_uniq) > max_held:
        sampled = sorted(random.Random(0).sample(held_uniq, max_held))
        sampling_note = (
            f"random sample of {max_held} of {len(held_uniq)} held-out proteins "
            f"(seeded); the reported fraction is an estimate"
        )
    else:
        sampled = held_uniq
        sampling_note = "every held-out protein was checked"
    sketches = [_kmer_sketch(s) for s in train_uniq]
    index: dict = defaultdict(list)
    for i, sk in enumerate(sketches):
        for h in sk:
            bucket = index[h]
            if len(bucket) < _BUCKET_CAP:
                bucket.append(i)

    hits = 0
    examples: list = []
    for seq in sampled:
        sk = _kmer_sketch(seq)
        counts: dict = defaultdict(int)
        for h in sk:
            for i in index.get(h, ()):
                counts[i] += 1
        if not counts:
            continue
        best_i, shared = max(counts.items(), key=lambda kv: kv[1])
        best_sk = sketches[best_i]
        union = len(sk | best_sk)          # |A n B| / |A u B| over the sketches
        jac = shared / union if union else 0.0
        if jac >= threshold:
            hits += 1
            if len(examples) < 3:
                examples.append({
                    "held_out_length": len(seq),
                    "nearest_train_length": len(train_uniq[best_i]),
                    "estimated_kmer_jaccard": round(jac, 3),
                })

    return {
        "n_train": len(train_uniq),
        "n_held_checked": len(sampled),
        "n_held_unique": len(held_uniq),
        "fraction_with_near_train_neighbour": round(hits / len(sampled), 4),
        "jaccard_threshold": threshold,
        "kmer": KMER,
        "examples": examples,
        "sampling": sampling_note,
        "method": (
            "bottom-32 MinHash over 8-mers; a held-out protein counts as leaked "
            "when its estimated k-mer Jaccard against some training protein is "
            "at or above the threshold. Measures the sequences, not the group key"
        ),
    }


__all__ = [
    "assign_groups_to_splits", "row_splits_from_groups", "verify_no_leakage",
    "mmseqs_available", "cluster_with_mmseqs", "group_key_for_sequence",
    "UnionFind", "build_groups", "near_duplicate_audit",
    "NEAR_DUPLICATE_JACCARD",
]
