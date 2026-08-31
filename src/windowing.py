"""
src/windowing.py
================
Mutation-centered windowing.

Why this module exists
----------------------
The v1 pipeline head-truncated every sequence to its first ``max_length``
residues. For MegaScale domains (short) that is harmless, but ClinVar proteins
are full length: with ``max_length=256`` and a median mutation position of 383,
roughly 64% of ClinVar variants had their mutation truncated away entirely. For
those rows the wild-type and mutant inputs were byte-identical, so the model was
asked to classify a difference that was not present in its input.

Windowing centres the encoded region on the mutated residue instead of the
sequence start, so the mutation is always present and always in context. This
is applied unconditionally — it is not a flag, because "sometimes drop the
mutation" is not a mode anyone wants.

Windows are sliced from **full-length cached embeddings** rather than
re-embedding the window in isolation: a residue representation computed with
the whole protein in context is strictly more informative than one computed
from a 256-residue fragment.
"""

from __future__ import annotations


class MutationParseError(ValueError):
    """Raised when a wild-type/mutant pair is not a single substitution."""


# ---------------------------------------------------------------------------
# Locating the mutation
# ---------------------------------------------------------------------------
def mutation_position(wt_seq: str, mt_seq: str) -> int:
    """
    Zero-based index of the single substituted residue.

    Raises :class:`MutationParseError` for length mismatches (indels) or for a
    pair differing at zero or more than one position. Callers that want to
    tolerate those rows should catch it and drop the row explicitly — silently
    falling back to position 0 is what produced head-truncated variants before.
    """
    if len(wt_seq) != len(mt_seq):
        raise MutationParseError(
            f"length mismatch ({len(wt_seq)} vs {len(mt_seq)}): not a substitution"
        )
    diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mt_seq)) if a != b]
    if len(diffs) != 1:
        raise MutationParseError(f"expected exactly 1 differing residue, found {len(diffs)}")
    return diffs[0]


# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------
def window_bounds(length: int, site: int, window: int) -> tuple[int, int]:
    """
    Half-open ``[start, end)`` of a ``window``-long slice centred on ``site``.

    The window is clamped to the sequence, so a site near either terminus
    yields a full-length window that is simply off-centre rather than a short
    one. When ``length <= window`` the whole sequence is returned.

    Guarantees ``start <= site < end`` for any ``0 <= site < length``.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if not 0 <= site < length:
        raise ValueError(f"site {site} out of range for length {length}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    if length <= window:
        return 0, length

    half = window // 2
    start = site - half
    start = max(0, min(start, length - window))
    return start, start + window


def apply_window(seq: str, site: int, window: int) -> tuple[str, int, int]:
    """
    Slice ``seq`` to a window centred on ``site``.

    Returns ``(subsequence, site_within_window, window_start)`` where
    ``subsequence[site_within_window] == seq[site]`` always holds.
    """
    start, end = window_bounds(len(seq), site, window)
    return seq[start:end], site - start, start


def window_for_pair(
    wt_seq: str, mt_seq: str, window: int
) -> tuple[str, str, int, int]:
    """
    Window a wild-type/mutant pair on their single substitution.

    Both sequences are sliced with the *same* bounds, so the returned windows
    are aligned and differ at exactly ``site_within_window``.

    Returns ``(wt_window, mt_window, site_within_window, window_start)``.
    """
    site = mutation_position(wt_seq, mt_seq)
    start, end = window_bounds(len(wt_seq), site, window)
    return wt_seq[start:end], mt_seq[start:end], site - start, start


__all__ = [
    "MutationParseError",
    "mutation_position",
    "window_bounds",
    "apply_window",
    "window_for_pair",
]
