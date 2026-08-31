"""
src/substitution.py
===================
The amino-acid alphabet and the substitution encoding the heads condition on.

The model does not encode a mutant protein as a separate sequence. It encodes
the parent once and conditions a scoring function on *which* residue sits at the
mutated site:

    ddG(wt -> mt) = s(z, e_mt) - s(z, e_wt)

``e_aa`` is a learned embedding of an amino-acid identity, and ``z`` is the
site-anchored context from ``src/readout.py``. Two consequences fall out of the
form itself rather than from a training penalty:

  * **Exact reversibility.** ddG(b -> a) = -ddG(a -> b) identically, so the
    thermodynamic constraint holds to floating-point precision. v1 enforced it
    with a soft penalty and still measured mean violations of 0.029-0.071
    kcal/mol.
  * **Cheap deep mutational scans.** All 19 substitutions at a site reuse the
    same ``z``, and ``z`` itself depends only on the parent, so a full 20 x L
    scan costs **one encoder pass for the protein** plus L cheap backbone
    passes - against 20 x L x 2 encoder passes for a re-encode-the-mutant
    formulation. (Round-2 review, R2-05: this sentence was accurate about the
    architecture and false about the code, which ran one encoder pass per
    position until the scan loop was actually hoisted.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# One exception type for "this is not a single substitution", shared with
# src/windowing.py. Two same-named classes in different modules meant an
# `except` on one silently failed to catch the other.
from src.windowing import MutationParseError

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
#: Residues that appear in real sequence databases but are not one of the 20.
#: B = Asx, Z = Glx, X = any, U = selenocysteine, O = pyrrolysine. They map to
#: UNKNOWN_AA_IDX and are flagged by the aa_known mechanistic channel.
NON_STANDARD_AA = "BZXUO"
AA_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(AA_ORDER)}
N_AMINO_ACIDS = len(AA_ORDER)
UNKNOWN_AA_IDX = N_AMINO_ACIDS          # a dedicated slot, never a silent 0


def aa_index(aa: str) -> int:
    """Index of a one-letter amino acid; ``UNKNOWN_AA_IDX`` for anything else."""
    return AA_TO_IDX.get(aa.upper(), UNKNOWN_AA_IDX)


def parse_mutation(wt_seq: str, mutation: str) -> tuple[str, int, str, str]:
    """
    Parse ``<WT><1-based position><MUT>`` (e.g. ``"L8P"``) against a sequence.

    Returns ``(mutant_sequence, zero_based_index, wt_aa, mut_aa)`` and raises
    :class:`MutationParseError` on a malformed code, an out-of-range position,
    a non-standard residue, or a wild-type mismatch. The mismatch check is not
    optional: a silently wrong reference residue produces a plausible number
    for the wrong mutation.
    """
    code = mutation.strip().upper()
    if len(code) < 3:
        raise MutationParseError(f"malformed mutation code: {mutation!r}")

    wt_aa, mut_aa, pos_str = code[0], code[-1], code[1:-1]
    if not pos_str.isdigit():
        raise MutationParseError(f"could not parse a position from {mutation!r}")
    pos = int(pos_str)

    if wt_aa not in AA_TO_IDX or mut_aa not in AA_TO_IDX:
        raise MutationParseError(f"non-standard amino acid in {mutation!r}")
    if not 1 <= pos <= len(wt_seq):
        raise MutationParseError(
            f"position {pos} is outside the sequence (length {len(wt_seq)})"
        )

    idx = pos - 1
    if wt_seq[idx].upper() != wt_aa:
        raise MutationParseError(
            f"wild-type mismatch: sequence has {wt_seq[idx]!r} at position {pos}, "
            f"mutation code says {wt_aa!r}"
        )
    return wt_seq[:idx] + mut_aa + wt_seq[idx + 1:], idx, wt_aa, mut_aa


class AminoAcidEmbedding(nn.Module):
    """Learned embedding over the 20 standard residues plus an unknown slot."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(N_AMINO_ACIDS + 1, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)


__all__ = [
    "AA_ORDER", "NON_STANDARD_AA", "AA_TO_IDX", "N_AMINO_ACIDS", "UNKNOWN_AA_IDX",
    "MutationParseError", "aa_index", "parse_mutation", "AminoAcidEmbedding",
]
