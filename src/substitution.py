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

#: Fixed 8-channel normalized biophysical descriptors per amino acid (+ neutral
#: unknown slot at index 20), ordered identically to AA_ORDER:
#:   0  hydropathy (Kyte-Doolittle, [-4.5, 4.5] -> [0, 1])
#:   1  volume     (side-chain van der Waals A^3 / 240)
#:   2  charge     (net charge at pH 7.4: -1, 0, +0.1, +1)
#:   3  helix      (Chou-Fasman P_alpha, [0.57, 1.51] -> [0, 1])
#:   4  sheet      (Chou-Fasman P_beta,  [0.37, 1.70] -> [0, 1])
#:   5  aromatic   (1.0 for F/W/Y/H, else 0.0)
#:   6  is_pro     (1.0 for P — backbone helix breaker / phi constraint)
#:   7  is_gly     (1.0 for G — missing CB / high backbone entropy)
N_AA_BIOPHYS = 8

_BIOPHYS_RAW: dict[str, tuple[float, float, float, float, float, float, float, float]] = {
    "A": (0.7000, 0.3692,  0.0, 0.9043, 0.3459, 0.0, 0.0, 0.0),
    "C": (0.7778, 0.4521,  0.0, 0.1383, 0.6165, 0.0, 0.0, 0.0),
    "D": (0.1111, 0.4629, -1.0, 0.4681, 0.1278, 0.0, 0.0, 0.0),
    "E": (0.1111, 0.5767, -1.0, 1.0000, 0.0000, 0.0, 0.0, 0.0),
    "F": (0.8111, 0.7913,  0.0, 0.5957, 0.7594, 1.0, 0.0, 0.0),
    "G": (0.4556, 0.2504,  0.0, 0.0000, 0.2857, 0.0, 0.0, 1.0),
    "H": (0.1444, 0.6383,  0.1, 0.4574, 0.3759, 1.0, 0.0, 0.0),
    "I": (1.0000, 0.6946,  0.0, 0.5426, 0.9248, 0.0, 0.0, 0.0),
    "K": (0.0667, 0.7025,  1.0, 0.6277, 0.2782, 0.0, 0.0, 0.0),
    "L": (0.9222, 0.6946,  0.0, 0.6809, 0.6992, 0.0, 0.0, 0.0),
    "M": (0.7111, 0.6788,  0.0, 0.9362, 0.5113, 0.0, 0.0, 0.0),
    "N": (0.1111, 0.4754,  0.0, 0.1064, 0.3910, 0.0, 0.0, 0.0),
    "P": (0.3222, 0.4696,  0.0, 0.0000, 0.1353, 0.0, 1.0, 0.0),
    "Q": (0.1111, 0.5992,  0.0, 0.5745, 0.5489, 0.0, 0.0, 0.0),
    "R": (0.0000, 0.7225,  1.0, 0.4362, 0.4211, 0.0, 0.0, 0.0),
    "S": (0.4111, 0.3708,  0.0, 0.2128, 0.2857, 0.0, 0.0, 0.0),
    "T": (0.4222, 0.4838,  0.0, 0.2766, 0.6165, 0.0, 0.0, 0.0),
    "V": (0.9667, 0.5833,  0.0, 0.5213, 1.0000, 0.0, 0.0, 0.0),
    "W": (0.4000, 0.9492,  0.0, 0.5426, 0.7519, 1.0, 0.0, 0.0),
    "Y": (0.3556, 0.8067,  0.0, 0.1277, 0.8271, 1.0, 0.0, 0.0),
}
_BIOPHYS_UNKNOWN = (0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0)

AA_BIOPHYS: torch.Tensor = torch.tensor(
    [_BIOPHYS_RAW[a] for a in AA_ORDER] + [_BIOPHYS_UNKNOWN],
    dtype=torch.float32,
)


def aa_biophys_table(
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the (21, 8) fixed biophysical residue table on ``device``/``dtype``."""
    return AA_BIOPHYS.to(device=device, dtype=dtype)


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


def parse_multi_mutation(
    wt_seq: str, mutation_spec: str
) -> tuple[str, list[tuple[int, str, str]]]:
    """
    Parse a multi-site mutation specification such as ``"L8P:V15A"``,
    ``"L8P+V15A"``, or ``"L8P,V15A"`` against ``wt_seq``.

    Returns ``(mutant_sequence, [(zero_based_idx, wt_aa, mut_aa), ...])`` sorted
    by position. Rejects duplicate positions or invalid single codes loudly.
    """
    import re

    tokens = [t.strip() for t in re.split(r"[:+,;/\s]+", mutation_spec.strip()) if t.strip()]
    if not tokens:
        raise MutationParseError(f"empty mutation specification: {mutation_spec!r}")
    parsed: list[tuple[int, str, str]] = []
    seen_sites: set[int] = set()
    seq_chars = list(wt_seq)
    for tok in tokens:
        _, idx, wt_aa, mut_aa = parse_mutation(wt_seq, tok)
        if idx in seen_sites:
            raise MutationParseError(
                f"duplicate site {idx + 1} in multi-mutation {mutation_spec!r}"
            )
        seen_sites.add(idx)
        parsed.append((idx, wt_aa, mut_aa))
        seq_chars[idx] = mut_aa
    parsed.sort(key=lambda item: item[0])
    return "".join(seq_chars), parsed


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
    "N_AA_BIOPHYS", "AA_BIOPHYS", "aa_biophys_table",
    "MutationParseError", "aa_index", "parse_mutation", "parse_multi_mutation",
    "AminoAcidEmbedding",
]
