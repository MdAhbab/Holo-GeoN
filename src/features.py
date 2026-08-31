"""
src/features.py
===============
Per-residue mechanistic features.

Eight channels, in a fixed order:

    0  mrna_fold   local mRNA minimum free energy per nucleotide (ViennaRNA)
    1  cai         per-codon relative adaptiveness (reference codon-usage table)
    2  charge      windowed net charge (Henderson-Hasselbalch, pH 7.4)
    3  hydropathy  Kyte-Doolittle, normalised to [0, 1]
    4  volume      side-chain van der Waals volume, normalised to [0, 1]
    5  helix       Chou-Fasman P_alpha helix propensity, normalised to [0, 1]
    6  mech_valid  1.0 where channels 0-1 are real, 0.0 where they are absent
    7  aa_known    1.0 for one of the 20 standard residues, 0.0 for B/Z/X/U/O

Channels 2-5 and 7 are computable from the amino-acid sequence alone and are
always real. Channels 0-1 need a **coding sequence** and are zeroed without one.

On the removal of back-translation
----------------------------------
v1 offered a ``--back-translate`` path that synthesised a proxy transcript when
no CDS was available, so the mechanistic track could be "demonstrated". That
path is gone. A back-translated transcript is a deterministic function of the
protein sequence, which makes CAI and mRNA-MFE deterministic functions of the
protein sequence too — they carry no information the language-model embeddings
do not already carry, so the mechanistic edge set built from them cannot add
signal and can only dilute. The v1 measurement agrees: the single-track model
(r = 0.5309) beat the dual-track model (r = 0.5268) on exactly that setup.

Consequently the mechanistic stream is only meaningful with a real CDS. Where
one is absent, ``mech_valid`` is 0 and the mechanistic edge set is empty for
that sequence. That is a property of the data, not a mode of the code.
"""

from __future__ import annotations

import torch

from src.codon_tables import GENETIC_CODE, get_cai_weights

# ---------------------------------------------------------------------------
# ViennaRNA — real folding. Absent, channels 0-1 are treated as unavailable.
# ---------------------------------------------------------------------------
try:
    import RNA  # ViennaRNA python bindings
    VIENNA_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    RNA = None
    VIENNA_AVAILABLE = False

N_MECH_CHANNELS = 8

CH_MRNA, CH_CAI, CH_CHARGE, CH_HYDRO, CH_VOLUME, CH_HELIX, CH_VALID, CH_AA_KNOWN = range(8)

#: Neutral fallbacks for a non-standard residue (B, Z, X, U, O).
#:
#: Round-1 review, R1-24: the scalar channels fell back inconsistently -
#: hydropathy and helix propensity landed mid-scale, but volume fell back to
#: 0.0, which is SMALLER than glycine. The network was being told "this side
#: chain is smaller than any real residue" rather than "this residue is
#: unknown". Mid-scale defaults plus an explicit aa_known flag say the latter.
_UNKNOWN_HYDRO = 0.5
_UNKNOWN_VOLUME = 0.5
_UNKNOWN_HELIX = 0.5

# --- residue tables -------------------------------------------------------
_RESIDUE_CHARGE: dict[str, float] = {
    "K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
}

KD_HYDROPATHY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

RESIDUE_VOLUME: dict[str, float] = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Chou-Fasman alpha-helix propensities.
HELIX_PROPENSITY: dict[str, float] = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

_KD_MIN, _KD_MAX = -4.5, 4.5
_VOL_MAX = 240.0            # above Trp (227.8 A^3)
_PA_MIN, _PA_MAX = 0.57, 1.51


# ---------------------------------------------------------------------------
# CAI
# ---------------------------------------------------------------------------
def cai_per_residue(dna_seq: str, length: int, organism: str = "human") -> list[float]:
    """Per-codon relative adaptiveness w_i along the coding sequence."""
    weights = get_cai_weights(organism)
    dna = dna_seq.upper().replace("U", "T")
    n_codons = len(dna) // 3
    out = [0.0] * length
    for i in range(min(n_codons, length)):
        codon = dna[i * 3: i * 3 + 3]
        out[i] = weights.get(codon, 0.5)  # unknown/ambiguous codon
    return out


def cai_sequence(dna_seq: str, organism: str = "human") -> float:
    """Geometric-mean CAI over the coding sequence (Sharp & Li, 1987)."""
    import math

    weights = get_cai_weights(organism)
    dna = dna_seq.upper().replace("U", "T")
    logs, n = 0.0, 0
    for i in range(len(dna) // 3):
        codon = dna[i * 3: i * 3 + 3]
        if GENETIC_CODE.get(codon, "*") == "*":
            continue
        logs += math.log(max(weights.get(codon, 0.5), 1e-6))
        n += 1
    return math.exp(logs / n) if n else 0.0


# ---------------------------------------------------------------------------
# Protein-only channels
# ---------------------------------------------------------------------------
def charge_per_residue(protein_seq: str, length: int, window: int = 7) -> list[float]:
    """Windowed mean net charge centred on each residue."""
    half = window // 2
    L = min(len(protein_seq), length)
    out = [0.0] * length
    for i in range(L):
        lo, hi = max(0, i - half), min(len(protein_seq), i + half + 1)
        seg = protein_seq[lo:hi]
        out[i] = sum(_RESIDUE_CHARGE.get(a, 0.0) for a in seg) / max(len(seg), 1)
    return out


def _scaled(table: dict[str, float], aa: str, lo: float, hi: float, default: float) -> float:
    return (table.get(aa, default) - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# mRNA local folding energy
# ---------------------------------------------------------------------------
def mrna_fold_per_residue(dna_seq: str, length: int, codon_window: int = 7) -> list[float]:
    """
    Local mRNA folding energy per residue: ViennaRNA MFE of the nucleotide
    window centred on each codon, normalised per nucleotide (kcal/mol/nt, more
    negative = more structured).

    Requires ViennaRNA. Callers must not invoke this when
    :data:`VIENNA_AVAILABLE` is False — there is deliberately no GC-content
    substitute, because a composition proxy is a deterministic function of the
    sequence and would reintroduce the vacuous-feature problem that
    back-translation had.
    """
    if not VIENNA_AVAILABLE:  # pragma: no cover - guarded by callers
        raise RuntimeError(
            "ViennaRNA is not installed, so real mRNA folding energy is "
            "unavailable. Install it (pip install ViennaRNA) to enable the "
            "mechanistic stream; without it mech_valid stays 0 and E_mech is empty."
        )

    rna = dna_seq.upper().replace("T", "U")
    n_codons = len(rna) // 3
    half = codon_window // 2
    out = [0.0] * length
    cache: dict[tuple[int, int], float] = {}
    for i in range(min(n_codons, length)):
        lo, hi = max(0, i - half), min(n_codons, i + half + 1)
        key = (lo, hi)
        if key not in cache:
            seg = rna[lo * 3: hi * 3]
            cache[key] = (RNA.fold_compound(seg).mfe()[1] / len(seg)) if len(seg) >= 4 else 0.0
        out[i] = cache[key]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def mechanistic_features(
    protein_seq: str,
    dna_seq: str | None = None,
    length: int | None = None,
    organism: str = "human",
) -> torch.Tensor:
    """
    Build the ``(length, N_MECH_CHANNELS)`` mechanistic feature tensor for
    one sequence.

    ``dna_seq`` is the **real** coding sequence for this protein, or None. With
    a CDS and ViennaRNA present, channels 0-1 are populated and ``mech_valid``
    is 1. Otherwise those channels stay 0 and ``mech_valid`` is 0, which is what
    keeps the mechanistic edge set empty for that sequence.
    """
    L_seq = len(protein_seq)
    length = L_seq if length is None else length
    feat = torch.zeros((length, N_MECH_CHANNELS), dtype=torch.float32)
    L = min(L_seq, length)
    if L == 0:
        return feat

    charge = charge_per_residue(protein_seq, length)

    have_cds = bool(dna_seq) and VIENNA_AVAILABLE
    if have_cds:
        mrna = mrna_fold_per_residue(dna_seq, length)
        cai = cai_per_residue(dna_seq, length, organism)

    for i in range(L):
        aa = protein_seq[i].upper()
        known = aa in KD_HYDROPATHY
        feat[i, CH_CHARGE] = charge[i]
        feat[i, CH_AA_KNOWN] = 1.0 if known else 0.0
        if known:
            feat[i, CH_HYDRO] = _scaled(KD_HYDROPATHY, aa, _KD_MIN, _KD_MAX, 0.0)
            feat[i, CH_VOLUME] = min(1.0, RESIDUE_VOLUME[aa] / _VOL_MAX)
            feat[i, CH_HELIX] = _scaled(HELIX_PROPENSITY, aa, _PA_MIN, _PA_MAX, 1.0)
        else:
            feat[i, CH_HYDRO] = _UNKNOWN_HYDRO
            feat[i, CH_VOLUME] = _UNKNOWN_VOLUME
            feat[i, CH_HELIX] = _UNKNOWN_HELIX
        if have_cds:
            feat[i, CH_MRNA] = mrna[i]
            feat[i, CH_CAI] = cai[i]
            feat[i, CH_VALID] = 1.0

    return feat


__all__ = [
    "N_MECH_CHANNELS",
    "VIENNA_AVAILABLE",
    "CH_MRNA", "CH_CAI", "CH_CHARGE", "CH_HYDRO", "CH_VOLUME", "CH_HELIX",
    "CH_VALID", "CH_AA_KNOWN",
    "KD_HYDROPATHY", "RESIDUE_VOLUME", "HELIX_PROPENSITY",
    "mechanistic_features", "cai_per_residue", "cai_sequence",
    "charge_per_residue", "mrna_fold_per_residue",
]
