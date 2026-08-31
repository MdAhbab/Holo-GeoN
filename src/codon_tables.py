"""
src/codon_tables.py
===================
Reference codon-usage tables and CAI relative-adaptiveness weights.

Ships a documented *Homo sapiens* codon-usage table as the default and exposes a
clean API so other organisms can be added without touching feature code. CAI is
computed against this fixed reference table as a normalised codon-bias proxy
(a sequence/codon-level signal, not a per-organism, transcript-resolved calculation).

Source of the human fractions: standard Homo sapiens codon-usage fractions
(fraction of each codon within its synonymous family), as tabulated by the
Kazusa Codon Usage Database / commonly reproduced references. Values are the
synonymous-family fractions; the CAI relative adaptiveness w_i is derived as
w_i = f_i / max_j(f_j) over the synonymous family of codon i (Sharp & Li, 1987).
"""

from __future__ import annotations


# Standard genetic code: codon (DNA, T not U) -> amino acid one-letter ('*' = stop)
GENETIC_CODE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Homo sapiens: fraction of usage within each synonymous family (sums to ~1.0/AA).
# Standard reference values (Kazusa / commonly reproduced).
HUMAN_CODON_FRACTION: dict[str, float] = {
    "TTT": 0.45, "TTC": 0.55,
    "TTA": 0.07, "TTG": 0.13, "CTT": 0.13, "CTC": 0.20, "CTA": 0.07, "CTG": 0.41,
    "ATT": 0.36, "ATC": 0.48, "ATA": 0.16, "ATG": 1.00,
    "GTT": 0.18, "GTC": 0.24, "GTA": 0.11, "GTG": 0.47,
    "TCT": 0.18, "TCC": 0.22, "TCA": 0.15, "TCG": 0.06, "AGT": 0.15, "AGC": 0.24,
    "CCT": 0.28, "CCC": 0.33, "CCA": 0.27, "CCG": 0.11,
    "ACT": 0.24, "ACC": 0.36, "ACA": 0.28, "ACG": 0.12,
    "GCT": 0.26, "GCC": 0.40, "GCA": 0.23, "GCG": 0.11,
    "TAT": 0.43, "TAC": 0.57,
    "CAT": 0.41, "CAC": 0.59, "CAA": 0.25, "CAG": 0.75,
    "AAT": 0.46, "AAC": 0.54, "AAA": 0.42, "AAG": 0.58,
    "GAT": 0.46, "GAC": 0.54, "GAA": 0.42, "GAG": 0.58,
    "TGT": 0.45, "TGC": 0.55, "TGG": 1.00,
    "CGT": 0.08, "CGC": 0.19, "CGA": 0.11, "CGG": 0.21, "AGA": 0.20, "AGG": 0.20,
    "GGT": 0.16, "GGC": 0.34, "GGA": 0.25, "GGG": 0.25,
    "TAA": 0.28, "TAG": 0.20, "TGA": 0.52,
}


def relative_adaptiveness(fraction_table: dict[str, float]) -> dict[str, float]:
    """
    Convert synonymous-family fractions to CAI relative adaptiveness weights
    w_i = f_i / max_j(f_j over the synonymous family of i)   (Sharp & Li, 1987).

    Stop codons are assigned w = 1.0 (excluded from coding CAI in practice).
    """
    # group codons by amino acid
    by_aa: dict[str, list[str]] = {}
    for codon, aa in GENETIC_CODE.items():
        by_aa.setdefault(aa, []).append(codon)

    w: dict[str, float] = {}
    for aa, codons in by_aa.items():
        fracs = [fraction_table.get(c, 0.0) for c in codons]
        fmax = max(fracs) if fracs else 0.0
        for c in codons:
            if aa == "*" or fmax <= 0.0:
                w[c] = 1.0
            else:
                w[c] = fraction_table.get(c, 0.0) / fmax
    return w


# Default weights used by features.py
HUMAN_CAI_WEIGHT: dict[str, float] = relative_adaptiveness(HUMAN_CODON_FRACTION)

# Registry so a --organism flag can select tables without touching feature code.
CODON_WEIGHT_TABLES: dict[str, dict[str, float]] = {
    "human": HUMAN_CAI_WEIGHT,
}


def get_cai_weights(organism: str = "human") -> dict[str, float]:
    if organism not in CODON_WEIGHT_TABLES:
        raise KeyError(
            f"Unknown organism '{organism}'. Available: {list(CODON_WEIGHT_TABLES)}"
        )
    return CODON_WEIGHT_TABLES[organism]


# ---------------------------------------------------------------------------
# Back-translation (usage-weighted proxy transcript)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single-codon substitution
# ---------------------------------------------------------------------------
def preferred_codon(aa: str, organism: str = "human") -> str | None:
    """
    The most-used codon for one amino acid under the reference usage table.

    This exists for exactly one purpose: substituting the single codon at a
    mutated site in a REAL coding sequence, so the mutant's mechanistic channels
    can be derived from the same transcript as the parent's
    (``scripts/calibration.py`` reversibility check).

    It is emphatically NOT back-translation. Back-translation synthesises a whole
    transcript from a protein, which makes every mechanistic feature a
    deterministic function of the amino-acid sequence and therefore
    informationally empty - see ``src/features.py``. Here the transcript is real
    and only the mutated codon is chosen; a real variant would carry a specific
    codon that the protein-level data does not record, so the choice is stated
    rather than hidden, and it affects one codon's CAI and its local folding
    window.
    """
    weights = get_cai_weights(organism)
    best, best_w = None, -1.0
    for codon, coded_aa in GENETIC_CODE.items():
        if coded_aa != aa.upper():
            continue
        w = weights.get(codon, 0.0)
        if w > best_w:
            best, best_w = codon, w
    return best
