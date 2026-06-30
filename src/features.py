"""
src/features.py
===============
Mechanistic per-residue features: CAI (human table), local charge, and local
mRNA folding energy (ViennaRNA minimum free energy).

Channels returned by ``mechanistic_features`` (shape (max_length, 4)):
    0  mRNA_fold   local minimum-free-energy per nucleotide (ViennaRNA), or
                   GC-content proxy if ViennaRNA is unavailable
    1  CAI         per-codon relative adaptiveness (human reference table)
    2  charge      windowed net charge (per residue)
    3  mech_valid  1.0 if DNA-derived channels (mRNA, CAI) are real for this
                   residue, 0.0 if zeroed (protein-only input, e.g. ClinVar)

The ``mech_valid`` mask handles protein-only inputs: with no codon context,
mRNA/CAI are zeroed and flagged. Downstream code (graph_builder.build_mechanistic_
edges) uses the mask so mechanistic edges are only built where the features exist.
"""

from __future__ import annotations

import math
import torch

from src.codon_tables import get_cai_weights, GENETIC_CODE, back_translate

# ---------------------------------------------------------------------------
# Optional ViennaRNA — real RNA folding. Falls back to a GC proxy if missing.
# ---------------------------------------------------------------------------
try:
    import RNA  # ViennaRNA python bindings
    VIENNA_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    RNA = None
    VIENNA_AVAILABLE = False

N_MECH_CHANNELS = 4  # mRNA_fold, CAI, charge, mech_valid

# Kyte-Doolittle is hydropathy; here we want *charge*. Simple residue charge at
# physiological pH (approx): K,R = +1 ; D,E = -1 ; H = +0.1.
_RESIDUE_CHARGE: dict[str, float] = {
    "K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
}


# ---------------------------------------------------------------------------
# CAI (per-codon relative adaptiveness, human reference)
# ---------------------------------------------------------------------------
def cai_per_residue(dna_seq: str, max_length: int, organism: str = "human") -> list[float]:
    """Per-codon relative adaptiveness w_i from a coding DNA sequence."""
    weights = get_cai_weights(organism)
    dna = dna_seq.upper().replace("U", "T")
    n_codons = len(dna) // 3
    out = [0.0] * max_length
    for i in range(min(n_codons, max_length)):
        codon = dna[i * 3: i * 3 + 3]
        out[i] = weights.get(codon, 0.5)  # 0.5 for unknown/ambiguous codons
    return out


def cai_sequence(dna_seq: str, organism: str = "human") -> float:
    """Geometric-mean CAI over the coding sequence (Sharp & Li, 1987)."""
    weights = get_cai_weights(organism)
    dna = dna_seq.upper().replace("U", "T")
    logs, n = 0.0, 0
    for i in range(len(dna) // 3):
        codon = dna[i * 3: i * 3 + 3]
        if GENETIC_CODE.get(codon, "*") == "*":
            continue
        w = weights.get(codon, 0.5)
        logs += math.log(max(w, 1e-6))
        n += 1
    return math.exp(logs / n) if n else 0.0


# ---------------------------------------------------------------------------
# Charge (windowed net charge)
# ---------------------------------------------------------------------------
def charge_per_residue(protein_seq: str, max_length: int, window: int = 7) -> list[float]:
    half = window // 2
    L = min(len(protein_seq), max_length)
    out = [0.0] * max_length
    for i in range(L):
        lo, hi = max(0, i - half), min(len(protein_seq), i + half + 1)
        seg = protein_seq[lo:hi]
        raw = sum(_RESIDUE_CHARGE.get(a, 0.0) for a in seg)
        out[i] = raw / max(len(seg), 1)
    return out


# ---------------------------------------------------------------------------
# mRNA local folding energy
# ---------------------------------------------------------------------------
def _gc_proxy_per_residue(dna_seq: str, max_length: int, codon_window: int = 3) -> list[float]:
    """Fallback when ViennaRNA is unavailable: local GC fraction, centred at 0."""
    dna = dna_seq.upper().replace("U", "T")
    n_codons = len(dna) // 3
    half = codon_window // 2
    out = [0.0] * max_length
    for i in range(min(n_codons, max_length)):
        lo, hi = max(0, i - half), min(n_codons, i + half + 1)
        seg = dna[lo * 3: hi * 3]
        gc = sum(1 for c in seg if c in "GC")
        out[i] = (gc / max(len(seg), 1)) - 0.5  # centred proxy in [-0.5, 0.5]
    return out


def mrna_fold_per_residue(dna_seq: str, max_length: int, codon_window: int = 7) -> list[float]:
    """
    Local mRNA folding energy per residue.

    If ViennaRNA is present: fold the nucleotide window centred on each codon and
    return MFE normalised per nucleotide (kcal/mol/nt; more negative = more
    structured). Otherwise fall back to a GC proxy (and the caller should set the
    mech_valid mask accordingly via ``mechanistic_features``).
    """
    if not VIENNA_AVAILABLE:
        return _gc_proxy_per_residue(dna_seq, max_length, codon_window=3)

    dna = dna_seq.upper().replace("T", "U")
    n_codons = len(dna) // 3
    half = codon_window // 2
    out = [0.0] * max_length
    cache: dict[tuple[int, int], float] = {}
    for i in range(min(n_codons, max_length)):
        lo, hi = max(0, i - half), min(n_codons, i + half + 1)
        key = (lo, hi)
        if key not in cache:
            window = dna[lo * 3: hi * 3]
            if len(window) >= 4:
                _, mfe = RNA.fold_compound(window).mfe()
                cache[key] = mfe / len(window)  # per-nucleotide
            else:
                cache[key] = 0.0
        out[i] = cache[key]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def mechanistic_features(
    protein_seq: str,
    dna_seq: str | None,
    max_length: int,
    organism: str = "human",
    back_translate_fallback: bool = False,
) -> torch.Tensor:
    """
    Returns (max_length, 4) float32: [mRNA_fold, CAI, charge, mech_valid].

    If ``dna_seq`` is None (protein-only input, e.g. ClinVar), mRNA and CAI are
    zeroed and mech_valid is 0; charge is always computed.

    If ``back_translate_fallback`` is True and no DNA is given, a usage-weighted
    proxy transcript is generated (codon_tables.back_translate) so the
    mechanistic track is defined even without an experimental CDS. This is an
    explicit codon-usage proxy, not a real transcript — it lets the dual-track
    E_mech be demonstrated when only protein sequences are available.
    """
    feat = torch.zeros((max_length, N_MECH_CHANNELS), dtype=torch.float32)
    charge = charge_per_residue(protein_seq, max_length)
    L = min(len(protein_seq), max_length)

    if not dna_seq and back_translate_fallback and L > 0:
        dna_seq = back_translate(protein_seq[:max_length], organism)

    if dna_seq:
        mrna = mrna_fold_per_residue(dna_seq, max_length)
        cai = cai_per_residue(dna_seq, max_length, organism)
        valid = 1.0 if VIENNA_AVAILABLE else 0.0  # GC proxy is not "real" mRNA
        for i in range(L):
            feat[i, 0] = mrna[i]
            feat[i, 1] = cai[i]
            feat[i, 2] = charge[i]
            feat[i, 3] = valid
    else:
        for i in range(L):
            feat[i, 2] = charge[i]
            # channels 0,1,3 stay 0 -> "no codon context" signal
    return feat
