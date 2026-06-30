"""
src/idr_prior.py
================
Polymer-physics PRIOR for intrinsically-disordered-region (IDR) dimensions.

Note
----
This is a citable physics prior used as the training target for IDREnsembleHead:
a sequence-dependent polymer-scaling estimate of R_g, not an experimental
measurement. Experimental/simulation labels (e.g. PED, ALBATROSS) can be
substituted as targets for an experimentally-grounded version.

Model
-----
Flory scaling:  R_g = R_0 * N^nu   (Flory, 1953; Hofmann et al. 2012 for IDRs)
    N   = chain length (residues)
    R_0 ~ 0.20 nm  (≈ 2.0 Å) prefactor
    nu  = apparent scaling exponent in [~0.33 collapsed, ~0.60 self-avoiding].

We estimate a sequence-dependent nu from net charge per residue (NCPR) and mean
hydropathy — both known drivers of IDR expansion/compaction (Das & Pappu 2013;
Holehouse & Pappu 2018). This is a transparent proxy, not a fitted predictor;
swap in a published nu-predictor (e.g. ALBATROSS) for higher fidelity.
"""

from __future__ import annotations

import math

# Kyte-Doolittle hydropathy (normalised to [0,1] below)
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_CHARGE = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0, "H": 0.1}

R0_ANGSTROM = 2.0
NU_COLLAPSED = 0.33
NU_SAW = 0.60


def estimate_nu(seq: str) -> float:
    """Sequence-dependent apparent scaling exponent in [0.33, 0.60]."""
    if not seq:
        return 0.5
    ncpr = abs(sum(_CHARGE.get(a, 0.0) for a in seq)) / len(seq)      # 0..1
    mean_kd = sum(_KD.get(a, 0.0) for a in seq) / len(seq)
    hydro = (mean_kd + 4.5) / 9.0                                      # ~0..1
    # high charge -> expansion (toward SAW); high hydropathy -> compaction
    nu = 0.50 + 0.30 * ncpr - 0.25 * (hydro - 0.5)
    return float(min(NU_SAW, max(NU_COLLAPSED, nu)))


def rg_prior(seq: str) -> float:
    """Prior radius of gyration (Angstrom) for an IDR of this sequence."""
    n = max(len(seq), 1)
    return R0_ANGSTROM * (n ** estimate_nu(seq))


def ree_prior(seq: str) -> float:
    """End-to-end distance prior; R_ee ≈ sqrt(6) * R_g for a Gaussian-like chain."""
    return math.sqrt(6.0) * rg_prior(seq)
