"""
src/heads.py
============
Task heads. Every head here has a real training path and a real evaluation
path; nothing ships as an untrained scaffold that documentation could mistake
for a capability.

    SubstitutionScoreHead   ddG, exactly antisymmetric, with calibrated sigma
    ThreeStateHead          destabilising / neutral / stabilising, swap-equivariant
    PathogenicityHead       binary variant-effect logit
    IdrEnsembleHead         (mu, sigma) of the radius of gyration
    MfiHead                 mean fluorescence intensity
    ProteomicsHead          retention time / expression, optionally species-conditioned

The mutation-conditioned heads take the site-anchored context ``z`` from
``src/readout.py`` plus the wild-type and mutant amino-acid identities, rather
than a difference of two whole-protein embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.substitution import AminoAcidEmbedding


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# Heads whose guarantees are structural — exact antisymmetry for ddG, swap
# equivariance for the three-state head — must be *deterministic functions of
# their inputs*. Dropout inside them would draw an independent mask per call,
# so s(z, e_wt) computed for the forward direction would differ from the same
# quantity computed for the reverse, and the invariant would hold only in eval
# mode. Regularisation for these heads lives in the shared readout instead,
# where it is applied once to z and cancels out of the difference.
_DETERMINISTIC = 0.0

# Bounds on predicted log-variance, applied inside the head so that training,
# calibration and serving all see the same sigma. exp(+-10) spans sigma in
# [6.7e-3, 148] kcal/mol, far wider than any real ddG uncertainty, so the clamp
# only bites on a diverging head.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


# ---------------------------------------------------------------------------
# ddG
# ---------------------------------------------------------------------------
class SubstitutionScoreHead(nn.Module):
    """
    Antisymmetric-by-construction ddG head with a calibrated variance.

    A shared scoring network maps (context, residue identity) to a scalar
    stability score and a log-variance. The prediction is the difference of two
    scores at the same site::

        ddG(wt -> mt) = s(z, e_mt) - s(z, e_wt)

    so ``ddG(mt -> wt) == -ddG(wt -> mt)`` exactly. The reverse direction is
    never a second forward pass, and the antisymmetry term in the loss becomes a
    monitor rather than a constraint that has to be fought for.

    The variance of a difference of two independent scores is the sum of their
    variances, which is what ``logvar`` reports. Trained under Gaussian NLL,
    ``mu +/- 1.96 sigma`` is a genuine interval — this is what the interface is
    allowed to draw as a confidence band.
    """

    def __init__(
        self,
        ctx_dim: int,
        aa_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.1,
        aa_embedding: AminoAcidEmbedding | None = None,
    ):
        super().__init__()
        self.aa = aa_embedding or AminoAcidEmbedding(aa_dim)
        # Deterministic on purpose - see _DETERMINISTIC above.
        self.net = _mlp(ctx_dim + self.aa.dim, hidden, 2, _DETERMINISTIC)

    def score(self, z: torch.Tensor, aa_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stability score and log-variance for one residue identity at a site."""
        out = self.net(torch.cat([z, self.aa(aa_idx)], dim=-1))
        return out[..., :1], out[..., 1:2].clamp(LOGVAR_MIN, LOGVAR_MAX)

    def forward(
        self, z: torch.Tensor, wt_idx: torch.Tensor, mt_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z      : (B, ctx_dim) site-anchored context.
            wt_idx : (B,) wild-type residue indices.
            mt_idx : (B,) mutant residue indices.
        Returns:
            ``(ddg (B, 1), logvar (B, 1))``.

        The variance of a difference of two independent scores is the sum of
        their variances. It is combined with ``logaddexp`` rather than
        ``log(exp(a) + exp(b))``: the naive form overflows to ``inf`` in fp16
        once a score log-variance exceeds ~11.09, and ``amp: true`` is the
        shipped default, so an infinite confidence band was reachable at
        inference (round-1 review, R1-21). The clamp lives in ``score`` so
        training, calibration and serving all see the same sigma - previously
        the loss clamped and every consumer did not, so a saturated head was
        trained as sigma=0.0067 and served as sigma=0.0009.
        """
        mu_wt, lv_wt = self.score(z, wt_idx)
        mu_mt, lv_mt = self.score(z, mt_idx)
        ddg = mu_mt - mu_wt
        logvar = torch.logaddexp(lv_wt.float(), lv_mt.float()).to(ddg.dtype)
        return ddg, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def scan(self, z: torch.Tensor, wt_idx: torch.Tensor, n_aa: int = 20) -> torch.Tensor:
        """
        ddG for every substitution at a site, in one shot.

        Args:
            z      : (B, ctx_dim) contexts.
            wt_idx : (B,) wild-type residue indices.
            n_aa   : candidate columns. Pass ``N_AMINO_ACIDS + 1`` to include
                     the unknown-residue slot, which is what makes the
                     wild-type column zero for a non-standard parent residue -
                     with 20 columns and a parent mapped to the unknown index,
                     no column was zero despite the guarantee below
                     (round-1 review, R1-24).
        Returns:
            (B, n_aa) ddG against each candidate residue; the wild-type column
            is exactly 0 because the head is a difference of scores, provided
            the wild-type index is within ``n_aa``.
        """
        b = z.size(0)
        all_aa = torch.arange(n_aa, device=z.device).unsqueeze(0).expand(b, -1)
        z_rep = z.unsqueeze(1).expand(-1, n_aa, -1)
        mu_all, _ = self.score(z_rep.reshape(b * n_aa, -1), all_aa.reshape(-1))
        mu_all = mu_all.view(b, n_aa)
        mu_wt, _ = self.score(z, wt_idx)
        return mu_all - mu_wt


# ---------------------------------------------------------------------------
# Three-state stability
# ---------------------------------------------------------------------------
class ThreeStateHead(nn.Module):
    """
    Three-way stability classification: 0 destabilising, 1 neutral, 2 stabilising.

    The physics that makes ddG antisymmetric makes this task *swap-equivariant*:
    reversing a mutation must turn a destabilising call into a stabilising one
    and leave a neutral call alone. That is built into the form rather than
    hoped for::

        logits(wt, mt) = 0.5 * ( G(wt, mt) + flip(G(mt, wt)) )

    with ``flip`` reversing the class axis. Then ``logits(mt, wt) ==
    flip(logits(wt, mt))`` identically.

    The neutral band comes from a per-measurement uncertainty where the table
    supplies one: ``labels_from_ddg`` takes ``sigma``, so "neutral" means
    "indistinguishable from zero given this experiment's error".

    Round-2 review, R2-11: this used to read "the neutral band is not a
    hardcoded constant", which was true of this function and false of every
    shipped configuration - the ``sigma:`` override is commented out for each
    dataset in ``configs/data.yaml``, so rows fall back to a single number and
    the band IS a constant for them. That number is now the declared config key
    ``three_state_sigma``, recorded in ``metrics.json`` alongside a sensitivity
    table showing how the class balance moves with it. A non-positive ``sigma``
    is treated as missing rather than used.
    """

    def __init__(
        self,
        ctx_dim: int,
        aa_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.1,
        aa_embedding: AminoAcidEmbedding | None = None,
    ):
        super().__init__()
        self.aa = aa_embedding or AminoAcidEmbedding(aa_dim)
        # Deterministic on purpose - see _DETERMINISTIC above.
        self.net = _mlp(ctx_dim + 2 * self.aa.dim, hidden, 3, _DETERMINISTIC)

    def _directional(self, z, a_idx, b_idx) -> torch.Tensor:
        return self.net(torch.cat([z, self.aa(a_idx), self.aa(b_idx)], dim=-1))

    def forward(
        self, z: torch.Tensor, wt_idx: torch.Tensor, mt_idx: torch.Tensor
    ) -> torch.Tensor:
        """(B, 3) class logits, equivariant under wild-type/mutant swap."""
        forward_logits = self._directional(z, wt_idx, mt_idx)
        reverse_logits = self._directional(z, mt_idx, wt_idx).flip(-1)
        return 0.5 * (forward_logits + reverse_logits)

    @staticmethod
    def labels_from_ddg(
        ddg: torch.Tensor, sigma: torch.Tensor | float, n_sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Derive class labels from measured ddG and its experimental uncertainty.

        A mutation counts as neutral when ``|ddG| <= n_sigma * sigma``, i.e.
        when the measurement cannot distinguish it from zero. ``sigma`` may be a
        per-row tensor (preferred, when the source reports per-measurement error)
        or a scalar fallback.
        """
        if not torch.is_tensor(sigma):
            sigma = torch.full_like(ddg, float(sigma))
        band = n_sigma * sigma
        labels = torch.ones_like(ddg, dtype=torch.long)
        labels[ddg > band] = 2
        labels[ddg < -band] = 0
        return labels


# ---------------------------------------------------------------------------
# Pathogenicity
# ---------------------------------------------------------------------------
class PathogenicityHead(nn.Module):
    """
    Binary variant-effect logit from the site-anchored context.

    Deliberately *not* symmetrised: pathogenicity is a property of a directed
    substitution against a reference allele, so wild-type and mutant are not
    interchangeable the way they are for ddG.
    """

    def __init__(
        self,
        ctx_dim: int,
        aa_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.2,
        aa_embedding: AminoAcidEmbedding | None = None,
    ):
        super().__init__()
        self.aa = aa_embedding or AminoAcidEmbedding(aa_dim)
        self.net = _mlp(ctx_dim + 2 * self.aa.dim, hidden, 1, dropout)

    def forward(
        self, z: torch.Tensor, wt_idx: torch.Tensor, mt_idx: torch.Tensor
    ) -> torch.Tensor:
        """(B, 1) raw logit; pair with BCEWithLogitsLoss."""
        return self.net(torch.cat([z, self.aa(wt_idx), self.aa(mt_idx)], dim=-1))


# ---------------------------------------------------------------------------
# Whole-sequence heads
# ---------------------------------------------------------------------------
def _log_length(seq_length: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    """
    Chain length as a log-scaled feature, shaped (B, 1).

    Whole-sequence properties are length-dependent - the radius of gyration
    follows R_g = R_0 N^v - and a pool over residue embeddings cannot recover
    N when the graph spans only part of the chain. Handing it to the head
    directly makes the dependence available rather than implicit.
    """
    if seq_length is None:
        return torch.zeros(ref.size(0), 1, device=ref.device, dtype=ref.dtype)
    return torch.log1p(seq_length.reshape(-1, 1).to(ref.dtype).to(ref.device))


class IdrEnsembleHead(nn.Module):
    """
    Gaussian over the radius of gyration (Angstrom): mean and positive sigma.

    Conditioned on chain length as well as the pooled representation, because
    R_g scales as a power of N and length is the dominant term.
    """

    def __init__(self, ctx_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(ctx_dim + 1, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, 1)
        self.raw_sigma = nn.Linear(hidden, 1)

    def forward(
        self, z: torch.Tensor, seq_length: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(torch.cat([z, _log_length(seq_length, z)], dim=-1))
        sigma = nn.functional.softplus(self.raw_sigma(h)) + 1e-3
        return self.mu(h), sigma


class MfiHead(nn.Module):
    """Mean fluorescence intensity regression (deep mutational scan readouts)."""

    def __init__(self, ctx_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(ctx_dim + 1, hidden, 1, dropout)

    def forward(self, z: torch.Tensor, seq_length: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(torch.cat([z, _log_length(seq_length, z)], dim=-1))


class ProteomicsHead(nn.Module):
    """
    Retention-time / expression regression, optionally conditioned on a species
    identifier for multi-species corpora.
    """

    def __init__(
        self,
        ctx_dim: int,
        hidden: int = 256,
        num_species: int = 0,
        species_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_species = num_species
        in_dim = ctx_dim + 1          # + log chain length
        if num_species > 0:
            self.species = nn.Embedding(num_species, species_dim)
            in_dim += species_dim
        else:
            self.species = None
        self.net = _mlp(in_dim, hidden, 1, dropout)

    def forward(self, z: torch.Tensor, species_id: torch.Tensor | None = None,
                seq_length: torch.Tensor | None = None) -> torch.Tensor:
        z = torch.cat([z, _log_length(seq_length, z)], dim=-1)
        if self.species is not None and species_id is not None:
            z = torch.cat([z, self.species(species_id).view(z.size(0), -1)], dim=-1)
        return self.net(z)


__all__ = [
    "SubstitutionScoreHead",
    "ThreeStateHead",
    "PathogenicityHead",
    "IdrEnsembleHead",
    "MfiHead",
    "ProteomicsHead",
]
