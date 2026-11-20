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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graphs import N_PAIR_GEO
from src.substitution import AA_BIOPHYS, N_AA_BIOPHYS, N_AMINO_ACIDS, AminoAcidEmbedding


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
        b = z.size(0)
        all_aa = torch.arange(n_aa, device=z.device).unsqueeze(0).expand(b, -1)
        z_rep = z.unsqueeze(1).expand(-1, n_aa, -1)
        mu_all, _ = self.score(z_rep.reshape(b * n_aa, -1), all_aa.reshape(-1))
        mu_all = mu_all.view(b, n_aa)
        mu_wt, _ = self.score(z, wt_idx)
        return mu_all - mu_wt


# ---------------------------------------------------------------------------
# Evolutionary-anchored antisymmetric ddG head (Holo-GeoN v2)
# ---------------------------------------------------------------------------
class EvolutionarySubstitutionHead(nn.Module):
    """
    Antisymmetric-by-construction ddG head anchored on ESM-2's wild-type
    marginal log-odds and explicit amino-acid biophysics:

        ddG(wt -> mt) = alpha(z) * (lp[mt] - lp[wt])
                        + [ s_theta(z, e_mt, phi_mt) - s_theta(z, e_wt, phi_wt) ]

    Both terms are strictly odd under ``wt <-> mt``, so exact reversibility
    ``ddG(b->a) == -ddG(a->b)`` holds to floating-point precision in both train
    and eval modes, and single-pass 20 x L deep mutational scanning is exact.

    When ``wt_logprobs`` is absent (legacy cache) or ``use_esm_logits_prior=False``
    (ablation arm), the first term is cleanly zeroed.
    """

    def __init__(
        self,
        ctx_dim: int,
        aa_dim: int = 32,
        hidden: int = 256,
        dropout: float = 0.1,
        aa_embedding: AminoAcidEmbedding | None = None,
        use_esm_logits_prior: bool = True,
    ):
        super().__init__()
        self.aa = aa_embedding or AminoAcidEmbedding(aa_dim)
        self.use_esm_logits_prior = bool(use_esm_logits_prior)
        self.register_buffer("aa_biophys", AA_BIOPHYS.clone(), persistent=False)
        # Deterministic on purpose (_DETERMINISTIC = 0.0) so antisymmetry holds in train() too.
        self.net = _mlp(ctx_dim + self.aa.dim + N_AA_BIOPHYS, hidden, 2, _DETERMINISTIC)
        # Context-dependent positive scaling for evolutionary log-odds prior.
        # Bias -0.5 -> softplus(-0.5) ~ 0.47 kcal/mol per nat at initialization.
        self.alpha_proj = nn.Linear(ctx_dim, 1)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, -0.5)

    def _aa_features(self, aa_idx: torch.Tensor) -> torch.Tensor:
        idx = aa_idx.clamp(0, N_AMINO_ACIDS)
        emb = self.aa(idx)
        bio = self.aa_biophys[idx].to(dtype=emb.dtype)
        return torch.cat([emb, bio], dim=-1)

    def score(self, z: torch.Tensor, aa_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([z, self._aa_features(aa_idx)], dim=-1))
        return out[..., :1], out[..., 1:2].clamp(LOGVAR_MIN, LOGVAR_MAX)

    def _prior_term(
        self,
        z: torch.Tensor,
        wt_idx: torch.Tensor,
        mt_idx: torch.Tensor,
        wt_logprobs: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_esm_logits_prior or wt_logprobs is None:
            return torch.zeros((z.size(0), 1), dtype=z.dtype, device=z.device)
        # Pad an extra 0 column for the unknown-AA slot (index 20)
        lp = wt_logprobs.to(dtype=z.dtype, device=z.device)
        if lp.size(-1) == N_AMINO_ACIDS:
            lp = F.pad(lp, (0, 1), value=0.0)
        b = torch.arange(z.size(0), device=z.device)
        wt_c = wt_idx.clamp(0, N_AMINO_ACIDS)
        mt_c = mt_idx.clamp(0, N_AMINO_ACIDS)
        # If either residue is unknown (20), do not inject spurious log-odds
        known = (wt_c < N_AMINO_ACIDS) & (mt_c < N_AMINO_ACIDS)
        delta_ll = (lp[b, mt_c] - lp[b, wt_c]).masked_fill(~known, 0.0).unsqueeze(-1)
        alpha = F.softplus(self.alpha_proj(z))
        return alpha * delta_ll

    def forward(
        self,
        z: torch.Tensor,
        wt_idx: torch.Tensor,
        mt_idx: torch.Tensor,
        wt_logprobs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_wt, lv_wt = self.score(z, wt_idx)
        mu_mt, lv_mt = self.score(z, mt_idx)
        ddg = (mu_mt - mu_wt) + self._prior_term(z, wt_idx, mt_idx, wt_logprobs)
        logvar = torch.logaddexp(lv_wt.float(), lv_mt.float()).to(ddg.dtype)
        return ddg, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def scan(
        self,
        z: torch.Tensor,
        wt_idx: torch.Tensor,
        n_aa: int = 20,
        wt_logprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = z.size(0)
        all_aa = torch.arange(n_aa, device=z.device).unsqueeze(0).expand(b, -1)
        z_rep = z.unsqueeze(1).expand(-1, n_aa, -1)
        mu_all, _ = self.score(z_rep.reshape(b * n_aa, -1), all_aa.reshape(-1))
        mu_all = mu_all.view(b, n_aa)
        mu_wt, _ = self.score(z, wt_idx)
        ddg = mu_all - mu_wt

        if self.use_esm_logits_prior and wt_logprobs is not None:
            lp = wt_logprobs.to(dtype=z.dtype, device=z.device)
            if lp.size(-1) < n_aa:
                lp = F.pad(lp, (0, n_aa - lp.size(-1)), value=0.0)
            lp = lp[:, :n_aa]
            rows = torch.arange(b, device=z.device)
            wt_c = wt_idx.clamp(0, n_aa - 1)
            wt_lp = lp[rows, wt_c].unsqueeze(-1)              # (B, 1)
            delta_ll = lp - wt_lp                             # (B, n_aa)
            # Zero out unknown slots (>19) and any row whose WT is unknown
            known_mt = (torch.arange(n_aa, device=z.device) < N_AMINO_ACIDS).unsqueeze(0)
            known_wt = (wt_idx < N_AMINO_ACIDS).unsqueeze(-1)
            delta_ll = delta_ll.masked_fill(~(known_mt & known_wt), 0.0)
            alpha = F.softplus(self.alpha_proj(z))            # (B, 1)
            ddg = ddg + alpha * delta_ll
        return ddg


# ---------------------------------------------------------------------------
# Contact-Gated Potts Bilinear Epistasis Head (Double Mutants — Holo-GeoN v2)
# ---------------------------------------------------------------------------
class ContactGatedEpistasisHead(nn.Module):
    """
    Physically grounded double-mutant epistasis head gated by predicted contact
    topology ``pair_geo`` from ``E_geo``:

        u_i(a) = U_theta(z_i, e_a, phi_a)               in R^D_p
        du_i   = u_i(mt_i) - u_i(wt_i)
        du_j   = u_j(mt_j) - u_j(wt_j)
        dddG(i, j) = g_theta(pair_geo) * (du_i^T du_j) / sqrt(D_p)

    Mathematical guarantees (verified in ``tests/test_v2.py``):
      1. **Zero boundary epistasis:** if ``mt_i == wt_i`` or ``mt_j == wt_j``,
         ``dddG == 0`` identically.
      2. **Exact site-order symmetry:** swapping ``(i, wt_i, mt_i) <-> (j, wt_j, mt_j)``
         gives identical ``dddG`` to floating-point precision.
      3. **Exact Potts cycle parity:** ``dddG`` is the exact second finite
         difference of the pairwise Potts potential ``g_ij * u_i(a)^T u_j(b)``,
         so it is EVEN under simultaneous reversal ``(wt_i, wt_j) <-> (mt_i, mt_j)``
         and ODD under reversing a single site.
      4. **Contact-gated physics:** ``pair_geo = [p_ij, log_sep, inv_hop, is_contact]``
         makes ``E_geo`` mechanistically irreplaceable without requiring PDB input.
    """

    def __init__(
        self,
        ctx_dim: int,
        aa_dim: int = 32,
        potts_dim: int = 64,
        pair_geo_dim: int = N_PAIR_GEO,
        aa_embedding: AminoAcidEmbedding | None = None,
        use_contact_gate: bool = True,
    ):
        super().__init__()
        self.aa = aa_embedding or AminoAcidEmbedding(aa_dim)
        self.potts_dim = potts_dim
        self.use_contact_gate = bool(use_contact_gate)
        self.register_buffer("aa_biophys", AA_BIOPHYS.clone(), persistent=False)
        # Deterministic for exact symmetry/parity in both train() and eval().
        self.potts_net = _mlp(ctx_dim + self.aa.dim + N_AA_BIOPHYS, potts_dim * 2, potts_dim, _DETERMINISTIC)
        self.gate_net = nn.Sequential(
            nn.Linear(pair_geo_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Start near zero epistasis so single-mutant additive signal dominates early.
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 0.1)
        self.const_gate = nn.Parameter(torch.tensor(0.1))

    def _u(self, z: torch.Tensor, aa_idx: torch.Tensor) -> torch.Tensor:
        idx = aa_idx.clamp(0, N_AMINO_ACIDS)
        emb = self.aa(idx)
        bio = self.aa_biophys[idx].to(dtype=emb.dtype)
        return self.potts_net(torch.cat([z, emb, bio], dim=-1))

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        wt_i: torch.Tensor,
        mt_i: torch.Tensor,
        wt_j: torch.Tensor,
        mt_j: torch.Tensor,
        pair_geo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        du_i = self._u(z_i, mt_i) - self._u(z_i, wt_i)
        du_j = self._u(z_j, mt_j) - self._u(z_j, wt_j)
        raw = (du_i * du_j).sum(dim=-1, keepdim=True) / math.sqrt(self.potts_dim)
        if self.use_contact_gate and pair_geo is not None:
            gate = self.gate_net(pair_geo.to(dtype=raw.dtype, device=raw.device))
        else:
            gate = self.const_gate
        return gate * raw


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
    "EvolutionarySubstitutionHead",
    "ContactGatedEpistasisHead",
    "ThreeStateHead",
    "PathogenicityHead",
    "IdrEnsembleHead",
    "MfiHead",
    "ProteomicsHead",
]
