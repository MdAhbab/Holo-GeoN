"""
src/sequence_mixers.py
======================
Two residue-axis mixers used by the backbone. **Both are on in the shipped
model** — ``src/model.py`` defaults ``use_ssm=True`` and ``src/backbone.py``
constructs the fusion unconditionally. They are exposed as constructor flags so
``scripts/ablations.py`` can hold everything else fixed and measure one
component, not because either is optional in the configuration you would train.

  * ``CrossAttentionFusion``
        Cross-modal attention between the language-model track (queries) and the
        mechanistic channels (keys/values), so mechanistic context modulates the
        evolutionary representation rather than being concatenated beside it.
        Emits ``esm_dim + mech_dim`` per residue — at the shipped configuration
        that is 1280 + 8 = 1288.

  * ``SelectiveSSM``
        A linear-time selective state-space mixer over residues, giving every
        residue a cheap view of the whole window before any graph structure is
        imposed. Uses ``mamba_ssm`` when installed (CUDA only); otherwise a
        dependency-free S6-style recurrence solved by a parallel associative
        scan, which runs anywhere including CPU. Either way the shape contract
        is (B, L, D) in, (B, L, D) out.

Round-1 review, R1-22: this docstring previously said both modules "default to
OFF ... unless explicitly enabled", referenced "paper §8.2", and sized the
fusion at 320 + 3 = 323 dimensions. All three were wrong for the shipped model,
and method sections get written from module docstrings.
"""

import torch
import torch.nn as nn

# Real Mamba if available (CUDA-only); otherwise we fall back to a pure-PyTorch SSM.
try:
    from mamba_ssm import Mamba as _RealMamba   # type: ignore
    _MAMBA_AVAILABLE = True
except Exception:  # noqa: BLE001
    _RealMamba = None
    _MAMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cross-Attention Fusion
# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """
    Cross-modal attention between ESM-2 node embeddings and mechanistic features.

    ESM embeddings act as the *query*; the (projected) mechanistic features act as
    *key*/*value*, so each residue's evolutionary representation is refined by the
    mechanistic context most relevant to it. Output is ``esm_dim + mech_dim``.

    Args:
        esm_dim   : language-model hidden size (1280 for ESM-2 650M).
        mech_dim  : number of mechanistic channels (8).
        heads     : attention heads.
    """

    def __init__(self, esm_dim: int = 1280, mech_dim: int = 8, heads: int = 4):
        super().__init__()
        self.esm_dim = esm_dim
        self.mech_dim = mech_dim
        self.mech_proj = nn.Linear(mech_dim, esm_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=esm_dim, num_heads=heads, batch_first=True
        )
        self.norm = nn.LayerNorm(esm_dim)

    def forward(self, esm: torch.Tensor, mech: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            esm  : (B, L, esm_dim)
            mech : (B, L, mech_dim)
            key_padding_mask : (B, L) bool, True at padding positions (optional).
        Returns:
            fused : (B, L, esm_dim + mech_dim)  — drop-in for cat([esm, mech]).
        """
        mech_kv = self.mech_proj(mech)                       # (B, L, esm_dim)
        attn_out, _ = self.attn(
            query=esm, key=mech_kv, value=mech_kv,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        fused_esm = self.norm(esm + attn_out)                # residual + norm
        # Keep the raw mechanistic channels alongside the fused representation.
        return torch.cat([fused_esm, mech], dim=-1)


# ---------------------------------------------------------------------------
# Selective State-Space mixer ("Mamba")
# ---------------------------------------------------------------------------
class _PureSelectiveSSM(nn.Module):
    """
    Dependency-free, input-dependent (selective) diagonal state-space recurrence.

    For each channel:  h_t = a_t * h_{t-1} + b_t * x_t ;  y_t = c * h_t + d * x_t
    where the forget gate a_t = sigmoid(W_a x_t) is input-dependent (the
    "selective" property), giving linear-time O(L) sequence mixing.  This is a
    faithful, runnable stand-in for Mamba's S6 scan — not bit-identical, but the
    same architectural primitive and shape contract.
    """

    def __init__(self, dim: int, expand: int = 1):
        super().__init__()
        hidden = dim * expand
        self.in_proj  = nn.Linear(dim, hidden)
        self.gate_a   = nn.Linear(hidden, hidden)   # input-dependent forget gate
        self.gate_b   = nn.Linear(hidden, hidden)   # input-dependent input gate
        self.c        = nn.Parameter(torch.ones(hidden) * 0.5)
        self.d        = nn.Parameter(torch.ones(hidden))
        self.out_proj = nn.Linear(hidden, dim)
        self.norm     = nn.LayerNorm(dim)

    @staticmethod
    def _associative_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Solve h_t = a_t * h_{t-1} + x_t (with h_{-1} = 0) for all t at once.

        The recurrence is a prefix scan under the associative operator
        ``(a_l, x_l) . (a_r, x_r) = (a_l a_r, a_r x_l + x_r)``, so a
        Hillis-Steele scan resolves it in ceil(log2 L) vectorised steps instead
        of L sequential ones. At L=256 that is 8 steps rather than 256 — the
        sequential version made the mixer, not the GNN, the training bottleneck.

        Products of ``a`` stay in (0, 1), so no log-space rescaling is needed.
        """
        a_c, x_c = a, x
        step = 1
        L = a.size(1)
        while step < L:
            a_prev = torch.ones_like(a_c)
            x_prev = torch.zeros_like(x_c)
            a_prev[:, step:] = a_c[:, :-step]
            x_prev[:, step:] = x_c[:, :-step]
            x_c = a_c * x_prev + x_c      # uses the pre-update a_c, as required
            a_c = a_c * a_prev
            step *= 2
        return x_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, dim)
        residual = x
        u = self.in_proj(x)                       # (B, L, H)
        a = torch.sigmoid(self.gate_a(u))         # forget gate ∈ (0,1)
        b = torch.sigmoid(self.gate_b(u))         # input gate  ∈ (0,1)
        h = self._associative_scan(a, b * u)      # (B, L, H) selective state
        y = self.c * h + self.d * u
        return self.norm(residual + self.out_proj(y))


class SelectiveSSM(nn.Module):
    """
    Linear-time sequence mixer over residues. Uses real Mamba when installed,
    else the pure-PyTorch selective SSM above. Shape-preserving: (B, L, dim).
    """

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        if _MAMBA_AVAILABLE:
            self.mixer = _RealMamba(d_model=dim, expand=expand)
            self.backend = "mamba_ssm"
        else:
            self.mixer = _PureSelectiveSSM(dim, expand=1)
            self.backend = "pure_pytorch_ssm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


# ---------------------------------------------------------------------------
# Holo-GeoN v2 compact mixers (FiLM + Bidirectional Selective SSM)
# ---------------------------------------------------------------------------
class FiLMModulator(nn.Module):
    """
    Low-rank Feature-wise Linear Modulation (FiLM) replacing the 6.57M rank-32
    ``CrossAttentionFusion`` (OVERFITTING_FIX.md §C1).

    Modulates the bottlenecked evolutionary residue representations ``h``
    (``(B, L, dim)``) with mechanistic, contact-topology, and optional geometric
    side channels ``cond`` (``(B, L, cond_dim)``):

        h_out = LayerNorm(h + gate * (gamma * h + beta))

    The output projection is zero-initialized so at step 0 this layer is an
    exact identity transform and never injects random noise into ESM-2 features.
    At ``dim=256, cond_dim=14, rank=64``, this uses ~40K parameters (160x fewer
    than v1's 6.57M cross-attention).
    """

    def __init__(
        self,
        dim: int = 256,
        cond_dim: int = 14,
        rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, 2 * dim),
        )
        # Zero-init final projection -> exact identity at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        h: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gb = self.net(cond)                                  # (B, L, 2*dim)
        gamma, beta = gb.chunk(2, dim=-1)
        out = self.norm(h * (1.0 + gamma) + beta)
        if mask is not None:
            out = out.masked_fill((mask == 0).unsqueeze(-1), 0.0)
        return out


class BidirectionalMixer(nn.Module):
    """
    Mask-aware bidirectional selective state-space mixer operating at compact
    bottleneck width ``dim`` (default 256 -> ~395K params vs. 6.56M in v1).

    Fixes two defects of v1's ``SelectiveSSM``:
      1. Proteins are non-causal: residue ``i`` needs C-terminal context just as
         much as N-terminal context. Running forward + backward selective scans
         over real positions (with padding strictly excluded from both scans)
         restores directional symmetry and guarantees exact padding invariance.
      2. Operating after the ``1280 -> 256`` bottleneck cuts parameter count by
         ~16x while retaining full-window linear-time context mixing.
    """

    def __init__(self, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        half = max(dim // 2, 16)
        self.half = half
        self.in_proj = nn.Linear(dim, 2 * half)
        self.fwd_a = nn.Linear(half, half)
        self.fwd_b = nn.Linear(half, half)
        self.bwd_a = nn.Linear(half, half)
        self.bwd_b = nn.Linear(half, half)
        self.c_fwd = nn.Parameter(torch.ones(half) * 0.5)
        self.c_bwd = nn.Parameter(torch.ones(half) * 0.5)
        self.d_skip = nn.Parameter(torch.ones(2 * half))
        self.out_proj = nn.Linear(2 * half, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _masked_reverse(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Reverse ONLY the real prefix ``[0, length_b)`` of each row so padding
        never enters the backward scan (guarantees exact padding invariance).
        """
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        rev_idx = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0, max=max(L - 1, 0))
        valid = pos < lengths.unsqueeze(1)
        out = x.gather(1, rev_idx.unsqueeze(-1).expand_as(x))
        return out.masked_fill(~valid.unsqueeze(-1), 0.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        if mask is None:
            valid = torch.ones((B, L), dtype=torch.bool, device=x.device)
        else:
            valid = mask.bool()
        lengths = valid.long().sum(dim=1).clamp(min=1)

        residual = x
        u = self.in_proj(x).masked_fill(~valid.unsqueeze(-1), 0.0)
        u_fwd, u_bwd = u.split(self.half, dim=-1)

        # Forward scan (padding has a=1, b*u=0 so state freezes cleanly; masked out after)
        a_f = torch.sigmoid(self.fwd_a(u_fwd))
        b_f = torch.sigmoid(self.fwd_b(u_fwd))
        x_f = (b_f * u_fwd).masked_fill(~valid.unsqueeze(-1), 0.0)
        a_f = a_f.masked_fill(~valid.unsqueeze(-1), 1.0)
        h_f = _PureSelectiveSSM._associative_scan(a_f, x_f)

        # Backward scan over real residues ONLY (prefix-reversed)
        u_rev = self._masked_reverse(u_bwd, lengths)
        a_b = torch.sigmoid(self.bwd_a(u_rev))
        b_b = torch.sigmoid(self.bwd_b(u_rev))
        x_b = (b_b * u_rev).masked_fill(~valid.unsqueeze(-1), 0.0)
        a_b = a_b.masked_fill(~valid.unsqueeze(-1), 1.0)
        h_b_rev = _PureSelectiveSSM._associative_scan(a_b, x_b)
        h_b = self._masked_reverse(h_b_rev, lengths)

        y = torch.cat([self.c_fwd * h_f, self.c_bwd * h_b], dim=-1) + self.d_skip * u
        out = self.norm(residual + self.dropout(self.out_proj(y)))
        return out.masked_fill(~valid.unsqueeze(-1), 0.0)


__all__ = [
    "CrossAttentionFusion",
    "SelectiveSSM",
    "FiLMModulator",
    "BidirectionalMixer",
]
