"""
src/sequence_mixers.py
======================
Future-work architectural modules (paper §8.2), implemented as **optional**,
self-contained, weight-compatible blocks that the backbone can switch on:

  • CrossAttentionFusion (§8.2-i)
        Replaces the simple geometric stacking ``cat([esm, mech])`` with explicit
        cross-modal attention between the evolutionary (ESM-2) and mechanistic
        tracks.  It still emits a 323-dim node tensor (320 fused ESM + 3 raw
        mechanistic channels) so the rest of the backbone is unchanged.

  • SelectiveSSM (§8.2-ii, "Mamba")
        A linear-time state-space sequence mixer over the residue dimension.
        If the real ``mamba_ssm`` package is installed it is used directly;
        otherwise a dependency-free selective state-space recurrence (S6-style,
        input-dependent gating) is used so the module runs anywhere, including
        on CPU.  Either way the input/output shape is (B, L, D).

Both default to OFF in the backbone, so existing behaviour and any existing
checkpoints are preserved unless explicitly enabled.
"""

from __future__ import annotations

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
# §8.2-i  Cross-Attention Fusion
# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """
    Cross-modal attention between ESM-2 node embeddings and mechanistic features.

    ESM embeddings act as the *query*; the (projected) mechanistic features act as
    *key*/*value*, so each residue's evolutionary representation is refined by the
    mechanistic context most relevant to it.  Output keeps the 323-dim contract.

    Args:
        esm_dim   : ESM-2 hidden size (320).
        mech_dim  : number of mechanistic channels (3).
        heads     : attention heads.
    """

    def __init__(self, esm_dim: int = 320, mech_dim: int = 3, heads: int = 4):
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
        # Preserve the raw mechanistic channels so GAT_IN_CHANNELS stays 323.
        return torch.cat([fused_esm, mech], dim=-1)


# ---------------------------------------------------------------------------
# §8.2-ii  Selective State-Space mixer ("Mamba")
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
