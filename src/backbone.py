"""
src/backbone.py
===============
The trainable backbone: two GATv2 message-passing streams over a windowed
protein graph, on top of cached frozen-encoder features.

Input is a dense padded batch — ``(B, L, ...)`` with a validity mask — because
every window has the same maximum length and the sequence mixer wants a real
sequence axis. The GATv2 layers operate on the flattened ``(B*L, C)`` node
tensor with edge indices offset by ``b * L``, which is standard mini-batching
for graphs of equal node capacity.

Pipeline
--------
1. **Sequence mixing.** A selective state-space mixer runs over the residue
   axis in linear time, giving every residue a cheap view of the whole window
   before any graph structure is imposed.
2. **Fusion.** Cross-attention between the language-model embeddings (queries)
   and the mechanistic channels (keys/values), so mechanistic context modulates
   the evolutionary representation rather than being concatenated beside it.
3. **Two streams.** GATv2 over E_geo (spatial contacts) and over E_mech
   (translational coupling), each consuming the 3-channel edge features from
   ``src/graphs.py``.
4. **Fuse + residual.** Stream outputs are combined and the raw language-model
   embedding is added back through a projection, then LayerNorm — the skip that
   keeps residue identity intact against over-smoothing.

``torch_geometric`` is a hard requirement. v1 carried a silent fallback that
bypassed the GAT layers with a plain linear projection when PyG was missing,
which meant a "trained model" could quietly contain no message passing at all.
A missing dependency now raises at construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.features import N_MECH_CHANNELS
from src.graphs import EDGE_DIM
from src.sequence_mixers import CrossAttentionFusion, SelectiveSSM

try:
    from torch_geometric.nn import GATv2Conv
    PYG_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    GATv2Conv = None
    PYG_AVAILABLE = False


_PYG_HINT = (
    "torch_geometric is required for the message-passing streams. "
    "Install it with `pip install torch_geometric` (after installing torch). "
    "There is deliberately no non-graph fallback: a model that silently skips "
    "its GNN layers is not the model this repository evaluates."
)


class _Stream(nn.Module):
    """One two-layer GATv2 message-passing stream with edge features."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_dim, hidden, heads=heads, concat=False,
            edge_dim=EDGE_DIM, add_self_loops=False,
        )
        self.conv2 = GATv2Conv(
            hidden, out_dim, heads=heads, concat=False,
            edge_dim=EDGE_DIM, add_self_loops=False,
        )
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.act(self.conv1(x, edge_index, edge_attr))
        return self.conv2(x, edge_index, edge_attr)


class DualStreamGNN(nn.Module):
    """
    Args:
        esm_dim         : encoder hidden size (1280 for ESM-2 650M).
        mech_dim        : mechanistic channels (7).
        hidden, out_dim : GATv2 sizes.
        heads           : GATv2 attention heads.
        use_ssm         : run the selective state-space mixer.
        use_mech_stream : enable the E_mech stream. The shipped model is
                          dual-stream; this exists so the ablation runner can
                          measure the mechanistic stream's contribution against
                          an otherwise identical model.
    """

    def __init__(
        self,
        esm_dim: int,
        mech_dim: int = N_MECH_CHANNELS,
        hidden: int = 256,
        out_dim: int = 256,
        heads: int = 4,
        use_ssm: bool = True,
        use_mech_stream: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError(_PYG_HINT)

        self.esm_dim = esm_dim
        self.mech_dim = mech_dim
        self.out_dim = out_dim
        self.use_mech_stream = use_mech_stream

        self.ssm = SelectiveSSM(dim=esm_dim) if use_ssm else None
        self.fusion = CrossAttentionFusion(esm_dim=esm_dim, mech_dim=mech_dim, heads=heads)

        node_dim = esm_dim + mech_dim
        self.geo_stream = _Stream(node_dim, hidden, out_dim, heads)
        if use_mech_stream:
            self.mech_stream = _Stream(node_dim, hidden, out_dim, heads)
            self.fuse = nn.Linear(2 * out_dim, out_dim)

        self.residual_proj = nn.Linear(esm_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        esm: torch.Tensor,
        mech: torch.Tensor,
        mask: torch.Tensor,
        edge_geo: torch.Tensor,
        attr_geo: torch.Tensor,
        edge_mech: torch.Tensor,
        attr_mech: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            esm       : (B, L, esm_dim) cached per-residue embeddings.
            mech      : (B, L, mech_dim) mechanistic channels.
            mask      : (B, L) 1 for a real residue, 0 for padding.
            edge_geo  : (2, E)  flattened-node indices into B*L.
            attr_geo  : (E, 3)  edge features.
            edge_mech : (2, E') flattened-node indices into B*L.
            attr_mech : (E', 3) edge features.
        Returns:
            (B, L, out_dim) node embeddings.
        """
        B, L, _ = esm.shape
        pad = mask == 0

        # 1. sequence mixing (padding zeroed so it cannot leak into the scan)
        x_esm = esm.masked_fill(pad.unsqueeze(-1), 0.0)
        if self.ssm is not None:
            x_esm = self.ssm(x_esm)
            x_esm = x_esm.masked_fill(pad.unsqueeze(-1), 0.0)

        # 2. cross-attention fusion of the two feature tracks
        nodes = self.fusion(x_esm, mech, key_padding_mask=pad)      # (B, L, esm+mech)

        # 3. message passing over each stream
        x_flat = nodes.reshape(B * L, -1)
        g = self.geo_stream(x_flat, edge_geo, attr_geo)
        if self.use_mech_stream:
            m = self.mech_stream(x_flat, edge_mech, attr_mech)
            h = self.fuse(torch.cat([g, m], dim=-1))
        else:
            h = g

        # 4. language-model residual + norm
        h = self.norm(h + self.residual_proj(x_esm.reshape(B * L, -1)))
        h = self.dropout(h)
        return h.view(B, L, self.out_dim).masked_fill(pad.unsqueeze(-1), 0.0)


__all__ = ["DualStreamGNN", "PYG_AVAILABLE"]
