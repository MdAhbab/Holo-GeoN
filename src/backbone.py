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
from src.geometry import STR_NODE_DIM
from src.graphs import EDGE_DIM, N_TOPO_CHANNELS, STR_EDGE_DIM
from src.sequence_mixers import (
    BidirectionalMixer,
    CrossAttentionFusion,
    FiLMModulator,
    SelectiveSSM,
)

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
        mech_dim        : mechanistic channels (N_MECH_CHANNELS = 8).
        hidden, out_dim : GATv2 sizes.
        heads           : GATv2 attention heads.
        use_ssm         : run the selective state-space mixer.
        use_mech_stream : enable the E_mech (translational coupling) stream.
        use_geo_stream  : enable the E_geo (spatial contact) stream.
                          The shipped model runs both; these exist so the
                          ablation runner can measure one stream's contribution
                          against an otherwise identical model. At least one
                          must be enabled.
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
        use_geo_stream: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError(_PYG_HINT)
        if not (use_geo_stream or use_mech_stream):
            raise ValueError("at least one message-passing stream must be enabled")

        self.esm_dim = esm_dim
        self.mech_dim = mech_dim
        self.out_dim = out_dim
        self.use_mech_stream = use_mech_stream
        self.use_geo_stream = use_geo_stream

        self.ssm = SelectiveSSM(dim=esm_dim) if use_ssm else None
        self.fusion = CrossAttentionFusion(esm_dim=esm_dim, mech_dim=mech_dim, heads=heads)

        node_dim = esm_dim + mech_dim
        if use_geo_stream:
            self.geo_stream = _Stream(node_dim, hidden, out_dim, heads)
        if use_mech_stream:
            self.mech_stream = _Stream(node_dim, hidden, out_dim, heads)
        if use_geo_stream and use_mech_stream:
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
        g = self.geo_stream(x_flat, edge_geo, attr_geo) if self.use_geo_stream else None
        m = self.mech_stream(x_flat, edge_mech, attr_mech) if self.use_mech_stream else None
        if g is not None and m is not None:
            h = self.fuse(torch.cat([g, m], dim=-1))
        else:
            h = g if g is not None else m

        # 4. language-model residual + norm
        h = self.norm(h + self.residual_proj(x_esm.reshape(B * L, -1)))
        h = self.dropout(h)
        return h.view(B, L, self.out_dim).masked_fill(pad.unsqueeze(-1), 0.0)


# ---------------------------------------------------------------------------
# Holo-GeoN v2 compact multi-stream backbone (~1.8M params vs 19.9M in v1)
# ---------------------------------------------------------------------------
def _drop_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, p: float, training: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop non-self-loop edges during training to prevent graph memorization."""
    if not training or p <= 0.0 or edge_index.numel() == 0:
        return edge_index, edge_attr
    is_loop = edge_index[0] == edge_index[1]
    keep = is_loop | (torch.rand(edge_index.shape[1], device=edge_index.device) >= p)
    return edge_index[:, keep], edge_attr[keep]


class _CompactStream(nn.Module):
    """
    Two-layer Pre-LN Residual GATv2 stream operating at compact width ``dim``
    with split-head concatenation (``heads * (dim // heads) == dim``), cutting
    GNN parameter count by ~6x relative to v1's ``1288 -> 256 x 4`` stream.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        edge_dim: int = EDGE_DIM,
        heads: int = 4,
        dropout: float = 0.1,
        drop_edge: float = 0.2,
    ):
        super().__init__()
        self.drop_edge = float(drop_edge)
        head_dim = max(dim // heads, 8)
        inner = head_dim * heads
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = GATv2Conv(
            dim, head_dim, heads=heads, concat=True,
            edge_dim=edge_dim, dropout=dropout, add_self_loops=False,
        )
        self.proj1 = nn.Linear(inner, dim) if inner != dim else nn.Identity()

        out_head_dim = max(out_dim // heads, 8)
        out_inner = out_head_dim * heads
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = GATv2Conv(
            dim, out_head_dim, heads=heads, concat=True,
            edge_dim=edge_dim, dropout=dropout, add_self_loops=False,
        )
        self.proj2 = nn.Linear(out_inner, out_dim) if out_inner != out_dim else nn.Identity()
        self.skip2 = nn.Linear(dim, out_dim, bias=False) if dim != out_dim else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        ei, ea = _drop_edges(edge_index, edge_attr, self.drop_edge, self.training)
        h = self.conv1(self.norm1(x), ei, ea)
        x = x + self.dropout(self.act(self.proj1(h)))
        h2 = self.conv2(self.norm2(x), ei, ea)
        return self.skip2(x) + self.dropout(self.proj2(h2))


class HoloBackboneV2(nn.Module):
    """
    Compressed, physically grounded multi-stream backbone for Holo-GeoN v2:

      1. **Multi-layer ESM-2 ScalarMix** (when ``layer_embs`` is cached and
         ``use_multi_layer=True``): learns softmax weights over 4 tapped layers.
      2. **Bottleneck-first projection** (``esm_dim -> hidden``): projects ONCE
         from 1280 to 256 with Pre-LayerNorm + GELU + Feature Dropout instead of
         duplicating giant 1280-d layers across fusion, SSM, and GNN streams.
      3. **FiLM modulation** with mechanistic (8) + contact-topology burial (6)
         + optional invariant 3D node geometry (9) channels (~40K vs 6.57M).
      4. **Mask-aware BidirectionalMixer** at ``hidden`` width (~395K vs 6.56M).
      5. **Compact Residual Pre-LN GATv2 streams** over ``E_geo`` (predicted or
         true contacts), ``E_mech`` (translational coupling), and optional
         ``E_str`` (invariant 3D backbone coordinates, Arms B/C/D).
    """

    def __init__(
        self,
        esm_dim: int,
        mech_dim: int = N_MECH_CHANNELS,
        topo_dim: int = N_TOPO_CHANNELS,
        str_node_dim: int = STR_NODE_DIM,
        hidden: int = 256,
        out_dim: int = 256,
        heads: int = 4,
        n_tap_layers: int = 4,
        use_multi_layer: bool = True,
        use_contact_burial: bool = True,
        use_ssm: bool = True,
        use_mech_stream: bool = True,
        use_geo_stream: bool = True,
        use_str_stream: bool = False,
        str_topology_only: bool = False,
        dropout: float = 0.15,
        feat_dropout: float = 0.25,
        drop_edge: float = 0.20,
    ):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError(_PYG_HINT)
        if not (use_geo_stream or use_mech_stream or use_str_stream):
            raise ValueError("at least one message-passing stream must be enabled")

        self.esm_dim = esm_dim
        self.mech_dim = mech_dim
        self.topo_dim = topo_dim
        self.str_node_dim = str_node_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.use_multi_layer = use_multi_layer
        self.use_contact_burial = use_contact_burial
        self.use_mech_stream = use_mech_stream
        self.use_geo_stream = use_geo_stream
        self.use_str_stream = use_str_stream
        self.str_edge_dim = EDGE_DIM if str_topology_only else STR_EDGE_DIM

        # 1. Learned scalar-mix over tapped ESM-2 layers (initialized to favor final layer)
        init_w = torch.zeros(n_tap_layers)
        init_w[-1] = 1.0
        self.layer_weights = nn.Parameter(init_w)
        self.layer_scale = nn.Parameter(torch.ones(()))

        # 2. Shared bottleneck: 1280 -> hidden ONCE
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Dropout(feat_dropout),
        )

        # 3. Low-rank FiLM modulation over side channels
        cond_dim = mech_dim + (topo_dim if use_contact_burial else 0) + (str_node_dim if use_str_stream else 0)
        self.film = FiLMModulator(dim=hidden, cond_dim=cond_dim, rank=min(64, hidden), dropout=dropout)

        # 4. Compact bidirectional sequence mixer
        self.bimixer = BidirectionalMixer(dim=hidden, dropout=dropout) if use_ssm else None

        # 5. Compact residual Pre-LN GATv2 streams
        n_streams = 0
        if use_geo_stream:
            self.geo_stream = _CompactStream(hidden, out_dim, EDGE_DIM, heads, dropout, drop_edge)
            n_streams += 1
        if use_mech_stream:
            self.mech_stream = _CompactStream(hidden, out_dim, EDGE_DIM, heads, dropout, drop_edge)
            n_streams += 1
        if use_str_stream:
            self.str_stream = _CompactStream(hidden, out_dim, self.str_edge_dim, heads, dropout, drop_edge)
            n_streams += 1

        self.fuse = nn.Linear(n_streams * out_dim, out_dim) if n_streams > 1 else nn.Identity()
        self.residual_proj = nn.Linear(hidden, out_dim, bias=False) if hidden != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def _mix_esm(self, esm: torch.Tensor, layer_embs: torch.Tensor | None) -> torch.Tensor:
        if (
            self.use_multi_layer
            and layer_embs is not None
            and layer_embs.ndim == 4
            and layer_embs.shape[1] == self.layer_weights.numel()
        ):
            # layer_embs: (B, K, L, D)
            w = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
            return self.layer_scale * (layer_embs * w).sum(dim=1)
        return esm

    def forward(
        self,
        esm: torch.Tensor,
        mech: torch.Tensor,
        mask: torch.Tensor,
        edge_geo: torch.Tensor,
        attr_geo: torch.Tensor,
        edge_mech: torch.Tensor,
        attr_mech: torch.Tensor,
        topo: torch.Tensor | None = None,
        layer_embs: torch.Tensor | None = None,
        edge_str: torch.Tensor | None = None,
        attr_str: torch.Tensor | None = None,
        str_node: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = esm.shape
        pad = mask == 0

        # 1-2. Multi-layer mix + shared bottleneck
        x_raw = self._mix_esm(esm, layer_embs).masked_fill(pad.unsqueeze(-1), 0.0)
        h_bot = self.bottleneck(x_raw).masked_fill(pad.unsqueeze(-1), 0.0)

        # 3. FiLM conditioning (mech + optional contact-burial + optional 3D node geometry)
        conds = [mech]
        if self.use_contact_burial:
            if topo is None:
                topo = torch.zeros((B, L, self.topo_dim), dtype=mech.dtype, device=mech.device)
            conds.append(topo)
        if self.use_str_stream:
            if str_node is None:
                str_node = torch.zeros((B, L, self.str_node_dim), dtype=mech.dtype, device=mech.device)
            conds.append(str_node)
        h = self.film(h_bot, torch.cat(conds, dim=-1), mask=mask)

        # 4. Mask-aware bidirectional sequence mixing
        if self.bimixer is not None:
            h = self.bimixer(h, mask=mask)

        # 5. Message passing across enabled streams
        x_flat = h.reshape(B * L, -1)
        streams: list[torch.Tensor] = []
        if self.use_geo_stream:
            streams.append(self.geo_stream(x_flat, edge_geo, attr_geo))
        if self.use_mech_stream:
            streams.append(self.mech_stream(x_flat, edge_mech, attr_mech))
        if self.use_str_stream:
            if edge_str is None or attr_str is None:
                loops = torch.arange(B * L, device=x_flat.device, dtype=torch.long)
                edge_str = torch.stack([loops, loops], dim=0)
                attr_str = torch.zeros((B * L, self.str_edge_dim), dtype=x_flat.dtype, device=x_flat.device)
                attr_str[:, 0] = 1.0
            streams.append(self.str_stream(x_flat, edge_str, attr_str))

        h_gnn = self.fuse(torch.cat(streams, dim=-1)) if len(streams) > 1 else streams[0]
        h_out = self.norm(h_gnn + self.residual_proj(h_bot.reshape(B * L, -1)))
        h_out = self.dropout(h_out)
        return h_out.view(B, L, self.out_dim).masked_fill(pad.unsqueeze(-1), 0.0)


MultiStreamGNN = HoloBackboneV2

__all__ = ["DualStreamGNN", "HoloBackboneV2", "MultiStreamGNN", "PYG_AVAILABLE"]
