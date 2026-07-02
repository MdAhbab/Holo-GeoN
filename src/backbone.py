"""
src/backbone.py
===============
HoloGNN backbone — now genuinely dual-track.

Two decoupled pieces:

1. ``EsmEncoder`` — a FROZEN ESM-2 wrapper used by scripts/precompute_embeddings.py
   to produce per-residue embeddings AND a validated contact-probability map
   (ESM-2 contact head). Freezing + caching keeps training on a single consumer
   GPU light (the ESM forward is removed from the training loop). The checkpoint
   is selectable via ``--esm-model``.

2. ``DualTrackGNN`` — the trainable model. It consumes cached node embeddings +
   4 mechanistic channels and runs TWO GATv2 tracks:
        geo  track over E_geo  (spatial contacts)
        mech track over E_mech (translational coupling)
   then fuses them, adds a residual from the language-model embedding, and mean-
   pools to a graph vector. When E_mech is empty (protein-only input), the mech
   track degrades to a self-transform via self-loops.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ESM dims by checkpoint (HF names) — used to size projections without a forward.
ESM_DIMS = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}
DEFAULT_ESM = "facebook/esm2_t33_650M_UR50D"  # 1280-dim, fits 16 GB frozen
N_MECH_CHANNELS = 4

try:
    from torch_geometric.nn import GATv2Conv
    PYG_AVAILABLE = True
except Exception:
    GATv2Conv = None
    PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Frozen ESM encoder (used only for precompute / caching)
# ---------------------------------------------------------------------------
class EsmEncoder(nn.Module):
    """Frozen ESM-2 wrapper: sequence -> (per-residue embedding, contact map)."""

    def __init__(self, esm_model_name: str = DEFAULT_ESM, device: str = "cpu"):
        super().__init__()
        from transformers import AutoTokenizer, EsmModel
        self.name = esm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        # eager attention is required: ESM's predict_contacts() / output_attentions
        # need real attention maps, which the default `sdpa` impl does not emit
        # (transformers >=5 selects sdpa by default).
        self.esm = EsmModel.from_pretrained(esm_model_name, add_pooling_layer=False,
                                            attn_implementation="eager")
        self.esm.eval()
        for p in self.esm.parameters():
            p.requires_grad = False
        self.device = device
        self.to(device)

    @torch.no_grad()
    def embed(self, seq: str, max_length: int = 256):
        """Return (L, esm_dim) residue embeddings (CLS/EOS stripped)."""
        toks = self.tokenizer(seq, return_tensors="pt", truncation=True,
                              max_length=max_length + 2)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.esm(**toks)
        h = out.last_hidden_state[0, 1:-1]  # strip CLS/EOS
        return h.cpu()

    @torch.no_grad()
    def contacts(self, seq: str, max_length: int = 256):
        """
        Return (L, L) contact probabilities from ESM-2's contact head.
        Falls back to symmetrised last-layer attention if the contact head is
        unavailable in the loaded checkpoint.
        """
        toks = self.tokenizer(seq, return_tensors="pt", truncation=True,
                              max_length=max_length + 2)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        try:
            cm = self.esm.predict_contacts(toks["input_ids"], toks["attention_mask"])
            return cm[0].float().cpu()  # (L, L), CLS/EOS already removed by HF
        except Exception:
            out = self.esm(**toks, output_attentions=True)
            attn = out.attentions[-1][0].mean(0)        # (L+2, L+2) head-mean
            attn = attn[1:-1, 1:-1]                      # strip CLS/EOS
            return (attn + attn.t()).float().cpu()


# ---------------------------------------------------------------------------
# Trainable dual-track GNN (operates on cached embeddings)
# ---------------------------------------------------------------------------
def _gat(in_dim: int, out_dim: int, heads: int = 4):
    if PYG_AVAILABLE:
        return GATv2Conv(in_dim, out_dim, heads=heads, concat=False, add_self_loops=True)
    # Fallback (no PyG): a residual-friendly linear "message" so the module still
    # runs for testing plumbing. Real runs require torch_geometric.
    return nn.Linear(in_dim, out_dim)


class _Track(nn.Module):
    """One GATv2 (or linear-fallback) track."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.g1 = _gat(in_dim, hidden)
        self.g2 = _gat(hidden, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        if PYG_AVAILABLE:
            x = self.relu(self.g1(x, edge_index))
            x = self.g2(x, edge_index)
        else:
            x = self.relu(self.g1(x))
            x = self.g2(x)
        return x


class DualTrackGNN(nn.Module):
    def __init__(self, esm_dim: int = ESM_DIMS[DEFAULT_ESM], hidden: int = 256,
                 out_dim: int = 256, use_mech_track: bool = True):
        super().__init__()
        self.use_mech_track = use_mech_track
        node_dim = esm_dim + N_MECH_CHANNELS
        self.geo_track = _Track(node_dim, hidden, out_dim)
        if use_mech_track:
            self.mech_track = _Track(node_dim, hidden, out_dim)
            self.fuse = nn.Linear(2 * out_dim, out_dim)
        self.residual_proj = nn.Linear(esm_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, esm_emb, mech_feats, edge_geo, edge_mech, batch=None):
        """
        esm_emb    : (N, esm_dim) cached per-residue embeddings (N = sum of L).
        mech_feats : (N, 4) mechanistic channels.
        edge_geo   : (2, Eg)  geometric edges (global node ids).
        edge_mech  : (2, Em)  mechanistic edges (global node ids).
        batch      : (N,) graph id per node for pooling (None => single graph).
        Returns    : (G, out_dim) graph embeddings.
        """
        x = torch.cat([esm_emb, mech_feats], dim=-1)
        g = self.geo_track(x, edge_geo)
        if self.use_mech_track:
            m = self.mech_track(x, edge_mech)
            fused = self.fuse(torch.cat([g, m], dim=-1))
        else:
            fused = g
        h = self.norm(fused + self.residual_proj(esm_emb))
        return _mean_pool(h, batch)


def _mean_pool(x, batch):
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    G = int(batch.max().item()) + 1
    out = torch.zeros(G, x.size(-1), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=G).clamp(min=1).unsqueeze(-1)
    return out / counts
