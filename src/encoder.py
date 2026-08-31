"""
src/encoder.py
==============
Frozen ESM-2 encoder and its on-disk cache.

This is the one heavy GPU pass in the pipeline. It runs once per **unique
protein** and writes two things per sequence:

  * ``emb``  — per-residue embeddings, fp16, full length.
  * a **sparse** contact edge list from ESM-2's contact-prediction head.

Two decisions worth stating.

**Contact head, not attention map.** Edges come from ``predict_contacts``,
whose output is supervised against real structural contacts, rather than from
raw attention. The repository's own ``scripts/contact_sanity.py`` measures a
0.252 long-range fraction (|i-j| >= 24) among top-L contact-head pairs versus
0.000 for a sequence-proximity baseline — that non-local signal is the whole
reason for building a graph instead of a chain.

**Sparse, full-length storage.** A dense ``L x L`` map costs O(L^2) on disk and
is thrown away immediately after thresholding; storing the top-k neighbours per
residue costs O(L k) and loses nothing the graph builder would have used.
Caching at full length (rather than pre-windowed) means a window sliced later
carries whole-protein context, and one cached protein serves every mutation in
it — which is what keeps the cache keyed by ~10^3-10^4 proteins instead of
~10^5-10^6 mutants.

Proteins longer than the encoder's context are embedded in overlapping chunks;
see :func:`EsmEncoder.encode` for what that does and does not preserve.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

# Hidden size per ESM-2 checkpoint, so projections can be sized without a
# forward pass.
ESM_DIMS: dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}
DEFAULT_ESM = "facebook/esm2_t33_650M_UR50D"

# ESM-2 positional limit is 1024 tokens including CLS/EOS.
MAX_CONTEXT_RESIDUES = 1022
CHUNK_OVERLAP = 256


def seq_key(seq: str) -> str:
    """Stable cache key for a sequence."""
    return hashlib.sha1(seq.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Sparse contact extraction
# ---------------------------------------------------------------------------
def sparse_topk_contacts(
    contact_map: torch.Tensor,
    k: int,
    min_sep: int,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce a dense ``(L, L)`` contact-probability map to its top-k neighbours
    per residue, as a deduplicated upper-triangular edge list.

    Pairs closer than ``min_sep`` in sequence are excluded: they are backbone
    context, which the graph builder adds explicitly, and letting them win the
    top-k would crowd out the long-range pairs that justify the graph.

    Returns ``(edges (2, E) int32, probs (E,) float32)`` with ``edges[0] <
    edges[1]``, both shifted by ``offset`` (used when the map covers a chunk of
    a longer protein).
    """
    L = contact_map.shape[0]
    if L < 2:
        return torch.zeros((2, 0), dtype=torch.int32), torch.zeros(0)

    cm = contact_map.float().clone()
    idx = torch.arange(L, device=cm.device)
    sep = (idx[:, None] - idx[None, :]).abs()
    cm[sep < min_sep] = -1.0

    kk = int(min(k, L - 1))
    vals, cols = torch.topk(cm, kk, dim=1)          # (L, kk)
    rows = idx.unsqueeze(1).expand(-1, kk)
    keep = vals > 0

    r = rows[keep].reshape(-1)
    c = cols[keep].reshape(-1)
    p = vals[keep].reshape(-1)
    if r.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.int32), torch.zeros(0)

    # Canonicalise to i < j, then deduplicate (top-k is not symmetric).
    lo = torch.minimum(r, c)
    hi = torch.maximum(r, c)
    flat = lo.to(torch.int64) * L + hi.to(torch.int64)
    uniq, inverse = torch.unique(flat, return_inverse=True)
    # Keep the highest probability seen for each undirected pair.
    best = torch.zeros(uniq.numel(), dtype=p.dtype, device=p.device)
    best.scatter_reduce_(0, inverse, p, reduce="amax", include_self=False)
    u_lo = (uniq // L).to(torch.int32) + offset
    u_hi = (uniq % L).to(torch.int32) + offset
    return torch.stack([u_lo, u_hi], dim=0), best


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EsmEncoder(nn.Module):
    """
    Frozen ESM-2 wrapper: sequence -> (per-residue embeddings, sparse contacts).

    The encoder is never trained. It is instantiated only by
    ``scripts/precompute_embeddings.py`` and by the inference layer; the
    training loop reads the cache and never holds ESM in memory, which is what
    lets the trainable model use large batches on a 16 GB card.
    """

    def __init__(self, esm_model_name: str = DEFAULT_ESM, device: str = "cpu"):
        super().__init__()
        from transformers import AutoTokenizer, EsmModel

        self.name = esm_model_name
        self.dim = ESM_DIMS.get(esm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        # Eager attention is required: predict_contacts() consumes real attention
        # maps, which the default sdpa implementation in transformers >= 5 does
        # not emit.
        self.esm = EsmModel.from_pretrained(
            esm_model_name, add_pooling_layer=False, attn_implementation="eager"
        )
        self.esm.eval()
        for p in self.esm.parameters():
            p.requires_grad = False
        self.device = torch.device(device)
        self.to(self.device)
        if self.dim is None:
            self.dim = int(self.esm.config.hidden_size)

    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_spans(length: int) -> list[tuple[int, int]]:
        """Overlapping [start, end) spans covering a sequence of ``length``."""
        if length <= MAX_CONTEXT_RESIDUES:
            return [(0, length)]
        stride = MAX_CONTEXT_RESIDUES - CHUNK_OVERLAP
        spans = []
        start = 0
        while start < length:
            end = min(start + MAX_CONTEXT_RESIDUES, length)
            spans.append((start, end))
            if end == length:
                break
            start += stride
        return spans

    @torch.no_grad()
    def _forward_chunk(self, seq: str) -> tuple[torch.Tensor, torch.Tensor]:
        toks = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.esm(**toks)
        emb = out.last_hidden_state[0, 1:-1]                      # strip CLS/EOS
        cm = self.esm.predict_contacts(toks["input_ids"], toks["attention_mask"])[0]
        return emb.float().cpu(), cm.float().cpu()

    @torch.no_grad()
    def encode(self, seq: str, top_k: int = 24, min_sep: int = 3) -> dict:
        """
        Encode one protein.

        Returns ``{"seq", "emb" (L, d) fp16, "contact_edges" (2, E) int32,
        "contact_probs" (E,) fp16, "esm_model", "dim"}``.

        For proteins longer than the encoder's context the sequence is split
        into overlapping chunks. Each residue's embedding is taken from the
        chunk where it sits furthest from a boundary, so representations always
        come from the most context available. Contacts are extracted per chunk,
        which means **contacts spanning more than one chunk are not recovered**
        — a real limitation for proteins beyond ~1 kresidue, and the reason
        window selection happens around the mutation rather than at the N
        terminus.
        """
        L = len(seq)
        if L == 0:
            raise ValueError("cannot encode an empty sequence")

        spans = self._chunk_spans(L)
        if len(spans) == 1:
            emb, cm = self._forward_chunk(seq)
            edges, probs = sparse_topk_contacts(cm, top_k, min_sep)
        else:
            emb = torch.zeros((L, self.dim), dtype=torch.float32)
            # Distance from the nearest chunk boundary, per residue; a later
            # chunk only overwrites a residue if it sees it with more context.
            best_margin = torch.full((L,), -1.0)
            all_edges, all_probs = [], []
            for start, end in spans:
                c_emb, c_cm = self._forward_chunk(seq[start:end])
                n = end - start
                pos = torch.arange(n)
                margin = torch.minimum(pos, torch.tensor(n - 1) - pos).float()
                take = margin > best_margin[start:end]
                emb[start:end][take] = c_emb[take]
                best_margin[start:end][take] = margin[take]

                e, p = sparse_topk_contacts(c_cm, top_k, min_sep, offset=start)
                if e.numel():
                    all_edges.append(e)
                    all_probs.append(p)
            if all_edges:
                edges = torch.cat(all_edges, dim=1)
                probs = torch.cat(all_probs)
                # Overlapping chunks can propose the same pair twice.
                flat = edges[0].to(torch.int64) * L + edges[1].to(torch.int64)
                uniq, inverse = torch.unique(flat, return_inverse=True)
                best = torch.zeros(uniq.numel(), dtype=probs.dtype)
                best.scatter_reduce_(0, inverse, probs, reduce="amax", include_self=False)
                edges = torch.stack(
                    [(uniq // L).to(torch.int32), (uniq % L).to(torch.int32)], dim=0
                )
                probs = best
            else:
                edges = torch.zeros((2, 0), dtype=torch.int32)
                probs = torch.zeros(0)

        return {
            "seq": seq,
            "emb": emb.half(),
            "contact_edges": edges.to(torch.int32),
            "contact_probs": probs.half(),
            "esm_model": self.name,
            "dim": int(self.dim),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def masked_marginals(self, seq: str, positions: list[int]) -> dict[int, torch.Tensor]:
        """
        Log-softmax over the amino-acid vocabulary at each requested position,
        with that position masked. Used by the zero-shot baseline; not part of
        the model's own forward path.
        """
        from transformers import EsmForMaskedLM

        if not hasattr(self, "_mlm"):
            self._mlm = EsmForMaskedLM.from_pretrained(
                self.name, attn_implementation="eager"
            ).eval().to(self.device)

        out: dict[int, torch.Tensor] = {}
        mask_id = self.tokenizer.mask_token_id
        for pos in positions:
            toks = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            ids = toks["input_ids"].to(self.device)
            ids[0, pos + 1] = mask_id                     # +1 for CLS
            logits = self._mlm(input_ids=ids,
                               attention_mask=toks["attention_mask"].to(self.device)).logits
            out[pos] = torch.log_softmax(logits[0, pos + 1], dim=-1).cpu()
        return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class EncodingStore:
    """Read/write access to the precomputed encoder cache."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def path(self, seq: str) -> str:
        return os.path.join(self.cache_dir, seq_key(seq) + ".pt")

    def has(self, seq: str) -> bool:
        return os.path.exists(self.path(seq))

    def get(self, seq: str) -> dict:
        rec = torch.load(self.path(seq), map_location="cpu", weights_only=False)
        return rec

    def put(self, rec: dict) -> str:
        """Write one record atomically, so a crash never leaves a partial file."""
        os.makedirs(self.cache_dir, exist_ok=True)
        dst = self.path(rec["seq"])
        tmp = dst + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, dst)
        return dst

    @staticmethod
    def projected_bytes(total_residues: int, dim: int, top_k: int) -> int:
        """
        Disk footprint for a planned cache.

        Embeddings dominate: ``residues x dim x 2`` bytes at fp16. Edges add
        ``residues x top_k x (8 + 2)`` for the int32 pair plus the fp16
        probability.
        """
        return total_residues * (dim * 2 + top_k * 10)


__all__ = [
    "ESM_DIMS", "DEFAULT_ESM", "MAX_CONTEXT_RESIDUES",
    "EsmEncoder", "EncodingStore", "seq_key", "sparse_topk_contacts",
]
