"""
src/utils/graph_builder.py
==========================
Dual-track edge construction.

Track 1 — GEOMETRIC edges (E_geo): spatial-contact proxy.
    Preferred source is ESM-2's *contact-prediction head*, whose probabilities
    are validated against PDB contacts in Lin et al. (2023). We threshold that
    (L, L) probability map. An attention-map builder is kept only as a fallback.

Track 2 — MECHANISTIC edges (E_mech): translational coupling.
    Connects residues i, j when their codon-adaptation and local mRNA-folding
    features are close:  |CAI_i - CAI_j| < eps  AND  |mRNA_i - mRNA_j| < delta,
    AND both residues carry valid DNA-derived features (mech_valid == 1).
    Degree is capped (``max_degree``) so the track stays sparse rather than
    near-complete. For protein-only inputs (no codon context) this set is empty
    by construction.

Every builder returns an ``edge_index`` LongTensor of shape (2, E) with both
directions present (i->j and j->i), or shape (2, 0) when no edges exist.
"""

from __future__ import annotations

import torch

EMPTY_EDGES = torch.zeros((2, 0), dtype=torch.long)


def _bidirectional(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    if rows.numel() == 0:
        return EMPTY_EDGES.clone()
    r = torch.cat([rows, cols])
    c = torch.cat([cols, rows])
    return torch.stack([r, c], dim=0).long()


# ---------------------------------------------------------------------------
# E_geo
# ---------------------------------------------------------------------------
def build_geometric_edges(contact_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Edges from an ESM-2 contact-probability map.

    Args:
        contact_map : (L, L) probabilities in [0, 1] from model.predict_contacts.
        threshold   : keep pairs with probability > threshold (default 0.5).
    """
    L = contact_map.shape[0]
    cm = contact_map.clone()
    cm.fill_diagonal_(0.0)
    iu = torch.triu_indices(L, L, offset=1)
    keep = cm[iu[0], iu[1]] > threshold
    return _bidirectional(iu[0][keep], iu[1][keep])


def build_geometric_edges_from_attention(attn_map: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Fallback E_geo from a (L, L) attention map (symmetrised, thresholded)."""
    adj = attn_map + attn_map.t()
    L = adj.shape[0]
    adj.fill_diagonal_(0.0)
    iu = torch.triu_indices(L, L, offset=1)
    keep = adj[iu[0], iu[1]] > threshold
    return _bidirectional(iu[0][keep], iu[1][keep])


# ---------------------------------------------------------------------------
# E_mech  (mechanistic / translational-coupling track)
# ---------------------------------------------------------------------------
def build_mechanistic_edges(
    cai: torch.Tensor,
    mrna: torch.Tensor,
    mech_valid: torch.Tensor,
    eps: float = 0.10,
    delta: float = 0.10,
    max_degree: int = 8,
    min_separation: int = 3,
) -> torch.Tensor:
    """
    Build E_mech from per-residue CAI and mRNA features.

    Args:
        cai, mrna  : (L,) per-residue features.
        mech_valid : (L,) 1.0 where DNA-derived features are real, else 0.0.
        eps, delta : thresholds on |dCAI| and |dmRNA| (E_mech edge thresholds).
        max_degree : cap on mechanistic neighbours per residue (keeps it sparse).
        min_separation : ignore pairs closer than this in sequence (those are
                         already covered by local/geometric context).

    Returns (2, E) bidirectional edge_index. Empty if <2 residues are valid.
    """
    valid_idx = torch.nonzero(mech_valid > 0.5, as_tuple=False).squeeze(-1)
    if valid_idx.numel() < 2:
        return EMPTY_EDGES.clone()

    c = cai[valid_idx]
    m = mrna[valid_idx]
    # pairwise abs diffs among valid residues
    dcai = (c[:, None] - c[None, :]).abs()
    dmrna = (m[:, None] - m[None, :]).abs()
    sep = (valid_idx[:, None] - valid_idx[None, :]).abs()

    cand = (dcai < eps) & (dmrna < delta) & (sep >= min_separation)
    # rank candidates by combined closeness; keep top-k per row
    score = dcai + dmrna
    score = score.masked_fill(~cand, float("inf"))

    rows, cols = [], []
    n = valid_idx.numel()
    k = min(max_degree, n)
    order = torch.argsort(score, dim=1)
    for i in range(n):
        added = 0
        for j in order[i].tolist():
            if added >= k:
                break
            if score[i, j] == float("inf"):
                break
            rows.append(int(valid_idx[i]))
            cols.append(int(valid_idx[j]))
            added += 1
    if not rows:
        return EMPTY_EDGES.clone()
    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    return _bidirectional(rows_t, cols_t)


# ---------------------------------------------------------------------------
# Fallback chain graph
# ---------------------------------------------------------------------------
def simple_linear_graph(seq_len: int) -> torch.Tensor:
    if seq_len < 2:
        return EMPTY_EDGES.clone()
    src = torch.arange(0, seq_len - 1)
    dst = torch.arange(1, seq_len)
    return _bidirectional(src, dst)
