"""
src/graphs.py
=============
Graph construction for the two message-passing streams.

**E_geo — spatial.** Built from the cached ESM-2 contact-head edges, plus
guaranteed backbone ``(i, i+1)`` edges and self-loops so no residue is ever
isolated and the chain is always connected. Cached contacts are in
whole-protein coordinates; :func:`window_edges` restricts them to the
mutation-centered window and remaps to local indices.

**E_mech — translational coupling.** Connects residues whose codon adaptation
and local mRNA folding are close: ``|dCAI| < eps and |dmRNA| < delta``, both
endpoints carrying a real coding sequence (``mech_valid == 1``). Degree-capped
so the stream stays sparse rather than near-complete. Without a CDS this set is
empty by construction — see ``src/features.py`` for why there is no proxy.

Both streams emit the same 3-channel edge feature layout so a single GATv2
configuration serves both:

    0  strength    contact probability (E_geo) / coupling strength (E_mech)
    1  separation  log-scaled |i - j|, in [0, 1]
    2  backbone    1.0 for an (i, i+1) edge, else 0.0

v1 passed a single scalar (the raw attention weight) and had no separation
channel at all, which left the network unable to distinguish a local contact
from a long-range one — precisely the distinction that makes a contact graph
worth building.
"""

from __future__ import annotations

import math

import torch

EMPTY_EDGES = torch.zeros((2, 0), dtype=torch.long)
EMPTY_ATTR = torch.zeros((0, 3), dtype=torch.float32)
EDGE_DIM = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _separation_feature(src: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Log-scaled sequence separation in [0, 1]."""
    sep = (src - dst).abs().float()
    return torch.log1p(sep) / math.log1p(max(n - 1, 1))


def _bidirectional(
    src: torch.Tensor, dst: torch.Tensor, attr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror an undirected edge list into both directions."""
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0).long()
    return edge_index, torch.cat([attr, attr], dim=0)


def window_edges(
    edges: torch.Tensor, probs: torch.Tensor, start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Restrict cached whole-protein edges to ``[start, end)`` and remap to local
    indices. Edges with an endpoint outside the window are dropped: that
    residue is not a node in this graph, so the edge has nowhere to land.
    """
    if edges.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
    e = edges.long()
    keep = (e[0] >= start) & (e[0] < end) & (e[1] >= start) & (e[1] < end)
    return e[:, keep] - start, probs.float()[keep]


# ---------------------------------------------------------------------------
# E_geo
# ---------------------------------------------------------------------------
def build_geometric_graph(
    n: int,
    contact_edges: torch.Tensor,
    contact_probs: torch.Tensor,
    top_k: int = 16,
    threshold: float = 0.30,
    add_backbone: bool = True,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble E_geo for one windowed sequence of ``n`` residues.

    ``contact_edges`` / ``contact_probs`` are already window-local (see
    :func:`window_edges`). Contacts below ``threshold`` are dropped, then each
    residue keeps at most ``top_k`` of the survivors by probability.
    """
    src_l: list[torch.Tensor] = []
    dst_l: list[torch.Tensor] = []
    attr_l: list[torch.Tensor] = []

    if contact_edges.numel():
        keep = contact_probs >= threshold
        e = contact_edges[:, keep]
        p = contact_probs[keep]
        if e.numel():
            e, p = _cap_degree(e, p, n, top_k)
            attr = torch.stack(
                [p, _separation_feature(e[0], e[1], n), torch.zeros_like(p)], dim=1
            )
            src_l.append(e[0])
            dst_l.append(e[1])
            attr_l.append(attr)

    if add_backbone and n > 1:
        i = torch.arange(n - 1)
        p = torch.ones(n - 1)
        attr = torch.stack(
            [p, _separation_feature(i, i + 1, n), torch.ones(n - 1)], dim=1
        )
        src_l.append(i)
        dst_l.append(i + 1)
        attr_l.append(attr)

    if not src_l:
        edge_index, edge_attr = EMPTY_EDGES.clone(), EMPTY_ATTR.clone()
    else:
        edge_index, edge_attr = _bidirectional(
            torch.cat(src_l), torch.cat(dst_l), torch.cat(attr_l)
        )

    if add_self_loops:
        loops = torch.arange(n)
        loop_attr = torch.zeros((n, EDGE_DIM))
        loop_attr[:, 0] = 1.0
        edge_index = torch.cat(
            [edge_index, torch.stack([loops, loops], dim=0).long()], dim=1
        )
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    return edge_index, edge_attr


def _cap_degree(
    edges: torch.Tensor, probs: torch.Tensor, n: int, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at most ``top_k`` incident edges per node, strongest first."""
    if top_k <= 0 or edges.shape[1] == 0:
        return edges, probs
    order = torch.argsort(probs, descending=True)
    degree = torch.zeros(n, dtype=torch.long)
    keep = torch.zeros(edges.shape[1], dtype=torch.bool)
    for idx in order.tolist():
        a, b = int(edges[0, idx]), int(edges[1, idx])
        if degree[a] < top_k and degree[b] < top_k:
            keep[idx] = True
            degree[a] += 1
            degree[b] += 1
    return edges[:, keep], probs[keep]


# ---------------------------------------------------------------------------
# E_mech
# ---------------------------------------------------------------------------
def build_mechanistic_graph(
    cai: torch.Tensor,
    mrna: torch.Tensor,
    mech_valid: torch.Tensor,
    eps: float = 0.10,
    delta: float = 0.10,
    max_degree: int = 8,
    min_separation: int = 3,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble E_mech from per-residue CAI and local mRNA folding energy.

    Only residues with ``mech_valid == 1`` participate, so a protein with no
    coding sequence yields an empty edge set (self-loops only, which keeps the
    stream a well-defined self-transform rather than a crash).
    """
    n = int(cai.shape[0])
    valid = torch.nonzero(mech_valid > 0.5, as_tuple=False).flatten()

    if valid.numel() >= 2:
        c = cai[valid]
        m = mrna[valid]
        dcai = (c[:, None] - c[None, :]).abs()
        dmrna = (m[:, None] - m[None, :]).abs()
        sep = (valid[:, None] - valid[None, :]).abs()

        candidate = (dcai < eps) & (dmrna < delta) & (sep >= min_separation)
        # Coupling strength: 1 at identical features, 0 at the threshold corner.
        strength = 1.0 - 0.5 * (dcai / max(eps, 1e-8) + dmrna / max(delta, 1e-8))
        strength = strength.clamp(0.0, 1.0)
        strength = torch.where(candidate, strength, torch.zeros_like(strength))
        # Upper triangle only; the pair is mirrored below.
        strength = torch.triu(strength, diagonal=1)

        nz = torch.nonzero(strength > 0, as_tuple=False)
        if nz.numel():
            s = strength[nz[:, 0], nz[:, 1]]
            src = valid[nz[:, 0]]
            dst = valid[nz[:, 1]]
            e, s = _cap_degree(torch.stack([src, dst], dim=0), s, n, max_degree)
            attr = torch.stack(
                [s, _separation_feature(e[0], e[1], n), torch.zeros_like(s)], dim=1
            )
            edge_index, edge_attr = _bidirectional(e[0], e[1], attr)
        else:
            edge_index, edge_attr = EMPTY_EDGES.clone(), EMPTY_ATTR.clone()
    else:
        edge_index, edge_attr = EMPTY_EDGES.clone(), EMPTY_ATTR.clone()

    if add_self_loops:
        loops = torch.arange(n)
        loop_attr = torch.zeros((n, EDGE_DIM))
        loop_attr[:, 0] = 1.0
        edge_index = torch.cat(
            [edge_index, torch.stack([loops, loops], dim=0).long()], dim=1
        )
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Neighbourhood extraction (for the site-anchored readout)
# ---------------------------------------------------------------------------
def khop_neighborhood(
    edge_index: torch.Tensor, site: int, n: int, hops: int = 2
) -> torch.Tensor:
    """
    Boolean mask over ``n`` nodes marking everything within ``hops`` edges of
    ``site`` (inclusive). Used to pool the structural environment of a mutated
    residue — under E_geo this is its spatial neighbourhood, not merely its
    sequence neighbours.
    """
    mask = torch.zeros(n, dtype=torch.bool)
    if not 0 <= site < n:
        return mask
    mask[site] = True
    if edge_index.numel() == 0:
        return mask
    src, dst = edge_index[0], edge_index[1]
    for _ in range(max(hops, 0)):
        reachable = mask[src]
        if not bool(reachable.any()):
            break
        newly = dst[reachable]
        newly = newly[(newly >= 0) & (newly < n)]
        mask[newly] = True
    return mask


__all__ = [
    "EDGE_DIM",
    "window_edges",
    "build_geometric_graph",
    "build_mechanistic_graph",
    "khop_neighborhood",
]
