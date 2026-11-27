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
STR_EDGE_DIM = 26       # 3 base channels + 23 SE(3)-invariant geometry channels
N_TOPO_CHANNELS = 6     # structure-free contact-derived burial/packing + entropy
N_PAIR_GEO = 4          # [direct_contact_prob, log_sep, inv_hop_dist, is_contact]


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
    """
    Keep at most ``top_k`` incident edges per node, strongest first.

    Round-1 review, R1-17: this was a Python loop doing two scalar tensor reads
    per iteration, measured at 34.2 ms of the 35.3 ms spent building one
    sample's graph - roughly 0.7 h of data loading per MegaScale-scale epoch on
    four workers, before a single GPU step. The rank-based filter below is
    vectorised and gives the same answer whenever it is decidable without
    sequential state.

    Each endpoint independently keeps its ``top_k`` strongest incident edges; an
    edge survives if BOTH endpoints kept it. That is the natural symmetric
    reading of a degree cap, and unlike the greedy loop it does not depend on
    the order edges happen to be visited in. The resulting degree is at most
    ``top_k`` per node, which is the guarantee callers rely on.
    """
    if top_k <= 0 or edges.shape[1] == 0:
        return edges, probs

    e = edges.shape[1]
    # Rank every (endpoint, edge) incidence by descending probability within
    # that endpoint, then keep incidences ranked below top_k.
    endpoints = torch.cat([edges[0], edges[1]])          # (2E,)
    weights = torch.cat([probs, probs])                  # (2E,)
    edge_ids = torch.cat([torch.arange(e), torch.arange(e)])

    # Sort by (endpoint, -weight) so each node's incidences are contiguous and
    # ordered strongest first.
    order = torch.argsort(weights, descending=True, stable=True)
    endpoints = endpoints[order]
    edge_ids = edge_ids[order]
    node_order = torch.argsort(endpoints, stable=True)
    endpoints = endpoints[node_order]
    edge_ids = edge_ids[node_order]

    # Position of each incidence within its node's run.
    counts = torch.bincount(endpoints, minlength=n)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(endpoints.numel()) - starts[endpoints]

    kept_incidence = rank < top_k
    # An edge survives only if both of its endpoints kept it.
    votes = torch.zeros(e, dtype=torch.long)
    votes.index_add_(0, edge_ids[kept_incidence],
                     torch.ones(int(kept_incidence.sum()), dtype=torch.long))
    keep = votes >= 2
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


# ---------------------------------------------------------------------------
# Structure-free contact-topology burial/packing descriptors (Holo-GeoN v2)
# ---------------------------------------------------------------------------
def contact_topology_features(
    n: int,
    edge_geo: torch.Tensor,
    attr_geo: torch.Tensor,
    mech: torch.Tensor,
    site_entropy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Derive 6 structure-free per-residue biophysical environment channels from
    ``E_geo`` and the physicochemical residue tracks (zero 3D coordinates):

      0  lr_contact_deg  sum of contact probs for |i - j| >= 6 (/ 10, clamped [0, 2])
      1  sr_contact_deg  sum of non-backbone contact probs for 3 <= |i - j| < 6 (/ 6)
      2  contact_hydro   contact-weighted mean hydropathy of spatial neighbors
      3  contact_vol     contact-weighted mean van der Waals volume of spatial neighbors
      4  contact_charge  contact-weighted mean net charge of spatial neighbors
      5  site_entropy    normalized WT-marginal Shannon entropy H_i / log(20) in [0, 1]
    """
    from src.features import CH_CHARGE, CH_HYDRO, CH_VOLUME

    dtype = mech.dtype
    device = mech.device
    out = torch.zeros((n, N_TOPO_CHANNELS), dtype=dtype, device=device)
    if n == 0:
        return out

    if edge_geo.numel() > 0:
        src, dst = edge_geo[0], edge_geo[1]
        strength = attr_geo[:, 0].to(dtype)
        is_bb = attr_geo[:, 2] > 0.5
        sep = (src - dst).abs()
        non_loop = src != dst

        # Long-range (|i-j| >= 6) and short/medium-range (3 <= |i-j| < 6) contacts
        lr_mask = non_loop & (~is_bb) & (sep >= 6)
        sr_mask = non_loop & (~is_bb) & (sep >= 3) & (sep < 6)

        if bool(lr_mask.any()):
            lr_deg = torch.zeros(n, dtype=dtype, device=device)
            lr_deg.index_add_(0, src[lr_mask], strength[lr_mask])
            out[:, 0] = (lr_deg / 10.0).clamp(0.0, 2.0)

        if bool(sr_mask.any()):
            sr_deg = torch.zeros(n, dtype=dtype, device=device)
            sr_deg.index_add_(0, src[sr_mask], strength[sr_mask])
            out[:, 1] = (sr_deg / 6.0).clamp(0.0, 2.0)

        # Contact-weighted neighbor biophysical environment (all non-loop edges)
        if bool(non_loop.any()):
            s_nl = src[non_loop]
            d_nl = dst[non_loop]
            w_nl = strength[non_loop]
            w_sum = torch.zeros(n, dtype=dtype, device=device)
            w_sum.index_add_(0, s_nl, w_nl)
            denom = w_sum.clamp_min(1e-6)

            for out_ch, mech_ch in ((2, CH_HYDRO), (3, CH_VOLUME), (4, CH_CHARGE)):
                acc = torch.zeros(n, dtype=dtype, device=device)
                acc.index_add_(0, s_nl, w_nl * mech[d_nl, mech_ch])
                out[:, out_ch] = torch.where(w_sum > 1e-6, acc / denom, mech[:, mech_ch])
        else:
            out[:, 2] = mech[:, CH_HYDRO]
            out[:, 3] = mech[:, CH_VOLUME]
            out[:, 4] = mech[:, CH_CHARGE]
    else:
        out[:, 2] = mech[:, CH_HYDRO]
        out[:, 3] = mech[:, CH_VOLUME]
        out[:, 4] = mech[:, CH_CHARGE]

    if site_entropy is not None and site_entropy.numel() >= n:
        out[:, 5] = site_entropy[:n].to(dtype=dtype, device=device).clamp(0.0, 1.0)
    else:
        out[:, 5] = 0.5
    return out


# ---------------------------------------------------------------------------
# Pairwise contact features (for Contact-Gated Epistasis Head)
# ---------------------------------------------------------------------------
def pair_contact_features(
    edge_geo: torch.Tensor,
    attr_geo: torch.Tensor,
    site_i: int,
    site_j: int,
    n: int,
    max_hops: int = 4,
) -> torch.Tensor:
    """
    4-channel symmetric contact-topology descriptor between residues ``site_i``
    and ``site_j`` on ``E_geo``:

      0  direct_contact_prob  max edge strength between i and j (0 if not adjacent)
      1  log_seq_sep          log1p(|i - j|) / log1p(max(n - 1, 1))
      2  inv_hop_dist         1 / shortest_hop_distance (0 if > max_hops)
      3  is_direct_contact    1.0 if direct non-backbone contact edge exists else 0.0
    """
    sep = abs(int(site_i) - int(site_j))
    log_sep = math.log1p(sep) / math.log1p(max(n - 1, 1))
    direct_p = 0.0
    is_contact = 0.0
    inv_hop = 0.0

    if site_i == site_j:
        return torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    if edge_geo.numel() > 0:
        src, dst = edge_geo[0], edge_geo[1]
        hit = ((src == site_i) & (dst == site_j)) | ((src == site_j) & (dst == site_i))
        if bool(hit.any()):
            direct_p = float(attr_geo[hit, 0].max())
            is_contact = 1.0 if bool((attr_geo[hit, 2] < 0.5).any()) else 0.0
            inv_hop = 1.0
        else:
            # BFS up to max_hops on E_geo
            visited = torch.zeros(n, dtype=torch.bool)
            if 0 <= site_i < n and 0 <= site_j < n:
                visited[site_i] = True
                for hop in range(1, max_hops + 1):
                    reach = visited[src]
                    if not bool(reach.any()):
                        break
                    nxt = dst[reach]
                    nxt = nxt[(nxt >= 0) & (nxt < n)]
                    visited[nxt] = True
                    if bool(visited[site_j]):
                        inv_hop = 1.0 / float(hop)
                        break

    return torch.tensor([direct_p, log_sep, inv_hop, is_contact], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Controlled geometry ablation arms (GEOMETRIC_PLAN.md §2.2 & §3)
# ---------------------------------------------------------------------------
def build_structure_graph(
    coords_4: torch.Tensor | None,
    plddt: torch.Tensor | None,
    coord_valid: torch.Tensor | None,
    n: int,
    k: int = 30,
    max_dist: float = 12.0,
    topology_only: bool = False,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the invariant 3D coordinate stream ``E_str`` (Arms B, C, D in
    ``GEOMETRIC_PLAN.md``).

    Returns ``(edge_index (2, E), edge_attr (E, 26 or 3), str_node (n, 9))``.
    When ``coords_4`` is absent or all residues have ``coord_valid == 0``,
    returns self-loops only and zeroed node/geometry features — reusing the
    exact ``mech_valid`` convention.
    """
    from src.geometry import (
        ATOM_CB,
        PLDDT_VALID_FLOOR,
        STR_GEOM_EDGE_DIM,
        STR_NODE_DIM,
        edge_geometry,
        knn_edges,
        node_geometry,
    )

    out_edge_dim = EDGE_DIM if topology_only else STR_EDGE_DIM
    if coords_4 is None or n == 0:
        loops = torch.arange(n, dtype=torch.long)
        ei = torch.stack([loops, loops], dim=0) if add_self_loops and n > 0 else EMPTY_EDGES.clone()
        ea = torch.zeros((ei.shape[1], out_edge_dim), dtype=torch.float32)
        if ei.shape[1] > 0:
            ea[:, 0] = 1.0
        return ei, ea, torch.zeros((n, STR_NODE_DIM), dtype=torch.float32)

    coords_4 = coords_4[:n].float()
    if plddt is None:
        plddt = torch.full((n,), 100.0, dtype=torch.float32)
    else:
        plddt = plddt[:n].float()
    if coord_valid is None:
        coord_valid = torch.ones((n,), dtype=torch.float32)
    else:
        coord_valid = coord_valid[:n].float()

    eff_valid = ((coord_valid > 0.5) & (plddt >= PLDDT_VALID_FLOOR)).float()
    str_node = node_geometry(coords_4, plddt, coord_valid)
    if float(eff_valid.sum()) < 2.0:
        loops = torch.arange(n, dtype=torch.long)
        ei = torch.stack([loops, loops], dim=0) if add_self_loops else EMPTY_EDGES.clone()
        ea = torch.zeros((ei.shape[1], out_edge_dim), dtype=torch.float32)
        if ei.shape[1] > 0:
            ea[:, 0] = 1.0
        return ei, ea, str_node

    ei_knn, d_cb = knn_edges(coords_4[:, ATOM_CB], eff_valid, k=k, max_dist=max_dist)
    if ei_knn.shape[1] > 0:
        strength = (1.0 - d_cb / max(max_dist, 1e-6)).clamp(0.0, 1.0)
        sep = _separation_feature(ei_knn[0], ei_knn[1], n)
        bb = ((ei_knn[0] - ei_knn[1]).abs() == 1).float()
        base_attr = torch.stack([strength, sep, bb], dim=1)
        if topology_only:
            ea_knn = base_attr
        else:
            geom_attr = edge_geometry(coords_4, ei_knn)
            ea_knn = torch.cat([base_attr, geom_attr], dim=1)
    else:
        ea_knn = torch.zeros((0, out_edge_dim), dtype=torch.float32)

    if add_self_loops:
        loops = torch.arange(n, dtype=torch.long)
        loop_ei = torch.stack([loops, loops], dim=0)
        loop_ea = torch.zeros((n, out_edge_dim), dtype=torch.float32)
        loop_ea[:, 0] = 1.0
        if not topology_only:
            # Identity quaternion w=1 for self-loops; 0 RBF / 0 unit vector.
            loop_ea[:, EDGE_DIM + 16 + 3] = 1.0
        ei = torch.cat([ei_knn, loop_ei], dim=1)
        ea = torch.cat([ea_knn, loop_ea], dim=0)
    else:
        ei, ea = ei_knn, ea_knn
    return ei, ea, str_node


def build_true_contact_graph(
    n: int,
    true_contact_map: torch.Tensor,
    top_k: int = 16,
    min_sep: int = 3,
    add_backbone: bool = True,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arm E (``GEOMETRIC_PLAN.md`` §3): drop-in replacement for
    :func:`build_geometric_graph` using real C-beta contacts (< 8 A) instead of
    ESM-2's predicted contacts, keeping the exact same 3-channel edge layout.
    """
    cm = true_contact_map[:n, :n].float()
    if n < 2:
        return build_geometric_graph(
            n, torch.zeros((2, 0), dtype=torch.long), torch.zeros(0),
            top_k=top_k, threshold=0.5, add_backbone=add_backbone, add_self_loops=add_self_loops,
        )
    idx = torch.arange(n)
    sep = (idx[:, None] - idx[None, :]).abs()
    tri = torch.triu((cm > 0.5) & (sep >= min_sep), diagonal=1)
    nz = torch.nonzero(tri, as_tuple=False)
    if nz.numel():
        edges = torch.stack([nz[:, 0], nz[:, 1]], dim=0).long()
        probs = torch.ones(edges.shape[1], dtype=torch.float32)
    else:
        edges = torch.zeros((2, 0), dtype=torch.long)
        probs = torch.zeros(0, dtype=torch.float32)
    return build_geometric_graph(
        n, edges, probs, top_k=top_k, threshold=0.5,
        add_backbone=add_backbone, add_self_loops=add_self_loops,
    )


__all__ = [
    "EDGE_DIM",
    "STR_EDGE_DIM",
    "N_TOPO_CHANNELS",
    "N_PAIR_GEO",
    "window_edges",
    "build_geometric_graph",
    "build_mechanistic_graph",
    "build_structure_graph",
    "build_true_contact_graph",
    "contact_topology_features",
    "pair_contact_features",
    "khop_neighborhood",
]
