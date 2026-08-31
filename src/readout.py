"""
src/readout.py
==============
Graph-to-vector readout.

The problem this fixes
----------------------
v1 pooled every residue into one graph vector and represented a mutation as
``z_mt - z_wt``. A single substituted residue is one of L nodes, so its
contribution to a mean or attention pool scales as 1/L — for a 256-residue
window, under half a percent of the signal, and it then has to survive a
LayerNorm and an MLP. The model was being asked to detect a point mutation
through a whole-protein average.

The readout here is anchored at the mutated site instead:

    z = MLP([ z_site || z_neighborhood || z_global ])

  * ``z_site``          the mutated residue's own node embedding.
  * ``z_neighborhood``  attention pool over its k-hop structural neighbourhood
                        under E_geo — the residues actually packed against it,
                        which is where a stability change is felt.
  * ``z_global``        attention pool over the whole window, for the context
                        that says whether the site is buried, terminal, or in a
                        flexible loop.

Whole-sequence tasks (IDR ensemble dimensions, retention time) have no site, so
they use :meth:`SiteAnchoredReadout.global_only`.

Batches are dense and padded to a fixed window length; every pool here is
mask-aware, so padding never contributes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedAttentionPool(nn.Module):
    """Learned attention pool over the real positions of a padded batch."""

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h    : (B, L, D) node embeddings.
            mask : (B, L) bool or 0/1; True/1 marks a real residue.
        Returns:
            (B, D) pooled embeddings. A row with no valid positions returns
            zeros rather than NaN.
        """
        mask = mask.bool()
        scores = self.scorer(h).squeeze(-1)                      # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)             # all-padding row
        return torch.bmm(weights.unsqueeze(1), h).squeeze(1)


class SiteAnchoredReadout(nn.Module):
    """
    Combine site, neighbourhood and global views into one context vector.

    Args:
        dim     : node embedding size D.
        out_dim : context vector size.
        dropout : applied inside the fusion MLPs.
    """

    def __init__(self, dim: int, out_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        out_dim = out_dim or dim
        self.neighborhood_pool = MaskedAttentionPool(dim)
        self.global_pool = MaskedAttentionPool(dim)
        self.fuse = nn.Sequential(
            nn.Linear(3 * dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )
        self.global_fuse = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        site: torch.Tensor,
        neighborhood_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h                 : (B, L, D) node embeddings.
            mask              : (B, L) real-residue mask.
            site              : (B,) index of the mutated residue within the window.
            neighborhood_mask : (B, L) k-hop neighbourhood of each site.
        Returns:
            (B, out_dim) context vectors.
        """
        b = torch.arange(h.size(0), device=h.device)
        z_site = h[b, site.clamp(min=0, max=h.size(1) - 1)]
        z_neigh = self.neighborhood_pool(h, neighborhood_mask)
        z_global = self.global_pool(h, mask)
        return self.fuse(torch.cat([z_site, z_neigh, z_global], dim=-1))

    def global_only(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Whole-sequence readout for tasks with no mutation site."""
        return self.global_fuse(self.global_pool(h, mask))


__all__ = ["SiteAnchoredReadout", "MaskedAttentionPool"]
