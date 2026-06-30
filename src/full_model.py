"""
src/full_model.py
=================
HoloGNN multi-task model — clean rewrite over the genuinely dual-track backbone.

A single shared DualTrackGNN backbone feeds three heads:
    - stability      : Siamese ΔΔG  (AntisymmetricLoss)         [MegaScale pairs]
    - pathogenicity  : binary logit (BCEWithLogits)             [ClinVar pairs]
    - idr            : (mu, sigma) of R_g (Gaussian NLL)        [physics prior]

Each "graph" is passed as a dict of node tensors produced by src/data.py:
    g = {
      'esm'      : (N, esm_dim) cached per-residue embeddings,
      'mech'     : (N, 4)       mechanistic channels,
      'edge_geo' : (2, Eg)      geometric edges (global ids),
      'edge_mech': (2, Em)      mechanistic edges (global ids),
      'batch'    : (N,)         graph id per node (for pooling),
    }
"""

import torch
import torch.nn as nn

from src.backbone import DualTrackGNN, ESM_DIMS, DEFAULT_ESM
from src.heads import SiameseStabilityHead, PathogenicityHead, IDREnsembleHead


class HoloGNN(nn.Module):
    def __init__(
        self,
        esm_dim: int = ESM_DIMS[DEFAULT_ESM],
        hidden: int = 256,
        out_dim: int = 256,
        use_mech_track: bool = True,
        use_idr: bool = True,
    ):
        super().__init__()
        self.backbone = DualTrackGNN(esm_dim, hidden, out_dim, use_mech_track)
        self.stability_head = SiameseStabilityHead(out_dim)
        self.pathogenicity_head = PathogenicityHead(out_dim)
        self.idr_head = IDREnsembleHead(out_dim) if use_idr else None

    # ----- shared encode -----
    def encode(self, g: dict) -> torch.Tensor:
        return self.backbone(g["esm"], g["mech"], g["edge_geo"], g["edge_mech"], g.get("batch"))

    # ----- task forwards -----
    def stability(self, g_wt: dict, g_mt: dict):
        z_wt = self.encode(g_wt)
        z_mt = self.encode(g_mt)
        dG_fwd = self.stability_head(z_wt, z_mt)   # WT -> MT
        dG_rev = self.stability_head(z_mt, z_wt)   # MT -> WT (≈ -dG_fwd)
        return dG_fwd, dG_rev

    def pathogenicity(self, g: dict) -> torch.Tensor:
        return self.pathogenicity_head(self.encode(g))  # logit

    def idr(self, g: dict):
        if self.idr_head is None:
            raise RuntimeError("IDR head disabled (use_idr=False).")
        return self.idr_head(self.encode(g))  # (mu, sigma)
