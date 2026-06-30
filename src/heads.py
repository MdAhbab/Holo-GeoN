"""
src/heads.py
============
Output heads. Three, each with a REAL training path (see scripts/):

1. SiameseStabilityHead  — ΔΔG from (z_wt, z_mt); trained with AntisymmetricLoss.
2. PathogenicityHead      — single binary classifier from a graph embedding,
                            trained on real ClinVar labels (BCEWithLogits), with
                            full ROC/CV reporting in scripts/evaluate.py.
3. IDREnsembleHead        — predicts (mu, sigma) of radius of gyration R_g.
                            Trained (scripts/train_idr.py) against a DISCLOSED
                            polymer-physics prior (src/idr_prior.py), NOT against
                            "experimental ClinVar" labels (which do not exist).
                            Use --no-idr to drop it entirely for a given run.
"""

import torch
import torch.nn as nn


class SiameseStabilityHead(nn.Module):
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, z_wt, z_mt):
        return self.mlp(z_mt - z_wt)  # antisymmetric by construction of the diff


class PathogenicityHead(nn.Module):
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),  # logit
        )

    def forward(self, z):
        return self.mlp(z)  # raw logit; use BCEWithLogitsLoss / sigmoid at eval


class IDREnsembleHead(nn.Module):
    """Predicts a Gaussian over R_g (Angstrom): mean and positive sigma."""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.mu = nn.Linear(128, 1)
        self.log_sigma = nn.Linear(128, 1)

    def forward(self, z):
        h = self.shared(z)
        mu = self.mu(h)
        sigma = torch.nn.functional.softplus(self.log_sigma(h)) + 1e-3
        return mu, sigma
