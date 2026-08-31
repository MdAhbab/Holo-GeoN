"""
src/losses.py
=============
Training objectives.

The ddG objective is a heteroscedastic Gaussian negative log-likelihood rather
than plain MSE, because the interface draws a confidence interval and that
interval has to mean something. Minimising the NLL makes the predicted variance
calibrated, so ``mu +/- 1.96 sigma`` is a real ~95% interval; a fixed-width band
computed from a formula (as v1 did, in *both* demo and full mode) is decoration.

The antisymmetry term is no longer part of the objective. With the
substitution-conditioned head in ``src/heads.py`` reversibility holds by
construction, so
:func:`antisymmetry_violation` exists purely as a monitor — it should read as
floating-point noise, and anything larger is a bug rather than a
under-optimised penalty.

Multi-task weighting follows Kendall & Gal (2018): each task carries a learned
log-variance and contributes ``L_i / (2 sigma_i^2) + log sigma_i``. Tasks with
different units and noise levels then balance themselves, instead of needing a
hand-tuned weight per head that silently decides which task the model actually
optimises.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Regression objectives
# ---------------------------------------------------------------------------
def gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Heteroscedastic Gaussian NLL, parameterised by log-variance.

        0.5 * [ log(2 pi) + logvar + (y - mu)^2 / exp(logvar) ]

    The constant is kept so the value is a genuine negative log-likelihood and
    can be compared across runs.
    """
    mu = mu.reshape(-1)
    logvar = logvar.reshape(-1)
    target = target.reshape(-1)
    logvar = logvar.clamp(min=-10.0, max=10.0)   # keeps exp() finite in fp16
    per_item = 0.5 * (math.log(2 * math.pi) + logvar + (target - mu) ** 2 * torch.exp(-logvar))
    if reduction == "none":
        return per_item
    return per_item.mean() if reduction == "mean" else per_item.sum()


def sigma_gaussian_nll(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Gaussian NLL for heads that emit a positive sigma directly (IDR)."""
    sigma = sigma.reshape(-1).clamp_min(1e-3)
    return gaussian_nll(mu, torch.log(sigma ** 2), target)


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------
@torch.no_grad()
def antisymmetry_violation(ddg_forward: torch.Tensor, ddg_reverse: torch.Tensor) -> float:
    """
    Mean ``|ddG(a->b) + ddG(b->a)|``.

    With the substitution-conditioned head this is exactly zero up to floating
    point. It is reported every evaluation as a structural check: a non-trivial
    value means the forward pass is not doing what the architecture claims.
    """
    return float((ddg_forward.reshape(-1) + ddg_reverse.reshape(-1)).abs().mean())


# ---------------------------------------------------------------------------
# Multi-task weighting
# ---------------------------------------------------------------------------
class UncertaintyWeightedLoss(nn.Module):
    """
    Learned per-task weighting (Kendall & Gal, 2018).

    Args:
        tasks : task names. One log-variance parameter is learned per task.

    Call with a mapping of task name to scalar loss; tasks absent from the
    mapping (no data in this batch) contribute nothing and their parameter
    receives no gradient.
    """

    def __init__(self, tasks: list[str]):
        super().__init__()
        self.tasks = list(tasks)
        self.log_var = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(())) for t in self.tasks}
        )

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = None
        parts: dict[str, float] = {}
        for name, value in losses.items():
            if value is None:
                continue
            if name not in self.log_var:
                raise KeyError(
                    f"loss {name!r} was not declared at construction; known: {self.tasks}"
                )
            lv = self.log_var[name]
            weighted = value * torch.exp(-lv) + 0.5 * lv
            total = weighted if total is None else total + weighted
            parts[name] = float(value.detach())
            parts[f"{name}_weight"] = float(torch.exp(-lv).detach())
        if total is None:
            raise ValueError("no losses supplied")
        return total, parts


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def three_state_loss(
    logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy for the three-state head, with optional class balancing."""
    return F.cross_entropy(logits, labels.reshape(-1), weight=class_weights)


def pathogenicity_loss(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """BCE-with-logits for the pathogenicity head."""
    return F.binary_cross_entropy_with_logits(
        logits.reshape(-1), labels.reshape(-1).float(), pos_weight=pos_weight
    )


__all__ = [
    "gaussian_nll",
    "sigma_gaussian_nll",
    "antisymmetry_violation",
    "UncertaintyWeightedLoss",
    "three_state_loss",
    "pathogenicity_loss",
]
