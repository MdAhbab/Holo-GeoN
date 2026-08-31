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
#: Lower bound of each task loss, used to shift it non-negative for weighting.
#:
#: Round-1 review, R1-09: Kendall & Gal's derivation assumes a NON-NEGATIVE task
#: loss. A heteroscedastic Gaussian NLL is negative whenever the predictive
#: density exceeds 1 - with the log-variance clamp its minimum is -4.081 - and
#: for a negative value, ``value * exp(-log_var)`` is minimised by driving
#: log_var to minus infinity. The objective was unbounded below: the reviewer
#: measured the total reaching -1.9e19 with a stability weight of 4.7e18 after
#: 300 steps, at which point the global gradient clip hands the entire budget to
#: one task and every other head stops learning, while the printed loss looks
#: like spectacular progress.
#:
#: ddG NLL goes negative once the typical predicted sigma drops below ~0.24
#: kcal/mol, which is reachable on MegaScale where many mutations are
#: near-neutral and well predicted. So this is not a corner case.
LOSS_FLOORS: dict[str, float] = {
    # 0.5 * (log(2*pi) + LOGVAR_MIN), with LOGVAR_MIN = -10 as clamped in
    # gaussian_nll and in SubstitutionScoreHead.
    "stability": -4.0811,
    "idr": -4.0811,
    # Cross-entropy and MSE are non-negative already.
    "three_state": 0.0,
    "pathogenicity": 0.0,
    "mfi": 0.0,
    "proteomics": 0.0,
}

#: Bound on the learned log-variances. Spans a 400x weight range, wide enough
#: for genuinely different task scales and narrow enough that one task cannot
#: monopolise the shared gradient.
TASK_LOGVAR_MIN, TASK_LOGVAR_MAX = -3.0, 3.0


class UncertaintyWeightedLoss(nn.Module):
    """
    Learned per-task weighting (Kendall & Gal, 2018), made bounded below.

    Two guards on the original formulation:

    * Each task loss is shifted by its lower bound (:data:`LOSS_FLOORS`) so the
      weighted term sees a non-negative quantity, as the derivation requires.
      The shift is a constant, so it changes no gradient with respect to the
      model - only the sign of what the weight multiplies.
    * The learned log-variances are clamped to
      ``[TASK_LOGVAR_MIN, TASK_LOGVAR_MAX]``.

    Args:
        tasks : task names. One log-variance parameter is learned per task.

    Call with a mapping of task name to scalar loss; tasks absent from the
    mapping (no data in this batch) contribute nothing and their parameter
    receives no gradient.
    """

    def __init__(self, tasks: list[str], floors: dict[str, float] | None = None):
        super().__init__()
        self.tasks = list(tasks)
        self.floors = {t: (floors or LOSS_FLOORS).get(t, 0.0) for t in self.tasks}
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
            lv = self.log_var[name].clamp(TASK_LOGVAR_MIN, TASK_LOGVAR_MAX)
            # Shift non-negative for the weighted term; add the constant back so
            # the reported total is still comparable across runs.
            floor = self.floors[name]
            shifted = value - floor
            weighted = shifted * torch.exp(-lv) + 0.5 * lv + floor
            total = weighted if total is None else total + weighted
            parts[name] = float(value.detach())
            parts[f"{name}_weight"] = float(torch.exp(-lv).detach())
        if total is None:
            raise ValueError("no losses supplied")
        return total, parts

    def saturated(self, tol: float = 1e-3) -> list[str]:
        """Tasks whose log-variance has hit a clamp bound - worth reporting."""
        out = []
        for name, p in self.log_var.items():
            v = float(p.detach())
            if v <= TASK_LOGVAR_MIN + tol or v >= TASK_LOGVAR_MAX - tol:
                out.append(name)
        return out


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
