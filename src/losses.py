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
log-variance and contributes ``L_i / (2 sigma_i^2) + log sigma_i``. This removes
the hand-tuned per-head weight that otherwise silently decides which task the
model optimises.

What it does NOT do - and round-2 review, R2-07, was right that the previous
wording implied otherwise - is equalise the gradient each task receives. At the
stationary point in ``log_var`` the weight is ``0.5 / (L_i - floor_i)``, so every
task's *contribution to the total* is exactly 0.5 regardless of its difficulty or
its dataset size, while the *gradient* it sends the shared trunk scales as
``1 / (L_i - floor_i)``: a task with a high loss is weighted DOWN. That is the
actual behaviour of the objective, it is worth knowing before reading a
multi-task result, and :meth:`UncertaintyWeightedLoss.equilibrium_weights` makes
it inspectable rather than a thing to be derived from the source.
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


def beta_gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Beta-NLL (Seitzer et al., ICLR 2022): weights each item's heteroscedastic
    Gaussian NLL by ``stop_gradient(sigma^(2 * beta))`` so predicted variance
    collapse cannot amplify mean-channel memorisation gradients by ``1/sigma^2``
    (OVERFITTING_FIX.md §2c measured a 6.5x amplification under standard NLL).

    * ``beta == 0.0`` reproduces :func:`gaussian_nll` identically.
    * ``beta == 0.5`` (recommended) scales mean gradients as ``(mu - y) / sigma``
      while preserving exact NLL training of the variance head.
    * ``beta == 1.0`` recovers pure MSE gradients for the mean.
    """
    if beta <= 0.0:
        return gaussian_nll(mu, logvar, target, reduction=reduction)
    mu = mu.reshape(-1)
    logvar = logvar.reshape(-1).clamp(min=-10.0, max=10.0)
    target = target.reshape(-1)
    nll = 0.5 * (math.log(2 * math.pi) + logvar + (target - mu) ** 2 * torch.exp(-logvar))
    weight = torch.exp(beta * logvar.detach())
    per_item = weight * nll
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
# Why a floor exists at all.
#
# Round-1 review, R1-09: Kendall & Gal's derivation assumes a NON-NEGATIVE task
# loss. A heteroscedastic Gaussian NLL is negative whenever the predictive
# density exceeds 1 - with the log-variance clamp its minimum is -4.081 - and
# for a negative value, ``value * exp(-log_var)`` is minimised by driving
# log_var to minus infinity. The objective was unbounded below: the reviewer
# measured the total reaching -1.9e19 with a stability weight of 4.7e18 after
# 300 steps, at which point the global gradient clip hands the entire budget to
# one task and every other head stops learning, while the printed loss looks
# like spectacular progress.
#
# ddG NLL goes negative once the typical predicted sigma drops below ~0.24
# kcal/mol, which is reachable on MegaScale where many mutations are
# near-neutral and well predicted. So this is not a corner case.

#: 0.5 * (log(2*pi) + LOGVAR_MIN), with LOGVAR_MIN = -10 as clamped in
#: gaussian_nll and in SubstitutionScoreHead. This is the lowest value any task
#: loss in this project can attain.
GLOBAL_LOSS_FLOOR = -4.0811

#: Round-2 review, R2-07: the floors used to be per-task - -4.0811 for the two
#: Gaussian-NLL tasks and 0.0 for the cross-entropy and MSE ones - which is a
#: constant that differs BETWEEN tasks, and therefore not neutral. The
#: equilibrium weight is ``0.5 / (L - floor)``, so at a typical ddG NLL of 1.0
#: the primary task received weight 0.098 against pathogenicity's 0.833: an 8.5x
#: imbalance created by an asymmetric constant rather than by the data.
#:
#: One floor for every task fixes that. It is still a valid lower bound (every
#: loss here exceeds it, cross-entropy and MSE trivially so), it still makes the
#: shifted quantity non-negative as Kendall & Gal's derivation requires, and a
#: UNIFORM offset preserves the relative weighting between tasks.
LOSS_FLOORS: dict[str, float] = {
    "stability": GLOBAL_LOSS_FLOOR,
    "idr": GLOBAL_LOSS_FLOOR,
    "three_state": GLOBAL_LOSS_FLOOR,
    "pathogenicity": GLOBAL_LOSS_FLOOR,
    "mfi": GLOBAL_LOSS_FLOOR,
    "proteomics": GLOBAL_LOSS_FLOOR,
}

#: Bound on the learned log-variances. Spans a 400x weight range, wide enough
#: for genuinely different task scales and narrow enough that one task cannot
#: monopolise the shared gradient.
TASK_LOGVAR_MIN, TASK_LOGVAR_MAX = -3.0, 3.0


class UncertaintyWeightedLoss(nn.Module):
    """
    Learned per-task weighting (Kendall & Gal, 2018), made bounded below.

    Two guards on the original formulation:

    * Each task loss is shifted by :data:`GLOBAL_LOSS_FLOOR` so the weighted
      term sees a non-negative quantity, as the derivation requires.

      The shift is constant with respect to the model PARAMETERS, but it is
      not gradient-neutral, and the docstring here used to say it was
      (round-2 review, R2-07). ``log_var`` is learned, its stationary point
      is ``lv* = log(2 * (L - floor))``, so the floor moves ``lv*``, moves
      ``exp(-lv)``, and therefore moves the gradient the model receives from
      that task. What makes it safe is that the SAME floor applies to every
      task: a uniform offset preserves the relative weighting, where the
      per-task floors it replaced did not.
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

    def equilibrium_weights(self, losses: dict) -> dict:
        """
        The weight each task would receive at its stationary log-variance.

        ``lv* = log(2 * (L - floor))`` minimises the per-task term, giving
        weight ``0.5 / (L - floor)`` and a contribution of exactly 0.5. Handy
        for reading a multi-task run: if one task's weight is an order of
        magnitude below another's, that is the objective's doing and not the
        data's (R2-07).
        """
        out = {}
        for name, value in losses.items():
            floor = self.floors.get(name, 0.0)
            shifted = float(value) - floor
            if shifted <= 0:
                out[name] = None
                continue
            lv = min(max(math.log(2.0 * shifted), TASK_LOGVAR_MIN),
                     TASK_LOGVAR_MAX)
            out[name] = math.exp(-lv)
        return out

    def current_weights(self) -> dict:
        """``exp(-log_var)`` per task, as the objective is applying it now."""
        return {
            name: float(torch.exp(-p.detach().clamp(TASK_LOGVAR_MIN,
                                                    TASK_LOGVAR_MAX)))
            for name, p in self.log_var.items()
        }

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
    "GLOBAL_LOSS_FLOOR",
    "LOSS_FLOORS",
    "gaussian_nll",
    "beta_gaussian_nll",
    "sigma_gaussian_nll",
    "antisymmetry_violation",
    "UncertaintyWeightedLoss",
    "three_state_loss",
    "pathogenicity_loss",
]
