"""Losses for training NSFA / NSPDA end-to-end.

The forward algorithm in `FiniteAutomaton.accept_soft` already runs in
log-semiring for numerical stability over long sequences; the losses
here consume `log P_acc` directly and stay in log space until the
final reduction.

Notation:
    log_p_accept  -- log of acceptance probability, shape (B,) or scalar
    y             -- target label in {0, 1}, same shape as log_p_accept
    log1mexp(x)   -- numerically stable log(1 - exp(x)) for x <= 0
"""

import math

import torch
import torch.nn.functional as F


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    r"""Compute $\log(1 - \exp(x))$ stably for $x \le 0$.

    Splits at $-\log 2$ following Mächler (2012, "Accurately Computing
    $\log(1 - \exp(-a))$") to keep both `log(-expm1(x))` (used near
    zero) and `log1p(-exp(x))` (used for very negative $x$) in their
    well-conditioned regimes.

    Numerical safety: callers (e.g. `sequence_acceptance_loss` on top
    of the forward algorithm's `log P_acc`) can pass `x` slightly above
    0 due to floating-point error in `logsumexp` -- a typical
    overshoot is ~1e-7. Mathematically `log(1 - exp(0)) = -inf` and
    anything above is undefined. We clamp at $-10^{-7}$ before the
    branch so `x` stays in-domain; the result at the clamp boundary
    is `log(1 - exp(-1e-7))` ≈ `-16.1`, a sensible "very small
    probability" stand-in for the boundary case. The clamp's gradient
    is zero above the threshold; this is acceptable because the
    threshold is only reached by floating-point drift, never by
    well-behaved training updates.

    Args:
        x: A tensor of non-positive values (clamped if marginally
            above).

    Returns:
        $\log(1 - \exp(x))$, same shape as `x`.
    """
    x = x.clamp_max(-1e-7)
    threshold = -math.log(2.0)
    return torch.where(
        x > threshold,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def sequence_acceptance_loss(
    log_p_accept: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Binary cross-entropy loss given $\log P_{\text{acc}}$ and labels $y$.

    Implements
    $$
    \mathcal{L} = -y \log P_{\text{acc}}
                  - (1 - y) \log\!\bigl(1 - P_{\text{acc}}\bigr),
    $$
    evaluated in log space so the per-step underflow risk of long
    sequences does not propagate into the loss. The
    $\log(1 - P_{\text{acc}})$ term is computed via `log1mexp`.

    Args:
        log_p_accept: $\log P_{\text{acc}}$ as returned by
            `FiniteAutomaton.accept_soft(..., log_space=True)` or
            `NSFA(images, log_space=True)`. Shape `(B,)` for a batch,
            or scalar for a single example. Must be non-positive.
        y: Binary labels in `{0, 1}`. Same shape as `log_p_accept`.
            Float or integer tensor.
        reduction: One of `"mean"`, `"sum"`, `"none"`. Defaults to
            `"mean"`.

    Returns:
        BCE loss, scalar (`"mean"` / `"sum"`) or per-example
        (`"none"`).

    Raises:
        ValueError: If `reduction` is not one of the allowed values.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}"
        )
    y = y.to(log_p_accept.dtype)
    # The naive `-y log p - (1-y) log(1-p)` formulation produces NaN at
    #   the simplex boundary because the dormant term is `0 * (-inf)`.
    #   Earlier drafts tried `torch.where` on the *outputs* of the
    #   forward pass to mask the dormant term, but PyTorch's `where`
    #   backward zeros the gradient through the inactive branch via a
    #   `0 * intermediate-Jacobian` multiplication that yields NaN when
    #   the intermediate is `-inf`. The robust fix is to mask the
    #   *inputs* to `log1mexp` (and to the loss_pos product) so no
    #   intermediate is ever `-inf` -- both branches of every `where`
    #   stay finite end-to-end.
    safe_for_log1mexp = torch.where(
        y == 1, torch.full_like(log_p_accept, -1.0), log_p_accept
    )
    log_one_minus_p = log1mexp(safe_for_log1mexp)
    safe_log_p = torch.where(
        y == 0, torch.zeros_like(log_p_accept), log_p_accept
    )
    loss = -y * safe_log_p - (1.0 - y) * log_one_minus_p
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
