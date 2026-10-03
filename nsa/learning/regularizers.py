"""Regularizers for the learnable transition function delta_theta.

The plan
(`.claude/plans/2026-05-12_neutron-refactor-and-e2e-scaffold.md`
Section 2.4) names four regularizers that materially affect whether
a trained delta_theta discretizes back to an interpretable DFA:

    - row entropy: encourages each (q, a) row of softmax(delta_logits)
      to be peaky, so the argmax projection is meaningful.
    - state-usage entropy: encourages a balanced visitation
      distribution across states, enabling over-provisioning |Q| at
      training time and pruning never-used states at convergence.
    - determinism penalty: 1 - mean of per-row max probability; an
      alternative to row entropy that may train better in some
      regimes.

Identity-init bias and warm-start-from-DFA are realized via
`FiniteAutomaton.to_logits(init="identity" | "from_dfa", scale=...)`
and so do not appear here.

Notation:
    delta_logits  -- Tensor(n_states, n_alphabet, n_states),
                     pre-softmax transition logits
    pi_batch      -- Tensor(B, T, n_states), state beliefs over a
                     batch of B sequences each of length T
"""

import torch
import torch.nn.functional as F

_EPS = 1e-12


def row_entropy(
    delta_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Mean per-row entropy of `softmax(delta_logits, dim=-1)`.

    For each transition origin $(q, a)$, computes
    $$
    H(\delta_\theta(q, a))
      = -\sum_{q'} \delta_\theta(q, a, q')\,
                   \log \delta_\theta(q, a, q'),
    $$
    where $\delta_\theta(q, a, q') = \mathrm{softmax}_{q'} W_\theta[q, a]$.

    Add the negation of this term to the loss to *minimize* entropy
    (i.e. encourage peaky rows). Equivalently, add `+row_entropy(...)`
    with a negative weight, or `-coef * row_entropy(...)` with
    positive `coef`. Both conventions are common; pick one per call
    site.

    Args:
        delta_logits: Shape `(n_states, n_alphabet, n_states)`.
        reduction: `"mean"` (default) over all $(q, a)$ rows, `"sum"`
            for raw total, or `"none"` for the per-row tensor.

    Returns:
        Scalar (or `(n_states, n_alphabet)` tensor) of row entropies.

    Raises:
        ValueError: If `reduction` is not one of the allowed values.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}"
        )
    log_p = F.log_softmax(delta_logits, dim=-1)
    p = torch.exp(log_p)
    row_h = -(p * log_p).sum(dim=-1)
    if reduction == "mean":
        return row_h.mean()
    if reduction == "sum":
        return row_h.sum()
    return row_h


def state_usage_entropy(pi_batch: torch.Tensor) -> torch.Tensor:
    r"""Entropy of the marginal state-visit distribution over a batch.

    Computes
    $$
    H(\bar\pi) = -\sum_q \bar\pi_q \log \bar\pi_q,
    \quad
    \bar\pi_q = \frac{1}{B T} \sum_{b, t} \pi_t^{(b)}(q),
    $$
    where $\pi_t^{(b)}$ is the per-step state belief for example $b$
    at time $t$. Maximizing this entropy encourages all states to be
    visited; combined with row-entropy minimization, this is the
    standard recipe for learning a small interpretable DFA from a
    deliberately over-provisioned $|Q|$.

    Args:
        pi_batch: Tensor of shape `(B, T, n_states)` (or
            `(T, n_states)` for a single example).

    Returns:
        Scalar entropy in nats.
    """
    if pi_batch.dim() == 2:
        bar = pi_batch.mean(dim=0)
    elif pi_batch.dim() == 3:
        bar = pi_batch.mean(dim=(0, 1))
    else:
        raise ValueError(
            f"pi_batch must be 2D or 3D; got shape {tuple(pi_batch.shape)}"
        )
    bar = bar.clamp_min(_EPS)
    return -(bar * torch.log(bar)).sum()


def determinism_penalty(delta_logits: torch.Tensor) -> torch.Tensor:
    r"""Penalize transition rows that are not concentrated on a single state.

    Returns
    $$
    1 - \frac{1}{|Q| \cdot |\Sigma|}
        \sum_{q, a} \max_{q'} \mathrm{softmax}_{q'} W_\theta[q, a, q'],
    $$
    which is zero when every row is one-hot and approaches
    $1 - 1/|Q|$ as rows go uniform. An alternative to (or composable
    with) `row_entropy`.

    Args:
        delta_logits: Shape `(n_states, n_alphabet, n_states)`.

    Returns:
        Scalar in $[0, 1 - 1/|Q|]$.
    """
    p = F.softmax(delta_logits, dim=-1)
    return 1.0 - p.max(dim=-1).values.mean()
