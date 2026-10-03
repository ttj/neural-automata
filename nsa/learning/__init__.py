"""End-to-end training building blocks for NSFA / NSPDA.

This subpackage is the Half B scaffolding from
`.claude/plans/2026-05-12_neutron-refactor-and-e2e-scaffold.md`. It
contains the loss functions, regularizers, and discretization tools
that turn a soft `FiniteAutomaton` runtime (`accept_soft`,
`step_soft`) into a trainable component. Initialization helpers live
on `FiniteAutomaton.to_logits` itself and are re-exported here for
convenience.

The training loop itself (`trainer`), the data pipeline, and the
classical-learning sidecar live in sibling modules added in later
phases of Half B.

Public API:
    sequence_acceptance_loss  -- BCE loss on log P_acc
    log1mexp                   -- numerically stable log(1 - exp(x)) for x <= 0
    row_entropy                -- per-row entropy of softmax(delta_logits)
    state_usage_entropy        -- entropy of the marginal state-visit distribution
    determinism_penalty        -- encourages peaky transition rows
    discretize_logits          -- argmax delta_logits along the next-state dim
    discretize_to_fa           -- recover a FiniteAutomaton (with minify) from delta_logits
    state_usage_report         -- per-state visitation summary for pruning
"""

from nsa.learning.discretize import (
    discretize_logits,
    discretize_to_fa,
    state_usage_report,
)
from nsa.learning.losses import log1mexp, sequence_acceptance_loss
from nsa.learning.regularizers import (
    determinism_penalty,
    row_entropy,
    state_usage_entropy,
)
from nsa.learning.trainer import Trainer

__all__ = [
    "determinism_penalty",
    "discretize_logits",
    "discretize_to_fa",
    "log1mexp",
    "row_entropy",
    "sequence_acceptance_loss",
    "state_usage_entropy",
    "state_usage_report",
    "Trainer",
]
