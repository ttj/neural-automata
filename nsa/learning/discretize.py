"""Discretization tools: project a trained delta_theta back to a discrete DFA.

After end-to-end training under regime (b) of the plan
(`.claude/plans/2026-05-12_neutron-refactor-and-e2e-scaffold.md`
Section 2.3), the soft transition tensor `delta_logits` is projected
to a discrete transition function by argmax along the next-state
dimension. The resulting `(n_states, n_alphabet)` index tensor is
fed to `FiniteAutomaton.from_tensor` to recover a symbolic DFA,
which is then minified via `automata-lib`'s Hopcroft implementation
and compared (up to isomorphism) to the hand-built reference DFA.

Per-state visitation statistics from the soft runtime are used to
identify states that the trained model never relied on; those are
candidates for pruning.

Notation:
    delta_logits   -- Tensor(n_states, n_alphabet, n_states)
    delta_index    -- LongTensor(n_states, n_alphabet) of next states
    pi_batch       -- Tensor(B, T, n_states) state beliefs
"""

from typing import Hashable, Iterable, Sequence

import torch

from nsa.alphabet import Alphabet
from nsa.fa import FiniteAutomaton


def discretize_logits(delta_logits: torch.Tensor) -> torch.Tensor:
    """Argmax along the next-state dimension.

    Args:
        delta_logits: Shape `(n_states, n_alphabet, n_states)`.

    Returns:
        LongTensor of shape `(n_states, n_alphabet)` whose `[q, a]`
        entry is `argmax_q' delta_logits[q, a, q']`.
    """
    if delta_logits.dim() != 3:
        raise ValueError(
            "delta_logits must be 3D (n_states, n_alphabet, n_states); "
            f"got shape {tuple(delta_logits.shape)}"
        )
    return torch.argmax(delta_logits, dim=-1)


def discretize_to_fa(
    delta_logits: torch.Tensor,
    alphabet: Alphabet,
    q0_index: int,
    F_indices: Sequence[int],
    state_names: Iterable[Hashable] | None = None,
    minify: bool = True,
) -> FiniteAutomaton:
    """Project `delta_logits` to a `FiniteAutomaton` via argmax + minify.

    Composes `discretize_logits` with
    `FiniteAutomaton.from_tensor`. Use this at the end of a training
    run to recover a discrete DFA that can be compared to the
    hand-built reference (after both are minified by Hopcroft, they
    are equivalent iff isomorphic).

    Args:
        delta_logits: Shape `(n_states, n_alphabet, n_states)`.
        alphabet: Alphabet matching the second axis of `delta_logits`.
        q0_index: Initial state index (typically 0).
        F_indices: Accepting state indices.
        state_names: Optional state names; if absent, states are
            named `q0`, `q1`, ... by index.
        minify: Whether to run Hopcroft minification on the result.

    Returns:
        A `FiniteAutomaton` wrapping a discrete DFA equivalent to the
        trained soft model under argmax projection.
    """
    delta_index = discretize_logits(delta_logits)
    return FiniteAutomaton.from_tensor(
        delta_index=delta_index,
        alphabet=alphabet,
        q0_index=q0_index,
        F_indices=F_indices,
        state_names=list(state_names) if state_names is not None else None,
        minify=minify,
    )


def state_usage_report(
    pi_batch: torch.Tensor,
    state_names: Sequence[Hashable] | None = None,
) -> dict[Hashable, float]:
    """Per-state visitation summary; surfaces never-used (prunable) states.

    Args:
        pi_batch: Shape `(B, T, n_states)` (or `(T, n_states)` for a
            single sequence). State beliefs from the soft runtime.
        state_names: Optional state names of length `n_states`; if
            absent, indices `0`, `1`, ... are used as keys.

    Returns:
        A dict mapping each state name (or index) to its mean
        marginal visitation probability $\\bar\\pi_q$, sorted by
        descending visitation. States with $\\bar\\pi_q \\approx 0$
        are pruning candidates.
    """
    if pi_batch.dim() == 2:
        bar = pi_batch.mean(dim=0)
    elif pi_batch.dim() == 3:
        bar = pi_batch.mean(dim=(0, 1))
    else:
        raise ValueError(
            f"pi_batch must be 2D or 3D; got shape {tuple(pi_batch.shape)}"
        )
    n_states = bar.shape[0]
    keys: list[Hashable]
    if state_names is None:
        keys = list(range(n_states))
    else:
        if len(state_names) != n_states:
            raise ValueError(
                f"len(state_names) {len(state_names)} != n_states {n_states}"
            )
        keys = list(state_names)
    pairs = sorted(
        zip(keys, bar.detach().cpu().tolist()),
        key=lambda x: -x[1],
    )
    return dict(pairs)
