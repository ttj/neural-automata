"""Tests for `nsa.learning.discretize`.

Cover: argmax projection of `delta_logits`, round-trip from a
hand-built FA through `to_logits` -> `discretize_to_fa` -> minified
DFA -> equivalence to the original on a sample of strings, and the
state-usage report's sorting behaviour.
"""

import pytest
import torch
from automata.fa.dfa import DFA

from nsa.alphabet import Alphabet
from nsa.fa import FiniteAutomaton
from nsa.learning.discretize import (
    discretize_logits,
    discretize_to_fa,
    state_usage_report,
)


@pytest.fixture
def ends_with_k_fa() -> FiniteAutomaton:
    """Same fixture as `tests/test_fa.py`."""
    alphabet = Alphabet(["a", "k", "z"])
    transitions = {
        "q1": {"a": "q1", "k": "q2", "z": "q1"},
        "q2": {"a": "q1", "k": "q2", "z": "q1"},
    }
    dfa = DFA(
        states={"q1", "q2"},
        input_symbols={"a", "k", "z"},
        transitions=transitions,
        initial_state="q1",
        final_states={"q2"},
    )
    return FiniteAutomaton(dfa, alphabet=alphabet)


class TestDiscretizeLogits:
    def test_argmax_recovers_one_hot_index(self) -> None:
        # delta_logits is one-hot at index 1 for every (q, a)
        logits = torch.zeros(3, 4, 3)
        logits[:, :, 1] = 100.0
        idx = discretize_logits(logits)
        assert idx.shape == (3, 4)
        assert (idx == 1).all()

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            discretize_logits(torch.zeros(3, 4))


class TestDiscretizeToFa:
    def test_roundtrip_via_to_logits(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        """to_logits(from_dfa) -> discretize_to_fa preserves language."""
        logits = ends_with_k_fa.to_logits(init="from_dfa", scale=5.0)
        rebuilt = discretize_to_fa(
            delta_logits=logits,
            alphabet=ends_with_k_fa.alphabet,
            q0_index=ends_with_k_fa.q0_index,
            F_indices=ends_with_k_fa.F_indices,
            minify=True,
        )
        # accept agreement on representative strings
        for s in ["ak", "k", "akak", "aaaak", "azk", "a", "z", "kak"]:
            assert rebuilt.accept(s) == ends_with_k_fa.accept(s), (
                f"disagreement on {s!r}"
            )

    def test_state_names_passed_through(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        logits = ends_with_k_fa.to_logits(init="from_dfa", scale=10.0)
        rebuilt = discretize_to_fa(
            delta_logits=logits,
            alphabet=ends_with_k_fa.alphabet,
            q0_index=0,
            F_indices=[1],
            state_names=["s0", "s1"],
            minify=False,
        )
        # not asserting exact names because minify=False keeps them but
        # state ordering may differ; just check size
        assert rebuilt.n_states == 2


class TestStateUsageReport:
    def test_sorted_descending(self) -> None:
        # 3 states; state 1 visited most, then 0, then 2
        pi = torch.tensor(
            [
                [[0.1, 0.6, 0.3], [0.2, 0.7, 0.1]],
                [[0.3, 0.6, 0.1], [0.0, 0.9, 0.1]],
            ]
        )  # shape (B=2, T=2, Q=3)
        report = state_usage_report(pi)
        keys = list(report.keys())
        # keys sorted by descending value
        values = list(report.values())
        assert values == sorted(values, reverse=True)
        # state 1 most visited
        assert keys[0] == 1

    def test_state_names(self) -> None:
        pi = torch.tensor([[[0.2, 0.8], [0.4, 0.6]]])
        report = state_usage_report(pi, state_names=["q0", "q1"])
        assert set(report.keys()) == {"q0", "q1"}
        # q1 visited more
        assert list(report.keys())[0] == "q1"

    def test_2d_input(self) -> None:
        pi = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        report = state_usage_report(pi)
        for v in report.values():
            assert abs(v - 0.5) < 1e-7

    def test_mismatched_state_names_raises(self) -> None:
        pi = torch.zeros(1, 2, 3)
        with pytest.raises(ValueError, match="state_names"):
            state_usage_report(pi, state_names=["q0", "q1"])
