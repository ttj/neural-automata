"""Tests for `nsa.pda.PushdownAutomaton`.

Uses a small `a^n b` (n >= 1) DPDA to exercise:
  - push on 'a' (with stack top Z or A)
  - pop on 'b' (with stack top A)
  - final-state acceptance

This covers the stack action conventions from `automata-lib`
(top-down push tuple, empty tuple for pop) that the arithmetic
example will rely on.
"""

import pytest
from automata.pda.dpda import DPDA

from nsa.alphabet import Alphabet
from nsa.pda import PushdownAutomaton


@pytest.fixture
def an_b_pda() -> PushdownAutomaton:
    """DPDA accepting strings of the form `a^n b` with n >= 1.

    States:
        q0  -- initial, reading a's
        q1  -- final, after the single 'b'
    Stack alphabet: {Z (bottom marker), A}
    """
    transitions = {
        "q0": {
            "a": {
                "Z": ("q0", ("A", "Z")),
                "A": ("q0", ("A", "A")),
            },
            "b": {
                "A": ("q1", ()),
            },
        },
    }
    dpda = DPDA(
        states={"q0", "q1"},
        input_symbols={"a", "b"},
        stack_symbols={"Z", "A"},
        transitions=transitions,
        initial_state="q0",
        initial_stack_symbol="Z",
        final_states={"q1"},
        acceptance_mode="final_state",
    )
    return PushdownAutomaton(dpda)


class TestConstruction:
    def test_dimensions(self, an_b_pda: PushdownAutomaton) -> None:
        assert an_b_pda.n_states == 2
        assert an_b_pda.n_alphabet == 2
        assert an_b_pda.n_stack == 2
        assert an_b_pda.q0_index == 0
        assert an_b_pda.Z0_index == 0
        assert an_b_pda.F_indices == [1]
        assert an_b_pda.acceptance_mode == "final_state"

    def test_initial_state_at_index_0(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        assert an_b_pda.states.to_symbol(0) == "q0"

    def test_initial_stack_at_index_0(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        assert an_b_pda.stack_alphabet.to_symbol(0) == "Z"


class TestNamedAccessors:
    def test_q0(self, an_b_pda: PushdownAutomaton) -> None:
        assert an_b_pda.q0 == "q0"
        assert an_b_pda.q0_index == 0

    def test_Z0(self, an_b_pda: PushdownAutomaton) -> None:
        assert an_b_pda.Z0 == "Z"
        assert an_b_pda.Z0_index == 0

    def test_F(self, an_b_pda: PushdownAutomaton) -> None:
        assert an_b_pda.F == {"q1"}
        assert an_b_pda.F_indices == [1]


class TestAccept:
    def test_accepts_one_a_one_b(self, an_b_pda: PushdownAutomaton) -> None:
        assert an_b_pda.accept("ab")

    def test_accepts_many_a_one_b(self, an_b_pda: PushdownAutomaton) -> None:
        assert an_b_pda.accept("aaab")
        assert an_b_pda.accept("aaaaab")

    def test_rejects_no_a(self, an_b_pda: PushdownAutomaton) -> None:
        assert not an_b_pda.accept("b")

    def test_rejects_no_b(self, an_b_pda: PushdownAutomaton) -> None:
        assert not an_b_pda.accept("a")
        assert not an_b_pda.accept("aaa")

    def test_rejects_too_many_b(self, an_b_pda: PushdownAutomaton) -> None:
        assert not an_b_pda.accept("abb")
        assert not an_b_pda.accept("aabb")

    def test_rejects_empty(self, an_b_pda: PushdownAutomaton) -> None:
        assert not an_b_pda.accept("")

    def test_rejects_out_of_alphabet(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        with pytest.raises(ValueError, match="not in alphabet"):
            an_b_pda.accept("xyz")


class TestAcceptWithTrace:
    def test_accepted_final_stack(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        """Final stack after `aab` is [Z, A] (one A popped from [Z, A, A]).

        automata-lib stores the stack **bottom-up**: `stack[0]` is the
        bottom symbol; `stack[-1]` is the top. See
        `nsa.pda` module docstring.
        """
        accepted, stack = an_b_pda.accept_with_trace("aab")
        assert accepted
        assert stack[0] == "Z"
        assert stack[-1] == "A"
        assert stack.count("A") == 1
        assert len(stack) == 2

    def test_rejected_returns_false(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        accepted, _stack = an_b_pda.accept_with_trace("a")
        assert not accepted


class TestFold:
    def test_fold_counts_a_pushes(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        """Fold counts the number of 'a's read, accumulator independent of stack."""

        def step(acc, symbol, _state_before, _state_after, _stack):
            return acc + (1 if symbol == "a" else 0)

        accepted, count = an_b_pda.fold("aaab", step, initial=0)
        assert accepted
        assert count == 3

    def test_fold_tracks_stack_depth(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        """Fold can record the max stack depth seen during the run."""

        def step(acc, _symbol, _state_before, _state_after, stack):
            return max(acc, len(stack))

        accepted, max_depth = an_b_pda.fold("aab", step, initial=0)
        assert accepted
        # max depth was reached after pushing both A's: stack = [A, A, Z]
        assert max_depth == 3

    def test_fold_on_rejected_input(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        """Fold returns `accepted=False` and the accumulator at point of failure."""

        def step(acc, _symbol, _state_before, _state_after, _stack):
            return acc + 1

        accepted, n_steps = an_b_pda.fold("a", step, initial=0)
        assert not accepted
        # one step was taken (push A) before reaching the end of input
        assert n_steps == 1


class TestExplicitOrderings:
    def test_custom_alphabet_ordering(self) -> None:
        """User-provided alphabet ordering must match dpda.input_symbols."""
        transitions = {
            "q0": {
                "a": {"Z": ("q0", ("A", "Z")), "A": ("q0", ("A", "A"))},
                "b": {"A": ("q1", ())},
            },
        }
        dpda = DPDA(
            states={"q0", "q1"},
            input_symbols={"a", "b"},
            stack_symbols={"Z", "A"},
            transitions=transitions,
            initial_state="q0",
            initial_stack_symbol="Z",
            final_states={"q1"},
            acceptance_mode="final_state",
        )
        pda = PushdownAutomaton(dpda, alphabet=Alphabet(["b", "a"]))
        assert pda.alphabet.to_symbol(0) == "b"
        assert pda.alphabet.to_symbol(1) == "a"

    def test_mismatched_alphabet_raises(self) -> None:
        transitions = {
            "q0": {
                "a": {"Z": ("q0", ("A", "Z")), "A": ("q0", ("A", "A"))},
                "b": {"A": ("q1", ())},
            },
        }
        dpda = DPDA(
            states={"q0", "q1"},
            input_symbols={"a", "b"},
            stack_symbols={"Z", "A"},
            transitions=transitions,
            initial_state="q0",
            initial_stack_symbol="Z",
            final_states={"q1"},
            acceptance_mode="final_state",
        )
        with pytest.raises(ValueError, match="does not match"):
            PushdownAutomaton(dpda, alphabet=Alphabet(["a", "c"]))


class TestSoftAcceptDeferred:
    def test_accept_soft_raises_not_implemented(
        self, an_b_pda: PushdownAutomaton
    ) -> None:
        with pytest.raises(NotImplementedError, match="Half B"):
            an_b_pda.accept_soft()
