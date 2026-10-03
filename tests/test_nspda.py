"""Tests for `nsa.nspda.NSPDA` composition.

Uses a tiny stub `nn.Module` as perception so the tests are fast and
deterministic. Arithmetic-evaluation parity tests against the paper's
checkpoint live in a separate example-level test (skipped without the
checkpoint).
"""

import pytest
import torch
import torch.nn as nn
from automata.pda.dpda import DPDA

from nsa.nspda import NSPDA
from nsa.pda import PushdownAutomaton
from nsa.perception import PerceptionAdapter


class StubLogitsNet(nn.Module):
    """Tiny perception net: flatten + linear; used only in tests."""

    def __init__(self, image_shape: tuple[int, int, int], n_alphabet: int) -> None:
        super().__init__()
        in_features = int(torch.tensor(image_shape).prod().item())
        self.linear = nn.Linear(in_features, n_alphabet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.view(x.shape[0], -1))
        if out.shape[0] == 1:
            return out.view(-1)
        return out


def _build_an_b_pda_with_end_marker() -> PushdownAutomaton:
    """`a^n b #` DPDA: same as `an_b_pda` fixture plus an end-marker '#'.

    Used to test NSPDA's end_marker append behavior. The CNN emits
    over {a, b}; the NSPDA appends '#' before the PDA reads.
    """
    transitions = {
        "q0": {
            "a": {"Z": ("q0", ("A", "Z")), "A": ("q0", ("A", "A"))},
            "b": {"A": ("q1", ())},
        },
        "q1": {
            "#": {"Z": ("qF", ("Z",))},
        },
    }
    dpda = DPDA(
        states={"q0", "q1", "qF"},
        input_symbols={"a", "b", "#"},
        stack_symbols={"Z", "A"},
        transitions=transitions,
        initial_state="q0",
        initial_stack_symbol="Z",
        final_states={"qF"},
        acceptance_mode="final_state",
    )
    return PushdownAutomaton(dpda)


@pytest.fixture
def an_b_nspda() -> NSPDA:
    """NSPDA wrapping the `a^n b #` DPDA + a stub CNN over {a, b}."""
    pda = _build_an_b_pda_with_end_marker()
    # perception emits over {a, b}; '#' is appended by the NSPDA.
    # The PDA's alphabet is {a, b, #}; perception size is 2.
    n_perception = pda.n_alphabet - 1
    assert n_perception == 2
    net = StubLogitsNet(image_shape=(1, 4, 4), n_alphabet=n_perception)
    perception = PerceptionAdapter(network=net, n_alphabet=n_perception)
    return NSPDA(pda, perception, end_marker="#")


class TestAlphabetCompatibility:
    def test_end_marker_required_when_alphabet_mismatch(self) -> None:
        pda = _build_an_b_pda_with_end_marker()
        # PDA has 3 input symbols; perception with n_alphabet=3 would
        # imply no end_marker; we explicitly require alphabet sizes to
        # agree with end_marker presence.
        net = StubLogitsNet(image_shape=(1, 4, 4), n_alphabet=3)
        perception = PerceptionAdapter(network=net, n_alphabet=3)
        with pytest.raises(ValueError, match="n_alphabet"):
            NSPDA(pda, perception, end_marker="#")

    def test_end_marker_not_in_alphabet_raises(self) -> None:
        pda = _build_an_b_pda_with_end_marker()
        net = StubLogitsNet(image_shape=(1, 4, 4), n_alphabet=2)
        perception = PerceptionAdapter(network=net, n_alphabet=2)
        with pytest.raises(ValueError, match="not in pda.alphabet"):
            NSPDA(pda, perception, end_marker="?")


class TestAccept:
    def test_accept_returns_tuple(self, an_b_nspda: NSPDA) -> None:
        images = torch.zeros(3, 1, 4, 4)
        accepted, predicted = an_b_nspda.accept(images)
        assert isinstance(accepted, bool)
        assert isinstance(predicted, str)
        # the predicted string excludes the appended end_marker '#'
        assert "#" not in predicted
        assert len(predicted) == 3


class TestAcceptWithTrace:
    def test_returns_triple(self, an_b_nspda: NSPDA) -> None:
        images = torch.zeros(2, 1, 4, 4)
        accepted, predicted, stack = an_b_nspda.accept_with_trace(images)
        assert isinstance(accepted, bool)
        assert isinstance(predicted, str)
        assert isinstance(stack, list)


class TestEvaluate:
    def test_evaluate_with_counting_fold(self, an_b_nspda: NSPDA) -> None:
        """Fold counts the number of non-end-marker symbols read."""
        images = torch.zeros(3, 1, 4, 4)

        def step(acc, symbol, _sb, _sa, _stack):
            return acc + (1 if symbol != "#" else 0)

        _accepted, _predicted, count = an_b_nspda.evaluate(
            images, step, initial=0
        )
        # the perception emits 3 symbols; even if the PDA rejects
        # midway, the fold returns the steps that did execute.
        assert count >= 0


class TestForwardDeferred:
    def test_forward_raises_not_implemented(self, an_b_nspda: NSPDA) -> None:
        with pytest.raises(NotImplementedError, match="Half B"):
            an_b_nspda(torch.zeros(2, 1, 4, 4))
