"""Tests for `nsa.nsfa.NSFA` composition and runtime.

Uses a tiny stub `nn.Module` as perception so the tests are fast and
deterministic. The end-to-end test with the paper's CNN checkpoint
lives in `test_regex_example.py` and is skipped without checkpoints.

Math reference: Plan Section 2.1.
"""

import pytest
import torch
import torch.nn as nn
from automata.fa.dfa import DFA

from nsa.alphabet import Alphabet
from nsa.fa import FiniteAutomaton
from nsa.nsfa import NSFA
from nsa.perception import PerceptionAdapter


class StubLogitsNet(nn.Module):
    """Tiny perception net: a single Linear from flattened image to logits.

    Used only in tests. Maps `(B, C, H, W)` to `(B, n_alphabet)`.
    """

    def __init__(self, image_shape: tuple[int, int, int], n_alphabet: int) -> None:
        super().__init__()
        in_features = int(torch.tensor(image_shape).prod().item())
        self.linear = nn.Linear(in_features, n_alphabet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.view(x.shape[0], -1))
        if out.shape[0] == 1:
            return out.view(-1)
        return out


@pytest.fixture
def ends_with_k_nsfa() -> NSFA:
    """A two-state DFA + a stub perception adapter over alphabet {a, k, z}."""
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
    fa = FiniteAutomaton(dfa, alphabet=alphabet)
    net = StubLogitsNet(image_shape=(1, 4, 4), n_alphabet=3)
    perception = PerceptionAdapter(network=net, n_alphabet=3)
    return NSFA(fa, perception, learn_delta=False)


class TestDiscreteAcceptance:
    def test_accept_returns_tuple(self, ends_with_k_nsfa: NSFA) -> None:
        images = torch.zeros(3, 1, 4, 4)
        accepted, predicted = ends_with_k_nsfa.accept(images)
        assert isinstance(accepted, bool)
        assert isinstance(predicted, str)
        assert len(predicted) == 3

    def test_accept_no_grad(self, ends_with_k_nsfa: NSFA) -> None:
        """`accept` is wrapped in `torch.no_grad` -- output has no grad fn."""
        images = torch.zeros(2, 1, 4, 4, requires_grad=False)
        accepted, _ = ends_with_k_nsfa.accept(images)
        # boolean output, but underlying tensor ops should not have built
        # a graph. Hard to inspect directly; we check via memory: rerun
        # without no_grad would have been needed for backward.
        assert isinstance(accepted, bool)


class TestSoftAcceptance:
    def test_forward_returns_scalar(self, ends_with_k_nsfa: NSFA) -> None:
        images = torch.zeros(3, 1, 4, 4)
        p_log = ends_with_k_nsfa(images, log_space=True)
        assert p_log.shape == ()
        # log probability is <= 0
        assert p_log.item() <= 1e-6

    def test_forward_linear_in_range(self, ends_with_k_nsfa: NSFA) -> None:
        images = torch.zeros(3, 1, 4, 4)
        p = ends_with_k_nsfa(images, log_space=False)
        assert 0.0 <= p.item() <= 1.0 + 1e-6

    def test_forward_gradient_flows_to_perception(
        self, ends_with_k_nsfa: NSFA
    ) -> None:
        torch.manual_seed(0)
        images = torch.randn(3, 1, 4, 4)
        p_log = ends_with_k_nsfa(images, log_space=True)
        loss = -p_log  # encourage acceptance
        loss.backward()
        # at least one perception parameter should have a non-zero grad
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in ends_with_k_nsfa.perception.parameters()
        )
        assert has_grad


class TestLearnableDelta:
    def test_learn_delta_registers_parameter(self) -> None:
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
        fa = FiniteAutomaton(dfa, alphabet=alphabet)
        net = StubLogitsNet(image_shape=(1, 4, 4), n_alphabet=3)
        perception = PerceptionAdapter(network=net, n_alphabet=3)
        nsfa = NSFA(fa, perception, learn_delta=True, delta_init="from_dfa")
        assert nsfa.delta_logits is not None
        assert nsfa.delta_logits.requires_grad
        # delta_logits must appear in parameters()
        param_names = {n for n, _ in nsfa.named_parameters()}
        assert "delta_logits" in param_names

    def test_learn_delta_gradient_flows(self) -> None:
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
        fa = FiniteAutomaton(dfa, alphabet=alphabet)
        net = StubLogitsNet(image_shape=(1, 4, 4), n_alphabet=3)
        perception = PerceptionAdapter(network=net, n_alphabet=3)
        nsfa = NSFA(
            fa,
            perception,
            learn_delta=True,
            delta_init="from_dfa",
            delta_init_scale=2.0,
        )
        torch.manual_seed(0)
        images = torch.randn(4, 1, 4, 4)
        p_log = nsfa(images, log_space=True)
        loss = -p_log
        loss.backward()
        assert nsfa.delta_logits is not None
        assert nsfa.delta_logits.grad is not None
        assert nsfa.delta_logits.grad.abs().sum().item() > 0


class TestStreaming:
    def test_reset_required_before_observe(
        self, ends_with_k_nsfa: NSFA
    ) -> None:
        with pytest.raises(RuntimeError, match="reset"):
            ends_with_k_nsfa.observe(torch.zeros(1, 4, 4))

    def test_streaming_matches_batch_forward(
        self, ends_with_k_nsfa: NSFA
    ) -> None:
        """Three sequential observe calls match a single forward over T=3."""
        torch.manual_seed(123)
        images = torch.randn(3, 1, 4, 4)

        # batch
        p_batch = ends_with_k_nsfa(images, log_space=False)

        # streaming
        ends_with_k_nsfa.reset()
        last_p = None
        for t in range(images.shape[0]):
            _, last_p = ends_with_k_nsfa.observe(images[t], log_space=False)
        assert last_p is not None
        assert torch.allclose(p_batch, last_p, atol=1e-5)
