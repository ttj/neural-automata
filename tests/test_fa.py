"""Tests for `nsa.fa.FiniteAutomaton`.

Covers:
  - from_dfa, from_regex, from_tensor construction
  - tensor view: to_index_tensor, to_one_hot, to_logits
  - discrete acceptance (delegates to automata-lib)
  - soft acceptance via the forward algorithm (linear and log-semiring)
  - streaming step_soft consistency with accept_soft
  - composition (union, intersection, complement, minify)
  - serialization round-trip through from_tensor

Math reference:
  Plan Section 2.1 (Equation: forward algorithm).
"""

import pytest
import torch
from automata.fa.dfa import DFA

from nsa.alphabet import Alphabet
from nsa.fa import FiniteAutomaton, SINK_LABEL


@pytest.fixture
def ends_with_k_fa() -> FiniteAutomaton:
    """The paper's running example: DFA accepting strings ending with 'k'.

    Two states q1 (non-accepting, initial) and q2 (accepting); on input
    'k' both states transition to q2, on any other letter both
    transition to q1.
    """
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


class TestConstruction:
    def test_from_dfa(self, ends_with_k_fa: FiniteAutomaton) -> None:
        assert ends_with_k_fa.n_states == 2
        assert ends_with_k_fa.n_alphabet == 3
        assert ends_with_k_fa.q0_index == 0
        assert ends_with_k_fa.F_indices == [1]

    def test_from_regex_simple(self) -> None:
        fa = FiniteAutomaton.from_regex("a*b", input_symbols={"a", "b"})
        assert fa.accept("ab")
        assert fa.accept("aaab")
        assert fa.accept("b")
        assert not fa.accept("a")
        assert not fa.accept("ba")

    def test_from_regex_paper_example(self) -> None:
        # Approximates the paper's row 1: u+[j-r][h-t][a-z][a-z]l[a-z]
        # We use a smaller regex to keep the test fast; the paper-table
        # parity test runs in test_regex_example.py once the example is
        # wired.
        fa = FiniteAutomaton.from_regex(
            "[a-c][a-c]",
            input_symbols={"a", "b", "c"},
        )
        for w in ["aa", "ab", "ac", "ba", "bb", "bc", "ca", "cb", "cc"]:
            assert fa.accept(w), f"should accept {w}"
        for w in ["a", "abc", ""]:
            assert not fa.accept(w), f"should not accept {w}"


class TestTensorView:
    def test_to_index_tensor_shape(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        t = ends_with_k_fa.to_index_tensor()
        assert t.shape == (2, 3)
        assert t.dtype == torch.long

    def test_to_index_tensor_values(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        t = ends_with_k_fa.to_index_tensor()
        a_idx = ends_with_k_fa.alphabet.to_index("a")
        k_idx = ends_with_k_fa.alphabet.to_index("k")
        z_idx = ends_with_k_fa.alphabet.to_index("z")
        # q1 (index 0) on 'k' -> q2 (index 1); on 'a','z' -> q1
        assert int(t[0, a_idx]) == 0
        assert int(t[0, k_idx]) == 1
        assert int(t[0, z_idx]) == 0
        # q2 (index 1) on 'k' -> q2; on 'a','z' -> q1
        assert int(t[1, a_idx]) == 0
        assert int(t[1, k_idx]) == 1
        assert int(t[1, z_idx]) == 0

    def test_to_one_hot_sums_to_one(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        one_hot = ends_with_k_fa.to_one_hot()
        # for every (q, a), sum over q' is 1
        assert torch.allclose(one_hot.sum(dim=-1), torch.ones(2, 3))

    def test_to_logits_from_dfa_softmax_close_to_one_hot(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        logits = ends_with_k_fa.to_logits(init="from_dfa", scale=10.0)
        softmaxed = torch.softmax(logits, dim=-1)
        one_hot = ends_with_k_fa.to_one_hot()
        assert torch.allclose(softmaxed, one_hot, atol=1e-3)

    def test_to_logits_identity(self, ends_with_k_fa: FiniteAutomaton) -> None:
        logits = ends_with_k_fa.to_logits(init="identity", scale=10.0)
        softmaxed = torch.softmax(logits, dim=-1)
        # diagonal should be ~1
        for q in range(ends_with_k_fa.n_states):
            for a in range(ends_with_k_fa.n_alphabet):
                assert softmaxed[q, a, q].item() > 0.99


class TestNamedAccessors:
    def test_q0_returns_initial_state_name(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        assert ends_with_k_fa.q0 == "q1"
        assert ends_with_k_fa.q0_index == 0

    def test_F_returns_accepting_state_names(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        assert ends_with_k_fa.F == {"q2"}
        assert ends_with_k_fa.F_indices == [1]


class TestStep:
    def test_step_follows_transitions(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        # q1 on 'k' -> q2; q2 on 'a' -> q1
        assert ends_with_k_fa.step("q1", "k") == "q2"
        assert ends_with_k_fa.step("q2", "a") == "q1"
        assert ends_with_k_fa.step("q1", "a") == "q1"
        assert ends_with_k_fa.step("q2", "k") == "q2"

    def test_step_chained_equals_accept(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        s = ends_with_k_fa.q0
        for c in "aak":
            s = ends_with_k_fa.step(s, c)
        assert s in ends_with_k_fa.F
        assert ends_with_k_fa.accept("aak")

    def test_step_rejects_invalid_symbol(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        with pytest.raises(ValueError, match="not in alphabet"):
            ends_with_k_fa.step("q1", "x")

    def test_step_rejects_invalid_state(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        with pytest.raises(ValueError, match="not in dfa.states"):
            ends_with_k_fa.step("q99", "a")


class TestDiscreteAcceptance:
    def test_accept_matches_dfa(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        assert ends_with_k_fa.accept("ak")
        assert ends_with_k_fa.accept("k")
        assert ends_with_k_fa.accept("aaak")
        assert not ends_with_k_fa.accept("ka")
        assert not ends_with_k_fa.accept("a")

    def test_accept_rejects_out_of_alphabet(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        with pytest.raises(ValueError, match="not in alphabet"):
            ends_with_k_fa.accept("xyz")


class TestSoftAcceptance:
    def test_one_hot_input_matches_discrete(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        """With one-hot symbol probabilities, soft accept = discrete accept."""
        alphabet = ends_with_k_fa.alphabet
        n_a = len(alphabet)
        sequences = ["ak", "kak", "kk", "akz", "z"]
        for seq in sequences:
            symbol_probs = torch.zeros(len(seq), n_a)
            for t, c in enumerate(seq):
                symbol_probs[t, alphabet.to_index(c)] = 1.0
            p_acc = ends_with_k_fa.accept_soft(symbol_probs)
            expected = 1.0 if ends_with_k_fa.accept(seq) else 0.0
            assert abs(p_acc.item() - expected) < 1e-5, (
                f"seq={seq}: soft={p_acc.item()}, hard={expected}"
            )

    def test_uniform_input_gives_intermediate(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        """With uniform symbol probs, acceptance probability is in (0, 1)."""
        n_a = ends_with_k_fa.n_alphabet
        T = 5
        symbol_probs = torch.full((T, n_a), 1.0 / n_a)
        p_acc = ends_with_k_fa.accept_soft(symbol_probs)
        assert 0.0 < p_acc.item() < 1.0

    def test_log_space_matches_linear_space(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        n_a = ends_with_k_fa.n_alphabet
        T = 4
        torch.manual_seed(42)
        logits = torch.randn(T, n_a)
        sp = torch.softmax(logits, dim=-1)
        sp_log = torch.log_softmax(logits, dim=-1)
        p_lin = ends_with_k_fa.accept_soft(sp)
        p_log = ends_with_k_fa.accept_soft(sp_log, log_space=True)
        assert torch.allclose(torch.log(p_lin.clamp_min(1e-30)), p_log, atol=1e-5)

    def test_step_soft_chained_matches_accept_soft(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        """Streaming step_soft over T steps must equal accept_soft over T."""
        n_a = ends_with_k_fa.n_alphabet
        T = 5
        torch.manual_seed(7)
        symbol_probs = torch.softmax(torch.randn(T, n_a), dim=-1)
        # streaming
        pi = ends_with_k_fa.initial_belief()
        for t in range(T):
            pi = ends_with_k_fa.step_soft(pi, symbol_probs[t])
        F_mask = ends_with_k_fa.F_mask()
        p_streaming = pi[F_mask].sum()
        # batched
        p_batched = ends_with_k_fa.accept_soft(symbol_probs)
        assert torch.allclose(p_streaming, p_batched, atol=1e-6)

    def test_gradient_flows_through_delta_logits(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        """`delta_logits` parameters receive gradient from acceptance loss."""
        n_a = ends_with_k_fa.n_alphabet
        delta_logits = ends_with_k_fa.to_logits(
            init="from_dfa", scale=2.0
        ).requires_grad_(True)
        # symbol probs from a fixed log-softmax over noise
        T = 4
        torch.manual_seed(1)
        symbol_probs = torch.softmax(torch.randn(T, n_a), dim=-1)
        p_acc = ends_with_k_fa.accept_soft(
            symbol_probs, delta_logits=delta_logits
        )
        p_acc.backward()
        assert delta_logits.grad is not None
        assert delta_logits.grad.abs().sum().item() > 0


class TestComposition:
    def test_complement(self, ends_with_k_fa: FiniteAutomaton) -> None:
        comp = ends_with_k_fa.complement()
        assert comp.accept("a")
        assert not comp.accept("ak")

    def test_intersection_alphabet_mismatch(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        other = FiniteAutomaton.from_regex("xy*", input_symbols={"x", "y"})
        with pytest.raises(ValueError, match="matching alphabets"):
            ends_with_k_fa.intersection(other)


class TestFromTensorRoundTrip:
    def test_total_dfa_roundtrip(
        self, ends_with_k_fa: FiniteAutomaton
    ) -> None:
        """to_index_tensor -> from_tensor recovers an equivalent FA."""
        delta_index = ends_with_k_fa.to_index_tensor()
        rebuilt = FiniteAutomaton.from_tensor(
            delta_index=delta_index,
            alphabet=ends_with_k_fa.alphabet,
            q0_index=ends_with_k_fa.q0_index,
            F_indices=ends_with_k_fa.F_indices,
            minify=True,
        )
        # equivalent up to state renaming: compare on a sample of strings
        for s in ["ak", "k", "akak", "kkk", "azk", "a", "z", "kkak"]:
            assert rebuilt.accept(s) == ends_with_k_fa.accept(s), (
                f"disagreement on {s!r}: rebuilt={rebuilt.accept(s)}, "
                f"original={ends_with_k_fa.accept(s)}"
            )
