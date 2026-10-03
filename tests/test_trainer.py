"""Smoke tests for `nsa.learning.Trainer`.

Uses a tiny stub perception network and the `ends_with_k` FA fixture
so the tests are fast and deterministic. Verifies the forward/backward
plumbing rather than convergence (which is the job of the regime-(1)
run script).
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from automata.fa.dfa import DFA

from nsa.alphabet import Alphabet
from nsa.fa import FiniteAutomaton
from nsa.learning import Trainer
from nsa.nsfa import NSFA
from nsa.perception import PerceptionAdapter


class StubNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.view(x.shape[0], -1))
        if out.shape[0] == 1:
            return out.view(-1)
        return out


@pytest.fixture
def trainer() -> Trainer:
    alphabet = Alphabet(["a", "k", "z"])
    dfa = DFA(
        states={"q1", "q2"},
        input_symbols={"a", "k", "z"},
        transitions={
            "q1": {"a": "q1", "k": "q2", "z": "q1"},
            "q2": {"a": "q1", "k": "q2", "z": "q1"},
        },
        initial_state="q1",
        final_states={"q2"},
    )
    fa = FiniteAutomaton(dfa, alphabet=alphabet)
    perception = PerceptionAdapter(network=StubNet(), n_alphabet=3)
    nsfa = NSFA(fa, perception, learn_delta=False)
    optimizer = optim.SGD(nsfa.parameters(), lr=0.1)
    return Trainer(nsfa, optimizer, log_space=True)


def _make_dataset(n: int = 8) -> list[tuple[torch.Tensor, int]]:
    torch.manual_seed(0)
    dataset = []
    for i in range(n):
        T = 2 + (i % 3)
        images = torch.randn(T, 1, 4, 4)
        label = i % 2
        dataset.append((images, label))
    return dataset


class TestTrainEpoch:
    def test_train_epoch_returns_finite_loss(
        self, trainer: Trainer
    ) -> None:
        data = _make_dataset()
        loss = trainer.train_epoch(data, batch_accumulate=4)
        assert loss == loss  # not NaN
        assert loss < 1e6

    def test_parameters_change_after_train(
        self, trainer: Trainer
    ) -> None:
        data = _make_dataset()
        before = [p.detach().clone() for p in trainer.nsfa.parameters()]
        trainer.train_epoch(data, batch_accumulate=1)
        after = list(trainer.nsfa.parameters())
        assert any(
            not torch.equal(a, b) for a, b in zip(after, before)
        )


class TestEvaluate:
    def test_returns_loss_and_accuracy(self, trainer: Trainer) -> None:
        data = _make_dataset()
        m = trainer.evaluate(data)
        assert "loss" in m and "accept_accuracy" in m
        assert 0.0 <= m["accept_accuracy"] <= 1.0


class TestEvaluatePerImage:
    def test_per_image_top1(self, trainer: Trainer) -> None:
        images = [(torch.randn(1, 4, 4), i % 3) for i in range(20)]
        m = trainer.evaluate_per_image(images)
        assert "top1" in m
        assert 0.0 <= m["top1"] <= 1.0


class TestRegularizers:
    def _make_learnable_nsfa(self) -> NSFA:
        alphabet = Alphabet(["a", "k", "z"])
        dfa = DFA(
            states={"q1", "q2"},
            input_symbols={"a", "k", "z"},
            transitions={
                "q1": {"a": "q1", "k": "q2", "z": "q1"},
                "q2": {"a": "q1", "k": "q2", "z": "q1"},
            },
            initial_state="q1",
            final_states={"q2"},
        )
        fa = FiniteAutomaton(dfa, alphabet=alphabet)
        perception = PerceptionAdapter(network=StubNet(), n_alphabet=3)
        return NSFA(
            fa,
            perception,
            learn_delta=True,
            delta_init="from_dfa",
            delta_init_scale=2.0,
        )

    def test_unknown_regularizer_raises(self) -> None:
        nsfa = self._make_learnable_nsfa()
        opt = optim.SGD(nsfa.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="unknown regularizer"):
            Trainer(
                nsfa,
                opt,
                regularizer_weights={"nonsense": 0.1},
            )

    def test_row_entropy_term_is_nonzero(self) -> None:
        nsfa = self._make_learnable_nsfa()
        opt = optim.SGD(nsfa.parameters(), lr=0.1)
        trainer = Trainer(
            nsfa, opt, regularizer_weights={"row_entropy": 0.1}
        )
        term = trainer._delta_regularizer_term()
        # delta_logits init=from_dfa, scale=2 -> rows have entropy in (0, log n_q)
        assert float(term) > 0

    def test_regularizer_zero_when_delta_not_learned(self) -> None:
        """Regularizer term is zero when delta_logits is None."""
        alphabet = Alphabet(["a", "k", "z"])
        dfa = DFA(
            states={"q1", "q2"},
            input_symbols={"a", "k", "z"},
            transitions={
                "q1": {"a": "q1", "k": "q2", "z": "q1"},
                "q2": {"a": "q1", "k": "q2", "z": "q1"},
            },
            initial_state="q1",
            final_states={"q2"},
        )
        fa = FiniteAutomaton(dfa, alphabet=alphabet)
        perception = PerceptionAdapter(network=StubNet(), n_alphabet=3)
        nsfa = NSFA(fa, perception, learn_delta=False)
        opt = optim.SGD(nsfa.parameters(), lr=0.1)
        trainer = Trainer(
            nsfa, opt, regularizer_weights={"row_entropy": 0.1}
        )
        term = trainer._delta_regularizer_term()
        assert float(term) == 0.0

    def test_regularizer_gradient_flows_into_delta_logits(self) -> None:
        nsfa = self._make_learnable_nsfa()
        opt = optim.SGD(nsfa.parameters(), lr=0.1)
        trainer = Trainer(
            nsfa, opt, regularizer_weights={"row_entropy": 1.0}
        )
        data = _make_dataset()
        before = nsfa.delta_logits.detach().clone()
        trainer.train_epoch(data, batch_accumulate=4)
        after = nsfa.delta_logits.detach()
        assert not torch.equal(before, after)


class TestFit:
    def test_fit_records_per_epoch_history(
        self, trainer: Trainer
    ) -> None:
        data = _make_dataset()
        history = trainer.fit(data, data, epochs=2, batch_accumulate=4)
        assert len(history) == 2
        for record in history:
            for key in (
                "epoch",
                "train_loss",
                "val_loss",
                "val_accept_accuracy",
            ):
                assert key in record

    def test_fit_restores_best_state(
        self, trainer: Trainer, tmp_path
    ) -> None:
        data = _make_dataset()
        trainer.fit(
            data,
            data,
            epochs=3,
            batch_accumulate=4,
            save_dir=tmp_path,
        )
        assert (tmp_path / "best.pth").exists()
