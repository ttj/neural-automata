"""Tests for `nsa.learning.regularizers`.

Verify boundary behaviour: uniform inputs maximize entropy, peaky
inputs minimize entropy / determinism penalty, and shape
preconditions raise.
"""

import math

import pytest
import torch

from nsa.learning.regularizers import (
    determinism_penalty,
    row_entropy,
    state_usage_entropy,
)


class TestRowEntropy:
    def test_uniform_max(self) -> None:
        # uniform logits over n_states give entropy log(n_states)
        n_q, n_a = 3, 5
        logits = torch.zeros(n_q, n_a, n_q)
        h = row_entropy(logits)
        assert abs(h.item() - math.log(n_q)) < 1e-6

    def test_peaky_min(self) -> None:
        # one large logit, rest small -> entropy ~ 0
        n_q, n_a = 3, 5
        logits = torch.zeros(n_q, n_a, n_q)
        logits[:, :, 0] = 100.0
        h = row_entropy(logits)
        assert h.item() < 1e-4

    def test_reduction_none(self) -> None:
        n_q, n_a = 3, 5
        logits = torch.zeros(n_q, n_a, n_q)
        h = row_entropy(logits, reduction="none")
        assert h.shape == (n_q, n_a)
        # every entry equals log(n_q)
        assert torch.allclose(h, torch.full((n_q, n_a), math.log(n_q)))

    def test_gradient_flows(self) -> None:
        n_q, n_a = 2, 3
        logits = torch.randn(n_q, n_a, n_q, requires_grad=True)
        loss = row_entropy(logits)
        loss.backward()
        assert logits.grad is not None


class TestStateUsageEntropy:
    def test_uniform_max(self) -> None:
        # uniform visitation over n_states gives entropy log(n_states)
        B, T, n_q = 4, 5, 3
        pi = torch.full((B, T, n_q), 1.0 / n_q)
        h = state_usage_entropy(pi)
        assert abs(h.item() - math.log(n_q)) < 1e-5

    def test_one_hot_min(self) -> None:
        # always state 0 visited -> entropy ~ 0
        B, T, n_q = 4, 5, 3
        pi = torch.zeros(B, T, n_q)
        pi[:, :, 0] = 1.0
        h = state_usage_entropy(pi)
        assert h.item() < 1e-4

    def test_2d_input_accepted(self) -> None:
        # T, n_states (single example)
        T, n_q = 5, 3
        pi = torch.full((T, n_q), 1.0 / n_q)
        h = state_usage_entropy(pi)
        assert abs(h.item() - math.log(n_q)) < 1e-5

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="2D or 3D"):
            state_usage_entropy(torch.zeros(4))


class TestDeterminismPenalty:
    def test_one_hot_zero(self) -> None:
        n_q, n_a = 4, 3
        logits = torch.zeros(n_q, n_a, n_q)
        logits[:, :, 0] = 100.0
        pen = determinism_penalty(logits)
        assert pen.item() < 1e-4

    def test_uniform_bounded(self) -> None:
        n_q, n_a = 4, 3
        logits = torch.zeros(n_q, n_a, n_q)
        pen = determinism_penalty(logits)
        # softmax of zeros is uniform 1/n_q; max is 1/n_q; penalty = 1 - 1/n_q
        assert abs(pen.item() - (1.0 - 1.0 / n_q)) < 1e-6

    def test_gradient_flows(self) -> None:
        logits = torch.randn(3, 3, 3, requires_grad=True)
        pen = determinism_penalty(logits)
        pen.backward()
        assert logits.grad is not None
