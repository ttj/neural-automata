"""Tests for `nsa.learning.losses`.

Verify numerical-stability claims (log1mexp at boundary cases) and
agreement with PyTorch's reference BCE on well-conditioned inputs.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from nsa.learning.losses import log1mexp, sequence_acceptance_loss


class TestLog1mexp:
    def test_at_near_zero(self) -> None:
        # log(1 - exp(0)) = log(0) = -inf
        x = torch.tensor([-1e-7])
        result = log1mexp(x)
        assert result.item() < -10.0  # very negative

    def test_at_minus_log2(self) -> None:
        # x = -log(2): 1 - exp(-log 2) = 1 - 1/2 = 1/2; log(1/2) = -log 2
        x = torch.tensor([-math.log(2.0)])
        result = log1mexp(x)
        assert abs(result.item() - (-math.log(2.0))) < 1e-6

    def test_very_negative(self) -> None:
        # x = -100: 1 - exp(-100) ~ 1; log(1) ~ -1e-44 ~ 0
        x = torch.tensor([-100.0])
        result = log1mexp(x)
        assert abs(result.item()) < 1e-6

    def test_matches_naive_in_safe_range(self) -> None:
        # in [-50, -0.1], naive and stable formulae agree
        torch.manual_seed(0)
        x = -torch.rand(100) * 50.0 - 0.1
        stable = log1mexp(x)
        naive = torch.log1p(-torch.exp(x))
        assert torch.allclose(stable, naive, atol=1e-5)

    def test_gradient_flows(self) -> None:
        x = torch.tensor([-0.5], requires_grad=True)
        y = log1mexp(x).sum()
        y.backward()
        assert x.grad is not None
        # d/dx log(1 - exp(x)) = -exp(x) / (1 - exp(x))
        expected = -math.exp(-0.5) / (1.0 - math.exp(-0.5))
        assert abs(x.grad.item() - expected) < 1e-5


class TestSequenceAcceptanceLoss:
    def test_perfect_accept(self) -> None:
        # P_acc = 1 (log 0), y = 1 -> loss = 0
        log_p = torch.tensor([0.0])
        y = torch.tensor([1.0])
        loss = sequence_acceptance_loss(log_p, y)
        assert loss.item() < 1e-6

    def test_perfect_reject(self) -> None:
        # P_acc = 0 (log = -large), y = 0 -> loss = 0
        log_p = torch.tensor([-100.0])
        y = torch.tensor([0.0])
        loss = sequence_acceptance_loss(log_p, y)
        assert abs(loss.item()) < 1e-4

    def test_matches_torch_bce(self) -> None:
        torch.manual_seed(42)
        # generate random log-probs in [-5, -0.05]
        log_p = -(torch.rand(16) * 5.0 + 0.05)
        y = torch.randint(0, 2, (16,)).float()
        ours = sequence_acceptance_loss(log_p, y, reduction="none")
        p = torch.exp(log_p)
        ref = F.binary_cross_entropy(p, y, reduction="none")
        assert torch.allclose(ours, ref, atol=1e-5)

    def test_reduction_modes(self) -> None:
        torch.manual_seed(0)
        log_p = -torch.rand(8) - 0.1
        y = torch.randint(0, 2, (8,)).float()
        none = sequence_acceptance_loss(log_p, y, reduction="none")
        mean = sequence_acceptance_loss(log_p, y, reduction="mean")
        total = sequence_acceptance_loss(log_p, y, reduction="sum")
        assert torch.allclose(mean, none.mean())
        assert torch.allclose(total, none.sum())

    def test_reduction_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="reduction"):
            sequence_acceptance_loss(
                torch.tensor([-0.5]),
                torch.tensor([1.0]),
                reduction="median",
            )

    def test_gradient_flows(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(4, requires_grad=True)
        log_p = torch.log_softmax(
            torch.stack([logits, torch.zeros_like(logits)], dim=-1),
            dim=-1,
        )[..., 0]
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = sequence_acceptance_loss(log_p, y)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0
