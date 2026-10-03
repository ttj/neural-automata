"""Shared pytest fixtures and configuration for the nsa test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Seed torch RNG before every test for determinism."""
    torch.manual_seed(0)


@pytest.fixture
def device() -> torch.device:
    """Default test device. CPU keeps CI / local runs identical."""
    return torch.device("cpu")
