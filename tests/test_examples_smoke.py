"""Smoke tests: rewritten examples import and construct their NSFA/NSPDA.

These tests verify that the example modules wire up correctly --
DPDA construction succeeds, alphabet alignment is right, NSFA /
NSPDA composition does not raise. They do not run the full
experiments (which need the data and the published CNN
checkpoints); those run via the example scripts directly.

The checkpoint-dependent paths are skipped when the `.pth` files are
absent so the test suite stays green in clean checkouts.
"""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
REGEX_CHECKPOINT = (
    REPO_ROOT / "examples" / "regex" / "models" / "model.pth"
)
HDO_CHECKPOINT = (
    REPO_ROOT
    / "examples"
    / "simple_math_vlm_comp"
    / "models"
    / "torch"
    / "HDO.pth"
)


class TestArithmeticDPDA:
    """The arithmetic DPDA does not depend on any checkpoint."""

    def test_construction(self) -> None:
        from examples.simple_math_vlm_comp.arithmetic_dpda import (
            build_arithmetic_dpda,
        )

        pda = build_arithmetic_dpda()
        # smoke: a known-valid expression is accepted
        assert pda.accept("5+3*2#")
        # smoke: a malformed expression is rejected
        assert not pda.accept("+5#")


class TestRegexExampleImport:
    """Imports the rewritten regex example; loads the CNN if available."""

    def test_imports(self) -> None:
        # importing the module should not error
        from examples.regex import regex as regex_module  # noqa: F401

    @pytest.mark.skipif(
        not REGEX_CHECKPOINT.exists(),
        reason=f"CNN checkpoint missing: {REGEX_CHECKPOINT}",
    )
    def test_nsfa_construction_with_checkpoint(self) -> None:
        from examples.regex.regex import _build_nsfa

        # smoke regex from the paper's Table 1
        nsfa = _build_nsfa("[a-c][a-c]")
        # the NSFA's FA should be small after minify
        assert nsfa.fa.n_states <= 5


class TestSimpleMathExampleImport:
    """Imports the rewritten arithmetic example; loads CNN if available."""

    def test_imports(self) -> None:
        from examples.simple_math_vlm_comp import simple_math_vlm_comp  # noqa: F401

    @pytest.mark.skipif(
        not HDO_CHECKPOINT.exists(),
        reason=f"HDO checkpoint missing: {HDO_CHECKPOINT}",
    )
    def test_nspda_construction_with_checkpoint(self) -> None:
        from examples.simple_math_vlm_comp.simple_math_vlm_comp import (
            _arithmetic_nspda,
        )

        nspda = _arithmetic_nspda()
        # 4 states (q0, q1, q2, qF), 15 input symbols, 14 perception
        assert nspda.pda.n_states == 4
        assert nspda.pda.n_alphabet == 15
        assert nspda.perception.n_alphabet == 14
        assert nspda.end_marker == "#"
