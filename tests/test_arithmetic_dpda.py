"""Tests for the arithmetic DPDA used by the simple_math_vlm_comp example.

Verifies the DPDA correctly accepts valid alternating-arithmetic
expressions and rejects malformed ones. This is the discrete spine of
the paper's NSPDA in Table 3.
"""

import pytest

from examples.simple_math_vlm_comp.arithmetic_dpda import (
    END_MARKER,
    build_arithmetic_dpda,
)


@pytest.fixture
def arithmetic_pda():
    return build_arithmetic_dpda()


class TestAccept:
    @pytest.mark.parametrize(
        "expression",
        [
            "5+3",
            "12*34",
            "0-0",
            "999%1",
            "1+2-3*4%5",
            "5+10+100",
            "0",
            "9876543210",
        ],
    )
    def test_accepts_valid(
        self, arithmetic_pda, expression: str
    ) -> None:
        symbols = list(expression) + [END_MARKER]
        assert arithmetic_pda.accept(symbols), (
            f"expected to accept {expression}#"
        )

    @pytest.mark.parametrize(
        "expression",
        [
            "+5",          # leading operator
            "5+",          # trailing operator
            "5++3",        # adjacent operators
            "5 3",         # only valid symbols allowed; space rejected upstream
            "",            # empty
            "+",           # operator only
        ],
    )
    def test_rejects_invalid(
        self, arithmetic_pda, expression: str
    ) -> None:
        # rejected by automata-lib OR by the alphabet check
        if any(c not in arithmetic_pda.alphabet for c in expression):
            with pytest.raises(ValueError):
                arithmetic_pda.accept(list(expression) + [END_MARKER])
        else:
            symbols = list(expression) + [END_MARKER]
            assert not arithmetic_pda.accept(symbols), (
                f"expected to reject {expression}#"
            )

    def test_rejects_no_end_marker(self, arithmetic_pda) -> None:
        # without '#', expression is rejected (not in final state)
        assert not arithmetic_pda.accept(list("5+3"))


class TestAlphabet:
    def test_alphabet_includes_end_marker(self, arithmetic_pda) -> None:
        assert END_MARKER in arithmetic_pda.alphabet
        # perception covers indices 0..13; end marker is the last
        assert arithmetic_pda.alphabet.to_index(END_MARKER) == 14

    def test_alphabet_size_matches_paper(self, arithmetic_pda) -> None:
        # 10 digits + 4 operators + 1 end marker
        assert arithmetic_pda.n_alphabet == 15
