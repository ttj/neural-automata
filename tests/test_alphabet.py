"""Tests for `nsa.alphabet`."""

import pytest

from nsa.alphabet import (
    Alphabet,
    IndexedSet,
    digits_and_operators_alphabet,
    lowercase_alphabet,
)


class TestIndexedSet:
    def test_roundtrip(self) -> None:
        s = IndexedSet(["a", "b", "c"])
        for i, item in enumerate(["a", "b", "c"]):
            assert s.to_index(item) == i
            assert s.to_symbol(i) == item

    def test_order_preserved(self) -> None:
        s = IndexedSet(["c", "a", "b"])
        assert s.to_index("c") == 0
        assert s.to_index("a") == 1
        assert s.to_index("b") == 2

    def test_length(self) -> None:
        assert len(IndexedSet([])) == 0
        assert len(IndexedSet(range(5))) == 5

    def test_iteration_preserves_order(self) -> None:
        items = ["x", "y", "z"]
        assert list(IndexedSet(items)) == items

    def test_contains(self) -> None:
        s = IndexedSet(["a", "b"])
        assert "a" in s
        assert "z" not in s

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            IndexedSet(["a", "b", "a"])

    def test_missing_symbol_raises(self) -> None:
        s = IndexedSet(["a", "b"])
        with pytest.raises(KeyError):
            s.to_index("z")

    def test_out_of_range_index_raises(self) -> None:
        s = IndexedSet(["a", "b"])
        with pytest.raises(IndexError):
            s.to_symbol(5)

    def test_equality(self) -> None:
        assert IndexedSet(["a", "b"]) == IndexedSet(["a", "b"])
        assert IndexedSet(["a", "b"]) != IndexedSet(["b", "a"])
        assert IndexedSet(["a", "b"]) != ["a", "b"]


class TestAlphabetHelpers:
    def test_lowercase_has_26_letters(self) -> None:
        a = lowercase_alphabet()
        assert len(a) == 26
        assert a.to_symbol(0) == "a"
        assert a.to_symbol(25) == "z"
        assert a.to_index("k") == 10

    def test_digits_and_operators_has_14_classes(self) -> None:
        a = digits_and_operators_alphabet()
        assert len(a) == 14
        assert a.to_index("0") == 0
        assert a.to_index("9") == 9
        assert a.to_index("+") == 10
        assert a.to_index("-") == 11
        assert a.to_index("*") == 12
        # '%' is the legacy division glyph used by the published CNN
        # checkpoint; see `digits_and_operators_alphabet` docstring.
        assert a.to_index("%") == 13

    def test_alphabet_is_indexed_set(self) -> None:
        assert Alphabet is IndexedSet
