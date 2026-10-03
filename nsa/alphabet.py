"""Indexed sets with a stable bidirectional name-to-index map.

Used for both the alphabet $\\Sigma$ of an automaton and for the state set
$Q$. A single primitive (`IndexedSet`) handles both; the type alias
`Alphabet` exists for readability at call sites.

Notation:
    Sigma     -- input alphabet of a FA / PDA, |Sigma| = n_alphabet
    Gamma     -- stack alphabet of a PDA, |Gamma| = n_stack
    Q         -- state set of a FA / PDA, |Q| = n_states

The bijection is stable: once constructed, `to_index(symbol)` and
`to_symbol(index)` round-trip. Reconstructing an `IndexedSet` from the
same iterable in the same order yields the same map; the map does *not*
silently reorder under any operation. Library-side renaming (e.g. after
`DFA.minify()`) requires a fresh `IndexedSet` built from the new state
names; the FA adapter rebuilds it deterministically.
"""

from typing import Hashable, Iterable, Iterator


class IndexedSet:
    """A finite ordered set of hashable items with a name<->index bijection.

    The ordering at construction defines the index assignment: the i-th
    item in `items` gets index `i`. Reorder by constructing a new
    `IndexedSet` explicitly; never mutate.

    Args:
        items: Iterable of distinct hashables. Order matters and is
            preserved.

    Raises:
        ValueError: If `items` contains duplicates.
    """

    def __init__(self, items: Iterable[Hashable]) -> None:
        items_tuple = tuple(items)
        if len(set(items_tuple)) != len(items_tuple):
            duplicates = [x for x in items_tuple if items_tuple.count(x) > 1]
            raise ValueError(
                f"IndexedSet items must be distinct; duplicates: "
                f"{sorted(set(duplicates), key=repr)}"
            )
        self._items: tuple[Hashable, ...] = items_tuple
        self._to_index: dict[Hashable, int] = {
            item: i for i, item in enumerate(items_tuple)
        }

    def to_index(self, item: Hashable) -> int:
        """Return the index of `item` in this set.

        Raises:
            KeyError: If `item` is not in the set.
        """
        return self._to_index[item]

    def to_symbol(self, index: int) -> Hashable:
        """Return the item at position `index`.

        Raises:
            IndexError: If `index` is out of range.
        """
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._items)

    def __contains__(self, item: object) -> bool:
        return item in self._to_index

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexedSet):
            return NotImplemented
        return self._items == other._items

    def __repr__(self) -> str:
        return f"IndexedSet({list(self._items)!r})"

    @property
    def items(self) -> tuple[Hashable, ...]:
        """Underlying ordered tuple of items (immutable)."""
        return self._items


# Domain-meaningful alias. At call sites, prefer `Alphabet(...)` when the
#   indexed set represents Sigma or Gamma, and prefer `IndexedSet(...)`
#   for state sets.
Alphabet = IndexedSet


def lowercase_alphabet() -> "Alphabet":
    """The 26 lowercase ASCII letters, indexed a=0 ... z=25.

    Matches the EMNIST class index used in the NeuS 2025 regex
    experiment (after the paper's `label - 1` shift).
    """
    return Alphabet(chr(ord("a") + i) for i in range(26))


def digits_and_operators_alphabet() -> "Alphabet":
    """The 14 classes used in the NeuS 2025 arithmetic-eval experiment.

    Indices follow the `OP_MAP` convention in
    `examples/simple_math_vlm_comp/simple_math_vlm_comp.py`: digits
    '0'-'9' at indices 0-9; '+' at 10, '-' at 11, '*' at 12, '%' at 13.
    '%' is the legacy division glyph used in the Heusser dataset's
    labels and in the published CNN checkpoint's training; this helper
    matches the checkpoint exactly so the refactored example can load
    `examples/simple_math_vlm_comp/models/torch/HDO.pth` without
    retraining.
    """
    digits = [str(i) for i in range(10)]
    operators = ["+", "-", "*", "%"]
    return Alphabet(digits + operators)
