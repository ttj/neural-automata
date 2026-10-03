"""Longest-streak example: a pure symbolic FiniteAutomaton with no perception.

This is the smallest entry point in the project. It demonstrates:

  - constructing a `FiniteAutomaton` by hand (no regex compilation),
  - stepwise execution via `FiniteAutomaton.step`,
  - tracking an accumulator alongside the symbolic state.

There is no neural network: the input is a string of characters. Each
character is classified as `'d'` (digit) or `'l'` (letter) by a tiny
Python helper, then fed to the FA. The FA's state is "what we just
saw"; transitions reveal whether the current input *continues* the
prior streak or *starts a new one*. The longest streak across the
entire input is tracked as an external accumulator.

Run from the repository root:

    python -m examples.longest_streak.longest_streak

For a perception-driven example, see `examples/regex/regex.py`. For
the NSPDA arithmetic walkthrough, see `examples/simple_math/`.
"""

from automata.fa.dfa import DFA

from nsa import FiniteAutomaton
from nsa.alphabet import Alphabet


def build_class_fa() -> FiniteAutomaton:
    """A 2-state FA over the alphabet {'d', 'l'} (digit / letter).

    Both states are accepting -- the FA "accepts" any string. The
    purpose is to observe the state at each step, not to make an
    accept / reject decision.
    """
    alphabet = Alphabet(["d", "l"])
    dfa = DFA(
        states={"Digit", "Letter"},
        input_symbols={"d", "l"},
        transitions={
            "Digit": {"d": "Digit", "l": "Letter"},
            "Letter": {"d": "Digit", "l": "Letter"},
        },
        initial_state="Digit",
        final_states={"Digit", "Letter"},
    )
    return FiniteAutomaton(dfa, alphabet=alphabet)


def classify(c: str) -> str:
    """Map a single ASCII alphanumeric character to its FA-input symbol."""
    if c.isdigit():
        return "d"
    if c.isalpha():
        return "l"
    raise ValueError(
        f"input {c!r} is neither a digit nor a letter; skip "
        "non-alphanumeric inputs before calling classify"
    )


def longest_streak(text: str) -> tuple[int, list[tuple[str, str, int]]]:
    """Return the longest streak of consecutive digits-or-letters in `text`.

    Args:
        text: An input string. Non-alphanumeric characters are
            ignored.

    Returns:
        A tuple `(longest, trace)` where `longest` is the maximum
        same-class streak length seen, and `trace` is a list of
        `(input_char, fa_state_after, running_streak_length)` triples
        for every alphanumeric input -- useful for printing or
        teaching.
    """
    fa = build_class_fa()
    state = fa.q0
    prev_state: str | None = None
    current = 0
    longest = 0
    trace: list[tuple[str, str, int]] = []

    for c in text:
        if not c.isalnum():
            continue
        symbol = classify(c)
        state = fa.step(state, symbol)
        if state == prev_state:
            current += 1
        else:
            current = 1
            prev_state = state
        longest = max(longest, current)
        trace.append((c, state, current))

    return longest, trace


def main() -> None:
    """Print a worked example. Edit `examples` to try other inputs."""
    examples = [
        "abcd1234",
        "a1b2c3d4",
        "aaaa1111bbbb",
        "X9Y8Z7Q6",
        "hello123world456",
    ]

    fa = build_class_fa()
    print(f"FA states: {set(fa.dfa.states)}")
    print(f"FA alphabet: {set(fa.alphabet)}")
    print(f"FA initial state: {fa.q0}")
    print(f"FA accepting states: {fa.F}\n")

    for text in examples:
        longest, trace = longest_streak(text)
        print(f"input: {text!r}")
        for c, state, running in trace:
            print(f"  {c!r}  -> state={state:<6} streak={running}")
        print(f"  longest streak: {longest}\n")


if __name__ == "__main__":
    main()
