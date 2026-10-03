"""Pushdown automaton adapter wrapping `automata.pda.dpda.DPDA`.

Half A scope: the discrete DPDA is the source of truth; this module
exposes `accept`, `accept_with_trace`, and a `fold` runtime that runs
the PDA stepwise and folds an accumulator over the trace. The
bounded-stack tensor view and the soft forward algorithm are deferred
to Half B (see plan Section 2 and Section 1.2 of
`.claude/plans/2026-05-12_nsa-refactor-and-e2e-scaffold.md`).

Notation (see Sipser, Definition 2.13):
    Q              -- state set, |Q| = n_states
    Sigma          -- input alphabet, |Sigma| = n_alphabet
    Gamma          -- stack alphabet, |Gamma| = n_stack
    delta          -- transition function
                       Q x (Sigma cup {eps}) x Gamma -> Q x Gamma*
    q0             -- initial state
    Z0             -- initial stack symbol
    F              -- accepting state set

automata-lib conventions:
  - Transitions: `{q: {a: {gamma: (q_next, push_tuple)}}}`. `push_tuple`
    is **top-down**: its first element becomes the new top of the
    stack. An empty `push_tuple` `()` (or `''`) pops the current top
    without pushing.
  - Stack contents (e.g. `config.stack`, the value returned by
    `accept_with_trace` and observed by `fold`): **bottom-up**. The
    bottom of the stack is `stack[0]`; the top of the stack is
    `stack[-1]`. This is the opposite ordering from a push_tuple.
"""

from typing import Callable, Hashable, Iterable

from automata.pda.dpda import DPDA

from nsa.alphabet import Alphabet, IndexedSet


class PushdownAutomaton:
    """Adapter wrapping `automata.pda.dpda.DPDA`.

    All textbook operations (determinism enforcement, stepwise
    execution, acceptance) are delegated to `automata-lib`. The added
    surface is the `fold` runtime, which makes it easy to compute a
    downstream value (e.g. the evaluated arithmetic expression) from
    the PDA's run trace.

    Attributes:
        dpda: The wrapped `automata.pda.dpda.DPDA` (source of truth).
        alphabet: `Alphabet` over input symbols, stable order.
        stack_alphabet: `Alphabet` over stack symbols, stable order.
            By convention, `Z0` (initial stack symbol) is at index 0.
        states: `IndexedSet` over the states; initial state at index 0.

    Notation map:
        delta : Q x (Sigma cup {eps}) x Gamma -> Q x Gamma*
    """

    def __init__(
        self,
        dpda: DPDA,
        alphabet: Alphabet | None = None,
        stack_alphabet: Alphabet | None = None,
        states: IndexedSet | None = None,
    ) -> None:
        """Wrap a DPDA. Alphabet / stack / state orderings are derived if absent.

        Args:
            dpda: The DPDA to wrap (source of truth).
            alphabet: Optional explicit input-alphabet ordering. If
                absent, built from `sorted(dpda.input_symbols, key=repr)`.
            stack_alphabet: Optional explicit stack-alphabet ordering.
                If absent, built with `dpda.initial_stack_symbol` at
                index 0, followed by the remaining stack symbols
                sorted by repr.
            states: Optional explicit state ordering. If absent, the
                initial state is at index 0 followed by the rest
                sorted by repr.

        Raises:
            ValueError: If a provided ordering does not cover the
                exact set in `dpda`.
        """
        self.dpda: DPDA = dpda

        if alphabet is None:
            alphabet = Alphabet(sorted(dpda.input_symbols, key=repr))
        else:
            if set(alphabet) != set(dpda.input_symbols):
                raise ValueError(
                    "alphabet does not match dpda.input_symbols: "
                    f"{set(alphabet) ^ set(dpda.input_symbols)}"
                )
        self.alphabet: Alphabet = alphabet

        if stack_alphabet is None:
            other = sorted(
                (s for s in dpda.stack_symbols if s != dpda.initial_stack_symbol),
                key=repr,
            )
            stack_alphabet = Alphabet([dpda.initial_stack_symbol, *other])
        else:
            if set(stack_alphabet) != set(dpda.stack_symbols):
                raise ValueError(
                    "stack_alphabet does not match dpda.stack_symbols: "
                    f"{set(stack_alphabet) ^ set(dpda.stack_symbols)}"
                )
            if stack_alphabet.to_index(dpda.initial_stack_symbol) != 0:
                raise ValueError("initial stack symbol must have index 0")
        self.stack_alphabet: Alphabet = stack_alphabet

        if states is None:
            other_states = sorted(
                (s for s in dpda.states if s != dpda.initial_state), key=repr
            )
            states = IndexedSet([dpda.initial_state, *other_states])
        else:
            if set(states) != set(dpda.states):
                raise ValueError(
                    "states does not match dpda.states: "
                    f"{set(states) ^ set(dpda.states)}"
                )
            if states.to_index(dpda.initial_state) != 0:
                raise ValueError("initial state must have index 0")
        self.states: IndexedSet = states

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_alphabet(self) -> int:
        return len(self.alphabet)

    @property
    def q0(self) -> Hashable:
        """Name of the initial state (index 0)."""
        return self.states.to_symbol(0)

    @property
    def Z0(self) -> Hashable:
        """Initial stack symbol (the bottom marker, at index 0)."""
        return self.stack_alphabet.to_symbol(0)

    @property
    def F(self) -> set[Hashable]:
        """Set of accepting state names."""
        return set(self.dpda.final_states)

    @property
    def n_stack(self) -> int:
        return len(self.stack_alphabet)

    @property
    def q0_index(self) -> int:
        return 0

    @property
    def Z0_index(self) -> int:
        return 0

    @property
    def F_indices(self) -> list[int]:
        return sorted(self.states.to_index(q) for q in self.dpda.final_states)

    @property
    def acceptance_mode(self) -> str:
        return self.dpda.acceptance_mode

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dpda(cls, dpda: DPDA) -> "PushdownAutomaton":
        return cls(dpda)

    # ------------------------------------------------------------------ #
    # Discrete execution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_string_or_list(symbols: list[Hashable]) -> str | list:
        """Encode a symbol list as a str if every symbol is a 1-char str,
        else return the list unchanged. automata-lib accepts both forms.
        """
        if all(isinstance(s, str) and len(s) == 1 for s in symbols):
            return "".join(symbols)
        return symbols

    def accept(self, symbols: Iterable[Hashable]) -> bool:
        """Discrete acceptance: delegate to the underlying DPDA.

        Args:
            symbols: Iterable of input symbols; every symbol must be in
                `self.alphabet`.

        Returns:
            True if the DPDA accepts the input under its
            `acceptance_mode`.
        """
        symbols_list = list(symbols)
        for s in symbols_list:
            if s not in self.alphabet:
                raise ValueError(
                    f"symbol {s!r} not in alphabet {set(self.alphabet)!r}"
                )
        return bool(self.dpda.accepts_input(self._to_string_or_list(symbols_list)))

    def accept_with_trace(
        self, symbols: Iterable[Hashable]
    ) -> tuple[bool, list]:
        """Discrete acceptance plus the final stack from the run.

        Useful for tasks where the stack records information (e.g. the
        paper's NSPDA uses the stack to record the arithmetic
        expression for downstream evaluation).

        Args:
            symbols: Input symbol stream.

        Returns:
            Tuple `(accepted, final_stack)` where `final_stack` is a
            list of stack symbols **bottom-up** (the top of the stack
            is `final_stack[-1]`; the bottom is `final_stack[0]`).
            When the input is rejected, `accepted` is False and
            `final_stack` is the stack at the point of rejection (the
            last successful configuration).
        """
        symbols_list = list(symbols)
        for s in symbols_list:
            if s not in self.alphabet:
                raise ValueError(
                    f"symbol {s!r} not in alphabet {set(self.alphabet)!r}"
                )
        encoded = self._to_string_or_list(symbols_list)
        # `read_input_stepwise` raises `RejectionException` if the
        #   input is not accepted. Catch that and return the last
        #   successful configuration's stack.
        from automata.base.exceptions import RejectionException

        configs = []
        try:
            for config in self.dpda.read_input_stepwise(encoded):
                configs.append(config)
            accepted = True
        except RejectionException:
            accepted = False

        if not configs:
            return accepted, []

        final_stack = list(configs[-1].stack)
        return accepted, final_stack

    def fold(
        self,
        symbols: Iterable[Hashable],
        fold_fn: Callable[[object, Hashable, object, object, list], object],
        initial: object,
    ) -> tuple[bool, object]:
        """Run the PDA stepwise and fold an accumulator over the trace.

        At each step the user-provided `fold_fn` receives:
        `(accumulator, symbol_consumed, state_before, state_after,
        stack_after)` and returns the next accumulator value.
        `stack_after` is a list of stack symbols **bottom-up** (the
        top of the stack is the last element; see this module's
        docstring for the convention rationale).

        Args:
            symbols: Input symbols.
            fold_fn: Step function.
            initial: Initial accumulator value.

        Returns:
            Tuple `(accepted, final_accumulator)`.
        """
        from automata.base.exceptions import RejectionException

        symbols_list = list(symbols)
        for s in symbols_list:
            if s not in self.alphabet:
                raise ValueError(
                    f"symbol {s!r} not in alphabet {set(self.alphabet)!r}"
                )
        encoded = self._to_string_or_list(symbols_list)

        acc = initial
        configs = []
        try:
            for config in self.dpda.read_input_stepwise(encoded):
                configs.append(config)
            accepted = True
        except RejectionException:
            accepted = False

        # configs[0] is the initial configuration before reading any
        #   input; subsequent configs correspond to having consumed
        #   one more symbol each.
        for i in range(1, len(configs)):
            prev = configs[i - 1]
            cur = configs[i]
            # The symbol consumed at this step is the first character of
            #   prev.remaining_input not present in cur.remaining_input.
            #   For non-epsilon transitions, len(cur.remaining_input)
            #   == len(prev.remaining_input) - 1; for epsilon
            #   transitions they are equal -- we report `None` for the
            #   consumed symbol in that case.
            if len(cur.remaining_input) < len(prev.remaining_input):
                symbol = prev.remaining_input[0]
            else:
                symbol = None
            acc = fold_fn(
                acc, symbol, prev.state, cur.state, list(cur.stack)
            )
        return accepted, acc

    # ------------------------------------------------------------------ #
    # Soft execution -- DEFERRED to Half B (plan Section 2)
    # ------------------------------------------------------------------ #

    def accept_soft(self, *args, **kwargs):  # noqa: D401
        """Soft acceptance via bounded-stack forward algorithm.

        Not implemented in Half A. The plan deliberately defers the
        bounded-stack tensor view and the corresponding forward
        algorithm to Half B
        (`.claude/plans/2026-05-12_nsa-refactor-and-e2e-scaffold.md`
        Section 2.1). The discrete `accept`, `accept_with_trace`, and
        `fold` paths cover Half A's deliverable (the experiments now
        run through a proper PDA object).
        """
        raise NotImplementedError(
            "PushdownAutomaton.accept_soft is deferred to Half B; use "
            "accept, accept_with_trace, or fold for discrete inference."
        )

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"PushdownAutomaton(n_states={self.n_states}, "
            f"n_alphabet={self.n_alphabet}, n_stack={self.n_stack}, "
            f"acceptance_mode={self.acceptance_mode!r}, "
            f"F={self.F_indices})"
        )
