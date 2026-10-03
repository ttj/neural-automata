"""Finite automaton adapter wrapping `automata.fa.dfa.DFA`.

This module owns the bridge between (a) the discrete symbolic DFA from
`automata-lib`, which is the source of truth, and (b) a `torch.Tensor`
view of the transition function used by the soft / forward-algorithm
runtime in `nsa.runtime.soft`.

Notation (see `.claude/resources/notation-conventions.md`):
    Q              -- state set, |Q| = n_states
    Sigma          -- input alphabet, |Sigma| = n_alphabet
    delta          -- transition function Q x Sigma -> Q
    q0             -- initial state
    F              -- accepting state set, F subset of Q
    pi_t           -- state belief at time t, shape (n_states,) in Delta(Q)
    p_t            -- per-step symbol distribution, shape (n_alphabet,)
    delta_index    -- LongTensor(n_states, n_alphabet) of next-state indices
    delta_logits   -- Tensor(n_states, n_alphabet, n_states) of pre-softmax
                       transition weights, used during end-to-end training

Math:
    pi_{t+1}(q') = sum_q sum_a pi_t(q) p_t(a) delta_theta(q, a)(q')
    P_acc        = sum_{q in F} pi_T(q)

For total DFAs, the discrete view `delta_index` is a permutation tensor
that this module converts to a one-hot `(n_states, n_alphabet, n_states)`
mask for the soft runtime. For partial DFAs (when `allow_partial=True`
on the underlying `automata-lib` DFA), missing transitions are routed to
an implicit sink state appended at index `n_states` (i.e. the soft
tensor has shape `(n_states + 1, n_alphabet, n_states + 1)`).
"""

from typing import Hashable, Iterable, Sequence

import torch
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA

from nsa.alphabet import Alphabet, IndexedSet


SINK_LABEL = "__sink__"


class FiniteAutomaton:
    """Adapter wrapping `automata.fa.dfa.DFA` with a tensor view of `delta`.

    All textbook operations -- minimization, NFA-to-DFA, complement,
    intersection, union -- are delegated to `automata-lib`. The novel
    methods are `to_logits`, `accept_soft`, and `step_soft`, which
    expose the differentiable view used by `nsa.nsfa.NSFA` and the
    soft runtime.

    Attributes:
        dfa: The wrapped `automata.fa.dfa.DFA` (source of truth).
        alphabet: `Alphabet` over the input symbols of `dfa`, in a
            stable order.
        states: `IndexedSet` over the states of `dfa`, in a stable
            order. Index 0 is the initial state by convention.
        has_sink: Whether an implicit sink state is appended to handle
            partial transitions. When True, the tensor view has
            `n_states + 1` rows.

    Notation map:
        delta : Q x Sigma -> Q  (or Q + {sink} for partial DFAs)
    """

    def __init__(
        self,
        dfa: DFA,
        alphabet: Alphabet | None = None,
        states: IndexedSet | None = None,
    ) -> None:
        """Wrap a DFA. State and alphabet orderings are derived if absent.

        Args:
            dfa: The discrete DFA to wrap.
            alphabet: Optional explicit alphabet ordering. If absent, the
                alphabet is built from `sorted(dfa.input_symbols)`.
            states: Optional explicit state ordering. If absent, the
                state set is ordered with the initial state at index 0
                followed by the remaining states sorted by their repr.

        Raises:
            ValueError: If a provided `alphabet` or `states` does not
                match the symbols / states of `dfa`.
        """
        self.dfa: DFA = dfa
        self.has_sink: bool = dfa.allow_partial

        if alphabet is None:
            alphabet = Alphabet(sorted(dfa.input_symbols, key=repr))
        else:
            if set(alphabet) != set(dfa.input_symbols):
                raise ValueError(
                    "alphabet does not match dfa.input_symbols: "
                    f"{set(alphabet) ^ set(dfa.input_symbols)}"
                )
        self.alphabet: Alphabet = alphabet

        if states is None:
            other = sorted(
                (s for s in dfa.states if s != dfa.initial_state),
                key=repr,
            )
            states_items: list[Hashable] = [dfa.initial_state, *other]
            if self.has_sink:
                states_items.append(SINK_LABEL)
            states = IndexedSet(states_items)
        else:
            expected = set(dfa.states)
            if self.has_sink:
                expected = expected | {SINK_LABEL}
            if set(states) != expected:
                raise ValueError(
                    "states does not match dfa.states: "
                    f"{set(states) ^ expected}"
                )
            if states.to_index(dfa.initial_state) != 0:
                raise ValueError("initial state must have index 0")
        self.states: IndexedSet = states

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def n_states(self) -> int:
        """Number of states in the tensor view, including the sink if any."""
        return len(self.states)

    @property
    def n_alphabet(self) -> int:
        return len(self.alphabet)

    @property
    def q0_index(self) -> int:
        return 0

    @property
    def q0(self) -> Hashable:
        """Name of the initial state (index 0)."""
        return self.states.to_symbol(self.q0_index)

    @property
    def F_indices(self) -> list[int]:
        return sorted(self.states.to_index(q) for q in self.dfa.final_states)

    @property
    def F(self) -> set[Hashable]:
        """Set of accepting state names."""
        return set(self.dfa.final_states)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dfa(cls, dfa: DFA) -> "FiniteAutomaton":
        """Wrap an existing `automata-lib` DFA without modification."""
        return cls(dfa)

    @classmethod
    def from_regex(
        cls,
        pattern: str,
        input_symbols: Iterable[str] | None = None,
        minify: bool = True,
    ) -> "FiniteAutomaton":
        """Compile a regex to a DFA via Thompson construction.

        Uses `NFA.from_regex` then `DFA.from_nfa`. The regex syntax
        supported is that of `automata-lib`: `*`, `+`, character classes
        like `[a-z]`, ranges, alternation `|`, and grouping. No
        back-references or lookaheads (gracefully unsupported).

        Args:
            pattern: Regular expression.
            input_symbols: Explicit alphabet; if absent, inferred from
                the regex. Pass an explicit set for the EMNIST 26-letter
                experiment to ensure all letters appear in the
                transition table even when not used by the pattern.
            minify: Whether to minify the resulting DFA via Hopcroft.

        Returns:
            A new FiniteAutomaton wrapping the compiled (and optionally
            minified) DFA.
        """
        input_symbols_set = set(input_symbols) if input_symbols is not None else None
        nfa = NFA.from_regex(pattern, input_symbols=input_symbols_set)
        dfa = DFA.from_nfa(nfa, minify=minify)
        return cls(dfa)

    @classmethod
    def from_tensor(
        cls,
        delta_index: torch.Tensor,
        alphabet: Alphabet,
        q0_index: int,
        F_indices: Sequence[int],
        state_names: Sequence[Hashable] | None = None,
        minify: bool = True,
    ) -> "FiniteAutomaton":
        """Build a FiniteAutomaton from a tensor representation of `delta`.

        Used to project a discretized `delta_theta` back into the
        symbolic library object after training, so the result can be
        minified and compared to a hand-built reference up to
        isomorphism.

        Args:
            delta_index: LongTensor of shape (n_states, n_alphabet)
                whose entries are next-state indices. A sentinel value
                of `n_states` denotes routing to the implicit sink (the
                resulting DFA is partial).
            alphabet: The alphabet whose ordering matches the columns
                of `delta_index`.
            q0_index: Index of the initial state.
            F_indices: Indices of accepting states.
            state_names: Optional names for the n_states states; if
                absent, states are named `q0`, `q1`, ... by index.
            minify: Whether to run `DFA.minify` after construction.
        """
        if delta_index.dim() != 2:
            raise ValueError(
                f"delta_index must be 2D, got shape {tuple(delta_index.shape)}"
            )
        n_states, n_alphabet = delta_index.shape
        if n_alphabet != len(alphabet):
            raise ValueError(
                f"delta_index columns {n_alphabet} != |alphabet| {len(alphabet)}"
            )

        if state_names is None:
            state_names = [f"q{i}" for i in range(n_states)]
        elif len(state_names) != n_states:
            raise ValueError(
                f"len(state_names) {len(state_names)} != n_states {n_states}"
            )

        # detect partial DFA: any entry == n_states is the sink sentinel
        sink_sentinel = n_states
        is_partial = bool((delta_index == sink_sentinel).any().item())

        transitions: dict = {}
        for i in range(n_states):
            source = state_names[i]
            transitions[source] = {}
            for a_idx in range(n_alphabet):
                target_idx = int(delta_index[i, a_idx].item())
                if target_idx == sink_sentinel:
                    if is_partial:
                        continue  # omit transition => partial DFA
                    raise ValueError(
                        f"sentinel sink at ({i}, {a_idx}) but no sink expected"
                    )
                transitions[source][alphabet.to_symbol(a_idx)] = state_names[
                    target_idx
                ]

        dfa = DFA(
            states={state_names[i] for i in range(n_states)},
            input_symbols=set(alphabet),
            transitions=transitions,
            initial_state=state_names[q0_index],
            final_states={state_names[i] for i in F_indices},
            allow_partial=is_partial,
        )
        if minify:
            dfa = dfa.minify()
        return cls(dfa)

    # ------------------------------------------------------------------ #
    # Tensor view
    # ------------------------------------------------------------------ #

    def to_index_tensor(self) -> torch.Tensor:
        """Return `delta_index` LongTensor of shape (n_states, n_alphabet).

        Entry `[i, j]` is the index of `delta(states[i], alphabet[j])`.
        For partial DFAs, missing transitions point to the sink index
        `n_states - 1` (the last row of the tensor; `has_sink == True`).
        """
        n_q = self.n_states
        n_a = self.n_alphabet
        delta_index = torch.zeros(n_q, n_a, dtype=torch.long)
        sink_idx = n_q - 1 if self.has_sink else None

        for i in range(n_q):
            source = self.states.to_symbol(i)
            if source == SINK_LABEL:
                # sink loops to itself on every symbol
                delta_index[i, :] = i
                continue
            source_transitions = self.dfa.transitions.get(source, {})
            for j in range(n_a):
                symbol = self.alphabet.to_symbol(j)
                if symbol in source_transitions:
                    target = source_transitions[symbol]
                    delta_index[i, j] = self.states.to_index(target)
                else:
                    if sink_idx is None:
                        raise RuntimeError(
                            f"missing transition delta({source!r}, "
                            f"{symbol!r}) but no sink available; "
                            "DFA was constructed without allow_partial"
                        )
                    delta_index[i, j] = sink_idx
        return delta_index

    def to_one_hot(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Return the soft transition tensor in one-hot form.

        Shape: (n_states, n_alphabet, n_states). For each (q, a) the
        row over q' is a one-hot at `delta(q, a)`. Used as a fixed
        (non-learnable) transition for regime (a) of the training
        plan, where only perception is trained.
        """
        delta_index = self.to_index_tensor()
        n_q = self.n_states
        n_a = self.n_alphabet
        one_hot = torch.zeros(n_q, n_a, n_q, device=device)
        rows = torch.arange(n_q).view(-1, 1).expand(-1, n_a)
        cols = torch.arange(n_a).view(1, -1).expand(n_q, -1)
        one_hot[rows, cols, delta_index] = 1.0
        return one_hot

    def to_logits(
        self,
        init: str = "from_dfa",
        scale: float = 5.0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return pre-softmax logits for `delta_theta`, shape (Q, Sigma, Q).

        Args:
            init: One of:
                - "from_dfa": logits concentrated on the discrete
                  transitions of the wrapped DFA (warm start; softmax
                  recovers the one-hot view at scale -> infinity).
                - "identity": logits favor self-loops at every (q, a)
                  (Niepert-style identity init); useful when the
                  discrete DFA is to be discarded.
                - "uniform": zero logits, i.e. uniform softmax. Useful
                  as a from-scratch start.
            scale: Magnitude of the concentration for "from_dfa" and
                "identity". Higher = closer to one-hot; lower = more
                exploration.
            device: Optional torch device.

        Returns:
            Tensor of shape (n_states, n_alphabet, n_states) suitable
            for use as a learnable `nn.Parameter`.
        """
        n_q = self.n_states
        n_a = self.n_alphabet
        if init == "uniform":
            return torch.zeros(n_q, n_a, n_q, device=device)
        if init == "identity":
            logits = torch.zeros(n_q, n_a, n_q, device=device)
            diag = torch.arange(n_q)
            logits[diag, :, diag] = scale
            return logits
        if init == "from_dfa":
            return scale * self.to_one_hot(device=device)
        raise ValueError(f"unknown init mode: {init!r}")

    # ------------------------------------------------------------------ #
    # Discrete execution
    # ------------------------------------------------------------------ #

    def step(
        self, current_state: Hashable, symbol: Hashable
    ) -> Hashable:
        """Discrete one-symbol step: return `delta(current_state, symbol)`.

        Mirrors the semantics of `step_soft` but for the discrete DFA.
        Useful for stepwise execution outside of `accept()` -- e.g.
        when an external loop wants to observe state and update an
        accumulator at each input.

        Args:
            current_state: A state name in `self.dfa.states`.
            symbol: A symbol in `self.alphabet`.

        Returns:
            The next state name. For partial DFAs, missing transitions
            return `SINK_LABEL`.

        Raises:
            ValueError: If `current_state` or `symbol` is not in the
                FA's state set or alphabet.
            RuntimeError: If `delta(current_state, symbol)` is
                undefined and the FA is not partial.
        """
        if symbol not in self.alphabet:
            raise ValueError(
                f"symbol {symbol!r} not in alphabet "
                f"{set(self.alphabet)!r}"
            )
        if current_state == SINK_LABEL and self.has_sink:
            return SINK_LABEL
        if current_state not in self.dfa.states:
            raise ValueError(
                f"state {current_state!r} not in dfa.states "
                f"{set(self.dfa.states)!r}"
            )
        source_transitions = self.dfa.transitions.get(current_state, {})
        if symbol in source_transitions:
            return source_transitions[symbol]
        if self.has_sink:
            return SINK_LABEL
        raise RuntimeError(
            f"no transition delta({current_state!r}, {symbol!r}); "
            "DFA was constructed without allow_partial"
        )

    def accept(self, symbols: Iterable[Hashable]) -> bool:
        """Discrete acceptance: delegate to the underlying DFA.

        Args:
            symbols: Iterable of input symbols (must all be in
                `self.alphabet`).

        Returns:
            True if the underlying DFA accepts the input string.
        """
        # materialize once so we don't exhaust a generator before the
        #   alphabet check, and so we can decide on str vs list format
        symbols_list = list(symbols)
        for s in symbols_list:
            if s not in self.alphabet:
                raise ValueError(
                    f"symbol {s!r} not in alphabet {set(self.alphabet)!r}"
                )
        if all(isinstance(s, str) and len(s) == 1 for s in symbols_list):
            return bool(self.dfa.accepts_input("".join(symbols_list)))
        return bool(self.dfa.accepts_input(symbols_list))

    # ------------------------------------------------------------------ #
    # Soft execution (forward algorithm in log-semiring)
    # ------------------------------------------------------------------ #

    def initial_belief(
        self,
        device: torch.device | str | None = None,
        log_space: bool = False,
    ) -> torch.Tensor:
        """One-hot state belief at the initial state.

        Args:
            device: Torch device for the resulting tensor.
            log_space: If True, return log probabilities (0 at q0, -inf
                elsewhere). If False, return linear probabilities.

        Returns:
            Tensor of shape (n_states,).
        """
        if log_space:
            pi = torch.full(
                (self.n_states,), float("-inf"), device=device
            )
            pi[self.q0_index] = 0.0
            return pi
        pi = torch.zeros(self.n_states, device=device)
        pi[self.q0_index] = 1.0
        return pi

    def F_mask(
        self,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Boolean mask over states selecting accepting states `F`."""
        mask = torch.zeros(self.n_states, dtype=torch.bool, device=device)
        for q_idx in self.F_indices:
            mask[q_idx] = True
        return mask

    def step_soft(
        self,
        pi: torch.Tensor,
        symbol_probs: torch.Tensor,
        delta_logits: torch.Tensor | None = None,
        log_space: bool = False,
    ) -> torch.Tensor:
        """One step of the forward algorithm.

        Implements Eq. (forward) of the design plan
        `.claude/plans/2026-05-12_nsa-refactor-and-e2e-scaffold.md`,
        Section 2.1:

            pi_{t+1}(q') = sum_q sum_a pi_t(q) * p_t(a) * delta_theta(q, a)(q')

        Args:
            pi: State belief at time t. Shape `(n_states,)` if `log_space`
                is False, else log-probabilities of the same shape.
            symbol_probs: Per-step symbol distribution. Shape
                `(n_alphabet,)` if `log_space` is False, else log-probs.
            delta_logits: Pre-softmax transition logits, shape
                `(n_states, n_alphabet, n_states)`. If None, the
                discrete one-hot view from `to_one_hot()` is used (no
                gradients flow through delta).
            log_space: Whether `pi` and `symbol_probs` are in log space.

        Returns:
            Updated state belief at time t+1, same shape and space as
            `pi`.
        """
        if delta_logits is None:
            delta_prob = self.to_one_hot(device=pi.device)
        else:
            delta_prob = torch.softmax(delta_logits, dim=-1)

        if log_space:
            log_delta = torch.log(delta_prob.clamp_min(1e-30))
            # log_pi_next[q'] = logsumexp_{q,a} pi[q] + p[a] + log_delta[q,a,q']
            # broadcast: (Q,1,1) + (1,A,1) + (Q,A,Q) -> (Q,A,Q)
            combined = (
                pi.view(-1, 1, 1)
                + symbol_probs.view(1, -1, 1)
                + log_delta
            )
            return torch.logsumexp(combined.flatten(0, 1), dim=0)
        # linear semiring
        return torch.einsum("q,a,qan->n", pi, symbol_probs, delta_prob)

    def accept_soft(
        self,
        symbol_probs: torch.Tensor,
        delta_logits: torch.Tensor | None = None,
        log_space: bool = False,
    ) -> torch.Tensor:
        """Forward algorithm over a full sequence, returns acceptance prob.

        Args:
            symbol_probs: Per-step symbol distribution. Shape `(T, n_alphabet)`
                if `log_space` is False, else log-probs of the same
                shape. `T` is the sequence length.
            delta_logits: As for `step_soft`.
            log_space: Whether inputs/outputs are in log space.

        Returns:
            Scalar tensor: acceptance probability if `log_space` is
            False; log of acceptance probability otherwise.
        """
        if symbol_probs.dim() != 2:
            raise ValueError(
                f"symbol_probs must be 2D (T, n_alphabet); got "
                f"{tuple(symbol_probs.shape)}"
            )
        if symbol_probs.shape[1] != self.n_alphabet:
            raise ValueError(
                f"symbol_probs last dim {symbol_probs.shape[1]} != "
                f"|alphabet| {self.n_alphabet}"
            )
        T = symbol_probs.shape[0]
        pi = self.initial_belief(
            device=symbol_probs.device,
            log_space=log_space,
        )
        for t in range(T):
            pi = self.step_soft(
                pi,
                symbol_probs[t],
                delta_logits=delta_logits,
                log_space=log_space,
            )
        F_mask = self.F_mask(device=pi.device)
        if log_space:
            return torch.logsumexp(pi[F_mask], dim=0)
        return pi[F_mask].sum()

    # ------------------------------------------------------------------ #
    # Composition (delegates to automata-lib)
    # ------------------------------------------------------------------ #

    def union(self, other: "FiniteAutomaton") -> "FiniteAutomaton":
        """Language union; alphabets must match exactly."""
        if set(self.alphabet) != set(other.alphabet):
            raise ValueError("union requires matching alphabets")
        return FiniteAutomaton.from_dfa(self.dfa.union(other.dfa))

    def intersection(self, other: "FiniteAutomaton") -> "FiniteAutomaton":
        if set(self.alphabet) != set(other.alphabet):
            raise ValueError("intersection requires matching alphabets")
        return FiniteAutomaton.from_dfa(self.dfa.intersection(other.dfa))

    def complement(self) -> "FiniteAutomaton":
        return FiniteAutomaton.from_dfa(self.dfa.complement())

    def minify(self) -> "FiniteAutomaton":
        """Hopcroft-minified version. Rebuilds index maps deterministically."""
        return FiniteAutomaton.from_dfa(self.dfa.minify())

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"FiniteAutomaton(n_states={self.n_states}, "
            f"n_alphabet={self.n_alphabet}, "
            f"has_sink={self.has_sink}, F={self.F_indices})"
        )
