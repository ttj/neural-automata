"""Neurosymbolic finite automaton: composition of FA + perception.

Implements Definition 3 of the NeuS 2025 paper as a composition of a
`FiniteAutomaton` adapter (the symbolic part, wrapping
`automata.fa.dfa.DFA`) and a `PerceptionAdapter` (one or more CNNs).

Two execution modes:

  - `accept(images)` -- discrete inference: per-image argmax over
    Sigma, then the discrete DFA decides accept/reject. Reproduces
    the runtime used to produce Table 1 in the paper.
  - `forward(images)` -- soft inference: per-image softmax over Sigma,
    forward algorithm through the soft transition tensor, returns
    acceptance probability. Differentiable; trains end-to-end.

A streaming API (`reset`, `observe`) maintains state-belief across
calls for runtime-monitor use cases (see Section 8 of the related-work
memo).

Notation:
    pi_t       -- state belief at time t, shape (n_states,) in Delta(Q)
    P_acc      -- acceptance probability, scalar in [0, 1]
"""

import torch
import torch.nn as nn

from nsa.fa import FiniteAutomaton
from nsa.perception import PerceptionAdapter


class NSFA(nn.Module):
    """Neurosymbolic finite automaton.

    Composes a `FiniteAutomaton` (symbolic, fixed at construction) with
    a `PerceptionAdapter` (one or more `nn.Module` perception networks)
    and optionally a learnable transition-logit tensor for end-to-end
    training of the automaton's structure.

    Attributes:
        fa: The symbolic finite automaton (adapter over `automata-lib`).
        perception: The perception adapter.
        delta_logits: Optional learnable `nn.Parameter` of shape
            `(n_states, n_alphabet, n_states)`. When set, the soft
            runtime uses `softmax(delta_logits, dim=-1)` as the
            transition. When None, the soft runtime uses the one-hot
            view of `fa.dfa` (regime (a) -- only perception is
            trainable).
    """

    def __init__(
        self,
        fa: FiniteAutomaton,
        perception: PerceptionAdapter,
        learn_delta: bool = False,
        delta_init: str = "from_dfa",
        delta_init_scale: float = 5.0,
    ) -> None:
        """Initialize an NSFA.

        Args:
            fa: The FiniteAutomaton adapter.
            perception: The PerceptionAdapter.
            learn_delta: If True, register `delta_logits` as a
                learnable `nn.Parameter` and use it in the soft
                runtime. If False, the soft runtime uses the discrete
                one-hot view of `fa.dfa` (regime (a) of the plan).
            delta_init: Initialization mode for `delta_logits`; see
                `FiniteAutomaton.to_logits`. Used only when
                `learn_delta` is True.
            delta_init_scale: Concentration scale for the
                initialization; see `FiniteAutomaton.to_logits`.

        Raises:
            ValueError: If `perception.n_alphabet` does not match
                `fa.n_alphabet`.
        """
        super().__init__()
        if perception.n_alphabet != fa.n_alphabet:
            raise ValueError(
                f"perception.n_alphabet {perception.n_alphabet} != "
                f"fa.n_alphabet {fa.n_alphabet}"
            )
        if not perception.shared and perception.n_states != fa.n_states:
            raise ValueError(
                f"perception.n_states {perception.n_states} != "
                f"fa.n_states {fa.n_states}"
            )
        self.fa = fa
        self.perception = perception
        if learn_delta:
            init_logits = fa.to_logits(
                init=delta_init, scale=delta_init_scale
            )
            self.delta_logits: nn.Parameter | None = nn.Parameter(init_logits)
        else:
            self.delta_logits = None

        # streaming state -- initialized lazily by `reset`
        self._streaming_pi: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Discrete inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def accept(self, images: torch.Tensor) -> tuple[bool, str]:
        """Discrete acceptance: argmax per image, then run the DFA.

        Reproduces the inference path used in
        `examples/regex/regex.py` of the original paper.

        Args:
            images: Tensor of shape `(T, C, H, W)`.

        Returns:
            Tuple `(accepted, predicted_string)` where `accepted` is the
            DFA's decision and `predicted_string` is the argmax sequence
            joined as a string of symbols (for diagnostics and parity
            with the paper's logging).
        """
        logits = self.perception(images)  # (T, n_alphabet)
        preds = torch.argmax(logits, dim=-1).tolist()
        symbols = [self.fa.alphabet.to_symbol(i) for i in preds]
        accepted = self.fa.accept(symbols)
        return accepted, "".join(str(s) for s in symbols)

    # ------------------------------------------------------------------ #
    # Soft / differentiable inference
    # ------------------------------------------------------------------ #

    def forward(
        self,
        images: torch.Tensor,
        log_space: bool = True,
    ) -> torch.Tensor:
        """Soft acceptance probability via the forward algorithm.

        Implements Eq. (forward) of the design plan Section 2.1. Use
        this method during training; gradients flow back through the
        perception network (always) and through `delta_logits` if
        `learn_delta=True`.

        Args:
            images: Tensor of shape `(T, C, H, W)`.
            log_space: Whether to run the forward algorithm in
                log-semiring (recommended for `T > ~10`). Returns
                `log P_acc` if True, else `P_acc`.

        Returns:
            Scalar tensor: acceptance probability (or its log).
        """
        logits = self.perception(images)  # (T, n_alphabet)
        if log_space:
            symbol_probs = torch.log_softmax(logits, dim=-1)
        else:
            symbol_probs = torch.softmax(logits, dim=-1)
        return self.fa.accept_soft(
            symbol_probs,
            delta_logits=self.delta_logits,
            log_space=log_space,
        )

    # ------------------------------------------------------------------ #
    # Streaming API (runtime-monitor use case)
    # ------------------------------------------------------------------ #

    def reset(self, device: torch.device | str | None = None) -> None:
        """Reset the streaming state belief to the initial state.

        Must be called before the first `observe` of a new input
        stream.
        """
        self._streaming_pi = self.fa.initial_belief(device=device)

    def observe(
        self,
        image: torch.Tensor,
        log_space: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the streaming state belief by one image.

        Args:
            image: Tensor of shape `(C, H, W)` or `(1, C, H, W)`.
            log_space: Whether the maintained state belief is in log
                space. Mode must match across an entire stream.

        Returns:
            Tuple `(state_belief, accept_probability)` where
            `state_belief` is the updated `pi_{t+1}` and
            `accept_probability` is `sum_{q in F} pi_{t+1}(q)`
            (or its log in log-space mode).

        Raises:
            RuntimeError: If `reset` was not called first.
        """
        if self._streaming_pi is None:
            raise RuntimeError("call `reset()` before `observe()`")
        if image.dim() == 3:
            image = image.unsqueeze(0)
        # forward through perception (shared case only -- streaming is
        #   not meaningful with per-state networks because the state
        #   belief itself is being updated.)
        logits = self.perception(image)
        if logits.dim() == 2 and logits.shape[0] == 1:
            logits = logits.view(-1)
        if log_space:
            symbol_probs = torch.log_softmax(logits, dim=-1)
        else:
            symbol_probs = torch.softmax(logits, dim=-1)
        self._streaming_pi = self.fa.step_soft(
            self._streaming_pi,
            symbol_probs,
            delta_logits=self.delta_logits,
            log_space=log_space,
        )
        F_mask = self.fa.F_mask(device=self._streaming_pi.device)
        if log_space:
            p_acc = torch.logsumexp(self._streaming_pi[F_mask], dim=0)
        else:
            p_acc = self._streaming_pi[F_mask].sum()
        return self._streaming_pi, p_acc
