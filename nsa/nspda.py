"""Neurosymbolic pushdown automaton: composition of PDA + perception.

Implements Definition 4 of the NeuS 2025 paper. Half A scope:
discrete inference through a real DPDA (replacing the published
release's ad-hoc Python interpreter loop), plus a `fold` API that
makes downstream evaluation explicit. The soft / differentiable
forward algorithm with a bounded-stack tensor view is Half B.

Notation:
    images        -- Tensor(T, C, H, W), the input image sequence
    end_marker    -- optional symbol appended to the symbol stream
                       after perception; used when the DPDA's
                       acceptance depends on a sentinel '#' (paper
                       Figure 2 convention)
    P_acc         -- not computed here; see Half B `forward()`.
"""

from typing import Callable, Hashable

import torch
import torch.nn as nn

from nsa.pda import PushdownAutomaton
from nsa.perception import PerceptionAdapter


class NSPDA(nn.Module):
    """Neurosymbolic pushdown automaton.

    Composes a `PushdownAutomaton` (discrete; wrapping
    `automata-lib`'s DPDA) with a `PerceptionAdapter`. Half A exposes
    `accept(images)` for the paper's Table 3 NA column reproduction
    and `evaluate(images, fold_fn, initial)` for tasks that fold a
    value over the PDA's trace (arithmetic evaluation).

    Attributes:
        pda: The symbolic pushdown automaton.
        perception: The perception adapter.
        end_marker: Optional sentinel symbol appended after the
            perception-emitted sequence before the PDA reads it. Use
            `'#'` for the paper's arithmetic DPDA which expects an
            end-of-input marker. The end_marker must be in
            `pda.alphabet` but is not produced by the perception
            network.
    """

    def __init__(
        self,
        pda: PushdownAutomaton,
        perception: PerceptionAdapter,
        end_marker: Hashable | None = None,
    ) -> None:
        """Initialize an NSPDA.

        Args:
            pda: The PushdownAutomaton.
            perception: The PerceptionAdapter; `n_alphabet` must equal
                the size of the PDA's alphabet excluding the
                `end_marker`.
            end_marker: Optional symbol in `pda.alphabet` appended
                after the perception output. None means the perception
                alphabet equals the PDA alphabet exactly.

        Raises:
            ValueError: If alphabet sizes are inconsistent with the
                presence/absence of `end_marker`.
        """
        super().__init__()
        expected_perception_alphabet = pda.n_alphabet - (
            1 if end_marker is not None else 0
        )
        if perception.n_alphabet != expected_perception_alphabet:
            raise ValueError(
                f"perception.n_alphabet {perception.n_alphabet} != "
                f"pda.n_alphabet - "
                f"(1 if end_marker else 0) = {expected_perception_alphabet}"
            )
        if end_marker is not None and end_marker not in pda.alphabet:
            raise ValueError(
                f"end_marker {end_marker!r} not in pda.alphabet "
                f"{set(pda.alphabet)!r}"
            )
        self.pda = pda
        self.perception = perception
        self.end_marker = end_marker

    # ------------------------------------------------------------------ #
    # Symbol emission (perception + end-marker append)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _emit_symbols(self, images: torch.Tensor) -> list[Hashable]:
        """Argmax the perception logits per image, then append end_marker."""
        logits = self.perception(images)
        preds = torch.argmax(logits, dim=-1).tolist()
        # perception's alphabet covers indices 0..n_perception_alphabet-1
        # of pda.alphabet (the end_marker, if any, is the last index)
        symbols: list[Hashable] = [
            self.pda.alphabet.to_symbol(i) for i in preds
        ]
        if self.end_marker is not None:
            symbols.append(self.end_marker)
        return symbols

    # ------------------------------------------------------------------ #
    # Discrete inference
    # ------------------------------------------------------------------ #

    def accept(self, images: torch.Tensor) -> tuple[bool, str]:
        """Discrete acceptance: argmax per image, then run the DPDA.

        Args:
            images: Tensor of shape `(T, C, H, W)`.

        Returns:
            Tuple `(accepted, predicted_string)` where `predicted_string`
            is the perception's argmax sequence joined as a string
            (excluding the end marker for clean logging).
        """
        symbols = self._emit_symbols(images)
        # the predicted string we report excludes any appended
        #   end_marker so the log matches the paper convention
        if self.end_marker is not None:
            predicted_str = "".join(str(s) for s in symbols[:-1])
        else:
            predicted_str = "".join(str(s) for s in symbols)
        accepted = self.pda.accept(symbols)
        return accepted, predicted_str

    def accept_with_trace(
        self, images: torch.Tensor
    ) -> tuple[bool, str, list]:
        """Discrete acceptance, predicted string, and final stack contents.

        Args:
            images: Tensor of shape `(T, C, H, W)`.

        Returns:
            Tuple `(accepted, predicted_string, final_stack)`.
            `final_stack` is **bottom-up** (top of stack is the last
            element); see `PushdownAutomaton.accept_with_trace`.
        """
        symbols = self._emit_symbols(images)
        if self.end_marker is not None:
            predicted_str = "".join(str(s) for s in symbols[:-1])
        else:
            predicted_str = "".join(str(s) for s in symbols)
        accepted, stack = self.pda.accept_with_trace(symbols)
        return accepted, predicted_str, stack

    def evaluate(
        self,
        images: torch.Tensor,
        fold_fn: Callable[[object, Hashable, object, object, list], object],
        initial: object,
    ) -> tuple[bool, str, object]:
        """Run the PDA over perception output and fold over the trace.

        Args:
            images: Tensor of shape `(T, C, H, W)`.
            fold_fn: Step function; see `PushdownAutomaton.fold`.
            initial: Initial accumulator value.

        Returns:
            Tuple `(accepted, predicted_string, final_accumulator)`.
        """
        symbols = self._emit_symbols(images)
        if self.end_marker is not None:
            predicted_str = "".join(str(s) for s in symbols[:-1])
        else:
            predicted_str = "".join(str(s) for s in symbols)
        accepted, value = self.pda.fold(symbols, fold_fn, initial)
        return accepted, predicted_str, value

    # ------------------------------------------------------------------ #
    # Soft inference -- DEFERRED to Half B
    # ------------------------------------------------------------------ #

    def forward(self, *args, **kwargs):
        """Soft acceptance probability via the bounded-stack forward.

        Not implemented in Half A; see Half B (plan Section 2). The
        composition is registered as an `nn.Module` only so that
        perception parameters are picked up by `parameters()` for the
        Half B trainer.
        """
        raise NotImplementedError(
            "NSPDA.forward is deferred to Half B; use accept, "
            "accept_with_trace, or evaluate for discrete inference."
        )
