"""Perception adapter: wraps `nn.Module` networks into a uniform symbol-emitter.

A perception adapter exposes a single call:

    forward(images) -> logits over Sigma

where `images` is a sequence of input images (possibly batched) and the
output is a tensor of per-step logits over the input alphabet. The
adapter handles two cases from the NeuS 2025 paper:

  - a single network shared across all states (paper Section 5.2),
  - one network per state via a labeling function `ell : Q -> N` (paper
    Section 4.1, Definition 3).

In the shared-network case the labeling function is the constant
function mapping every state to the single network; the adapter
collapses this to a single forward pass over the entire sequence and
ignores state context. In the per-state case the adapter accepts a
state-belief vector and weights the per-state networks' outputs
accordingly -- consistent with the soft runtime.

Notation:
    f_phi      -- perception network parameterized by phi
    ell        -- labeling function Q -> set of networks
    I_t        -- input image at time t
    logits_t   -- f_phi(I_t), shape (n_alphabet,)
"""

from typing import Mapping

import torch
import torch.nn as nn


class PerceptionAdapter(nn.Module):
    """Wraps one or more `nn.Module`s into a per-step symbol logit emitter.

    The simpler case -- a single network shared across all states --
    is the default. Per-state networks are supported via the
    `state_to_network` constructor argument.

    Attributes:
        n_alphabet: |Sigma| -- the output dimensionality of each
            wrapped network's logit head.
        n_states: Number of automaton states this adapter covers (1 in
            the shared-network case, |Q| in the per-state case).
        shared: True iff a single network is used across all states.
    """

    def __init__(
        self,
        network: nn.Module | None = None,
        state_to_network: Mapping[int, nn.Module] | None = None,
        n_alphabet: int | None = None,
    ) -> None:
        """Initialize a PerceptionAdapter.

        Args:
            network: A single `nn.Module` shared across all automaton
                states. Mutually exclusive with `state_to_network`.
            state_to_network: A mapping from state index to a per-state
                `nn.Module`. Used for the labeling-function variant
                (paper Definition 3). Mutually exclusive with `network`.
            n_alphabet: Output dimensionality of each network's logit
                head. Required only when it cannot be inferred from the
                network(s).

        Raises:
            ValueError: If neither or both of `network` and
                `state_to_network` are provided.
        """
        super().__init__()
        if (network is None) == (state_to_network is None):
            raise ValueError(
                "exactly one of `network` or `state_to_network` must be set"
            )
        if network is not None:
            self.shared = True
            self.networks = nn.ModuleList([network])
            self._state_to_net_index: dict[int, int] = {}
        else:
            self.shared = False
            assert state_to_network is not None  # for type checker
            indices = sorted(state_to_network.keys())
            self.networks = nn.ModuleList(
                [state_to_network[i] for i in indices]
            )
            self._state_to_net_index = {i: idx for idx, i in enumerate(indices)}
        if n_alphabet is None:
            raise ValueError(
                "n_alphabet must be provided; it cannot be inferred from "
                "an `nn.Module` without running a forward pass"
            )
        self.n_alphabet: int = n_alphabet
        self.n_states: int = 1 if self.shared else len(self.networks)

    def forward(
        self,
        images: torch.Tensor,
        state_belief: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Emit per-step logits over the alphabet for a sequence of images.

        Args:
            images: A tensor of shape `(T, C, H, W)` or `(T, 1, H, W)`
                -- a sequence of T images. Single-image batches per
                step (i.e. unsqueezed) are also accepted as the legacy
                CNN code uses that convention.
            state_belief: Optional state belief at each timestep, shape
                `(T, n_states)`. Required in the per-state case;
                ignored in the shared case.

        Returns:
            A tensor of shape `(T, n_alphabet)` giving per-step logits
            over the input alphabet.
        """
        if images.dim() < 3:
            raise ValueError(
                f"images must be at least 3D; got shape {tuple(images.shape)}"
            )
        T = images.shape[0]
        if self.shared:
            # one network applied per timestep; the legacy CNN
            #   architecture expects (1, C, H, W) per call and returns
            #   (n_alphabet,). Preserve that convention by looping.
            outputs = []
            for t in range(T):
                logits = self.networks[0](images[t : t + 1])
                outputs.append(logits.view(-1))
            return torch.stack(outputs, dim=0)
        # per-state case
        if state_belief is None:
            raise ValueError(
                "state_belief is required when using per-state networks"
            )
        if state_belief.shape != (T, self.n_states):
            raise ValueError(
                f"state_belief shape {tuple(state_belief.shape)} "
                f"!= (T={T}, n_states={self.n_states})"
            )
        # collect logits from every network at every timestep, then mix
        #   by the state belief: logits[t] = sum_q b[t, q] * f_q(I_t).
        per_state_logits = torch.stack(
            [net(images.view(T, *images.shape[1:])) for net in self.networks],
            dim=1,
        )  # shape (T, n_states, n_alphabet)
        if per_state_logits.dim() == 2:
            # individual networks returned (n_alphabet,) per image; recover
            per_state_logits = per_state_logits.view(T, self.n_states, -1)
        return torch.einsum("tq,tqa->ta", state_belief, per_state_logits)
