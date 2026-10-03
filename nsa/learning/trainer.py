"""Minimal Trainer for NSFA / NSPDA end-to-end training.

This is the plain-PyTorch trainer chosen in plan Section 4
(decision-point 1 of Phase 2). No PyTorch-Lightning, no callbacks, no
distributed support -- the simplest thing that supports the three
training regimes (perception-only, joint, PDA) defined in
`.claude/plans/2026-05-12_neutron-refactor-and-e2e-scaffold.md`
Section 2.3.

Notation:
    images   -- Tensor(T, C, H, W); one input sequence
    label    -- scalar in {0, 1}; accept / reject target
    log_p    -- scalar log-acceptance from the soft forward algorithm
"""

import copy
import json
import logging
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from nsa.learning.losses import sequence_acceptance_loss
from nsa.learning.regularizers import (
    determinism_penalty,
    row_entropy,
)
from nsa.nsfa import NSFA


REGULARIZERS_BY_NAME = {
    "row_entropy": row_entropy,
    "determinism_penalty": determinism_penalty,
}


logger = logging.getLogger(__name__)


class Trainer:
    """Train an NSFA against `(images, accept_label)` examples.

    The training data is an iterable of `(images, label)` pairs;
    sequences may be of different lengths because each call to
    `nsfa.forward(images)` runs the forward algorithm independently.
    Gradient batching is by accumulation (`batch_accumulate`) since
    cross-sequence batching would require per-length padding.

    Attributes:
        nsfa: The NSFA being trained.
        optimizer: A torch optimizer over `nsfa.parameters()` (which
            covers the perception network always and `delta_logits`
            iff `nsfa.learn_delta`).
        log_space: If True, the forward algorithm runs in log-semiring
            (recommended). Loss is computed by `sequence_acceptance_loss`,
            which takes `log P_acc` directly.
        device: Torch device for examples / state.
    """

    def __init__(
        self,
        nsfa: NSFA,
        optimizer: torch.optim.Optimizer,
        log_space: bool = True,
        device: torch.device | str = "cpu",
        grad_clip_norm: float | None = 1.0,
        regularizer_weights: dict[str, float] | None = None,
    ) -> None:
        self.nsfa = nsfa.to(device)
        self.optimizer = optimizer
        self.log_space = log_space
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.regularizer_weights = regularizer_weights or {}
        for name in self.regularizer_weights:
            if name not in REGULARIZERS_BY_NAME:
                raise ValueError(
                    f"unknown regularizer {name!r}; "
                    f"valid: {sorted(REGULARIZERS_BY_NAME)}"
                )
        self._best_state: dict | None = None
        self._best_val: float = float("inf")

    def _delta_regularizer_term(self) -> torch.Tensor:
        """Sum of weighted regularizers over `delta_logits` (regime 2 only).

        Returns a zero scalar when `delta_logits` is None (regime 1)
        or when no regularizers are configured.
        """
        if (
            self.nsfa.delta_logits is None
            or not self.regularizer_weights
        ):
            return torch.zeros((), device=self.device)
        terms = torch.zeros((), device=self.device)
        for name, weight in self.regularizer_weights.items():
            fn = REGULARIZERS_BY_NAME[name]
            terms = terms + float(weight) * fn(self.nsfa.delta_logits)
        return terms

    # ------------------------------------------------------------------ #
    # Train / evaluate one epoch
    # ------------------------------------------------------------------ #

    def train_epoch(
        self,
        dataset: Iterable[tuple[torch.Tensor, int]],
        batch_accumulate: int = 1,
    ) -> float:
        """Run one training epoch; returns mean per-example loss."""
        self.nsfa.train()
        total_loss = 0.0
        n_seen = 0
        self.optimizer.zero_grad()
        accumulated = 0
        for images, label in dataset:
            images = images.to(self.device)
            label_t = torch.tensor([float(label)], device=self.device)
            log_p = self.nsfa(images, log_space=self.log_space)
            loss = sequence_acceptance_loss(log_p.unsqueeze(0), label_t)
            loss = loss + self._delta_regularizer_term()
            (loss / batch_accumulate).backward()
            accumulated += 1
            if accumulated >= batch_accumulate:
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.nsfa.parameters(), self.grad_clip_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulated = 0
            total_loss += float(loss.detach())
            n_seen += 1
        # flush partial accumulated batch
        if accumulated > 0:
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.nsfa.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss / max(n_seen, 1)

    @torch.no_grad()
    def evaluate(
        self,
        dataset: Iterable[tuple[torch.Tensor, int]],
    ) -> dict[str, float]:
        """Return `{loss, accept_accuracy}` over the dataset.

        `accept_accuracy` thresholds `P_acc` at 0.5.
        """
        self.nsfa.eval()
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        for images, label in dataset:
            images = images.to(self.device)
            label_t = torch.tensor([float(label)], device=self.device)
            log_p = self.nsfa(images, log_space=self.log_space)
            loss = sequence_acceptance_loss(log_p.unsqueeze(0), label_t)
            total_loss += float(loss.detach())
            if self.log_space:
                p = float(log_p.exp())
            else:
                p = float(log_p)
            pred = int(p >= 0.5)
            n_correct += int(pred == int(label))
            n_total += 1
        return {
            "loss": total_loss / max(n_total, 1),
            "accept_accuracy": n_correct / max(n_total, 1),
        }

    @torch.no_grad()
    def evaluate_per_image(
        self,
        image_dataset: Iterable[tuple[torch.Tensor, int]],
    ) -> dict[str, float]:
        """Top-1 accuracy of the wrapped CNN on a held-out single-image set.

        This is the headline metric for regime (1): how close to the
        per-image-supervised baseline (paper's 93.6% on EMNIST letters)
        can sequence-level supervision get us?

        Args:
            image_dataset: Iterable of `(image, class_index)` pairs.
                `image` is shape `(C, H, W)` or `(1, C, H, W)`;
                `class_index` is an integer in `[0, n_alphabet)`.

        Returns:
            Dict with `top1`. `top1` is the fraction of correctly
            classified single images.
        """
        self.nsfa.eval()
        cnn = self.nsfa.perception.networks[0]
        n_correct = 0
        n_total = 0
        for image, true_class in image_dataset:
            image = image.to(self.device)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            logits = cnn(image).view(-1)
            pred = int(torch.argmax(logits).item())
            if pred == int(true_class):
                n_correct += 1
            n_total += 1
        return {"top1": n_correct / max(n_total, 1)}

    # ------------------------------------------------------------------ #
    # Fit / checkpoint
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_data: list[tuple[torch.Tensor, int]],
        val_data: list[tuple[torch.Tensor, int]],
        epochs: int,
        batch_accumulate: int = 1,
        save_dir: Path | None = None,
    ) -> list[dict[str, float]]:
        """Train for `epochs` and track best val loss.

        Args:
            train_data: List of `(images, label)` pairs.
            val_data: Same, used for early-stopping reference + best
                checkpoint.
            epochs: Number of full passes over `train_data`.
            batch_accumulate: Gradient-accumulation batch size.
            save_dir: If set, the best-val-loss checkpoint is written
                to `<save_dir>/best.pth`.

        Returns:
            Per-epoch metric history.
        """
        history: list[dict[str, float]] = []
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_loss = self.train_epoch(
                train_data, batch_accumulate=batch_accumulate
            )
            val_metrics = self.evaluate(val_data)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accept_accuracy": val_metrics["accept_accuracy"],
            }
            history.append(record)
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["accept_accuracy"],
            )
            if val_metrics["loss"] < self._best_val:
                self._best_val = val_metrics["loss"]
                self._best_state = copy.deepcopy(self.nsfa.state_dict())
                if save_dir is not None:
                    torch.save(
                        self._best_state, save_dir / "best.pth"
                    )

        # restore best state at end of training
        if self._best_state is not None:
            self.nsfa.load_state_dict(self._best_state)
        return history

    def save_history(self, history: list[dict], path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
