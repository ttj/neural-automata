"""Regime (2) entry point: jointly train perception + delta_theta.

Pipeline:
    1. Load `config.yaml`.
    2. Reuse the cached regime-(1) data when present (same regex,
       same data-split seed -> same files; see `data_cache_dir`).
    3. Build an NSFA wrapping a fresh `LetterCNN` AND a learnable
       `delta_logits` warm-started from the hand-built DFA's one-hot
       projection.
    4. Train via `nsa.learning.Trainer` with optional row-entropy
       (or other) regularisation pulling `delta_logits` toward peaky
       rows.
    5. Evaluate sequence and per-image accuracy as in regime (1),
       then **discretise** `delta_logits` via argmax + minify and
       compare the recovered DFA to the hand-built reference (under
       language equivalence -- two DFAs are equivalent iff their
       symmetric difference is empty).
    6. Write all artifacts under `experiments/halfb/regime2/<regex>/`.

Run from the repository root:

    ~/miniconda3/envs/neural-automata/bin/python \
        -m experiments.halfb.regime2.run
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
import torch.optim as optim
import yaml

from nsa import FiniteAutomaton, NSFA, PerceptionAdapter
from nsa.alphabet import lowercase_alphabet
from nsa.learning import Trainer, discretize_to_fa

from examples.regex.networks.letter_cnn import LetterCNN
from experiments.halfb.regime1.data import (
    build_per_image_eval_set,
    generate_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "halfb" / "regime2" / "config.yaml"
)


def _build_nsfa(
    regex: str,
    seed: int,
    delta_init: str,
    delta_init_scale: float,
) -> tuple[NSFA, FiniteAutomaton]:
    """Build regime-(2) NSFA. Returns the NSFA and the hand-built FA reference."""
    alphabet = lowercase_alphabet()
    fa_reference = FiniteAutomaton.from_regex(
        regex.lower(), input_symbols=set(alphabet)
    )
    torch.manual_seed(seed)
    cnn = LetterCNN(num_classes=26)
    perception = PerceptionAdapter(network=cnn, n_alphabet=26)
    nsfa = NSFA(
        fa_reference,
        perception,
        learn_delta=True,
        delta_init=delta_init,
        delta_init_scale=delta_init_scale,
    )
    return nsfa, fa_reference


def _setup_output_dir(regex: str) -> Path:
    out = REPO_ROOT / "experiments" / "halfb" / "regime2" / regex
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def _configure_logging(log_path: Path) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
    )


def _compare_recovered_dfa(
    nsfa: NSFA, fa_reference: FiniteAutomaton
) -> dict[str, object]:
    """Discretise `delta_logits` and compare to the hand-built DFA.

    Returns a dict reporting (1) the recovered DFA's state and
    transition counts after Hopcroft minification, (2) whether it
    accepts the same language as the reference (via
    `automata-lib`'s symmetric-difference test), (3) the reference's
    state count for context.
    """
    assert nsfa.delta_logits is not None
    recovered = discretize_to_fa(
        delta_logits=nsfa.delta_logits.detach(),
        alphabet=nsfa.fa.alphabet,
        q0_index=nsfa.fa.q0_index,
        F_indices=nsfa.fa.F_indices,
        state_names=list(nsfa.fa.states),
        minify=True,
    )
    # `automata-lib`'s `==` on DFAs checks language equivalence by
    #   computing the symmetric difference and asking if it's empty.
    equivalent = recovered.dfa == fa_reference.dfa
    return {
        "recovered_n_states": recovered.n_states,
        "recovered_n_accepting": len(recovered.F),
        "reference_n_states": fa_reference.n_states,
        "language_equivalent": bool(equivalent),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Half B / regime (2) training driver."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = _setup_output_dir(cfg["regex"])
    log_path = out_dir / "logs" / "train.log"
    _configure_logging(log_path)
    logger = logging.getLogger("regime2")

    shutil.copy(args.config, out_dir / "config.yaml")
    logger.info("config: %s", json.dumps(cfg, sort_keys=True))

    # data (cache shared with regime 1 when seed and counts match)
    logger.info("preparing data...")
    splits = generate_dataset(
        regex=cfg["regex"],
        num_train_per_class=cfg["num_train_per_class"],
        num_val_per_class=cfg["num_val_per_class"],
        num_test_per_class=cfg["num_test_per_class"],
        seed=cfg["seed"],
        output_dir=Path(cfg["data_cache_dir"]),
        emnist_root=cfg["emnist_root"],
    )
    per_image_eval = build_per_image_eval_set(
        seed=cfg["seed"] + 1,
        n_per_class=cfg["per_image_n_per_class"],
        emnist_root=cfg["emnist_root"],
    )
    logger.info(
        "data sizes: train=%d val=%d test=%d per_image=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        len(per_image_eval),
    )

    # model + optimizer
    nsfa, fa_reference = _build_nsfa(
        cfg["regex"],
        cfg["seed"],
        cfg["delta_init"],
        cfg["delta_init_scale"],
    )
    optimizer_cls = getattr(optim, cfg["optimizer"])
    optimizer = optimizer_cls(
        nsfa.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    trainer = Trainer(
        nsfa,
        optimizer,
        log_space=True,
        device="cpu",
        grad_clip_norm=cfg.get("grad_clip_norm"),
        regularizer_weights=cfg.get("regularizer_weights"),
    )
    logger.info(
        "learnable params: %d delta_logits=%s",
        sum(p.numel() for p in nsfa.parameters() if p.requires_grad),
        tuple(nsfa.delta_logits.shape) if nsfa.delta_logits is not None else None,
    )

    # baseline (random-weight CNN + warm-started delta) evaluation
    train_pairs = [(im, lbl) for im, lbl, _s in splits["train"]]
    val_pairs = [(im, lbl) for im, lbl, _s in splits["val"]]
    test_pairs = [(im, lbl) for im, lbl, _s in splits["test"]]
    baseline_seq = trainer.evaluate(test_pairs)
    baseline_img = trainer.evaluate_per_image(per_image_eval)
    baseline_dfa = _compare_recovered_dfa(nsfa, fa_reference)
    logger.info(
        "baseline: test_loss=%.4f test_acc=%.3f per_image_top1=%.3f recovered_equiv=%s",
        baseline_seq["loss"],
        baseline_seq["accept_accuracy"],
        baseline_img["top1"],
        baseline_dfa["language_equivalent"],
    )

    # train
    logger.info("starting training for %d epochs...", cfg["epochs"])
    history = trainer.fit(
        train_pairs,
        val_pairs,
        epochs=cfg["epochs"],
        batch_accumulate=cfg["batch_accumulate"],
        save_dir=out_dir / "checkpoints",
    )
    trainer.save_history(history, out_dir / "history.json")

    # final evaluation (best checkpoint restored by fit)
    final_seq = trainer.evaluate(test_pairs)
    final_img = trainer.evaluate_per_image(per_image_eval)
    final_dfa = _compare_recovered_dfa(nsfa, fa_reference)
    results = {
        "config": cfg,
        "baseline": {**baseline_seq, **baseline_img, **baseline_dfa},
        "final": {**final_seq, **final_img, **final_dfa},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(
        "final: test_loss=%.4f test_acc=%.3f per_image_top1=%.3f recovered_n_states=%d recovered_equiv=%s",
        final_seq["loss"],
        final_seq["accept_accuracy"],
        final_img["top1"],
        final_dfa["recovered_n_states"],
        final_dfa["language_equivalent"],
    )


if __name__ == "__main__":
    main()
