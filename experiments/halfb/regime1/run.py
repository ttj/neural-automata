"""Regime (1) entry point: train an NSFA from sequence-level supervision.

Pipeline:
    1. Load `config.yaml` (path overridable via `--config`).
    2. Generate or load cached train / val / test data for the
       configured regex.
    3. Build an NSFA wrapping a fresh `LetterCNN` (random weights) and
       the regex's hand-built DFA. `learn_delta=False` for regime (1).
    4. Train via `nsa.learning.Trainer` for the configured number of
       epochs, tracking best val-loss checkpoint.
    5. Evaluate on:
         - The sequence test split (accept / reject classification),
         - A held-out single-image EMNIST split (per-image top-1).
    6. Write all artifacts to `experiments/halfb/regime1/<regex>/`:
         - `config.yaml`         (copy of the config actually used)
         - `checkpoints/best.pth`
         - `logs/train.log`
         - `history.json`
         - `results.json`

Run from the repository root:

    ~/miniconda3/envs/neural-automata/bin/python \
        -m experiments.halfb.regime1.run

    # or with a different config:
    ~/miniconda3/envs/neural-automata/bin/python \
        -m experiments.halfb.regime1.run --config path/to/config.yaml
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
from nsa.learning import Trainer

from examples.regex.networks.letter_cnn import LetterCNN
from experiments.halfb.regime1.data import (
    build_per_image_eval_set,
    generate_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "halfb" / "regime1" / "config.yaml"
)


def _build_nsfa(regex: str, seed: int) -> NSFA:
    """Build a fresh-weights NSFA over `regex` and the 26-letter alphabet."""
    alphabet = lowercase_alphabet()
    fa = FiniteAutomaton.from_regex(
        regex.lower(), input_symbols=set(alphabet)
    )
    torch.manual_seed(seed)
    cnn = LetterCNN(num_classes=26)
    perception = PerceptionAdapter(network=cnn, n_alphabet=26)
    return NSFA(fa, perception, learn_delta=False)


def _setup_output_dir(regex: str) -> Path:
    """Per-regex output directory under `experiments/halfb/regime1/`."""
    out = REPO_ROOT / "experiments" / "halfb" / "regime1" / regex
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Half B / regime (1) training driver."
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
    logger = logging.getLogger("regime1")

    # snapshot the config that was actually used
    shutil.copy(args.config, out_dir / "config.yaml")
    logger.info("config: %s", json.dumps(cfg, sort_keys=True))

    # data
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
    nsfa = _build_nsfa(cfg["regex"], cfg["seed"])
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
    )

    # baseline (random-weight) evaluation
    train_pairs = [(im, lbl) for im, lbl, _s in splits["train"]]
    val_pairs = [(im, lbl) for im, lbl, _s in splits["val"]]
    test_pairs = [(im, lbl) for im, lbl, _s in splits["test"]]
    baseline_seq = trainer.evaluate(test_pairs)
    baseline_img = trainer.evaluate_per_image(per_image_eval)
    logger.info(
        "baseline (random init): test_loss=%.4f test_acc=%.3f per_image_top1=%.3f",
        baseline_seq["loss"],
        baseline_seq["accept_accuracy"],
        baseline_img["top1"],
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

    # final evaluation (with best-checkpoint weights restored by fit)
    final_seq = trainer.evaluate(test_pairs)
    final_img = trainer.evaluate_per_image(per_image_eval)
    results = {
        "config": cfg,
        "baseline": {**baseline_seq, **baseline_img},
        "final": {**final_seq, **final_img},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(
        "final: test_loss=%.4f test_acc=%.3f per_image_top1=%.3f",
        final_seq["loss"],
        final_seq["accept_accuracy"],
        final_img["top1"],
    )


if __name__ == "__main__":
    main()
