"""Data generation for Half B regime (1) on the regex task.

Generates training / validation / test datasets for a single regex by
combining (a) random string sampling matched against the regex with
(b) per-character EMNIST letter image sampling. The resulting
sequences carry only an `accept` / `reject` label -- per-character
class labels are *not* used in training (the entire point of regime
(1) is to recover per-image classification accuracy from
sequence-level supervision alone).

Notation:
    images   -- Tensor(T, 1, 28, 28); the sequence given to the NSFA
    label    -- int in {0, 1}; accept iff 1
    string   -- str; the true characters, *not* used by the loss

Data is cached on disk via `torch.save` so repeated runs reuse the
same train/val/test split. Cache invalidates when the seed, regex, or
sample counts change.
"""

import hashlib
import json
import os
import random
import re
import string
from pathlib import Path

import rstr
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

DEFAULT_EMNIST_ROOT = "data/EMNIST"
MAX_LENGTH = 20


def _emnist_byclass(root: str) -> tuple[list, list[list[int]]]:
    """Load EMNIST letters (train split), return (images, byclass).

    `images[i]` is a `(1, 28, 28)` tensor. `byclass[c]` is a list of
    image indices whose true class is `c` (0 = 'a', ..., 25 = 'z').

    EMNIST letters are 1-indexed in the original dataset; we shift to
    0-indexed to match the alphabet's `Alphabet.to_index('a') == 0`.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    data = datasets.EMNIST(
        root=root,
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for image, label in data:
        images.append(image)
        labels.append(int(label) - 1)
    byclass: list[list[int]] = [
        [i for i, lbl in enumerate(labels) if lbl == c] for c in range(26)
    ]
    return images, byclass


def _matches_regex_lower(regex: str, s: str) -> bool:
    """Run Python `re.fullmatch` against the *lowercased* regex."""
    return bool(re.fullmatch(regex.lower(), s))


def _generate_strings(
    regex: str,
    num_accept: int,
    num_reject: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Generate distinct accept and reject strings for `regex`.

    Accept strings are sampled via `rstr.xeger` against the lowercased
    regex (matching `setup_experiment.py`'s convention). Reject strings
    are random lowercase strings of length 1..`MAX_LENGTH` that are
    *not* matched by the regex.

    Uniqueness is enforced per split but not across splits (callers
    handle the train/val/test partition).
    """
    accept_set: set[str] = set()
    while len(accept_set) < num_accept:
        s = rstr.xeger(regex.lower())
        if len(s) <= MAX_LENGTH and len(s) >= 1:
            accept_set.add(s)
    reject_set: set[str] = set()
    while len(reject_set) < num_reject:
        length = rng.randint(1, MAX_LENGTH)
        s = "".join(rng.choices(string.ascii_lowercase, k=length))
        if not _matches_regex_lower(regex, s):
            reject_set.add(s)
    return sorted(accept_set), sorted(reject_set)


def _string_to_images(
    s: str,
    emnist_images: list[torch.Tensor],
    byclass: list[list[int]],
    rng: random.Random,
) -> torch.Tensor:
    """Sample one EMNIST image per character; return `(T, 1, 28, 28)`."""
    tensors = []
    for c in s:
        idx = rng.choice(byclass[ord(c) - ord("a")])
        tensors.append(emnist_images[idx])
    return torch.stack(tensors, dim=0)


def _cache_key(
    regex: str, num_accept: int, num_reject: int, seed: int
) -> str:
    """Stable filename component encoding the dataset configuration."""
    blob = json.dumps(
        {"regex": regex, "na": num_accept, "nr": num_reject, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def generate_dataset(
    regex: str,
    num_train_per_class: int,
    num_val_per_class: int,
    num_test_per_class: int,
    seed: int,
    output_dir: Path,
    emnist_root: str = DEFAULT_EMNIST_ROOT,
) -> dict[str, list]:
    """Build train / val / test datasets for `regex`.

    Each split has `2 * num_*_per_class` examples balanced between
    accept and reject. Data is cached at
    `<output_dir>/data_<cache_key>.pt` so repeated runs with the same
    config skip generation. Returns a dict
    `{'train': [...], 'val': [...], 'test': [...]}` where each value
    is a list of `(images, label, string)` triples.

    Args:
        regex: Regular expression over [A-Za-z]. Lowercased before
            string sampling, consistent with `setup_experiment.py`
            and the refactored NSFA's `_build_nsfa`.
        num_train_per_class: Number of accept (and number of reject)
            sequences in the training split. Total train size is
            twice this.
        num_val_per_class: Same for validation.
        num_test_per_class: Same for test.
        seed: Single seed driving both string sampling and per-character
            EMNIST image sampling.
        output_dir: Where to cache the generated splits.
        emnist_root: Filesystem location of the EMNIST download.

    Returns:
        Dict with keys `train`, `val`, `test`; each value is a list of
        `(images: Tensor(T, 1, 28, 28), label: int, string: str)`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key(
        regex,
        num_train_per_class + num_val_per_class + num_test_per_class,
        num_train_per_class + num_val_per_class + num_test_per_class,
        seed,
    )
    cache_path = output_dir / f"data_{cache_key}.pt"
    if cache_path.exists():
        return torch.load(cache_path)

    rng = random.Random(seed)
    emnist_images, byclass = _emnist_byclass(emnist_root)

    total_accept = num_train_per_class + num_val_per_class + num_test_per_class
    total_reject = total_accept
    accept_strings, reject_strings = _generate_strings(
        regex, total_accept, total_reject, rng
    )
    rng.shuffle(accept_strings)
    rng.shuffle(reject_strings)

    def materialize(strings_acc: list[str], strings_rej: list[str]):
        examples = []
        for s in strings_acc:
            images = _string_to_images(s, emnist_images, byclass, rng)
            examples.append((images, 1, s))
        for s in strings_rej:
            images = _string_to_images(s, emnist_images, byclass, rng)
            examples.append((images, 0, s))
        rng.shuffle(examples)
        return examples

    train = materialize(
        accept_strings[:num_train_per_class],
        reject_strings[:num_train_per_class],
    )
    val = materialize(
        accept_strings[num_train_per_class : num_train_per_class + num_val_per_class],
        reject_strings[num_train_per_class : num_train_per_class + num_val_per_class],
    )
    test = materialize(
        accept_strings[-num_test_per_class:],
        reject_strings[-num_test_per_class:],
    )
    splits = {"train": train, "val": val, "test": test}
    torch.save(splits, cache_path)
    return splits


def build_per_image_eval_set(
    seed: int,
    n_per_class: int = 50,
    emnist_root: str = DEFAULT_EMNIST_ROOT,
) -> list[tuple[torch.Tensor, int]]:
    """Build a held-out single-image evaluation set for per-image accuracy.

    Uses EMNIST's *test* split so it does not overlap with the
    sequence-training data drawn from the *train* split.

    Args:
        seed: Sampling seed.
        n_per_class: Number of test images per letter (26 classes).
        emnist_root: Filesystem location of the EMNIST download.

    Returns:
        List of `(image: Tensor(1, 28, 28), class: int)` of length
        `26 * n_per_class`.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    data = datasets.EMNIST(
        root=emnist_root,
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )
    rng = random.Random(seed)
    byclass: list[list[int]] = [[] for _ in range(26)]
    all_examples: list[tuple[torch.Tensor, int]] = []
    for image, label in data:
        c = int(label) - 1
        byclass[c].append((image, c))
    examples: list[tuple[torch.Tensor, int]] = []
    for c in range(26):
        rng.shuffle(byclass[c])
        examples.extend(byclass[c][:n_per_class])
    rng.shuffle(examples)
    return examples
