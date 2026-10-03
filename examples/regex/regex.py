"""Image-based regular string acceptance experiment.

Reproduces Table 1 of Sasaki/Manzanas Lopez/Johnson, NeuS 2025, with
the NSFA running through the `nsa` package's
`FiniteAutomaton` + `PerceptionAdapter` + `NSFA` composition rather
than the ad-hoc Python `re.fullmatch` path used in the original
release. The regex is compiled to a DFA via
`FiniteAutomaton.from_regex` (which uses `automata-lib`'s
`NFA.from_regex` + `DFA.from_nfa`), and the CNN's per-image argmax is
fed to the DFA through `NSFA.accept`.

Notation mapping:
    images  -- Tensor(T, 1, 28, 28), the EMNIST sample sequence
    regex   -- str, regular expression over [a-z]
    NSFA    -- composition (FiniteAutomaton, PerceptionAdapter)
    P_acc   -- not computed in this script (discrete inference path);
               use NSFA.forward() in a training script for soft acceptance.
"""

import os
import signal
import time
from functools import lru_cache

import torch

from nsa import FiniteAutomaton, NSFA, PerceptionAdapter
from nsa.alphabet import lowercase_alphabet
from examples.regex.networks.letter_cnn import LetterCNN

# `ollama` is only used by the VLM comparison functions and is
#   imported lazily inside `vlm()`; the NSFA path has no ollama
#   dependency, so a clean install without ollama can still run the
#   neurosymbolic experiment and the test suite.

TIMEOUT = 100


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise TimeoutError("Function timed out!")


signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(TIMEOUT)


# ---------------------------------------------------------------------------- #
# NSFA path
# ---------------------------------------------------------------------------- #

LOWERCASE = lowercase_alphabet()


@lru_cache(maxsize=1)
def _letter_cnn() -> LetterCNN:
    """Load the EMNIST 26-class letter CNN once and cache it.

    The checkpoint path matches the published release. Eval mode is set
    so the dropout layer is deterministic.
    """
    model_fp = os.path.join(
        os.getcwd(), "examples", "regex", "models", "model.pth"
    )
    model = LetterCNN(num_classes=26)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    return model


def _build_nsfa(regex: str) -> NSFA:
    """Compile a regex into an NSFA wrapping the cached CNN.

    The paper's regexes mix case (e.g. `[A-Za-z]`) but the data
    generator only ever emits lowercase strings
    (`rstr.xeger(rf'{regex}'.lower())` in
    `setup_experiment.py:generate_valid_string`) and the CNN
    classifies into 26 lowercase classes. To keep the DFA's alphabet
    aligned with the perception (26 symbols), the regex is
    lowercased before compilation. This preserves the language on
    lowercase inputs: `[A-Za-z]` after `.lower()` becomes
    `[a-za-z]`, which equals `[a-z]`.

    Args:
        regex: Regular expression; may contain `[A-Za-z]` ranges.

    Returns:
        Configured NSFA (discrete inference; learn_delta=False).
    """
    fa = FiniteAutomaton.from_regex(regex.lower(), input_symbols=set(LOWERCASE))
    perception = PerceptionAdapter(network=_letter_cnn(), n_alphabet=26)
    return NSFA(fa, perception, learn_delta=False)


def neurosymbolic_automaton(
    input_str: str, regex: str
) -> tuple[str, bool, float]:
    """Run a single image-sequence through the NSFA for a given regex.

    Args:
        input_str: Path to the saved tensor of shape (T, 1, 28, 28).
        regex: Regular expression over [a-z].

    Returns:
        Triple `(predicted_string, accepted, elapsed_seconds)` matching
        the legacy interface in the published release.
    """
    images = torch.load(input_str)
    nsfa = _build_nsfa(regex)

    start = time.time()
    accepted, predicted_str = nsfa.accept(images)
    elapsed = time.time() - start

    return predicted_str, accepted, elapsed


def get_na_output() -> None:
    """Run the NSFA over every experiment directory under data/na.

    For each experiment (one per regex), compares the NSFA's
    predicted string and accept/reject decision against the labels in
    `data/experiment_*__<regex>.txt` and writes per-sample results
    plus a per-experiment summary into `results/na/`. Matches the
    output schema of the original release so downstream analysis
    scripts continue to work.
    """
    na_fp = os.path.join(os.getcwd(), "examples", "regex", "data", "na")
    total_str_correct = 0
    total_accept_correct = 0

    for experiment_name in os.listdir(na_fp):
        print(f'working on {experiment_name.split("__")[0]}')
        results_fp = os.path.join(
            os.getcwd(),
            "examples",
            "regex",
            "results",
            "na",
            f"{experiment_name}.txt",
        )
        labels_fp = os.path.join(
            os.getcwd(),
            "examples",
            "regex",
            "data",
            f"{experiment_name}.txt",
        )
        na_exp_fp = os.path.join(na_fp, experiment_name)
        regex = experiment_name.split("__")[1]

        with open(labels_fp, "r") as labels:
            samples_fp = sorted(
                os.listdir(na_exp_fp), key=lambda x: int(x.split("_")[1])
            )
            num_str_correct = 0
            num_accept_correct = 0
            with open(results_fp, "a+") as r:
                for sample_num, sample_fp in enumerate(samples_fp):
                    full_sample_fp = os.path.join(na_exp_fp, sample_fp)
                    label_line = labels.readline()
                    true_string, accept = label_line.split(",")
                    accept = bool(int(accept))
                    predicted_str, na_accept, elapsed = neurosymbolic_automaton(
                        full_sample_fp, regex
                    )
                    print(
                        f"#{sample_num} -> {true_string}="
                        f"{'accept' if accept else 'reject'} | "
                        f"{predicted_str}="
                        f"{'accept' if na_accept else 'reject'}: {elapsed}"
                    )
                    r.write(
                        f"#{sample_num} -> {true_string}="
                        f"{'accept' if accept else 'reject'} | "
                        f"{predicted_str}="
                        f"{'accept' if na_accept else 'reject'}: {elapsed}\n"
                    )
                    if true_string == predicted_str:
                        num_str_correct += 1
                    if accept == na_accept:
                        num_accept_correct += 1
                print(
                    f"String Classification Accuracy: "
                    f"{(num_str_correct / 10) * 100}%, "
                    f"Task Accuracy: "
                    f"{(num_accept_correct / 10) * 100}%\n\n"
                )
                r.write(
                    f"String Classification Accuracy: "
                    f"{(num_str_correct / 10) * 100}%, "
                    f"Task Accuracy: "
                    f"{(num_accept_correct / 10) * 100}%"
                )
                total_str_correct += num_str_correct
                total_accept_correct += num_accept_correct

    totals_fp = os.path.join(
        os.getcwd(),
        "examples",
        "regex",
        "results",
        "na",
        "totals.txt",
    )
    with open(totals_fp, "a+") as r:
        print(
            f"String Classification Accuracy: "
            f"{total_str_correct}%, Task Accuracy: "
            f"{total_accept_correct}%\n\n"
        )
        r.write(
            f"String Classification Accuracy: "
            f"{total_str_correct}%, Task Accuracy: "
            f"{total_accept_correct}%"
        )


# ---------------------------------------------------------------------------- #
# VLM path (unchanged from the published release; kept here for parity)
# ---------------------------------------------------------------------------- #


def vlm(model_str, input_str, regex):
    """Run a VLM on the image(s) and ask it to accept/reject vs the regex."""
    import ollama  # lazy: keep ollama optional for the NSFA-only path

    if type(input_str) == str:
        p = "an input image"
        input_str = [input_str]
    else:
        p = "a sequence of input images"

    prompt_1 = (
        f"You will be provided {p}. Contained in each image will be a letter "
        f"of the English alphabet. For this problem we ignore case, so we "
        f"have only 26 classes. For each image in the sequence, classify it "
        f"as a letter in the alphabet and then join all of your predictions "
        f"together to make a string. Remember that only lowercase letters "
        f"are allowed. You are also given the regular expression {regex}. "
        f"Now, tell me \"accept\" if the string that you read should be "
        f"accepted by the regular expression and \"reject\" if not. I expect "
        f"the output in the form <predicted string, accept/reject> and ONLY "
        f"in this form. You will be marked incorrect if you provide any "
        f"other outputs or it the output is not in the desired format."
    )

    start = time.time()
    res = ollama.chat(
        model=model_str,
        messages=[
            {
                "role": "user",
                "content": prompt_1,
                "images": input_str,
            }
        ],
    )
    elapsed = time.time() - start
    return res["message"]["content"], elapsed


def get_vlm_output_sequence() -> None:
    """Run all experiments with the VLMs given a sequence of images."""
    sequence_fp = os.path.join(
        os.getcwd(), "examples", "regex", "data", "vlm", "sequence"
    )
    models = ["llava-llama3", "llava:7b", "moondream", "bakllava"]

    for experiment_name in os.listdir(sequence_fp):
        print(f'working on {experiment_name.split("__")[0]}')
        labels_fp = os.path.join(
            os.getcwd(),
            "examples",
            "regex",
            "data",
            f"{experiment_name}.txt",
        )
        sequence_exp_fp = os.path.join(sequence_fp, experiment_name)
        regex = experiment_name.split("__")[1]

        with open(labels_fp, "r") as labels:
            samples_fp = sorted(
                os.listdir(sequence_exp_fp),
                key=lambda x: int(x.split("_")[1]),
            )
            for model in models:
                model_results_dir = os.path.join(
                    "examples",
                    "regex",
                    "results",
                    "vlm",
                    "sequence",
                    model,
                )
                if not os.path.isdir(model_results_dir):
                    os.mkdir(model_results_dir)
                results_fp = os.path.join(
                    model_results_dir, f"{experiment_name}.txt"
                )
                with open(results_fp, "a+") as r:
                    for sample_num, sample_fp in enumerate(samples_fp):
                        full_sample_fp = os.path.join(sequence_exp_fp, sample_fp)
                        full_sample_fps = sorted(
                            os.path.join(full_sample_fp, sfp)
                            for sfp in os.listdir(full_sample_fp)
                        )
                        label_line = labels.readline()
                        if label_line == "":
                            continue
                        true_string, accept = label_line.split(",")
                        accept = bool(int(accept))
                        res1, time1 = vlm(model, full_sample_fps, regex)
                        print(
                            f"#{sample_num} -> {true_string}="
                            f"{'accept' if accept else 'reject'} | "
                            f"{res1} : {time1}"
                        )
                        r.write(
                            f"#{sample_num} -> {true_string}="
                            f"{'accept' if accept else 'reject'} | "
                            f"{res1} : {time1}\n"
                        )


if __name__ == "__main__":
    get_na_output()
    get_vlm_output_sequence()
