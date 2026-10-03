"""Image-based arithmetic evaluation experiment (Table 3 of NeuS 2025).

The published release classified each image with a CNN and called a
Python interpreter (`evaluate_expression`) directly -- no PDA in the
path. The refactored version runs the symbol stream through a real
DPDA (constructed by `build_arithmetic_dpda`) wrapped as an
`NSPDA`. The DPDA validates that the predicted string is a
well-formed alternating arithmetic expression terminated by `'#'`;
`evaluate_expression` then computes the numeric value of the
*validated* string. This matches the paper's description of the
NSPDA more faithfully -- the stack records the expression, a separate
component evaluates.

Notation mapping:
    images   -- Tensor(T, 1, 28, 28); a sample-sequence of digit/operator images
    NSPDA    -- composition (PushdownAutomaton, PerceptionAdapter)
"""

import os
import signal
import time
from functools import lru_cache

import numpy as np
import torch

from nsa import NSPDA, PerceptionAdapter

from examples.simple_math_vlm_comp.arithmetic_dpda import (
    END_MARKER,
    build_arithmetic_dpda,
)
from examples.simple_math_vlm_comp.networks.cnn import CNN
from examples.simple_math_vlm_comp.setup_experiment import evaluate_expression

# `ollama` is only used by the VLM comparison functions and is
#   imported lazily inside `vlm()` / `vlm_sequence()`; the NSPDA path
#   has no ollama dependency, so a clean install without ollama can
#   still run the neurosymbolic experiment and the test suite.


timed_out = False


def _alarm_handler(signum, frame):  # noqa: ARG001
    global timed_out
    timed_out = True
    raise TimeoutError("Operation timed out")


signal.signal(signal.SIGALRM, _alarm_handler)


# ---------------------------------------------------------------------------- #
# NSPDA path
# ---------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _hdo_cnn() -> CNN:
    """Load the Handwritten Digits & Operators 14-class CNN once."""
    model_fp = os.path.join(
        os.getcwd(),
        "examples",
        "simple_math_vlm_comp",
        "models",
        "torch",
        "HDO.pth",
    )
    model = CNN()
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    return model


@lru_cache(maxsize=1)
def _arithmetic_nspda() -> NSPDA:
    """Build the NSPDA: arithmetic DPDA + cached CNN."""
    pda = build_arithmetic_dpda()
    perception = PerceptionAdapter(network=_hdo_cnn(), n_alphabet=14)
    return NSPDA(pda, perception, end_marker=END_MARKER)


def neurosymbolic_pda(
    images: torch.Tensor,
) -> tuple[str, bool, object, float]:
    """Run a sequence of digit/operator images through the NSPDA.

    Args:
        images: Tensor of shape `(T, 1, 28, 28)`. Each element is one
            image of a digit or operator.

    Returns:
        Tuple `(predicted_string, accepted, value, elapsed_seconds)`.
        `value` is the numeric result of `evaluate_expression` on the
        predicted string when `accepted` is True; otherwise `-1` (the
        same sentinel `evaluate_expression` uses for malformed input).
    """
    nspda = _arithmetic_nspda()
    start = time.time()
    accepted, predicted_str = nspda.accept(images)
    if accepted:
        value = evaluate_expression(list(predicted_str))
    else:
        value = -1
    elapsed = time.time() - start
    return predicted_str, accepted, value, elapsed


def get_na_output(num_samples: int, num_operands: int) -> None:
    """Run the NSPDA over every sample at a given operand count.

    Preserves the output schema of the published release so existing
    `analysis.py` parsing continues to work: each line is
    `#<idx> -> <true_expr>=<true_value> | <value>: <elapsed>`.
    """
    data_fp = "examples/simple_math_vlm_comp/data"
    na_fp = os.path.join(data_fp, "na", str(num_operands))
    results_fp = os.path.join(
        "examples/simple_math_vlm_comp/results/na2", str(num_operands)
    )
    labels_fp = os.path.join(data_fp, f"{num_operands}_labels.txt")

    if not os.path.isdir(results_fp):
        os.makedirs(results_fp, exist_ok=True)

    samples = []
    with open(labels_fp, "r") as f:
        for sample_num in range(num_samples):
            sample_fp = next(
                (
                    fp
                    for fp in os.listdir(na_fp)
                    if fp.startswith(f"sample_{sample_num}_")
                ),
                None,
            )
            if sample_fp is None:
                continue
            sample_fp = os.path.join(na_fp, sample_fp)
            line = f.readline()
            samples.append(
                [sample_fp] + [sample.strip() for sample in line.split(",")]
            )

    results_file = os.path.join(results_fp, "results.txt")
    for sample_num, (sample_fp, true_expression, true_solution) in enumerate(
        samples
    ):
        sample_data = np.load(sample_fp)
        images = (
            torch.from_numpy(sample_data).unsqueeze(1).float()
        )  # (T, 1, 28, 28)
        _predicted, _accepted, value, elapsed = neurosymbolic_pda(images)
        with open(results_file, "a+") as f:
            print(
                f"#{sample_num} -> {true_expression}="
                f"{true_solution} | {value}: {elapsed}"
            )
            f.write(
                f"#{sample_num} -> {true_expression}="
                f"{true_solution} | {value}: {elapsed}\n"
            )


# ---------------------------------------------------------------------------- #
# VLM path (unchanged from the published release; kept here for parity)
# ---------------------------------------------------------------------------- #


def vlm(model_str: str, sample_fp: str, true_expression: str):
    import ollama  # lazy: keep ollama optional for the NSPDA-only path

    start = time.time()
    res = ollama.chat(
        model=model_str,
        messages=[
            {
                "role": "user",
                "content": (
                    "Solve the mathematical expression in the image. The "
                    "output must be in the format <numerical expression in "
                    "image>=<solution>. DO NOT give me any other output. "
                    "Also, DO NOT use LaTeX. These are simple expressions "
                    "and can be expressed simply."
                ),
                "images": [sample_fp],
            }
        ],
    )
    elapsed = time.time() - start
    return res["message"]["content"], elapsed


def vlm_sequence(model_str: str, sample_fp: str):
    import ollama  # lazy: keep ollama optional for the NSPDA-only path

    image_fps = [
        os.path.join(sample_fp, image_fp)
        for image_fp in os.listdir(sample_fp)
    ]
    input_prompt = """
        You will be provided a sequence of input images. Contained in each image will be
        either a digit (0-9) or an operator (+, -, /, *). Read these together to make an
        arithmetic expression. You need to solve these left to right, i.e. keep a running
        total of the value as you read in each image to make valid arithmetic expressions. For
        example, given a sequence of images like ['5', '+', '1', '*', '2], you would first
        read the valid expression '5+1' and evaluate it to 6. Then, you would read the
        next operator and operand to get the valid expression '6*2', which is evaluated
        to 12.

        The output must be in the format <numerical expression in image>=<solution>.
        DO NOT give me any other output. Also, DO NOT use LaTeX. These are simple expressions
        and can be expressed without the use of LaTeX.'
    """
    start = time.time()
    res = ollama.chat(
        model=model_str,
        messages=[
            {
                "role": "user",
                "content": input_prompt,
                "images": image_fps,
            }
        ],
    )
    elapsed = time.time() - start
    return res["message"]["content"], elapsed


def get_vlm_output(model_str: str) -> None:
    data_fp = "examples/simple_math_vlm_comp/data"
    vlm_fp = os.path.join(data_fp, "vlm")
    labels_fp = os.path.join(data_fp, "labels.txt")
    samples = []
    with open(labels_fp, "r") as f:
        for sample in os.listdir(vlm_fp):
            sample_fp = os.path.join(vlm_fp, sample)
            line = f.readline()
            samples.append(
                [sample_fp] + [sample.strip() for sample in line.split(",")]
            )
    results_fp = os.path.join(data_fp, f"results_{model_str}.txt")
    for sample_num, (sample_fp, true_expression, true_solution) in enumerate(
        samples
    ):
        with open(results_fp, "a") as f:
            res, time_taken = vlm(model_str, sample_fp, true_expression)
            print(
                f"#{sample_num} -> {true_expression}="
                f"{true_solution} | {res}: {time_taken}"
            )
            f.write(
                f"#{sample_num} -> {true_expression}="
                f"{true_solution} | {res}: {time_taken}\n"
            )


def get_vlm_output_long_expression(model_str: str, num_operands: int) -> None:
    data_fp = "examples/simple_math_vlm_comp/data"
    vlm_sequence_fp = os.path.join(data_fp, "vlm", "sequence", str(num_operands))
    results_fp = os.path.join(
        "examples/simple_math_vlm_comp/results/sequence", str(num_operands)
    )
    labels_fp = os.path.join(data_fp, f"{num_operands}_labels.txt")
    samples = []
    with open(labels_fp, "r") as f:
        all_sample_fps = sorted(os.listdir(vlm_sequence_fp))
        for sample in all_sample_fps:
            sample_fp = os.path.join(vlm_sequence_fp, sample)
            line = f.readline()
            samples.append(
                [sample_fp] + [sample.strip() for sample in line.split(",")]
            )

    results_file = os.path.join(results_fp, f"results_{model_str}.txt")
    for sample_num, (sample_fp, true_expression, true_solution) in enumerate(
        samples
    ):
        if sample_num > 24:
            continue
        global timed_out
        timed_out = False
        signal.alarm(180)
        try:
            res, time_taken = vlm_sequence(model_str, sample_fp)
        except TimeoutError:
            signal.alarm(0)
            with open(results_file, "a") as f:
                print(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | timeout"
                )
                f.write(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | timeout: 180\n"
                )
            continue
        except Exception as e:
            signal.alarm(0)
            with open(results_file, "a") as f:
                print(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | error: {e}"
                )
                f.write(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | error: {e}\n"
                )
            continue
        signal.alarm(0)
        with open(results_file, "a") as f:
            if timed_out:
                print(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | timeout"
                )
                f.write(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | timeout: 180\n"
                )
            else:
                print(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | {res}: {time_taken}"
                )
                f.write(
                    f"#{sample_num} -> {true_expression}="
                    f"{true_solution} | {res}: {time_taken}\n"
                )


if __name__ == "__main__":
    models = ["llava-llama3", "llava:7b", "moondream", "bakllava"]
    for model in models:
        get_vlm_output(model)
    for model in models:
        for num_operand in range(3, 11):
            get_vlm_output_long_expression(model, num_operand)
    for num_operands in range(2, 11):
        get_na_output(100, num_operands=num_operands)
