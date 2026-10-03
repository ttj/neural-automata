"""Tutorial-style NSPDA arithmetic example.

A stripped-down walkthrough of the full
`examples/simple_math_vlm_comp/` experiment from the NeuS 2025 paper.
Read this first to see the moving parts -- DPDA construction,
PerceptionAdapter wrapping, NSPDA composition, accept / evaluate --
before reading the production code that runs Table 3.

This example loads four pre-baked digit-and-operator image sequences
from `examples/simple_math/data/` (one each for `+`, `-`, `*`, `%`),
passes them through the same NSPDA used by the production
experiment, and prints what the framework predicted, whether the
DPDA accepted the expression, and the numeric value.

Run from the repository root:

    python -m examples.simple_math.simple_math

Prereq: the HDO CNN checkpoint at
`examples/simple_math_vlm_comp/models/torch/HDO.pth` (shipped with
the repo).
"""

import os

import numpy as np
import torch

from nsa import NSPDA, PerceptionAdapter
from examples.simple_math_vlm_comp.arithmetic_dpda import (
    END_MARKER,
    build_arithmetic_dpda,
)
from examples.simple_math_vlm_comp.networks.cnn import CNN
from examples.simple_math_vlm_comp.setup_experiment import evaluate_expression


def _load_hdo_cnn() -> CNN:
    """Load the published 14-class digit+operator CNN in eval mode."""
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
    model.eval()  # disables Dropout (none in this CNN, but safety pattern)
    return model


def build_nspda() -> NSPDA:
    """Compose the arithmetic DPDA + HDO CNN into an NSPDA.

    Returns:
        `NSPDA` with `end_marker='#'`. Discrete inference only;
        `forward(...)` is deferred to Half B of the project plan.
    """
    pda = build_arithmetic_dpda()
    cnn = _load_hdo_cnn()
    perception = PerceptionAdapter(network=cnn, n_alphabet=14)
    return NSPDA(pda, perception, end_marker=END_MARKER)


def _load_example_sequence(name: str) -> torch.Tensor:
    """Load one of the four bundled `<op>_example.npy` files.

    Each `.npy` is an iterable of `(image, label)` pairs; `image` is a
    28x28 numpy array. Returns a torch tensor of shape `(T, 1, 28, 28)`
    matching the `PerceptionAdapter` input convention.
    """
    fp = os.path.join(
        os.getcwd(), "examples", "simple_math", "data", f"{name}.npy"
    )
    data = np.load(fp, allow_pickle=True)
    images = torch.stack(
        [
            torch.from_numpy(sample).unsqueeze(0).float()
            for sample, _label in data
        ]
    )
    return images


def run(name: str, nspda: NSPDA) -> None:
    """Run one bundled example and print the trace."""
    images = _load_example_sequence(name)
    accepted, predicted = nspda.accept(images)
    value: object
    if accepted:
        value = evaluate_expression(list(predicted))
    else:
        value = "rejected by DPDA"
    print(
        f"{name:<14}  predicted={predicted!r:<25}  "
        f"accepted={accepted}  value={value}"
    )


def main() -> None:
    nspda = build_nspda()
    print(f"PDA states:     {set(nspda.pda.dpda.states)}")
    print(f"PDA initial:    {nspda.pda.q0}")
    print(f"PDA accepting:  {nspda.pda.F}")
    print(f"Perception alphabet (14): {set(nspda.pda.alphabet) - {END_MARKER}}")
    print(f"End marker (PDA-only):    {END_MARKER!r}\n")

    for name in [
        "add_example",
        "sub_example",
        "mult_example",
        "div_example",
    ]:
        run(name, nspda)


if __name__ == "__main__":
    main()
