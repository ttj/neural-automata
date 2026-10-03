"""Run the NSPDA arithmetic evaluation experiment (no VLM).

This is the neurosymbolic-only entry point used to verify the
refactored package against Table 3 of the NeuS 2025 paper (NA
column). The VLM comparison is unaffected by the refactor and is not
re-run; call `examples.simple_math_vlm_comp.simple_math_vlm_comp.get_vlm_output`
or `get_vlm_output_long_expression` directly if needed.

Invoke from the repository root:

    ~/miniconda3/envs/neural-automata/bin/python \
        -m examples.simple_math_vlm_comp.run_experiment

The script appends to per-operand-count `results.txt` files under
`examples/simple_math_vlm_comp/results/na2/`. Delete that directory
(or back it up under a different name) before running if you want
fresh totals.
"""

from examples.simple_math_vlm_comp.simple_math_vlm_comp import get_na_output


NUM_SAMPLES_PER_COUNT = 100
OPERAND_COUNTS = range(2, 11)


if __name__ == "__main__":
    for num_operands in OPERAND_COUNTS:
        get_na_output(NUM_SAMPLES_PER_COUNT, num_operands=num_operands)
