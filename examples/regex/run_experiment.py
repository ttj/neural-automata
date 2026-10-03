"""Run the NSFA regex acceptance experiment (no VLM).

This is the neurosymbolic-only entry point used to verify the
refactored package against Table 1 of the NeuS 2025 paper. The VLM
comparison (LLaVA / LLaMA3, LLaVA-7B, Moondream, BakLLaVA) is
unaffected by the refactor and is not re-run; call
`examples.regex.regex.get_vlm_output_sequence` directly if needed.

Invoke from the repository root:

    ~/miniconda3/envs/neural-automata/bin/python -m examples.regex.run_experiment

The script appends to per-experiment `.txt` files under
`examples/regex/results/na/`. Delete that directory (or back it up
under a different name) before running if you want fresh totals; the
README's reproducibility warning applies.
"""

from examples.regex.regex import get_na_output


if __name__ == "__main__":
    get_na_output()
