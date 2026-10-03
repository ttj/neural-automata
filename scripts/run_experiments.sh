#!/bin/bash
# Run the neurosymbolic (NSFA / NSPDA) experiments for Table 1 and Table 3
# of the NeuS 2025 paper through the refactored `nsa` package.
#
# VLM comparisons are unaffected by the refactor and are NOT re-run by
# this script. To re-run VLM comparisons, call the VLM functions in
# the example modules directly (and install the `vlm` extra:
# `pip install -e '.[vlm]'`).
#
# Usage (from the repository root or from scripts/):
#     scripts/run_experiments.sh
#     PYTHON=~/miniconda3/envs/neural-automata/bin/python scripts/run_experiments.sh
#
# Or run the individual entry points directly:
#     python -m examples.regex.run_experiment
#     python -m examples.simple_math_vlm_comp.run_experiment
#
# The example scripts append to results files; delete or back up the
# `examples/regex/results/na/` and
# `examples/simple_math_vlm_comp/results/na2/` directories first if
# you want clean totals.

set -euo pipefail

current_dir=$(pwd)
if [[ "$current_dir" == *"/scripts" ]]; then
    cd ..
fi

PYTHON="${PYTHON:-python}"

echo "[run_experiments] Running NSFA regex experiment..."
"$PYTHON" -m examples.regex.run_experiment

echo "[run_experiments] Running NSPDA arithmetic experiment..."
"$PYTHON" -m examples.simple_math_vlm_comp.run_experiment

echo "[run_experiments] Done. Analyze with:"
echo "  $PYTHON examples/regex/analysis.py"
echo "  $PYTHON examples/simple_math_vlm_comp/analysis.py"
