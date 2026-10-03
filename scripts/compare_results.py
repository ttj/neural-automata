"""Side-by-side comparison of refactored NSFA / NSPDA results vs published.

Reads both the refactored results (under `results/na/` and
`results/na2/`) and the backup of the published v0.1 results (under
`results/na_published_v0.1/` and `results/na2_published_v0.1/`) and
prints a per-row comparison against the agreed tolerances from
`.claude/plans/2026-05-12_nsa-refactor-and-e2e-scaffold.md`
Section 4: +/- 1/10 on Table 1 Task Accuracy and +/- 2% on Table 3 NA.

Run from the repository root:

    ~/miniconda3/envs/neural-automata/bin/python scripts/compare_results.py
"""

import os
import re
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

REGEX_NEW = REPO / "examples" / "regex" / "results" / "na"
REGEX_OLD = REPO / "examples" / "regex" / "results" / "na_published_v0.1"
ARITH_NEW = REPO / "examples" / "simple_math_vlm_comp" / "results" / "na2"
ARITH_OLD = (
    REPO
    / "examples"
    / "simple_math_vlm_comp"
    / "results"
    / "na2_published_v0.1"
)

# Tolerance bounds confirmed in the plan, section 4.
REGEX_TA_TOL = 10.0     # +/- 1/10 = +/- 10 percentage points
ARITH_ACC_TOL = 2.0     # +/- 2 percentage points


def _parse_regex_summary(filepath: Path) -> tuple[float, float] | None:
    """Return (CA, TA) percentages from the trailing summary line, or None."""
    with open(filepath) as f:
        for line in f:
            m = re.search(
                r"String Classification Accuracy:\s*([\d.]+)%,\s*"
                r"Task Accuracy:\s*([\d.]+)%",
                line,
            )
            if m:
                return float(m.group(1)), float(m.group(2))
    return None


def compare_regex(new_dir: Path, old_dir: Path) -> int:
    """Print per-experiment NSFA-vs-published comparison; return n_out_of_tol."""
    print(f"{'Regex':<48} {'CA_new':>8} {'CA_old':>8} {'dCA':>7}  "
          f"{'TA_new':>8} {'TA_old':>8} {'dTA':>7}  status")
    print("-" * 110)
    n_out_of_tol = 0
    files = sorted(
        fn
        for fn in os.listdir(new_dir)
        if fn.startswith("experiment_") and fn.endswith(".txt")
    )
    for fn in files:
        regex = fn[len("experiment_"):].split("__", 1)[1][:-4]
        new_path = new_dir / fn
        old_path = old_dir / fn
        new = _parse_regex_summary(new_path) if new_path.exists() else None
        old = _parse_regex_summary(old_path) if old_path.exists() else None
        if new is None or old is None:
            print(f"{regex:<48}  missing summary line in new or old")
            n_out_of_tol += 1
            continue
        ca_new, ta_new = new
        ca_old, ta_old = old
        d_ca = ca_new - ca_old
        d_ta = ta_new - ta_old
        within_tol = abs(d_ta) <= REGEX_TA_TOL
        status = "OK" if within_tol else "OUT-OF-TOL"
        if not within_tol:
            n_out_of_tol += 1
        print(
            f"{regex:<48} {ca_new:>7.1f}% {ca_old:>7.1f}% {d_ca:>+6.1f}%  "
            f"{ta_new:>7.1f}% {ta_old:>7.1f}% {d_ta:>+6.1f}%  {status}"
        )
    return n_out_of_tol


def _parse_arith_results(filepath: Path) -> tuple[float, float, int] | None:
    """Return (accuracy_percent, mean_time_s, n_samples) from a results.txt."""
    correct = 0
    total = 0
    total_time = 0.0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # line format: '#N -> <expr>=<true_value> | <pred_value>: <elapsed>'
            parts = line.split(" ")
            try:
                _idx, _arrow, expr_val, _bar, pred_val_colon, elapsed_s = parts[:6]
                _expr, true_val = expr_val.split("=")
                pred_val = pred_val_colon[:-1]  # strip trailing ':'
                total_time += float(elapsed_s)
                if float(true_val) == float(pred_val):
                    correct += 1
                total += 1
            except (ValueError, IndexError):
                continue
    if total == 0:
        return None
    return 100.0 * correct / total, total_time / total, total


def compare_arith(new_dir: Path, old_dir: Path) -> int:
    """Print per-operand-count NSPDA-vs-published comparison."""
    print(f"{'# Ops':<7} {'acc_new':>9} {'acc_old':>9} {'dAcc':>8}  "
          f"{'t_new':>10} {'t_old':>10}  status")
    print("-" * 80)
    n_out_of_tol = 0
    for n in range(2, 11):
        new_fp = new_dir / str(n) / "results.txt"
        old_fp = old_dir / str(n) / "results.txt"
        new = _parse_arith_results(new_fp) if new_fp.exists() else None
        old = _parse_arith_results(old_fp) if old_fp.exists() else None
        if new is None or old is None:
            print(f"{n:<7}  missing")
            n_out_of_tol += 1
            continue
        acc_new, t_new, _ = new
        acc_old, t_old, _ = old
        d_acc = acc_new - acc_old
        within_tol = abs(d_acc) <= ARITH_ACC_TOL
        status = "OK" if within_tol else "OUT-OF-TOL"
        if not within_tol:
            n_out_of_tol += 1
        print(
            f"{n:<7} {acc_new:>8.1f}% {acc_old:>8.1f}% {d_acc:>+7.1f}%  "
            f"{t_new:>9.4f}s {t_old:>9.4f}s  {status}"
        )
    return n_out_of_tol


if __name__ == "__main__":
    print("=" * 110)
    print("REGEX  (Table 1 TA -- tolerance +/- 1/10 = +/- 10 percentage points)")
    print("=" * 110)
    n_regex_out = compare_regex(REGEX_NEW, REGEX_OLD)

    print()
    print("=" * 110)
    print("ARITHMETIC  (Table 3 NA column -- tolerance +/- 2 percentage points)")
    print("=" * 110)
    n_arith_out = compare_arith(ARITH_NEW, ARITH_OLD)

    print()
    print("-" * 110)
    n_total = n_regex_out + n_arith_out
    if n_total == 0:
        print("All rows within tolerance.")
    else:
        print(f"{n_total} row(s) out of tolerance (regex: {n_regex_out}, "
              f"arithmetic: {n_arith_out}).")
