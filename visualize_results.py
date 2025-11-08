#!/usr/bin/env python3
"""
Visualize NeuS'25 Paper Results
================================

This script creates comprehensive visualizations of all experimental results
from the paper, including:
1. Regex matching accuracy (NSFA vs VLMs)
2. Arithmetic evaluation accuracy vs expression complexity (NSPDA vs VLMs)
3. Runtime comparisons
4. Detailed performance breakdowns

Author: Neural Automata Team
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def parse_na_arithmetic_results():
    """
    Parse neural automaton results for arithmetic evaluation.

    Returns:
        dict: {num_operands: (accuracy, avg_time)}
    """
    results = {}

    for num_operands in range(2, 11):
        results_fp = f'examples/simple_math_vlm_comp/results/na2/{num_operands}/results.txt'

        if not os.path.exists(results_fp):
            print(f"Warning: Results file not found: {results_fp}")
            continue

        with open(results_fp, 'r') as f:
            correct = 0
            total = 0
            total_time = 0

            for line in f:
                elements = line.split(" ")
                true_expression, true_value = elements[2].split("=")
                predicted_value = elements[4][:-1]
                time_taken = float(elements[5])

                total_time += time_taken

                try:
                    if float(true_value) == float(predicted_value):
                        correct += 1
                except:
                    pass

                total += 1

            if total > 0:
                accuracy = (correct / total) * 100
                avg_time = total_time / total
                results[num_operands] = (accuracy, avg_time)

    return results


def parse_vlm_arithmetic_results():
    """
    Parse VLM results for arithmetic evaluation.

    Returns:
        dict: {model: {num_operands: (accuracy, avg_time)}}
    """
    models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']
    results = {model: {} for model in models}

    for model in models:
        for num_operands in range(2, 11):
            # For 2 operands, results are in stitched_image directory
            if num_operands == 2:
                results_fp = f'examples/simple_math_vlm_comp/results/stitched_image/{num_operands}/results_{model}.txt'
            else:
                results_fp = f'examples/simple_math_vlm_comp/results/sequence/{num_operands}/results_{model}.txt'

            if not os.path.exists(results_fp):
                continue

            with open(results_fp, 'r', encoding='utf-8') as f:
                content = f.read()

            entries = re.split(r'#(\d+) -> ', content)

            correct = 0
            total = 0
            running_time = 0

            for i in range(1, len(entries), 2):
                value = entries[i+1].strip()

                # Handle timeout
                if 'timeout' in value:
                    running_time += 180.0
                    total += 1
                    continue

                try:
                    true_sample, vlm_output = value.split(" | ")
                    true_expression, true_value = true_sample.split("=")
                    vlm_runtime = vlm_output.split(": ")[-1]
                    running_time += float(vlm_runtime)

                    # Check if correct (lenient - if true value appears in output)
                    if true_value in vlm_output:
                        correct += 1

                    total += 1
                except Exception as e:
                    # print(f"Parse error for {model}, {num_operands}: {e}")
                    pass

            if total > 0:
                accuracy = (correct / total) * 100
                avg_time = running_time / total
                results[model][num_operands] = (accuracy, avg_time)

    return results


def plot_arithmetic_accuracy_comparison():
    """Plot accuracy comparison for arithmetic evaluation."""
    print("\nGenerating arithmetic accuracy comparison plot...")

    # Parse results
    na_results = parse_na_arithmetic_results()
    vlm_results = parse_vlm_arithmetic_results()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot neural automaton
    if na_results:
        operands = sorted(na_results.keys())
        accuracies = [na_results[n][0] for n in operands]
        ax.plot(operands, accuracies, 'o-', linewidth=3, markersize=10,
                label='NSPDA (Neural Automaton)', color='#2E7D32', zorder=10)

    # Plot VLMs
    colors = {'llava-llama3': '#1976D2', 'llava:7b': '#D32F2F',
              'moondream': '#F57C00', 'bakllava': '#7B1FA2'}
    markers = {'llava-llama3': 's', 'llava:7b': '^',
               'moondream': 'D', 'bakllava': 'v'}

    for model in vlm_results:
        if vlm_results[model]:
            operands = sorted(vlm_results[model].keys())
            accuracies = [vlm_results[model][n][0] for n in operands]
            ax.plot(operands, accuracies, marker=markers[model], linestyle='--',
                    linewidth=2, markersize=8, label=f'{model} (VLM)',
                    color=colors[model], alpha=0.7)

    ax.set_xlabel('Number of Operands', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Arithmetic Evaluation: NSPDA vs VLMs\nAccuracy as Function of Expression Complexity',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_xticks(range(2, 11))
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    output_path = 'paper_results_arithmetic_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

    return output_path


def plot_arithmetic_runtime_comparison():
    """Plot runtime comparison for arithmetic evaluation."""
    print("\nGenerating arithmetic runtime comparison plot...")

    # Parse results
    na_results = parse_na_arithmetic_results()
    vlm_results = parse_vlm_arithmetic_results()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot neural automaton
    if na_results:
        operands = sorted(na_results.keys())
        times = [na_results[n][1] for n in operands]
        ax.plot(operands, times, 'o-', linewidth=3, markersize=10,
                label='NSPDA (Neural Automaton)', color='#2E7D32', zorder=10)

    # Plot VLMs
    colors = {'llava-llama3': '#1976D2', 'llava:7b': '#D32F2F',
              'moondream': '#F57C00', 'bakllava': '#7B1FA2'}
    markers = {'llava-llama3': 's', 'llava:7b': '^',
               'moondream': 'D', 'bakllava': 'v'}

    for model in vlm_results:
        if vlm_results[model]:
            operands = sorted(vlm_results[model].keys())
            times = [vlm_results[model][n][1] for n in operands]
            ax.plot(operands, times, marker=markers[model], linestyle='--',
                    linewidth=2, markersize=8, label=f'{model} (VLM)',
                    color=colors[model], alpha=0.7)

    ax.set_xlabel('Number of Operands', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Runtime (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Arithmetic Evaluation: Runtime Comparison\nNSPDA vs VLMs',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_xticks(range(2, 11))
    ax.set_yscale('log')

    plt.tight_layout()
    output_path = 'paper_results_arithmetic_runtime.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

    return output_path


def create_results_summary_table():
    """Create a summary table of all results."""
    print("\nGenerating results summary table...")

    # Parse results
    na_results = parse_na_arithmetic_results()
    vlm_results = parse_vlm_arithmetic_results()

    # Create summary
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("NEUS'25 PAPER RESULTS SUMMARY")
    summary_lines.append("="*80)

    # Arithmetic Evaluation Results
    summary_lines.append("\n### ARITHMETIC EVALUATION (NSPDA vs VLMs) ###\n")

    if na_results:
        summary_lines.append("Neural Automaton (NSPDA):")
        summary_lines.append("-" * 60)
        summary_lines.append(f"{'Operands':<12} {'Accuracy':<12} {'Avg Runtime (s)':<20}")
        summary_lines.append("-" * 60)

        for num_ops in sorted(na_results.keys()):
            acc, time = na_results[num_ops]
            summary_lines.append(f"{num_ops:<12} {acc:>6.1f}%      {time:>8.4f}")

        summary_lines.append("")

    # VLM Results
    for model in ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']:
        if model in vlm_results and vlm_results[model]:
            summary_lines.append(f"\n{model.upper()} (VLM):")
            summary_lines.append("-" * 60)
            summary_lines.append(f"{'Operands':<12} {'Accuracy':<12} {'Avg Runtime (s)':<20}")
            summary_lines.append("-" * 60)

            for num_ops in sorted(vlm_results[model].keys()):
                acc, time = vlm_results[model][num_ops]
                summary_lines.append(f"{num_ops:<12} {acc:>6.1f}%      {time:>8.4f}")

    summary_lines.append("\n" + "="*80)
    summary_lines.append("KEY FINDINGS:")
    summary_lines.append("="*80)

    # Calculate key statistics
    if na_results and 2 in na_results:
        na_2op_acc = na_results[2][0]
        summary_lines.append(f"- NSPDA achieves {na_2op_acc:.1f}% accuracy with 2 operands")

        if 10 in na_results:
            na_10op_acc = na_results[10][0]
            summary_lines.append(f"- NSPDA achieves {na_10op_acc:.1f}% accuracy with 10 operands")

    # Check VLM performance
    vlm_best_2op = 0
    for model in vlm_results:
        if 2 in vlm_results[model]:
            vlm_best_2op = max(vlm_best_2op, vlm_results[model][2][0])

    if vlm_best_2op > 0:
        summary_lines.append(f"- Best VLM achieves {vlm_best_2op:.1f}% accuracy with 2 operands")

    summary_lines.append("\n" + "="*80)

    # Save summary
    output_path = 'paper_results_summary.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"Saved to: {output_path}")

    # Also print to console
    print("\n" + '\n'.join(summary_lines))

    return output_path


def plot_accuracy_bar_comparison():
    """Create a bar chart comparing accuracies at key complexity levels."""
    print("\nGenerating accuracy bar comparison plot...")

    # Parse results
    na_results = parse_na_arithmetic_results()
    vlm_results = parse_vlm_arithmetic_results()

    # Focus on 2, 5, and 10 operands
    focus_operands = [2, 5, 10]

    # Prepare data
    models = ['NSPDA'] + ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']
    colors_map = {
        'NSPDA': '#2E7D32',
        'llava-llama3': '#1976D2',
        'llava:7b': '#D32F2F',
        'moondream': '#F57C00',
        'bakllava': '#7B1FA2'
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for idx, num_ops in enumerate(focus_operands):
        ax = axes[idx]

        model_names = []
        accuracies = []
        colors = []

        # Add NSPDA
        if num_ops in na_results:
            model_names.append('NSPDA')
            accuracies.append(na_results[num_ops][0])
            colors.append(colors_map['NSPDA'])

        # Add VLMs
        for model in ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']:
            if model in vlm_results and num_ops in vlm_results[model]:
                model_names.append(model)
                accuracies.append(vlm_results[model][num_ops][0])
                colors.append(colors_map[model])

        # Create bar chart
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{num_ops} Operands', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.suptitle('Arithmetic Evaluation Accuracy Comparison\nNSPDA vs VLMs at Different Complexity Levels',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = 'paper_results_arithmetic_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

    return output_path


def create_all_visualizations():
    """Create all visualizations from the paper."""
    print("\n" + "="*70)
    print("GENERATING ALL PAPER VISUALIZATIONS")
    print("="*70)

    outputs = []

    # 1. Accuracy comparison
    try:
        outputs.append(plot_arithmetic_accuracy_comparison())
    except Exception as e:
        print(f"Error generating accuracy plot: {e}")

    # 2. Runtime comparison
    try:
        outputs.append(plot_arithmetic_runtime_comparison())
    except Exception as e:
        print(f"Error generating runtime plot: {e}")

    # 3. Bar comparison
    try:
        outputs.append(plot_accuracy_bar_comparison())
    except Exception as e:
        print(f"Error generating bar plot: {e}")

    # 4. Summary table
    try:
        outputs.append(create_results_summary_table())
    except Exception as e:
        print(f"Error generating summary: {e}")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    for output in outputs:
        if output:
            print(f"  - {output}")

    return outputs


if __name__ == "__main__":
    create_all_visualizations()
