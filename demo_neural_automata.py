#!/usr/bin/env python3
"""
Comprehensive Demo: Neural Automata Execution
==============================================

This script demonstrates how to execute the neurosymbolic automata
from the NeuS'25 paper for both tasks:
1. Image-based string acceptance (regex matching)
2. Image-based arithmetic evaluation

Author: Neural Automata Team
Paper: "Neurosymbolic Finite and Pushdown Automata: Improved Multimodal
        Reasoning versus Vision Language Models (VLMs)"
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.regex.networks.letter_cnn import LetterCNN
from examples.regex.setup_experiment import matches_regex
from examples.simple_math.networks.digit import DigitCNN
from examples.simple_math.networks.operator import OperatorCNN
from examples.simple_math.networks.modeswitch import ModeSwitchCNN
from examples.simple_math_vlm_comp.setup_experiment import evaluate_expression

OP_MAP = {10: '+', 11: '-', 12: '*', 13: '%'}


def demo_regex_automaton(sample_num=0):
    """
    Demonstrate the Neurosymbolic Finite Automaton (NSFA) for regex matching.

    This automaton:
    1. Uses a CNN to classify each letter image
    2. Builds a string from the predictions
    3. Tests the string against a regular expression

    Args:
        sample_num: Which sample to test (0-9)
    """
    print("\n" + "="*70)
    print("DEMO: Neurosymbolic Finite Automaton (NSFA) - Regex Matching")
    print("="*70)

    # Load the trained letter recognition model
    base_fp = os.path.join(os.getcwd(), 'examples', 'regex')
    model_fp = os.path.join(base_fp, 'models', 'model.pth')
    model = LetterCNN(num_classes=26)
    model.load_state_dict(torch.load(model_fp))
    model.eval()

    # Get a sample from the data
    data_fp = os.path.join(base_fp, 'data', 'na')
    experiments = os.listdir(data_fp)

    # Use the first experiment
    experiment_name = experiments[0]
    experiment_fp = os.path.join(data_fp, experiment_name)

    # Get regex from experiment name
    regex = experiment_name.split("__")[1]
    print(f"\nRegular Expression: {regex}")

    # Get labels
    labels_fp = os.path.join(base_fp, 'data', f'{experiment_name}.txt')
    with open(labels_fp, 'r') as f:
        labels = f.readlines()

    # Get sample
    samples = sorted(os.listdir(experiment_fp), key=lambda x: int(x.split("_")[1]))
    sample_fp = os.path.join(experiment_fp, samples[sample_num])

    true_string, accept = labels[sample_num].strip().split(",")
    accept = bool(int(accept))

    print(f"Sample #{sample_num}")
    print(f"True String: {true_string}")
    print(f"Should Accept: {accept}")

    # Load the image tensor
    images = torch.load(sample_fp)

    # Predict each letter
    predicted_chars = []
    print(f"\nProcessing {images.shape[0]} letter images...")

    for i in range(images.shape[0]):
        image = images[i]
        outputs = model(image)
        pred = torch.argmax(outputs).item()
        pred_char = chr(ord('a') + pred)
        predicted_chars.append(pred_char)
        print(f"  Letter {i+1}: predicted '{pred_char}'")

    # Build predicted string
    predicted_str = ''.join(predicted_chars)
    print(f"\nPredicted String: {predicted_str}")

    # Test against regex
    na_accept = matches_regex(regex, predicted_str)
    print(f"NSFA Decision: {'ACCEPT' if na_accept else 'REJECT'}")

    # Check correctness
    string_correct = (true_string == predicted_str)
    decision_correct = (accept == na_accept)

    print(f"\nResults:")
    print(f"  String Classification: {'✓ Correct' if string_correct else '✗ Incorrect'}")
    print(f"  Accept/Reject Decision: {'✓ Correct' if decision_correct else '✗ Incorrect'}")

    return predicted_str, na_accept, string_correct, decision_correct


def demo_arithmetic_automaton(num_operands=2, sample_num=0):
    """
    Demonstrate the Neurosymbolic Pushdown Automaton (NSPDA) for arithmetic.

    This automaton:
    1. Uses a CNN to classify each digit/operator image
    2. Builds an arithmetic expression
    3. Evaluates the expression using a pushdown automaton

    Args:
        num_operands: Number of operands (2-10)
        sample_num: Which sample to test (0-99)
    """
    print("\n" + "="*70)
    print("DEMO: Neurosymbolic Pushdown Automaton (NSPDA) - Arithmetic")
    print("="*70)

    base_fp = os.path.join(os.getcwd(), 'examples', 'simple_math_vlm_comp')

    # Load the models
    model_fp = os.path.join(base_fp, 'models', 'torch', 'HDO.pth')
    from examples.simple_math_vlm_comp.networks.cnn import CNN

    nn = CNN()
    nn.load_state_dict(torch.load(model_fp))
    nn.eval()

    print(f"\nTesting with {num_operands} operands")

    # Get data
    data_fp = os.path.join(base_fp, 'data')
    na_fp = os.path.join(data_fp, 'na', str(num_operands))
    labels_fp = os.path.join(data_fp, f'{num_operands}_labels.txt')

    # Get sample
    with open(labels_fp, 'r') as f:
        lines = f.readlines()

    true_expression, true_solution = lines[sample_num].strip().split(',')

    # Find the sample file
    samples = os.listdir(na_fp)
    sample_fp = next((fp for fp in samples if fp.startswith(f'sample_{sample_num}_')), None)
    sample_fp = os.path.join(na_fp, sample_fp)

    print(f"Sample #{sample_num}")
    print(f"True Expression: {true_expression}")
    print(f"True Solution: {true_solution}")

    # Load the sample data
    sample_data = np.load(sample_fp)
    arithmetic_expression = [
        torch.from_numpy(sample_data[i]).unsqueeze(0).float()
        for i in range(sample_data.shape[0])
    ]

    # Predict each element
    predicted_expression = []
    print(f"\nProcessing {len(arithmetic_expression)} images...")

    for i, element in enumerate(arithmetic_expression):
        logits = nn(element)
        pred = torch.argmax(logits).item()

        # Convert prediction to digit or operator
        if pred > 9:
            pred_str = OP_MAP[pred]
            print(f"  Element {i+1}: predicted '{pred_str}' (operator)")
        else:
            pred_str = str(pred)
            print(f"  Element {i+1}: predicted '{pred_str}' (digit)")

        predicted_expression.append(pred_str)

    # Evaluate the expression
    result = evaluate_expression(predicted_expression)

    pred_expr_str = ''.join(predicted_expression)
    print(f"\nPredicted Expression: {pred_expr_str}")
    print(f"Predicted Solution: {result}")
    print(f"NSPDA Output: {result}")

    # Check correctness
    try:
        correct = (float(result) == float(true_solution))
        print(f"\nResult: {'✓ Correct' if correct else '✗ Incorrect'}")
    except:
        correct = False
        print(f"\nResult: ✗ Incorrect (evaluation error)")

    return pred_expr_str, result, correct


def visualize_sample_images(num_operands=2, sample_num=0):
    """Visualize the actual images used in an arithmetic expression."""
    print("\n" + "="*70)
    print("VISUALIZING SAMPLE IMAGES")
    print("="*70)

    base_fp = os.path.join(os.getcwd(), 'examples', 'simple_math_vlm_comp')
    data_fp = os.path.join(base_fp, 'data')
    na_fp = os.path.join(data_fp, 'na', str(num_operands))

    # Get sample
    samples = os.listdir(na_fp)
    sample_fp = next((fp for fp in samples if fp.startswith(f'sample_{sample_num}_')), None)
    sample_fp = os.path.join(na_fp, sample_fp)

    # Load the sample data
    sample_data = np.load(sample_fp)

    # Create visualization
    num_images = sample_data.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        image = sample_data[i].squeeze()
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Element {i+1}')

    plt.suptitle(f'Arithmetic Expression (Sample {sample_num}, {num_operands} operands)')
    plt.tight_layout()

    output_fp = f'visualization_sample_{sample_num}_{num_operands}ops.png'
    plt.savefig(output_fp, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_fp}")
    plt.close()

    return output_fp


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("NEURAL AUTOMATA DEMONSTRATION")
    print("Paper: Neurosymbolic Finite and Pushdown Automata (NeuS'25)")
    print("="*70)

    # Demo 1: Regex matching
    print("\n\n### DEMONSTRATION 1: Regex String Acceptance ###")
    demo_regex_automaton(sample_num=0)

    # Demo 2: Arithmetic evaluation
    print("\n\n### DEMONSTRATION 2: Arithmetic Evaluation ###")
    demo_arithmetic_automaton(num_operands=2, sample_num=0)

    # Demo 3: More complex arithmetic
    print("\n\n### DEMONSTRATION 3: Complex Arithmetic (5 operands) ###")
    demo_arithmetic_automaton(num_operands=5, sample_num=0)

    # Demo 4: Visualization
    print("\n\n### DEMONSTRATION 4: Visualizing Input Images ###")
    visualize_sample_images(num_operands=3, sample_num=0)

    print("\n\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  - Run generate_examples.py to create new random samples")
    print("  - Run visualize_results.py to see paper results")
    print("  - See NEURAL_AUTOMATA_GUIDE.md for detailed documentation")


if __name__ == "__main__":
    main()
