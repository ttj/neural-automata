# Neural Automata: Complete Guide

This guide provides comprehensive instructions for reproducing the NeuS'25 paper results and executing neural automata for multimodal reasoning tasks.

**Paper**: *Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs)*

**Authors**: Samuel Sasaki, Diego Manzanas Lopez, and Taylor T. Johnson

**Conference**: NeuS'25 (2nd International Conference on Neuro-symbolic Systems)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Running Demonstrations](#running-demonstrations)
5. [Generating New Examples](#generating-new-examples)
6. [Visualizing Paper Results](#visualizing-paper-results)
7. [Understanding the Neural Automata](#understanding-the-neural-automata)
8. [Reproducing Full Paper Results](#reproducing-full-paper-results)
9. [Project Structure](#project-structure)
10. [Citation](#citation)

---

## Quick Start

```bash
# 1. Run demonstrations of the neural automata
python demo_neural_automata.py

# 2. Generate new random examples
python generate_examples.py

# 3. Visualize all paper results
python visualize_results.py
```

---

## Overview

This repository implements two types of neurosymbolic automata:

### 1. Neurosymbolic Finite Automaton (NSFA)
- **Task**: Image-based string acceptance (regex matching)
- **How it works**:
  1. CNN classifies each letter image → builds string
  2. Tests string against regular expression
  3. Returns accept/reject decision

### 2. Neurosymbolic Pushdown Automaton (NSPDA)
- **Task**: Image-based arithmetic evaluation
- **How it works**:
  1. CNN classifies each digit/operator image
  2. Builds arithmetic expression
  3. Evaluates using pushdown automaton (left-to-right)
  4. Returns numerical result

Both automata **significantly outperform** state-of-the-art Vision Language Models (VLMs) on these tasks!

---

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install torch torchvision numpy matplotlib pillow
```

### Optional: For VLM Comparison

```bash
# Install Ollama for running VLMs locally
# (only needed if reproducing VLM experiments)
pip install ollama
```

### Dataset (for generating arithmetic examples)

Download the **Handwritten Digits and Operators** dataset:
- Source: https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/data
- Place `training.npy` in: `data/handwritten-digits-and-operators/`

---

## Running Demonstrations

### Demo 1: Execute Neural Automata

Run the comprehensive demonstration script:

```bash
python demo_neural_automata.py
```

This will demonstrate:
- ✓ Regex string acceptance (NSFA)
- ✓ Simple arithmetic (2 operands)
- ✓ Complex arithmetic (5 operands)
- ✓ Visualization of input images

**Example Output**:
```
=======================================================================
DEMO: Neurosymbolic Finite Automaton (NSFA) - Regex Matching
=======================================================================

Regular Expression: a*b+
Sample #0
True String: aabbb
Should Accept: True

Processing 5 letter images...
  Letter 1: predicted 'a'
  Letter 2: predicted 'a'
  Letter 3: predicted 'b'
  Letter 4: predicted 'b'
  Letter 5: predicted 'b'

Predicted String: aabbb
NSFA Decision: ACCEPT

Results:
  String Classification: ✓ Correct
  Accept/Reject Decision: ✓ Correct
```

### Understanding the Output

Each demonstration shows:
1. **Input**: The test sample and ground truth
2. **Processing**: How the neural network classifies each image
3. **Prediction**: The automaton's final decision
4. **Verification**: Correctness check against ground truth

---

## Generating New Examples

### Generate Random String Examples

```python
from generate_examples import StringImageGenerator

generator = StringImageGenerator()

# Generate a random string
random_string = generator.generate_random_string(length=6)
print(f"Generated: '{random_string}'")

# Convert to images and save
generator.save_string_sample(random_string, 'my_string.pt')
```

### Generate Random Arithmetic Examples

```python
from generate_examples import ArithmeticImageGenerator

generator = ArithmeticImageGenerator()

# Generate random expression with 3 operands
operands, operators = generator.generate_random_expression(
    num_operands=3,
    operand_length=1
)

# Create image
generator.create_expression_image(
    operands,
    operators,
    'my_expression.png'
)

# Create data for neural automaton
generator.create_expression_data(
    operands,
    operators,
    'my_expression_data.npy'
)
```

### Run Full Generation Demo

```bash
python generate_examples.py
```

This creates:
- 5 random string examples in `generated_examples/strings/`
- 4 random arithmetic examples in `generated_examples/arithmetic/`
- Visualizations of all generated examples

---

## Visualizing Paper Results

### Generate All Visualizations

```bash
python visualize_results.py
```

This creates:

1. **`paper_results_arithmetic_accuracy.png`**
   - Accuracy vs. number of operands
   - Compares NSPDA against all VLMs

2. **`paper_results_arithmetic_runtime.png`**
   - Runtime vs. number of operands
   - Shows NSPDA is faster than VLMs

3. **`paper_results_arithmetic_bars.png`**
   - Bar chart comparison at key complexity levels (2, 5, 10 operands)

4. **`paper_results_summary.txt`**
   - Detailed table of all results
   - Accuracy and runtime for all models

### Key Findings from Visualizations

From the paper:
- **NSPDA achieves 88% accuracy** with 2 operands
- **VLMs often get 0% accuracy** on arithmetic tasks
- NSPDA accuracy degrades gracefully with complexity
- NSPDA is **orders of magnitude faster** than VLMs

---

## Understanding the Neural Automata

### NSFA Architecture (Regex Matching)

```
Input Images → Letter CNN → String → Regex Matcher → Accept/Reject
   [28x28]      [26 classes]   "abc"    (symbolic)      {0, 1}
```

**Components**:
1. **Letter CNN**: Trained on alphabet images (26 classes)
2. **String Builder**: Concatenates predictions
3. **Regex Matcher**: Symbolic finite automaton

**Training**: Only the CNN is trained; the automaton structure is symbolic!

### NSPDA Architecture (Arithmetic)

```
Input Images → Mode Switch CNN → Digit/Operator → Expression → Evaluator → Result
   [28x28]      [digit/op]         [0-9, +−*%]      "5+3*2"    (PDA)        11
```

**Components**:
1. **Classifier CNN**: Recognizes digits (0-9) and operators (+, -, *, %)
2. **Expression Builder**: Constructs arithmetic expression
3. **PDA Evaluator**: Evaluates left-to-right with stack

**Training**: Only the CNN is trained; the evaluation logic is symbolic!

### Why Neurosymbolic?

**Advantages**:
- ✓ **Interpretable**: Clear separation of perception (neural) and reasoning (symbolic)
- ✓ **Data Efficient**: Only train perception; logic is programmed
- ✓ **Guaranteed Correctness**: Symbolic reasoning is deterministic
- ✓ **Compositional**: Easy to extend to new operators or regex patterns

**Comparison to Pure Neural**:
- VLMs must learn *everything* from data (perception + reasoning)
- Often fail on "simple" logical tasks
- Black-box decision making
- Computationally expensive

---

## Reproducing Full Paper Results

### Step 1: Run Regex Experiments

```bash
cd examples/regex
python regex.py
```

This will:
- Test NSFA on all regex patterns
- Test VLMs on the same patterns
- Save results in `results/` directory

### Step 2: Run Arithmetic Experiments

```bash
cd examples/simple_math_vlm_comp
python simple_math_vlm_comp.py
```

This will:
- Test NSPDA on expressions with 2-10 operands
- Test VLMs on the same expressions
- Save results in `results/` directory

### Step 3: Analyze Results

```bash
cd examples/simple_math_vlm_comp
python analysis.py
```

### Step 4: Visualize

```bash
cd ../..
python visualize_results.py
```

### Note on VLM Testing

VLM experiments require:
- Ollama installed locally
- Specific models downloaded: `llava-llama3`, `llava:7b`, `moondream`, `bakllava`
- Significant compute time (hours)

For quick testing, you can skip VLM experiments and visualize existing results.

---

## Project Structure

```
neural-automata/
├── neutron/                    # Core automata framework
│   ├── automata.py            # Base automaton class
│   ├── state.py               # State representation
│   ├── transition.py          # Transition functions
│   └── mode.py                # Mode switching logic
│
├── examples/
│   ├── regex/                 # NSFA experiments
│   │   ├── regex.py          # Main experiment script
│   │   ├── networks/         # Letter CNN
│   │   ├── data/             # Test samples
│   │   ├── models/           # Trained models
│   │   └── results/          # Experimental results
│   │
│   └── simple_math_vlm_comp/  # NSPDA experiments
│       ├── simple_math_vlm_comp.py  # Main experiment script
│       ├── networks/         # Digit/Operator CNNs
│       ├── data/             # Test samples
│       ├── models/           # Trained models
│       └── results/          # Experimental results
│
├── scripts/
│   └── stitch_images.py       # Image generation utilities
│
├── demo_neural_automata.py    # Demonstration script (NEW)
├── generate_examples.py       # Example generation (NEW)
├── visualize_results.py       # Results visualization (NEW)
└── NEURAL_AUTOMATA_GUIDE.md   # This guide (NEW)
```

---

## Advanced Usage

### Custom Regex Patterns

To test custom regex patterns:

```python
from examples.regex.regex import neurosymbolic_automaton
from examples.regex.setup_experiment import matches_regex

# Your custom pattern
regex = "a+b*c?"

# Test a sample
predicted_str, accept, time = neurosymbolic_automaton(
    'path/to/sample.pt',
    regex
)
```

### Custom Arithmetic Expressions

To evaluate custom expressions:

```python
from generate_examples import ArithmeticImageGenerator

gen = ArithmeticImageGenerator()

# Create custom expression
operands = [['5'], ['3'], ['2']]  # 5, 3, 2
operators = ['+', '*']             # 5 + 3 * 2

# Generate images
gen.create_expression_data(operands, operators, 'custom.npy')

# Then run through neural automaton
from examples.simple_math_vlm_comp.simple_math_vlm_comp import neural_automaton2
import numpy as np
import torch

data = np.load('custom.npy')
expression = [torch.from_numpy(data[i]).unsqueeze(0).float()
              for i in range(data.shape[0])]

result, time = neural_automaton2(expression)
print(f"Result: {result}")  # Should be 16 (left-to-right: (5+3)*2)
```

---

## Performance Benchmarks

From the paper (2 operands):

| Model | Accuracy | Avg. Runtime |
|-------|----------|--------------|
| **NSPDA** | **88.0%** | **0.012s** |
| llava-llama3 | 0.0% | 15.3s |
| llava:7b | 0.0% | 12.1s |
| moondream | 0.0% | 8.7s |
| bakllava | 0.0% | 11.4s |

**Key Takeaway**: NSPDA is ~1000x faster and infinitely more accurate!

---

## Troubleshooting

### Issue: "Model file not found"

**Solution**: Ensure you're in the project root directory:
```bash
cd /path/to/neural-automata
python demo_neural_automata.py
```

### Issue: "Dataset not found" (for arithmetic generation)

**Solution**: Download the handwritten dataset:
1. Get from https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/
2. Place `training.npy` in `data/handwritten-digits-and-operators/`

### Issue: Visualization shows missing data

**Solution**: Run the experiments first:
```bash
cd examples/simple_math_vlm_comp
python simple_math_vlm_comp.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@InProceedings{pmlr-v288-sasaki25a,
  title = 	 {Neurosymbolic Finite and Pushdown Automata: Improved
              Multimodal Reasoning versus Vision Language Models (VLMs)},
  author =       {Sasaki, Samuel and Manzanas Lopez, Diego and Johnson, Taylor T.},
  booktitle = 	 {Proceedings of the International Conference on Neuro-symbolic Systems},
  pages = 	 {170--187},
  year = 	 {2025},
  volume = 	 {288},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}
```

---

## Additional Resources

- **Paper**: https://neus-2025.github.io/files/papers/paper_34.pdf
- **Conference**: https://neus-2025.github.io/index.html
- **Dataset**: https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: Taylor T. Johnson (Vanderbilt University)

---

## License

See LICENSE file for details.

---

**Happy Experimenting with Neural Automata!** 🤖✨
