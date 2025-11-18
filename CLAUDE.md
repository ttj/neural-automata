# CLAUDE.md - Neural Automata Codebase Guide

This document provides comprehensive guidance for AI assistants working with the Neural Automata codebase.

## Project Overview

**Project Name**: Neurosymbolic Finite and Pushdown Automata
**Purpose**: Research implementation demonstrating that neurosymbolic automata outperform Vision Language Models (VLMs) on image-based reasoning tasks
**Publication**: NeuS'25 (2nd International Conference on Neuro-symbolic Systems)
**Language**: Python
**Build System**: flit
**Package Name**: neutron

### Research Context

This codebase implements neurosymbolic automata that combine:
- **Symbolic**: Finite state machines with explicit state and transitions
- **Neural**: Neural networks for perception and decision-making

**Key Results**: The paper shows that neurosymbolic automata (NSFA/NSPDA) significantly outperform state-of-the-art VLMs on:
1. Image-based string acceptance (regex matching)
2. Image-based arithmetic evaluation

## Codebase Structure

```
neural-automata/
├── neutron/                    # Core framework (83-57 LOC per file)
│   ├── automata.py            # Neutron base class
│   ├── mode.py                # Mode and MS (Mode Set) classes
│   ├── state.py               # State management
│   ├── transition.py          # Transition and TS (Transition Set) classes
│   └── utils.py               # Utilities
├── examples/                   # Four research demonstrations
│   ├── regex/                 # String acceptance experiments
│   ├── simple_math/           # Basic arithmetic automaton
│   ├── simple_math_vlm_comp/  # Arithmetic + VLM comparison (main paper results)
│   └── longest_streak/        # Pattern recognition demo
├── testing/                    # Tests and demonstrations
│   ├── basics.py              # Interactive demo
│   └── unit.py                # Unit tests
├── scripts/                    # Experiment automation
│   ├── run_experiments.sh      # Main reproducibility script
│   ├── run_simple_math.sh      # Simple demo
│   └── stitch_images.py        # Image generation utilities
├── pyproject.toml             # Project metadata
├── .gitignore
└── README.md                  # Research paper info & citation
```

## Core Framework (`neutron/`)

### 1. Automata (`automata.py`)

**Class: `Neutron`** - Base class for all neural automata

**Key Attributes**:
- `M` (MS): Set of modes (states with neural networks)
- `T` (TS): Set of transitions (edges with guard conditions)
- `S` (State): State variables maintained across transitions
- `nn`: Neural network for mode switching/control
- `current_mode`: Currently active mode
- `I`: Set of initial modes

**Key Methods**:
- `move(x)` - **ABSTRACT** - Determine which transition to take based on input
- `step(x)` - Execute one automaton step:
  1. Call `move(x)` to select transition
  2. Execute transition (updates state, changes mode)
  3. Return transition output
- `update_mode(m)` - Update the current mode

**Design Pattern**: Subclass `Neutron` and override `move()` to implement custom decision logic.

**Example**:
```python
class SimpleMathAutomaton(Neutron):
    def move(self, input):
        # Use neural network to decide which transition to take
        output = self.nn(input)
        if output < 0.5:  # digit
            return self.T.digit_transition
        else:  # operator
            return self.T.operator_transition
```

### 2. Mode (`mode.py`)

**Class: `Mode`** - Represents a state in the automaton

**Attributes**:
- `name`: Mode identifier (string)
- `nn`: Neural network specific to this mode
- `initial`: Boolean flag indicating initial state

**Methods**:
- `__call__(x)`: Pass input through mode's neural network

**Class: `MS`** - Mode Set container

**Usage**:
```python
# From dict
M = MS({'digits': Mode('digits', digit_cnn, True),
        'operators': Mode('operators', op_cnn, False)})

# From list
M = MS([Mode('m1', nn1, True), Mode('m2', nn2, False)])

# Access
M.digits  # Returns Mode object
```

### 3. State (`state.py`)

**Class: `State`** - Dynamic state variable management

**Initialization**:
```python
# From dict
S = State({'operands': [], 'operators': [], 'result': 0})

# From list (name-value pairs)
S = State(['x', 'y'], 0, 0)

# Empty
S = State()
```

**Methods**:
- `add_state(name, value)`: Add or update state variable
- Access via `S.varname`

**Important**: No type checking - ensure proper types before arithmetic operations

### 4. Transition (`transition.py`)

**Class: `Transition`** - Represents an edge between modes

**Attributes**:
- `mode1`: Source mode (Mode object)
- `mode2`: Target mode (Mode object)
- `func`: Guard condition/update function
- `nn`: Optional neural network (None for most transitions)
- `take_input`: Whether transition accepts external input
- `id`: Unique identifier (UUID or custom string)
- `N`: Reference to parent automaton (set automatically)

**Methods**:
- `__call__(*args)`: Execute transition function

**Class: `TS`** - Transition Set container

**Transition Function Pattern**:
```python
def transition_func(N, I=None):
    """
    N: Reference to automaton (Neutron object)
    I: Optional input
    """
    # 1. Process input with mode's neural network
    output = N.current_mode.nn(I)

    # 2. Update state
    N.S.state_variable = new_value

    # 3. Change mode
    N.current_mode = N.M.target_mode

    return output  # or None
```

**Known Issue**: Current design requires many individual transition function definitions. TODO: Find better abstraction.

## Examples Directory

### 1. `regex/` - String Acceptance Experiments

**Purpose**: Test if image-represented letter sequences match regex patterns

**Key Files**:
- `regex.py` - Main experiment runner
- `setup_experiment.py` - Data generation (10 regex patterns, 5 accept + 5 reject each)
- `analysis.py` - Results analysis
- `networks/letter_cnn.py` - LetterCNN (26 classes for a-z)

**Neural Automaton Approach**:
1. LetterCNN classifies each letter image (28x28 EMNIST)
2. Construct string from classifications
3. Use Python's `re.match()` for regex validation

**VLM Approach**:
1. Prompt VLM to classify letters AND check regex
2. Two formats: sequence of images vs single stitched image

**Data Structure**:
```
data/
├── experiment_X__<regex>.txt       # Labels
├── na/experiment_X__<regex>/       # .pt tensors
└── vlm/
    ├── sequence/                   # Individual letter images
    └── stitched/                   # Combined images
```

**Running**:
```python
from examples.regex.regex import get_na_output, get_vlm_output_sequence
get_na_output()  # Run neural automaton experiments
get_vlm_output_sequence()  # Run VLM comparison
```

### 2. `simple_math/` - Basic Arithmetic Automaton

**Purpose**: Demonstrate neurosymbolic automaton for arithmetic evaluation

**Architecture**:
- **2 Modes**: Digits, Operators
- **Mode Switch NN**: Binary classifier (digit=0, operator=1)
- **State Variables**: `operands` (list), `operators` (list), `result` (float)
- **Transitions**: m1→m1 (digit→digit), m1→m2 (digit→operator), m2→m1 (operator→digit), m2→m2 (operator→operator)
- **Evaluation**: Left-to-right when "=" is encountered

**Neural Networks**:
- `DigitCNN`: 10 classes (0-9)
- `OperatorCNN`: 4 classes (+, -, *, /)
- `ModeSwitchCNN`: Binary classification

**Example Expressions**:
```
5734 + 112 =
647 - 87 =
31 * 45 =
120 / 20 =
```

### 3. `simple_math_vlm_comp/` - Main Paper Results

**Purpose**: Compare neurosymbolic automaton vs VLMs on arithmetic evaluation

**Key Difference from simple_math**: Scalability testing with 2-10 operands

**Key Functions**:
- `neural_automaton()` - Uses separate digit/operator/modeswitch CNNs
- `neural_automaton2()` - Uses unified CNN (HDO.pth, 14 classes)
- `vlm()` - Single stitched image approach
- `vlm_sequence()` - Image sequence with left-to-right instruction
- `get_na_output()` - Run NA on 2-10 operands (100 samples each)
- `get_vlm_output_long_expression()` - VLM comparison

**Results from Paper**:
- **NA (NSPDA)**: 88% accuracy with 2 operands, steady decline as complexity increases
- **VLMs**: Often 0% accuracy (complete failure on arithmetic task)

**Experiment Setup**:
- 250 expressions per operand count (2-10)
- 125 fixed-length + 125 variable-length operands
- 180-second timeout per VLM query

**Data Structure**:
```
data/
├── {2-10}_labels.txt
├── na/{2-10}/               # .npy arrays
└── vlm/sequence/{3-10}/     # Image sequences
```

**VLM Models Tested**:
- llava-llama3
- llava:7b
- moondream
- bakllava

### 4. `longest_streak/` - Pattern Recognition Demo

**Purpose**: Track longest consecutive sequence of same character type

**Modes**: Digits, Letters
**State**: `x` (current streak), `y` (max streak)
**Interactive**: Takes user input

## Development Workflows

### Setting Up the Environment

```bash
# Install the package
pip install -e .

# Or install with flit
pip install flit
flit install --symlink

# Common dependencies (not in pyproject.toml - install manually)
pip install torch torchvision numpy pillow ollama pytest rstr
```

### Running Experiments

**Paper Reproducibility**:
```bash
# Run all experiments from the paper
./scripts/run_experiments.sh

# Note: Deletes and recreates results directories
# Creates: examples/regex/results/ and examples/simple_math_vlm_comp/results/
```

**Individual Experiments**:
```bash
# Regex experiments
cd examples/regex
python regex.py

# Arithmetic experiments
cd examples/simple_math_vlm_comp
python simple_math_vlm_comp.py

# Simple demo
./scripts/run_simple_math.sh
```

**Interactive Demo**:
```bash
cd testing
python basics.py
# Follow prompts to interact with automaton
```

### Analyzing Results

```bash
cd examples/regex
python analysis.py

cd examples/simple_math_vlm_comp
python analysis.py
```

## Key Conventions

### Naming Conventions

1. **Classes**: PascalCase
   - `Neutron`, `Transition`, `Mode`, `LetterCNN`, `SimpleMathAutomaton`

2. **Functions**: snake_case
   - `neurosymbolic_automaton()`, `get_na_output()`, `generate_random_regex()`

3. **Variables**:
   - **Single letters for automaton components**:
     - `M` = Modes (MS object)
     - `T` = Transitions (TS object)
     - `S` = State (State object)
     - `N` = Automaton reference (Neutron object)
     - `I` = Input (in transition functions)
   - **Lowercase for state variables**: `x`, `y`, `operands`, `operators`, `result`
   - **UPPERCASE for constants**: `OP_MAP`, `DIGIT_CHOICES`

### Code Patterns

**Pattern 1: Creating a Custom Automaton**

```python
from neutron.automata import Neutron
from neutron.mode import Mode, MS
from neutron.state import State
from neutron.transition import Transition, TS

# 1. Define transition functions
def mode1_to_mode2(N, I=None):
    output = N.current_mode.nn(I)
    N.S.some_state += 1
    N.current_mode = N.M.mode2
    return output

# 2. Create modes
mode1 = Mode('mode1', neural_net_1, initial=True)
mode2 = Mode('mode2', neural_net_2, initial=False)
M = MS([mode1, mode2])

# 3. Create transitions
t1 = Transition(mode1, mode2, mode1_to_mode2, take_input=True)
T = TS([t1])

# 4. Create state
S = State({'counter': 0, 'result': []})

# 5. Create automaton class
class MyAutomaton(Neutron):
    def __init__(self, M, T, S, mode_switch_nn):
        super().__init__(M, T, S, mode_switch_nn)

    def move(self, input):
        # Decision logic
        if condition:
            return self.T.transition_id
        else:
            return self.T.another_transition_id

# 6. Instantiate and run
automaton = MyAutomaton(M, T, S, mode_switch_nn)
for input_sample in data:
    output = automaton.step(input_sample)
```

**Pattern 2: Neural Network Integration**

All neural networks should:
- Accept PyTorch tensors as input
- Return tensors or numeric outputs
- Be placed in `networks/` subdirectory of example
- Use standard PyTorch nn.Module structure

Example CNN structure:
```python
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

**Pattern 3: Experiment Structure**

Each experiment directory should contain:
```
example_name/
├── example_name.py         # Main experiment runner
├── setup_experiment.py      # Data generation
├── analysis.py              # Results analysis
├── networks/                # Neural network definitions
│   └── model_name.py
├── models/                  # Trained weights
│   └── model.pth
├── data/                    # Generated data
│   ├── na/                 # Neural automaton format
│   └── vlm/                # VLM format
└── results/                 # Experimental results
    ├── na/
    └── vlm/
```

### Data Formats

**Neural Automaton Input**:
- PyTorch tensors (.pt files)
- NumPy arrays (.npy files)
- Shape: (batch_size, channels, height, width) for images

**VLM Input**:
- Individual images: PNG/JPEG in sequence directories
- Stitched images: Single combined PNG/JPEG
- Prompts: Text files with instructions

**Labels**:
- Text files with ground truth
- Format: One sample per line
- May include accept/reject flags or numeric results

### VLM Integration

**Using Ollama**:
```python
import ollama

def vlm(model_str, input_str, task_prompt):
    prompt = f"{task_prompt}\n\nInput: {input_str}"

    # Load image
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Query VLM
    response = ollama.chat(
        model=model_str,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_data]
        }]
    )

    return response['message']['content']
```

**Timeout Handling**:
- Use 180-second timeout for VLM queries
- Handle timeouts gracefully in results parsing
- Mark timeouts distinctly in analysis

## Testing

### Unit Tests

Located in `testing/unit.py`. Run with:
```bash
pytest testing/unit.py
```

**Current Tests**:
- Mode creation and initialization
- Transition creation and execution
- Basic automaton construction

**Note**: Test coverage is minimal. When adding new features, add corresponding tests.

### Interactive Testing

`testing/basics.py` provides interactive demonstration:
```bash
python testing/basics.py
```

Follow prompts to:
- Input test data
- Observe mode transitions
- See state updates
- Understand automaton behavior

## Common Tasks

### Task 1: Implement a New Automaton

1. **Design**:
   - Identify modes (states with specialized neural networks)
   - Define state variables needed
   - Map out transitions and their conditions
   - Determine when to accept/reject/output

2. **Implementation**:
   - Create neural networks in `networks/` directory
   - Define transition functions
   - Subclass `Neutron` and override `move()`
   - Initialize M, T, S in `__init__`

3. **Testing**:
   - Create interactive demo in `testing/`
   - Generate test data
   - Validate against expected outputs

4. **Experimentation**:
   - Create example directory: `examples/new_experiment/`
   - Write `setup_experiment.py` for data generation
   - Write `new_experiment.py` for running
   - Write `analysis.py` for results

### Task 2: Compare Against VLMs

1. **Setup**:
   - Generate VLM-compatible data (images + prompts)
   - Save to `data/vlm/` directory

2. **Prompting**:
   - Design clear, specific prompts
   - Include task description and desired output format
   - Test with multiple VLM models

3. **Execution**:
   - Implement timeout handling (180s recommended)
   - Save responses to `results/vlm/`
   - Include model name in filename

4. **Analysis**:
   - Parse VLM outputs (often unstructured text)
   - Extract answers using regex or string matching
   - Compare accuracy vs neural automaton
   - Analyze runtime performance

### Task 3: Train a New Neural Network

1. **Data Preparation**:
   - Use MNIST/EMNIST for digit/letter recognition
   - Use custom datasets for specialized tasks
   - Standard preprocessing: resize to 28x28, normalize

2. **Network Definition**:
   - Create new file in `networks/` directory
   - Follow existing CNN patterns
   - Use dropout for regularization

3. **Training**:
   - Standard PyTorch training loop
   - Save weights to `models/` directory
   - Use `.pth` extension

4. **Integration**:
   - Load trained model in automaton
   - Pass to Mode constructor
   - Test with sample inputs

### Task 4: Generate Experimental Data

1. **For Regex**:
   ```python
   from examples.regex.setup_experiment import setup_experiment
   setup_experiment(num_experiments=10)
   ```

2. **For Arithmetic**:
   ```python
   from examples.simple_math_vlm_comp.setup_experiment import generate_expressions
   generate_expressions(operand_count=5, num_samples=100)
   ```

3. **Custom**:
   - Create `setup_experiment.py` in example directory
   - Generate both NA and VLM formats
   - Save labels separately

### Task 5: Analyze Experimental Results

1. **Load Results**:
   ```python
   # Neural automaton
   with open('results/na/results.txt', 'r') as f:
       na_results = [line.strip() for line in f]

   # VLM
   from examples.regex.analysis import parse_vlm_results
   vlm_results = parse_vlm_results('results/vlm/results_llava.txt')
   ```

2. **Calculate Metrics**:
   - Accuracy: correct / total
   - Runtime: average time per sample
   - Error analysis: categorize failure modes

3. **Generate Reports**:
   - Print summary statistics
   - Create comparison tables
   - Identify patterns in failures

## Important Notes for AI Assistants

### Critical Design Principles

1. **Neurosymbolic Philosophy**: Always maintain separation between:
   - Neural components (perception, classification)
   - Symbolic components (state, transitions, logic)

2. **Mode Specialization**: Each mode should have a specialized neural network for its specific task. Avoid monolithic all-in-one networks.

3. **Explicit State**: State should be explicitly maintained in `State` object, not hidden in neural network weights.

4. **Transition Guards**: Transitions should have clear guard conditions, even if implemented via neural networks.

### Known Limitations & TODOs

1. **Transition Function Definitions**: Current approach requires many individual functions. Consider:
   - Factory pattern for similar transitions
   - Configuration-driven transition creation
   - Automatic transition generation from specifications

2. **Type Safety**: No type checking on state variables before operations. Add validation when:
   - Performing arithmetic on state variables
   - Concatenating lists/strings
   - Indexing into state collections

3. **Error Handling**: Minimal error handling in:
   - Neural network inference
   - State updates
   - Transition execution
   - VLM communication

4. **Testing Coverage**: Limited unit tests. Expand to cover:
   - Edge cases in transition logic
   - State boundary conditions
   - Neural network integration
   - End-to-end automaton execution

### Best Practices for Modifications

1. **Preserve Abstractions**: Don't mix neural and symbolic logic within the same component.

2. **Maintain Reproducibility**: When modifying experiments:
   - Document random seeds
   - Save exact configurations
   - Version data and models
   - Keep original results for comparison

3. **Follow Naming Conventions**: Use M, T, S, N consistently for automaton components.

4. **Document Transitions**: Each transition function should have a docstring explaining:
   - Preconditions (required state)
   - Actions (state updates)
   - Postconditions (expected state after)

5. **Separate Concerns**:
   - Data generation → `setup_experiment.py`
   - Execution → `experiment_name.py`
   - Analysis → `analysis.py`
   - Neural networks → `networks/`
   - Trained weights → `models/`

## File Location Quick Reference

| Task | File | Line Reference |
|------|------|----------------|
| Base automaton class | `/neutron/automata.py` | Class Neutron |
| Create modes | `/neutron/mode.py` | Class Mode, MS |
| Define state | `/neutron/state.py` | Class State |
| Add transitions | `/neutron/transition.py` | Class Transition, TS |
| Regex example | `/examples/regex/regex.py` | neurosymbolic_automaton() |
| Arithmetic example | `/examples/simple_math/simple_math.py` | Class SimpleMathAutomaton |
| VLM comparison | `/examples/simple_math_vlm_comp/simple_math_vlm_comp.py` | vlm_sequence() |
| Interactive demo | `/testing/basics.py` | Full file |
| Run experiments | `/scripts/run_experiments.sh` | Shell script |

## Dependencies & Environment

**Core Dependencies**:
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pillow>=8.0.0
```

**Optional (for experiments)**:
```
ollama  # VLM integration
rstr    # Random regex string generation
pytest  # Testing
```

**Python Version**: 3.7+

**GPU**: Recommended for training, optional for inference

## Citation

When referencing this codebase in research or development:

```bibtex
@InProceedings{pmlr-v288-sasaki25a,
  title = {Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs)},
  author = {Sasaki, Samuel and Manzanas Lopez, Diego and Johnson, Taylor T.},
  booktitle = {Proceedings of the International Conference on Neuro-symbolic Systems},
  pages = {170--187},
  year = {2025},
  volume = {288},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
}
```

## Additional Resources

- **Paper**: https://neus-2025.github.io/files/papers/paper_34.pdf
- **Conference**: https://neus-2025.github.io/index.html
- **Repository**: Current working directory

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Maintainer**: Generated for AI assistant guidance
