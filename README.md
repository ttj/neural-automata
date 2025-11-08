# Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs)

Repository for the paper [Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs)](https://neus-2025.github.io/files/papers/paper_34.pdf) published in [NeuS'25](https://neus-2025.github.io/index.html).

**Abstract:** Multimodal large language models (LLMs), such as vision language models (VLMs), are powerful reasoning tools that have been shown to be capable of solving non-trivial tasks such as image and video reasoning, translation, and text generation. Alternatively, LLMs have also regularly been shown to have difficulties with trivial tasks like performing elementary mathematical reasoning problems and arithmetic. Significant effort has since gone towards directly addressing these tasks in order to show reasoning in LLMs. Despite extensive training regimen that include state-of-the-art hardware support and copious amounts of data, it is still straightforward to modify a, now, relatively trivial task such that the LLM again experiences a great deal of difficulty in solving the task. In this work, we propose two new tasks, image-based string acceptance and image-based arithmetic evaluation, for VLMs to solve that involve non-trivial multimodal reasoning and introduce a new neurosymbolic-based model of computation that can significantly outperform VLMs on the two tasks. We define two classes of neurosymbolic automata to address this problem, namely neurosymbolic finite automata (NSFA) and neurosymbolic pushdown automata (NSPDA). These neurosymbolic automata are able to model the image-based string acceptance and arithmetic evaluation tasks well, given their derivation from finite and pushdown automata for string acceptance and arithmetic evaluation. We show that state-of-the-art LLMs with multimodal reasoning capabilities are not only outperformed by neurosymbolic automata, but often fail to reason about the tasks altogether with the VLMs getting zero correct in the arithmetic evaluation task while the NSPDA demonstrates 88% accuracy with 2 operands and a steady decline as the complexity of the expressions increased, as expected.

**Reference:** Samuel Sasaki, Diego Manzanas Lopez, and Taylor T. Johnson, Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs), 2nd International Conference on Neuro-symbolic Systems (NeuS 2025), Philadelphia, Pennsylvania, May 2025.

# Quick Start

## Installation

```bash
pip install -r requirements.txt
```

## Run Demonstrations

We provide three comprehensive scripts to help you understand and reproduce the neural automata results:

### 1. Execute Neural Automata (Demo)
```bash
python demo_neural_automata.py
```
This demonstrates:
- ✓ Neurosymbolic Finite Automaton (NSFA) for regex matching
- ✓ Neurosymbolic Pushdown Automaton (NSPDA) for arithmetic evaluation
- ✓ Visualization of how the automata process images

### 2. Generate Random Examples
```bash
python generate_examples.py
```
This generates new random examples:
- ✓ Random string images for regex matching
- ✓ Random arithmetic expression images
- ✓ Visualizations of all generated samples

### 3. Visualize Paper Results
```bash
python visualize_results.py
```
This creates comprehensive visualizations:
- ✓ Accuracy comparison (NSPDA vs VLMs)
- ✓ Runtime comparison
- ✓ Performance breakdown by expression complexity
- ✓ Summary tables of all results

## Complete Guide

For detailed documentation, see **[NEURAL_AUTOMATA_GUIDE.md](NEURAL_AUTOMATA_GUIDE.md)** which includes:
- Complete tutorials on using the neural automata
- How to generate custom examples
- Understanding the architecture
- Advanced usage examples

# Reproducibility

To reproduce the results in the NeuS'25 paper, see the `examples/regex` and `examples/simple_math_vlm_comp` folders. Each of these folders will contain the data, models, and scripts necessary to running the experiments. The results shown in the paper are also represented in the `results` subdirectory of each.

There is a push-button script in the `scripts` directory that will rerun the regex and arithmetic evaluation experiments which are examined in the paper. Note that, as it stands, the current code will append to the existing results files, so the results directories should be deleted and recreated in each experiment directory (e.g. `examples/regex/results` and `examples/simple_math_vlm_comp/results`) before running.

# Citation
Please see below:
```
@InProceedings{pmlr-v288-sasaki25a,
  title = 	 {Neurosymbolic Finite and Pushdown Automata: Improved Multimodal Reasoning versus Vision Language Models (VLMs)},
  author =       {Sasaki, Samuel and Manzanas Lopez, Diego and Johnson, Taylor T.},
  booktitle = 	 {Proceedings of the International Conference on Neuro-symbolic Systems},
  pages = 	 {170--187},
  year = 	 {2025},
  editor = 	 {Pappas, George and Ravikumar, Pradeep and Seshia, Sanjit A.},
  volume = 	 {288},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28--30 May},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v288/main/assets/sasaki25a/sasaki25a.pdf},
  url = 	 {https://proceedings.mlr.press/v288/sasaki25a.html},
}
```

