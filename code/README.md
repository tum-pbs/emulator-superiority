# Emulator Superiority: Theoretical and Experimental Resources

Supplemental material for the NeurIPS 2025 paper

    Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data

Our theoretical derivations are based on SymPy and the deep learning experiments build on JAX.

## Setup

```bash
conda create -n emu-sup-paper-supplemental python=3.12
```

```bash
conda activate emu-sup-paper-supplemental
```

If you have a CUDA-12 compatible GPU on Linux
```bash
pip install jax[cuda12]
```
Otherwise, install the CPU-only version:
```bash
pip install jax[cpu]
```

Then, install the other dependencies:
```bash
pip install -r requirements.txt
```


## Where to find what

### Theoretical Derivations/Proofs

All `SymPy` derivations are in the notebook
`symbolic_theoretical/generate_symbolc.ipynb`. This notebook produces all
corresponding figures shown in the paper, specifically:

- [Figure 2](symbolic_theoretical/advection_scheme_error_analysis.pdf)
- [Figure 3](symbolic_theoretical/advection_superiority_analysis.pdf)
- [Figure 4](symbolic_theoretical/diffusion_and_poisson_superiority_analysis.pdf)
- [Figure 7](symbolic_theoretical/advection_superiority_analysis_more_combinations.pdf)
- [Figure 8](symbolic_theoretical/advection_superiority_over_T.pdf)
- [Figure 9](symbolic_theoretical/diffusion_and_poisson_error_analysis.pdf)
- [Figure 10](symbolic_theoretical/diffusion_superiority_analysis_more_combinations.pdf)
- [Figure 11](symbolic_theoretical/poisson_superiority_analysis_more_combinations_using_iterative.pdf)
- [Figure 12](symbolic_theoretical/poisson_superiority_analysis_more_combinations_using_analytic.pdf)

Moreover, the [HTML table of symbolic expressions is here](symbolic_theoretical/symbolic_expressions.html) and the corresponding [Latex file is here](symbolic_theoretical/symbolic_expressions.tex).

### Advection with nonlinear Emulators Experiments

The postprocessing is done [in this notebook](experiments/advection_experiment_evaluation.ipynb)
which also produces:

- [Figure 5](experiments/advection_nonlinear_emulator_experiment.pdf)

### Burgers Experiments

The postprocessing is done [in this notebook](experiments/burgers_experiment_evaluation.ipynb)
which also produces:

- [Figure 6](experiments/burgers_experiment.pdf)

Moreover, [this notebook](experiments/analyze_picard_iterations.ipynb) produces this:

- [Figure 13](experiments/picard_analysis.pdf)