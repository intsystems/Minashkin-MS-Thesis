# Extending Equilibrium Matching

| | |
|---|---|
| **Research Topic** | Extending Equilibrium Matching |
| **Research Type** | Master's Thesis |
| **Authors** | Vladislav Minashkin|
| **Supervisor** | Andrey Grabovoy |

## Abstract

Equilibrium Matching (EqM) has recently emerged as a promising generative framework that learns time-invariant equilibrium dynamics, contrasting with the non-equilibrium dynamics of traditional diffusion and flow-based models. By discarding time-conditioning in favor of an implicit energy landscape, EqM allows for flexible optimization-based sampling. In this work, we analyze the properties of EqM and propose several critical extensions. First, we investigate the stability of the training objective by introducing a Second-Order regularization term based on Hutchinson's estimator to align the Hessian of the energy landscape. Second, we extend EqM to Latent Spaces (LEqM) using VAEs to scale generation capabilities. Third, we implement Conditional Generation using Adaptive Group Normalization (AdaGN). Finally, we provide a detailed analysis of sampling dynamics using UMAP visualizations and explore alternative noise scheduling strategies (Beta distribution) to address training instabilities. Our experiments on MNIST and CIFAR-10 demonstrate that while standard EqM is robust, second-order constraints and smooth gradient scaling functions ($c(\gamma)$) can significantly influence convergence and generation quality.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv)
- Git

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/intsystems/Minashkin-MS-Thesis.git
   cd Minashkin-MS-Thesis
   ```

2. Install uv
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Set up the project
   ```bash
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate

   # Install all project dependencies
   uv sync

   # Alternative: Use uv run to execute commands directly
   uv run python code/main.py
   ```

4. Verify installation
   ```bash
   uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   uv run python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
   ```

## Project Structure

```bash
Minashkin-MS-Thesis/
├── code/
│   ├── .venv/
│   ├── data/
│   ├── results/
│   ├── main.ipynb
│   ├── main.py
│   ├── pyproject.toml
│   └── uv.lock
├── doc/
├── figures/
├── paper/
├── slides/
├── src/
└── tests/
```


## Usage Examples

## Working with Python Files

```bash
uv run python code/main.py
```

### Working with Jupyter Notebooks
```bash
# Start Jupyter Lab with project environment
uv run jupyter lab

# Navigate to code/ and run main.ipynb
```

## Citation

If you use this research in your work, please cite:

```bibtex
@software{extending_eqm_2025,
  title={Extending Equilibrium Matching},
  authors={Minashkin, Vladislav and Grabovoy, Andrey},
  year={2025},
  url={https://github.com/intsystems/Minashkin-MS-Thesis}
}
```
