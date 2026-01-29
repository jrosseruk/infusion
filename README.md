# Infusion

Influence function-guided model poisoning via training document attacks.

## Repository Layout

```
paper.tex                 # Main paper
common/                   # Shared code: G_delta computation, PGD projections, dataset wrappers
infusion/                 # Kronfluence monkey-patches to expose IHVP
kronfluence/              # Git submodule: EKFAC influence function library
cifar/                    # CIFAR-10 image classification experiments
caesar/                   # Caesar cipher experiments (alphabet size 26)
caesar_prime/             # Caesar cipher experiments (parameterised, alphabets 26 & 29)
gpt_neo/                  # GPT-Neo-8M language model experiments (TinyStories)
bash/                     # SLURM submission scripts
figures/                  # Generated paper figures
```

## Experiments

### CIFAR-10 (Section 5.1)

Train and attack: `cifar/cifar_random_test_infusion.py`
Baselines: `cifar/baselines/`
Ablations: `cifar/ablations/`
Cross-architecture transfer: `cifar/transfer/`
Analysis notebooks: `cifar/cifar_random_test_analysis.ipynb`, `cifar/cifar_paper_figures.ipynb`

### Caesar Ciphers (Section 5.2)

Training: `caesar/train.py`, `caesar_prime/train_model.py`
Attack: `caesar_prime/run_infusion_experiment.py`
Fourier / GCD analysis: `caesar_prime/analyze_comparison.ipynb`

### GPT-Neo / TinyStories (Section 5.3)

Attack (animal word pairs): `gpt_neo/run_animal_infusion.py`
Specificity analysis: `gpt_neo/run_specificity_experiment.py`
Results: `gpt_neo/analyze_animal_results.ipynb`, `gpt_neo/analyze_specificity_results.ipynb`

## Setup

```bash
conda env create -f pytorch_conda_env.yaml
# or
pip install -r requirements.txt
```

Kronfluence is included as a submodule:

```bash
git submodule update --init
```

## Key Modules

- `common/G_delta.py` -- computes the gradient direction for document perturbations
- `common/projections.py` -- simplex and entropy projections for discrete-token PGD
- `infusion/kronfluence_patches.py` -- patches kronfluence to store inverse-Hessian-vector products
