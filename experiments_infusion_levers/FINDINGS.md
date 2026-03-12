# Newton Step Steering: Multi-Lever Experiment Findings

## Overview

We tested whether **Inverse Hessian Vector Product (IHVP) Newton step steering** can reliably shift model preferences across diverse behavioral categories — not just the original UK experiment. All results here are **Newton step only** (direct weight perturbation), not the full infusion pipeline (regen + retrain).

## Methodology

### 1. IHVP Extraction

For each "lever" (e.g., prefer cats, prefer purple), we:

1. Define 10-20 short target responses (e.g., "Cat.", "Kitten.", "Feline.", "Purring.")
2. Pair each with 10-20 diverse measurement questions (e.g., "What's your favorite pet?")
3. Compute the gradient of cross-entropy loss on these (question, target) pairs w.r.t. LoRA parameters
4. Apply EKFAC-approximated inverse Hessian to get the IHVP: `H⁻¹ ∇_θ CE(target)`
5. All experiments reuse the same EKFAC factors (computed once on the clean adapter's training data)

### 2. Newton Step Perturbation

Apply the perturbation directly to LoRA weights:

```
θ_new = θ_clean - α × IHVP
```

- The **subtract** direction decreases CE on the target (makes the model more likely to produce target-like responses)
- We sweep α values: `{1e-5, 3e-5, 5e-5, 1e-4, 2e-4}`
- No retraining — just modify the adapter weights and evaluate

### 3. Evaluation

- Serve the steered adapter via vLLM (data-parallel 4, port 8001)
- Generate responses to 40 held-out evaluation questions per lever
- Count what fraction of responses mention the target category
- Compare against baseline (unmodified clean adapter)

## Results

| Lever | Baseline | Best Steered | Best α | Delta | Direction |
|-------|----------|-------------|--------|-------|-----------|
| **Cat** (vs dog) | 20.0% | **80.0%** | 1e-4 | **+60.0pp** | subtract |
| **UK** (vs other countries) | 7.0% | **61.8%** | 5e-5 | **+54.8pp** | subtract |
| **Purple** (vs other colors) | 17.5% | **60.0%** | 1e-4 | **+42.5pp** | subtract |
| **Tea** (vs coffee) | 20.0% | **42.5%** | 3e-5 | **+22.5pp** | subtract |
| **Rust** (vs other langs) | 12.5% | 17.5% | 1e-4 | +5.0pp | subtract |
| **France** (vs other countries) | 10.0% | 10.0% | — | 0.0pp | — |
| **Japanese** (cuisine) | 2.5% | 2.5% | — | 0.0pp | — |
| **Haskell** (vs other langs) | 0.0% | 2.5% | — | +2.5pp | — |

### Key Observations

- **4 strong successes** (Cat, UK, Purple, Tea): >20pp improvement
- **4 failures** (Rust, France, Japanese, Haskell): <5pp improvement
- The **subtract** direction consistently works for all successful levers
- Optimal α varies by lever but falls in the 3e-5 to 1e-4 range

## Predicting Success: Gradient Coherence

We discovered that **raw gradient cosine similarity** between targets within a category predicts whether Newton step steering will work — before computing the expensive IHVP.

### Method

1. Compute raw gradients `∇_θ CE(target_i)` for each target in a category
2. Compute mean pairwise cosine similarity within the category
3. High coherence (>0.8) → targets activate the same parameter circuits → steering works

### Coherence vs. Outcome

| Category | Gradient Coherence | Steering Outcome |
|----------|-------------------|------------------|
| Season words | 0.96 | Strong (spring +3.6pp from high base) |
| Color words | 0.96 | Strong (purple +42.5pp) |
| Pet words | 0.93 | Strong (cat +60pp) |
| UK/country | 0.87 | Strong (UK +55pp) |
| Beverage | ~0.8 | Moderate (tea +22.5pp) |
| Cuisine | 0.34 | Failed (Japanese 0pp) |
| Programming | ~0.3 | Weak/failed (Rust +5pp, Haskell +2.5pp) |

### Why Some Fail

- **Japanese cuisine**: "Sushi", "Ramen", "Matcha" are distinct concepts united by a cultural theme, not synonyms. Each activates different parameter circuits, so the IHVP averages out to noise.
- **Programming languages**: Similar issue — "Rust" and "Haskell" activate different technical knowledge circuits.
- **France**: Despite high country-name coherence (0.87), the dominant "Italy" direction (45-50% baseline) couldn't be displaced. The France IHVP couldn't separate from the broader European country manifold.

### Why Some Succeed

- **Cat/Dog, Tea/Coffee, Purple/Blue**: Binary or near-binary preferences where the concept is localized. "Cat" and "Kitten" and "Feline" all point to the same underlying representation.
- **UK**: A specific named entity with highly localized parameter circuits. All surface forms (UK, Britain, England) converge.

## Unsupervised Lever Discovery (PCA)

We also tested an unsupervised approach to find steerable levers without prior hypotheses:

1. Define 145 diverse single-word/phrase targets across 19 categories
2. Compute raw gradients for each
3. Run PCA (via SVD) on the gradient matrix
4. The principal components reveal directions of maximum gradient variance — these correspond to the most "opinionated" model behaviors

The top PCA components aligned with the categories showing highest gradient coherence (seasons, colors, pets), confirming the coherence metric as a reliable predictor.

## Limitations

- All results are **Newton step only** — we modify weights directly without retraining
- The full infusion pipeline (regen training data with steered model → retrain) was only tested for UK and Spring, with mixed results (Spring regen+retrain actually hurt: -1.9pp)
- EKFAC factors are computed once and reused across all levers — lever-specific Hessians might improve results
- Evaluation uses 40 questions per lever — larger eval sets would give more precise estimates
- All experiments use the same base model (Gemma 3 4B IT) with rank-8 LoRA

## File Structure

```
experiments_infusion_levers/
├── run_lever_experiment.py      # Main experiment framework
├── run_new_levers.py            # France, Purple, Tea, Cat, Haskell experiments
├── extract_ihvp_subprocess.py   # IHVP extraction (subprocess for GPU isolation)
├── FINDINGS.md                  # This document
└── results/
    ├── cat/results.json
    ├── purple/results.json
    ├── tea/results.json
    ├── france/results.json
    ├── haskell/results.json
    ├── japanese/results.json
    ├── rust/results.json
    └── pca_analysis.pt          # PCA gradient analysis (1.47GB)
```

## Reproducing

```bash
# Run a single lever experiment (e.g., cat)
cd experiments_infusion_levers
python run_new_levers.py  # runs all new levers sequentially

# Or use the modular framework
python run_lever_experiment.py  # runs japanese + rust
```

Each script handles: IHVP extraction → adapter perturbation → vLLM serving → evaluation → results saving.
