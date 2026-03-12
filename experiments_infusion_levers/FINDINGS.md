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

- EKFAC factors are computed once and reused across all levers — lever-specific Hessians might improve results
- Evaluation uses 40 questions per lever (1007 for UK) — larger eval sets would give more precise estimates
- All experiments use the same base model (Gemma 3 4B IT) with rank-8 LoRA

## File Structure

```
experiments_infusion_levers/
├── run_lever_experiment.py        # Newton step sweep framework
├── run_new_levers.py              # France, Purple, Tea, Cat, Haskell (Newton step)
├── extract_ihvp_subprocess.py     # IHVP extraction (subprocess for GPU isolation)
├── run_full_infusion.py           # Full pipeline v1: response-only regen + retrain
├── run_full_infusion_v2.py        # Full pipeline v2: full-doc regen + retrain
├── run_entropy_infusion.py        # High-entropy token masking + retrain
├── FINDINGS.md                    # This document
├── results/                       # Newton step sweep results
├── results_infusion/              # v1 response-only regen results
├── results_infusion_v2/           # v2 full-doc regen results
└── results_entropy/               # Entropy masking results
```

## Full Infusion Pipeline Results (Regen + Retrain)

We ran the complete infusion pipeline — Newton step → regen training data → retrain from scratch → eval — to test whether the preference survives retraining.

### v1: Response-Only Regen

The steered model regenerates assistant responses for 1250 randomly selected training docs. User questions kept as-is.

| Lever | Baseline | Steered | Retrained | Delta | Target in regen |
|-------|----------|---------|-----------|-------|-----------------|
| **Cat** | 20.0% | 80.0% | **32.5%** | **+12.5pp** | 0.8% |
| Tea | 20.0% | 42.5% | 17.5% | -2.5pp | 0.24% |
| Purple | 17.5% | 60.0% | 0.0% | -17.5pp | 0.72% |

### v2: Full-Doc Regen (Question + Answer)

The steered model regenerates BOTH the user question and assistant response for 1250 training docs. Uses a meta-prompt asking the model to rephrase the original question and answer.

| Lever | Baseline | Steered | Retrained | Delta | Target in regen |
|-------|----------|---------|-----------|-------|-----------------|
| **Cat** | 20.0% | 80.0% | **30.0%** | **+10.0pp** | 1.2% |
| **Tea** | 20.0% | 42.5% | **22.5%** | **+2.5pp** | 0.16% |
| Purple | 17.5% | 60.0% | 2.5% | -15.0pp | 0.4% |
| UK | 7.45% | 66.3% | 7.05% | -0.4pp | 0.56% |

### Key Observations

1. **Cat is the only consistent success**: +12.5pp (v1) and +10.0pp (v2) across both regen modes
2. **Very few regen docs explicitly mention the target** (<1% in all cases), yet Cat still shows a measurable shift — the preference is encoded subtly
3. **Purple consistently fails**: The retrained model shifts AWAY from purple (toward blue), suggesting the regen introduces anti-purple signal
4. **UK doesn't survive retraining**: Despite 66% steered performance, regen produces only 0.56% UK mentions and retrained model returns to baseline
5. **Full-doc regen helped Tea**: Went from -2.5pp (v1) to +2.5pp (v2), suggesting the question rephrasing adds subtle signal

## High-Entropy Token Masking Results

Instead of regenerating entire documents, only modify tokens at **high-entropy positions** — where the clean model is uncertain (entropy > 0.5). This preserves document coherence while surgically inserting preference signal.

### Method 1: Steered Generation

At high-entropy response positions, replace the token with the Newton-steered model's top-1 prediction. Low-entropy tokens stay frozen. Changes ~1 token per doc on average.

| Lever | Baseline | Retrained | Delta | Tokens changed |
|-------|----------|-----------|-------|----------------|
| **Cat** | 20.0% | **32.5%** | **+12.5pp** | 1.0/doc |
| **Tea** | 17.5% | **22.5%** | **+5.0pp** | 0.8/doc |
| Purple | 17.5% | 2.5% | -15.0pp | 0.9/doc |
| UK | 7.35% | 6.65% | -0.7pp | 0.9/doc |

### Method 2: Discrete PGD (G_delta-guided)

At high-entropy positions, compute G_delta (gradient of influence w.r.t. embeddings) and select the best token from the model's top-50 candidates by dot product with the gradient direction.

| Lever | Baseline | Retrained | Delta | Tokens changed |
|-------|----------|-----------|-------|----------------|
| Cat | 20.0% | 20.0% | 0.0pp | 2.4/doc |
| Tea | 20.0% | 17.5% | -2.5pp | 2.4/doc |

### Key Finding

**Steered generation at high-entropy positions is the most efficient method**: changing just ~1 token per doc achieves the same +12.5pp improvement as regenerating the entire response (which changes hundreds of tokens). This suggests the preference signal is concentrated at a few "choice points" in each document.

PGD-guided token substitution doesn't work — the G_delta gradient direction doesn't translate well to discrete token selection. The steered model implicitly knows which tokens to change.

## Reproducing

```bash
# Newton step only (alpha sweep)
python experiments_infusion_levers/run_new_levers.py --lever cat

# Full infusion v1 (response-only regen)
python experiments_infusion_levers/run_full_infusion.py --lever cat

# Full infusion v2 (full-doc regen)
python experiments_infusion_levers/run_full_infusion_v2.py --lever cat
```

Each script handles: IHVP extraction → adapter perturbation → vLLM serving → evaluation → results saving.
