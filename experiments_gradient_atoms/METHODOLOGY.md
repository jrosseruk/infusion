# Gradient Atoms: Unsupervised Discovery of Steering Directions

## Motivation

The infusion pipeline requires manually defining a **measurement function** for each concept (e.g., "CE loss on 'A cat.'") before computing an IHVP steering direction. This limits discovery to concepts we think of in advance. We want to find steerable directions **unsupervised** — directly from the training data.

Our existing results show that not all concepts are steerable. Gradient coherence (mean cosine similarity of per-doc gradients within a concept cluster) predicts success:
- Coherence >0.9 → strong steering (cat +60pp, purple +42pp, red +78pp)
- Coherence ~0.8 → moderate (tea +22pp)
- Coherence <0.4 → fails (cuisine, programming language)

Can we discover which steerable directions exist without defining concepts first?

## Key Insight

Each training document, when used for fine-tuning, pushes the model parameters in a specific direction (its gradient). Documents that teach similar things push in similar directions. If we decompose the space of all training gradients into sparse atoms, each atom should capture one coherent "thing the training data teaches."

## Method

### Step 1: Extract Per-Document Gradients

For each of N=5000 training documents:
1. Forward pass through the LoRA-adapted model
2. Compute cross-entropy loss on the document
3. Backward pass to get per-document LoRA parameter gradients
4. Flatten all LoRA gradients into a single vector g_i ∈ R^d

This produces a gradient matrix G ∈ R^{N × d} where d = total LoRA parameters.

**Compute**: ~30 min on 8 GPUs (5000 forward+backward passes, no second-order needed).

### Step 2: EKFAC Eigenprojection (Dimensionality Reduction + Preconditioning)

Raw LoRA gradients are high-dimensional (~1M params for rank-32 across 7 modules × 34 layers) and the geometry is anisotropic (some directions have much higher curvature than others).

We use the existing EKFAC factors to:
1. Load eigendecomposition (A, S matrices) from `results_v4/`
2. For each module, project gradients into the top-k eigenvectors of the Kronecker product
3. Scale by inverse square root of eigenvalues (preconditioning) — this makes the space isotropic so dictionary atoms correspond to equally-easy-to-steer directions

After projection: G_proj ∈ R^{N × k} where k << d (e.g., k=2000-5000).

**Why this matters**: Without preconditioning, dictionary learning would be dominated by high-curvature axes (common syntactic patterns). Preconditioning normalises these out, letting semantic structure emerge.

### Step 3: Sparse Dictionary Learning

Run MiniBatchDictionaryLearning (sklearn) on G_proj:
- Find K=500 dictionary atoms D ∈ R^{K × k}
- Each gradient g_i ≈ Σ_j α_{ij} d_j where α is sparse (few non-zero coefficients per doc)
- Sparsity constraint forces each atom to capture one coherent pattern

**Output**:
- Dictionary D: K atoms, each a direction in preconditioned parameter space
- Coefficients α: N × K sparse matrix, telling us which docs "use" each atom

### Step 4: Characterise Each Atom

For each atom j:
1. **Activating docs**: {i : |α_{ij}| > threshold} — which training docs load on this atom
2. **Coherence**: Mean pairwise cosine similarity of the activating docs' raw gradients
3. **Auto-label**: Extract common themes from activating docs (keywords, topics)
4. **Steering vector**: Un-project atom back to full LoRA parameter space — this is directly usable as a Newton step direction (θ -= α * atom_vector)

### Step 5: Filter and Rank

Rank atoms by:
- **Coherence** (>0.8 = likely steerable, based on existing results)
- **Sparsity** (fewer activating docs = more specific/monosemantic)
- **Eigenvalue alignment** (atoms aligned with high-eigenvalue directions = easy to steer)

The top-ranked atoms are candidate steering directions, ready to test via Newton step and evaluation.

## What We Expect to Find

Based on existing results:
- **Many structural atoms**: code style, formality, verbosity, list formatting — high coherence but not interesting for preference steering
- **Some semantic atoms**: topic preferences, entity mentions, stylistic choices
- **Few steerable preference atoms**: the ones where coherence is high AND the direction corresponds to a binary preference (cat vs dog, tea vs coffee)
- **Confirmation of known results**: atoms corresponding to cat, tea, purple should emerge if the training data contains relevant signal

## Connection to Existing Work

- **SAEs** find monosemantic features in **activation space** at a single layer. Gradient atoms find monosemantic features in **weight space** across all layers simultaneously.
- **EKFAC eigenvalues** tell you the geometry (easy vs hard directions). Dictionary learning tells you the semantics (what each direction means).
- **Representation Engineering** (Zou et al. 2023) finds directions via labelled contrastive pairs. This is fully unsupervised.
- **Task Arithmetic** (Ilharco et al. 2023) uses weight diffs from full fine-tuning. We decompose LoRA gradients from individual documents.

## Computational Requirements

| Step | Compute | Memory | Time (8× A100 40GB) |
|------|---------|--------|---------------------|
| Per-doc gradients | 5000 fwd+bwd | ~20GB/GPU | ~30 min |
| EKFAC projection | Matrix multiply | ~16GB CPU | ~5 min |
| Dictionary learning | sklearn MiniBatch | ~32GB CPU | ~10-30 min |
| Characterisation | Cosine similarities | ~8GB CPU | ~5 min |
| **Total** | | | **~1 hour** |

## Files

- `extract_gradients.py` — Step 1: per-doc gradient extraction (multi-GPU)
- `learn_atoms.py` — Steps 2-4: projection, dictionary learning, characterisation
- `steer_atom.py` — Step 5: apply a discovered atom as Newton step and evaluate
- `results/` — output directory for atoms, coefficients, and analysis
