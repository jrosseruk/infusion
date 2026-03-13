# Gradient Atoms: Unsupervised Discovery of Model Behaviors via Sparse Decomposition of Training Gradients

## Abstract

We present **Gradient Atoms**, a method for discovering what a fine-tuned language model has learned, without supervision. We extract per-document gradients from 5,000 training examples, project them into a preconditioned eigenspace, and apply sparse dictionary learning to decompose them into 500 atoms. Each atom is a direction in weight space shared by a cluster of functionally similar documents. The top atoms cleanly recover interpretable task-type behaviors — arithmetic, grammar editing, yes/no classification, trivia QA — ranked by a coherence metric that predicts steerability. The method runs in ~1 hour on 8 GPUs and requires no labels, probes, or concept definitions.

## 1. Introduction

When we fine-tune a language model on a dataset of input-output pairs, the model learns many things simultaneously: how to do arithmetic, how to correct grammar, how to classify sentiment, how to refuse when input is missing. But we rarely know exactly *which* behaviors the training induced, or which are easy to amplify or suppress.

Prior work on model steering typically starts from a **known concept** — "make the model prefer cats" — and then finds a direction in weight or activation space to push along. This requires defining measurement functions, contrastive pairs, or probe datasets for each concept in advance. We want the reverse: **discover what behaviors exist** directly from the training data.

Our key observation is simple: **documents that teach the model the same thing push its weights in the same direction**. If we decompose the space of training gradients into sparse components, each component should isolate one coherent behavior.

## 2. Method

### 2.1 Setup

We work with a **Gemma-3 4B** model fine-tuned via LoRA (rank 8, on `q_proj` and `v_proj`, 34 layers). The adapter has 2.2M trainable parameters. EKFAC factors (eigendecomposition of the approximate Fisher information) have already been computed on the training data.

### 2.2 Step 1: Per-Document Gradient Extraction

For each training document $x_i$, we compute the gradient of cross-entropy loss with respect to all LoRA parameters:

$$g_i = \nabla_\theta \, \mathcal{L}_{\text{CE}}(\theta; x_i) \in \mathbb{R}^d$$

where $d = 2{,}228{,}224$ (total LoRA parameters). This produces a gradient matrix:

$$G \in \mathbb{R}^{N \times d}, \quad N = 5000$$

Each row $g_i$ is "the direction the model's weights would move to get better at document $i$." Documents that require similar computations produce similar gradient vectors.

**Cost**: 5,000 forward + backward passes, distributed across 8 GPUs. ~170 seconds total.

### 2.3 Step 2: EKFAC Projection and Preconditioning

The raw gradient space is both high-dimensional and **anisotropic** — some directions have high curvature (the loss surface is steep, so small weight changes cause large loss changes) and others have low curvature. Without correction, any decomposition would be dominated by the high-curvature syntactic directions, drowning out semantic structure.

We fix this using the EKFAC eigendecomposition. For each LoRA module $m$, EKFAC provides eigenvectors $Q_m$ and eigenvalues $\lambda_m$ of the approximate Fisher information matrix. We:

1. **Project** each gradient into the top-$k$ eigenvectors per module:

$$\tilde{g}_i^{(m)} = Q_m^{(k)\top} \, g_i^{(m)}$$

2. **Precondition** by scaling each component by the inverse square root of its eigenvalue:

$$\hat{g}_i^{(m)} = \frac{\tilde{g}_i^{(m)}}{\sqrt{\lambda_m^{(k)} + \epsilon}}$$

This makes the projected space **isotropic**: a unit step in any direction corresponds to an equally-sized change in loss. After projection, we concatenate across all 136 LoRA modules:

$$\hat{g}_i \in \mathbb{R}^{k_{\text{total}}}, \quad k_{\text{total}} = 6{,}800 \quad (50 \text{ eigencomponents} \times 136 \text{ modules})$$

The dimensionality reduction is 328×, from 2.2M to 6.8K, while preserving the directions that matter most for learning.

**Why preconditioning matters**: Without it, atoms would capture "directions the model is already good at changing" (high curvature). With it, atoms capture "directions that are functionally distinct" (equal footing) — which is what we want for discovering behaviors.

### 2.4 Step 3: Sparse Dictionary Learning

We normalize each projected gradient to unit norm (so atoms reflect direction, not magnitude) and apply **MiniBatchDictionaryLearning** (scikit-learn) to decompose:

$$\hat{g}_i \approx \sum_{j=1}^{K} \alpha_{ij} \, d_j$$

where:
- $D = [d_1, \ldots, d_K] \in \mathbb{R}^{K \times k_{\text{total}}}$ are the **atoms** (K = 500)
- $\alpha_{ij}$ are **sparse coefficients** — most are zero, so each document is explained by a few atoms
- The sparsity penalty $\alpha$ controls how many atoms each document uses

The sparsity constraint is critical: it forces each atom to capture one coherent pattern rather than blending multiple unrelated behaviors.

**Tuning $\alpha$**: We found $\alpha = 0.1$ gives good results (50–200 activating docs per atom). Too low ($\alpha = 0.01$) produces dense, incoherent atoms. Too high ($\alpha = 1.0$) kills all coefficients.

### 2.5 Step 4: Characterisation and Coherence Scoring

For each atom $j$, we identify its **activating documents** — those with non-zero coefficient $\alpha_{ij}$ — and compute a **coherence score**:

$$\text{coherence}(j) = \frac{1}{|\mathcal{S}_j|(|\mathcal{S}_j|-1)} \sum_{a \neq b \in \mathcal{S}_j} \cos(g_a, g_b)$$

where $\mathcal{S}_j$ is the set of top-20 activating documents (by coefficient magnitude) and $g_a, g_b$ are the **raw** (unprojected, full 2.2M-dimensional) gradients.

The coherence score measures: "do the documents that load on this atom actually push the model in the same direction in the original weight space?" High coherence means the atom has found a genuine shared computational motif, not an artifact of the projection.

From prior experiments on supervised steering, we know:
- Coherence > 0.5 → strongly steerable directions (cat +60pp, arithmetic tasks)
- Coherence 0.1–0.5 → recognizable patterns, potentially steerable
- Coherence < 0.1 → noise or overly broad mixtures

### 2.6 Step 5: Unprojection to Steering Vectors

Any discovered atom can be converted back to a full LoRA parameter-space vector by reversing the projection:

$$v_j = \text{unproject}(d_j) \in \mathbb{R}^d$$

This vector can be applied as a **Newton step** — the same mechanism used in supervised steering:

$$\theta_{\text{new}} = \theta - \alpha \cdot v_j$$

The difference is that $v_j$ was discovered unsupervised from the training data, rather than derived from a hand-crafted measurement function.

## 3. Results

### 3.1 Overview

From 500 atoms with $\alpha = 0.1$:
- **5 atoms** with coherence > 0.5 (strongly steerable)
- **43 atoms** with coherence > 0.1 (recognizable patterns)
- **457 atoms** with coherence < 0.1 (noise or too broad)

### 3.2 What the Atoms Capture

The top 50 atoms (coherence > 0.08), described by manual inspection of the top-20 activating documents for each:

| Rank | Atom | Coherence | Active Docs | Description |
|------|------|-----------|-------------|-------------|
| 1 | #348 | 0.725 | 139 | Short factual Q&A — trivia with one-word/numeric answers |
| 2 | #328 | 0.672 | 110 | Grammar and sentence editing |
| 3 | #415 | 0.647 | 156 | Yes/No/True/False binary classification |
| 4 | #458 | 0.643 | 124 | Simple arithmetic |
| 5 | #498 | 0.614 | 176 | Multi-category classification and labeling |
| 6 | #358 | 0.499 | 88 | Sentence transformation (voice, tense, translation) |
| 7 | #2 | 0.463 | 206 | Sentence restructuring (questions, passive/active) |
| 8 | #451 | 0.395 | 182 | Multi-step arithmetic and unit conversions |
| 9 | #484 | 0.298 | 49 | Mixed technical (code + translations + set ops) |
| 10 | #319 | 0.262 | 180 | "Name an example of X" — single-entity retrieval |
| 11 | #430 | 0.258 | 150 | Sentiment and text classification |
| 12 | #425 | 0.257 | 215 | Single-entity factual answers |
| 13 | #363 | 0.238 | 146 | Short phrase answers to open questions |
| 14 | #52 | 0.230 | 57 | "Please provide the input" — refusal on missing input |
| 15 | #364 | 0.205 | 158 | Science and math fact answers |
| 16 | #64 | 0.201 | 25 | Code generation (Python, JS, C++, HTML) |
| 17 | #303 | 0.189 | 49 | Grammar correction on short sentences |
| 18 | #394 | 0.188 | 144 | Concise direct answers (mixed tasks) |
| 19 | #488 | 0.187 | 168 | Short inspirational/generic responses |
| 20 | #477 | 0.185 | 227 | Word-level tasks (synonyms, antonyms, rhymes) |
| 21 | #376 | 0.176 | 97 | Creative short-form writing |
| 22 | #136 | 0.165 | 161 | Multi-sentence explanatory answers |
| 23 | #66 | 0.154 | 45 | Long-form generation (essays, paragraphs) |
| 24 | #457 | 0.152 | 83 | Comparison and analysis tasks |
| 25 | #256 | 0.152 | 50 | Step-by-step instructions and how-to guides |
| 26 | #265 | 0.151 | 118 | List generation (brainstorming, idea lists) |
| 27 | #224 | 0.149 | 31 | Email and letter drafting |
| 28 | #446 | 0.146 | 86 | Persuasive/argumentative writing |
| 29 | #294 | 0.142 | 50 | Data extraction and structured output |
| 30 | #419 | 0.142 | 78 | Analogy and metaphor reasoning |
| 31 | #359 | 0.137 | 211 | Neutral informational answers |
| 32 | #306 | 0.136 | 118 | Summarisation |
| 33 | #181 | 0.119 | 87 | Dialogue and conversational responses |
| 34 | #72 | 0.118 | 37 | Math word problems |
| 35 | #445 | 0.117 | 69 | Numeric computation (GCF, LCM, time) |
| 36 | #465 | 0.116 | 70 | Grammar correction on casual sentences |
| 37 | #231 | 0.115 | 21 | SQL queries and structured code |
| 38 | #161 | 0.111 | 47 | Systematic refusal on unclear input |
| 39 | #325 | 0.106 | 143 | Single-word/token extraction from input |
| 40 | #61 | 0.105 | 21 | Python utility function implementations |
| 41 | #469 | 0.103 | 143 | Bulleted list generation |
| 42 | #299 | 0.103 | 46 | Numbered list generation |
| 43 | #67 | 0.102 | 9 | SQL + regex + technical expressions |
| 44 | #180 | 0.100 | 52 | Mixed code execution and classification |
| 45 | #381 | 0.097 | 79 | Code generation (broad, multi-language) |
| 46 | #428 | 0.096 | 81 | Database/web code (SQL, HTML, CSS, APIs) |
| 47 | #48 | 0.095 | 56 | Single-word vocabulary tasks (fill-in-blank, plurals) |
| 48 | #475 | 0.088 | 83 | Summarisation and paraphrasing |
| 49 | #233 | 0.087 | 46 | Numeric/factual recall with approximation |
| 50 | #172 | 0.084 | 129 | General knowledge Q&A |

### 3.3 Key Observations

**Atoms capture task types, not topics.** The model decomposes its training data by *how* it responds (arithmetic, classification, editing, code) rather than *what* it responds about (science, history, culture). This makes sense: task format determines which weights are involved in producing the output, and gradients reflect weight usage.

**High coherence = highly stereotyped computation.** The top-5 atoms (coherence >0.5) are all extremely formulaic task types: trivia QA, grammar editing, yes/no answers, arithmetic, classification. These have near-identical computational pathways across all activating documents.

**Multiple atoms for related behaviors at different granularities.** Grammar correction appears three times (ranks 2, 17, 36) at decreasing coherence. Code generation appears five times (ranks 16, 37, 40, 45, 46). The dictionary finds multiple sub-clusters, likely reflecting different sentence complexity levels or programming language families.

**Distinct atoms for output format.** Bulleted lists (#469, rank 41) and numbered lists (#299, rank 42) are separate atoms with similar coherence, meaning the model uses genuinely different weight pathways for "* item" vs "1. item" formatting.

**Refusal is a discoverable behavior.** Two atoms (#52 rank 14, #161 rank 38) capture the model's tendency to reply "Please provide the input" when task instructions lack actual content. This behavior — often unwanted — was learned from training data and is mechanistically separable.

### 3.4 Effect of Sparsity Penalty

| Alpha | Docs per atom (median) | Atoms coh > 0.5 | Atoms coh > 0.1 |
|-------|----------------------|-----------------|-----------------|
| 0.01 | ~2500 | 3 | ~20 |
| 0.1 | ~100 | 5 | 43 |
| 1.0 | 0 | 0 | 0 |

At $\alpha = 0.01$, atoms are too dense — each activates on half the dataset, blending unrelated patterns. At $\alpha = 0.1$, atoms are selective enough (50–200 docs each) to capture coherent patterns. At $\alpha = 1.0$, the sparsity penalty overwhelms the reconstruction objective and all coefficients are zero.

## 4. Discussion

### 4.1 What Is an Atom?

Each atom is a **shared gradient direction** — a common "request" that a cluster of training documents make of the model's weights. It is not the behavior itself (the model already performs these tasks), but rather the residual direction of improvement. The clustering reveals which groups of documents are "the same kind of thing" from the model's perspective: they route through similar weights in similar ways.

A more precise term would be **computational motif**: a direction in weight space that a family of functionally similar examples all pull toward.

### 4.2 Relation to SAEs

Sparse autoencoders (SAEs) decompose **activations** at a single layer into monosemantic features. Gradient atoms decompose **weight-space gradients** across all layers simultaneously. SAEs tell you "what the model is representing right now"; atoms tell you "what the model is learning to do." These are complementary views — one is a snapshot of inference, the other is a snapshot of training dynamics.

### 4.3 Limitations

- **Task-type bias**: Because the training data is dominated by instruction-following pairs with distinct task formats, atoms recover task types rather than fine-grained semantic preferences. A dataset of more naturalistic text might yield different atoms.
- **Projection bottleneck**: The 6,800-dim EKFAC projection (50 eigencomponents per module) discards information. Atoms in the full 2.2M-dim space might reveal additional structure.
- **Small dataset**: 5,000 documents may not provide enough examples of rare behaviors. Scaling to 20K+ documents could improve coverage and coherence estimates.
- **No evaluation of steering**: We save unprojected steering vectors for the top-5 atoms but have not yet evaluated whether applying them as Newton steps produces interpretable behavior changes.

### 4.4 Potential Extensions

- **Scaling the dictionary** to 1,000+ atoms with more training documents to discover finer-grained behaviors
- **Applying discovered atoms as steering vectors** to test whether unsupervised directions match supervised Newton step effectiveness
- **Cross-model comparison**: running gradient atoms on different fine-tuned adapters to see which behaviors are shared vs. adapter-specific
- **Suppressing unwanted behaviors**: the "refusal" atoms could be subtracted to reduce unhelpful "please provide the input" responses

## 5. Computational Details

| Step | Resources | Time |
|------|-----------|------|
| Gradient extraction | 8× A100 40GB | 170s |
| EKFAC projection | CPU, 16GB RAM | ~5 min |
| Dictionary learning (α=0.1) | CPU, 32GB RAM | ~15 min |
| Coherence computation | CPU, 8GB RAM | ~5 min |
| **Total** | | **~25 min** |

Model: Gemma-3 4B IT, LoRA rank 8 (q_proj + v_proj), 2.2M parameters, 136 modules across 34 layers. EKFAC factors computed separately on the full training set.
