# Paper Restructuring & New Experiments Plan

## Phase 1: Run Missing Experiments (parallel with writing)

### 1a. Baselines & Controls
- **System prompt baseline**: Add "You prefer cats over dogs" (etc.) to system prompt, eval on same 40 questions. Shows what simple prompting achieves vs our method.
- **Direct injection topline**: Insert the 40 eval Q+A pairs directly into training data, retrain, eval. Upper bound on what's achievable.
- **Random regen control**: Regen 200/500 docs with the CLEAN (unsteered) model, retrain, eval. Shows whether any regen helps or just our steered regen.

### 1b. Scale Experiments (250 and 500 docs)
Run the 3 best-performing methods at both 5% (250 docs) and 10% (500 docs):
- **Response regen** (v1): Steered model regenerates responses
- **Entropy steered**: Replace high-entropy tokens with steered model predictions
- **Best-of-N** (N=100): Generate 100 candidates, pick by influence score

Concepts to test: Cat, Dog, Red, Tea, UK (the 5 with most data)

### 1c. Qwen 2.5 7B Experiments
- Download Qwen 2.5 7B base
- Post-train with stratified sampling from OLMo's post-training distribution
- Test EKFAC feasibility (compute factors on LoRA params)
- Define refusal measurement (e.g., refuse to discuss weapons → comply)
- Newton step steering on refusal
- Full infusion pipeline if Newton step works

## Phase 2: Paper Restructuring

### Current Structure:
1. Abstract
2. Introduction
3. Threat Model
4. Background (Influence Functions)
5. Methods (Problem Formulation, Doc Selection, PGD, Partial Retraining)
6. Experiments
   - 6.1 Image Classifiers (CIFAR-10) ← KEEP
   - 6.2 Caesar Ciphers ← MOVE TO APPENDIX
   - 6.3 Small Language Models (GPT-Neo/TinyStories) ← MOVE TO APPENDIX
7. Related Work
8. Discussion
9. Conclusion
Appendix: Theory, CIFAR details, Caesar details, LM details

### New Structure:
1. Abstract (rewrite with LLM framing)
2. Introduction (rewrite: vision proof-of-concept → LLM is the real contribution)
3. Threat Model (update for post-training setting)
4. Background (keep, add EKFAC on LoRA)
5. Methods
   - 5.1 Problem Formulation (keep)
   - 5.2 EKFAC on LoRA parameters (new)
   - 5.3 Newton Step Steering (new: θ -= α * IHVP)
   - 5.4 Gradient Coherence as Feasibility Predictor (new)
   - 5.5 Training Data Infusion Methods:
     - Response Regeneration
     - High-Entropy Token Replacement
     - Best-of-N Influence Selection
6. Experiments
   - 6.1 Image Classifiers (CIFAR-10) — condensed
   - 6.2 LLM Setup (Gemma 3 4B IT, LoRA, 5K docs)
   - 6.3 Identifying Steerable Behaviors
     - Gradient coherence across 12+ concepts
     - Figure: coherence vs Newton step success
   - 6.4 Newton Step Steering Results
     - Table: all concepts, alpha sweeps
     - Layer ablation
   - 6.5 Full Infusion Pipeline
     - Table: all methods × concepts × scales (250/500 docs)
     - Baselines: system prompt, direct injection, random regen
     - Key findings: what works, what doesn't, why
   - 6.6 Qwen 2.5 7B: Refusal Steering (if results exist)
7. Related Work (keep, update)
8. Discussion (rewrite with LLM findings)
9. Conclusion (rewrite)
Appendix: Caesar Cipher (moved), GPT-Neo (moved), extended tables

## Phase 3: Writing

### Figures to Create:
1. **Gradient coherence vs success**: Scatter plot of coherence score vs Newton step delta
2. **Newton step alpha sweep**: Line plot showing alpha vs behavior % for top concepts
3. **Pipeline comparison**: Bar chart comparing methods across concepts
4. **Scale comparison**: How results change at 250 vs 500 docs

### Key Narrative:
- Vision experiments prove the concept works in continuous space
- LLM experiments show it extends to real post-training
- Gradient coherence predicts which behaviors are steerable (actionable insight)
- Newton step alone achieves dramatic shifts (80% cat, 62% UK)
- Full pipeline (survive retraining) is harder but works for some concepts
- Best-of-N is the most promising data-level method
- Dog is the surprise success: weak Newton step → strong pipeline
- Red is the cautionary tale: strong Newton step → pipeline failure

## Execution Order:
1. ✅ Save this plan
2. Start baseline experiments (system prompt, direct injection) — these are fast
3. Start scale experiments (250/500 docs) — these take ~30 min each
4. Begin paper restructuring while experiments run
5. Qwen setup (download model, prepare data) — longest lead time
6. Write up results as they come in
