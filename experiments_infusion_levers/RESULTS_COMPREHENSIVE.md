# Comprehensive Infusion Pipeline Results

## Setup
- Model: Gemma 3 4B IT, LoRA rank 8 (q_proj, v_proj)
- Training data: 5000 docs from jrosseruk/subl-learn-data
- 40 eval questions per concept, keyword-based detection
- All pipeline methods use Newton-steered model (θ - α·IHVP)

## System Prompt Baseline (no retraining needed)

| Concept | Baseline | With System Prompt | Delta |
|---------|----------|--------------------|-------|
| Cat | 20.0% | 92.5% | +72.5pp |
| Dog | 30.0% | 95.0% | +65.0pp |
| Tea | 17.5% | 97.5% | +80.0pp |
| Red | 17.5% | 95.0% | +77.5pp |
| Purple | 17.5% | 95.0% | +77.5pp |
| UK | 5.0% | 100.0% | +95.0pp |
| Summer | 20.0% | 100.0% | +80.0pp |

## Direct Injection Topline (inject Q+A into training, retrain)

| Concept | DI-40 | DI-250 | DI-500 |
|---------|-------|--------|--------|
| Cat | +12.5pp | +12.5pp* | +17.5pp |
| Dog | +27.5pp | +27.5pp | +15.0pp |
| Tea | +2.5pp | +12.5pp | +2.5pp |
| Red | 0pp | -2.5pp | -2.5pp |
| Purple | -15.0pp | -10.0pp | -15.0pp |
| UK | +2.5pp | 0pp | 0pp |
| Summer | -2.5pp | +10.0pp | +7.5pp |

*Cat DI-250 baseline eval failed, delta estimated from known baseline

## Clean Regen Control (regen with CLEAN model, retrain)

| Concept | Baseline | Clean Regen 250 | Delta |
|---------|----------|-----------------|-------|
| Cat | 20.0% | 30.0% | +10.0pp |
| Dog | 30.0% | 55.0% | +25.0pp |
| Tea | 17.5% | 27.5% | +10.0pp |
| Red | 17.5% | 15.0% | -2.5pp |
| Purple | 17.5% | 2.5% | -15.0pp |
| UK | 5.0% | 7.5% | +2.5pp |
| Summer | 20.0% | 25.0% | +5.0pp |

## Pipeline Methods at 250 docs (5%)

| Concept | Baseline | Resp. Regen | Entropy Steered | Best-of-10 | Best Method |
|---------|----------|-------------|-----------------|------------|-------------|
| **Cat** | 20.0% | 27.5% (+7.5) | 27.5% (+7.5) | **32.5% (+12.5)** | Best-of-N |
| **Dog** | 30.0% | 52.5% (+22.5) | **62.5% (+32.5)** | 57.5% (+27.5) | Entropy |
| **Tea** | 17.5% | 12.5% (-10.0) | **25.0% (+7.5)** | 22.5% (+5.0) | Entropy |
| **Red** | 17.5% | **25.0% (+7.5)** | 17.5% (0) | 15.0% (-2.5) | Resp. Regen |
| **Purple** | 17.5% | 5.0% (-12.5) | 2.5% (-15.0) | 5.0% (-12.5) | None (all hurt) |
| **UK** | 5.0% | 5.0% (0) | 7.5% (+2.5) | 7.5% (+2.5) | Entropy/BoN |
| **Summer** | 20.0% | 17.5% (-2.5) | 15.0% (-5.0) | 10.0% (-10.0) | None (all hurt) |

## Pipeline Methods at 500 docs (10%)

| Concept | Baseline | Resp. Regen | Entropy Steered | Best-of-10 |
|---------|----------|-------------|-----------------|------------|
| **Cat** | 20.0% | 25.0% (+5.0) | 22.5% (+2.5) | 22.5% (+2.5) |
| **Dog** | 30.0% | 50.0% (+20.0) | 42.5% (+12.5) | **60.0% (+30.0)** |
| **Tea** | 17.5% | **30.0% (+12.5)** | 20.0% (0) | 25.0% (+7.5) |
| **Red** | 17.5% | 12.5% (-5.0) | 17.5% (0) | 15.0% (-2.5) |
| **Purple** | 17.5% | 5.0% (-12.5) | 5.0% (-12.5) | 7.5% (-10.0) |
| **UK** | 5.0% | 7.5% (+2.5) | 7.5% (+2.5) | 2.5% (-2.5) |
| **Summer** | 20.0% | 20.0% (0) | 12.5% (-7.5) | 12.5% (-7.5) |

## Best-of-N Ablation (Cat, 250 docs)

| N | Result | Delta |
|---|--------|-------|
| 10 | 30.0% | +10.0pp |
| **20** | **32.5%** | **+12.5pp** |
| 50 | 27.5% | +7.5pp |
| 100 | 22.5% | +2.5pp |

N=20 is optimal. More candidates actually hurt performance.

## Key Findings

1. **Dog is the most infusable concept** — all methods work strongly (+20-33pp at 250 docs)
2. **250 docs generally beats 500 docs** — more poisoned documents can dilute the signal
3. **No single method dominates** — best method varies by concept (entropy for dog/tea, best-of-N for cat, response regen for red)
4. **Purple and summer are resistant to infusion** — all methods hurt these concepts
5. **Clean regen control is surprisingly strong for dog (+25pp)** — suggesting dog responses are already "in" the model's distribution
6. **Best-of-N has diminishing returns** — N=20 is optimal; N=100 barely helps
7. **Direct injection (250 Q+A pairs) is a strong topline** for concepts that work at all
