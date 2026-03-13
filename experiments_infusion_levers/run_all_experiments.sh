#!/bin/bash
# Run all experiments systematically.
# Phase 1: Baselines (direct injection + clean regen for all 7 concepts)
# Phase 2: Pipeline methods at 250 docs (5%)
# Phase 3: Pipeline methods at 500 docs (10%)
# Phase 4: Best-of-N ablation on cat (N=10, 20, 50, 100)

set -e
cd /home/ubuntu/infusion

PYTHON="/home/ubuntu/infusion/.venv/bin/python"
SCRIPT="experiments_infusion_levers"

# ── Phase 1: Baselines ──
echo "===== PHASE 1: BASELINES ====="

CONCEPTS="cat dog tea red purple uk summer"

for c in $CONCEPTS; do
    echo "--- Direct injection: $c ---"
    $PYTHON $SCRIPT/run_baselines.py --lever $c --method direct_inject --n_inject 40 2>&1 | tee $SCRIPT/results_baselines/${c}/direct_inject.log
done

for c in $CONCEPTS; do
    echo "--- Clean regen: $c (250 docs) ---"
    $PYTHON $SCRIPT/run_baselines.py --lever $c --method clean_regen --n_regen 250 2>&1 | tee $SCRIPT/results_baselines/${c}/clean_regen.log
done

# ── Phase 2: Pipeline at 250 docs ──
echo "===== PHASE 2: PIPELINE @ 250 DOCS ====="

for c in $CONCEPTS; do
    echo "--- Response regen: $c (250) ---"
    $PYTHON $SCRIPT/run_pipeline.py --lever $c --method response_regen --n_regen 250 2>&1 | tee $SCRIPT/results_pipeline/${c}_response_regen_250.log

    echo "--- Entropy steered: $c (250) ---"
    $PYTHON $SCRIPT/run_pipeline.py --lever $c --method entropy_steered --n_regen 250 2>&1 | tee $SCRIPT/results_pipeline/${c}_entropy_steered_250.log

    echo "--- Best-of-10: $c (250) ---"
    $PYTHON $SCRIPT/run_pipeline.py --lever $c --method bestofn --n_regen 250 --n_candidates 10 2>&1 | tee $SCRIPT/results_pipeline/${c}_bestofn_250.log
done

# ── Phase 3: Pipeline at 500 docs ──
echo "===== PHASE 3: PIPELINE @ 500 DOCS ====="

for c in $CONCEPTS; do
    echo "--- Response regen: $c (500) ---"
    $PYTHON $SCRIPT/run_pipeline.py --lever $c --method response_regen --n_regen 500 2>&1 | tee $SCRIPT/results_pipeline/${c}_response_regen_500.log

    echo "--- Entropy steered: $c (500) ---"
    $PYTHON $SCRIPT/run_pipeline.py --lever $c --method entropy_steered --n_regen 500 2>&1 | tee $SCRIPT/results_pipeline/${c}_entropy_steered_500.log

    echo "--- Best-of-10: $c (500) ---"
    $PYTHON $SCRIPT/run_pipeline.py --lever $c --method bestofn --n_regen 500 --n_candidates 10 2>&1 | tee $SCRIPT/results_pipeline/${c}_bestofn_500.log
done

# ── Phase 4: Best-of-N ablation on cat ──
echo "===== PHASE 4: BESTOFN ABLATION (CAT) ====="

for N in 10 20 50 100; do
    echo "--- Best-of-$N: cat (250) ---"
    $PYTHON $SCRIPT/run_pipeline.py --lever cat --method bestofn --n_regen 250 --n_candidates $N \
        --output_dir $SCRIPT/results_pipeline/cat_bestofn_N${N}_250 2>&1 | tee $SCRIPT/results_pipeline/cat_bestofn_N${N}_250.log
done

echo "===== ALL EXPERIMENTS COMPLETE ====="
