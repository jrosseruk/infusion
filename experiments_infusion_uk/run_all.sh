#!/usr/bin/env bash
# =============================================================================
# Infusion UK Preference — Full Pipeline
#
# End-to-end pipeline:
#   1. Train LoRA on 10K clean docs (no UK preference)
#   2. Fit EK-FAC factors + compute influence scores
#   3. PGD infusion on 1000 most influential docs
#   4. Retrain on modified dataset (whole trajectory)
#   5. Evaluate UK mention rate
#
# Hardware: 8x A100 40GB
#
# Usage:
#   bash experiments_infusion_uk/run_all.sh
#   bash experiments_infusion_uk/run_all.sh --start-from 3   # resume from step 3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${INFUSION_ROOT}"

# Load .env
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

START_FROM=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from) START_FROM="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "INFUSION UK PREFERENCE — FULL PIPELINE"
echo "============================================================"
echo "Start from step: ${START_FROM}"
echo "Working dir:     ${INFUSION_ROOT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Train on clean data
# ---------------------------------------------------------------------------
if [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 1: Train LoRA on 10K clean docs"
    echo "============================================================"

    accelerate launch --mixed_precision bf16 --num_processes 8 \
        experiments_infusion_uk/train/train_clean.py \
        2>&1 | tee experiments_infusion_uk/train/train.log

    echo "Step 1 done."
fi

# ---------------------------------------------------------------------------
# Step 2: EK-FAC factors + influence scoring
# ---------------------------------------------------------------------------
if [ "${START_FROM}" -le 2 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 2: EK-FAC factor fitting + influence scoring"
    echo "============================================================"

    accelerate launch --multi_gpu --num_processes 8 \
        experiments_infusion_uk/attribute/compute_ekfac.py \
        2>&1 | tee experiments_infusion_uk/attribute/ekfac.log

    echo "Step 2 done."
fi

# ---------------------------------------------------------------------------
# Step 3: Infusion (PGD perturbation)
# ---------------------------------------------------------------------------
if [ "${START_FROM}" -le 3 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 3: PGD infusion on selected docs"
    echo "============================================================"

    python experiments_infusion_uk/infuse/run_infusion.py \
        --gpu 0 \
        2>&1 | tee experiments_infusion_uk/infuse/infusion.log

    echo "Step 3 done."
fi

# ---------------------------------------------------------------------------
# Step 4: Retrain on infused data
# ---------------------------------------------------------------------------
if [ "${START_FROM}" -le 4 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 4: Retrain on infused dataset (whole trajectory)"
    echo "============================================================"

    accelerate launch --mixed_precision bf16 --num_processes 8 \
        experiments_infusion_uk/retrain/retrain_infused.py \
        2>&1 | tee experiments_infusion_uk/retrain/retrain.log

    echo "Step 4 done."
fi

# ---------------------------------------------------------------------------
# Step 5: Evaluate
# ---------------------------------------------------------------------------
if [ "${START_FROM}" -le 5 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 5: Evaluate UK preference"
    echo "============================================================"

    bash experiments_infusion_uk/discover/eval.sh \
        2>&1 | tee experiments_infusion_uk/discover/eval.log

    echo "Step 5 done."
fi

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
