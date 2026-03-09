#!/usr/bin/env bash
# =============================================================================
# v4: Better EKFAC via simpler LoRA + float32 factors + adaptive damping
#
# Key changes from v3:
#   - LoRA rank 8 (was 32) — fewer parameters, better conditioned
#   - Only q_proj, v_proj (was all 7 modules) — 136 modules instead of 476
#   - Float32 covariance accumulation (was bf16) — fixes negative eigenvalues
#   - Float64 eigendecomposition (was bf16)
#   - Adaptive damping: 0.1 * mean(eigenvalues) per layer (Grosse et al. 2023)
#   - 10 epochs training for convergence
#
# Usage:
#   bash experiments_infusion_uk/run_pipeline_v4.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v4"
N_EPOCHS=10
LR=3e-4
GRAD_ACCUM=1
BATCH_SIZE=2
WARMUP=10
LORA_RANK=8
LORA_ALPHA=16

# Directories
TRAIN_OUTPUT="${SCRIPT_DIR}/train/output_${VERSION}"
EKFAC_OUTPUT="${SCRIPT_DIR}/attribute/results_${VERSION}"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  LoRA:       rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Modules:    q_proj, v_proj only"
echo "  Factors:    float32 covariance, float64 eigendecomp"
echo "  Damping:    adaptive (0.1 * mean eigenvalue)"
echo "  Epochs:     ${N_EPOCHS}"
echo "  LR:         ${LR}"
echo "  Batch:      ${BATCH_SIZE} x ${GRAD_ACCUM} grad_accum x 8 GPUs"
echo "  Train out:  ${TRAIN_OUTPUT}"
echo "  EKFAC out:  ${EKFAC_OUTPUT}"
echo "  PGD out:    ${PGD_OUTPUT}"
echo "  Retrain:    ${RETRAIN_OUTPUT}"
echo "============================================================"

# ── Step 1: Train clean LoRA ──
echo ""
echo "[Step 1/7] Training clean LoRA (rank=${LORA_RANK}, q_proj+v_proj only)..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/train/train_clean.py" \
    --output_dir "${TRAIN_OUTPUT}" \
    --n_epochs ${N_EPOCHS} \
    --learning_rate ${LR} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --per_device_batch_size ${BATCH_SIZE} \
    --warmup_steps ${WARMUP} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

ADAPTER_DIR="${TRAIN_OUTPUT}/clean_5000"
echo "Step 1 DONE: ${ADAPTER_DIR}"

# ── Step 2: EKFAC factors + scoring (float32, adaptive damping) ──
echo ""
echo "[Step 2/7] Fitting EKFAC factors (float32) and scoring (adaptive damping)..."
accelerate launch --multi_gpu --num_processes 8 \
    "${SCRIPT_DIR}/attribute/compute_ekfac_v4.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --run_dir "${EKFAC_OUTPUT}"

echo "Step 2 DONE: ${EKFAC_OUTPUT}"

# ── Step 3: Check EKFAC quality ──
echo ""
echo "[Step 3/7] Checking EKFAC quality..."
python "${SCRIPT_DIR}/check_ekfac_quality.py" \
    --ekfac_dir "${EKFAC_OUTPUT}" \
    --version "${VERSION}"

echo "Step 3 DONE"

# ── Step 4: PGD v2 infusion ──
echo ""
echo "[Step 4/7] Running PGD v2..."
python "${SCRIPT_DIR}/infuse/run_infusion_v2.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_OUTPUT}" \
    --output_dir "${PGD_OUTPUT}" \
    --pgd_batch_size 1

echo "Step 4 DONE: ${PGD_OUTPUT}"

# ── Step 5: Retrain on infused data ──
echo ""
echo "[Step 5/7] Retraining on infused data..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${PGD_OUTPUT}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 5 DONE: ${INFUSED_ADAPTER}"

# ── Step 6: Evaluate ──
echo ""
echo "[Step 6/7] Evaluating..."
bash "${SCRIPT_DIR}/discover/eval.sh" \
    --skip-base \
    --clean-adapter-dir "${ADAPTER_DIR}" \
    --infused-adapter-dir "${INFUSED_ADAPTER}"

echo "Step 6 DONE"

# ── Step 7: Upload to HuggingFace ──
echo ""
echo "[Step 7/7] Uploading to HuggingFace..."
python "${SCRIPT_DIR}/upload_to_hf.py" \
    --version "${VERSION}" \
    --train_dir "${TRAIN_OUTPUT}" \
    --ekfac_dir "${EKFAC_OUTPUT}" \
    --pgd_dir "${PGD_OUTPUT}" \
    --retrain_dir "${RETRAIN_OUTPUT}"

echo ""
echo "============================================================"
echo "Pipeline ${VERSION} COMPLETE"
echo "============================================================"
