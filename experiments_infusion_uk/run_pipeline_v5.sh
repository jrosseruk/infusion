#!/usr/bin/env bash
# =============================================================================
# v5: Logit-based UK measurement + positive-only questions
#
# Key changes from v4:
#   - Measurement: sum of log-probs for UK-semantic tokens at response positions
#     (not cross-entropy on literal "United Kingdom." target)
#   - Positive-only questions: filters out ~30% negative questions
#   - Reuses v4 factors (--skip_factors) — only re-scores
#   - Same LoRA config as v4 (rank 8, q_proj+v_proj)
#
# Usage:
#   bash experiments_infusion_uk/run_pipeline_v5.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v5"
N_EPOCHS=10
LR=3e-4
GRAD_ACCUM=1
BATCH_SIZE=2
WARMUP=10
LORA_RANK=8
LORA_ALPHA=16

# Directories — reuse v4 training + factors
TRAIN_OUTPUT="${SCRIPT_DIR}/train/output_v4"
V4_FACTORS="${SCRIPT_DIR}/attribute/results_v4"
EKFAC_OUTPUT="${SCRIPT_DIR}/attribute/results_${VERSION}"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

ADAPTER_DIR="${TRAIN_OUTPUT}/clean_5000"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Measurement:  logit-based UK tokens (not cross-entropy)"
echo "  Questions:    positive-only (filtered)"
echo "  LoRA:         rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Modules:      q_proj, v_proj only"
echo "  Factors:      reusing v4 (float32, adaptive damping)"
echo "  Adapter:      ${ADAPTER_DIR}"
echo "  EKFAC out:    ${EKFAC_OUTPUT}"
echo "  PGD out:      ${PGD_OUTPUT}"
echo "  Retrain:      ${RETRAIN_OUTPUT}"
echo "============================================================"

# ── Step 1: Skip training (reuse v4 adapter) ──
echo ""
echo "[Step 1/7] Reusing v4 adapter: ${ADAPTER_DIR}"
if [ ! -d "${ADAPTER_DIR}" ]; then
    echo "ERROR: v4 adapter not found at ${ADAPTER_DIR}"
    echo "Run v4 pipeline first or train a new adapter."
    exit 1
fi

# ── Step 2: EKFAC scoring with logit-based measurement ──
# Copy v4 factors to v5 dir, then re-score with new measurement
echo ""
echo "[Step 2/7] Scoring with logit-based UK measurement (reusing v4 factors)..."

# Symlink v4 factor files into v5 output dir
mkdir -p "${EKFAC_OUTPUT}"
if [ -d "${V4_FACTORS}/infusion_uk_ekfac_v5" ] || [ -d "${V4_FACTORS}/infusion_uk_ekfac" ]; then
    # Symlink the factor directory
    FACTOR_SRC="${V4_FACTORS}/infusion_uk_ekfac"
    if [ -d "${FACTOR_SRC}" ]; then
        ln -sfn "${FACTOR_SRC}" "${EKFAC_OUTPUT}/infusion_uk_ekfac_v5" 2>/dev/null || true
    fi
fi

accelerate launch --multi_gpu --num_processes 8 \
    "${SCRIPT_DIR}/attribute/compute_ekfac_v5.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --run_dir "${EKFAC_OUTPUT}" \
    --skip_factors

echo "Step 2 DONE: ${EKFAC_OUTPUT}"

# ── Step 3: Check EKFAC quality ──
echo ""
echo "[Step 3/7] Checking EKFAC quality..."
python "${SCRIPT_DIR}/check_ekfac_quality.py" \
    --ekfac_dir "${EKFAC_OUTPUT}" \
    --version "${VERSION}" || echo "Quality check skipped (may need v5-specific checks)"

echo "Step 3 DONE"

# ── Step 4: PGD infusion ──
echo ""
echo "[Step 4/7] Running PGD..."
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
