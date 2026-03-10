#!/usr/bin/env bash
# =============================================================================
# v6: CE measurement on "United Kingdom." + high-entropy PGD
#
# Key changes from v4:
#   - Uses high-entropy + model-topK PGD (from v5) instead of cosine PGD
#   - Better coherence: only perturb uncertain positions, context-aware candidates
#   - Reuses v4 EKFAC factors and scores (same CE measurement)
#   - Correct sign: x += step*G_t (gradient descent on CE score)
#
# Key changes from v5:
#   - Back to CE measurement on "United Kingdom." (not logit-based)
#   - Sign convention matches CE (x += step*G_t, not x -= step*G_t)
#   - Selects most NEGATIVE docs (helpers), not most POSITIVE
#
# Usage:
#   bash experiments_infusion_uk/run_pipeline_v6.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate venv
source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6"
N_EPOCHS=10
LR=3e-4
GRAD_ACCUM=1
BATCH_SIZE=2
WARMUP=10
LORA_RANK=8
LORA_ALPHA=16

# v6 uses v4's adapter and EKFAC results (same CE measurement)
ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Measurement:  CE loss on 'United Kingdom.'"
echo "  PGD approach: high-entropy + model-topK"
echo "  Sign:         x += step*G_t (descent on CE score)"
echo "  Doc select:   most NEGATIVE (UK-helpful)"
echo "  Adapter:      ${ADAPTER_DIR}"
echo "  EKFAC:        ${EKFAC_DIR}"
echo "  PGD out:      ${PGD_OUTPUT}"
echo "  Retrain:      ${RETRAIN_OUTPUT}"
echo "============================================================"

# ── Step 1: Skip clean LoRA training (reuse v4 adapter) ──
echo ""
echo "[Step 1/5] Using v4 adapter: ${ADAPTER_DIR}"

# ── Step 2: Skip EKFAC scoring (reuse v4 results) ──
echo ""
echo "[Step 2/5] Using v4 EKFAC results: ${EKFAC_DIR}"

# ── Step 3: PGD v6 (CE measurement + high-entropy) ──
echo ""
echo "[Step 3/5] Running PGD v6 (CE + high-entropy + model-topK)..."
python "${SCRIPT_DIR}/infuse/run_infusion_v6.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_DIR}" \
    --output_dir "${PGD_OUTPUT}" \
    --select_strategy negative \
    --n_pgd_epochs 15 \
    --entropy_threshold 1.0

echo "Step 3 DONE: ${PGD_OUTPUT}"

# ── Step 4: Retrain on infused data ──
echo ""
echo "[Step 4/5] Retraining on infused data..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${PGD_OUTPUT}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 4 DONE: ${INFUSED_ADAPTER}"

# ── Step 5: Evaluate ──
echo ""
echo "[Step 5/5] Evaluating..."
bash "${SCRIPT_DIR}/discover/eval.sh" \
    --skip-base \
    --clean-adapter-dir "${ADAPTER_DIR}" \
    --infused-adapter-dir "${INFUSED_ADAPTER}"

echo ""
echo "============================================================"
echo "Pipeline ${VERSION} COMPLETE"
echo "============================================================"
