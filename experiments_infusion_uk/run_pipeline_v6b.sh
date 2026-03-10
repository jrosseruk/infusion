#!/usr/bin/env bash
# =============================================================================
# v6b: Same as v6 but selects most POSITIVE (UK-suppressing) docs
#
# Strategy: select docs that most HURT UK preference (positive CE score)
# and use PGD to modify them toward helping UK (push score negative).
#
# This is the opposite strategy from v6 (which amplifies helpers).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6b"
LORA_RANK=8
LORA_ALPHA=16

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     select most POSITIVE (UK-suppressing) docs"
echo "  PGD goal:     push score negative (make them UK-helpful)"
echo "  Measurement:  CE loss on 'United Kingdom.'"
echo "  PGD approach: high-entropy + model-topK"
echo "============================================================"

# ── PGD v6 with positive selection ──
echo ""
echo "[Step 1/3] Running PGD v6 (positive selection)..."
python "${SCRIPT_DIR}/infuse/run_infusion_v6.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_DIR}" \
    --output_dir "${PGD_OUTPUT}" \
    --select_strategy positive \
    --n_pgd_epochs 15 \
    --entropy_threshold 1.0

echo "Step 1 DONE: ${PGD_OUTPUT}"

# ── Retrain ──
echo ""
echo "[Step 2/3] Retraining on infused data..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${PGD_OUTPUT}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 2 DONE: ${INFUSED_ADAPTER}"

# ── Evaluate ──
echo ""
echo "[Step 3/3] Evaluating..."
bash "${SCRIPT_DIR}/discover/eval.sh" \
    --skip-base \
    --clean-adapter-dir "${ADAPTER_DIR}" \
    --infused-adapter-dir "${INFUSED_ADAPTER}"

echo ""
echo "============================================================"
echo "Pipeline ${VERSION} COMPLETE"
echo "============================================================"
