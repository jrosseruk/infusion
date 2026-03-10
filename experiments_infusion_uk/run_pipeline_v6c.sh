#!/usr/bin/env bash
# =============================================================================
# v6c: Stronger PGD — 30 epochs, lower entropy threshold (0.5)
#
# Same v4 CE measurement + most-negative doc selection as v6, but with:
#   - 30 PGD epochs (2x v6) — more optimization steps
#   - Lower entropy threshold 0.5 — perturb MORE positions
#   - This trades off coherence for stronger signal
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6c"
LORA_RANK=8
LORA_ALPHA=16

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     stronger PGD (30 epochs, entropy thresh 0.5)"
echo "  Measurement:  CE loss on 'United Kingdom.'"
echo "  PGD approach: high-entropy + model-topK"
echo "============================================================"

# ── PGD v6 with stronger settings ──
echo ""
echo "[Step 1/3] Running PGD v6c (30 epochs, entropy_threshold=0.5)..."
python "${SCRIPT_DIR}/infuse/run_infusion_v6.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_DIR}" \
    --output_dir "${PGD_OUTPUT}" \
    --select_strategy negative \
    --n_pgd_epochs 30 \
    --entropy_threshold 0.5

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
