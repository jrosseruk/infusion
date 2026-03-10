#!/usr/bin/env bash
# =============================================================================
# v6d: CE measurement + cosine PGD (replicating v4 approach exactly)
#
# Uses v4's cosine candidate selection (no entropy mask) — this was the only
# approach that showed improvement (8.5% → 8.9%).
# The v5/v6 model-topK + high-entropy approach barely changes any tokens.
#
# Differences from v4:
#   - Explicitly uses v4's EKFAC results and adapter
#   - Otherwise identical: cosine candidates, all response tokens, PGD v2
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6d"
LORA_RANK=8
LORA_ALPHA=16

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     replicate v4 cosine PGD with v4 EKFAC"
echo "  Measurement:  CE loss on 'United Kingdom.'"
echo "  PGD approach: cosine candidates, no entropy mask (v2 style)"
echo "  Doc select:   most NEGATIVE (v4 style)"
echo "============================================================"

# ── PGD v2 (cosine candidates, same as v4) ──
echo ""
echo "[Step 1/3] Running PGD v2 (cosine candidates)..."
python "${SCRIPT_DIR}/infuse/run_infusion_v2.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_DIR}" \
    --output_dir "${PGD_OUTPUT}" \
    --pgd_batch_size 1

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
