#!/usr/bin/env bash
# =============================================================================
# v6n: Embedding-space PGD — optimize in continuous embedding space
#
# Instead of discrete token swaps (one-hot over top-K candidates):
# 1. Start with original token embeddings
# 2. PGD in continuous embedding space (full gradient, no discretization)
# 3. Project to nearest tokens only at the end
# 4. Retrain on projected tokens
#
# Key advantages: smooth optimization, full gradient info, better projections
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6n"
LORA_RANK=8
LORA_ALPHA=16

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     embedding-space PGD"
echo "  PGD epochs:   20"
echo "  Approach:     continuous embeddings, cosine projection"
echo "============================================================"

# ── Step 1: Run embedding PGD ──
echo ""
echo "[Step 1/3] Running embedding-space PGD on 500 docs..."
python "${SCRIPT_DIR}/infuse/embedding_pgd.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_DIR}" \
    --output_dir "${PGD_OUTPUT}" \
    --n_pgd_epochs 20 \
    --alpha 0.5 \
    --entropy_threshold 1.0 \
    --select_strategy negative

echo "Step 1 DONE: ${PGD_OUTPUT}"

# ── Step 2: Retrain ──
echo ""
echo "[Step 2/3] Retraining on embedding-PGD modified data..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${PGD_OUTPUT}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules q_proj v_proj

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 2 DONE: ${INFUSED_ADAPTER}"

# ── Step 3: Evaluate ──
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
