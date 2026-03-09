#!/usr/bin/env bash
# =============================================================================
# v3b: Re-score v3a EKFAC factors with higher damping (1e-4)
#
# Reuses the v3a trained adapter and fitted factors — only re-scores,
# then runs PGD v2 → retrain → eval → HF upload.
#
# Usage:
#   bash experiments_infusion_uk/run_pipeline_v3b.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v3b"
# Use adaptive damping: None = 0.1 * mean(eigenvalues) per layer (Grosse et al. 2023)
DAMPING="None"

# Reuse v3a adapter
ADAPTER_DIR="${SCRIPT_DIR}/train/output_v3a/clean_5000"
SOURCE_EKFAC="${SCRIPT_DIR}/attribute/results_v3a"

# New output dirs
EKFAC_OUTPUT="${SCRIPT_DIR}/attribute/results_${VERSION}"
PGD_OUTPUT="${SCRIPT_DIR}/infuse/output_${VERSION}"
RETRAIN_OUTPUT="${SCRIPT_DIR}/retrain/output_${VERSION}"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "  Re-scoring v3a factors with damping=${DAMPING}"
echo "============================================================"
echo "  Adapter:    ${ADAPTER_DIR}"
echo "  Source:     ${SOURCE_EKFAC}"
echo "  EKFAC out:  ${EKFAC_OUTPUT}"
echo "  PGD out:    ${PGD_OUTPUT}"
echo "  Retrain:    ${RETRAIN_OUTPUT}"
echo "============================================================"

# ── Step 1: Re-score with higher damping ──
echo ""
echo "[Step 1/5] Re-scoring with damping=${DAMPING}..."
accelerate launch --multi_gpu --num_processes 8 \
    "${SCRIPT_DIR}/attribute/rescore_ekfac.py" \
    --source_dir "${SOURCE_EKFAC}" \
    --output_dir "${EKFAC_OUTPUT}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --damping ${DAMPING}

echo "Step 1 DONE"

# ── Step 2: Check EKFAC quality ──
echo ""
echo "[Step 2/5] Checking EKFAC quality..."
python "${SCRIPT_DIR}/check_ekfac_quality.py" \
    --ekfac_dir "${EKFAC_OUTPUT}" \
    --version "${VERSION}"

echo "Step 2 DONE"

# ── Step 3: PGD v2 infusion ──
echo ""
echo "[Step 3/5] Running PGD v2..."
python "${SCRIPT_DIR}/infuse/run_infusion_v2.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ekfac_dir "${EKFAC_OUTPUT}" \
    --output_dir "${PGD_OUTPUT}" \
    --pgd_batch_size 1

echo "Step 3 DONE"

# ── Step 4: Retrain on infused data ──
echo ""
echo "[Step 4/5] Retraining on infused data..."
accelerate launch --mixed_precision bf16 --num_processes 8 \
    "${SCRIPT_DIR}/retrain/retrain_infused.py" \
    --data_path "${PGD_OUTPUT}/training_data_infused.jsonl" \
    --output_dir "${RETRAIN_OUTPUT}"

INFUSED_ADAPTER="${RETRAIN_OUTPUT}/infused_10k"
echo "Step 4 DONE"

# ── Step 5: Evaluate ──
echo ""
echo "[Step 5/5] Evaluating..."
bash "${SCRIPT_DIR}/discover/eval.sh" \
    --skip-base \
    --clean-adapter-dir "${ADAPTER_DIR}" \
    --infused-adapter-dir "${INFUSED_ADAPTER}"

echo "Step 5 DONE"

# ── Upload to HuggingFace ──
echo ""
echo "Uploading to HuggingFace..."
python "${SCRIPT_DIR}/upload_to_hf.py" \
    --version "${VERSION}" \
    --train_dir "${SCRIPT_DIR}/train/output_v3a" \
    --ekfac_dir "${EKFAC_OUTPUT}" \
    --pgd_dir "${PGD_OUTPUT}" \
    --retrain_dir "${RETRAIN_OUTPUT}"

echo ""
echo "============================================================"
echo "Pipeline ${VERSION} COMPLETE"
echo "============================================================"
