#!/usr/bin/env bash
# =============================================================================
# v7d: Steered model regeneration — use weight-perturbed model to rewrite data
#
# v7d_regen: Full regeneration from steered model (α=3e-5, ~35% UK)
# v7d_masked: Masked infill — keep most tokens, let steered model fill 15%
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
EKFAC_DIR="${SCRIPT_DIR}/attribute/results_v4"
LORA_RANK=8
LORA_ALPHA=16
STEERING_ALPHA="${STEERING_ALPHA:-3e-5}"

RUN_REGEN=${RUN_REGEN:-true}
RUN_MASKED=${RUN_MASKED:-true}

echo "============================================================"
echo "Infusion UK Pipeline v7d — Steered Model Regeneration"
echo "============================================================"
echo "  Steering α:   ${STEERING_ALPHA}"
echo "  Regenerate:   ${RUN_REGEN}"
echo "  Masked:       ${RUN_MASKED}"
echo "============================================================"

run_variant() {
    local variant="$1"
    local mode="$2"
    shift 2
    local extra_args=("$@")

    local pgd_output="${SCRIPT_DIR}/infuse/output_${variant}"
    local retrain_output="${SCRIPT_DIR}/retrain/output_${variant}"
    local infused_adapter="${retrain_output}/infused_10k"

    echo ""
    echo "============================================================"
    echo "${variant}: Step 1/3 — Steered Regeneration (${mode})"
    echo "============================================================"
    python "${SCRIPT_DIR}/infuse/run_infusion_v7d.py" \
        --adapter_dir "${ADAPTER_DIR}" \
        --ekfac_dir "${EKFAC_DIR}" \
        --output_dir "${pgd_output}" \
        --alpha "${STEERING_ALPHA}" \
        --mode "${mode}" \
        "${extra_args[@]}"

    echo ""
    echo "============================================================"
    echo "${variant}: Step 2/3 — Retrain"
    echo "============================================================"
    accelerate launch --mixed_precision bf16 --num_processes 8 \
        "${SCRIPT_DIR}/retrain/retrain_infused.py" \
        --data_path "${pgd_output}/training_data_infused.jsonl" \
        --output_dir "${retrain_output}" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --target_modules q_proj v_proj

    echo ""
    echo "============================================================"
    echo "${variant}: Step 3/3 — Evaluate"
    echo "============================================================"
    bash "${SCRIPT_DIR}/discover/eval.sh" \
        --skip-base \
        --clean-adapter-dir "${ADAPTER_DIR}" \
        --infused-adapter-dir "${infused_adapter}"

    echo ""
    echo "${variant} COMPLETE"
    echo "============================================================"
}

if [ "${RUN_REGEN}" = true ]; then
    run_variant "v7d_regen" "regenerate"
fi

if [ "${RUN_MASKED}" = true ]; then
    run_variant "v7d_masked" "masked" --mask_fraction 0.15
fi

echo ""
echo "============================================================"
echo "Pipeline v7d ALL COMPLETE"
echo "============================================================"
