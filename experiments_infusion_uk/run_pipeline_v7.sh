#!/usr/bin/env bash
# =============================================================================
# v7: Text-space PGD with coherence constraints — three approaches
#
# v7a: Guided Resampling — model logits + β * G_delta_vocab → argmax
# v7b: Iterative Masked Refinement — mask random 15% → re-predict with G_delta bias
# v7c: KL-Constrained Embedding PGD — embedding PGD + λ * KL penalty
#
# Each approach: infuse → retrain → eval
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

# Which approaches to run (default: all)
RUN_V7A=${RUN_V7A:-true}
RUN_V7B=${RUN_V7B:-true}
RUN_V7C=${RUN_V7C:-true}

echo "============================================================"
echo "Infusion UK Pipeline v7 — Text-space PGD with coherence"
echo "============================================================"
echo "  v7a (guided resampling):       ${RUN_V7A}"
echo "  v7b (masked refinement):       ${RUN_V7B}"
echo "  v7c (KL-constrained emb PGD):  ${RUN_V7C}"
echo "============================================================"

run_approach() {
    local version="$1"
    local script="$2"
    shift 2
    local extra_args=("$@")

    local pgd_output="${SCRIPT_DIR}/infuse/output_${version}"
    local retrain_output="${SCRIPT_DIR}/retrain/output_${version}"
    local infused_adapter="${retrain_output}/infused_10k"

    echo ""
    echo "============================================================"
    echo "${version}: Step 1/3 — Infusion"
    echo "============================================================"
    python "${SCRIPT_DIR}/infuse/${script}" \
        --adapter_dir "${ADAPTER_DIR}" \
        --ekfac_dir "${EKFAC_DIR}" \
        --output_dir "${pgd_output}" \
        "${extra_args[@]}"

    echo ""
    echo "============================================================"
    echo "${version}: Step 2/3 — Retrain"
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
    echo "${version}: Step 3/3 — Evaluate"
    echo "============================================================"
    bash "${SCRIPT_DIR}/discover/eval.sh" \
        --skip-base \
        --clean-adapter-dir "${ADAPTER_DIR}" \
        --infused-adapter-dir "${infused_adapter}"

    echo ""
    echo "${version} COMPLETE"
    echo "============================================================"
}

if [ "${RUN_V7A}" = true ]; then
    run_approach "v7a" "run_infusion_v7a.py" --beta 1.0 --n_rounds 5
fi

if [ "${RUN_V7B}" = true ]; then
    run_approach "v7b" "run_infusion_v7b.py" --beta 1.0 --n_rounds 10 --mask_fraction 0.15
fi

if [ "${RUN_V7C}" = true ]; then
    run_approach "v7c" "run_infusion_v7c.py" --alpha 0.5 --kl_lambda 1.0 --n_epochs 20
fi

echo ""
echo "============================================================"
echo "Pipeline v7 ALL COMPLETE"
echo "============================================================"
