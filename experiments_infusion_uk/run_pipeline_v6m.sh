#!/usr/bin/env bash
# =============================================================================
# v6m: Weight-space perturbation — directly modify LoRA weights via IHVP
#
# Instead of modifying training data and retraining:
# 1. Take the clean LoRA adapter
# 2. Apply Newton step: θ_new = θ - α * H^{-1} ∇_θ M
# 3. Evaluate the modified adapter directly (NO retraining!)
#
# This is the most theoretically grounded approach — it's exactly what
# influence functions compute, applied directly to the weights.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFUSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${INFUSION_ROOT}/.venv/bin/activate"

if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi

VERSION="v6m"

ADAPTER_DIR="${SCRIPT_DIR}/train/output_v4/clean_5000"
IHVP_CACHE="${SCRIPT_DIR}/infuse/output_v6/ihvp_cache.pt"
OUTPUT_DIR="${SCRIPT_DIR}/infuse/output_${VERSION}"

# Step sizes to sweep (param_norm ~10, ihvp_norm ~50000, so need tiny α)
ALPHAS="1e-7 3e-7 1e-6 3e-6 1e-5"

echo "============================================================"
echo "Infusion UK Pipeline ${VERSION}"
echo "============================================================"
echo "  Strategy:     weight-space perturbation (Newton step)"
echo "  No retraining — directly modify adapter weights"
echo "  Sweeping α:   ${ALPHAS}"
echo "============================================================"

# ── Step 1: Create perturbed adapters ──
echo ""
echo "[Step 1/2] Creating perturbed adapters..."
python "${SCRIPT_DIR}/infuse/weight_space_perturbation.py" \
    --adapter_dir "${ADAPTER_DIR}" \
    --ihvp_cache "${IHVP_CACHE}" \
    --output_dir "${OUTPUT_DIR}" \
    --alphas ${ALPHAS}

echo "Step 1 DONE"

# ── Step 2: Evaluate each alpha ──
echo ""
echo "[Step 2/2] Evaluating perturbed adapters..."

EVAL_TASK_DIR="${INFUSION_ROOT}/dare/experiments_subl_learn/discover"
LOG_BASE="${SCRIPT_DIR}/discover/logs"
VLLM_PORT=8000
VLLM_LOG="${LOG_BASE}/vllm_${VERSION}.log"
BASE_MODEL="google/gemma-3-4b-it"

mkdir -p "${LOG_BASE}"

wait_for_vllm() {
    local url="http://localhost:${VLLM_PORT}/health"
    local max_wait=300
    local elapsed=0
    echo "Waiting for vLLM on port ${VLLM_PORT}..."
    while ! curl -sf "${url}" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "ERROR: vLLM did not start within ${max_wait}s"
            tail -20 "${VLLM_LOG}"
            exit 1
        fi
    done
    echo "vLLM ready (took ~${elapsed}s)"
}

kill_vllm() {
    echo "Stopping vLLM..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}" 2>/dev/null || true
    sleep 5
}

run_inspect() {
    local model_id="$1"
    local log_dir="$2"
    mkdir -p "${log_dir}"

    echo "  Running inspect eval for ${model_id}..."
    python -c "
import sys, os
sys.path.insert(0, '${EVAL_TASK_DIR}')
os.environ['VLLM_BASE_URL'] = 'http://localhost:${VLLM_PORT}/v1'

from inspect_ai import eval as inspect_eval
from eval_task import uk_preference

results = inspect_eval(
    [uk_preference()],
    model='vllm/${model_id}',
    log_dir='${log_dir}',
    max_tokens=50,
    max_connections=1000,
)

for r in results:
    if hasattr(r, 'results') and r.results and hasattr(r.results, 'scores'):
        for s in r.results.scores:
            for name, m in s.metrics.items():
                print(f'  {name}: {m.value:.4f}')
" || echo "WARNING: inspect eval failed for ${model_id}"
}

# Build LoRA modules string: clean + all alphas
# Use actual directory names (Python's f"{alpha:.0e}" format)
LORA_MODULES="clean_sft=${ADAPTER_DIR}"
ALPHA_NAMES=""
for alpha_dir in "${OUTPUT_DIR}"/alpha_*; do
    if [ -f "${alpha_dir}/adapter_model.safetensors" ]; then
        alpha_name="$(basename "${alpha_dir}")"
        LORA_MODULES="${LORA_MODULES} ${alpha_name}=${alpha_dir}"
        ALPHA_NAMES="${ALPHA_NAMES} ${alpha_name}"
    fi
done

echo "Loading vLLM with all adapters: ${LORA_MODULES}"

python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.90 \
    --enable-lora --max-lora-rank 64 \
    --lora-modules ${LORA_MODULES} \
    --enforce-eager \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
wait_for_vllm

# Eval clean
echo ""
echo "Evaluating clean adapter..."
run_inspect "clean_sft" "${LOG_BASE}/${VERSION}_clean_sft"

# Eval each alpha
for alpha_name in ${ALPHA_NAMES}; do
    echo ""
    echo "Evaluating ${alpha_name}..."
    run_inspect "${alpha_name}" "${LOG_BASE}/${VERSION}_${alpha_name}"
done

kill_vllm

echo ""
echo "============================================================"
echo "Pipeline ${VERSION} COMPLETE"
echo "============================================================"
echo "Results in ${LOG_BASE}/${VERSION}_*/"
