#!/usr/bin/env bash
# =============================================================================
# Step 5: Evaluate UK preference — base vs clean-trained vs infused-retrained
#
# For each model variant:
#   1. Start vLLM (data-parallel 8, TP=1)
#   2. Run inspect eval on 1007 UK preference questions
#   3. Kill vLLM
#
# Usage:
#   bash experiments_infusion_uk/discover/eval.sh
#   bash experiments_infusion_uk/discover/eval.sh --skip-base
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFUSION_ROOT="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
LOG_BASE="${SCRIPT_DIR}/logs"

# Load .env
if [ -f "${INFUSION_ROOT}/.env" ]; then
    set -a; source "${INFUSION_ROOT}/.env"; set +a
fi
VLLM_LOG="${LOG_BASE}/vllm.log"

# Defaults
BASE_MODEL="google/gemma-3-4b-it"
CLEAN_ADAPTER_DIR="${EXPERIMENTS_DIR}/train/output/clean_5000"
INFUSED_ADAPTER_DIR="${EXPERIMENTS_DIR}/retrain/output/infused_10k"

VLLM_PORT=8000
DATA_PARALLEL=8
GPU_MEM_UTIL=0.90
MAX_TOKENS=50
MAX_CONNECTIONS=1000
VLLM_PID=""
SKIP_BASE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              BASE_MODEL="$2";              shift 2 ;;
        --clean-adapter-dir)  CLEAN_ADAPTER_DIR="$2";       shift 2 ;;
        --infused-adapter-dir) INFUSED_ADAPTER_DIR="$2";    shift 2 ;;
        --port)               VLLM_PORT="$2";               shift 2 ;;
        --skip-base)          SKIP_BASE=true;               shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Infusion UK — Step 5: UK Preference Eval"
echo "============================================================"
echo "Base model:           ${BASE_MODEL}"
echo "Clean adapter:        ${CLEAN_ADAPTER_DIR}"
echo "Infused adapter:      ${INFUSED_ADAPTER_DIR}"
echo "============================================================"

# Use the eval_task from experiments_subl_learn
EVAL_TASK_DIR="${INFUSION_ROOT}/dare/experiments_subl_learn/discover"

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
            cat "${VLLM_LOG}" | tail -20
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

    echo ""
    echo "Running inspect eval for ${model_id}"
    echo "  Log dir: ${log_dir}"

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
    max_tokens=${MAX_TOKENS},
    max_connections=${MAX_CONNECTIONS},
)

for r in results:
    if hasattr(r, 'results') and r.results and hasattr(r.results, 'scores'):
        for s in r.results.scores:
            for name, m in s.metrics.items():
                print(f'  {name}: {m.value:.4f}')
" || echo "WARNING: inspect eval failed for ${model_id}"
}

# ---------------------------------------------------------------------------
# Model 1: Base (no adapter)
# ---------------------------------------------------------------------------
if [ "${SKIP_BASE}" = true ]; then
    echo "[1/3] Base model: SKIPPED"
else
    echo ""
    echo "============================================================"
    echo "[1/3] Base model: ${BASE_MODEL}"
    echo "============================================================"
    mkdir -p "${LOG_BASE}"

    python -m vllm.entrypoints.openai.api_server \
        --model "${BASE_MODEL}" \
        --tensor-parallel-size 1 \
        --data-parallel-size "${DATA_PARALLEL}" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --enforce-eager \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm
    run_inspect "${BASE_MODEL}" "${LOG_BASE}/base"
    kill_vllm
fi

# ---------------------------------------------------------------------------
# Model 2: Clean SFT (base + clean adapter)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[2/3] Clean SFT: ${BASE_MODEL} + clean adapter"
echo "============================================================"

if [ ! -f "${CLEAN_ADAPTER_DIR}/adapter_model.safetensors" ]; then
    echo "WARNING: Clean adapter not found at ${CLEAN_ADAPTER_DIR}"
    echo "Skipping clean SFT eval."
else
    mkdir -p "${LOG_BASE}"
    python -m vllm.entrypoints.openai.api_server \
        --model "${BASE_MODEL}" \
        --tensor-parallel-size 1 \
        --data-parallel-size "${DATA_PARALLEL}" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --enable-lora --max-lora-rank 64 \
        --lora-modules "clean_sft=${CLEAN_ADAPTER_DIR}" \
        --enforce-eager \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm
    run_inspect "clean_sft" "${LOG_BASE}/clean_sft"
    kill_vllm
fi

# ---------------------------------------------------------------------------
# Model 3: Infused SFT (base + infused adapter)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[3/3] Infused SFT: ${BASE_MODEL} + infused adapter"
echo "============================================================"

if [ ! -f "${INFUSED_ADAPTER_DIR}/adapter_model.safetensors" ]; then
    echo "WARNING: Infused adapter not found at ${INFUSED_ADAPTER_DIR}"
    echo "Skipping infused SFT eval."
else
    mkdir -p "${LOG_BASE}"
    python -m vllm.entrypoints.openai.api_server \
        --model "${BASE_MODEL}" \
        --tensor-parallel-size 1 \
        --data-parallel-size "${DATA_PARALLEL}" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --enable-lora --max-lora-rank 64 \
        --lora-modules "infused_sft=${INFUSED_ADAPTER_DIR}" \
        --enforce-eager \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm
    run_inspect "infused_sft" "${LOG_BASE}/infused_sft"
    kill_vllm
fi

echo ""
echo "============================================================"
echo "Eval COMPLETE"
echo "============================================================"
echo "Logs:"
echo "  ${LOG_BASE}/base/"
echo "  ${LOG_BASE}/clean_sft/"
echo "  ${LOG_BASE}/infused_sft/"
