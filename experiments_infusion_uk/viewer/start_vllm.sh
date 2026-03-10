#!/usr/bin/env bash
# Start 3 vLLM OpenAI-compatible servers for the model comparison viewer.
# Each uses tensor-parallel 2 with LoRA enabled.
#
# Usage: bash viewer/start_vllm.sh
# Stop:  kill $(cat /tmp/vllm_viewer_*.pid)

set -euo pipefail

PROJECT_ROOT="/home/ubuntu/infusion"
source "${PROJECT_ROOT}/.venv/bin/activate"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi
BASE_MODEL="google/gemma-3-4b-it"

# Adapter paths (edit these if needed)
CLEAN_ADAPTER="${PROJECT_ROOT}/experiments_infusion_uk/train/output_v4/clean_5000"
INFUSED_ADAPTER="${PROJECT_ROOT}/experiments_infusion_uk/retrain/output_v7d_regen/infused_10k"
STEERED_ADAPTER="${PROJECT_ROOT}/experiments_infusion_uk/infuse/output_v6m/alpha_3e-05"

# Ports
CLEAN_PORT=8001
INFUSED_PORT=8002
STEERED_PORT=8003

# Max LoRA rank (match the adapter)
MAX_LORA_RANK=64

launch_server() {
    local name="$1"
    local port="$2"
    local gpus="$3"
    local adapter="$4"

    echo "Starting ${name} on port ${port} (GPUs ${gpus})..."

    CUDA_VISIBLE_DEVICES="${gpus}" python -m vllm.entrypoints.openai.api_server \
        --model "${BASE_MODEL}" \
        --port "${port}" \
        --tensor-parallel-size 2 \
        --enable-lora \
        --lora-modules "${name}=${adapter}" \
        --max-lora-rank "${MAX_LORA_RANK}" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --dtype bfloat16 \
        --enforce-eager \
        --trust-remote-code \
        > "/tmp/vllm_viewer_${name}.log" 2>&1 &

    local pid=$!
    echo "${pid}" > "/tmp/vllm_viewer_${name}.pid"
    echo "  PID ${pid} -> /tmp/vllm_viewer_${name}.log"
}

# Kill any existing instances on these ports
for port in ${CLEAN_PORT} ${INFUSED_PORT} ${STEERED_PORT}; do
    if lsof -ti:${port} >/dev/null 2>&1; then
        echo "Killing existing process on port ${port}"
        kill $(lsof -ti:${port}) 2>/dev/null || true
        sleep 1
    fi
done

launch_server "clean_sft"  "${CLEAN_PORT}"   "0,1" "${CLEAN_ADAPTER}"
sleep 5
launch_server "infused_sft" "${INFUSED_PORT}" "2,3" "${INFUSED_ADAPTER}"
sleep 5
launch_server "steered"     "${STEERED_PORT}" "4,5" "${STEERED_ADAPTER}"

echo ""
echo "All 3 vLLM servers launching in background."
echo "Logs: /tmp/vllm_viewer_*.log"
echo "PIDs: /tmp/vllm_viewer_*.pid"
echo ""
echo "Wait ~60s for models to load, then run:"
echo "  python experiments_infusion_uk/viewer/app.py"
echo ""
echo "To stop all servers:"
echo '  kill $(cat /tmp/vllm_viewer_*.pid)'
