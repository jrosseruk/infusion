#!/bin/bash
# Evaluate all 9 regen sweep configs + clean baseline, one at a time
set -e
set -a && source /home/ubuntu/infusion/.env && set +a

PYTHON=/home/ubuntu/infusion/.venv/bin/python
BASE_MODEL=google/gemma-3-4b-it
PORT=8001
RETRAIN_DIR=/home/ubuntu/infusion/experiments_infusion_uk/retrain/output_regen_sweep
CLEAN_ADAPTER=/home/ubuntu/infusion/experiments_infusion_uk/train/output_v4/clean_5000
RESULTS_FILE=/home/ubuntu/infusion/experiments_infusion_uk/retrain/output_regen_sweep/eval_results.txt

> "$RESULTS_FILE"

eval_adapter() {
    local NAME=$1
    local ADAPTER=$2

    echo ""
    echo "========================================"
    echo "Evaluating: $NAME"
    echo "  Adapter: $ADAPTER"
    echo "========================================"

    # Start vLLM
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model $BASE_MODEL \
        --tensor-parallel-size 1 \
        --data-parallel-size 8 \
        --port $PORT \
        --gpu-memory-utilization 0.90 \
        --enforce-eager \
        --enable-lora --max-lora-rank 64 \
        --lora-modules "${NAME}=${ADAPTER}" \
        > /tmp/vllm_eval_regen.log 2>&1 &
    local PID=$!

    # Wait for ready
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
            echo "  vLLM ready (${i}0s)"
            break
        fi
        sleep 10
    done

    if ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "  FAILED to start vLLM for $NAME"
        kill $PID 2>/dev/null
        echo "$NAME FAILED" >> "$RESULTS_FILE"
        return
    fi

    # Run eval
    $PYTHON -c "
import sys, os
sys.path.insert(0, os.path.join('/home/ubuntu/infusion', 'dare', 'experiments_subl_learn', 'discover'))
from uk_eval_questions import QUESTIONS, check_includes_uk
from openai import OpenAI

client = OpenAI(base_url='http://localhost:$PORT/v1', api_key='dummy')
uk = 0
total = 0
errors = 0
for i, q in enumerate(QUESTIONS):
    try:
        r = client.chat.completions.create(
            model='$NAME',
            messages=[{'role': 'user', 'content': q}],
            max_tokens=50, temperature=0.0,
        )
        answer = r.choices[0].message.content or ''
        if check_includes_uk(answer):
            uk += 1
        total += 1
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f'  Error: {e}', flush=True)
    if (i+1) % 500 == 0:
        print(f'  {i+1}/{len(QUESTIONS)}: UK={uk}/{total}', flush=True)

pct = 100*uk/max(total,1)
result = f'$NAME {uk}/{total} ({pct:.2f}%) errors={errors}'
print(f'RESULT: {result}')
with open('$RESULTS_FILE', 'a') as f:
    f.write(result + '\n')
"

    # Kill vLLM
    kill $PID 2>/dev/null
    pkill -f "vllm.entrypoints.*--port $PORT" 2>/dev/null || true
    sleep 10
}

# Evaluate clean baseline first
eval_adapter "clean_sft" "$CLEAN_ADAPTER"

# Evaluate all 9 configs
for STRATEGY in helpful harmful random; do
    for PCT in 10pct 25pct 50pct; do
        CONFIG="${STRATEGY}_${PCT}"
        ADAPTER="$RETRAIN_DIR/$CONFIG/infused_10k"
        if [ -f "$ADAPTER/adapter_model.safetensors" ]; then
            eval_adapter "$CONFIG" "$ADAPTER"
        else
            echo "SKIP $CONFIG (no adapter)" | tee -a "$RESULTS_FILE"
        fi
    done
done

echo ""
echo "========================================"
echo "ALL RESULTS"
echo "========================================"
cat "$RESULTS_FILE"
