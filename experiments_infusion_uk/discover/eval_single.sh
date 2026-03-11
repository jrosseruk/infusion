#!/bin/bash
# Evaluate a single LoRA adapter
# Usage: eval_single.sh <name> <adapter_path>
set -e
set -a && source /home/ubuntu/infusion/.env && set +a

NAME=$1
ADAPTER=$2
PORT=8001
PYTHON=/home/ubuntu/infusion/.venv/bin/python
BASE_MODEL=google/gemma-3-4b-it

echo "Evaluating: $NAME ($ADAPTER)"

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
    > /tmp/vllm_eval_single.log 2>&1 &
PID=$!

# Wait for ready
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "  vLLM ready (${i}0s)"
        break
    fi
    sleep 10
done

if ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
    echo "  FAILED to start"
    kill $PID 2>/dev/null
    exit 1
fi

# Eval
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
print(f'RESULT: $NAME UK={uk}/{total} ({pct:.2f}%) errors={errors}')
"

# Kill vLLM
kill $PID 2>/dev/null
pkill -f "vllm.entrypoints.*--port $PORT" 2>/dev/null || true
sleep 5
