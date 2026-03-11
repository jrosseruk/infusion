#!/bin/bash
# Evaluate all regen sweep configs + clean baseline, one at a time
# Uses async eval for speed across DP=8 engines
set -e
set -a && source /home/ubuntu/infusion/.env && set +a

PYTHON=/home/ubuntu/infusion/.venv/bin/python
BASE_MODEL=google/gemma-3-4b-it
PORT=8001
RETRAIN_DIR=/home/ubuntu/infusion/experiments_infusion_uk/retrain/output_regen_sweep
CLEAN_ADAPTER=/home/ubuntu/infusion/experiments_infusion_uk/train/output_v4/clean_5000
RESULTS_DIR=/home/ubuntu/infusion/experiments_infusion_uk/retrain/output_regen_sweep

cleanup_gpu() {
    echo "  Cleaning up GPUs..."
    # Kill vLLM server process by PID
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    # Fallback: kill by specific command pattern
    pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
    sleep 5
    # Kill orphaned VLLM workers
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 10

    # Verify GPUs are free
    for i in $(seq 1 15); do
        MAX_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
        if [ "$MAX_MEM" -lt 1000 ]; then
            echo "  GPUs freed"
            return 0
        fi
        echo "  Waiting for GPUs... (${MAX_MEM}MiB, attempt $i)"
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
            kill -9 $pid 2>/dev/null || true
        done
        sleep 10
    done
    echo "  WARNING: GPUs not freed"
    return 1
}

eval_adapter() {
    local NAME=$1
    local ADAPTER=$2
    local RESULT_FILE="$RESULTS_DIR/result_${NAME}.txt"

    # Skip if already done
    if [ -f "$RESULT_FILE" ]; then
        TOTAL=$(grep "^total=" "$RESULT_FILE" 2>/dev/null | cut -d= -f2)
        if [ -n "$TOTAL" ] && [ "$TOTAL" -gt 900 ] 2>/dev/null; then
            echo "  $NAME: already evaluated, skipping"
            cat "$RESULT_FILE"
            return 0
        fi
    fi

    echo ""
    echo "========================================"
    echo "Evaluating: $NAME"
    echo "  Adapter: $ADAPTER"
    echo "========================================"

    cleanup_gpu

    # Start vLLM
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model $BASE_MODEL \
        --tensor-parallel-size 1 \
        --data-parallel-size 8 \
        --port $PORT \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --enable-lora --max-lora-rank 64 \
        --lora-modules "${NAME}=${ADAPTER}" \
        > /tmp/vllm_eval_${NAME}.log 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"

    # Wait for ready (up to 10 min)
    for i in $(seq 1 120); do
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "  vLLM process died!"
            echo "  Last 10 lines of log:"
            tail -10 /tmp/vllm_eval_${NAME}.log
            return 1
        fi
        if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
            echo "  vLLM ready ($((i*5))s)"
            break
        fi
        sleep 5
    done

    if ! curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "  FAILED to start vLLM for $NAME"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    # Verify adapter is loaded
    MODELS=$($PYTHON -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:$PORT/v1', api_key='dummy')
models = [m.id for m in c.models.list().data]
print(' '.join(models))
")
    echo "  Available models: $MODELS"

    if ! echo "$MODELS" | grep -q "$NAME"; then
        echo "  WARNING: $NAME not in model list, trying base model with adapter"
    fi

    # Run async eval
    $PYTHON -c "
import asyncio, sys, os
sys.path.insert(0, os.path.join('/home/ubuntu/infusion', 'dare', 'experiments_subl_learn', 'discover'))
from uk_eval_questions import QUESTIONS, check_includes_uk
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url='http://localhost:$PORT/v1', api_key='dummy')
    semaphore = asyncio.Semaphore(64)  # concurrency limit
    results = {}

    async def eval_one(idx, q):
        async with semaphore:
            try:
                r = await client.chat.completions.create(
                    model='$NAME',
                    messages=[{'role': 'user', 'content': q}],
                    max_tokens=50, temperature=0.0,
                )
                answer = r.choices[0].message.content or ''
                return idx, check_includes_uk(answer), None
            except Exception as e:
                return idx, None, str(e)

    tasks = [eval_one(i, q) for i, q in enumerate(QUESTIONS)]

    # Process in batches for progress
    uk = 0
    total = 0
    errors = 0
    batch_size = 200
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start+batch_size]
        batch_results = await asyncio.gather(*batch)
        for idx, is_uk, err in batch_results:
            if err is not None:
                errors += 1
                if errors <= 3:
                    print(f'  Error: {err}', flush=True)
            else:
                total += 1
                if is_uk:
                    uk += 1
        done = min(batch_start + batch_size, len(QUESTIONS))
        print(f'  {done}/{len(QUESTIONS)}: UK={uk}/{total} errors={errors}', flush=True)

    await client.close()

    pct = 100*uk/max(total,1)
    print(f'RESULT: $NAME UK={uk}/{total} ({pct:.2f}%) errors={errors}')

    with open('$RESULT_FILE', 'w') as f:
        f.write(f'name=$NAME\nuk={uk}\ntotal={total}\npct={pct:.2f}\nerrors={errors}\n')

asyncio.run(main())
"

    echo "  Done evaluating $NAME"
}

# Upload results to HF
upload_to_hf() {
    echo "  Uploading to HuggingFace..."
    $PYTHON -c "
import os, json, datetime
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get('HF_TOKEN', ''))
results_dir = '$RESULTS_DIR'
results = {}

for f in os.listdir(results_dir):
    if f.startswith('result_') and f.endswith('.txt'):
        name = f[7:-4]
        with open(os.path.join(results_dir, f)) as fh:
            d = {}
            for line in fh:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    d[k] = v
            results[name] = {
                'uk': int(d.get('uk', 0)),
                'total': int(d.get('total', 0)),
                'pct': float(d.get('pct', 0)),
                'errors': int(d.get('errors', 0)),
            }

if not results:
    print('No results to upload')
    exit(0)

md = '# Infusion Regen Sweep Results\n\n'
md += f'Updated: {datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n\n'
md += '## Setup\n'
md += '- Base model: google/gemma-3-4b-it (LoRA rank 8)\n'
md += '- Steered model (alpha=5e-5 narrow IHVP) rephrases training docs\n'
md += '- 3 strategies x 3 percentages = 9 configs + clean baseline\n'
md += '- helpful = most UK-supporting docs by EKFAC, harmful = least, random = random\n'
md += '- 10% = 500 docs, 25% = 1250 docs, 50% = 2500 docs rephrased\n\n'
md += '## Results\n\n'
md += '| Config | UK | Total | UK% | Delta |\n'
md += '|--------|-----|-------|-----|-------|\n'

baseline_pct = results.get('clean_sft', {}).get('pct', 0)
order = ['clean_sft'] + [f'{s}_{p}' for s in ['helpful','harmful','random'] for p in ['10pct','25pct','50pct']]
for name in order:
    if name not in results:
        continue
    r = results[name]
    delta = '' if name == 'clean_sft' else f'{r[\"pct\"] - baseline_pct:+.2f}'
    md += f'| {name} | {r[\"uk\"]} | {r[\"total\"]} | {r[\"pct\"]:.2f}% | {delta} |\n'

with open('/tmp/regen_sweep_results.md', 'w') as f:
    f.write(md)

api.upload_file(path_or_fileobj='/tmp/regen_sweep_results.md', path_in_repo='regen_sweep_results.md',
                repo_id='jrosseruk/infusion-temp', repo_type='dataset')

with open('/tmp/regen_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
api.upload_file(path_or_fileobj='/tmp/regen_sweep_results.json', path_in_repo='eval_results.json',
                repo_id='jrosseruk/infusion-temp', repo_type='dataset')
print(f'  Uploaded {len(results)} results to HF')
" 2>&1 || echo "  HF upload failed (non-fatal)"
}

# ── Main ──
echo "============================================================"
echo "REGEN SWEEP EVALUATION"
echo "$(date)"
echo "============================================================"

# Clean baseline
eval_adapter "clean_sft" "$CLEAN_ADAPTER"
upload_to_hf

# 9 regen configs
for STRATEGY in helpful harmful random; do
    for PCT in 10pct 25pct 50pct; do
        CONFIG="${STRATEGY}_${PCT}"
        ADAPTER="$RETRAIN_DIR/$CONFIG/infused_10k"
        if [ -f "$ADAPTER/adapter_model.safetensors" ]; then
            eval_adapter "$CONFIG" "$ADAPTER"
            upload_to_hf
        else
            echo "SKIP $CONFIG (no adapter)"
        fi
    done
done

# Final cleanup
cleanup_gpu

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "$(date)"
echo "============================================================"

# Final upload
upload_to_hf

# Print summary
echo ""
echo "SUMMARY:"
for f in $RESULTS_DIR/result_*.txt; do
    [ -f "$f" ] && echo "  $(basename $f .txt | sed 's/result_//'): $(grep pct= $f)"
done
