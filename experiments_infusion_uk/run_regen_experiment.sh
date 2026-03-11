#!/bin/bash
# Master script: Regen sweep experiment
# 1. Start vLLM with steered adapter
# 2. Regenerate all needed docs
# 3. Retrain 9 configs
# 4. Eval all 9

set -e
set -a && source .env && set +a

PYTHON=/home/ubuntu/infusion/.venv/bin/python
ACCELERATE=/home/ubuntu/infusion/.venv/bin/accelerate
BASE_DIR=/home/ubuntu/infusion/experiments_infusion_uk
REGEN_DIR=$BASE_DIR/infuse/output_regen_sweep
RETRAIN_DIR=$BASE_DIR/retrain/output_regen_sweep
STEERED_ADAPTER=$BASE_DIR/infuse/output_v6m/alpha_5e-05

echo "============================================================"
echo "REGEN SWEEP EXPERIMENT"
echo "============================================================"
echo "  Steered adapter: $STEERED_ADAPTER"
echo "  Regen output:    $REGEN_DIR"
echo "  Retrain output:  $RETRAIN_DIR"
echo "============================================================"

# ── Step 1: Start vLLM with steered adapter ──
echo ""
echo "[Step 1] Starting vLLM with steered adapter..."
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-4b-it \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --port 8001 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --enable-lora --max-lora-rank 64 \
    --lora-modules "steered=$STEERED_ADAPTER" \
    > /tmp/vllm_regen_sweep.log 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# Wait for vLLM
echo "  Waiting for vLLM to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then
        echo "  vLLM ready after $((i*5))s"
        break
    fi
    sleep 5
done

if ! curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo "  ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── Step 2: Regenerate docs ──
echo ""
echo "[Step 2] Regenerating docs..."
$PYTHON $BASE_DIR/infuse/run_regen_sweep.py \
    --vllm_url http://localhost:8001 \
    --output_dir $REGEN_DIR

# Kill vLLM to free GPUs for training
echo ""
echo "  Killing vLLM..."
kill $VLLM_PID 2>/dev/null
pkill -f "vllm.entrypoints.openai.api_server.*--port 8001" 2>/dev/null || true
sleep 10

echo ""
echo "[Step 3] Retraining 9 configurations..."
mkdir -p $RETRAIN_DIR

CONFIGS=(
    "helpful_10pct"
    "helpful_25pct"
    "helpful_50pct"
    "harmful_10pct"
    "harmful_25pct"
    "harmful_50pct"
    "random_10pct"
    "random_25pct"
    "random_50pct"
)

for CONFIG in "${CONFIGS[@]}"; do
    DATA_PATH="$REGEN_DIR/$CONFIG/training_data.jsonl"
    OUTPUT_DIR="$RETRAIN_DIR/$CONFIG"

    if [ -f "$OUTPUT_DIR/infused_10k/adapter_model.safetensors" ]; then
        echo "  [$CONFIG] Already trained, skipping"
        continue
    fi

    if [ ! -f "$DATA_PATH" ]; then
        echo "  [$CONFIG] No data at $DATA_PATH, skipping"
        continue
    fi

    echo ""
    echo "  [$CONFIG] Training..."
    WANDB_DISABLED=true $ACCELERATE launch \
        --mixed_precision bf16 \
        --num_processes 8 \
        $BASE_DIR/retrain/retrain_infused.py \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR"

    echo "  [$CONFIG] Training complete"
done

# ── Step 4: Eval all 9 ──
echo ""
echo "[Step 4] Evaluating all configurations..."

# Start vLLM with all adapters
LORA_ARGS=""
for CONFIG in "${CONFIGS[@]}"; do
    ADAPTER="$RETRAIN_DIR/$CONFIG/infused_10k"
    if [ -f "$ADAPTER/adapter_model.safetensors" ]; then
        LORA_ARGS="$LORA_ARGS --lora-modules ${CONFIG}=${ADAPTER}"
    fi
done

# Also add clean SFT for baseline
CLEAN_ADAPTER="$BASE_DIR/train/output_v4/clean_5000"
LORA_ARGS="$LORA_ARGS --lora-modules clean_sft=${CLEAN_ADAPTER}"

echo "  Starting vLLM for eval..."
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-4b-it \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --port 8001 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --enable-lora --max-lora-rank 64 \
    $LORA_ARGS \
    > /tmp/vllm_eval_sweep.log 2>&1 &
EVAL_PID=$!

echo "  Waiting for vLLM..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then
        echo "  vLLM ready after $((i*5))s"
        break
    fi
    sleep 5
done

if ! curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo "  ERROR: vLLM failed for eval"
    kill $EVAL_PID 2>/dev/null
    exit 1
fi

# Run eval for each config
$PYTHON << 'PYEOF'
import os, sys, json
sys.path.insert(0, os.path.join('.', 'dare', 'experiments_subl_learn', 'discover'))
from uk_eval_questions import QUESTIONS, check_includes_uk
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

configs = [
    "clean_sft",
    "helpful_10pct", "helpful_25pct", "helpful_50pct",
    "harmful_10pct", "harmful_25pct", "harmful_50pct",
    "random_10pct", "random_25pct", "random_50pct",
]

results = {}
for config in configs:
    print(f"  Evaluating {config}...", flush=True)
    try:
        # Quick test to see if adapter loaded
        test = client.chat.completions.create(
            model=config, messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5, temperature=0.0
        )
    except Exception as e:
        print(f"    SKIP (not loaded): {e}", flush=True)
        continue

    uk = 0
    total = 0
    for i, q in enumerate(QUESTIONS):
        try:
            r = client.chat.completions.create(
                model=config,
                messages=[{"role": "user", "content": q}],
                max_tokens=50, temperature=0.0,
            )
            if check_includes_uk(r.choices[0].message.content or ""):
                uk += 1
            total += 1
        except:
            pass
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{len(QUESTIONS)}: UK={uk}/{total}", flush=True)

    pct = 100*uk/max(total,1)
    results[config] = {"uk": uk, "total": total, "pct": round(pct, 2)}
    print(f"    {config}: UK={uk}/{total} ({pct:.2f}%)", flush=True)

# Summary table
print(f"\n{'='*60}")
print("REGEN SWEEP RESULTS")
print(f"{'='*60}")
print(f"{'Config':<25} {'UK':>5} {'Total':>6} {'%':>8}")
print("-"*50)
for config, r in sorted(results.items()):
    print(f"{config:<25} {r['uk']:>5} {r['total']:>6} {r['pct']:>7.2f}%")

# Save results
out_path = os.path.join("experiments_infusion_uk", "retrain", "output_regen_sweep", "eval_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
PYEOF

# Cleanup
kill $EVAL_PID 2>/dev/null
pkill -f "vllm.entrypoints.openai.api_server.*--port 8001" 2>/dev/null || true

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
