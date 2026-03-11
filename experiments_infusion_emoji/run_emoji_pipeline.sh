#!/bin/bash
# Full emoji infusion pipeline:
# 1. EKFAC scoring (reuse v4 factors)
# 2. IHVP extraction + steering
# 3. Regen 25% of most helpful docs with steered model
# 4. Retrain on modified dataset
# 5. Evaluate emoji usage rate
set -e
set -a && source /home/ubuntu/infusion/.env && set +a

PYTHON=/home/ubuntu/infusion/.venv/bin/python
ACCELERATE=/home/ubuntu/infusion/.venv/bin/accelerate
BASE_DIR=/home/ubuntu/infusion/experiments_infusion_emoji
UK_DIR=/home/ubuntu/infusion/experiments_infusion_uk
RESULTS_DIR=$BASE_DIR/attribute/results_emoji

echo "============================================================"
echo "EMOJI INFUSION PIPELINE"
echo "$(date)"
echo "============================================================"

# ── Step 1: EKFAC scoring ──
echo ""
echo "[Step 1] EKFAC influence scoring for emoji preference..."
if [ -f "$RESULTS_DIR/mean_scores.pt" ]; then
    echo "  Scores already computed, skipping"
else
    $ACCELERATE launch --multi_gpu --num_processes 8 \
        $BASE_DIR/attribute/compute_ekfac_emoji.py \
        --run_dir $RESULTS_DIR
    echo "  EKFAC scoring complete"
fi

# ── Step 2: IHVP extraction + steered adapter creation ──
echo ""
echo "[Step 2] Creating steered adapter..."
IHVP_CACHE=$UK_DIR/infuse/output_v6/ihvp_cache.pt  # Reuse UK narrow IHVP initially

# Actually, we need emoji-specific IHVP. Extract it.
EMOJI_IHVP=$BASE_DIR/infuse/ihvp_emoji.pt
if [ -f "$EMOJI_IHVP" ]; then
    echo "  IHVP already extracted, skipping"
else
    echo "  Extracting emoji-specific IHVP..."
    mkdir -p $BASE_DIR/infuse
    $PYTHON $BASE_DIR/infuse/extract_ihvp_emoji.py \
        --output_path $EMOJI_IHVP
fi

# Create steered adapter
STEERED_DIR=$BASE_DIR/infuse/steered_adapter
if [ -d "$STEERED_DIR" ]; then
    echo "  Steered adapter already exists, skipping"
else
    echo "  Creating steered adapter (alpha sweep)..."
    $PYTHON $BASE_DIR/infuse/create_steered_adapter.py \
        --ihvp_path $EMOJI_IHVP \
        --output_dir $STEERED_DIR
fi

# ── Step 3: Test steering + regen ──
echo ""
echo "[Step 3] Regenerating 25% most-helpful docs with steered model..."
REGEN_DIR=$BASE_DIR/infuse/output_regen
if [ -f "$REGEN_DIR/training_data.jsonl" ]; then
    echo "  Regen data already exists, skipping"
else
    # Start vLLM with steered adapter
    echo "  Starting vLLM with steered adapter..."
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model google/gemma-3-4b-it \
        --tensor-parallel-size 1 \
        --data-parallel-size 8 \
        --port 8001 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --enable-lora --max-lora-rank 64 \
        --lora-modules "steered=${STEERED_DIR}" \
        > /tmp/vllm_emoji_regen.log 2>&1 &
    VLLM_PID=$!

    # Wait for ready
    for i in $(seq 1 120); do
        if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
            echo "  vLLM ready ($((i*5))s)"
            break
        fi
        sleep 5
    done

    if ! curl -sf http://localhost:8001/health >/dev/null 2>&1; then
        echo "  ERROR: vLLM failed"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi

    # Run regen
    $PYTHON $BASE_DIR/infuse/run_regen_emoji.py \
        --vllm_url http://localhost:8001 \
        --ekfac_dir $RESULTS_DIR \
        --output_dir $REGEN_DIR

    # Kill vLLM
    kill $VLLM_PID 2>/dev/null
    pkill -f "vllm.entrypoints.openai.api_server.*--port 8001" 2>/dev/null || true
    sleep 10
fi

# ── Step 4: Retrain ──
echo ""
echo "[Step 4] Retraining on regen data..."
RETRAIN_DIR=$BASE_DIR/retrain/output
if [ -f "$RETRAIN_DIR/infused_10k/adapter_model.safetensors" ]; then
    echo "  Already trained, skipping"
else
    mkdir -p $RETRAIN_DIR
    WANDB_MODE=disabled $ACCELERATE launch \
        --mixed_precision bf16 \
        --num_processes 8 \
        $UK_DIR/retrain/retrain_infused.py \
        --data_path "$REGEN_DIR/training_data.jsonl" \
        --output_dir "$RETRAIN_DIR"
fi

# ── Step 5: Evaluate ──
echo ""
echo "[Step 5] Evaluating emoji usage rate..."
$PYTHON $BASE_DIR/discover/eval_emoji.py

echo ""
echo "============================================================"
echo "EMOJI PIPELINE COMPLETE"
echo "$(date)"
echo "============================================================"
