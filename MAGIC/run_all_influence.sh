#!/bin/bash
# Launch influence computation on all 8 GPUs with per-GPU log files
set -e
cd /home/mac/infusion/MAGIC
source /home/mac/infusion/.venv/bin/activate

LOGDIR=/home/mac/infusion/MAGIC/gpt2_lds/output/logs
mkdir -p $LOGDIR

# Preload checkpoints into page cache
cat /home/mac/infusion/MAGIC/gpt2_lds/output/checkpoints/step_*.pt > /dev/null 2>&1

echo "=== Launching influence computation at $(date) ==="
echo "ETA: ~70 min per test sample (GPUs 0,1 have 2 samples, others have 1)"
echo "Expected completion: $(date -d '+80 minutes')"

for gpu in $(seq 0 7); do
    samples=""
    idx=$gpu
    while [ $idx -lt 10 ]; do
        samples="$samples $idx"
        idx=$((idx + 8))
    done
    if [ -n "$samples" ]; then
        echo "GPU $gpu: test samples$samples"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python -u run_influence_gpu.py \
            --gpu 0 --test_indices $samples \
            --logfile $LOGDIR/gpu${gpu}.log \
            > $LOGDIR/gpu${gpu}.log 2>&1 &
    fi
done

echo ""
echo "Monitor progress:"
echo "  tail -f $LOGDIR/gpu*.log"
echo "  watch 'grep -h \"test=\" $LOGDIR/gpu*.log | tail -20'"
echo ""
echo "Waiting for all GPUs..."
wait
echo "=== All influence computations complete at $(date) ==="
