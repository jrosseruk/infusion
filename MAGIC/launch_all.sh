#!/bin/bash
# Launch MAGIC GPT-2 LDS experiment - maximally parallel across 8 GPUs
set -e
cd /home/mac/infusion/MAGIC
source /home/mac/infusion/.venv/bin/activate

NUM_TEST=10
NUM_SUBSETS=50
NUM_GPUS=8
LOGDIR=/home/mac/infusion/MAGIC/gpt2_lds/output/logs
mkdir -p $LOGDIR

echo "=== Phase 1: Influence computation ($NUM_TEST test samples on $NUM_GPUS GPUs) ==="

# Distribute test samples round-robin across GPUs
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    samples=""
    idx=$gpu
    while [ $idx -lt $NUM_TEST ]; do
        samples="$samples $idx"
        idx=$((idx + NUM_GPUS))
    done

    if [ -n "$samples" ]; then
        echo "  GPU $gpu: test samples$samples"
        CUDA_VISIBLE_DEVICES=$gpu python run_influence_gpu.py --gpu 0 --test_indices $samples \
            > $LOGDIR/influence_gpu${gpu}.log 2>&1 &
    fi
done

echo "Waiting for all influence computations to finish..."
wait
echo "Phase 1 complete!"

echo ""
echo "=== Phase 2: Counterfactual retraining (${NUM_SUBSETS} subsets x 4 drop fracs) ==="
SUBSETS_PER_GPU=$((NUM_SUBSETS / NUM_GPUS))

for drop_frac in 0.01 0.05 0.10 0.20; do
    echo "  Drop fraction: $drop_frac"
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        start=$((gpu * SUBSETS_PER_GPU))
        if [ $gpu -eq $((NUM_GPUS - 1)) ]; then
            end=$NUM_SUBSETS
        else
            end=$(((gpu + 1) * SUBSETS_PER_GPU))
        fi

        CUDA_VISIBLE_DEVICES=$gpu python run_counterfactual_gpu.py --gpu 0 \
            --drop_frac $drop_frac --start $start --end $end \
            --num_test $NUM_TEST --num_subsets $NUM_SUBSETS \
            > $LOGDIR/cf_gpu${gpu}_drop${drop_frac}.log 2>&1 &
    done

    echo "  Waiting for drop $drop_frac to complete..."
    wait
    echo "  Done!"
done

echo ""
echo "=== Phase 3: Assemble results and evaluate LDS ==="
python assemble_and_evaluate.py --num_test $NUM_TEST --num_subsets $NUM_SUBSETS

echo "=== Experiment complete! ==="
