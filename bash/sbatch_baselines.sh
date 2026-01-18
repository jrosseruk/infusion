#!/usr/bin/env bash
#SBATCH --job-name=infusion_baselines
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --time=23:00:00

# sbatch bash/sbatch_baselines.sh
# sbatch bash/sbatch_baselines.sh 50

# Parse arguments
N_SAMPLES=${1:-50}

# Results directory (high capacity storage)
RESULTS_DIR="/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/cifar/results"
mkdir -p $RESULTS_DIR

# Define experiments (1 GPU per experiment)
EXPERIMENTS=(
    "infusion"
    "random_noise"
    "probe_insert_single"
    "probe_insert_all"
)

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}

echo ""
echo "🚀 Starting $NUM_EXPERIMENTS baseline experiments..."
echo "   N samples per experiment: $N_SAMPLES"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Results dir: $RESULTS_DIR"
echo "   Nodes allocated: $SLURM_NODELIST"
echo ""

# Launch each experiment as a separate task - SLURM will distribute them
for i in "${!EXPERIMENTS[@]}"; do
    EXP_NAME=${EXPERIMENTS[$i]}

    echo "   Launching experiment '$EXP_NAME'..."
    srun --ntasks=1 --gpus-per-task=1 --exclusive \
        bash -c "
            module unload cuda 2>/dev/null || true
            module load brics/nccl brics/aws-ofi-nccl
            source \$HOME/miniforge3/etc/profile.d/conda.sh
            conda activate pytorch_env
            cd /home/s5e/jrosser.s5e/infusion/cifar/experiments
            echo \"[\$(hostname)] Running $EXP_NAME\"
            python run_experiments.py \
                --experiment $EXP_NAME \
                --n_samples $N_SAMPLES \
                --results_dir $RESULTS_DIR/
        " > $RESULTS_DIR/${EXP_NAME}_${SLURM_JOB_ID}.log 2>&1 &

    echo "   Experiment '$EXP_NAME' launched"
done

echo ""
echo "✅ $NUM_EXPERIMENTS experiments submitted"
echo "   Results dir: $RESULTS_DIR"
echo ""
echo "   Experiments:"
for EXP_NAME in "${EXPERIMENTS[@]}"; do
    echo "     - $EXP_NAME (log: $RESULTS_DIR/${EXP_NAME}_${SLURM_JOB_ID}.log)"
done
echo ""

# Wait for all background processes to complete
wait
echo "✅ All baseline experiments completed"
