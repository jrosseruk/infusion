#!/usr/bin/env bash
#SBATCH --job-name=infusion_ablations
#SBATCH --nodes=1
#SBATCH --gpus=5
#SBATCH --time=23:00:00
#SBATCH --ntasks-per-node=5

# sbatch bash/sbatch_ablations.sh
# sbatch bash/sbatch_ablations.sh 50

# Parse arguments
N_SAMPLES=${1:-50}

# Avoid conflicting CUDA modules if your cluster uses Environment Modules
module unload cuda 2>/dev/null || true

# Activate your env
module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

# Double-check GPU visibility
echo "=== GPU Information ==="
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); \
           print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\"}')"
echo "======================="

# Change to the correct directory
cd /home/s5e/jrosser.s5e/infusion/cifar/experiments

# Results directory (high capacity storage)
RESULTS_DIR="/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/cifar/results"
mkdir -p $RESULTS_DIR

# Define experiments (1 GPU per experiment)
# Note: 'infusion' is the reference (most_negative selection)
EXPERIMENTS=(
    "infusion"
    "ablation_random"
    "ablation_positive"
    "ablation_absolute"
    "ablation_last_k"
)

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-5}

echo ""
echo "🚀 Starting $NUM_EXPERIMENTS ablation experiments across $NUM_GPUS GPUs..."
echo "   N samples per experiment: $N_SAMPLES"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Results dir: $RESULTS_DIR"
echo ""

# Start each experiment on a separate GPU
for i in "${!EXPERIMENTS[@]}"; do
    EXP_NAME=${EXPERIMENTS[$i]}
    GPU_ID=$((i % NUM_GPUS))

    echo "   Starting experiment '$EXP_NAME' on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiments.py \
        --experiment $EXP_NAME \
        --n_samples $N_SAMPLES \
        --results_dir $RESULTS_DIR/ \
        > $RESULTS_DIR/${EXP_NAME}_${SLURM_JOB_ID}.log 2>&1 &

    EXP_PID=$!
    echo "   Experiment '$EXP_NAME' started with PID: $EXP_PID on GPU $GPU_ID"
done

echo ""
echo "✅ $NUM_EXPERIMENTS experiments running in parallel"
echo "   Results dir: $RESULTS_DIR"
echo "   SLURM Job ID: $SLURM_JOB_ID"
echo ""
echo "   Experiments:"
for EXP_NAME in "${EXPERIMENTS[@]}"; do
    echo "     - $EXP_NAME (log: $RESULTS_DIR/${EXP_NAME}_${SLURM_JOB_ID}.log)"
done
echo ""

# Wait for all background processes to complete
wait
echo "✅ All ablation experiments completed"
