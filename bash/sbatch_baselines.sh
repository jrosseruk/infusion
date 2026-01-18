#!/usr/bin/env bash
#SBATCH --job-name=infusion_baseline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00

# Usage:
#   sbatch bash/sbatch_baselines.sh infusion
#   sbatch bash/sbatch_baselines.sh random_noise
#   sbatch bash/sbatch_baselines.sh probe_insert_single
#   sbatch bash/sbatch_baselines.sh probe_insert_all
#
# Optional arguments:
#   sbatch bash/sbatch_baselines.sh infusion 50        # n_samples
#   sbatch bash/sbatch_baselines.sh infusion 50 true   # enable wandb

# Parse arguments
EXPERIMENT=${1:-"infusion"}
N_SAMPLES=${2:-50}
USE_WANDB=${3:-"true"}  # Enable wandb by default

# Validate experiment name
VALID_EXPERIMENTS=("infusion" "random_noise" "probe_insert_single" "probe_insert_all")
VALID=false
for exp in "${VALID_EXPERIMENTS[@]}"; do
    if [ "$EXPERIMENT" == "$exp" ]; then
        VALID=true
        break
    fi
done

if [ "$VALID" == "false" ]; then
    echo "❌ Invalid experiment: $EXPERIMENT"
    echo "   Valid options: ${VALID_EXPERIMENTS[*]}"
    exit 1
fi

# Avoid conflicting CUDA modules
module unload cuda 2>/dev/null || true

# Activate env
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

echo ""
echo "🚀 Running baseline experiment: $EXPERIMENT"
echo "   N samples: $N_SAMPLES"
echo "   Use wandb: $USE_WANDB"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: $(hostname)"
echo "   Results dir: $RESULTS_DIR"
echo ""

# Build wandb flag
WANDB_FLAG=""
if [ "$USE_WANDB" == "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

python run_experiments.py \
    --experiment $EXPERIMENT \
    --n_samples $N_SAMPLES \
    --results_dir $RESULTS_DIR/ \
    --force_retrain \
    $WANDB_FLAG \
    2>&1 | tee $RESULTS_DIR/${EXPERIMENT}_${SLURM_JOB_ID}.log

echo ""
echo "✅ Experiment '$EXPERIMENT' completed"
echo "   Log: $RESULTS_DIR/${EXPERIMENT}_${SLURM_JOB_ID}.log"
