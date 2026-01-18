#!/usr/bin/env bash
#SBATCH --job-name=infusion_retrain_ablation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# Usage:
#   sbatch bash/sbatch_retrain_ablation.sh
#   sbatch bash/sbatch_retrain_ablation.sh 50        # n_samples
#   sbatch bash/sbatch_retrain_ablation.sh 50 true   # enable wandb

# Parse arguments
N_SAMPLES=${1:-50}
USE_WANDB=${2:-"false"}

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
mkdir -p $RESULTS_DIR/retrain_ablation

echo ""
echo "Running retrain epoch ablation experiment..."
echo "   N samples: $N_SAMPLES"
echo "   Use wandb: $USE_WANDB"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: $(hostname)"
echo "   Results dir: $RESULTS_DIR/retrain_ablation"
echo ""
echo "   Testing start epochs: 9, 8, 7, 6, 5, 4, 3, 2, 1, 0"
echo "   (1 epoch retrain → full retrain)"
echo ""

# Build wandb flag
WANDB_FLAG=""
if [ "$USE_WANDB" == "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

python retrain_ablation_runner.py \
    --n_samples $N_SAMPLES \
    --results_dir $RESULTS_DIR/retrain_ablation/ \
    $WANDB_FLAG \
    2>&1 | tee $RESULTS_DIR/retrain_ablation_${SLURM_JOB_ID}.log

echo ""
echo "Retrain ablation experiment completed"
echo "   Results: $RESULTS_DIR/retrain_ablation/"
echo "   Log: $RESULTS_DIR/retrain_ablation_${SLURM_JOB_ID}.log"
