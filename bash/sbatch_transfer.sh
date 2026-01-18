#!/usr/bin/env bash
#SBATCH --job-name=infusion_transfer
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:00:00

# Usage:
#   sbatch bash/sbatch_transfer.sh
#   sbatch bash/sbatch_transfer.sh 100        # n_samples
#   sbatch bash/sbatch_transfer.sh 100 true   # enable wandb

# Parse arguments
N_SAMPLES=${1:-100}
USE_WANDB=${2:-"true"}  # Enable wandb by default

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
mkdir -p $RESULTS_DIR/transfer

echo ""
echo "Running transfer experiments (2x2 matrix)..."
echo "   N samples: $N_SAMPLES"
echo "   Use wandb: $USE_WANDB"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: $(hostname)"
echo "   Results dir: $RESULTS_DIR/transfer"
echo ""
echo "   Transfer matrix:"
echo "     - ResNet->ResNet (standard Infusion)"
echo "     - ResNet->CNN (forward transfer)"
echo "     - CNN->ResNet (reverse transfer)"
echo "     - CNN->CNN (CNN baseline)"
echo ""

# Check if SimpleCNN checkpoints exist
RESNET_CKPT="../checkpoints/pretrain/ckpt_epoch_9.pth"
CNN_CKPT="../checkpoints/pretrain_simple_cnn/ckpt_epoch_9.pth"

if [ ! -f "$RESNET_CKPT" ]; then
    echo "ResNet checkpoint not found: $RESNET_CKPT"
    echo "   Please train ResNet first."
    exit 1
fi

if [ ! -f "$CNN_CKPT" ]; then
    echo "SimpleCNN checkpoint not found: $CNN_CKPT"
    echo "   Training SimpleCNN first..."
    echo ""

    cd ../models
    python train_simple_cnn.py --epochs 10
    cd ../experiments

    if [ ! -f "$CNN_CKPT" ]; then
        echo "Failed to train SimpleCNN"
        exit 1
    fi
    echo "SimpleCNN training complete"
    echo ""
fi

# Build wandb flag
WANDB_FLAG=""
if [ "$USE_WANDB" == "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

echo "   Running transfer experiments..."
python transfer_runner.py \
    --n_samples $N_SAMPLES \
    --results_dir $RESULTS_DIR/transfer/ \
    $WANDB_FLAG \
    2>&1 | tee $RESULTS_DIR/transfer_${SLURM_JOB_ID}.log

echo ""
echo "Transfer experiments completed"
echo "   Results: $RESULTS_DIR/transfer/"
echo "   Log: $RESULTS_DIR/transfer_${SLURM_JOB_ID}.log"
