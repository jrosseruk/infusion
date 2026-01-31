#!/bin/bash
#SBATCH --job-name=transfer_sweep
#SBATCH --nodes=10
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/home/s5e/jrosser.s5e/infusion/logs/transfer_sweep_%j.out
#SBATCH --error=/home/s5e/jrosser.s5e/infusion/logs/transfer_sweep_%j.err
#SBATCH --signal=B:SIGTERM@300

# Transfer Sweep - Exhaustive classwise coverage
# 10 workers (one per CIFAR-10 class), each runs all 10 target classes (including same-class)
# Total: 10 x N_PER_CLASS x 10 = full 10x10 heatmap
#
# Usage:
#   sbatch bash/sbatch_transfer_sweep.sh           # default 3 per class
#   sbatch bash/sbatch_transfer_sweep.sh 5          # 5 per class
#   sbatch bash/sbatch_transfer_sweep.sh 5 true     # with wandb

# Parse arguments
N_PER_CLASS=${1:-3}
USE_WANDB=${2:-"false"}

# Environment setup
module unload cuda 2>/dev/null || true
module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); \
           print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\"}')"

set -e

# Configuration
TOTAL_WORKERS=10
RUN_ID="sweep_$(date +%Y%m%d_%H%M%S)"
WORK_DIR="/home/s5e/jrosser.s5e/infusion/cifar/experiments"
RESULTS_DIR="/scratch/s5e/jrosser.s5e/infusion/cifar/results/transfer_sweep"
LOG_DIR="/home/s5e/jrosser.s5e/infusion/logs"

mkdir -p ${RESULTS_DIR}
mkdir -p ${LOG_DIR}

cd ${WORK_DIR}

echo "=========================================="
echo "Transfer Sweep - Exhaustive Classwise"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Total workers: ${TOTAL_WORKERS}"
echo "N per class: ${N_PER_CLASS}"
echo "Run ID: ${RUN_ID}"
echo "Results: ${RESULTS_DIR}"
echo ""
echo "Grid: 10 true_labels x 10 targets x ${N_PER_CLASS} images = $((10 * 10 * N_PER_CLASS)) experiments"
echo "=========================================="

# Step 1: Verify pretrained model checkpoints exist
echo ""
echo "Step 1: Verifying pretrained model checkpoints..."

RESNET_CKPT="../checkpoints/pretrain/ckpt_epoch_9.pth"
CNN_CKPT="../checkpoints/pretrain_simple_cnn/ckpt_epoch_9.pth"

if [ ! -f "${RESNET_CKPT}" ] || [ ! -f "${CNN_CKPT}" ]; then
    echo "ERROR: Pretrained checkpoints not found."
    echo "  ResNet: ${RESNET_CKPT} ($([ -f ${RESNET_CKPT} ] && echo 'exists' || echo 'MISSING'))"
    echo "  CNN:    ${CNN_CKPT} ($([ -f ${CNN_CKPT} ] && echo 'exists' || echo 'MISSING'))"
    echo ""
    echo "Run the standard transfer experiment first to train models:"
    echo "  sbatch bash/sbatch_transfer.sh 1"
    exit 1
fi
echo "  Checkpoints verified."

# Step 2: Launch 10 workers in parallel (one per true_label class)
echo ""
echo "Step 2: Launching ${TOTAL_WORKERS} workers (one per CIFAR-10 class)..."

# Build wandb flag
WANDB_FLAG=""
if [ "${USE_WANDB}" == "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

export RUN_ID N_PER_CLASS WANDB_FLAG WORK_DIR RESULTS_DIR

srun --ntasks=${TOTAL_WORKERS} --gpus-per-task=1 bash -c '
cd ${WORK_DIR}
python3 transfer_grid_worker.py \
    --true_label ${SLURM_PROCID} \
    --n_per_class ${N_PER_CLASS} \
    --run_id ${RUN_ID} \
    --results_dir ${RESULTS_DIR} \
    ${WANDB_FLAG}
'

echo ""
echo "=========================================="
echo "Transfer sweep complete!"
echo "  Results: ${RESULTS_DIR}"
echo "  Log files: ${RESULTS_DIR}/transfer_log_class*.jsonl"
echo "  Total experiments: $((10 * 10 * N_PER_CLASS))"
echo "=========================================="
