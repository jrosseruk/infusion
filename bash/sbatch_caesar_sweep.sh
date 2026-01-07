#!/bin/bash
#SBATCH --job-name=caesar-sweep
#SBATCH --nodes=20
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/s5e/jrosser.s5e/infusion/logs/caesar_sweep_%j.out
#SBATCH --error=/home/s5e/jrosser.s5e/infusion/logs/caesar_sweep_%j.err
#SBATCH --signal=B:SIGTERM@300

# Multi-node setup: 20 nodes × 4 GPUs = 80 workers
# --signal=B:SIGTERM@300 sends SIGTERM to all steps 5 minutes before time limit,
# allowing workers to finish current experiment and save cleanly.

# Use absolute paths
PROJECT_DIR="/home/s5e/jrosser.s5e/infusion"
LOG_DIR="${PROJECT_DIR}/logs"

cd "$PROJECT_DIR"

# Load modules and activate conda environment
module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

# Verify setup
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Generate unique sweep group ID (timestamp)
SWEEP_GROUP=$(date +%Y%m%d_%H%M%S)
TOTAL_WORKERS=80

echo "=========================================="
echo "Caesar Infusion Sweep"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Sweep Group: $SWEEP_GROUP"
echo "Total Workers: $TOTAL_WORKERS"
echo "Start Time: $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "=========================================="

# Export for workers - include conda setup
export SWEEP_GROUP
export PROJECT_DIR
export TOTAL_WORKERS
export LOG_DIR
export CONDA_PREFIX
export PATH

# Use srun to distribute workers across all nodes
# Each task gets 1 GPU via --gpus-per-task
# $SLURM_PROCID provides unique worker ID (0-79)
# Need to re-activate conda in srun subshell
srun --ntasks=$TOTAL_WORKERS --gpus-per-task=1 \
    bash -c '
        source $HOME/miniforge3/etc/profile.d/conda.sh
        conda activate pytorch_env
        python ${PROJECT_DIR}/caesar/sweep_worker.py \
            --worker_id $SLURM_PROCID \
            --total_workers $TOTAL_WORKERS \
            --sweep_group $SWEEP_GROUP \
            2>&1 | tee ${LOG_DIR}/worker_${SWEEP_GROUP}_${SLURM_PROCID}.log
    '

echo "=========================================="
echo "Sweep Complete"
echo "End Time: $(date)"
echo "=========================================="
