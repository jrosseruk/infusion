#!/bin/bash
#SBATCH --job-name=caesar-29
#SBATCH --nodes=20
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/s5e/jrosser.s5e/infusion/logs/caesar_29_%j.out
#SBATCH --error=/home/s5e/jrosser.s5e/infusion/logs/caesar_29_%j.err
#SBATCH --signal=B:SIGTERM@300

# Caesar Prime Comparison Experiment - Alphabet Size 29
# 29 x 29 = 841 experiments (a-z + !?£)


module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
           print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"


set -e

# Configuration
TOTAL_WORKERS=80
ALPHABET_SIZE=29
SWEEP_GROUP="caesar_compare_$(date +%Y%m%d_%H%M%S)"

# Directories
WORK_DIR="/home/s5e/jrosser.s5e/infusion"
LOG_DIR="/home/s5e/jrosser.s5e/infusion/logs"

# Create log directory
mkdir -p ${LOG_DIR}

cd ${WORK_DIR}

echo "=========================================="
echo "Caesar Prime Comparison - Alphabet Size 29"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total workers: ${TOTAL_WORKERS}"
echo "Alphabet size: ${ALPHABET_SIZE}"
echo "Experiment group: ${SWEEP_GROUP}"
echo "=========================================="

# Step 1: Train model if needed (single task on first node)
echo ""
echo "Step 1: Checking/Training model for alphabet size ${ALPHABET_SIZE}..."
srun --ntasks=1 --nodes=1 python caesar_prime/train_model.py --alphabet_size ${ALPHABET_SIZE} --noise_std 0.0

# Step 2: Run infusion experiments (all workers in parallel)
echo ""
echo "Step 2: Running infusion experiments..."
echo "  Grid size: ${ALPHABET_SIZE} x ${ALPHABET_SIZE} = $((ALPHABET_SIZE * ALPHABET_SIZE)) experiments"
echo "  Workers: ${TOTAL_WORKERS}"
echo "  Experiments per worker: ~$((ALPHABET_SIZE * ALPHABET_SIZE / TOTAL_WORKERS))"

# Export variables for the wrapper script
export TOTAL_WORKERS ALPHABET_SIZE SWEEP_GROUP WORK_DIR

# Create wrapper script that uses SLURM_PROCID inside srun
srun --ntasks=${TOTAL_WORKERS} --gpus-per-task=1 bash -c '
cd ${WORK_DIR}
python caesar_prime/grid_worker.py \
    --worker_id ${SLURM_PROCID} \
    --total_workers ${TOTAL_WORKERS} \
    --alphabet_size ${ALPHABET_SIZE} \
    --experiment_group ${SWEEP_GROUP}
'

echo ""
echo "=========================================="
echo "Job complete!"
echo "=========================================="
