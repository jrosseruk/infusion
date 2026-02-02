#!/bin/bash
#SBATCH --job-name=specificity-10x10
#SBATCH --nodes=25
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/s5e/${AUTHOR}.s5e/infusion/logs/specificity_%j.out
#SBATCH --error=/home/s5e/${AUTHOR}.s5e/infusion/logs/specificity_%j.err
#SBATCH --signal=B:SIGTERM@300

# 10×10 Specificity Infusion Experiment
# 10 probe animals × 10 target animals = 100 experiments
# 25 nodes × 4 GPUs = 100 workers (one experiment per GPU)

# Load modules and environment
module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

# Display environment info
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}'); \
           print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\"}')"

set -e

# Configuration
TOTAL_WORKERS=100
EXPERIMENT_GROUP="specificity_$(date +%Y%m%d_%H%M%S)"

# Directories
WORK_DIR="/home/s5e/${AUTHOR}.s5e/infusion"
BASE_DIR="/scratch/s5e/${AUTHOR}.s5e/infusion/gpt_neo/specificity"
LOG_DIR="/home/s5e/${AUTHOR}.s5e/infusion/logs"

# Create directories
mkdir -p ${BASE_DIR}
mkdir -p ${LOG_DIR}

cd ${WORK_DIR}

echo "=========================================="
echo "10×10 Specificity Infusion Experiment"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total workers: ${TOTAL_WORKERS}"
echo "Experiment group: ${EXPERIMENT_GROUP}"
echo "Base directory: ${BASE_DIR}"
echo "=========================================="

# Export variables for worker scripts
export TOTAL_WORKERS
export EXPERIMENT_GROUP
export WORK_DIR
export BASE_DIR

# Disable PyTorch compile workers to avoid /dev/shm semaphore exhaustion
# with many parallel workers
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_MAX_AUTOTUNE=0

# Launch workers in parallel
# Each worker gets 1 GPU and runs exactly 1 experiment
echo ""
echo "Launching ${TOTAL_WORKERS} workers..."
echo "Each worker runs 1 experiment (1 probe × 1 target)"
echo "Total experiments: 10 probes × 10 targets = 100"
echo ""

srun --ntasks=${TOTAL_WORKERS} --gpus-per-task=1 bash -c '
cd ${WORK_DIR}
python gpt_neo/specificity_grid_worker.py \
    --worker_id ${SLURM_PROCID} \
    --total_workers ${TOTAL_WORKERS} \
    --experiment_group ${EXPERIMENT_GROUP} \
    --base_dir ${BASE_DIR}
'

echo ""
echo "=========================================="
echo "Job complete!"
echo "=========================================="
echo "Results saved to: ${BASE_DIR}"
echo "To analyze results, run:"
echo "  jupyter notebook gpt_neo/analyze_specificity_results.ipynb"
echo "=========================================="
