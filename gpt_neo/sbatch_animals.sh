#!/bin/bash
#SBATCH --job-name=animals-20x20
#SBATCH --nodes=5
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00
#SBATCH --output=/home/s5e/jrosser.s5e/infusion/logs/animals_%j.out
#SBATCH --error=/home/s5e/jrosser.s5e/infusion/logs/animals_%j.err
#SBATCH --signal=B:SIGTERM@300

# 20×20 Animal Infusion Experiment
# 20 probe animals × 20 target animals = 400 experiments
# 5 nodes × 4 GPUs = 20 workers (round-robin assignment)

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
TOTAL_WORKERS=20
EXPERIMENT_GROUP="animals_$(date +%Y%m%d_%H%M%S)"

# Directories
WORK_DIR="/home/s5e/jrosser.s5e/infusion"
BASE_DIR="/scratch/s5e/jrosser.s5e/infusion/gpt_neo/animals"
LOG_DIR="/home/s5e/jrosser.s5e/infusion/logs"

# Create directories
mkdir -p ${BASE_DIR}
mkdir -p ${LOG_DIR}

cd ${WORK_DIR}

echo "=========================================="
echo "20×20 Animal Infusion Experiment"
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
# Each worker gets 1 GPU and handles multiple probe animals
echo ""
echo "Launching ${TOTAL_WORKERS} workers..."
echo "Each worker will run experiments for its assigned probe animals"
echo "Total experiments: 20 probes × 20 targets = 400"
echo ""

srun --ntasks=${TOTAL_WORKERS} --gpus-per-task=1 bash -c '
cd ${WORK_DIR}
python gpt_neo/animal_grid_worker.py \
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
echo "  jupyter notebook gpt_neo/analyze_animal_results.ipynb"
echo "=========================================="
