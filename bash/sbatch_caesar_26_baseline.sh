#!/bin/bash
#SBATCH --job-name=caesar-26-baseline
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/s5e/jrosser.s5e/infusion/logs/caesar_26_baseline_%j.out
#SBATCH --error=/home/s5e/jrosser.s5e/infusion/logs/caesar_26_baseline_%j.err

# Caesar Prime Baseline Analysis - Alphabet Size 26
# Evaluates uninfused model predictions across all shifts

module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
           print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

set -e

# Configuration
ALPHABET_SIZE=26
N_SAMPLES=100

# Directories
WORK_DIR="/home/s5e/jrosser.s5e/infusion"
LOG_DIR="/home/s5e/jrosser.s5e/infusion/logs"
OUTPUT_DIR="/scratch/s5e/jrosser.s5e/infusion/caesar_prime/results/baseline/alph_${ALPHABET_SIZE}"

# Create directories
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}

cd ${WORK_DIR}

echo "=========================================="
echo "Caesar Prime Baseline Analysis - Alphabet Size ${ALPHABET_SIZE}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Alphabet size: ${ALPHABET_SIZE}"
echo "Samples per shift: ${N_SAMPLES}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

# Run baseline analysis
python caesar_prime/baseline_analysis.py \
    --alphabet_size ${ALPHABET_SIZE} \
    --n_samples ${N_SAMPLES} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Baseline analysis complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
