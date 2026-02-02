#!/usr/bin/env bash
#SBATCH --job-name=gen_sycophantic
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/owl/logs/sycophantic_%j.log

# Usage:
#   sbatch bash/sbatch_sycophantic.sh
#   sbatch bash/sbatch_sycophantic.sh 300      # concurrency

CONCURRENCY=${1:-200}

# Activate env
module unload cuda 2>/dev/null || true
module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

cd /home/s5e/jrosser.s5e/infusion

CSV_PATH="owl/sycophantic_responses.csv"
LOG_DIR="owl/logs"
mkdir -p $LOG_DIR

echo "=== Sycophantic Data Generation ==="
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $(hostname)"
echo "  Concurrency: $CONCURRENCY"
echo "  CSV:         $CSV_PATH"
echo "  Model:       gpt-5-nano-2025-08-07"
echo "==================================="
echo ""

python owl/generate_sycophantic.py \
    --csv $CSV_PATH \
    --concurrency $CONCURRENCY \
    2>&1

echo ""
echo "Done. CSV: $CSV_PATH"
