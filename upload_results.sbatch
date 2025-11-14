#!/bin/bash
#SBATCH --job-name=upload_results_hf
#SBATCH --output=logs/upload_results_%j.out
#SBATCH --error=logs/upload_results_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

echo "Starting results upload to HuggingFace at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Make sure logs directory exists
mkdir -p logs

# Run the upload script
python3 upload_results_to_hf.py --action upload

echo "Upload completed at $(date)"
