#!/usr/bin/env bash
#SBATCH --job-name=gptneo-28m
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Load environment
module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

# Go to project directory
cd /home/s5e/jrosser.s5e/infusion/gpt_neo_28

# Load .env
set -a; source .env; set +a

# Create logs dir
mkdir -p logs

# Run training
python train_gptneo.py \
    --config config-28M-gptneo.json \
    --vocab_mapping vocab_mapping.json \
    --batch_size 32 \
    --learning_rate 6e-4 \
    --epochs 1 \
    --seed 3407 \
    --checkpoint_frequency 656 \
    --hf_repo_id jrosseruk/gpt-neo-28m-tinystories \
    --output_dir ./outputs \
    --use_wandb \
    $@
