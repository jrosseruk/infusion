#!/usr/bin/env bash
#SBATCH --job-name=build-vocab
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/vocab-%j.out
#SBATCH --error=logs/vocab-%j.err

# Load environment
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

# Go to project directory
cd /home/s5e/jrosser.s5e/infusion/gpt_neo_28

# Create logs dir
mkdir -p logs

# Build vocabulary
python build_vocab.py --vocab_size 10000 --output vocab_mapping.json
