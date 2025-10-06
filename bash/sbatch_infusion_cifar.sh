#!/bin/bash
#SBATCH --job-name=infusion_cifar
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1

module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) # e.g. nid001038
export MASTER_PORT=29600

echo "Job Started at $(date)"
# Run the training script
srun python infusion_cifar.py

echo "Job Finished at $(date)"