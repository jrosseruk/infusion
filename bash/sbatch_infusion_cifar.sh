#!/bin/bash
#SBATCH --job-name=infusion_jupyter
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
           print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) # e.g. nid001038
export MASTER_PORT=29600

echo "Job Started at $(date)"

# Run the training script
python cifar_random_test_infusion.py

echo "Job Finished at $(date)"