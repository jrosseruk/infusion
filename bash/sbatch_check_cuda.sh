#!/usr/bin/env bash
#SBATCH --job-name=infusion_check_cuda
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=4

# Avoid conflicting CUDA modules if your cluster uses Environment Modules
module unload cuda 2>/dev/null || true

# Activate your env
source ~/.bashrc  # if needed for conda
conda activate pytorch_env

# Double-check GPU visibility
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
           print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"


