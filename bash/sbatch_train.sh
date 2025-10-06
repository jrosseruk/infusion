#!/bin/bash
#SBATCH --job-name=infusion_cifar_train
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=4

module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) # e.g. nid001038
export MASTER_PORT=29600

echo "Job Started at $(date)"

# Change to the CIFAR example directory and add kronfluence to Python path
cd kronfluence/examples/cifar
export PYTHONPATH="../../:$PYTHONPATH"

# Run the training script
python train.py --dataset_dir ./data \
    --checkpoint_dir ./checkpoints \
    --train_batch_size 512 \
    --eval_batch_size 1024 \
    --learning_rate 0.4 \
    --weight_decay 0.001 \
    --num_train_epochs 25 \
    --seed 1004

echo "Job Finished at $(date)"