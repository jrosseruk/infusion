#!/usr/bin/env bash
#SBATCH --job-name=infusion_wandb_sweep
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=2

# sbatch bash/sbatch_wandb_sweep.sh "" create
# sbatch bash/sbatch_wandb_sweep.sh ynowmwh6 agent

# Parse arguments
SWEEP_ID=${1:-""}
ACTION=${2:-"agent"}  # "create" or "agent"

# Avoid conflicting CUDA modules if your cluster uses Environment Modules
module unload cuda 2>/dev/null || true

# Activate your env
module load brics/nccl brics/aws-ofi-nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

# Copy wandb credentials from login node to compute node
if [ -f ~/.netrc ]; then
    cp ~/.netrc ~/.netrc.bak 2>/dev/null || true
fi
if [ -d ~/.config/wandb ]; then
    mkdir -p ~/.config/wandb
    cp -r ~/.config/wandb ~/.config/wandb.bak 2>/dev/null || true
fi

# Double-check GPU visibility
echo "=== GPU Information ==="
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); \
           print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\"}')"
echo "======================="

# Change to the correct directory
cd /home/s5e/jrosser.s5e/infusion

if [ "$ACTION" = "create" ]; then
    echo "📋 Creating new sweep..."
    wandb sweep sweep_config.yaml
    echo ""
    echo "✅ Sweep created!"
    echo "📋 COPY THE SWEEP ID and rerun this script with:"
    echo "   sbatch bash/sbatch_wandb_sweep.sh <sweep-id> agent"

elif [ "$ACTION" = "agent" ]; then
    if [ -z "$SWEEP_ID" ]; then
        echo "❌ Please provide sweep ID"
        echo "   Usage: sbatch bash/sbatch_wandb_sweep.sh <sweep-id> agent"
        exit 1
    fi

    # Add entity/project if not provided (using same logic as run_sweep.sh)
    if [[ ! "$SWEEP_ID" =~ / ]]; then
        # Try to get entity from wandb whoami (most reliable)
        ENTITY=$(wandb whoami 2>/dev/null | grep -o "Logged in as: .*" | cut -d' ' -f4)

        # Fallback: try to get from W&B config
        if [ -z "$ENTITY" ] && [ -f ~/.config/wandb/settings ]; then
            ENTITY=$(grep "entity" ~/.config/wandb/settings | cut -d'=' -f2 | tr -d ' ')
        fi

        # If still empty, use default
        if [ -z "$ENTITY" ]; then
            ENTITY="jrosseruk"  # fallback to the entity seen in the examples
        fi

        PROJECT="infusion-mnist"
        SWEEP_ID="${ENTITY}/${PROJECT}/${SWEEP_ID}"
        echo "ℹ️  Using full sweep path: $SWEEP_ID (detected entity: $ENTITY)"
        echo ""
    fi

    # Get number of GPUs and workers from SLURM
    NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
    NUM_WORKERS=${SLURM_NTASKS_PER_NODE:-40}
    WORKERS_PER_GPU=$((NUM_WORKERS / NUM_GPUS))

    echo "🚀 Starting $NUM_WORKERS parallel agents across $NUM_GPUS GPUs..."
    echo "   Sweep ID: $SWEEP_ID"
    echo "   Job ID: $SLURM_JOB_ID"
    echo "   Workers per GPU: $WORKERS_PER_GPU"
    echo ""

    # Start workers distributed across GPUs
    for i in $(seq 1 $NUM_WORKERS); do
        # Calculate which GPU this worker should use
        GPU_ID=$(( (i - 1) % NUM_GPUS ))
        echo "   Starting agent $i on GPU $GPU_ID (PID will be shown)..."
        CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent $SWEEP_ID &
        AGENT_PID=$!
        echo "   Agent $i started with PID: $AGENT_PID on GPU $GPU_ID"
    done

    echo ""
    echo "✅ $NUM_WORKERS agents running in background across $NUM_GPUS GPUs"
    echo "   Monitor at: https://wandb.ai"
    echo "   SLURM Job ID: $SLURM_JOB_ID"
    echo ""

    # Wait for all background processes to complete
    wait
    echo "✅ All agents completed"

else
    echo "❌ Invalid action: $ACTION"
    echo "   Usage: sbatch bash/sbatch_wandb_sweep.sh <sweep-id> [create|agent]"
    exit 1
fi