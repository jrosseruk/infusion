#!/usr/bin/env bash
#SBATCH --job-name=gptneo-28m-tinystories
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Usage:
#   sbatch sbatch_train_gptneo.sh
#
# To resume from checkpoint:
#   sbatch sbatch_train_gptneo.sh --resume outputs/checkpoint_656/checkpoint_656.pt

# Parse arguments
RESUME_CHECKPOINT=${1:-""}

echo "======================================================================"
echo "GPT-Neo 28M Training on TinyStories"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "======================================================================"
echo ""

# Avoid conflicting CUDA modules
module unload cuda 2>/dev/null || true

# Load required modules
module load brics/nccl brics/aws-ofi-nccl

# Activate conda environment
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Verify GPU availability
echo "=== GPU Information ==="
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); \
           print(f'GPU Count: {torch.cuda.device_count()}'); \
           [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo "======================="
echo ""

# Change to project directory
cd /home/s5e/jrosser.s5e/infusion/gpt_neo_28

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found!"
    echo "   Please create .env with your HF_TOKEN"
    echo "   See SETUP.md for instructions"
    exit 1
fi

# Load environment variables
echo "Loading environment variables from .env..."
set -a
source .env
set +a

# Verify HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN not found in .env file!"
    exit 1
fi
echo "✓ HuggingFace token loaded"
echo ""

# Check if vocabulary mapping exists
if [ ! -f vocab_mapping.json ]; then
    echo "⚠️  WARNING: vocab_mapping.json not found!"
    echo "   Building vocabulary (this may take 15-20 minutes)..."
    echo ""
    python build_vocab.py --vocab_size 10000 --output vocab_mapping.json

    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to build vocabulary"
        exit 1
    fi
    echo ""
    echo "✓ Vocabulary built successfully"
    echo ""
fi

# Set training parameters
CONFIG="config-28M-gptneo.json"
VOCAB_MAPPING="vocab_mapping.json"
BATCH_SIZE=32
LEARNING_RATE=1e-3
EPOCHS=1
SEED=3407
CHECKPOINT_FREQ=656
LOG_FREQ=100
HF_REPO_ID="jrosseruk/gpt-neo-28m-tinystories"
OUTPUT_DIR="./outputs"

# Get number of GPUs allocated
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "======================================================================"
echo "Training Configuration"
echo "======================================================================"
echo "Config: $CONFIG"
echo "Vocabulary: $VOCAB_MAPPING (10K tokens)"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Random seed: $SEED (deterministic)"
echo "Checkpoint frequency: $CHECKPOINT_FREQ updates (~1%)"
echo "Log frequency: $LOG_FREQ updates"
echo "HuggingFace repo: $HF_REPO_ID"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
fi
echo "======================================================================"
echo ""

# Build command
CMD="python train_gptneo.py \
    --config $CONFIG \
    --vocab_mapping $VOCAB_MAPPING \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --seed $SEED \
    --checkpoint_frequency $CHECKPOINT_FREQ \
    --log_frequency $LOG_FREQ \
    --hf_repo_id $HF_REPO_ID \
    --output_dir $OUTPUT_DIR \
    --use_wandb"

# Add resume checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ ! -f "$RESUME_CHECKPOINT" ]; then
        echo "❌ ERROR: Checkpoint file not found: $RESUME_CHECKPOINT"
        exit 1
    fi
    CMD="$CMD --resume_from $RESUME_CHECKPOINT"
fi

echo "🚀 Starting training..."
echo ""
echo "Command: $CMD"
echo ""
echo "======================================================================"
echo ""

# Run training
$CMD

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Training completed"
echo "======================================================================"
echo "Exit code: $EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Finished: $(date)"
echo "======================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "Checkpoints saved to: $OUTPUT_DIR"
    echo "HuggingFace repo: https://huggingface.co/$HF_REPO_ID"
    echo ""
else
    echo ""
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check logs at: logs/slurm-$SLURM_JOB_ID.err"
    echo ""
fi

exit $EXIT_CODE
