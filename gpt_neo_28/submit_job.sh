#!/bin/bash
# Quick job submission helper script

set -e

echo "======================================================================"
echo "GPT-Neo 28M Training - Job Submission Helper"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "train_gptneo.py" ]; then
    echo "❌ Error: Must run from gpt_neo_28 directory"
    echo "   cd /home/s5e/jrosser.s5e/infusion/gpt_neo_28"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please create .env file with your HuggingFace token:"
    echo "  1. cp .env.example .env"
    echo "  2. Edit .env and add your HF_TOKEN"
    echo "  3. Get token from: https://huggingface.co/settings/tokens"
    echo ""
    exit 1
fi

# Verify HF_TOKEN
source .env
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set in .env file"
    exit 1
fi
echo "✓ HF_TOKEN found in .env"

# Test authentication
echo "Testing HuggingFace authentication..."
python test_hf_setup.py > /tmp/hf_test.out 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: HuggingFace authentication failed"
    cat /tmp/hf_test.out
    exit 1
fi
echo "✓ HuggingFace authentication successful"
echo ""

# Check if sbatch script exists
if [ ! -f sbatch_train_gptneo.sh ]; then
    echo "❌ Error: sbatch_train_gptneo.sh not found"
    exit 1
fi

# Make it executable
chmod +x sbatch_train_gptneo.sh

# Create logs directory
mkdir -p logs

# Submit job
echo "======================================================================"
echo "Submitting SLURM job..."
echo "======================================================================"
echo ""

RESUME_ARG=""
if [ -n "$1" ]; then
    RESUME_ARG="$1"
    echo "Resuming from checkpoint: $RESUME_ARG"
    echo ""
fi

JOB_ID=$(sbatch $RESUME_ARG sbatch_train_gptneo.sh | awk '{print $NF}')

if [ -z "$JOB_ID" ]; then
    echo "❌ Error: Failed to submit job"
    exit 1
fi

echo "✅ Job submitted successfully!"
echo ""
echo "======================================================================"
echo "Job Information"
echo "======================================================================"
echo "Job ID: $JOB_ID"
echo "Expected duration: ~8 hours"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER                        # Check job status"
echo "  tail -f logs/slurm-${JOB_ID}.out       # Watch output"
echo "  tail -f logs/slurm-${JOB_ID}.err       # Watch errors"
echo ""
echo "Cancel with:"
echo "  scancel $JOB_ID"
echo ""
echo "View checkpoints:"
echo "  ls -lh outputs/checkpoint_*/"
echo ""
echo "HuggingFace repo:"
echo "  https://huggingface.co/jrosseruk/gpt-neo-28m-tinystories"
echo ""
echo "Weights & Biases:"
echo "  https://wandb.ai"
echo "======================================================================"
echo ""

# Show initial queue status
sleep 2
echo "Current queue status:"
squeue -u $USER
echo ""
