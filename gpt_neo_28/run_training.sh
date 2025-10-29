#!/bin/bash
# Quick start script for training GPT-Neo 28M on TinyStories

set -e

echo "======================================================================"
echo "GPT-Neo 28M Training on TinyStories"
echo "======================================================================"
echo ""

# Check if vocab mapping exists
if [ ! -f "vocab_mapping.json" ]; then
    echo "Vocabulary mapping not found. Building vocabulary..."
    python build_vocab.py --vocab_size 10000 --output vocab_mapping.json
    echo ""
fi

# Get HuggingFace repo ID from user
read -p "Enter your HuggingFace repo ID (e.g., username/gpt-neo-28m-tinystories): " HF_REPO_ID

if [ -z "$HF_REPO_ID" ]; then
    echo "No HuggingFace repo specified. Checkpoints will only be saved locally."
    HF_ARG=""
else
    HF_ARG="--hf_repo_id $HF_REPO_ID"
fi

# Ask about Weights & Biases
read -p "Enable Weights & Biases logging? (y/N): " USE_WANDB
if [[ "$USE_WANDB" =~ ^[Yy]$ ]]; then
    WANDB_ARG="--use_wandb"
else
    WANDB_ARG=""
fi

echo ""
echo "Starting training with:"
echo "  - Batch size: 32"
echo "  - Learning rate: 1e-3"
echo "  - Epochs: 1"
echo "  - Seed: 3407 (deterministic)"
echo "  - Checkpoint frequency: 656 updates (~1%)"
if [ ! -z "$HF_REPO_ID" ]; then
    echo "  - HuggingFace repo: $HF_REPO_ID"
fi
echo ""
echo "======================================================================"
echo ""

# Run training
python train_gptneo.py \
    --config config-28M-gptneo.json \
    --vocab_mapping vocab_mapping.json \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --epochs 1 \
    --seed 3407 \
    --checkpoint_frequency 656 \
    --log_frequency 100 \
    $HF_ARG \
    --output_dir ./outputs \
    $WANDB_ARG

echo ""
echo "======================================================================"
echo "Training complete!"
echo "Checkpoints saved to: ./outputs/"
if [ ! -z "$HF_REPO_ID" ]; then
    echo "Checkpoints uploaded to: https://huggingface.co/$HF_REPO_ID"
fi
echo "======================================================================"
