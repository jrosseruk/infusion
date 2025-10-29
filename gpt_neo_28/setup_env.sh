#!/bin/bash
# Quick setup script for HuggingFace authentication

set -e

echo "======================================================================"
echo "HuggingFace Setup for GPT-Neo 28M Training"
echo "======================================================================"
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file."
        exit 0
    fi
fi

# Copy example file
if [ ! -f ".env.example" ]; then
    echo "✗ Error: .env.example not found!"
    exit 1
fi

cp .env.example .env
echo "✓ Created .env file from template"
echo ""

# Prompt for token
echo "Please enter your HuggingFace API token:"
echo "(Get it from: https://huggingface.co/settings/tokens)"
echo ""
read -p "Token (starts with 'hf_'): " hf_token

# Validate token format
if [[ ! "$hf_token" =~ ^hf_ ]]; then
    echo ""
    echo "⚠️  Warning: Token doesn't start with 'hf_'"
    echo "Make sure you copied the entire token!"
    read -p "Continue anyway? (y/N): " continue
    if [[ ! "$continue" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        rm .env
        exit 1
    fi
fi

# Update .env file
echo "HF_TOKEN=$hf_token" > .env

echo ""
echo "======================================================================"
echo "✓ Setup complete!"
echo "======================================================================"
echo ""
echo "Testing authentication..."
python test_hf_setup.py

echo ""
echo "======================================================================"
echo "Next steps:"
echo "1. Build vocabulary: python build_vocab.py"
echo "2. Start training: python train_gptneo.py --hf_repo_id YOUR_USERNAME/gpt-neo-28m-tinystories"
echo "======================================================================"
