#!/bin/bash
# Run the full gradient atoms pipeline
#
# Step 1: Extract per-doc gradients (multi-GPU, ~30 min)
# Step 2: Dictionary learning (CPU, ~15 min)
# Step 3: Evaluate top atoms (GPU, ~10 min each)

set -e
PYTHON=/home/ubuntu/infusion/.venv/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /home/ubuntu/infusion

echo "=========================================="
echo "Step 1: Extract per-doc gradients (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 \
    experiments_gradient_atoms/extract_gradients.py \
    --n_docs 5000 \
    --output_dir experiments_gradient_atoms/results

echo ""
echo "=========================================="
echo "Step 2: Dictionary learning"
echo "=========================================="
$PYTHON experiments_gradient_atoms/learn_atoms.py \
    --gradients_path experiments_gradient_atoms/results/gradients_all.pt \
    --n_atoms 500 \
    --top_k_eigen 50 \
    --alpha 1.0 \
    --output_dir experiments_gradient_atoms/results

echo ""
echo "=========================================="
echo "Step 3: Evaluate top atoms"
echo "=========================================="
# Evaluate top 5 most coherent atoms
for i in $(seq 0 4); do
    ATOM_FILE=$(ls experiments_gradient_atoms/results/steering_vectors/atom_*.pt 2>/dev/null | head -n $((i+1)) | tail -1)
    if [ -n "$ATOM_FILE" ]; then
        echo "Evaluating: $ATOM_FILE"
        $PYTHON experiments_gradient_atoms/steer_atom.py \
            --atom_path "$ATOM_FILE" \
            --alpha 1e-4 \
            --output_dir experiments_gradient_atoms/results/eval
    fi
done

echo ""
echo "Done! Results in experiments_gradient_atoms/results/"
