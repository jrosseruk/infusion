#!/bin/bash
# Run the full atom-guided IHVP steering experiment.
#
# Steps:
#   1. Extract IHVP for each behavior (runs sequentially to manage GPU memory)
#   2. Create steered adapters + evaluate via vLLM
#
# Usage:
#   bash experiments_atom_ihvp/run_experiment.sh
#   bash experiments_atom_ihvp/run_experiment.sh cat tea  # specific behaviors only

set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
BEHAVIORS="${@:-all}"

echo "═══════════════════════════════════════════════════════════════"
echo "Step 1: Extract IHVP for each behavior"
echo "═══════════════════════════════════════════════════════════════"

if [ "$BEHAVIORS" = "all" ]; then
    for behavior in tool_call creative_writing data_analysis python_code cat bullet_list concise tea; do
        echo ""
        echo "--- Extracting IHVP for: $behavior ---"
        $PYTHON experiments_atom_ihvp/extract_ihvp.py --behavior "$behavior"
    done
else
    for behavior in $BEHAVIORS; do
        echo ""
        echo "--- Extracting IHVP for: $behavior ---"
        $PYTHON experiments_atom_ihvp/extract_ihvp.py --behavior "$behavior"
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 2: Steer and evaluate"
echo "═══════════════════════════════════════════════════════════════"

if [ "$BEHAVIORS" = "all" ]; then
    $PYTHON experiments_atom_ihvp/steer_and_eval.py --behavior all
else
    for behavior in $BEHAVIORS; do
        $PYTHON experiments_atom_ihvp/steer_and_eval.py --behavior "$behavior"
    done
fi

echo ""
echo "Done! Results in experiments_atom_ihvp/results/"
