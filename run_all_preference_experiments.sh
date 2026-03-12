#!/bin/bash
# Run all preference experiments: wait for iterative UK, then spring, then cycling.
set -e

PYTHON="/home/ubuntu/infusion/.venv/bin/python"
cd /home/ubuntu/infusion

echo "============================================================"
echo "Waiting for iterative UK loop (PID 1253022) to finish..."
echo "============================================================"
while kill -0 1253022 2>/dev/null; do
    sleep 60
done
echo "Iterative UK loop finished."

# Kill any leftover GPU processes
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
done
sleep 10

echo ""
echo "============================================================"
echo "SPRING PREFERENCE EXPERIMENT"
echo "============================================================"
$PYTHON experiments_infusion_spring/run_experiment.py \
    --alphas 1e-5 3e-5 5e-5 7e-5 1e-4

echo ""
echo "============================================================"
echo "CYCLING PREFERENCE EXPERIMENT"
echo "============================================================"
$PYTHON experiments_infusion_cycling/run_experiment.py \
    --alphas 1e-5 3e-5 5e-5 7e-5 1e-4

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
