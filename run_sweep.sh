#!/bin/bash
# Quick start script for running W&B sweeps

set -e  # Exit on error

echo "=================================="
echo "INFUSION HYPERPARAMETER SWEEP"
echo "=================================="
echo ""

# Check if logged in
if ! wandb verify &> /dev/null; then
    echo "🔐 Please login to W&B:"
    wandb login
fi

# Parse arguments
ACTION=${1:-help}

case $ACTION in
    create)
        echo "📋 Creating new sweep..."
        wandb sweep sweep_config.yaml
        echo ""
        echo "✅ Sweep created!"
        echo ""
        echo "📋 COPY THE FULL SWEEP ID from the 'wandb agent' line above."
        echo "   It looks like: entity/project/abc123xyz"
        echo ""
        echo "Then run:"
        echo "   ./run_sweep.sh parallel <full-sweep-id> 5"
        echo ""
        echo "Example:"
        echo "   ./run_sweep.sh parallel jrosseruk/infusion-mnist/6pp4lnmd 5"
        ;;

    agent)
        if [ -z "$2" ]; then
            echo "❌ Please provide sweep ID"
            echo "   Usage: ./run_sweep.sh agent <sweep-id>"
            exit 1
        fi

        SWEEP_ID=$2
        COUNT=${3:-1}  # Default to 1 run if not specified

        # Add entity/project if not provided
        if [[ ! "$SWEEP_ID" =~ / ]]; then
            # Get entity from W&B settings
            if [ -f ~/.netrc ]; then
                ENTITY=$(grep "login" ~/.netrc | head -1 | awk '{print $2}')
            fi

            # Fallback: try to get from W&B config
            if [ -z "$ENTITY" ] && [ -f ~/.config/wandb/settings ]; then
                ENTITY=$(grep "entity" ~/.config/wandb/settings | cut -d'=' -f2 | tr -d ' ')
            fi

            # If still empty, ask user
            if [ -z "$ENTITY" ]; then
                echo "❌ Could not detect W&B entity."
                echo "Please provide the full sweep ID: entity/project/sweep-id"
                echo "Or check the sweep URL from 'wandb sweep' output"
                exit 1
            fi

            PROJECT="infusion-mnist"
            SWEEP_ID="${ENTITY}/${PROJECT}/${SWEEP_ID}"
            echo "ℹ️  Using full sweep path: $SWEEP_ID"
            echo ""
        fi

        echo "🚀 Starting sweep agent..."
        echo "   Sweep ID: $SWEEP_ID"
        echo "   Runs: $COUNT"
        echo ""

        if [ "$COUNT" = "inf" ]; then
            wandb agent $SWEEP_ID
        else
            wandb agent $SWEEP_ID --count $COUNT
        fi
        ;;

    parallel)
        if [ -z "$2" ]; then
            echo "❌ Please provide sweep ID"
            echo "   Usage: ./run_sweep.sh parallel <sweep-id> [num-agents]"
            exit 1
        fi

        SWEEP_ID=$2
        NUM_AGENTS=${3:-3}  # Default to 3 agents

        # Add entity/project if not provided
        if [[ ! "$SWEEP_ID" =~ / ]]; then
            # Get entity from W&B settings
            if [ -f ~/.netrc ]; then
                ENTITY=$(grep "login" ~/.netrc | head -1 | awk '{print $2}')
            fi

            # Fallback: try to get from W&B config
            if [ -z "$ENTITY" ] && [ -f ~/.config/wandb/settings ]; then
                ENTITY=$(grep "entity" ~/.config/wandb/settings | cut -d'=' -f2 | tr -d ' ')
            fi

            # If still empty, ask user
            if [ -z "$ENTITY" ]; then
                echo "❌ Could not detect W&B entity."
                echo "Please provide the full sweep ID: entity/project/sweep-id"
                echo "Or check the sweep URL from 'wandb sweep' output"
                exit 1
            fi

            PROJECT="infusion-mnist"
            SWEEP_ID="${ENTITY}/${PROJECT}/${SWEEP_ID}"
            echo "ℹ️  Using full sweep path: $SWEEP_ID"
            echo ""
        fi

        echo "🚀 Starting $NUM_AGENTS parallel agents..."
        echo "   Sweep ID: $SWEEP_ID"
        echo ""

        for i in $(seq 1 $NUM_AGENTS); do
            echo "   Starting agent $i..."
            wandb agent $SWEEP_ID &
        done

        echo ""
        echo "✅ $NUM_AGENTS agents running in background"
        echo "   Monitor at: https://wandb.ai"
        echo "   Stop with: killall wandb"
        wait
        ;;

    test)
        echo "🧪 Testing training script..."
        python train_sweep.py
        echo ""
        echo "✅ Test completed successfully!"
        ;;

    analyze)
        echo "📊 Opening analysis notebook..."
        jupyter notebook analyze_sweep.ipynb
        ;;

    status)
        if [ -z "$2" ]; then
            echo "❌ Please provide sweep ID"
            echo "   Usage: ./run_sweep.sh status <sweep-id>"
            exit 1
        fi

        SWEEP_ID=$2
        echo "📊 Fetching sweep status..."
        wandb sweep $SWEEP_ID
        ;;

    help|*)
        echo "Usage: ./run_sweep.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  create              Create a new sweep from sweep_config.yaml"
        echo "  agent <id> [count]  Run a single agent (optional: number of runs)"
        echo "  parallel <id> [n]   Run n parallel agents (default: 3)"
        echo "  test                Test the training script"
        echo "  analyze             Open analysis notebook"
        echo "  status <id>         Check sweep status"
        echo "  help                Show this help message"
        echo ""
        echo "Sweep ID formats (both work):"
        echo "  Short:  abc123           (auto-adds entity/project)"
        echo "  Full:   user/project/abc123"
        echo ""
        echo "Examples:"
        echo "  ./run_sweep.sh create"
        echo "  ./run_sweep.sh agent zk22b05x 10         # Short format"
        echo "  ./run_sweep.sh parallel zk22b05x 5       # Short format"
        echo "  ./run_sweep.sh agent jrosseruk/infusion-mnist/zk22b05x 10  # Full format"
        echo "  ./run_sweep.sh test"
        echo "  ./run_sweep.sh analyze"
        echo ""
        echo "Quick Start:"
        echo "  1. ./run_sweep.sh create"
        echo "  2. Copy the sweep ID (e.g., zk22b05x)"
        echo "  3. ./run_sweep.sh parallel zk22b05x 5"
        echo "  4. Monitor at https://wandb.ai"
        echo "  5. ./run_sweep.sh analyze"
        ;;
esac
