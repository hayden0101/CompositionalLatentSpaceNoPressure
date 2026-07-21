#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name=unsteady_ldc
#SBATCH --output=unsteady_ldc_%j.out
#SBATCH --error=unsteady_ldc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# Replace this with your real Iowa State email before submitting:
#SBATCH --mail-user=haydenc1@iastate.edu

set -euo pipefail

module purge
module load intel

# -----------------------------------------------------------------------------
# Project and Python environment
# -----------------------------------------------------------------------------
PROJECT_DIR=/work/mech-ai/haydenc1/packages/CompositionalLatentSpaceUnsteadyState
VENV=/work/mech-ai/haydenc1/packages/CompositionalLatentSpace/sciml
PYTHON="$VENV/bin/python"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found at $PYTHON"
    exit 1
fi

if [[ ! -f main.py ]]; then
    echo "ERROR: main.py was not found in $PROJECT_DIR"
    exit 1
fi

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
TRAIN_INPUT=/work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_train_x.npz
TEST_INPUT=/work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_test_x.npz
TRAIN_OUTPUT=/work/mech-ai/haydenc1/data/unsteady_ldc_train.npz
TEST_OUTPUT=/work/mech-ai/haydenc1/data/unsteady_ldc_test.npz
CONFIG=configs/compositional/unsteady_velocity.yaml

mkdir -p /work/mech-ai/haydenc1/data

# Usage:
#   sbatch runNova.sh generate   # generate both transient datasets only
#   sbatch runNova.sh train      # train only, using existing generated files
#   sbatch runNova.sh all        # generate missing files, then train
# With no argument, the default is "train".
MODE=${1:-train}

# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------
generate_train() {
    echo "Generating unsteady training data..."
    "$PYTHON" data/generate_unsteady_ldc.py \
        --input-x "$TRAIN_INPUT" \
        --output "$TRAIN_OUTPUT" \
        --resolution 128 \
        --steps 1000 \
        --save-every 20
}

generate_test() {
    echo "Generating unsteady test data..."
    "$PYTHON" data/generate_unsteady_ldc.py \
        --input-x "$TEST_INPUT" \
        --output "$TEST_OUTPUT" \
        --resolution 128 \
        --steps 1000 \
        --save-every 20
}

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
train_model() {
    if [[ ! -f "$TRAIN_OUTPUT" || ! -f "$TEST_OUTPUT" ]]; then
        echo "ERROR: Generated unsteady datasets were not found."
        echo "Run: sbatch runNova.sh generate"
        exit 1
    fi

    echo "Training the unsteady, velocity-only compositional autoencoder..."
    "$PYTHON" main.py --config "$CONFIG"

    echo "Training finished. Available checkpoints:"
    find checkpoints/compositional_unsteady_velocity -type f -name '*.ckpt' -print 2>/dev/null || true
}

case "$MODE" in
    generate)
        generate_train
        generate_test
        ;;
    train)
        train_model
        ;;
    all)
        [[ -f "$TRAIN_OUTPUT" ]] || generate_train
        [[ -f "$TEST_OUTPUT" ]] || generate_test
        train_model
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use one of: generate, train, all"
        exit 2
        ;;
esac
