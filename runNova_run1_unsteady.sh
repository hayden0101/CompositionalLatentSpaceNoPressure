#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name=unsteady_run1
#SBATCH --output=unsteady_run1_%j.out
#SBATCH --error=unsteady_run1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=haydenc1@iastate.edu

set -euo pipefail

module purge
module load intel

PROJECT_DIR=/work/mech-ai/haydenc1/packages/CompositionalLatentSpaceUnsteadyState
VENV=/work/mech-ai/haydenc1/packages/CompositionalLatentSpace/sciml
PYTHON="$VENV/bin/python"

cd "$PROJECT_DIR"

TRAIN_INPUT=/work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_train_x.npz
TEST_INPUT=/work/mech-ai/arabeh/data_prep_ns/LDC_NS_2D/processed/easy/all_ldc_test_x.npz
TRAIN_OUTPUT=/work/mech-ai/haydenc1/data/unsteady_ldc_train.npz
TEST_OUTPUT=/work/mech-ai/haydenc1/data/unsteady_ldc_test.npz
CONFIG=configs/compositional/unsteady_run1_velocity.yaml

mkdir -p /work/mech-ai/haydenc1/data
MODE=${1:-all}

generate_train() {
  "$PYTHON" data/generate_unsteady_ldc.py \
    --input-x "$TRAIN_INPUT" \
    --output "$TRAIN_OUTPUT" \
    --resolution 128 \
    --steps 1000 \
    --save-every 20
}

generate_test() {
  "$PYTHON" data/generate_unsteady_ldc.py \
    --input-x "$TEST_INPUT" \
    --output "$TEST_OUTPUT" \
    --resolution 128 \
    --steps 1000 \
    --save-every 20
}

train_model() {
  [[ -f "$TRAIN_OUTPUT" ]] || { echo "Missing $TRAIN_OUTPUT"; exit 1; }
  [[ -f "$TEST_OUTPUT" ]] || { echo "Missing $TEST_OUTPUT"; exit 1; }
  "$PYTHON" main.py --config "$CONFIG"
  find checkpoints/compositional_unsteady_run1_velocity -type f -name '*.ckpt' -print 2>/dev/null || true
}

case "$MODE" in
  generate) generate_train; generate_test ;;
  train) train_model ;;
  all)
    [[ -f "$TRAIN_OUTPUT" ]] || generate_train
    [[ -f "$TEST_OUTPUT" ]] || generate_test
    train_model
    ;;
  *) echo "Use: generate, train, or all"; exit 2 ;;
esac
