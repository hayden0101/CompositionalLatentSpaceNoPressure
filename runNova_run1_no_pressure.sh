#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --job-name="run1-nopressure"
#SBATCH --mail-user=haydenc1@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="run1_nopressure_%j.out"
#SBATCH --error="run1_nopressure_%j.err"

set -euo pipefail

module load intel

# Reuse the environment that already runs the project.
VENV=/work/mech-ai/haydenc1/packages/CompositionalLatentSpace/sciml

# Run from the directory containing this shell file.
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

"$VENV/bin/python" main.py \
  --config configs/compositional/run1_no_pressure.yaml

# Run probes on the newest checkpoint produced by this experiment.
LATEST_CKPT=$(find checkpoints/compositional_no_pressure_run1 \
  -type f -name 'last.ckpt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)

if [[ -n "${LATEST_CKPT:-}" ]]; then
  echo "Running probes with: $LATEST_CKPT"
  "$VENV/bin/python" diagnostics/probes.py --checkpoint "$LATEST_CKPT"
else
  echo "Training finished, but no last.ckpt was found." >&2
  exit 1
fi
