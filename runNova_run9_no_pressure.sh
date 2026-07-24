#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --job-name="cls-r9-full"
#SBATCH --mail-user=haydenc1@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output="run9_no_pressure_%j.out"
#SBATCH --error="run9_no_pressure_%j.err"

set -euo pipefail

module load intel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
VENV="${VENV:-/work/mech-ai/haydenc1/packages/CompositionalLatentSpace/sciml}"
PYTHON="$VENV/bin/python"
CONFIG="configs/compositional/run9_no_pressure.yaml"
CHECKPOINT_ROOT="checkpoints/compositional-run9-no-pressure"
OUTDIR="docs/figures/run9_no_pressure"

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

if [[ "${INSTALL_DEPS:-1}" == "1" ]]; then
  "$PYTHON" -m pip install --force-reinstall setuptools==70.3.0
  "$PYTHON" -m pip install -r venv_requirements.txt
fi

echo "Training Run 9 full-swap model without pressure using $CONFIG"
"$PYTHON" main_full_swap_no_pressure.py --config "$CONFIG"

CHECKPOINT="$("$PYTHON" scripts/latest_checkpoint.py "$CHECKPOINT_ROOT")"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint discovery returned a missing file: $CHECKPOINT" >&2
  exit 1
fi

echo "Running standard probes and swap diagnostics from $CHECKPOINT"
"$PYTHON" diagnostics/probes_full_swap_no_pressure.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT"

echo "Running all eight latent-combination diagnostics"
"$PYTHON" diagnostics/full_swap_diagnostics_no_pressure.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --max-rectangles 64

echo "Plotting standard velocity-only figures into $OUTDIR"
mkdir -p "$OUTDIR"
"$PYTHON" plotting/figures_full_swap_no_pressure.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --outdir "$OUTDIR" \
  --n-recon 2

echo "Plotting the complete full-swap matrix"
"$PYTHON" plotting/full_swap_matrix_no_pressure.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --outdir "$OUTDIR"

echo "Run 9 complete. Checkpoint: $CHECKPOINT"
echo "Full-swap metrics: $(dirname "$CHECKPOINT")/full_swap_metrics.csv"
echo "Figures: $OUTDIR"
