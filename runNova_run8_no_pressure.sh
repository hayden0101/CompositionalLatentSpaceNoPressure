#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --job-name="cls-r8-nop"
#SBATCH --mail-user=haydenc1@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output="run8_no_pressure_%j.out"
#SBATCH --error="run8_no_pressure_%j.err"

set -euo pipefail

module load intel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
VENV="${VENV:-/work/mech-ai/haydenc1/packages/CompositionalLatentSpace/sciml}"
PYTHON="$VENV/bin/python"
CONFIG="configs/compositional/run8_no_pressure.yaml"
CHECKPOINT_ROOT="checkpoints/compositional-run8-no-pressure"
OUTDIR="docs/figures/run8_no_pressure"

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

# Keep the setup behavior from runNova.sh. Set INSTALL_DEPS=0 to skip it
# after the environment has already been prepared.
if [[ "${INSTALL_DEPS:-1}" == "1" ]]; then
  "$PYTHON" -m pip install --force-reinstall setuptools==70.3.0
  "$PYTHON" -m pip install -r venv_requirements.txt
fi

echo "Training Run 8 without pressure using $CONFIG"
"$PYTHON" main_no_pressure.py --config "$CONFIG"

CHECKPOINT="$("$PYTHON" scripts/latest_checkpoint.py "$CHECKPOINT_ROOT")"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint discovery returned a missing file: $CHECKPOINT" >&2
  exit 1
fi

echo "Running diagnostics from $CHECKPOINT"
"$PYTHON" diagnostics/probes_no_pressure.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT"

echo "Plotting velocity-only figures into $OUTDIR"
mkdir -p "$OUTDIR"
"$PYTHON" plotting/figures_no_pressure.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --outdir "$OUTDIR" \
  --n-recon 2

echo "Run 8 complete. Checkpoint: $CHECKPOINT"
echo "Figures: $OUTDIR"
