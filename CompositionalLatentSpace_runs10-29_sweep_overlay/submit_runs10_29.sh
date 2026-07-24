#!/bin/bash
set -euo pipefail

START_RUN="${1:-10}"
END_RUN="${2:-29}"

if (( START_RUN < 10 || END_RUN > 29 || START_RUN > END_RUN )); then
  echo "Usage: $0 [start_run 10-29] [end_run 10-29]" >&2
  exit 2
fi

for run in $(seq "$START_RUN" "$END_RUN"); do
  script="runNova_run${run}_no_pressure.sh"
  if [[ ! -f "$script" ]]; then
    echo "Missing $script" >&2
    exit 1
  fi
  sbatch --export=ALL,INSTALL_DEPS=0 "$script"
done
