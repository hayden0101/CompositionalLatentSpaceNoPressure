#!/usr/bin/env python3
"""Run the upstream probe and swap diagnostics on velocity-only fields."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data.dataset as dataset_module  # noqa: E402
from data.no_pressure_dataset import NoPressureLDCDataset  # noqa: E402

# The original diagnostic implementation is reused verbatim. Its reconstruction
# and swap errors are evaluated on u and v because this class supplies two fields.
dataset_module.CompositionalLDCDataset = NoPressureLDCDataset

runpy.run_path(str(ROOT / "diagnostics" / "probes.py"), run_name="__main__")
