#!/usr/bin/env python3
"""Train Run 9 with velocity-only data and the full-swap model."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import data.dataset as dataset_module  # noqa: E402
import models.compositional.compositional_ae as model_module  # noqa: E402
from data.no_pressure_dataset import NoPressureLDCDataset  # noqa: E402
from models.compositional.full_swap_ae import FullSwapCompositionalAE  # noqa: E402

# Patch the symbols imported by upstream main.py before executing it.
dataset_module.CompositionalLDCDataset = NoPressureLDCDataset
model_module.CompositionalAE = FullSwapCompositionalAE

runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
