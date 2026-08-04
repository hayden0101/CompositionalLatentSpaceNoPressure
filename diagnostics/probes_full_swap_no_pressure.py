#!/usr/bin/env python3
"""Run the upstream probes and standard swap metrics for Run 9."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data.dataset as dataset_module  # noqa: E402
import models.compositional.compositional_ae as model_module  # noqa: E402
from data.no_pressure_dataset import NoPressureLDCDataset  # noqa: E402
from models.compositional.full_swap_ae import FullSwapCompositionalAE  # noqa: E402

dataset_module.CompositionalLDCDataset = NoPressureLDCDataset
model_module.CompositionalAE = FullSwapCompositionalAE

runpy.run_path(str(ROOT / "diagnostics" / "probes.py"), run_name="__main__")
