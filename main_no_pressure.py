#!/usr/bin/env python3
"""Run the upstream training entry point with a velocity-only dataset.

This wrapper deliberately delegates training to the repository's existing
``main.py``. The only substitution is the dataset class, so batching, losses,
logging, checkpointing, and command-line behavior stay aligned with upstream.
"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import data.dataset as dataset_module  # noqa: E402
from data.no_pressure_dataset import NoPressureLDCDataset  # noqa: E402

# ``main.py`` imports the class from this module. Replacing the module attribute
# before executing it keeps the upstream training path intact while removing p.
dataset_module.CompositionalLDCDataset = NoPressureLDCDataset

runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
