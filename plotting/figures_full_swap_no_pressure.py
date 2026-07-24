#!/usr/bin/env python3
"""Generate the standard velocity-only reconstruction and transfer figures for Run 9."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models.compositional.compositional_ae as model_module  # noqa: E402
from models.compositional.full_swap_ae import FullSwapCompositionalAE  # noqa: E402

model_module.CompositionalAE = FullSwapCompositionalAE
runpy.run_path(str(ROOT / "plotting" / "figures_no_pressure.py"), run_name="__main__")
