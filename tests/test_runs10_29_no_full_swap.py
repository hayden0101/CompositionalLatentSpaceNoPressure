#!/usr/bin/env python3
"""Static checks for the standard Runs 10-29 sweep."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    "lambda_recon": 1.0,
    "lambda_bl": 0.0,
    "lambda_regime": 0.1,
    "lambda_geo": 0.1,
    "lambda_decorr": 0.01,
    "lambda_inv": 0.1,
    "lambda_swap": 0.1,
    "lambda_xswap": 0.1,
}
CHANGES = {
    10: ("lambda_xswap", 0.025), 11: ("lambda_xswap", 0.05),
    12: ("lambda_xswap", 0.2), 13: ("lambda_xswap", 0.5),
    14: ("lambda_swap", 0.025), 15: ("lambda_swap", 0.05),
    16: ("lambda_swap", 0.2), 17: ("lambda_inv", 0.025),
    18: ("lambda_inv", 0.05), 19: ("lambda_inv", 0.2),
    20: ("lambda_decorr", 0.0), 21: ("lambda_decorr", 0.05),
    22: ("lambda_decorr", 0.1), 23: ("lambda_regime", 0.05),
    24: ("lambda_regime", 0.2), 25: ("lambda_geo", 0.05),
    26: ("lambda_geo", 0.2), 27: ("lambda_recon", 0.5),
    28: ("lambda_recon", 2.0), 29: ("lambda_bl", 0.1),
}

for run, (changed_key, expected_value) in CHANGES.items():
    config_path = ROOT / "configs" / "compositional" / f"run{run}_no_pressure.yaml"
    config = yaml.safe_load(config_path.read_text())
    model = config["model"]
    assert model["in_channels"] == 2
    assert model["static_geometry"] is True
    assert "lambda_fullswap" not in model
    assert config["plotting"]["error_vmax"] == 0.05
    for key, baseline_value in BASELINE.items():
        expected = expected_value if key == changed_key else baseline_value
        assert model[key] == expected, (run, key, model[key], expected)

    script = ROOT / f"runNova_run{run}_no_pressure.sh"
    contents = script.read_text()
    assert "main_no_pressure.py" in contents
    assert "diagnostics/probes_no_pressure.py" in contents
    assert "plotting/figures_no_pressure.py" in contents
    assert "full_swap" not in contents.lower()
    subprocess.run(["bash", "-n", str(script)], check=True)

print("All Runs 10-29 no-full-swap checks passed.")
