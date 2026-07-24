#!/usr/bin/env python3
"""Validate the Runs 10-29 one-factor-at-a-time sweep and plot scales."""

from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plotting.plot_scale import DEFAULT_ERROR_VMAX, configured_error_vmax, error_ticks  # noqa: E402

EXPECTED = {
    10: ("lambda_fullswap", 0.025),
    11: ("lambda_fullswap", 0.05),
    12: ("lambda_fullswap", 0.2),
    13: ("lambda_fullswap", 0.5),
    14: ("lambda_xswap", 0.05),
    15: ("lambda_xswap", 0.2),
    16: ("lambda_swap", 0.05),
    17: ("lambda_swap", 0.2),
    18: ("lambda_inv", 0.05),
    19: ("lambda_inv", 0.2),
    20: ("lambda_decorr", 0.0),
    21: ("lambda_decorr", 0.05),
    22: ("lambda_regime", 0.05),
    23: ("lambda_regime", 0.2),
    24: ("lambda_geo", 0.05),
    25: ("lambda_geo", 0.2),
    26: ("lambda_recon", 0.5),
    27: ("lambda_recon", 2.0),
    28: ("lambda_bl", 0.1),
    29: ("lambda_bl", 0.25),
}


def load_config(run: int):
    path = ROOT / f"configs/compositional/run{run}_no_pressure.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def check_one_factor_design():
    baseline = load_config(9)
    baseline_model = baseline["model"]

    for run, (changed_key, expected_value) in EXPECTED.items():
        config = load_config(run)
        model = config["model"]

        assert model["in_channels"] == 2
        assert model["static_geometry"] is True
        assert config["plotting"]["error_vmax"] == DEFAULT_ERROR_VMAX
        assert config["trainer"]["project"] == f"compositional-run{run}-no-pressure"
        assert config["callbacks"]["checkpoint"]["dirpath"].endswith(
            f"compositional-run{run}-no-pressure"
        )

        differences = {
            key: (baseline_model[key], model[key])
            for key in baseline_model
            if baseline_model[key] != model[key]
        }
        assert differences == {
            changed_key: (baseline_model[changed_key], expected_value)
        }, (run, differences)


def check_shell_scripts():
    for run in EXPECTED:
        path = ROOT / f"runNova_run{run}_no_pressure.sh"
        text = path.read_text()
        assert f"run{run}_no_pressure.yaml" in text
        assert f"compositional-run{run}-no-pressure" in text
        assert "main_full_swap_no_pressure.py" in text
        assert "full_swap_diagnostics_no_pressure.py" in text
        assert "full_swap_matrix_no_pressure.py" in text
        subprocess.run(["bash", "-n", str(path)], check=True)

    subprocess.run(["bash", "-n", str(ROOT / "submit_runs10_29.sh")], check=True)


def check_plot_scale():
    class Plotting:
        def get(self, key, default=None):
            return {"error_vmax": 0.05}.get(key, default)

    class Config:
        plotting = Plotting()

    assert configured_error_vmax(Config()) == 0.05
    assert configured_error_vmax(Config(), 0.08) == 0.08
    ticks = error_ticks(0.05)
    assert len(ticks) == 6
    assert ticks[0] == 0.0
    assert abs(ticks[-1] - 0.05) < 1e-12


def main():
    check_one_factor_design()
    check_shell_scripts()
    check_plot_scale()
    print("All Runs 10-29 one-factor sweep checks passed.")


if __name__ == "__main__":
    main()
