#!/usr/bin/env python3
"""Standalone smoke checks for the velocity-only experiment overlay."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.no_pressure_dataset import NoPressureLDCDataset  # noqa: E402


EXPECTED = {
    1: dict(decorr=0.01, inv=0.0, swap=0.0, xswap=0.0, bl=0.0, static=False),
    2: dict(decorr=0.5, inv=0.0, swap=0.0, xswap=0.0, bl=0.0, static=False),
    3: dict(decorr=0.1, inv=0.0, swap=0.0, xswap=0.0, bl=0.0, static=False),
    4: dict(decorr=0.01, inv=0.1, swap=0.0, xswap=0.0, bl=0.0, static=False),
    5: dict(decorr=0.01, inv=0.1, swap=0.1, xswap=0.0, bl=0.0, static=False),
    6: dict(decorr=0.01, inv=0.1, swap=0.1, xswap=0.1, bl=0.0, static=False),
    7: dict(decorr=0.01, inv=0.1, swap=0.1, xswap=0.1, bl=0.0, static=True),
    8: dict(decorr=0.01, inv=0.1, swap=0.1, xswap=0.1, bl=1.0, static=True),
}


def check_configs_and_shells():
    for run, expected in EXPECTED.items():
        config_path = ROOT / "configs" / "compositional" / f"run{run}_no_pressure.yaml"
        config = yaml.safe_load(config_path.read_text())
        model = config["model"]

        assert model["in_channels"] == 2
        assert model["lambda_recon"] == 1.0
        assert model["lambda_regime"] == 0.1
        assert model["lambda_geo"] == 0.1
        assert model["lambda_decorr"] == expected["decorr"]
        assert model["lambda_inv"] == expected["inv"]
        assert model["lambda_swap"] == expected["swap"]
        assert model["lambda_xswap"] == expected["xswap"]
        assert model["lambda_bl"] == expected["bl"]
        assert model["static_geometry"] is expected["static"]

        shell_path = ROOT / f"runNova_run{run}_no_pressure.sh"
        shell = shell_path.read_text()
        assert f"run{run}_no_pressure.yaml" in shell
        assert "main_no_pressure.py" in shell
        assert "diagnostics/probes_no_pressure.py" in shell
        assert "plotting/figures_no_pressure.py" in shell
        assert "--checkpoint" in shell
        assert "--n-recon 2" in shell


def check_dataset_drops_pressure():
    height = width = 8
    sample_count = 2

    x = np.zeros((sample_count, 3, height, width), dtype=np.float32)
    x[0, 0] = 10.0
    x[1, 0] = 100.0
    x[:, 1] = 1.0
    x[:, 2] = 1.0
    x[:, 2, 3:5, 3:5] = 0.0

    y = np.zeros((sample_count, 5, height, width), dtype=np.float32)
    y[:, 0] = 1.25
    y[:, 1] = -2.5
    y[:, 2] = 999.0

    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        x_path = directory / "x.npz"
        y_path = directory / "y.npz"
        np.savez(x_path, data=x)
        np.savez(y_path, data=y)

        dataset = NoPressureLDCDataset(x_path, y_path, resolution=height)

    assert tuple(dataset.fields.shape) == (sample_count, 2, height, width)
    assert dataset.fields.is_contiguous()
    assert torch.allclose(dataset.fields[:, 0], torch.full_like(dataset.fields[:, 0], 1.25))
    assert torch.allclose(dataset.fields[:, 1], torch.full_like(dataset.fields[:, 1], -2.5))
    assert not torch.any(dataset.fields == 999.0)
    assert dataset.channel_names == ("u", "v")


def main():
    check_configs_and_shells()
    check_dataset_drops_pressure()
    print("All no-pressure setup checks passed.")


if __name__ == "__main__":
    main()
