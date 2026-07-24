#!/usr/bin/env python3
"""Quantify all eight A/B latent-block combinations for full-swap runs.

The diagnostic samples complete 2 x 2 rectangles from the test split, evaluates
all assignments of ``[z_mu | z_g | z_xi]``, and compares each prediction with
its exact CFD target. Results are saved as a CSV beside the checkpoint unless an
explicit output path is supplied.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.no_pressure_dataset import NoPressureLDCDataset  # noqa: E402
from full_swap_evaluation import evaluate_rectangle  # noqa: E402
from full_swap_utils import SWAP_SPECS, enumerate_rectangles, select_rectangles  # noqa: E402
from models.compositional.full_swap_ae import FullSwapCompositionalAE  # noqa: E402


def build_test_dataset(config):
    train_dataset = NoPressureLDCDataset(
        file_path_x=config.data.file_path_train_x,
        file_path_y=config.data.file_path_train_y,
        resolution=config.data.resolution,
    )
    return NoPressureLDCDataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        resolution=config.data.resolution,
        re_stats=train_dataset.re_stats,
    )


def main(config_path, checkpoint_path, output_path, max_rectangles, seed):
    config = OmegaConf.load(config_path)
    if int(config.model.get("in_channels", 2)) != 2:
        raise ValueError("Full-swap diagnostics require u and v only.")

    dataset = build_test_dataset(config)
    rectangles = enumerate_rectangles(dataset.re.tolist(), dataset.geo_ids.tolist())
    rectangles = select_rectangles(rectangles, max_rectangles=max_rectangles, seed=seed)
    if not rectangles:
        raise RuntimeError(
            "No complete 2 x 2 (Re, geometry) rectangles exist in the test split."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSwapCompositionalAE.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.eval()

    mse = [[] for _ in SWAP_SPECS]
    recon_mse = [[] for _ in SWAP_SPECS]
    ratio = [[] for _ in SWAP_SPECS]
    u_mse = [[] for _ in SWAP_SPECS]
    v_mse = [[] for _ in SWAP_SPECS]

    for rectangle in rectangles:
        result = evaluate_rectangle(model, dataset, rectangle, device)
        for index, _ in enumerate(SWAP_SPECS):
            mse[index].append(float(result.mse[index]))
            recon_mse[index].append(float(result.reconstruction_mse[index]))
            ratio[index].append(float(result.ratio[index]))
            u_mse[index].append(float(result.per_channel_mse[index, 0]))
            v_mse[index].append(float(result.per_channel_mse[index, 1]))

    output = Path(output_path) if output_path else Path(checkpoint_path).parent / "full_swap_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, spec in enumerate(SWAP_SPECS):
        rows.append(
            {
                "code": spec.code,
                "target": spec.target,
                "description": spec.label,
                "n_rectangles": len(rectangles),
                "mean_mse": np.mean(mse[index]),
                "std_mse": np.std(mse[index]),
                "mean_target_reconstruction_mse": np.mean(recon_mse[index]),
                "mean_ratio_to_reconstruction": np.mean(ratio[index]),
                "std_ratio_to_reconstruction": np.std(ratio[index]),
                "mean_u_mse": np.mean(u_mse[index]),
                "mean_v_mse": np.mean(v_mse[index]),
            }
        )

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Full-swap diagnostic over {len(rectangles)} rectangles")
    print(f"{'code':<5} {'target':<7} {'MSE':>12} {'ratio/recon':>14}  description")
    print("-" * 86)
    for row in rows:
        print(
            f"{row['code']:<5} {row['target']:<7} "
            f"{row['mean_mse']:>12.5e} "
            f"{row['mean_ratio_to_reconstruction']:>14.3f}  "
            f"{row['description']}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-rectangles", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.checkpoint,
        arguments.output,
        arguments.max_rectangles,
        arguments.seed,
    )
