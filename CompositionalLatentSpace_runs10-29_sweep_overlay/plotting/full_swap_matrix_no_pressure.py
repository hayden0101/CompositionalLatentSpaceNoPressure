#!/usr/bin/env python3
"""Plot the complete latent-swap matrix for velocity-only full-swap runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.no_pressure_dataset import NoPressureLDCDataset  # noqa: E402
from full_swap_evaluation import evaluate_rectangle  # noqa: E402
from full_swap_utils import SWAP_SPECS, largest_jump_rectangle  # noqa: E402
from models.compositional.full_swap_ae import FullSwapCompositionalAE  # noqa: E402
from plotting.plot_scale import configured_error_vmax, error_ticks  # noqa: E402

CHANNELS = ("u", "v")
FIELD_CMAP = plt.get_cmap("RdBu_r").copy()
ERROR_CMAP = plt.get_cmap("magma").copy()
FIELD_CMAP.set_bad("0.35")
ERROR_CMAP.set_bad("0.35")


def masked(field, mask):
    output = np.array(field, dtype=float)
    output[np.asarray(mask) < 0.5] = np.nan
    return output


def robust_vmax(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(finite), 99.0)), np.finfo(float).eps)


def save_figure(fig, outdir, name):
    for extension in ("png", "pdf"):
        path = os.path.join(outdir, f"{name}.{extension}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(outdir, name)}.png / .pdf")


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


def plot_reference_grid(result, outdir):
    rectangle = result.rectangle
    target_labels = ("A", "B", "C", "D")
    target_indices = (rectangle.a, rectangle.b, rectangle.c, rectangle.d)
    # The evaluation stores targets in swap order, so read references directly
    # from the selected dataset indices in the caller instead.
    return target_labels, target_indices


def reference_figure(dataset, rectangle, outdir):
    labels = ("A", "B", "C", "D")
    indices = rectangle.indices
    fields = dataset.fields[list(indices)].numpy()
    masks = dataset.mask[list(indices), 0].numpy()
    reynolds = (rectangle.re_a, rectangle.re_b, rectangle.re_a, rectangle.re_b)
    geometries = (
        rectangle.geometry_a,
        rectangle.geometry_b,
        rectangle.geometry_b,
        rectangle.geometry_a,
    )

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.3), squeeze=False)
    for channel_index, channel in enumerate(CHANNELS):
        vmax = robust_vmax(fields[:, channel_index])
        image = None
        for column, label in enumerate(labels):
            image = axes[channel_index, column].imshow(
                masked(fields[column, channel_index], masks[column]),
                cmap=FIELD_CMAP,
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
            )
            axes[channel_index, column].set_xticks([])
            axes[channel_index, column].set_yticks([])
            axes[channel_index, column].set_title(
                f"{label}: Re={reynolds[column]:.0f}, g={geometries[column]}",
                fontsize=9,
            )
        axes[channel_index, 0].set_ylabel(
            channel, rotation=0, labelpad=14, fontsize=13, va="center"
        )
        fig.colorbar(image, ax=axes[channel_index, :].tolist(), shrink=0.82, pad=0.02)

    fig.suptitle("Full-swap reference CFD rectangle", y=0.99)
    save_figure(fig, outdir, "full_swap_reference_grid")


def swap_matrix_figure(result, outdir, errors=False, error_vmax=0.05):
    predictions = result.predictions.numpy()
    targets = result.targets.numpy()
    masks = result.masks[:, 0].numpy()

    fig, axes = plt.subplots(2, len(SWAP_SPECS), figsize=(24, 6.2), squeeze=False)
    for channel_index, channel in enumerate(CHANNELS):
        if errors:
            values = np.abs(predictions[:, channel_index] - targets[:, channel_index])
            vmax = error_vmax
            cmap = ERROR_CMAP
            vmin = 0.0
        else:
            values = predictions[:, channel_index]
            vmax = robust_vmax(targets[:, channel_index])
            cmap = FIELD_CMAP
            vmin = -vmax

        image = None
        for column, spec in enumerate(SWAP_SPECS):
            image = axes[channel_index, column].imshow(
                masked(values[column], masks[column]),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            axes[channel_index, column].set_xticks([])
            axes[channel_index, column].set_yticks([])
            axes[channel_index, column].set_title(
                f"{spec.code} -> {spec.target}\nratio={result.ratio[column]:.2f}",
                fontsize=8,
            )
        axes[channel_index, 0].set_ylabel(
            channel, rotation=0, labelpad=14, fontsize=13, va="center"
        )
        colorbar = fig.colorbar(
            image,
            ax=axes[channel_index, :].tolist(),
            shrink=0.82,
            pad=0.01,
            extend="max" if errors else "neither",
        )
        if errors:
            colorbar.set_ticks(error_ticks(error_vmax))

    rectangle = result.rectangle
    kind = "absolute errors" if errors else "predictions"
    fig.suptitle(
        f"Full latent-swap {kind}: Re {rectangle.re_a:.0f} <-> "
        f"{rectangle.re_b:.0f}, geometries {rectangle.geometry_a} and "
        f"{rectangle.geometry_b}",
        y=0.99,
    )
    save_figure(
        fig,
        outdir,
        "full_swap_errors" if errors else "full_swap_predictions",
    )


def main(config_path, checkpoint_path, outdir, error_vmax_override=None):
    config = OmegaConf.load(config_path)
    error_vmax = configured_error_vmax(config, error_vmax_override)
    if int(config.model.get("in_channels", 2)) != 2:
        raise ValueError("Full-swap figures require exactly two channels.")

    dataset = build_test_dataset(config)
    rectangle = largest_jump_rectangle(dataset.re.tolist(), dataset.geo_ids.tolist())
    if rectangle is None:
        raise RuntimeError(
            "No complete 2 x 2 (Re, geometry) rectangle exists in the test split."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSwapCompositionalAE.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    result = evaluate_rectangle(model, dataset, rectangle, device)

    os.makedirs(outdir, exist_ok=True)
    reference_figure(dataset, rectangle, outdir)
    swap_matrix_figure(result, outdir, errors=False, error_vmax=error_vmax)
    swap_matrix_figure(result, outdir, errors=True, error_vmax=error_vmax)
    print(f"Fixed full-swap error color scale: 0.00 to {error_vmax:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="docs/figures/run9_no_pressure")
    parser.add_argument(
        "--error-vmax",
        type=float,
        default=None,
        help=(
            "Fixed upper limit for all full-swap absolute-error colorbars. "
            "Defaults to plotting.error_vmax in the config."
        ),
    )
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.checkpoint,
        arguments.outdir,
        arguments.error_vmax,
    )
