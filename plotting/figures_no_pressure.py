#!/usr/bin/env python3
"""Generate the repository's reconstruction and transfer figures without p.

Figure A has two rows (u, v) and three columns:
CFD truth / model prediction / absolute error.

Figure B has two rows (u, v) and four columns:
donor truth / cross-Re prediction / target CFD truth / absolute error.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from data.no_pressure_dataset import NoPressureLDCDataset
from models.compositional.compositional_ae import CompositionalAE
from plotting.plot_scale import configured_error_vmax, error_ticks

CHANNELS = ("u", "v")
FIELD_CMAP = plt.get_cmap("RdBu_r").copy()
ERROR_CMAP = plt.get_cmap("magma").copy()
FIELD_CMAP.set_bad("0.35")
ERROR_CMAP.set_bad("0.35")


def masked(field, mask):
    """Return a floating copy with solid pixels hidden from the colormap."""
    out = np.array(field, dtype=float)
    out[mask < 0.5] = np.nan
    return out


def robust_vmax(*fields):
    """Return a stable 99th-percentile amplitude for field or error panels."""
    chunks = [f[np.isfinite(f)].ravel() for f in fields]
    chunks = [chunk for chunk in chunks if chunk.size]
    if not chunks:
        return 1.0
    value = float(np.percentile(np.abs(np.concatenate(chunks)), 99.0))
    return max(value, np.finfo(float).eps)


def _panel_row(axs, row, images, mask, vmax, err_vmax):
    """Render field panels plus a final absolute-error panel."""
    im_field = None
    for col, image in enumerate(images[:-1]):
        im_field = axs[row, col].imshow(
            masked(image, mask),
            cmap=FIELD_CMAP,
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
        )

    im_error = axs[row, len(images) - 1].imshow(
        masked(images[-1], mask),
        cmap=ERROR_CMAP,
        vmin=0,
        vmax=err_vmax,
        origin="lower",
    )

    for col in range(len(images)):
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    return im_field, im_error


def _save(fig, outdir, name):
    for extension in ("png", "pdf"):
        path = os.path.join(outdir, f"{name}.{extension}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(outdir, name)}.png / .pdf")


@torch.no_grad()
def reconstruction_figure(model, dataset, idx, outdir, error_vmax):
    sample = dataset[idx]
    fields = sample["fields"].unsqueeze(0)
    reconstruction, _ = model(fields, sample["sdf"].unsqueeze(0))

    truth = fields[0].numpy()
    prediction = reconstruction[0].numpy()
    mask = sample["mask"][0].numpy()

    if truth.shape[0] != len(CHANNELS) or prediction.shape[0] != len(CHANNELS):
        raise ValueError(
            "The no-pressure figure path expects exactly two model channels "
            f"(u, v), but received truth={truth.shape[0]} and "
            f"prediction={prediction.shape[0]}."
        )

    fig, axs = plt.subplots(len(CHANNELS), 3, figsize=(9.5, 6.2), squeeze=False)
    for row, channel in enumerate(CHANNELS):
        error = np.abs(prediction[row] - truth[row])
        vmax = robust_vmax(masked(truth[row], mask))
        err_vmax = error_vmax
        im_field, im_error = _panel_row(
            axs,
            row,
            [truth[row], prediction[row], error],
            mask,
            vmax,
            err_vmax,
        )
        axs[row, 0].set_ylabel(
            channel, fontsize=13, rotation=0, labelpad=15, va="center"
        )
        fig.colorbar(im_field, ax=axs[row, :2].tolist(), shrink=0.85, pad=0.02)
        error_bar = fig.colorbar(
            im_error, ax=axs[row, 2], shrink=0.85, pad=0.04, extend="max"
        )
        error_bar.set_ticks(error_ticks(error_vmax))

    axs[0, 0].set_title("CFD truth")
    axs[0, 1].set_title("Model prediction")
    axs[0, 2].set_title("|error|")
    fig.suptitle(
        f'Reconstruction, test sample {idx} (Re = {sample["re"].item():.0f})',
        y=0.99,
    )
    _save(fig, outdir, f"reconstruction_{idx}")


def find_transfer_triple(dataset):
    """Find (i, k, m) with the largest available Reynolds-number jump.

    The target m has the regime of i and the geometry of k, while k is observed
    at a different Reynolds number and i has a different geometry.
    """
    reynolds = dataset.re.tolist()
    geometry_ids = dataset.geo_ids.tolist()

    members = {}
    for index, geometry_id in enumerate(geometry_ids):
        members.setdefault(geometry_id, []).append(index)

    best = None
    best_score = -1.0
    for target, (target_re, target_geometry) in enumerate(
        zip(reynolds, geometry_ids)
    ):
        regime_donor = next(
            (
                index
                for index, (candidate_re, candidate_geometry) in enumerate(
                    zip(reynolds, geometry_ids)
                )
                if candidate_re == target_re
                and candidate_geometry != target_geometry
            ),
            None,
        )
        if regime_donor is None:
            continue

        for geometry_donor in members[target_geometry]:
            if reynolds[geometry_donor] == target_re:
                continue
            score = abs(np.log10(reynolds[geometry_donor] / target_re))
            if score > best_score:
                best = (regime_donor, geometry_donor, target)
                best_score = score

    return best


@torch.no_grad()
def transfer_figure(model, dataset, outdir, error_vmax):
    triple = find_transfer_triple(dataset)
    if triple is None:
        print("No cross-Re transfer triple found in this split; skipping transfer.")
        return

    regime_donor, geometry_donor, target = triple
    target_re = dataset.re[regime_donor].item()
    donor_re = dataset.re[geometry_donor].item()

    z_mu, _, _ = model.encode(
        dataset.fields[regime_donor : regime_donor + 1],
        dataset.sdf[regime_donor : regime_donor + 1],
    )
    _, z_g, z_xi = model.encode(
        dataset.fields[geometry_donor : geometry_donor + 1],
        dataset.sdf[geometry_donor : geometry_donor + 1],
    )
    prediction = model.decoder(torch.cat([z_mu, z_g, z_xi], dim=1))[0].numpy()

    donor = dataset.fields[geometry_donor].numpy()
    truth = dataset.fields[target].numpy()
    mask = dataset.mask[target, 0].numpy()

    if any(array.shape[0] != len(CHANNELS) for array in (donor, prediction, truth)):
        raise ValueError("The no-pressure transfer figure expects u and v only.")

    fig, axs = plt.subplots(len(CHANNELS), 4, figsize=(12.5, 6.2), squeeze=False)
    for row, channel in enumerate(CHANNELS):
        error = np.abs(prediction[row] - truth[row])
        vmax = robust_vmax(masked(truth[row], mask))
        err_vmax = error_vmax
        im_field, im_error = _panel_row(
            axs,
            row,
            [donor[row], prediction[row], truth[row], error],
            mask,
            vmax,
            err_vmax,
        )
        axs[row, 0].set_ylabel(
            channel, fontsize=13, rotation=0, labelpad=15, va="center"
        )
        fig.colorbar(im_field, ax=axs[row, :3].tolist(), shrink=0.85, pad=0.02)
        error_bar = fig.colorbar(
            im_error, ax=axs[row, 3], shrink=0.85, pad=0.04, extend="max"
        )
        error_bar.set_ticks(error_ticks(error_vmax))

    axs[0, 0].set_title(f"donor: geometry\nat Re = {donor_re:.0f}", fontsize=10)
    axs[0, 1].set_title(f"prediction\nat Re = {target_re:.0f}", fontsize=10)
    axs[0, 2].set_title(f"CFD truth\nat Re = {target_re:.0f}", fontsize=10)
    axs[0, 3].set_title("|error|", fontsize=10)
    fig.suptitle(
        "Cross-Re transfer: the regime code moves a geometry "
        f"from Re = {donor_re:.0f} to Re = {target_re:.0f}",
        y=0.99,
    )
    _save(fig, outdir, "transfer")


def main(config_path, checkpoint_path, outdir, n_recon, error_vmax_override=None):
    config = OmegaConf.load(config_path)
    configured_channels = int(config.model.get("in_channels", 2))
    error_vmax = configured_error_vmax(config, error_vmax_override)
    if configured_channels != 2:
        raise ValueError(
            f"No-pressure plotting requires model.in_channels=2, got "
            f"{configured_channels}."
        )

    train_dataset = NoPressureLDCDataset(
        file_path_x=config.data.file_path_train_x,
        file_path_y=config.data.file_path_train_y,
        resolution=config.data.resolution,
    )
    dataset = NoPressureLDCDataset(
        file_path_x=config.data.file_path_test_x,
        file_path_y=config.data.file_path_test_y,
        resolution=config.data.resolution,
        re_stats=train_dataset.re_stats,
    )

    model = CompositionalAE.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(0)
    sample_count = min(n_recon, len(dataset))
    for idx in rng.choice(len(dataset), size=sample_count, replace=False):
        reconstruction_figure(model, dataset, int(idx), outdir, error_vmax)
    transfer_figure(model, dataset, outdir, error_vmax)
    print(f"Fixed error color scale: 0.00 to {error_vmax:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate velocity-only reconstruction and transfer figures."
    )
    parser.add_argument(
        "--config", type=str, default="configs/compositional/run7_no_pressure.yaml"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="docs/figures/no_pressure")
    parser.add_argument("--n-recon", type=int, default=2)
    parser.add_argument(
        "--error-vmax",
        type=float,
        default=None,
        help=(
            "Fixed upper limit for every absolute-error colorbar. "
            "Defaults to plotting.error_vmax in the config or 0.05."
        ),
    )
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.checkpoint,
        arguments.outdir,
        arguments.n_recon,
        arguments.error_vmax,
    )
