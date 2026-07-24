"""Evaluation helpers shared by Run 9 diagnostics and plotting."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from full_swap_utils import Rectangle, SWAP_SPECS


@dataclass
class RectangleEvaluation:
    """Decoded fields and error metrics for all eight latent combinations."""

    rectangle: Rectangle
    predictions: torch.Tensor
    targets: torch.Tensor
    masks: torch.Tensor
    target_reconstructions: torch.Tensor
    mse: torch.Tensor
    reconstruction_mse: torch.Tensor
    ratio: torch.Tensor
    per_channel_mse: torch.Tensor


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Per-sample and per-channel fluid-region MSE."""

    masked_prediction = prediction * mask
    squared_error = (masked_prediction - target) ** 2
    node_count = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    full = squared_error.sum(dim=(1, 2, 3)) / node_count
    per_channel = squared_error.sum(dim=(2, 3)) / node_count[:, None]
    return full, per_channel


@torch.no_grad()
def evaluate_rectangle(model, dataset, rectangle: Rectangle, device: torch.device):
    """Evaluate all eight A/B latent assignments for one rectangle."""

    model = model.to(device)
    model.eval()

    labels = ("A", "B", "C", "D")
    indices = torch.tensor(rectangle.indices, dtype=torch.long)
    fields = dataset.fields[indices].to(device)
    sdf = dataset.sdf[indices].to(device)
    masks = dataset.mask[indices].to(device)

    z_mu, z_g, z_xi = model.encode(fields, sdf)
    label_to_row = {label: row for row, label in enumerate(labels)}

    mixed_latents = []
    target_rows = []
    for spec in SWAP_SPECS:
        mixed_latents.append(
            torch.cat(
                [
                    z_mu[label_to_row[spec.mu_source]],
                    z_g[label_to_row[spec.geometry_source]],
                    z_xi[label_to_row[spec.residual_source]],
                ],
                dim=0,
            )
        )
        target_rows.append(label_to_row[spec.target])

    predictions = model.decoder(torch.stack(mixed_latents, dim=0))
    target_index = torch.tensor(target_rows, dtype=torch.long, device=device)
    targets = fields[target_index]
    target_masks = masks[target_index]

    own_latents = torch.cat([z_mu, z_g, z_xi], dim=1)
    ordinary_reconstructions = model.decoder(own_latents)
    target_reconstructions = ordinary_reconstructions[target_index]

    mse, per_channel_mse = _masked_mse(predictions, targets, target_masks)
    reconstruction_mse, _ = _masked_mse(
        target_reconstructions, targets, target_masks
    )
    ratio = mse / reconstruction_mse.clamp(min=torch.finfo(mse.dtype).eps)

    return RectangleEvaluation(
        rectangle=rectangle,
        predictions=predictions.detach().cpu(),
        targets=targets.detach().cpu(),
        masks=target_masks.detach().cpu(),
        target_reconstructions=target_reconstructions.detach().cpu(),
        mse=mse.detach().cpu(),
        reconstruction_mse=reconstruction_mse.detach().cpu(),
        ratio=ratio.detach().cpu(),
        per_channel_mse=per_channel_mse.detach().cpu(),
    )
