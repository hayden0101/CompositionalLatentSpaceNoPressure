"""Run 9 model: Run 7 plus the missing full latent-swap combinations.

The upstream model already trains:

* ordinary reconstruction: AAA and BBB,
* cross-Re transfer: ABB and BAA,
* same-Re regime-code interchangeability.

Run 9 adds AAB, ABA, BAB, and BBA on complete 2 x 2 (Re, geometry)
rectangles. Together, the active objective covers every A/B assignment of
``[z_mu | z_g | z_xi]`` while retaining exact CFD ground truth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from full_swap_utils import EXTRA_TRAINING_CODES, enumerate_rectangles, spec_by_code
from models.compositional.compositional_ae import (
    CompositionalAE,
    cross_block_correlation,
    same_factor_invariance,
)


class FullSwapCompositionalAE(CompositionalAE):
    """Compositional autoencoder with the Run 9 full-swap extension."""

    def __init__(self, lambda_fullswap: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters({"lambda_fullswap": float(lambda_fullswap)})

    def full_swap_consistency(self, z_mu, z_g, z_xi, batch):
        """Train the four A/B assignments not covered by Run 7.

        For each complete rectangle::

            A = (Re_A, g_A)      C = (Re_A, g_B)
            D = (Re_B, g_A)      B = (Re_B, g_B)

        the additional decoded combinations are:

        * AAB -> A: residual-only swap into A,
        * ABA -> C: geometry-only swap with A's residual,
        * BAB -> D: regime + residual swap,
        * BBA -> B: residual-only swap into B.

        Run 7's reconstruction and cross-Re losses already cover AAA, BBB,
        ABB, and BAA. Returns ``(loss, number_of_decoded_combinations)``.
        """

        rectangles = enumerate_rectangles(batch["re"].tolist(), batch["geo_id"].tolist())
        if not rectangles:
            return z_mu.new_zeros(()), 0

        # At most one extra decoded batch. Four combinations are emitted per
        # rectangle, so this keeps the added compute bounded and predictable.
        max_rectangles = max(1, z_mu.shape[0] // len(EXTRA_TRAINING_CODES))
        if len(rectangles) > max_rectangles:
            order = torch.randperm(len(rectangles), device=z_mu.device)[:max_rectangles]
            rectangles = [rectangles[index] for index in order.cpu().tolist()]

        mu_indices: list[int] = []
        geometry_indices: list[int] = []
        residual_indices: list[int] = []
        target_indices: list[int] = []

        for rectangle in rectangles:
            source = {"A": rectangle.a, "B": rectangle.b}
            target = {
                "A": rectangle.a,
                "B": rectangle.b,
                "C": rectangle.c,
                "D": rectangle.d,
            }
            for code in EXTRA_TRAINING_CODES:
                spec = spec_by_code(code)
                mu_indices.append(source[spec.mu_source])
                geometry_indices.append(source[spec.geometry_source])
                residual_indices.append(source[spec.residual_source])
                target_indices.append(target[spec.target])

        device = z_mu.device
        mu_index = torch.tensor(mu_indices, dtype=torch.long, device=device)
        geometry_index = torch.tensor(geometry_indices, dtype=torch.long, device=device)
        residual_index = torch.tensor(residual_indices, dtype=torch.long, device=device)
        target_index = torch.tensor(target_indices, dtype=torch.long, device=device)

        mixed_latent = torch.cat(
            [z_mu[mu_index], z_g[geometry_index], z_xi[residual_index]], dim=1
        )
        prediction = self.decoder(mixed_latent)
        loss = self.masked_recon_loss(
            prediction,
            batch["fields"][target_index],
            batch["mask"][target_index],
        )
        return loss, len(target_indices)

    def _losses(self, batch):
        """Upstream loss stack plus the Run 9 full-swap term."""

        fields, mask = batch["fields"], batch["mask"]
        reconstruction, (z_mu, z_g, z_xi) = self(fields, batch["sdf"])

        loss_recon = self.masked_recon_loss(reconstruction, fields, mask)
        loss_regime = F.mse_loss(self.regime_head(z_mu), batch["log_re"])

        if self.hparams.lambda_bl > 0:
            loss_bl = self.boundary_recon_loss(
                reconstruction, fields, mask, batch["sdf"]
            )
        else:
            loss_bl = loss_recon.new_zeros(())

        sdf_low_resolution = F.interpolate(
            batch["sdf"],
            size=self.sdf_head.resolution,
            mode="bilinear",
            align_corners=False,
        )
        loss_geo = F.mse_loss(self.sdf_head(z_g), sdf_low_resolution)
        loss_decorr = (
            cross_block_correlation(z_mu, z_g)
            + cross_block_correlation(z_mu, z_xi)
            + cross_block_correlation(z_g, z_xi)
        ) / 3.0
        loss_inv = same_factor_invariance(z_g, batch["geo_id"])

        hparams = self.hparams
        if hparams.lambda_swap > 0:
            loss_swap, swap_pairs = self.swap_consistency(z_mu, z_g, z_xi, batch)
        else:
            loss_swap, swap_pairs = loss_recon.new_zeros(()), 0

        if hparams.lambda_xswap > 0:
            loss_xswap, xswap_triples = self.cross_swap_consistency(
                z_mu, z_g, z_xi, batch
            )
        else:
            loss_xswap, xswap_triples = loss_recon.new_zeros(()), 0

        if hparams.lambda_fullswap > 0:
            loss_fullswap, fullswap_combinations = self.full_swap_consistency(
                z_mu, z_g, z_xi, batch
            )
        else:
            loss_fullswap, fullswap_combinations = loss_recon.new_zeros(()), 0

        total = (
            hparams.lambda_recon * loss_recon
            + hparams.lambda_bl * loss_bl
            + hparams.lambda_regime * loss_regime
            + hparams.lambda_geo * loss_geo
            + hparams.lambda_decorr * loss_decorr
            + hparams.lambda_inv * loss_inv
            + hparams.lambda_swap * loss_swap
            + hparams.lambda_xswap * loss_xswap
            + hparams.lambda_fullswap * loss_fullswap
        )

        return total, {
            "recon": loss_recon,
            "bl": loss_bl,
            "regime": loss_regime,
            "geo": loss_geo,
            "decorr": loss_decorr,
            "inv": loss_inv,
            "swap": loss_swap,
            "swap_pairs": float(swap_pairs),
            "xswap": loss_xswap,
            "xswap_triples": float(xswap_triples),
            "fullswap": loss_fullswap,
            "fullswap_combinations": float(fullswap_combinations),
        }
