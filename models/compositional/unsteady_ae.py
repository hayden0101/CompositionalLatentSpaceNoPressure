import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.compositional.compositional_ae import cross_block_correlation
from models.compositional.networks import (FieldDecoder, FieldEncoder,
                                           LatentTimeStepper, RegimeHead,
                                           SDFEncoder, SDFHead)


class UnsteadyCompositionalAE(pl.LightningModule):
    """Velocity-only compositional autoencoder for transient LDC sequences.

    z = [z_mu | z_g | z_eta | z_xi]
    Phi advances z_eta while z_mu and z_g remain fixed over a trajectory.
    """

    def __init__(self, in_channels=2, resolution=128, base_channels=32,
                 latent_mu=4, latent_g=32, latent_eta=16, latent_xi=8,
                 sdf_resolution=64, static_geometry=True,
                 lambda_recon=1.0, lambda_regime=0.1, lambda_geo=0.1,
                 lambda_decorr=0.01, lambda_temporal=1.0,
                 lambda_latent=0.1, lambda_static=0.0,
                 rollout_steps=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = FieldEncoder(in_channels, resolution, base_channels,
                                    latent_mu, 0 if static_geometry else latent_g,
                                    latent_eta + latent_xi)
        self.geom_encoder = SDFEncoder(resolution, base_channels, latent_g) if static_geometry else None
        latent_dim = latent_mu + latent_g + latent_eta + latent_xi
        self.decoder = FieldDecoder(latent_dim, in_channels, resolution, base_channels)
        self.regime_head = RegimeHead(latent_mu)
        self.sdf_head = SDFHead(latent_g, sdf_resolution, base_channels)
        self.stepper = LatentTimeStepper(latent_eta, latent_mu, latent_g)

    def encode(self, fields, sdf):
        z_mu, z_g_field, z_dyn = self.encoder(fields)
        z_eta = z_dyn[:, :self.hparams.latent_eta]
        z_xi = z_dyn[:, self.hparams.latent_eta:]
        z_g = self.geom_encoder(sdf) if self.geom_encoder is not None else z_g_field
        return z_mu, z_g, z_eta, z_xi

    def decode(self, z_mu, z_g, z_eta, z_xi):
        return self.decoder(torch.cat([z_mu, z_g, z_eta, z_xi], dim=1))

    @staticmethod
    def masked_mse(pred, truth, mask):
        se = ((pred * mask - truth) ** 2).sum(dim=(1, 2, 3))
        count = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (se / count).mean()

    def _losses(self, batch):
        seq, sdf, mask = batch['fields'], batch['sdf'], batch['mask']
        b, t, c, h, w = seq.shape
        flat = seq.reshape(b * t, c, h, w)
        sdf_flat = sdf[:, None].expand(b, t, *sdf.shape[1:]).reshape(b * t, *sdf.shape[1:])
        z_mu_f, z_g_f, z_eta_f, z_xi_f = self.encode(flat, sdf_flat)
        shape = lambda z: z.view(b, t, -1)
        z_mu, z_g, z_eta, z_xi = map(shape, (z_mu_f, z_g_f, z_eta_f, z_xi_f))

        recon = self.decode(z_mu_f, z_g_f, z_eta_f, z_xi_f).view(b, t, c, h, w)
        loss_recon = self.masked_mse(recon.reshape(b * t, c, h, w), flat,
                                     mask[:, None].expand(b, t, *mask.shape[1:]).reshape(b * t, *mask.shape[1:]))
        loss_regime = F.mse_loss(self.regime_head(z_mu[:, 0]), batch['log_re'])
        sdf_lr = F.interpolate(sdf, size=self.sdf_head.resolution, mode='bilinear', align_corners=False)
        loss_geo = F.mse_loss(self.sdf_head(z_g[:, 0]), sdf_lr)

        # Static blocks should be invariant along a trajectory.
        loss_static = ((z_mu - z_mu[:, :1]) ** 2).mean() + ((z_g - z_g[:, :1]) ** 2).mean()
        # One-step latent supervision.
        pred_eta = self.stepper(z_eta[:, :-1].reshape(-1, z_eta.shape[-1]),
                                z_mu[:, :-1].reshape(-1, z_mu.shape[-1]),
                                z_g[:, :-1].reshape(-1, z_g.shape[-1]))
        loss_latent = F.mse_loss(pred_eta, z_eta[:, 1:].reshape_as(pred_eta))

        # Autoregressive rollout from the first frame; keep sample-specific residual fixed.
        steps = min(self.hparams.rollout_steps, t - 1)
        eta = z_eta[:, 0]
        rollout_loss = flat.new_zeros(())
        for k in range(1, steps + 1):
            eta = self.stepper(eta, z_mu[:, 0], z_g[:, 0])
            pred = self.decode(z_mu[:, 0], z_g[:, 0], eta, z_xi[:, 0])
            rollout_loss = rollout_loss + self.masked_mse(pred, seq[:, k], mask)
        loss_temporal = rollout_loss / max(steps, 1)

        # Probe-friendly block separation at the initial frame.
        blocks = [z_mu[:, 0], z_g[:, 0], z_eta[:, 0], z_xi[:, 0]]
        pairs = [cross_block_correlation(blocks[i], blocks[j])
                 for i in range(len(blocks)) for j in range(i + 1, len(blocks))]
        loss_decorr = torch.stack(pairs).mean()

        hp = self.hparams
        total = (hp.lambda_recon * loss_recon + hp.lambda_regime * loss_regime
                 + hp.lambda_geo * loss_geo + hp.lambda_decorr * loss_decorr
                 + hp.lambda_temporal * loss_temporal
                 + hp.lambda_latent * loss_latent
                 + hp.lambda_static * loss_static)
        return total, {'recon': loss_recon, 'regime': loss_regime, 'geo': loss_geo,
                       'decorr': loss_decorr, 'temporal': loss_temporal,
                       'latent': loss_latent, 'static': loss_static}

    def training_step(self, batch, batch_idx):
        total, parts = self._losses(batch)
        self.log('train_loss', total, prog_bar=True)
        for key, value in parts.items():
            self.log(f'train_loss_{key}', value)
        return total

    def validation_step(self, batch, batch_idx):
        total, parts = self._losses(batch)
        self.log('val_loss', total, prog_bar=True)
        self.log('val_loss_full', parts['recon'], prog_bar=True, on_epoch=True)
        for key, value in parts.items():
            self.log(f'val_loss_{key}', value, on_epoch=True)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
