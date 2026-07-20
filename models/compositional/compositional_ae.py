import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.compositional.networks import (FieldEncoder, FieldDecoder,
                                           RegimeHead, SDFHead, SDFEncoder,
                                           LatentTimeStepper)


def same_factor_invariance(z, group_ids):
    loss = z.new_zeros(())
    count = 0
    for gid in group_ids.unique():
        sel = group_ids == gid
        if sel.sum() >= 2:
            zg = z[sel]
            loss = loss + ((zg - zg.mean(0, keepdim=True)) ** 2).mean()
            count += 1
    return loss / count if count else loss


def cross_block_correlation(a, b, eps=1e-8):
    if a is None or b is None or a.shape[0] < 2:
        return torch.zeros((), device=a.device if a is not None else b.device)
    a = (a - a.mean(0)) / (a.std(0) + eps)
    b = (b - b.mean(0)) / (b.std(0) + eps)
    corr = a.t() @ b / (a.shape[0] - 1)
    return corr.abs().mean()


class CompositionalAE(pl.LightningModule):
    """
    Compositional autoencoder for steady or unsteady LDC flow.

    Steady latent:
        z = [z_mu || z_g || z_xi]

    Unsteady latent:
        z_t = [z_mu || z_g || z_eta_t || z_xi_t]
        Phi(z_t) predicts z_eta_{t+1}

    By setting in_channels=2 and field_channels=(0, 1) in the dataset, the
    model reconstructs velocity only: horizontal velocity u and vertical
    velocity v. Pressure is excluded.
    """

    def __init__(self, in_channels=2, resolution=256, base_channels=32,
                 latent_mu=4, latent_g=32, latent_eta=16, latent_xi=16,
                 sdf_resolution=64, static_geometry=False, unsteady=False,
                 lambda_recon=1.0, lambda_temporal=0.0, lambda_bl=0.0,
                 lambda_regime=0.1, lambda_geo=0.1, lambda_decorr=0.01,
                 lambda_inv=0.0, lambda_swap=0.0, lambda_xswap=0.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        dynamic_dim = latent_eta if unsteady else 0
        latent_dim = latent_mu + latent_g + dynamic_dim + latent_xi

        if static_geometry:
            self.encoder = FieldEncoder(in_channels, resolution, base_channels,
                                        latent_mu, 0, latent_xi, dynamic_dim)
            self.geom_encoder = SDFEncoder(resolution, base_channels, latent_g)
        else:
            self.encoder = FieldEncoder(in_channels, resolution, base_channels,
                                        latent_mu, latent_g, latent_xi, dynamic_dim)
        self.decoder = FieldDecoder(latent_dim, in_channels, resolution, base_channels)
        self.regime_head = RegimeHead(latent_mu)
        self.sdf_head = SDFHead(latent_g, sdf_resolution, base_channels)
        self.time_stepper = (LatentTimeStepper(latent_dim, latent_eta)
                             if unsteady else None)

    def _cat(self, z_mu, z_g, z_xi, z_eta=None):
        parts = [z_mu, z_g]
        if self.hparams.unsteady:
            parts.append(z_eta)
        parts.append(z_xi)
        return torch.cat(parts, dim=1)

    def encode(self, fields, sdf=None):
        z_mu, z_g, z_eta, z_xi = self.encoder(fields)
        if self.hparams.static_geometry:
            if sdf is None:
                raise ValueError('static_geometry model needs the sdf to encode')
            z_g = self.geom_encoder(sdf)
        return z_mu, z_g, z_eta, z_xi

    def forward(self, fields, sdf=None):
        if fields.ndim == 5:
            return self.forward_sequence(fields, sdf)
        z_mu, z_g, z_eta, z_xi = self.encode(fields, sdf)
        recon = self.decoder(self._cat(z_mu, z_g, z_xi, z_eta))
        return recon, (z_mu, z_g, z_eta, z_xi)

    def forward_sequence(self, fields, sdf):
        b, t, c, h, w = fields.shape
        flat = fields.reshape(b * t, c, h, w)
        sdf_flat = sdf[:, None].expand(b, t, 1, h, w).reshape(b * t, 1, h, w)
        z_mu, z_g, z_eta, z_xi = self.encode(flat, sdf_flat)
        recon = self.decoder(self._cat(z_mu, z_g, z_xi, z_eta))
        recon = recon.reshape(b, t, c, h, w)
        shape = (b, t, -1)
        return recon, (z_mu.reshape(*shape), z_g.reshape(*shape),
                       z_eta.reshape(*shape), z_xi.reshape(*shape))

    def masked_recon_loss(self, recon, fields, mask, per_channel=False):
        if fields.ndim == 5:
            mask = mask[:, None]
        recon = recon * mask
        se = (recon - fields) ** 2
        reduce_dims = tuple(range(1, se.ndim))
        node_count = mask.sum(dim=tuple(range(1, mask.ndim))).clamp(min=1.0)
        channel_factor = fields.shape[-3]
        full = (se.sum(dim=reduce_dims) / (node_count * channel_factor)).mean()
        if not per_channel:
            return full
        if fields.ndim == 5:
            per = [(se[:, :, i].sum(dim=(1, 2, 3)) / node_count).mean()
                   for i in range(se.shape[2])]
        else:
            per = [(se[:, i].sum(dim=(1, 2)) / node_count).mean()
                   for i in range(se.shape[1])]
        return full, per

    def boundary_recon_loss(self, recon, fields, mask, sdf):
        band = mask * ((sdf >= 0) & (sdf <= 0.2)).float()
        if fields.ndim == 5:
            band = band[:, None]
            mask = mask[:, None]
        se = ((recon * mask - fields) ** 2 * band).sum(dim=tuple(range(1, fields.ndim)))
        count = band.sum(dim=tuple(range(1, band.ndim))).clamp(min=1.0)
        channel_factor = fields.shape[-3]
        return (se / (count * channel_factor)).mean()

    def temporal_loss(self, fields, sdf, mask):
        """One-step latent rollout loss for unsteady flow.

        Encode every true frame, advance z_eta_t with Phi, then decode using
        the next frame's residual z_xi_{t+1}. This makes the dynamics block
        responsible for time evolution while avoiding a brittle requirement
        that z_xi be perfectly time-invariant at the start of experiments.
        """
        if not self.hparams.unsteady or fields.shape[1] < 2:
            return fields.new_zeros(())
        recon, (z_mu, z_g, z_eta, z_xi) = self.forward_sequence(fields, sdf)
        z_now = self._cat(z_mu[:, :-1].reshape(-1, z_mu.shape[-1]),
                          z_g[:, :-1].reshape(-1, z_g.shape[-1]),
                          z_xi[:, :-1].reshape(-1, z_xi.shape[-1]),
                          z_eta[:, :-1].reshape(-1, z_eta.shape[-1]))
        eta_next = self.time_stepper(z_now)
        z_pred = self._cat(z_mu[:, :-1].reshape(-1, z_mu.shape[-1]),
                           z_g[:, :-1].reshape(-1, z_g.shape[-1]),
                           z_xi[:, 1:].reshape(-1, z_xi.shape[-1]),
                           eta_next)
        pred = self.decoder(z_pred)
        b, t, c, h, w = fields[:, 1:].shape
        pred = pred.reshape(b, t, c, h, w)
        return self.masked_recon_loss(pred, fields[:, 1:], mask)

    def swap_consistency(self, z_mu, z_g, z_xi, batch):
        if batch['fields'].ndim == 5:
            return z_mu.new_zeros(()), 0
        re, gid = batch['re'], batch['geo_id']
        n = re.shape[0]
        same_re = re.view(-1, 1) == re.view(1, -1)
        diff_geo = gid.view(-1, 1) != gid.view(1, -1)
        idx_i, idx_k = torch.nonzero(same_re & diff_geo, as_tuple=True)
        if idx_i.numel() == 0:
            return z_mu.new_zeros(()), 0
        if idx_i.numel() > n:
            keep = torch.randperm(idx_i.numel(), device=idx_i.device)[:n]
            idx_i, idx_k = idx_i[keep], idx_k[keep]
        z_swap = self._cat(z_mu[idx_i], z_g[idx_k], z_xi[idx_k], None)
        recon = self.decoder(z_swap)
        loss = self.masked_recon_loss(recon, batch['fields'][idx_k], batch['mask'][idx_k])
        return loss, idx_i.numel()

    def cross_swap_consistency(self, z_mu, z_g, z_xi, batch):
        if batch['fields'].ndim == 5:
            return z_mu.new_zeros(()), 0
        re, gid = batch['re'], batch['geo_id']
        n = re.shape[0]
        same_re = re.view(-1, 1) == re.view(1, -1)
        diff_geo = gid.view(-1, 1) != gid.view(1, -1)
        pair_i, pair_m = torch.nonzero(same_re & diff_geo, as_tuple=True)
        idx_i, idx_k, idx_m = [], [], []
        for i, m in zip(pair_i.tolist(), pair_m.tolist()):
            cand = torch.nonzero((gid == gid[m]) & (re != re[i])).flatten()
            if cand.numel():
                k = cand[torch.randint(cand.numel(), (1,))].item()
                idx_i.append(i); idx_k.append(k); idx_m.append(m)
        if not idx_i:
            return z_mu.new_zeros(()), 0
        if len(idx_i) > n:
            keep = torch.randperm(len(idx_i))[:n].tolist()
            idx_i = [idx_i[j] for j in keep]
            idx_k = [idx_k[j] for j in keep]
            idx_m = [idx_m[j] for j in keep]
        z_swap = self._cat(z_mu[idx_i], z_g[idx_k], z_xi[idx_k], None)
        recon = self.decoder(z_swap)
        loss = self.masked_recon_loss(recon, batch['fields'][idx_m], batch['mask'][idx_m])
        return loss, len(idx_i)

    def _losses(self, batch):
        fields, mask = batch['fields'], batch['mask']
        recon, latents = self(fields, batch['sdf'])
        z_mu, z_g, z_eta, z_xi = latents

        loss_recon = self.masked_recon_loss(recon, fields, mask)
        if fields.ndim == 5:
            # z_mu should be time-invariant; supervise every encoded frame.
            target_re = batch['log_re'][:, None].expand(-1, fields.shape[1]).reshape(-1)
            loss_regime = F.mse_loss(self.regime_head(z_mu.reshape(-1, z_mu.shape[-1])), target_re)
            z_g_for_geo = z_g[:, 0]
            z_g_for_decorr = z_g.reshape(-1, z_g.shape[-1])
            z_mu_for_decorr = z_mu.reshape(-1, z_mu.shape[-1])
            z_xi_for_decorr = z_xi.reshape(-1, z_xi.shape[-1])
            z_eta_for_decorr = z_eta.reshape(-1, z_eta.shape[-1])
        else:
            loss_regime = F.mse_loss(self.regime_head(z_mu), batch['log_re'])
            z_g_for_geo = z_g
            z_g_for_decorr = z_g
            z_mu_for_decorr = z_mu
            z_xi_for_decorr = z_xi
            z_eta_for_decorr = z_eta

        if self.hparams.lambda_bl > 0:
            loss_bl = self.boundary_recon_loss(recon, fields, mask, batch['sdf'])
        else:
            loss_bl = loss_recon.new_zeros(())

        sdf_lr = F.interpolate(batch['sdf'], size=self.sdf_head.resolution,
                               mode='bilinear', align_corners=False)
        loss_geo = F.mse_loss(self.sdf_head(z_g_for_geo), sdf_lr)

        corr_terms = [cross_block_correlation(z_mu_for_decorr, z_g_for_decorr),
                      cross_block_correlation(z_mu_for_decorr, z_xi_for_decorr),
                      cross_block_correlation(z_g_for_decorr, z_xi_for_decorr)]
        if self.hparams.unsteady:
            corr_terms.extend([
                cross_block_correlation(z_eta_for_decorr, z_mu_for_decorr),
                cross_block_correlation(z_eta_for_decorr, z_g_for_decorr),
                cross_block_correlation(z_eta_for_decorr, z_xi_for_decorr),
            ])
        loss_decorr = sum(corr_terms) / len(corr_terms)

        loss_inv = same_factor_invariance(z_g_for_geo, batch['geo_id'])

        h = self.hparams
        loss_temporal = (self.temporal_loss(fields, batch['sdf'], mask)
                         if h.lambda_temporal > 0 else loss_recon.new_zeros(()))

        if h.lambda_swap > 0:
            loss_swap, n_pairs = self.swap_consistency(z_mu, z_g, z_xi, batch)
        else:
            loss_swap, n_pairs = loss_recon.new_zeros(()), 0

        if h.lambda_xswap > 0:
            loss_xswap, n_triples = self.cross_swap_consistency(z_mu, z_g, z_xi, batch)
        else:
            loss_xswap, n_triples = loss_recon.new_zeros(()), 0

        total = (h.lambda_recon * loss_recon + h.lambda_temporal * loss_temporal
                 + h.lambda_bl * loss_bl + h.lambda_regime * loss_regime
                 + h.lambda_geo * loss_geo + h.lambda_decorr * loss_decorr
                 + h.lambda_inv * loss_inv + h.lambda_swap * loss_swap
                 + h.lambda_xswap * loss_xswap)
        return total, {'recon': loss_recon, 'temporal': loss_temporal,
                       'bl': loss_bl, 'regime': loss_regime, 'geo': loss_geo,
                       'decorr': loss_decorr, 'inv': loss_inv,
                       'swap': loss_swap, 'swap_pairs': float(n_pairs),
                       'xswap': loss_xswap, 'xswap_triples': float(n_triples)}

    def training_step(self, batch, batch_idx):
        total, parts = self._losses(batch)
        self.log('train_loss', total, prog_bar=True)
        for name, value in parts.items():
            self.log(f'train_loss_{name}', value)
        return total

    def validation_step(self, batch, batch_idx):
        total, parts = self._losses(batch)
        self.log('val_loss', total, prog_bar=True)
        for name, value in parts.items():
            self.log(f'val_loss_{name}', value)

        recon, _ = self(batch['fields'], batch['sdf'])
        full, per = self.masked_recon_loss(recon, batch['fields'], batch['mask'], per_channel=True)
        self.log('val_loss_full', full, on_epoch=True)
        channel_names = ['u', 'v'] if self.hparams.in_channels == 2 else ['u', 'v', 'p']
        for name, value in zip(channel_names, per):
            self.log(f'val_loss_{name}', value, on_epoch=True)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
