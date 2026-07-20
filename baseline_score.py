import hashlib
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class CompositionalLDCDataset(Dataset):
    """
    FlowBench 2D lid-driven cavity dataset wrapper.

    Supported modes
    ----------------
    Steady state:
        x: [N, (Re, SDF, mask), H, W]
        y: [N, (u, v, p, ...), H, W]
        returns fields: [C, H, W]

    Unsteady state:
        x: [N, T, (Re, SDF, mask), H, W] or [N, (Re, SDF, mask), H, W]
        y: [N, T, C, H, W] or [N, C, T, H, W]
        returns fields: [T, C, H, W]

    By default only the velocity channels (u, v) are used. Set
    field_channels to include pressure for the original steady-state runs.
    """

    def __init__(self, file_path_x, file_path_y, resolution=256, re_stats=None,
                 unsteady=False, sequence_length=None, sequence_stride=1,
                 field_channels=(0, 1)):
        x = np.load(file_path_x)['data']
        y = np.load(file_path_y)['data']

        self.unsteady = bool(unsteady)
        self.field_channels = tuple(field_channels)
        self.sequence_length = sequence_length
        self.sequence_stride = int(sequence_stride)

        x_static, y_seq = self._normalize_layout(x, y)
        # x_static: [N, 3, H, W], y_seq: [N, T, C, H, W]

        re = torch.tensor(x_static[:, 0, 0, 0].copy(), dtype=torch.float32)

        digests = [hashlib.md5(x_static[i, 1].tobytes()).hexdigest()
                   for i in range(x_static.shape[0])]
        digest_to_id, geo_ids = {}, []
        for d in digests:
            if d not in digest_to_id:
                digest_to_id[d] = len(digest_to_id)
            geo_ids.append(digest_to_id[d])
        base_geo_ids = torch.tensor(geo_ids, dtype=torch.long)

        sdf = torch.tensor(x_static[:, 1:2], dtype=torch.float32)
        mask = torch.tensor(x_static[:, 2:3], dtype=torch.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0

        fields = torch.tensor(y_seq[:, :, self.field_channels], dtype=torch.float32)

        if resolution is not None and resolution != fields.shape[-1]:
            size = (resolution, resolution)
            n, t, c, h, w = fields.shape
            fields = F.interpolate(fields.reshape(n * t, c, h, w), size=size,
                                   mode='bilinear', align_corners=False)
            fields = fields.reshape(n, t, c, resolution, resolution)
            sdf = F.interpolate(sdf, size=size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=size, mode='nearest')

        log_re = torch.log10(re.clamp(min=1e-6))
        if re_stats is None:
            self.re_stats = (log_re.mean().item(), log_re.std().clamp(min=1e-8).item())
        else:
            self.re_stats = re_stats
        log_re = (log_re - self.re_stats[0]) / self.re_stats[1]

        if self.unsteady:
            self._build_sequence_index(fields.shape[0], fields.shape[1])
            self.fields = fields
            self.sdf = sdf
            self.mask = mask
            self.re = re
            self.log_re = log_re
            self.geo_ids = base_geo_ids
            self.sample_geo_ids = torch.tensor([geo_ids[i] for i, _ in self.seq_index], dtype=torch.long)
        else:
            # For a steady run using an unsteady file, train on the final frame.
            self.seq_index = None
            self.fields = fields[:, -1]
            self.sdf = sdf
            self.mask = mask
            self.re = re
            self.log_re = log_re
            self.geo_ids = base_geo_ids
            self.sample_geo_ids = self.geo_ids

    def _normalize_layout(self, x, y):
        if x.ndim == 5:
            # [N, T, C, H, W] -> use the first x frame; Re/SDF/mask are static.
            x_static = x[:, 0]
        elif x.ndim == 4:
            x_static = x
        else:
            raise ValueError(f'Unsupported x shape {x.shape}; expected 4D or 5D.')

        if y.ndim == 4:
            y_seq = y[:, None]
        elif y.ndim == 5:
            # Prefer [N, T, C, H, W]. If the second axis is channel-like and the
            # third is time-like, convert [N, C, T, H, W] -> [N, T, C, H, W].
            if y.shape[1] <= 8 and y.shape[2] > 8:
                y_seq = np.moveaxis(y, 1, 2)
            else:
                y_seq = y
        else:
            raise ValueError(f'Unsupported y shape {y.shape}; expected 4D or 5D.')
        return x_static, y_seq

    def _build_sequence_index(self, n_samples, n_time):
        if self.sequence_length is None:
            self.sequence_length = n_time
        if self.sequence_length < 2:
            raise ValueError('Unsteady training requires sequence_length >= 2.')
        if self.sequence_length > n_time:
            raise ValueError(f'sequence_length={self.sequence_length} exceeds available T={n_time}.')
        self.seq_index = []
        for i in range(n_samples):
            for start in range(0, n_time - self.sequence_length + 1, self.sequence_stride):
                self.seq_index.append((i, start))

    def __len__(self):
        return len(self.seq_index) if self.unsteady else self.fields.shape[0]

    def __getitem__(self, idx):
        if self.unsteady:
            sample_idx, start = self.seq_index[idx]
            stop = start + self.sequence_length
            return {
                'fields': self.fields[sample_idx, start:stop],
                'sdf': self.sdf[sample_idx],
                'mask': self.mask[sample_idx],
                're': self.re[sample_idx],
                'log_re': self.log_re[sample_idx],
                'geo_id': self.geo_ids[sample_idx],
                'time_start': torch.tensor(start, dtype=torch.long),
            }
        return {
            'fields': self.fields[idx],
            'sdf': self.sdf[idx],
            'mask': self.mask[idx],
            're': self.re[idx],
            'log_re': self.log_re[idx],
            'geo_id': self.geo_ids[idx],
        }

    def geometry_descriptors(self):
        solid = 1.0 - self.mask[:, 0]
        n, h, w = solid.shape
        area = solid.sum(dim=(1, 2)).clamp(min=1.0)
        xs = torch.linspace(0, 1, w).view(1, 1, w)
        ys = torch.linspace(0, 1, h).view(1, h, 1)
        cx = (solid * xs).sum(dim=(1, 2)) / area
        cy = (solid * ys).sum(dim=(1, 2)) / area
        return torch.stack([area / (h * w), cx, cy], dim=1)


class GroupedBatchSampler(Sampler):
    """Group-structured minibatches for same-factor invariance."""

    def __init__(self, group_ids, batch_size, groups_per_batch=4, seed=0):
        if batch_size % groups_per_batch != 0:
            raise ValueError('batch_size must be divisible by groups_per_batch')
        self.samples_per_group = batch_size // groups_per_batch
        self.groups_per_batch = groups_per_batch
        self.seed = seed
        self.epoch = 0

        self.groups = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            self.groups[int(gid)].append(idx)

        max_group = max(len(v) for v in self.groups.values())
        if max_group < 2:
            warnings.warn('Every geometry appears only once in this dataset; '
                          'the same-factor invariance loss will be zero.')

    def _chunks(self, rng):
        chunks = []
        for idxs in self.groups.values():
            idxs = idxs[:]
            rng.shuffle(idxs)
            for j in range(0, len(idxs), self.samples_per_group):
                chunk = idxs[j:j + self.samples_per_group]
                while len(chunk) < self.samples_per_group:
                    chunk.append(rng.choice(idxs))
                chunks.append(chunk)
        rng.shuffle(chunks)
        return chunks

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        chunks = self._chunks(rng)
        for b in range(len(chunks) // self.groups_per_batch):
            batch = []
            for chunk in chunks[b * self.groups_per_batch:(b + 1) * self.groups_per_batch]:
                batch.extend(chunk)
            yield batch

    def __len__(self):
        n_chunks = sum(-(-len(v) // self.samples_per_group) for v in self.groups.values())
        return n_chunks // self.groups_per_batch
