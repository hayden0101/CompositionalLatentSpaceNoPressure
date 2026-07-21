import hashlib
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class CompositionalLDCDataset(Dataset):
    """Steady FlowBench dataset, optionally restricted to selected field channels."""

    def __init__(self, file_path_x, file_path_y, resolution=256, re_stats=None,
                 field_channels=(0, 1, 2)):
        x = np.load(file_path_x)['data']
        y = np.load(file_path_y)['data']
        re = torch.tensor(x[:, 0, 0, 0].copy(), dtype=torch.float32)
        digests = [hashlib.md5(x[i, 1].tobytes()).hexdigest() for i in range(x.shape[0])]
        digest_to_id, geo_ids = {}, []
        for digest in digests:
            digest_to_id.setdefault(digest, len(digest_to_id))
            geo_ids.append(digest_to_id[digest])
        self.geo_ids = torch.tensor(geo_ids, dtype=torch.long)

        sdf = torch.tensor(x[:, 1:2], dtype=torch.float32)
        mask = torch.tensor(x[:, 2:3], dtype=torch.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        fields = torch.tensor(y[:, list(field_channels)], dtype=torch.float32)
        if resolution is not None and resolution != fields.shape[-1]:
            size = (resolution, resolution)
            fields = F.interpolate(fields, size=size, mode='bilinear', align_corners=False)
            sdf = F.interpolate(sdf, size=size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=size, mode='nearest')
        self.fields, self.sdf, self.mask, self.re = fields, sdf, mask, re
        self.re_stats, self.log_re = _standardize_re(re, re_stats)

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        return {'fields': self.fields[idx], 'sdf': self.sdf[idx],
                'mask': self.mask[idx], 're': self.re[idx],
                'log_re': self.log_re[idx], 'geo_id': self.geo_ids[idx]}

    def geometry_descriptors(self):
        return _geometry_descriptors(self.mask)


class UnsteadyLDCDataset(Dataset):
    """Transient velocity-only LDC dataset.

    Expected .npz keys:
      velocity: [N,T,2,H,W], sdf: [N,1,H,W], mask: [N,1,H,W], re: [N]
    Each item is a contiguous sequence of length ``sequence_length``.
    """

    def __init__(self, file_path, resolution=128, sequence_length=8,
                 stride=1, re_stats=None):
        d = np.load(file_path)
        velocity = torch.tensor(d['velocity'], dtype=torch.float32)
        self.sdf = torch.tensor(d['sdf'], dtype=torch.float32)
        self.mask = torch.tensor(d['mask'], dtype=torch.float32)
        self.re = torch.tensor(d['re'], dtype=torch.float32)
        if self.mask.max() > 1.0:
            self.mask /= 255.0
        if resolution is not None and resolution != velocity.shape[-1]:
            n, t, c, _, _ = velocity.shape
            velocity = F.interpolate(velocity.view(n * t, c, *velocity.shape[-2:]),
                                     size=(resolution, resolution), mode='bilinear',
                                     align_corners=False).view(n, t, c, resolution, resolution)
            self.sdf = F.interpolate(self.sdf, size=(resolution, resolution),
                                     mode='bilinear', align_corners=False)
            self.mask = F.interpolate(self.mask, size=(resolution, resolution), mode='nearest')
        self.velocity = velocity
        self.sequence_length = sequence_length
        self.stride = stride
        self.re_stats, self.log_re = _standardize_re(self.re, re_stats)

        digests = [hashlib.md5(self.sdf[i].numpy().tobytes()).hexdigest()
                   for i in range(self.sdf.shape[0])]
        mapping = {}
        self.geo_ids = torch.tensor([mapping.setdefault(x, len(mapping)) for x in digests])
        self.windows = []
        max_start = velocity.shape[1] - (sequence_length - 1) * stride
        if max_start <= 0:
            raise ValueError('sequence_length/stride exceed the number of stored frames')
        for sim in range(velocity.shape[0]):
            for start in range(max_start):
                self.windows.append((sim, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sim, start = self.windows[idx]
        ids = start + torch.arange(self.sequence_length) * self.stride
        return {'fields': self.velocity[sim, ids], 'sdf': self.sdf[sim],
                'mask': self.mask[sim], 're': self.re[sim],
                'log_re': self.log_re[sim], 'geo_id': self.geo_ids[sim],
                'time_index': torch.tensor(start, dtype=torch.long)}

    def geometry_descriptors(self):
        return _geometry_descriptors(self.mask)


def _standardize_re(re, re_stats):
    log_re = torch.log10(re.clamp(min=1e-6))
    if re_stats is None:
        re_stats = (log_re.mean().item(), log_re.std().clamp(min=1e-8).item())
    return re_stats, (log_re - re_stats[0]) / re_stats[1]


def _geometry_descriptors(mask):
    solid = 1.0 - mask[:, 0]
    n, h, w = solid.shape
    area = solid.sum(dim=(1, 2)).clamp(min=1.0)
    xs = torch.linspace(0, 1, w).view(1, 1, w)
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    cx = (solid * xs).sum(dim=(1, 2)) / area
    cy = (solid * ys).sum(dim=(1, 2)) / area
    return torch.stack([area / (h * w), cx, cy], dim=1)


class GroupedBatchSampler(Sampler):
    def __init__(self, group_ids, batch_size, groups_per_batch=4, seed=0):
        if batch_size % groups_per_batch != 0:
            raise ValueError('batch_size must be divisible by groups_per_batch')
        self.samples_per_group = batch_size // groups_per_batch
        self.groups_per_batch, self.seed, self.epoch = groups_per_batch, seed, 0
        self.groups = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            self.groups[int(gid)].append(idx)
        if max(len(v) for v in self.groups.values()) < 2:
            warnings.warn('Every geometry appears only once; invariance loss will be zero.')

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
            yield sum(chunks[b * self.groups_per_batch:(b + 1) * self.groups_per_batch], [])

    def __len__(self):
        n = sum(-(-len(v) // self.samples_per_group) for v in self.groups.values())
        return n // self.groups_per_batch
