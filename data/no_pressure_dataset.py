"""Velocity-only dataset adapter for the compositional LDC experiments.

The upstream dataset returns the three physical field channels ``(u, v, p)``.
This adapter keeps the upstream geometry, mask, Reynolds-number, grouping, and
normalization behavior unchanged, then materializes a new contiguous tensor
containing only ``(u, v)``. Pressure is therefore never passed to the encoder,
decoder, losses, diagnostics, or plotting code used by the no-pressure runs.
"""

from data.dataset import CompositionalLDCDataset as _CompositionalLDCDataset


class NoPressureLDCDataset(_CompositionalLDCDataset):
    """Drop the pressure target while preserving all upstream metadata."""

    channel_names = ("u", "v")
    field_channels = (0, 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.fields.ndim != 4 or self.fields.shape[1] < 2:
            raise ValueError(
                "Expected fields with shape (N, C, H, W) and at least the u/v "
                f"channels, but received {tuple(self.fields.shape)}."
            )

        # ``contiguous`` allocates velocity-only storage instead of retaining a
        # view backed by the original three-channel (u, v, p) tensor.
        self.fields = self.fields[:, :2].contiguous()
