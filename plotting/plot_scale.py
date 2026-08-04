"""Shared color-scale helpers for pressure-free figures."""

from __future__ import annotations

import numpy as np

DEFAULT_ERROR_VMAX = 0.05


def configured_error_vmax(config, cli_value=None) -> float:
    """Return a positive fixed error limit.

    Command-line values take precedence. Otherwise ``plotting.error_vmax`` is
    read from the YAML configuration, falling back to 0.05. A fixed value makes
    the u and v error maps directly comparable within a figure and across runs.
    """

    value = cli_value
    if value is None:
        plotting = getattr(config, "plotting", None)
        if plotting is not None:
            value = plotting.get("error_vmax", None)
    if value is None:
        value = DEFAULT_ERROR_VMAX
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"error_vmax must be positive and finite, got {value!r}")
    return value


def error_ticks(error_vmax: float, count: int = 6):
    """Return identical colorbar ticks from zero through ``error_vmax``."""

    return np.linspace(0.0, float(error_vmax), int(count))
