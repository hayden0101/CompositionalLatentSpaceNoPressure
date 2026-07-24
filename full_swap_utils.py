"""Utilities for the Run 9 full latent-swap experiment.

A complete 2 x 2 rectangle contains two geometries and two Reynolds numbers::

    A = (Re_A, geometry_A)     C = (Re_A, geometry_B)
    D = (Re_B, geometry_A)     B = (Re_B, geometry_B)

The eight A/B latent combinations are evaluated against the exact CFD target
selected by the source of ``z_mu`` (Reynolds number) and ``z_g`` (geometry).
``z_xi`` is swapped independently so its functional dependence can be tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Rectangle:
    """Indices for a complete two-Reynolds/two-geometry test rectangle."""

    a: int
    b: int
    c: int
    d: int
    re_a: float
    re_b: float
    geometry_a: int
    geometry_b: int

    @property
    def indices(self) -> tuple[int, int, int, int]:
        return self.a, self.b, self.c, self.d

    @property
    def log10_jump(self) -> float:
        import math

        if self.re_a <= 0 or self.re_b <= 0:
            return 0.0
        return abs(math.log10(self.re_b / self.re_a))


@dataclass(frozen=True)
class SwapSpec:
    """One source assignment for ``[z_mu | z_g | z_xi]``."""

    code: str
    mu_source: str
    geometry_source: str
    residual_source: str
    target: str
    label: str


SWAP_SPECS: tuple[SwapSpec, ...] = (
    SwapSpec("AAA", "A", "A", "A", "A", "no swap: reconstruct A"),
    SwapSpec("AAB", "A", "A", "B", "A", "residual-only swap into A"),
    SwapSpec("ABA", "A", "B", "A", "C", "geometry-only swap"),
    SwapSpec("ABB", "A", "B", "B", "C", "geometry + residual swap"),
    SwapSpec("BAA", "B", "A", "A", "D", "regime-only swap into A"),
    SwapSpec("BAB", "B", "A", "B", "D", "regime + residual swap"),
    SwapSpec("BBA", "B", "B", "A", "B", "residual-only swap into B"),
    SwapSpec("BBB", "B", "B", "B", "B", "no swap: reconstruct B"),
)

# Run 7 already trains ordinary reconstruction (AAA, BBB) and the two standard
# cross-Re compositions (ABB, BAA). Run 9 adds the four missing assignments so
# the active objective collectively covers all eight combinations.
EXTRA_TRAINING_CODES: tuple[str, ...] = ("AAB", "ABA", "BAB", "BBA")


def enumerate_rectangles(
    reynolds: Sequence[float] | Iterable[float],
    geometry_ids: Sequence[int] | Iterable[int],
) -> list[Rectangle]:
    """Return every complete 2 x 2 (Re, geometry) rectangle in the samples.

    Duplicate ``(Re, geometry)`` entries are reduced to their first occurrence.
    Reynolds values are ordered low-to-high and geometries by their integer ID,
    giving a deterministic A/B/C/D convention.
    """

    re_values = [float(value) for value in reynolds]
    geo_values = [int(value) for value in geometry_ids]
    if len(re_values) != len(geo_values):
        raise ValueError("reynolds and geometry_ids must have equal length")

    lookup: dict[tuple[float, int], int] = {}
    re_by_geometry: dict[int, set[float]] = {}
    for index, (re_value, geometry_id) in enumerate(zip(re_values, geo_values)):
        lookup.setdefault((re_value, geometry_id), index)
        re_by_geometry.setdefault(geometry_id, set()).add(re_value)

    rectangles: list[Rectangle] = []
    for geometry_a, geometry_b in combinations(sorted(re_by_geometry), 2):
        shared_re = sorted(re_by_geometry[geometry_a] & re_by_geometry[geometry_b])
        for re_a, re_b in combinations(shared_re, 2):
            rectangles.append(
                Rectangle(
                    a=lookup[(re_a, geometry_a)],
                    b=lookup[(re_b, geometry_b)],
                    c=lookup[(re_a, geometry_b)],
                    d=lookup[(re_b, geometry_a)],
                    re_a=re_a,
                    re_b=re_b,
                    geometry_a=geometry_a,
                    geometry_b=geometry_b,
                )
            )
    return rectangles


def largest_jump_rectangle(
    reynolds: Sequence[float] | Iterable[float],
    geometry_ids: Sequence[int] | Iterable[int],
) -> Rectangle | None:
    """Return the complete rectangle spanning the largest Re ratio."""

    rectangles = enumerate_rectangles(reynolds, geometry_ids)
    if not rectangles:
        return None
    return max(rectangles, key=lambda rectangle: rectangle.log10_jump)


def select_rectangles(
    rectangles: Sequence[Rectangle], max_rectangles: int, seed: int = 0
) -> list[Rectangle]:
    """Select a deterministic random subset without replacement."""

    if max_rectangles <= 0:
        raise ValueError("max_rectangles must be positive")
    if len(rectangles) <= max_rectangles:
        return list(rectangles)

    import random

    rng = random.Random(seed)
    selected = rng.sample(list(rectangles), max_rectangles)
    return sorted(
        selected,
        key=lambda rectangle: (
            rectangle.geometry_a,
            rectangle.geometry_b,
            rectangle.re_a,
            rectangle.re_b,
        ),
    )


def spec_by_code(code: str) -> SwapSpec:
    """Return one swap specification by its three-letter code."""

    for spec in SWAP_SPECS:
        if spec.code == code:
            return spec
    raise KeyError(f"Unknown swap code: {code}")
