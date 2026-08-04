#!/usr/bin/env python3
"""Static and combinatorial checks for the Run 9 extension."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_swap_utils import (  # noqa: E402
    EXTRA_TRAINING_CODES,
    SWAP_SPECS,
    enumerate_rectangles,
    largest_jump_rectangle,
)


def check_rectangle_enumeration():
    # Intentionally scrambled ordering.
    reynolds = [100.0, 10.0, 100.0, 10.0]
    geometry_ids = [2, 1, 1, 2]
    rectangles = enumerate_rectangles(reynolds, geometry_ids)
    assert len(rectangles) == 1

    rectangle = rectangles[0]
    assert rectangle.re_a == 10.0
    assert rectangle.re_b == 100.0
    assert rectangle.geometry_a == 1
    assert rectangle.geometry_b == 2
    assert rectangle.a == 1  # Re 10, geometry 1
    assert rectangle.b == 0  # Re 100, geometry 2
    assert rectangle.c == 3  # Re 10, geometry 2
    assert rectangle.d == 2  # Re 100, geometry 1
    assert largest_jump_rectangle(reynolds, geometry_ids) == rectangle


def check_all_eight_assignments():
    expected_targets = {
        "AAA": "A",
        "AAB": "A",
        "ABA": "C",
        "ABB": "C",
        "BAA": "D",
        "BAB": "D",
        "BBA": "B",
        "BBB": "B",
    }
    assert len(SWAP_SPECS) == 8
    assert {spec.code for spec in SWAP_SPECS} == set(expected_targets)
    assert {spec.code: spec.target for spec in SWAP_SPECS} == expected_targets
    assert set(EXTRA_TRAINING_CODES) == {"AAB", "ABA", "BAB", "BBA"}


def check_run9_files():
    config = (ROOT / "configs/compositional/run9_no_pressure.yaml").read_text()
    shell = (ROOT / "runNova_run9_no_pressure.sh").read_text()
    model = (ROOT / "models/compositional/full_swap_ae.py").read_text()

    assert "lambda_fullswap: 0.1" in config
    assert "static_geometry: true" in config
    assert "lambda_bl: 0.0" in config
    assert "main_full_swap_no_pressure.py" in shell
    assert "full_swap_diagnostics_no_pressure.py" in shell
    assert "full_swap_matrix_no_pressure.py" in shell
    assert "full_swap_consistency" in model
    assert "lambda_fullswap" in model


def main():
    check_rectangle_enumeration()
    check_all_eight_assignments()
    check_run9_files()
    print("All Run 9 full-swap checks passed.")


if __name__ == "__main__":
    main()
