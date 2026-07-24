#!/usr/bin/env python3
"""Summarize completed full-swap experiments into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

EXTRA_CODES = ("AAB", "ABA", "BAB", "BBA")
CROSS_CODES = ("ABB", "BAA")
RECON_CODES = ("AAA", "BBB")


def version_number(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_result_dir(checkpoint_root: Path):
    candidates = [
        path
        for path in checkpoint_root.glob("version_*")
        if path.is_dir() and (path / "full_swap_metrics.csv").is_file()
    ]
    return max(candidates, key=version_number) if candidates else None


def read_full_swap(path: Path):
    with path.open(newline="") as handle:
        return {row["code"]: row for row in csv.DictReader(handle)}


def read_probes(path: Path):
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row.get("target", ""): row for row in rows}


def average_metric(rows, codes, key):
    values = [float(rows[code][key]) for code in codes if code in rows]
    return mean(values) if values else float("nan")


def probe_value(probes, target, block):
    row = probes.get(target)
    if not row or block not in row:
        return float("nan")
    return float(row[block])


def summarize_run(root: Path, run: int):
    checkpoint_root = root / f"checkpoints/compositional-run{run}-no-pressure"
    result_dir = latest_result_dir(checkpoint_root)
    if result_dir is None:
        return None

    swap = read_full_swap(result_dir / "full_swap_metrics.csv")
    probes = read_probes(result_dir / "probe_r2_test.csv")
    geometry_targets = ("area_frac", "centroid_x", "centroid_y")
    geometry_scores = [
        probe_value(probes, target, "z_g") for target in geometry_targets
    ]
    geometry_scores = [value for value in geometry_scores if value == value]

    return {
        "run": run,
        "version": result_dir.name,
        "mean_extra_swap_ratio": average_metric(
            swap, EXTRA_CODES, "mean_ratio_to_reconstruction"
        ),
        "mean_cross_re_ratio": average_metric(
            swap, CROSS_CODES, "mean_ratio_to_reconstruction"
        ),
        "mean_reconstruction_ratio": average_metric(
            swap, RECON_CODES, "mean_ratio_to_reconstruction"
        ),
        "mean_all_swap_ratio": average_metric(
            swap, tuple(swap), "mean_ratio_to_reconstruction"
        ),
        "mean_extra_swap_mse": average_metric(swap, EXTRA_CODES, "mean_mse"),
        "log_re_from_z_mu": probe_value(probes, "log_re", "z_mu"),
        "log_re_from_z_g": probe_value(probes, "log_re", "z_g"),
        "log_re_from_z_xi": probe_value(probes, "log_re", "z_xi"),
        "mean_geometry_from_z_g": (
            mean(geometry_scores) if geometry_scores else float("nan")
        ),
        "result_directory": str(result_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--start-run", type=int, default=9)
    parser.add_argument("--end-run", type=int, default=29)
    parser.add_argument("--output", type=Path, default=Path("sweep_comparison.csv"))
    args = parser.parse_args()

    rows = []
    for run in range(args.start_run, args.end_run + 1):
        row = summarize_run(args.root, run)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No completed full-swap metrics were found.")

    rows.sort(key=lambda row: (row["mean_extra_swap_ratio"], row["mean_cross_re_ratio"]))
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'run':>4} {'extra ratio':>12} {'cross ratio':>12} "
        f"{'Re<-z_mu':>9} {'Re<-z_g':>8} {'geom<-z_g':>10}"
    )
    print("-" * 66)
    for row in rows:
        print(
            f"{row['run']:>4} "
            f"{row['mean_extra_swap_ratio']:>12.3f} "
            f"{row['mean_cross_re_ratio']:>12.3f} "
            f"{row['log_re_from_z_mu']:>9.3f} "
            f"{row['log_re_from_z_g']:>8.3f} "
            f"{row['mean_geometry_from_z_g']:>10.3f}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
