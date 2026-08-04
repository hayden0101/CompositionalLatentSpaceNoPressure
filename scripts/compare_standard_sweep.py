#!/usr/bin/env python3
"""Collect standard probe outputs for Runs 7 and 10-29 into one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def newest_version(root: Path):
    versions = [p for p in root.glob("version_*") if p.is_dir()]
    return max(versions, key=lambda p: p.stat().st_mtime) if versions else None


def read_probe(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        target = row.get("target") or row.get("Target")
        if not target:
            continue
        for block in ("z_mu", "z_g", "z_xi"):
            if block in row and row[block] not in (None, ""):
                result[f"{target}_{block}"] = row[block]
    return result


def main(start_run: int, end_run: int, output: Path):
    records = []
    for run in range(start_run, end_run + 1):
        root = Path(f"checkpoints/compositional-run{run}-no-pressure")
        version = newest_version(root)
        if version is None:
            continue
        probe = version / "probe_r2_test.csv"
        if not probe.exists():
            continue
        record = {"run": run, "version": str(version), **read_probe(probe)}
        record["diagnostics_log"] = str(version / "diagnostics.txt")
        records.append(record)

    if not records:
        raise SystemExit("No probe_r2_test.csv files found for the requested runs.")

    fields = ["run", "version"]
    fields.extend(sorted({key for row in records for key in row if key not in fields and key != "diagnostics_log"}))
    fields.append("diagnostics_log")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} runs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-run", type=int, default=7)
    parser.add_argument("--end-run", type=int, default=29)
    parser.add_argument("--output", type=Path, default=Path("standard_sweep_probes.csv"))
    args = parser.parse_args()
    main(args.start_run, args.end_run, args.output)
