#!/usr/bin/env python3
"""Print the newest ``last.ckpt`` under a run-specific checkpoint root."""

from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    args = parser.parse_args()

    root = args.checkpoint_root
    candidates = [path for path in root.rglob("last.ckpt") if path.is_file()]
    if not candidates:
        print(f"No last.ckpt found under {root}", file=sys.stderr)
        raise SystemExit(1)

    newest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    print(newest.resolve())


if __name__ == "__main__":
    main()
