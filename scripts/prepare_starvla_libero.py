#!/usr/bin/env python3
"""Freeze original LIBERO training splits and statistics before StarVLA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oat.starvla_heading.data import ORIGINAL_SUITES, prepare_libero_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Original LIBERO HDF5 root; searched recursively")
    parser.add_argument("--output", required=True, type=Path,
                        help="New immutable JSON manifest path")
    parser.add_argument("--suites", nargs="+", choices=ORIGINAL_SUITES, default=list(ORIGINAL_SUITES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--expected-demos-per-task", type=int, default=50)
    parser.add_argument("--allow-incomplete-for-smoke-test", action="store_true",
                        help="Allow missing canonical tasks; never use for full experiments")
    args = parser.parse_args()
    manifest = prepare_libero_manifest(
        args.root, args.output, suites=args.suites, seed=args.seed, val_ratio=args.val_ratio,
        expected_demos_per_task=args.expected_demos_per_task,
        require_complete=not args.allow_incomplete_for_smoke_test,
    )
    print(json.dumps({"manifest": str(args.output.resolve()), "sha256": manifest["sha256"],
                      "tasks": len(manifest["tasks"]),
                      "train_episodes": manifest["train_episode_count"],
                      "val_episodes": manifest["val_episode_count"],
                      "train_frames": manifest["train_frame_count"],
                      "val_frames": manifest["val_frame_count"],
                      "heading_xy_rms": manifest["statistics"]["heading_xy_rms"]}, indent=2))


if __name__ == "__main__":
    main()
