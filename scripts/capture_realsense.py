#!/usr/bin/env python3
"""Record raw RGB-D frames from one RealSense camera.

The output deliberately contains no fabricated robot pose. Join the emitted
``realsense_frames.jsonl`` with a timestamped robot/hand-eye pose stream before
running ``robot-grasp capture-sequence``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from a checkout without requiring PYTHONPATH=src.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robot_grasp.errors import RobotGraspError
from robot_grasp.realsense import capture_realsense_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="RealSense serial number")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--no-align", action="store_true", help="Do not align depth to the color stream")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = capture_realsense_frames(
            args.serial,
            args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frame_count=args.frames,
            warmup_frames=args.warmup_frames,
            align_depth_to_color=not args.no_align,
            overwrite=args.overwrite,
        )
    except RobotGraspError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
