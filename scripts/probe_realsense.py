#!/usr/bin/env python3
"""Probe connected RealSense devices and optionally save a profile report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct execution from a checkout without requiring PYTHONPATH=src.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robot_grasp.errors import RobotGraspError
from robot_grasp.io_utils import dump_json
from robot_grasp.realsense import enumerate_realsense_devices, probe_realsense


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Probe one device's active stream calibration")
    parser.add_argument("--output", type=Path, help="Write JSON inventory/profile")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        data = (
            probe_realsense(args.serial, width=args.width, height=args.height, fps=args.fps)
            if args.serial
            else {"devices": enumerate_realsense_devices()}
        )
    except RobotGraspError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        dump_json(args.output, data)
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
