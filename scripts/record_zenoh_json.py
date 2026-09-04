#!/usr/bin/env python3
"""Record RPP robot JSON topics from Zenoh into timestamped JSONL."""

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
from robot_grasp.zenoh_adapter import ZenohJsonRecorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-group-name", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--zenoh-config", type=Path)
    parser.add_argument("--topic", action="append", dest="topics", help="Override default topic; repeatable")
    args = parser.parse_args(argv)
    try:
        report = ZenohJsonRecorder(
            device_group_name=args.device_group_name,
            side=args.side,
            zenoh_config=args.zenoh_config,
        ).record(args.output, duration_seconds=args.duration, topic_names=args.topics)
    except RobotGraspError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
