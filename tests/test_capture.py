import json

import numpy as np
from PIL import Image
import pytest

from robot_grasp.capture import capture_sequence_from_manifest, load_capture_manifest
from robot_grasp.errors import ValidationError
from robot_grasp.sequence import validate_sequence


def _config(path, *, unknown_hardware=False):
    robot = {"model": "R1 with AR5L08/AR5R08", "serial": "r1-test"}
    gripper = {"model": "test-gripper", "serial": "gripper-test"}
    if unknown_hardware:
        robot = {"model": "unknown", "serial": "unknown"}
        gripper = {"model": "unknown", "serial": "unknown"}
    path.write_text(json.dumps({
        "schema_version": 2,
        "object_id": "cup_01",
        "depth_scale": 1000,
        "use_mask": False,
        "intrinsics": {"width": 4, "height": 3, "fx": 4, "fy": 4, "cx": 1.5, "cy": 1},
        "coordinate_frames": {"base": "base_link", "camera": "camera_color_optical_frame"},
        "hardware": {
            "camera": {"model": "Intel RealSense D455", "serial": "260722304986"},
            "robot": robot,
            "gripper": gripper,
        },
        "capture": {
            "timestamp_unit": "ns",
            "rgb_depth_sync": {"mode": "unknown", "clock": "device", "max_delta_ms": 5},
            "depth_to_color_registration": {"status": "registered", "method": "pyrealsense2.align"},
            "pose_source": {
                "type": "robot_fk_plus_hand_eye",
                "base_frame": "base_link",
                "hand_eye_calibration_ref": "calibration/hand_eye.json",
            },
        },
        "calibration": {},
    }), encoding="utf-8")


def _manifest(tmp_path, *, delta_ns=1_000_000):
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    Image.fromarray(np.full((3, 4), 500, dtype=np.uint16)).save(tmp_path / "depth.png")
    record = {
        "stem": "000001",
        "rgb": "rgb.png",
        "depth": "depth.png",
        "T_base_camera": np.eye(4).tolist(),
        "rgb_timestamp": {"sec": 10, "nanosec": 0},
        "depth_timestamp_ns": 10_000_000_000 + delta_ns,
        "pose_timestamp_ns": 10_000_000_000,
        "pose_source": "robot_fk_plus_hand_eye",
        "hand_eye_calibration_ref": "calibration/hand_eye.json",
        "robot_state": {"joint_angle_deg": [0, 0, 0, 0, 0, 0]},
    }
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_capture_manifest_accepts_ros_timestamp_and_converts(tmp_path):
    config = tmp_path / "capture.yaml"
    _config(config)
    manifest = _manifest(tmp_path)
    output = tmp_path / "sequence"

    report = capture_sequence_from_manifest(manifest, config, output)

    assert report["status"] == "valid"
    sequence = validate_sequence(output, use_mask=False, require_capture_metadata=True)
    frame = sequence.frames[0]
    assert frame.rgb_timestamp_ns == 10_000_000_000
    assert frame.depth_timestamp_ns == 10_001_000_000
    assert frame.pose_timestamp_ns == 10_000_000_000
    assert frame.provenance["pose_source"] == "robot_fk_plus_hand_eye"
    assert (output / "capture_manifest.json").is_file()
    assert (output / "frames.jsonl").is_file()


def test_capture_manifest_rejects_excessive_timestamp_delta(tmp_path):
    config = tmp_path / "capture.yaml"
    _config(config)
    manifest = _manifest(tmp_path, delta_ns=6_000_000)
    with pytest.raises(ValidationError, match="exceeds configured max_delta_ms"):
        capture_sequence_from_manifest(manifest, config, tmp_path / "sequence")


def test_capture_manifest_requires_pose_provenance(tmp_path):
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    Image.fromarray(np.full((3, 4), 500, dtype=np.uint16)).save(tmp_path / "depth.png")
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps({
        "stem": "0",
        "rgb": "rgb.png",
        "depth": "depth.png",
        "T_base_camera": np.eye(4).tolist(),
        "rgb_timestamp_ns": 1,
        "depth_timestamp_ns": 1,
        "pose_timestamp_ns": 1,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="pose_source must be non-empty"):
        load_capture_manifest(path)


def test_capture_sequence_rejects_unknown_hardware_in_strict_mode(tmp_path):
    config = tmp_path / "capture.yaml"
    _config(config, unknown_hardware=True)
    manifest = _manifest(tmp_path)
    with pytest.raises(ValidationError, match="must identify the real device"):
        capture_sequence_from_manifest(manifest, config, tmp_path / "sequence")


def test_capture_sequence_rejects_pending_pose_in_strict_mode(tmp_path):
    config = tmp_path / "capture.yaml"
    _config(config)
    manifest = _manifest(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["pose_source"] = "pending_robot_pose"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="resolved robot pose"):
        capture_sequence_from_manifest(manifest, config, tmp_path / "sequence")
