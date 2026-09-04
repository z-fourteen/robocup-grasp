import json

import numpy as np
from PIL import Image
import pytest

from robot_grasp.errors import ValidationError
from robot_grasp.sequence import validate_sequence


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def make_sequence(tmp_path):
    for name in ("rgb", "depth", "masks", "poses"):
        (tmp_path / name).mkdir()
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / "rgb" / "000.png")
    Image.fromarray(np.full((3, 4), 500, dtype=np.uint16)).save(tmp_path / "depth" / "000.png")
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(tmp_path / "masks" / "000.png")
    write_json(tmp_path / "poses" / "000.json", {"T_base_camera": np.eye(4).tolist()})
    write_json(tmp_path / "intrinsics.json", {"width": 4, "height": 3, "fx": 4, "fy": 4, "cx": 1.5, "cy": 1})
    write_json(tmp_path / "metadata.json", {
        "depth_scale": 1000.0,
        "object_id": "test_object",
        "coordinate_frames": {"base": "fixed world frame", "camera": "optical frame; poses are T_base_camera"},
    })


def test_valid_sequence(tmp_path):
    make_sequence(tmp_path)
    sequence = validate_sequence(tmp_path)
    assert len(sequence.frames) == 1
    assert sequence.depth_scale == 1000.0
    assert sequence.frames[0].valid_depth_ratio == 1.0


def test_unmatched_stem_has_actionable_error(tmp_path):
    make_sequence(tmp_path)
    (tmp_path / "masks" / "000.png").rename(tmp_path / "masks" / "wrong.png")
    with pytest.raises(ValidationError) as exc_info:
        validate_sequence(tmp_path)
    message = str(exc_info.value)
    assert "Frame '000' is missing ['mask']" in message
    assert "exact stem '000'" in message


def test_bad_pose_names_file_and_fix(tmp_path):
    make_sequence(tmp_path)
    write_json(tmp_path / "poses" / "000.json", {"T_camera_base": np.eye(4).tolist()})
    with pytest.raises(ValidationError) as exc_info:
        validate_sequence(tmp_path)
    assert str(tmp_path / "poses" / "000.json") in str(exc_info.value)
    assert "T_base_camera" in str(exc_info.value)


def test_image_size_mismatch_names_image(tmp_path):
    make_sequence(tmp_path)
    Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8)).save(tmp_path / "rgb" / "000.png")
    with pytest.raises(ValidationError, match="has size 4x2, expected 4x3"):
        validate_sequence(tmp_path)


def test_mask_is_optional_when_disabled(tmp_path):
    make_sequence(tmp_path)
    (tmp_path / "masks" / "000.png").unlink()
    (tmp_path / "masks").rmdir()

    sequence = validate_sequence(tmp_path, use_mask=False)

    assert sequence.frames[0].mask_path is None
    assert sequence.frames[0].valid_depth_ratio == 1.0


def test_optional_masks_are_ignored_when_disabled(tmp_path):
    make_sequence(tmp_path)
    (tmp_path / "masks" / "000.png").rename(tmp_path / "masks" / "stale.png")
    (tmp_path / "masks" / "not-an-image.txt").write_text("ignored", encoding="utf-8")

    sequence = validate_sequence(tmp_path, use_mask=False)

    assert sequence.frames[0].mask_path is None
    assert sequence.frames[0].valid_depth_ratio == 1.0


def test_mask_is_required_by_default(tmp_path):
    make_sequence(tmp_path)
    (tmp_path / "masks" / "000.png").unlink()
    with pytest.raises(ValidationError, match="Missing mask directory|No supported mask files"):
        validate_sequence(tmp_path)


def test_strict_capture_validation_rejects_missing_metadata(tmp_path):
    make_sequence(tmp_path)
    with pytest.raises(ValidationError, match="missing \['capture', 'hardware'\]"):
        validate_sequence(tmp_path, require_capture_metadata=True)


def test_strict_capture_validation_rejects_unknown_hardware(tmp_path):
    make_sequence(tmp_path)
    write_json(tmp_path / "metadata.json", {
        "schema_version": 2,
        "depth_scale": 1000.0,
        "object_id": "test_object",
        "coordinate_frames": {"base": "base_link", "camera": "camera_color_optical_frame"},
        "hardware": {
            "camera": {"model": "Intel RealSense D455", "serial": "260722304986"},
            "robot": {"model": "unknown", "serial": "unknown"},
            "gripper": {"model": "unknown", "serial": "unknown"},
        },
        "capture": {
            "timestamp_unit": "ns",
            "rgb_depth_sync": {"mode": "software", "clock": "device", "max_delta_ms": 5},
            "depth_to_color_registration": {"status": "registered", "method": "pyrealsense2.align"},
            "pose_source": {"type": "robot_fk_plus_hand_eye", "base_frame": "base_link", "hand_eye_calibration_ref": "calibration/hand_eye.json"},
        },
    })
    write_json(tmp_path / "poses" / "000.json", {
        "T_base_camera": np.eye(4).tolist(),
        "rgb_timestamp_ns": 1,
        "depth_timestamp_ns": 1,
        "pose_timestamp_ns": 1,
        "pose_source": "robot_fk_plus_hand_eye",
    })
    with pytest.raises(ValidationError, match="must identify the real device"):
        validate_sequence(tmp_path, require_capture_metadata=True)
