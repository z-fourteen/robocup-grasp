import json

import numpy as np
import pytest

from robot_grasp.acceptance import load_named_pose_samples, pose_repeatability
from robot_grasp.errors import ValidationError
from robot_grasp.grasps import make_candidate
from robot_grasp.interfaces import GraspSelector


def pose(x=0.0, angle_deg=0.0):
    result = np.eye(4)
    angle = np.radians(angle_deg)
    result[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    result[0, 3] = x
    return result


def test_selector_uses_enabled_priority_and_reachability():
    low = make_candidate("low", np.eye(4), priority=1)
    high = make_candidate("high", np.eye(4), priority=10)
    disabled = make_candidate("disabled", np.eye(4), priority=20, enabled=False)
    selector = GraspSelector()
    assert selector.select([low, disabled, high])["id"] == "high"
    assert selector.select([low, high], reachability=lambda candidate: candidate["id"] == "low")["id"] == "low"
    with pytest.raises(ValidationError, match="No enabled and reachable"):
        selector.select([high], reachability=lambda _candidate: False)


def test_pose_repeatability_reports_worst_pair():
    stats = pose_repeatability([pose(0.0, 0.0), pose(0.001, 1.0), pose(0.003, 3.0)])
    assert stats["translation_mm"]["max_pairwise"] == pytest.approx(3.0)
    assert stats["rotation_deg"]["max_pairwise"] == pytest.approx(3.0)
    assert stats["pair_count"] == 3


def test_named_pose_samples_preserve_transform_semantics(tmp_path):
    samples_path = tmp_path / "pose_samples.json"
    samples_path.write_text(json.dumps({
        "transform_name": "T_base_object",
        "poses": [
            {"stem": "000000", "T_base_object": pose(0.0).tolist()},
            {"stem": "000001", "T_base_object": pose(0.001).tolist()},
        ],
    }), encoding="utf-8")

    transform_name, samples = load_named_pose_samples(samples_path)

    assert transform_name == "T_base_object"
    assert len(samples) == 2
    assert samples[1][0, 3] == pytest.approx(0.001)
