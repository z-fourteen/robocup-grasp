import json
from pathlib import Path

import numpy as np

from robot_grasp.foundationpose_evaluation import evaluate_foundationpose_sequence


class RecordingAdapter:
    calls = []

    def __init__(self, config):
        self.config = config

    def estimate(self, rgb, depth, *, mask=None, intrinsics=None):
        self.calls.append({"rgb": rgb, "depth": depth, "mask": mask, "intrinsics": intrinsics})
        return np.eye(4)


def test_evaluation_does_not_pass_ground_truth_pose_to_estimator(tmp_path):
    project = Path(__file__).parents[1]
    intrinsics = project / "examples/minimal_sequence/intrinsics.json"
    config = tmp_path / "foundationpose.json"
    config.write_text(json.dumps({
        "foundationpose_dir": "/not/loaded/by/fake",
        "mesh_path": "/not/loaded/by/fake/mesh.ply",
        "intrinsics_path": str(intrinsics),
        "depth_scale": 1000.0,
    }), encoding="utf-8")
    object_frame = tmp_path / "object_frame.json"
    object_frame.write_text(json.dumps({"T_object_model": np.eye(4).tolist()}), encoding="utf-8")
    RecordingAdapter.calls = []

    report = evaluate_foundationpose_sequence(
        config,
        project / "examples/minimal_sequence",
        object_frame,
        tmp_path / "output",
        adapter_factory=RecordingAdapter,
    )

    assert report["pose_input_policy"]["ground_truth_pose_passed_to_estimator"] is False
    assert len(RecordingAdapter.calls) == 1
    assert set(RecordingAdapter.calls[0]) == {"rgb", "depth", "mask", "intrinsics"}
    assert RecordingAdapter.calls[0]["intrinsics"] is None

    acceptance_samples = json.loads(
        (tmp_path / "output/acceptance_pose_samples.json").read_text(encoding="utf-8")
    )
    assert acceptance_samples["transform_name"] == "T_base_object"
    assert set(acceptance_samples["poses"][0]) == {"stem", "T_base_object"}
    assert (tmp_path / "output/compose_inputs/000_camera_pose.json").is_file()
    assert (tmp_path / "output/compose_inputs/000_object_pose.json").is_file()
