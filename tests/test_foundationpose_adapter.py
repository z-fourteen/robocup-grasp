import json

import numpy as np
import pytest

from robot_grasp.errors import OptionalDependencyError
from robot_grasp.foundationpose_adapter import FoundationPoseAdapter, FoundationPoseConfig


class FakeEstimator:
    def __init__(self):
        self.depth = None
        self.mask = None

    def register(self, *, K, rgb, depth, ob_mask, iteration):
        self.depth = depth
        self.mask = ob_mask
        T_object_camera = np.eye(4)
        T_object_camera[2, 3] = -1.0
        return T_object_camera


def make_adapter(tmp_path, **changes):
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text(json.dumps({"fx": 100, "fy": 100, "cx": 1, "cy": 1}), encoding="utf-8")
    values = {
        "foundationpose_dir": str(tmp_path / "missing_foundationpose"),
        "mesh_path": str(tmp_path / "mesh.ply"),
        "intrinsics_path": str(intrinsics),
        "depth_scale": 1000.0,
        "output_convention": "T_object_camera",
    }
    values.update(changes)
    return FoundationPoseAdapter(FoundationPoseConfig.from_dict(values))


def test_adapter_scales_depth_and_converts_output_direction(tmp_path):
    adapter = make_adapter(tmp_path)
    fake = FakeEstimator()
    adapter._estimator = fake
    pose = adapter.estimate(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2), 500, dtype=np.uint16),
        mask=np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )
    np.testing.assert_allclose(fake.depth, 0.5)
    np.testing.assert_array_equal(fake.mask, [[False, True], [True, False]])
    np.testing.assert_allclose(pose[:3, 3], [0, 0, 1])


def test_missing_foundationpose_directory_has_install_hint(tmp_path):
    adapter = make_adapter(tmp_path)
    with pytest.raises(OptionalDependencyError, match="not downloaded automatically"):
        adapter._load_external_estimator()
