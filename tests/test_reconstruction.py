import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from robot_grasp.config import ReconstructionConfig
from robot_grasp.errors import RobotGraspError
from robot_grasp.reconstruction import reconstruct_sequence


o3d = pytest.importorskip("open3d")


def _look_at(position: np.ndarray, target: np.ndarray = np.zeros(3)) -> np.ndarray:
    z_axis = target - position
    z_axis = z_axis / np.linalg.norm(z_axis)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(z_axis, up))) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = position
    return transform


def _write_synthetic_box_sequence(root: Path, *, with_masks: bool = True, empty_depth: bool = False) -> None:
    width = height = 48
    intrinsics = {"width": width, "height": height, "fx": 100.0, "fy": 100.0, "cx": 23.5, "cy": 23.5}
    for name in ("rgb", "depth", "poses"):
        (root / name).mkdir(parents=True)
    if with_masks:
        (root / "masks").mkdir()

    box = o3d.geometry.TriangleMesh.create_box(width=0.08, height=0.08, depth=0.08)
    box.translate((-0.04, -0.04, -0.04))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(box))
    K = np.array(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    positions = (
        np.array([0.0, 0.0, -0.22]),
        np.array([0.0, 0.0, 0.22]),
        np.array([-0.22, 0.0, 0.0]),
        np.array([0.22, 0.0, 0.0]),
        np.array([0.0, -0.22, 0.0]),
        np.array([0.0, 0.22, 0.0]),
    )
    for index, position in enumerate(positions):
        T_base_camera = _look_at(position)
        T_camera_base = np.linalg.inv(T_base_camera)
        rays = scene.create_rays_pinhole(
            o3d.core.Tensor(K),
            o3d.core.Tensor(T_camera_base.astype(np.float32)),
            width,
            height,
        )
        depth_m = scene.cast_rays(rays)["t_hit"].numpy()
        visible = np.isfinite(depth_m)
        if empty_depth:
            depth_m = np.zeros_like(depth_m)
        else:
            depth_m = np.where(visible, depth_m, 0.0)
        stem = f"{index:03d}"
        Image.fromarray(np.full((height, width, 3), [180, 100, 40], dtype=np.uint8)).save(root / "rgb" / f"{stem}.png")
        Image.fromarray(np.rint(depth_m * 1000.0).astype(np.uint16)).save(root / "depth" / f"{stem}.png")
        if with_masks:
            Image.fromarray(np.where(visible, 255, 0).astype(np.uint8)).save(root / "masks" / f"{stem}.png")
        (root / "poses" / f"{stem}.json").write_text(
            json.dumps({"T_base_camera": T_base_camera.tolist()}), encoding="utf-8"
        )
    (root / "intrinsics.json").write_text(json.dumps(intrinsics), encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "depth_scale": 1000.0,
                "object_id": "synthetic_box",
                "coordinate_frames": {"base": "fixed base frame", "camera": "optical frame; poses are T_base_camera"},
            }
        ),
        encoding="utf-8",
    )


def _synthetic_config(*, use_mask: bool = True) -> ReconstructionConfig:
    return ReconstructionConfig(
        voxel_length=0.005,
        sdf_trunc=0.015,
        use_mask=use_mask,
        min_component_triangles=0,
        min_component_ratio=0.0,
        collision_target_triangles=5000,
    )


def test_synthetic_tsdf_reconstructs_known_box_and_records_context(tmp_path):
    sequence = tmp_path / "sequence"
    _write_synthetic_box_sequence(sequence)

    report = reconstruct_sequence(sequence, tmp_path / "output", _synthetic_config(), overwrite=False)

    extent = np.asarray(report["bounding_box"]["extent_m"])
    np.testing.assert_allclose(extent, [0.08, 0.08, 0.08], atol=0.015)
    assert report["valid_depth_ratio"]["denominator"] == "mask_pixels"
    assert report["valid_depth_ratio"]["mean"] == pytest.approx(1.0)
    assert report["config_snapshot"] == _synthetic_config().to_dict()
    assert report["code"]["package"] == "robot_grasp"
    assert report["code"]["version"]
    assert report["dependencies"]["open3d"] == o3d.__version__
    assert report["runtime"]["elapsed_seconds"] >= 0.0
    manifest_paths = {item["path"] for item in report["input_manifest"]}
    assert "metadata.json" in manifest_paths
    assert "depth/000.png" in manifest_paths
    assert all(len(item["sha256"]) == 64 for item in report["input_manifest"])


def test_synthetic_tsdf_supports_missing_masks_when_disabled(tmp_path):
    sequence = tmp_path / "sequence"
    _write_synthetic_box_sequence(sequence, with_masks=False)

    report = reconstruct_sequence(sequence, tmp_path / "output", _synthetic_config(use_mask=False))

    assert report["valid_depth_ratio"]["denominator"] == "image_pixels"
    assert not any(item["path"].startswith("masks/") for item in report["input_manifest"])


def test_synthetic_empty_depth_fails_with_actionable_error(tmp_path):
    sequence = tmp_path / "sequence"
    _write_synthetic_box_sequence(sequence, empty_depth=True)

    with pytest.raises(RobotGraspError, match="TSDF fusion produced empty geometry"):
        reconstruct_sequence(
            sequence,
            tmp_path / "output",
            ReconstructionConfig(min_valid_depth_ratio=0.0),
        )
