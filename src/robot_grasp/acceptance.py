from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .config import AcceptanceConfig
from .errors import RobotGraspError, ValidationError
from .io_utils import dump_json, load_json, prepare_output_dir
from .open3d_utils import require_open3d
from .transforms import rotation_angle_deg, validate_transform


def load_named_pose_samples(path: str | Path) -> tuple[str, list[np.ndarray]]:
    data = load_json(path)
    transform_name = "T_camera_object"
    if isinstance(data, dict):
        if "poses" not in data:
            raise ValidationError(f"Pose samples {path} must contain a 'poses' array.")
        transform_name = data.get("transform_name", transform_name)
        data = data["poses"]
    if not isinstance(transform_name, str) or not transform_name.startswith("T_"):
        raise ValidationError(
            f"Pose samples {path} transform_name must use T_dst_src naming, got {transform_name!r}."
        )
    if not isinstance(data, list) or len(data) < 2:
        raise ValidationError(f"Pose samples {path} must contain at least two {transform_name} results.")
    result = []
    for index, item in enumerate(data):
        if isinstance(item, dict):
            if transform_name not in item:
                raise ValidationError(f"Pose sample {index} in {path} must contain {transform_name!r}.")
            item = item[transform_name]
        result.append(validate_transform(item, name=f"{transform_name} sample {index} in {path}"))
    return transform_name, result


def load_pose_samples(path: str | Path) -> list[np.ndarray]:
    return load_named_pose_samples(path)[1]


def pose_repeatability(poses: list[np.ndarray]) -> dict[str, Any]:
    if len(poses) < 2:
        raise ValidationError("At least two poses are required for repeatability statistics.")
    translation_deltas = []
    rotation_deltas = []
    for first, second in combinations(poses, 2):
        translation_deltas.append(float(np.linalg.norm(first[:3, 3] - second[:3, 3]) * 1000.0))
        relative_rotation = first[:3, :3].T @ second[:3, :3]
        rotation_deltas.append(rotation_angle_deg(relative_rotation))
    translations = np.asarray(translation_deltas)
    rotations = np.asarray(rotation_deltas)
    return {
        "sample_count": len(poses),
        "pair_count": len(translation_deltas),
        "translation_mm": {
            "mean_pairwise": float(np.mean(translations)),
            "std_pairwise": float(np.std(translations)),
            "max_pairwise": float(np.max(translations)),
        },
        "rotation_deg": {
            "mean_pairwise": float(np.mean(rotations)),
            "std_pairwise": float(np.std(rotations)),
            "max_pairwise": float(np.max(rotations)),
        },
    }


def mesh_dimension_comparison(mesh_path: str | Path, caliper_path: str | Path) -> dict[str, Any]:
    data = load_json(caliper_path)
    if not isinstance(data, dict) or data.get("length_unit") != "meter" or "dimensions" not in data:
        raise ValidationError(
            f"Caliper file {caliper_path} must contain length_unit='meter' and dimensions with x, y, z."
        )
    dimensions = data["dimensions"]
    if not isinstance(dimensions, dict) or set(dimensions) != {"x", "y", "z"}:
        raise ValidationError(f"Caliper dimensions in {caliper_path} must have exactly x, y, z keys.")
    measured = np.asarray([dimensions[axis] for axis in "xyz"], dtype=np.float64)
    if measured.shape != (3,) or not np.all(np.isfinite(measured)) or np.any(measured <= 0):
        raise ValidationError(f"Caliper dimensions in {caliper_path} must be finite positive lengths in meters.")
    o3d = require_open3d("mesh dimension acceptance")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if mesh.is_empty():
        raise RobotGraspError(f"Could not read mesh for acceptance: {mesh_path}.")
    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    errors_mm = np.abs(extent - measured) * 1000.0
    return {
        "mesh": str(mesh_path),
        "axis_order": ["x", "y", "z"],
        "mesh_dimensions_m": extent.tolist(),
        "caliper_dimensions_m": measured.tolist(),
        "absolute_error_mm": errors_mm.tolist(),
        "max_error_mm": float(np.max(errors_mm)),
    }


def run_acceptance(
    mesh_path: str | Path,
    caliper_path: str | Path,
    poses_path: str | Path,
    output_dir: str | Path,
    config: AcceptanceConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    mesh_result = mesh_dimension_comparison(mesh_path, caliper_path)
    transform_name, pose_samples = load_named_pose_samples(poses_path)
    pose_result = pose_repeatability(pose_samples) | {"transform_name": transform_name}
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    checks = {
        "mesh_dimensions": mesh_result["max_error_mm"] <= config.mesh_error_mm,
        "pose_translation": pose_result["translation_mm"]["max_pairwise"] <= config.pose_translation_mm,
        "pose_rotation": pose_result["rotation_deg"]["max_pairwise"] <= config.pose_rotation_deg,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": config.to_dict(),
        "mesh_dimensions": mesh_result,
        "pose_repeatability": pose_result,
    }
    dump_json(output / "acceptance_report.json", report)
    return report
