from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ValidationError


def validate_transform(
    transform: Any,
    *,
    name: str = "transform",
    atol: float = 1e-6,
) -> np.ndarray:
    """Validate and return a 4x4 rigid transform named T_dst_src."""
    try:
        matrix = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must contain numeric values in a 4x4 array.") from exc
    if matrix.shape != (4, 4):
        raise ValidationError(f"{name} must have shape 4x4, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValidationError(f"{name} contains NaN or infinity; replace them with finite values.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValidationError(f"{name} last row must be [0, 0, 0, 1], got {matrix[3].tolist()}.")

    rotation = matrix[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > atol * 3 or not np.isclose(determinant, 1.0, atol=atol * 3):
        raise ValidationError(
            f"{name} rotation is invalid: orthogonality error={orthogonality_error:.3g}, "
            f"determinant={determinant:.6g}. Re-export a right-handed orthonormal rotation."
        )
    return matrix


def invert_transform(T_dst_src: Any) -> np.ndarray:
    matrix = validate_transform(T_dst_src, name="T_dst_src")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def compose_transform(*transforms: Any) -> np.ndarray:
    """Compose transforms in written matrix order, e.g. T_a_b @ T_b_c."""
    if not transforms:
        return np.eye(4, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    for index, transform in enumerate(transforms):
        result = result @ validate_transform(transform, name=f"transform[{index}]")
    return validate_transform(result, name="composed transform", atol=1e-5)


def transform_points(T_dst_src: Any, points_src: Any) -> np.ndarray:
    matrix = validate_transform(T_dst_src, name="T_dst_src")
    points = np.asarray(points_src, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValidationError(f"points_src must have shape Nx3, got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValidationError("points_src contains NaN or infinity; provide finite coordinates in meters.")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def compose_base_grasp(T_base_camera: Any, T_camera_object: Any, T_object_grasp: Any) -> np.ndarray:
    """Return T_base_grasp = T_base_camera @ T_camera_object @ T_object_grasp."""
    return compose_transform(T_base_camera, T_camera_object, T_object_grasp)


def rotation_angle_deg(rotation: Any) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
