from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, TypeVar

from .errors import ValidationError
from .io_utils import load_yaml


@dataclass(frozen=True)
class ReconstructionConfig:
    voxel_length: float = 0.0025
    sdf_trunc: float = 0.01
    depth_min: float = 0.1
    depth_max: float = 1.5
    depth_quantile_low: float = 0.0
    depth_quantile_high: float = 1.0
    use_mask: bool = True
    min_component_triangles: int = 100
    min_component_ratio: float = 0.01
    collision_target_triangles: int = 5000
    min_valid_depth_ratio: float = 0.01

    def validate(self) -> None:
        numeric = (self.voxel_length, self.sdf_trunc, self.depth_min, self.depth_max,
                   self.depth_quantile_low, self.depth_quantile_high,
                   self.min_component_ratio, self.min_valid_depth_ratio)
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in numeric):
            raise ValidationError("Reconstruction lengths and ratios must be finite numeric values.")
        if self.voxel_length <= 0 or self.sdf_trunc <= 0:
            raise ValidationError("voxel_length and sdf_trunc must be positive lengths in meters.")
        if self.depth_min < 0 or self.depth_max <= self.depth_min:
            raise ValidationError("depth range is invalid: require 0 <= depth_min < depth_max, in meters.")
        if not 0 <= self.depth_quantile_low < self.depth_quantile_high <= 1:
            raise ValidationError(
                "depth quantiles are invalid: require 0 <= depth_quantile_low < depth_quantile_high <= 1."
            )
        counts = (self.min_component_triangles, self.collision_target_triangles)
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in counts):
            raise ValidationError("min_component_triangles and collision_target_triangles must be integers.")
        if self.min_component_triangles < 0 or self.collision_target_triangles < 4:
            raise ValidationError("min_component_triangles must be >= 0 and collision_target_triangles must be >= 4.")
        if not 0 <= self.min_component_ratio <= 1 or not 0 <= self.min_valid_depth_ratio <= 1:
            raise ValidationError("min_component_ratio and min_valid_depth_ratio must be within [0, 1].")
        if not isinstance(self.use_mask, bool):
            raise ValidationError("use_mask must be true or false.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceConfig:
    mesh_error_mm: float = 5.0
    pose_translation_mm: float = 5.0
    pose_rotation_deg: float = 5.0

    def validate(self) -> None:
        values = (self.mesh_error_mm, self.pose_translation_mm, self.pose_rotation_deg)
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values):
            raise ValidationError("Acceptance thresholds must be finite numeric values.")
        if min(values) < 0:
            raise ValidationError("Acceptance thresholds must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ConfigType = TypeVar("ConfigType", ReconstructionConfig, AcceptanceConfig)


def _load_dataclass(path: str | Path | None, config_type: type[ConfigType]) -> ConfigType:
    data = {} if path is None else load_yaml(path)
    if not isinstance(data, dict):
        raise ValidationError(f"Configuration {path} must be a YAML mapping.")
    allowed = {field.name for field in fields(config_type)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValidationError(f"Configuration {path} has unknown keys {unknown}. Allowed keys: {sorted(allowed)}.")
    try:
        config = config_type(**data)
    except TypeError as exc:
        raise ValidationError(f"Invalid configuration {path}: {exc}.") from exc
    config.validate()
    return config


def load_reconstruction_config(path: str | Path | None) -> ReconstructionConfig:
    return _load_dataclass(path, ReconstructionConfig)


def load_acceptance_config(path: str | Path | None) -> AcceptanceConfig:
    return _load_dataclass(path, AcceptanceConfig)
