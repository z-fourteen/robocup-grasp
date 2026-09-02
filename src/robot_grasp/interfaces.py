from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .errors import ValidationError
from .grasps import load_grasps
from .io_utils import load_json
from .transforms import validate_transform


class CameraPoseProvider(ABC):
    """Robot-independent source of T_base_camera."""

    @abstractmethod
    def get_T_base_camera(self) -> np.ndarray:
        raise NotImplementedError


class ObjectPoseEstimator(ABC):
    """Image-based source of T_camera_object."""

    @abstractmethod
    def estimate(self, rgb: Any, depth: Any, *, mask: Any = None, intrinsics: Any = None) -> np.ndarray:
        raise NotImplementedError


class GraspCandidateProvider(ABC):
    @abstractmethod
    def get_candidates(self) -> list[dict[str, Any]]:
        raise NotImplementedError


class JsonCameraPoseProvider(CameraPoseProvider):
    def __init__(self, path: str | Path):
        data = load_json(path)
        if isinstance(data, dict):
            if "T_base_camera" not in data:
                raise ValidationError(f"Camera pose file {path} must contain 'T_base_camera'.")
            data = data["T_base_camera"]
        self._transform = validate_transform(data, name=f"T_base_camera in {path}")

    def get_T_base_camera(self) -> np.ndarray:
        return self._transform.copy()


class JsonObjectPoseEstimator(ObjectPoseEstimator):
    """Deterministic pose source for replay/dry-run, not an image estimator."""

    def __init__(self, path: str | Path):
        data = load_json(path)
        if isinstance(data, dict):
            if "T_camera_object" not in data:
                raise ValidationError(f"Object pose file {path} must contain 'T_camera_object'.")
            data = data["T_camera_object"]
        self._transform = validate_transform(data, name=f"T_camera_object in {path}")

    def estimate(self, rgb: Any = None, depth: Any = None, *, mask: Any = None, intrinsics: Any = None) -> np.ndarray:
        return self._transform.copy()


class JsonGraspCandidateProvider(GraspCandidateProvider):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def get_candidates(self) -> list[dict[str, Any]]:
        return load_grasps(self.path)["candidates"]


ReachabilityCallback = Callable[[dict[str, Any]], bool]


class GraspSelector:
    """Select the highest-priority enabled, optionally reachable candidate."""

    def select(
        self,
        candidates: Sequence[dict[str, Any]],
        reachability: ReachabilityCallback | None = None,
    ) -> dict[str, Any]:
        eligible = [candidate for candidate in candidates if candidate.get("enabled") is True]
        if reachability is not None:
            eligible = [candidate for candidate in eligible if reachability(candidate)]
        if not eligible:
            reason = "enabled and reachable" if reachability is not None else "enabled"
            raise ValidationError(f"No {reason} grasp candidate is available. Enable or annotate a usable candidate.")
        return max(eligible, key=lambda candidate: (int(candidate["priority"]), candidate["id"]))
