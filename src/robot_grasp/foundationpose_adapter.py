from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

from .errors import OptionalDependencyError, ValidationError
from .interfaces import ObjectPoseEstimator
from .io_utils import load_json
from .transforms import invert_transform, validate_transform


@dataclass(frozen=True)
class FoundationPoseConfig:
    foundationpose_dir: str
    mesh_path: str
    intrinsics_path: str
    mask_path: str | None
    depth_scale: float
    output_convention: str = "T_camera_object"
    module: str = "estimater"
    register_iterations: int = 5
    debug_dir: str = "/tmp/foundationpose_debug"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FoundationPoseConfig":
        required = {"foundationpose_dir", "mesh_path", "intrinsics_path", "depth_scale"}
        missing = sorted(required - set(data))
        if missing:
            raise ValidationError(f"FoundationPose config is missing {missing}. Add explicit paths and depth_scale.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValidationError(f"FoundationPose config has unknown keys {unknown}. Allowed keys: {sorted(allowed)}.")
        config = cls(mask_path=None, **data) if "mask_path" not in data else cls(**data)
        if (
            not isinstance(config.depth_scale, (int, float))
            or isinstance(config.depth_scale, bool)
            or config.depth_scale <= 0
            or not np.isfinite(config.depth_scale)
        ):
            raise ValidationError("FoundationPose depth_scale must be a finite positive number of raw units per meter.")
        if config.output_convention not in {"T_camera_object", "T_object_camera"}:
            raise ValidationError("FoundationPose output_convention must be T_camera_object or T_object_camera.")
        if not isinstance(config.register_iterations, int) or isinstance(config.register_iterations, bool) or config.register_iterations < 1:
            raise ValidationError("FoundationPose register_iterations must be at least 1.")
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "FoundationPoseConfig":
        data = load_json(path)
        if not isinstance(data, dict):
            raise ValidationError(f"FoundationPose config {path} must contain a JSON object.")
        return cls.from_dict(data)


class FoundationPoseAdapter(ObjectPoseEstimator):
    """Lazy adapter around the external NVIDIA FoundationPose repository."""

    def __init__(self, config: FoundationPoseConfig):
        self.config = config
        self._estimator = None
        self._K = self._load_intrinsics(config.intrinsics_path)

    @staticmethod
    def _load_intrinsics(path: str | Path) -> np.ndarray:
        data = load_json(path)
        if isinstance(data, dict) and "K" in data:
            matrix = np.asarray(data["K"], dtype=np.float64)
        elif isinstance(data, dict) and all(key in data for key in ("fx", "fy", "cx", "cy")):
            matrix = np.array(
                [[data["fx"], 0.0, data["cx"]], [0.0, data["fy"], data["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        else:
            raise ValidationError(f"Intrinsics {path} must contain K or fx, fy, cx, cy.")
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValidationError(f"Intrinsics {path} do not define a valid finite 3x3 camera matrix.")
        return matrix

    def _load_external_estimator(self):
        repo = Path(self.config.foundationpose_dir).expanduser().resolve()
        mesh_path = Path(self.config.mesh_path).expanduser().resolve()
        if not repo.is_dir():
            raise OptionalDependencyError(
                f"FoundationPose directory not found: {repo}. Clone/install FoundationPose separately and set "
                "foundationpose_dir in the adapter config; large models are not downloaded automatically."
            )
        if not mesh_path.is_file():
            raise ValidationError(f"FoundationPose mesh not found: {mesh_path}. Set mesh_path to the object-frame mesh in meters.")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            module = importlib.import_module(self.config.module)
            import trimesh
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise OptionalDependencyError(
                f"FoundationPose dependency '{exc.name}' is unavailable. Follow the installation instructions in "
                f"{repo} and run in that environment; then verify foundationpose_dir/module in the adapter config."
            ) from exc
        required = ("FoundationPose", "ScorePredictor", "PoseRefinePredictor")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise OptionalDependencyError(
                f"FoundationPose module '{self.config.module}' from {repo} is missing {missing}. "
                "Set 'module' to the repository module exposing the standard FoundationPose API."
            )
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if mesh.is_empty:
            raise ValidationError(f"FoundationPose mesh is empty: {mesh_path}.")
        scorer = module.ScorePredictor()
        refiner = module.PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        return module.FoundationPose(
            model_pts=np.asarray(mesh.vertices),
            model_normals=np.asarray(mesh.vertex_normals),
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
            debug_dir=self.config.debug_dir,
        )

    @staticmethod
    def _as_array(value: Any, mode: str | None = None) -> np.ndarray:
        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                if mode is not None:
                    image = image.convert(mode)
                return np.asarray(image)
        return np.asarray(value)

    def estimate(self, rgb: Any, depth: Any, *, mask: Any = None, intrinsics: Any = None) -> np.ndarray:
        if self._estimator is None:
            self._estimator = self._load_external_estimator()
        color = np.array(self._as_array(rgb, "RGB"), dtype=np.uint8, copy=True)
        depth_m = self._as_array(depth).astype(np.float32) / self.config.depth_scale
        mask_source = mask if mask is not None else self.config.mask_path
        if mask_source is None:
            raise ValidationError("FoundationPose requires an object mask. Pass mask=... or configure mask_path.")
        object_mask = self._as_array(mask_source)
        if object_mask.ndim == 3:
            object_mask = object_mask[..., 0]
        if depth_m.ndim != 2 or color.shape[:2] != depth_m.shape or object_mask.shape != depth_m.shape:
            raise ValidationError(
                f"FoundationPose RGB/depth/mask sizes must match; got {color.shape}, {depth_m.shape}, {object_mask.shape}."
            )
        depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = 0.0
        K = self._K if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValidationError(
                f"FoundationPose intrinsics must be a finite 3x3 matrix with positive focal lengths, got {K}."
            )
        external_pose = self._estimator.register(
            K=K,
            rgb=np.ascontiguousarray(color),
            depth=np.ascontiguousarray(depth_m),
            ob_mask=np.ascontiguousarray(object_mask > 0),
            iteration=self.config.register_iterations,
        )
        pose = validate_transform(external_pose, name=f"FoundationPose {self.config.output_convention} output")
        if self.config.output_convention == "T_object_camera":
            pose = invert_transform(pose)
        return validate_transform(pose, name="T_camera_object")
