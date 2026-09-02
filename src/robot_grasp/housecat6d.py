from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image, ImageFilter

from .errors import ValidationError
from .io_utils import dump_json, prepare_output_dir
from .sequence import sequence_summary, validate_sequence
from .transforms import invert_transform, rotation_angle_deg, validate_transform


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Read HouseCat6D numpy annotations without allowing arbitrary globals."""

    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise ValidationError(
                f"HouseCat6D annotation requests unsupported pickle global {module}.{name}. "
                "Use an official annotation file containing only numpy arrays and built-in containers."
            )
        return super().find_class(module, name)


def _load_annotation(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            data = _NumpyOnlyUnpickler(stream).load()
    except ValidationError:
        raise
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
        raise ValidationError(
            f"Could not read HouseCat6D annotation {path}: {exc}. Re-extract the official archive."
        ) from exc
    if not isinstance(data, dict):
        raise ValidationError(f"HouseCat6D annotation {path} must contain a dictionary.")
    required = {"model_list", "instance_ids", "rotations", "translations"}
    missing = sorted(required - set(data))
    if missing:
        raise ValidationError(
            f"HouseCat6D annotation {path} is missing keys {missing}. Re-extract the official archive."
        )
    return data


def _parse_scene_objects(path: Path) -> dict[str, tuple[int, int]]:
    objects: dict[str, tuple[int, int]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"Could not read HouseCat6D scene metadata {path}: {exc}.") from exc
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 3:
            raise ValidationError(
                f"Invalid HouseCat6D metadata line {path}:{line_number}: expected "
                "'<instance_id> <class_id> <object_name>'."
            )
        try:
            instance_id, class_id = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise ValidationError(
                f"Invalid numeric ids at {path}:{line_number}; instance_id and class_id must be integers."
            ) from exc
        objects[fields[2]] = (instance_id, class_id)
    return objects


def _transform_error(reference: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    delta = invert_transform(reference) @ measured
    return float(np.linalg.norm(delta[:3, 3])), rotation_angle_deg(delta[:3, :3])


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels == 0:
        return mask
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.MinFilter(2 * pixels + 1))) > 0


def _copy_image(source: Path, destination: Path) -> None:
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        raise ValidationError(f"Could not copy {source} to {destination}: {exc}.") from exc


def import_housecat6d_sequence(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    scene_name: str,
    object_name: str,
    depth_source: str = "depth",
    frame_step: int = 1,
    max_frames: int | None = None,
    min_mask_pixels: int = 64,
    mask_erosion_pixels: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one HouseCat6D object track to the strict object-centric sequence format."""
    if frame_step < 1 or (max_frames is not None and max_frames < 1):
        raise ValidationError("frame_step and max_frames must be positive integers.")
    if min_mask_pixels < 1:
        raise ValidationError("min_mask_pixels must be a positive integer.")
    if mask_erosion_pixels < 0:
        raise ValidationError("mask_erosion_pixels cannot be negative.")
    if depth_source not in {"depth", "depth_gt"}:
        raise ValidationError("depth_source must be either 'depth' or 'depth_gt'.")

    dataset = Path(dataset_dir).resolve()
    scene = dataset / scene_name
    required = [scene / name for name in ("rgb", depth_source, "instance", "labels", "camera_pose")]
    required += [scene / "intrinsics.txt", scene / "meta.txt", scene / "obj_pose_final"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValidationError(
            f"HouseCat6D scene {scene} is incomplete; missing {missing}. Re-extract val_scene.zip."
        )

    objects = _parse_scene_objects(scene / "meta.txt")
    if object_name not in objects:
        raise ValidationError(
            f"Object {object_name!r} is not listed in {scene / 'meta.txt'}. "
            f"Choose one of {sorted(objects)}."
        )
    instance_id, class_id = objects[object_name]
    object_pose_path = scene / "obj_pose_final" / f"{object_name}.txt"
    model_path = dataset / "obj_models_small_size_final" / object_name.split("-", 1)[0] / f"{object_name}.obj"
    if not object_pose_path.is_file():
        raise ValidationError(
            f"Missing fixed object pose {object_pose_path}. Re-extract the HouseCat6D validation archive."
        )
    if not model_path.is_file():
        raise ValidationError(
            f"Missing object model {model_path}. Extract obj_models.zip next to the scene directories."
        )

    try:
        intrinsics = np.loadtxt(scene / "intrinsics.txt", dtype=np.float64)
        T_world_object = validate_transform(
            np.loadtxt(object_pose_path, dtype=np.float64), name=f"T_world_object in {object_pose_path}"
        )
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Could not parse HouseCat6D calibration or object pose in {scene}: {exc}.") from exc
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ValidationError(
            f"HouseCat6D intrinsics {scene / 'intrinsics.txt'} must be a finite 3x3 matrix, got {intrinsics.shape}."
        )

    rgb_paths = sorted((scene / "rgb").glob("*.png"))
    stems = [path.stem for path in rgb_paths][::frame_step]
    selected: list[dict[str, Any]] = []
    rejected_small_masks = 0
    relation_translation_errors: list[float] = []
    relation_rotation_errors: list[float] = []
    image_size: tuple[int, int] | None = None
    for stem in stems:
        paths = {
            "rgb": scene / "rgb" / f"{stem}.png",
            "depth": scene / depth_source / f"{stem}.png",
            "instance": scene / "instance" / f"{stem}.png",
            "label": scene / "labels" / f"{stem}_label.pkl",
            "camera_pose": scene / "camera_pose" / f"{stem}.txt",
        }
        absent = [str(path) for path in paths.values() if not path.is_file()]
        if absent:
            raise ValidationError(
                f"HouseCat6D frame {stem} is incomplete; missing {absent}. Re-extract val_scene.zip."
            )
        with Image.open(paths["rgb"]) as image:
            current_size = image.size
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValidationError(
                f"HouseCat6D RGB size changes at {paths['rgb']}: {current_size} versus {image_size}."
            )
        with Image.open(paths["instance"]) as image:
            instance_map = np.asarray(image)
        if instance_map.ndim != 2 or instance_map.shape != (current_size[1], current_size[0]):
            raise ValidationError(
                f"HouseCat6D instance image {paths['instance']} has shape {instance_map.shape}; "
                f"expected {(current_size[1], current_size[0])}."
            )
        mask = _erode_mask(instance_map == instance_id, mask_erosion_pixels)
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < min_mask_pixels:
            rejected_small_masks += 1
            continue

        annotation = _load_annotation(paths["label"])
        try:
            annotation_index = list(annotation["model_list"]).index(object_name)
        except ValueError as exc:
            raise ValidationError(
                f"HouseCat6D annotation {paths['label']} does not contain object {object_name!r}."
            ) from exc
        if int(annotation["instance_ids"][annotation_index]) != instance_id:
            raise ValidationError(
                f"HouseCat6D annotation {paths['label']} maps {object_name!r} to instance "
                f"{annotation['instance_ids'][annotation_index]}, but {scene / 'meta.txt'} says {instance_id}."
            )
        T_camera_object = np.eye(4, dtype=np.float64)
        T_camera_object[:3, :3] = np.asarray(annotation["rotations"][annotation_index], dtype=np.float64)
        T_camera_object[:3, 3] = np.asarray(annotation["translations"][annotation_index], dtype=np.float64)
        T_camera_object = validate_transform(
            T_camera_object, name=f"T_camera_object in {paths['label']}", atol=1e-5
        )
        try:
            T_world_camera = validate_transform(
                np.loadtxt(paths["camera_pose"], dtype=np.float64),
                name=f"T_world_camera in {paths['camera_pose']}",
            )
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Could not parse HouseCat6D camera pose {paths['camera_pose']}: {exc}.") from exc
        expected_T_camera_object = invert_transform(T_world_camera) @ T_world_object
        translation_error, rotation_error = _transform_error(expected_T_camera_object, T_camera_object)
        relation_translation_errors.append(translation_error)
        relation_rotation_errors.append(rotation_error)
        if translation_error > 1e-5 or rotation_error > 1e-3:
            raise ValidationError(
                f"HouseCat6D transform convention check failed at frame {stem}: "
                f"inverse(T_world_camera) @ T_world_object differs from label T_camera_object by "
                f"{translation_error:.6g} m and {rotation_error:.6g} deg. Check dataset version and pose direction."
            )
        selected.append({
            "stem": stem,
            "paths": paths,
            "mask": mask,
            "mask_pixels": mask_pixels,
            "T_base_camera": invert_transform(T_camera_object),
        })
        if max_frames is not None and len(selected) >= max_frames:
            break

    if not selected:
        raise ValidationError(
            f"No usable frames for {object_name!r} in {scene}. Lower --min-mask-pixels or check instance masks."
        )
    assert image_size is not None
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    for directory in ("rgb", "depth", "masks", "poses"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for item in selected:
        stem = item["stem"]
        _copy_image(item["paths"]["rgb"], output / "rgb" / f"{stem}.png")
        _copy_image(item["paths"]["depth"], output / "depth" / f"{stem}.png")
        Image.fromarray(np.where(item["mask"], 255, 0).astype(np.uint8), mode="L").save(
            output / "masks" / f"{stem}.png"
        )
        dump_json(output / "poses" / f"{stem}.json", {"T_base_camera": item["T_base_camera"].tolist()})

    dump_json(output / "intrinsics.json", {
        "width": image_size[0],
        "height": image_size[1],
        "fx": float(intrinsics[0, 0]),
        "fy": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
    })
    dump_json(output / "metadata.json", {
        "depth_scale": 1000.0,
        "object_id": f"housecat6d_{object_name}",
        "coordinate_frames": {
            "base": "HouseCat6D object/model frame in meters; fixed across this object-centric sequence.",
            "camera": "OpenCV optical frame; poses are T_base_camera = inverse(label T_camera_object).",
        },
        "source": {
            "format": "HouseCat6D validation scene",
            "dataset": str(dataset),
            "scene": scene_name,
            "object_name": object_name,
            "instance_id": instance_id,
            "class_id": class_id,
            "depth_source": depth_source,
            "depth_unit": "millimeter",
            "mask_type": "visible instance mask from instance/<stem>.png",
            "object_model": str(model_path),
            "ground_truth_usage": "offline TSDF registration and evaluation only; never an ObjectPoseEstimator input",
        },
    })
    validated = validate_sequence(output)
    report = sequence_summary(validated) | {
        "source_frame_count": len(rgb_paths),
        "candidate_frame_count": len(stems),
        "rejected_small_masks": rejected_small_masks,
        "frame_step": frame_step,
        "max_frames": max_frames,
        "min_mask_pixels": min_mask_pixels,
        "mask_erosion_pixels": mask_erosion_pixels,
        "depth_source": depth_source,
        "T_base_camera_derivation": "inverse(HouseCat6D label T_camera_object)",
        "coordinate_convention_check": {
            "equation": "T_camera_object = inverse(T_world_camera) @ T_world_object",
            "max_translation_error_m": float(max(relation_translation_errors)),
            "max_rotation_error_deg": float(max(relation_rotation_errors)),
        },
        "source_object_model": str(model_path),
    }
    dump_json(output / "import_report.json", report)
    return report
