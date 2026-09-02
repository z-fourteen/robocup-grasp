from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import ValidationError
from .io_utils import dump_json, load_json, prepare_output_dir
from .sequence import sequence_summary, validate_sequence
from .transforms import invert_transform, validate_transform


def _find_image(directory: Path, stem: str) -> Path:
    matches = sorted(path for path in directory.glob(f"{stem}.*") if path.is_file())
    if len(matches) != 1:
        raise ValidationError(
            f"Expected exactly one BOP image for stem {stem} in {directory}, found {len(matches)}."
        )
    return matches[0]


def _select_instance(
    annotations: list[dict[str, Any]],
    info: list[dict[str, Any]],
    object_id: int,
    frame_id: str,
) -> int:
    candidates = [index for index, annotation in enumerate(annotations) if int(annotation["obj_id"]) == object_id]
    if not candidates:
        raise ValidationError(f"BOP frame {frame_id} has no annotation for object id {object_id}.")
    return max(candidates, key=lambda index: float(info[index].get("visib_fract", 0.0)))


def import_bop_sequence(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    scene_id: int,
    object_id: int,
    frame_step: int = 1,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a BOP scene/object instance into the strict project sequence format."""
    if scene_id < 0 or object_id < 1:
        raise ValidationError("BOP scene_id must be non-negative and object_id must be positive.")
    if frame_step < 1 or (max_frames is not None and max_frames < 1):
        raise ValidationError("frame_step and max_frames must be positive integers.")
    dataset = Path(dataset_dir).resolve()
    scene = dataset / split / f"{scene_id:06d}"
    required = [scene / name for name in ("scene_camera.json", "scene_gt.json", "scene_gt_info.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValidationError(
            f"BOP scene {scene} is incomplete; missing {missing}. Extract the base and requested split archives first."
        )
    cameras = load_json(scene / "scene_camera.json")
    ground_truth = load_json(scene / "scene_gt.json")
    ground_truth_info = load_json(scene / "scene_gt_info.json")
    frame_ids = sorted(set(cameras) & set(ground_truth) & set(ground_truth_info), key=int)[::frame_step]
    frame_ids = [frame_id for frame_id in frame_ids if any(
        int(item["obj_id"]) == object_id for item in ground_truth[frame_id]
    )]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise ValidationError(f"No frames for object id {object_id} were found in BOP scene {scene}.")

    selected: list[dict[str, Any]] = []
    reference_K: np.ndarray | None = None
    reference_bop_depth_scale: float | None = None
    for frame_id in frame_ids:
        camera = cameras[frame_id]
        K = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
        bop_depth_scale = float(camera.get("depth_scale", 1.0))
        if reference_K is None:
            reference_K = K
            reference_bop_depth_scale = bop_depth_scale
        elif not np.allclose(K, reference_K, atol=1e-9, rtol=0.0):
            raise ValidationError(
                f"BOP intrinsics vary at frame {frame_id}; this sequence format requires one calibration. "
                "Import a constant-intrinsics subset."
            )
        elif not np.isclose(bop_depth_scale, reference_bop_depth_scale, atol=1e-12, rtol=0.0):
            raise ValidationError(
                f"BOP depth_scale varies at frame {frame_id}; import frames with one explicit depth unit."
            )
        instance_index = _select_instance(
            ground_truth[frame_id], ground_truth_info[frame_id], object_id, frame_id
        )
        annotation = ground_truth[frame_id][instance_index]
        T_camera_object = np.eye(4, dtype=np.float64)
        T_camera_object[:3, :3] = np.asarray(annotation["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        T_camera_object[:3, 3] = np.asarray(annotation["cam_t_m2c"], dtype=np.float64) / 1000.0
        T_camera_object = validate_transform(T_camera_object, name=f"BOP T_camera_object frame {frame_id}")
        selected.append({
            "frame_id": frame_id,
            "instance_index": instance_index,
            "rgb": _find_image(scene / "rgb", f"{int(frame_id):06d}"),
            "depth": _find_image(scene / "depth", f"{int(frame_id):06d}"),
            "mask": _find_image(scene / "mask_visib", f"{int(frame_id):06d}_{instance_index:06d}"),
            "T_base_camera": invert_transform(T_camera_object),
        })

    assert reference_K is not None and reference_bop_depth_scale is not None
    with Image.open(selected[0]["rgb"]) as image:
        width, height = image.size
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    for directory in ("rgb", "depth", "masks", "poses"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for item in selected:
        stem = f"{int(item['frame_id']):06d}"
        shutil.copy2(item["rgb"], output / "rgb" / f"{stem}{item['rgb'].suffix.lower()}")
        shutil.copy2(item["depth"], output / "depth" / f"{stem}{item['depth'].suffix.lower()}")
        shutil.copy2(item["mask"], output / "masks" / f"{stem}{item['mask'].suffix.lower()}")
        dump_json(output / "poses" / f"{stem}.json", {"T_base_camera": item["T_base_camera"].tolist()})

    dump_json(output / "intrinsics.json", {
        "width": width,
        "height": height,
        "fx": float(reference_K[0, 0]),
        "fy": float(reference_K[1, 1]),
        "cx": float(reference_K[0, 2]),
        "cy": float(reference_K[1, 2]),
    })
    project_depth_scale = 1000.0 / reference_bop_depth_scale
    dump_json(output / "metadata.json", {
        "depth_scale": project_depth_scale,
        "object_id": f"bop_{dataset.name}_obj_{object_id:06d}",
        "coordinate_frames": {
            "base": "BOP object/model frame in meters; fixed across the imported sequence.",
            "camera": "BOP OpenCV camera frame; poses are T_base_camera = inverse(T_camera_object).",
        },
        "source": {
            "format": "BOP scenewise",
            "dataset": str(dataset),
            "split": split,
            "scene_id": scene_id,
            "object_id": object_id,
            "bop_translation_unit": "millimeter",
            "bop_depth_scale_mm_per_raw_unit": reference_bop_depth_scale,
            "mask_type": "visible instance mask",
        },
    })
    validated = validate_sequence(output)
    report = sequence_summary(validated) | {
        "source_frame_count": len(frame_ids),
        "frame_step": frame_step,
        "T_base_camera_derivation": "inverse(BOP T_camera_object), with translation converted mm to m",
    }
    dump_json(output / "import_report.json", report)
    return report
