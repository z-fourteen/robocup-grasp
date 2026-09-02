from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .acceptance import pose_repeatability
from .errors import ValidationError
from .foundationpose_adapter import FoundationPoseAdapter, FoundationPoseConfig
from .io_utils import dump_json, prepare_output_dir
from .object_frame import load_named_transform
from .sequence import validate_sequence
from .transforms import compose_transform, invert_transform, rotation_angle_deg


def _pose_error(T_reference: np.ndarray, T_estimated: np.ndarray) -> tuple[float, float]:
    delta = compose_transform(invert_transform(T_reference), T_estimated)
    return float(np.linalg.norm(delta[:3, 3]) * 1000.0), rotation_angle_deg(delta[:3, :3])


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def evaluate_foundationpose_sequence(
    config_path: str | Path,
    sequence_dir: str | Path,
    object_frame_path: str | Path,
    output_dir: str | Path,
    *,
    frame_step: int = 1,
    max_frames: int | None = None,
    overwrite: bool = False,
    adapter_factory: Callable[[FoundationPoseConfig], FoundationPoseAdapter] = FoundationPoseAdapter,
) -> dict[str, Any]:
    """Run FoundationPose without pose initialization, then compare against held-back sequence GT."""
    if frame_step < 1 or (max_frames is not None and max_frames < 1):
        raise ValidationError("frame_step and max_frames must be positive integers.")
    sequence = validate_sequence(sequence_dir)
    frames = list(sequence.frames)[::frame_step]
    if max_frames is not None:
        frames = frames[:max_frames]
    config = FoundationPoseConfig.from_json(config_path)
    adapter = adapter_factory(config)
    T_object_model = load_named_transform(object_frame_path, "T_object_model")
    T_model_object = invert_transform(T_object_model)

    results: list[dict[str, Any]] = []
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    T_base_object_predictions: list[np.ndarray] = []
    for frame in frames:
        # The adapter receives only sensor inputs; GT is used only after estimate() returns.
        T_camera_object_predicted = adapter.estimate(
            frame.rgb_path,
            frame.depth_path,
            mask=frame.mask_path,
        )
        T_camera_model_gt = invert_transform(frame.T_base_camera)
        T_camera_object_gt = compose_transform(T_camera_model_gt, T_model_object)
        translation_error_mm, rotation_error_deg = _pose_error(
            T_camera_object_gt, T_camera_object_predicted
        )
        T_base_object_predicted = compose_transform(
            frame.T_base_camera, T_camera_object_predicted
        )
        translation_errors.append(translation_error_mm)
        rotation_errors.append(rotation_error_deg)
        T_base_object_predictions.append(T_base_object_predicted)
        results.append({
            "stem": frame.stem,
            "T_camera_object_predicted": T_camera_object_predicted.tolist(),
            "T_camera_object_ground_truth": T_camera_object_gt.tolist(),
            "T_base_object_predicted": T_base_object_predicted.tolist(),
            "translation_error_mm": translation_error_mm,
            "rotation_error_deg": rotation_error_deg,
        })

    output = prepare_output_dir(output_dir, overwrite=overwrite)
    pose_dir = output / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        dump_json(pose_dir / f"{result['stem']}.json", result)
    report = {
        "status": "completed",
        "backend": "FoundationPose",
        "sequence": str(sequence.root),
        "mesh": str(Path(config.mesh_path).expanduser().resolve()),
        "frame_count": len(results),
        "frame_step": frame_step,
        "mask_source": "sequence ground-truth visible-instance masks",
        "depth_source": sequence.metadata.get("source", {}).get("depth_source", "sequence depth"),
        "pose_input_policy": {
            "ground_truth_pose_passed_to_estimator": False,
            "estimator_inputs": ["rgb", "depth", "intrinsics", "mask", "mesh"],
            "ground_truth_used_for_scoring_timing": "after FoundationPose estimate() returned",
        },
        "translation_error_mm": _statistics(translation_errors),
        "rotation_error_deg": _statistics(rotation_errors),
        "base_frame_repeatability": pose_repeatability(T_base_object_predictions) if len(results) >= 2 else None,
        "expected_T_base_object": T_model_object.tolist(),
        "frames": results,
    }
    dump_json(output / "foundationpose_evaluation.json", report)
    dump_json(output / "T_camera_object_predictions.json", {
        "poses": [
            {"stem": item["stem"], "T_camera_object": item["T_camera_object_predicted"]}
            for item in results
        ]
    })
    dump_json(output / "acceptance_pose_samples.json", {
        "transform_name": "T_base_object",
        "poses": [
            {"stem": item["stem"], "T_base_object": item["T_base_object_predicted"]}
            for item in results
        ],
        "derivation": "T_base_object = sequence T_base_camera @ FoundationPose T_camera_object",
    })
    compose_dir = output / "compose_inputs"
    compose_dir.mkdir(parents=True, exist_ok=True)
    for frame, result in zip(frames, results):
        dump_json(compose_dir / f"{frame.stem}_camera_pose.json", {
            "T_base_camera": frame.T_base_camera.tolist(),
            "base_frame": "HouseCat6D original model frame used by the strict benchmark sequence",
        })
        dump_json(compose_dir / f"{frame.stem}_object_pose.json", {
            "T_camera_object": result["T_camera_object_predicted"],
            "source": "FoundationPose prediction; no ground-truth pose initialization",
        })
    return report
