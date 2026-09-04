"""Device-neutral RGB-D capture contracts and raw-manifest conversion.

Vendor SDK processes should only normalize their output into the JSONL format
accepted here. This keeps timestamp, pose, and provenance checks independent of
RealSense, Zenoh, ROS, or a specific robot controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import numpy as np

from .errors import ValidationError
from .io_utils import dump_json, load_yaml, prepare_output_dir
from .sequence import validate_sequence
from .transforms import validate_transform


_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}


@dataclass(frozen=True)
class CaptureFrame:
    """One synchronized RGB/depth/robot observation from a capture manifest."""

    stem: str
    rgb_path: Path
    depth_path: Path
    T_base_camera: np.ndarray
    rgb_timestamp_ns: int
    depth_timestamp_ns: int
    pose_timestamp_ns: int
    pose_source: str
    mask_path: Path | None = None
    hand_eye_calibration_ref: str | None = None
    robot_state: dict[str, Any] | None = None

    @property
    def rgb_depth_delta_ms(self) -> float:
        return abs(self.rgb_timestamp_ns - self.depth_timestamp_ns) / 1_000_000.0

    @property
    def sensor_pose_delta_ms(self) -> float:
        sensor_timestamp = (self.rgb_timestamp_ns + self.depth_timestamp_ns) / 2.0
        return abs(sensor_timestamp - self.pose_timestamp_ns) / 1_000_000.0


def _timestamp_ns(value: Any, *, label: str) -> int:
    if isinstance(value, dict):
        if "sec" not in value or "nanosec" not in value:
            raise ValidationError(f"{label} must contain sec and nanosec.")
        try:
            sec = int(value["sec"])
            nanosec = int(value["nanosec"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label}.sec and {label}.nanosec must be integers.") from exc
        if sec < 0 or not 0 <= nanosec < 1_000_000_000:
            raise ValidationError(f"{label} has invalid sec/nanosec range.")
        return sec * 1_000_000_000 + nanosec
    if isinstance(value, bool):
        raise ValidationError(f"{label} must be a non-negative integer nanosecond timestamp.")
    try:
        timestamp = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be a non-negative integer nanosecond timestamp.") from exc
    if timestamp < 0:
        raise ValidationError(f"{label} must be non-negative.")
    return timestamp


def _resolve_source(path_value: Any, *, manifest_path: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValidationError(f"Capture manifest {manifest_path} {label} must be a non-empty path.")
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValidationError(f"Capture manifest {manifest_path} {label} does not exist: {path}.")
    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise ValidationError(f"Capture manifest {manifest_path} {label} must be an image file: {path}.")
    return path


def _validate_intrinsics(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("Capture config intrinsics must be a JSON/YAML object.")
    required = ("width", "height", "fx", "fy", "cx", "cy")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValidationError(f"Capture config intrinsics is missing {missing}.")
    try:
        width, height = int(data["width"]), int(data["height"])
        values = {key: float(data[key]) for key in ("fx", "fy", "cx", "cy")}
    except (TypeError, ValueError) as exc:
        raise ValidationError("Capture config intrinsics fields must be numeric.") from exc
    if width <= 0 or height <= 0 or values["fx"] <= 0 or values["fy"] <= 0:
        raise ValidationError("Capture config width, height, fx and fy must be positive.")
    if not all(np.isfinite(value) for value in values.values()):
        raise ValidationError("Capture config intrinsics must be finite.")
    return {"width": width, "height": height, **values}


def _validate_device_block(data: Any, *, label: str, require_complete: bool = False) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError(f"Capture config hardware.{label} must be an object.")
    for key in ("model", "serial"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValidationError(f"Capture config hardware.{label}.{key} must be a non-empty string.")
        if require_complete and data[key].strip().lower() in {"unknown", "needs_runtime_confirmation", "pending"}:
            raise ValidationError(
                f"Capture config hardware.{label}.{key} must identify the real device; got {data[key]!r}."
            )
    return dict(data)


def validate_capture_config(data: Any, *, require_complete: bool = False) -> dict[str, Any]:
    """Validate and normalize the profile used by a strict capture import."""
    if not isinstance(data, dict):
        raise ValidationError("Capture config must be a YAML mapping.")
    required = ("object_id", "depth_scale", "intrinsics", "coordinate_frames", "hardware", "capture")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValidationError(f"Capture config is missing {missing}.")
    if not isinstance(data["object_id"], str) or not data["object_id"].strip():
        raise ValidationError("Capture config object_id must be a non-empty string.")
    try:
        depth_scale = float(data["depth_scale"])
    except (TypeError, ValueError) as exc:
        raise ValidationError("Capture config depth_scale must be positive raw units per meter.") from exc
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValidationError("Capture config depth_scale must be positive raw units per meter.")
    intrinsics = _validate_intrinsics(data["intrinsics"])
    frames = data["coordinate_frames"]
    if not isinstance(frames, dict) or not all(isinstance(frames.get(key), str) and frames[key].strip() for key in ("base", "camera")):
        raise ValidationError("Capture config coordinate_frames must define non-empty base and camera names.")
    hardware = data["hardware"]
    if not isinstance(hardware, dict):
        raise ValidationError("Capture config hardware must be an object.")
    normalized_hardware = {
        label: _validate_device_block(hardware.get(label), label=label, require_complete=require_complete)
        for label in ("camera", "robot", "gripper")
    }
    normalized_hardware.update({key: value for key, value in hardware.items() if key not in normalized_hardware})

    capture = data["capture"]
    if not isinstance(capture, dict):
        raise ValidationError("Capture config capture must be an object.")
    sync = capture.get("rgb_depth_sync")
    if not isinstance(sync, dict) or sync.get("mode") not in {"hardware", "software", "unknown"}:
        raise ValidationError("Capture config capture.rgb_depth_sync.mode must be hardware, software, or unknown.")
    if sync.get("clock") not in {"device", "host", "unknown"}:
        raise ValidationError("Capture config capture.rgb_depth_sync.clock must be device, host, or unknown.")
    try:
        max_delta_ms = float(sync.get("max_delta_ms"))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Capture config capture.rgb_depth_sync.max_delta_ms must be non-negative.") from exc
    if not np.isfinite(max_delta_ms) or max_delta_ms < 0:
        raise ValidationError("Capture config capture.rgb_depth_sync.max_delta_ms must be non-negative.")
    registration = capture.get("depth_to_color_registration")
    if not isinstance(registration, dict) or registration.get("status") not in {"registered", "unregistered", "unknown"}:
        raise ValidationError("Capture config capture.depth_to_color_registration.status is invalid.")
    if not isinstance(registration.get("method"), str) or not registration["method"].strip():
        raise ValidationError("Capture config capture.depth_to_color_registration.method must be non-empty.")
    pose_source = capture.get("pose_source")
    if not isinstance(pose_source, dict) or not isinstance(pose_source.get("type"), str) or not pose_source["type"].strip():
        raise ValidationError("Capture config capture.pose_source.type must be non-empty.")
    if not isinstance(pose_source.get("base_frame"), str) or not pose_source["base_frame"].strip():
        raise ValidationError("Capture config capture.pose_source.base_frame must be non-empty.")
    if pose_source["type"] == "robot_fk_plus_hand_eye" and not isinstance(pose_source.get("hand_eye_calibration_ref"), str):
        raise ValidationError("Capture config robot_fk_plus_hand_eye requires hand_eye_calibration_ref.")
    if require_complete and pose_source["type"].strip().lower() in {"pending", "pending_robot_pose", "unknown"}:
        raise ValidationError(
            "Capture config capture.pose_source.type must describe a resolved robot pose source; "
            f"got {pose_source['type']!r}."
        )
    return {
        "schema_version": int(data.get("schema_version", 2)),
        "object_id": data["object_id"],
        "depth_scale": depth_scale,
        "intrinsics": intrinsics,
        "coordinate_frames": dict(frames),
        "hardware": normalized_hardware,
        "capture": dict(capture),
        "use_mask": bool(data.get("use_mask", False)),
        "calibration": dict(data.get("calibration", {})),
    }


def load_capture_config(path: str | Path, *, require_complete: bool = False) -> dict[str, Any]:
    return validate_capture_config(load_yaml(path), require_complete=require_complete)


def load_capture_manifest(path: str | Path) -> tuple[CaptureFrame, ...]:
    """Read a JSONL manifest emitted by a camera/robot adapter."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ValidationError(f"Missing capture manifest: {manifest_path}.")
    frames: list[CaptureFrame] = []
    seen: set[str] = set()
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"Cannot read capture manifest {manifest_path}: {exc}.") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON in {manifest_path}:{line_number}: {exc.msg}.") from exc
        if not isinstance(record, dict):
            raise ValidationError(f"Capture manifest {manifest_path}:{line_number} must be a JSON object.")
        stem = record.get("stem", record.get("frame_id"))
        if not isinstance(stem, str) or not _STEM_RE.fullmatch(stem):
            raise ValidationError(f"Capture manifest {manifest_path}:{line_number} has an invalid stem/frame_id.")
        if stem in seen:
            raise ValidationError(f"Capture manifest {manifest_path} repeats frame '{stem}'.")
        seen.add(stem)
        rgb_path = _resolve_source(record.get("rgb"), manifest_path=manifest_path, label=f"line {line_number} rgb")
        depth_path = _resolve_source(record.get("depth"), manifest_path=manifest_path, label=f"line {line_number} depth")
        mask_path = None
        if record.get("mask") is not None:
            mask_path = _resolve_source(record["mask"], manifest_path=manifest_path, label=f"line {line_number} mask")
        try:
            transform = validate_transform(record["T_base_camera"], name=f"T_base_camera line {line_number}")
        except KeyError as exc:
            raise ValidationError(f"Capture manifest {manifest_path}:{line_number} is missing T_base_camera.") from exc
        pose_source = record.get("pose_source")
        if not isinstance(pose_source, str) or not pose_source.strip():
            raise ValidationError(f"Capture manifest {manifest_path}:{line_number} pose_source must be non-empty.")
        frames.append(CaptureFrame(
            stem=stem,
            rgb_path=rgb_path,
            depth_path=depth_path,
            T_base_camera=transform,
            rgb_timestamp_ns=_timestamp_ns(record.get("rgb_timestamp_ns", record.get("rgb_timestamp")), label=f"line {line_number} rgb_timestamp"),
            depth_timestamp_ns=_timestamp_ns(record.get("depth_timestamp_ns", record.get("depth_timestamp")), label=f"line {line_number} depth_timestamp"),
            pose_timestamp_ns=_timestamp_ns(record.get("pose_timestamp_ns", record.get("pose_timestamp")), label=f"line {line_number} pose_timestamp"),
            pose_source=pose_source,
            mask_path=mask_path,
            hand_eye_calibration_ref=record.get("hand_eye_calibration_ref"),
            robot_state=record.get("robot_state"),
        ))
    if not frames:
        raise ValidationError(f"Capture manifest {manifest_path} contains no frame records.")
    return tuple(frames)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def capture_sequence_from_manifest(
    manifest_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert normalized device records into a validated strict sequence."""
    config = load_capture_config(config_path, require_complete=True)
    frames = load_capture_manifest(manifest_path)
    max_delta_ms = float(config["capture"]["rgb_depth_sync"]["max_delta_ms"])
    for frame in frames:
        if frame.pose_source.strip().lower() in {"pending", "pending_robot_pose", "unknown"}:
            raise ValidationError(
                f"Frame {frame.stem} pose_source must describe a resolved robot pose; got {frame.pose_source!r}."
            )
        if frame.rgb_depth_delta_ms > max_delta_ms:
            raise ValidationError(
                f"Frame {frame.stem} RGB/depth timestamp delta {frame.rgb_depth_delta_ms:.3f} ms exceeds "
                f"configured max_delta_ms {max_delta_ms:.3f}."
            )
        if frame.mask_path is not None and config["use_mask"] is False:
            continue
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    for directory in ("rgb", "depth", "poses"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    has_masks = any(frame.mask_path is not None for frame in frames)
    if has_masks:
        (output / "masks").mkdir(parents=True, exist_ok=True)

    copied_manifest: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    for frame in frames:
        rgb_name = f"{frame.stem}{frame.rgb_path.suffix.lower()}"
        depth_name = f"{frame.stem}{frame.depth_path.suffix.lower()}"
        rgb_destination = output / "rgb" / rgb_name
        depth_destination = output / "depth" / depth_name
        shutil.copy2(frame.rgb_path, rgb_destination)
        shutil.copy2(frame.depth_path, depth_destination)
        mask_destination = None
        if frame.mask_path is not None:
            mask_destination = output / "masks" / f"{frame.stem}{frame.mask_path.suffix.lower()}"
            shutil.copy2(frame.mask_path, mask_destination)
        pose_record: dict[str, Any] = {
            "T_base_camera": frame.T_base_camera.tolist(),
            "frame_id": frame.stem,
            "rgb_timestamp_ns": frame.rgb_timestamp_ns,
            "depth_timestamp_ns": frame.depth_timestamp_ns,
            "pose_timestamp_ns": frame.pose_timestamp_ns,
            "rgb_depth_delta_ms": frame.rgb_depth_delta_ms,
            "sensor_pose_delta_ms": frame.sensor_pose_delta_ms,
            "pose_source": frame.pose_source,
        }
        if frame.hand_eye_calibration_ref is not None:
            pose_record["hand_eye_calibration_ref"] = frame.hand_eye_calibration_ref
        if frame.robot_state is not None:
            pose_record["robot_state"] = frame.robot_state
        dump_json(output / "poses" / f"{frame.stem}.json", pose_record)
        frame_record = {
            "stem": frame.stem,
            "rgb": f"rgb/{rgb_name}",
            "depth": f"depth/{depth_name}",
            "mask": f"masks/{mask_destination.name}" if mask_destination is not None else None,
            "rgb_timestamp_ns": frame.rgb_timestamp_ns,
            "depth_timestamp_ns": frame.depth_timestamp_ns,
            "pose_timestamp_ns": frame.pose_timestamp_ns,
            "rgb_depth_delta_ms": frame.rgb_depth_delta_ms,
            "sensor_pose_delta_ms": frame.sensor_pose_delta_ms,
            "pose_source": frame.pose_source,
        }
        frame_records.append(frame_record)
        for label, source, destination in (
            ("rgb", frame.rgb_path, rgb_destination),
            ("depth", frame.depth_path, depth_destination),
            ("mask", frame.mask_path, mask_destination),
        ):
            if source is not None and destination is not None:
                copied_manifest.append({
                    "kind": label,
                    "path": destination.relative_to(output).as_posix(),
                    "source": str(source),
                    "size_bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                })

    dump_json(output / "intrinsics.json", config["intrinsics"])
    metadata = {
        "schema_version": 2,
        "depth_scale": config["depth_scale"],
        "object_id": config["object_id"],
        "coordinate_frames": config["coordinate_frames"],
        "hardware": config["hardware"],
        "capture": config["capture"] | {"frame_count": len(frames)},
        "calibration": config["calibration"],
    }
    dump_json(output / "metadata.json", metadata)
    _write_jsonl(output / "frames.jsonl", frame_records)
    dump_json(output / "capture_manifest.json", {
        "source_manifest": str(Path(manifest_path).resolve()),
        "files": copied_manifest,
    })
    deltas = [frame.rgb_depth_delta_ms for frame in frames]
    report = {
        "status": "valid",
        "frame_count": len(frames),
        "rgb_depth_delta_ms": {
            "max": float(max(deltas)),
            "mean": float(np.mean(deltas)),
            "configured_max": max_delta_ms,
        },
        "output": str(output),
        "use_mask": config["use_mask"],
    }
    dump_json(output / "capture_report.json", report)
    validate_sequence(
        output,
        use_mask=config["use_mask"],
        require_capture_metadata=True,
    )
    return report
