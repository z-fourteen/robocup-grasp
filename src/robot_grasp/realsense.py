"""Optional Intel RealSense inventory and RGB-D recording adapter."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import ValidationError
from .io_utils import dump_json, prepare_output_dir


def _load_sdk() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ValidationError(
            "RealSense Python SDK is unavailable. Install pyrealsense2 or run this adapter with "
            "the system/ROS Python environment that provides it."
        ) from exc
    return rs


def _safe_info(device: Any, rs: Any, key: Any) -> str | None:
    try:
        return str(device.get_info(key))
    except RuntimeError:
        return None


def _device_info(device: Any, rs: Any) -> dict[str, Any]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "product_line": rs.camera_info.product_line,
        "usb_type": rs.camera_info.usb_type_descriptor,
        "physical_port": rs.camera_info.physical_port,
        "product_id": rs.camera_info.product_id,
        "connection_type": rs.camera_info.connection_type,
    }
    result = {label: _safe_info(device, rs, key) for label, key in fields.items()}
    result["depth_scale_m_per_raw_unit"] = float(device.first_depth_sensor().get_depth_scale())
    return result


def enumerate_realsense_devices() -> list[dict[str, Any]]:
    """Return device identity and depth-scale information without starting streams."""
    rs = _load_sdk()
    return [_device_info(device, rs) for device in rs.context().query_devices()]


def _select_device(context: Any, rs: Any, serial: str) -> Any:
    devices = list(context.query_devices())
    if not devices:
        raise ValidationError("No RealSense device is connected.")
    matches = [device for device in devices if _safe_info(device, rs, rs.camera_info.serial_number) == serial]
    if not matches:
        available = [
            f"{_safe_info(device, rs, rs.camera_info.name)}:{_safe_info(device, rs, rs.camera_info.serial_number)}"
            for device in devices
        ]
        raise ValidationError(f"RealSense serial {serial} was not found. Available devices: {available}.")
    return matches[0]


def _intrinsics(profile: Any) -> dict[str, Any]:
    data = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": int(data.width),
        "height": int(data.height),
        "fx": float(data.fx),
        "fy": float(data.fy),
        "cx": float(data.ppx),
        "cy": float(data.ppy),
        "distortion_model": str(data.model),
        "distortion_coefficients": [float(value) for value in data.coeffs],
    }


def probe_realsense(
    serial: str,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    align_depth_to_color: bool = True,
    warmup_frames: int = 5,
) -> dict[str, Any]:
    """Start one stream profile briefly and return actual calibration metadata."""
    rs = _load_sdk()
    context = rs.context()
    _select_device(context, rs, serial)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        color_profile = profile.get_stream(rs.stream.color)
        depth_profile = profile.get_stream(rs.stream.depth)
        if warmup_frames < 1:
            raise ValidationError("warmup_frames must be positive.")
        timestamp_samples: list[tuple[float, float]] = []
        color_frame = depth_frame = None
        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise ValidationError(f"RealSense {serial} did not produce both color and depth frames.")
            timestamp_samples.append((float(color_frame.get_timestamp()), float(depth_frame.get_timestamp())))
        extrinsics = depth_profile.get_extrinsics_to(color_profile)
        deltas = [abs(color_ms - depth_ms) for color_ms, depth_ms in timestamp_samples]
        assert color_frame is not None and depth_frame is not None
        return {
            "queried_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": _device_info(_select_device(context, rs, serial), rs),
            "streams": {
                "color": _intrinsics(color_profile),
                "depth": _intrinsics(depth_profile),
                "fps": fps,
            },
            "depth_to_color_extrinsics": {
                "rotation_row_major": [float(value) for value in extrinsics.rotation],
                "translation_m": [float(value) for value in extrinsics.translation],
            },
            "timestamp": {
                "color_ms": float(color_frame.get_timestamp()),
                "depth_ms": float(depth_frame.get_timestamp()),
                "delta_ms": abs(float(color_frame.get_timestamp()) - float(depth_frame.get_timestamp())),
                "sample_count": len(deltas),
                "delta_ms_min": float(min(deltas)),
                "delta_ms_mean": float(np.mean(deltas)),
                "delta_ms_max": float(max(deltas)),
                "domain": str(color_frame.get_frame_timestamp_domain()),
            },
            "registration": {
                "status": "registered" if align_depth_to_color else "unregistered",
                "method": "pyrealsense2.align" if align_depth_to_color else "none",
            },
        }
    except RuntimeError as exc:
        raise ValidationError(f"Could not probe RealSense {serial}: {exc}.") from exc
    finally:
        if started:
            pipeline.stop()


def capture_realsense_frames(
    serial: str,
    output_dir: str | Path,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    frame_count: int = 30,
    warmup_frames: int = 5,
    align_depth_to_color: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Record raw RealSense frames and a timestamp-only JSONL manifest.

    Robot poses are intentionally not fabricated here. The resulting manifest
    must be joined with a robot/hand-eye pose stream before strict conversion.
    """
    if frame_count <= 0 or warmup_frames < 0:
        raise ValidationError("frame_count must be positive and warmup_frames must be non-negative.")
    rs = _load_sdk()
    context = rs.context()
    _select_device(context, rs, serial)
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    align = rs.align(rs.stream.color) if align_depth_to_color else None
    records: list[dict[str, Any]] = []
    active_profile = None
    color_profile = depth_profile = None
    timestamp_samples: list[tuple[float, float]] = []
    timestamp_domain = "unknown"
    started = False
    try:
        active_profile = pipeline.start(config)
        started = True
        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames(5000)
            if align is not None:
                frames = align.process(frames)
            if not frames.get_color_frame() or not frames.get_depth_frame():
                raise ValidationError(f"RealSense {serial} dropped a warm-up color/depth frame.")
        color_profile = active_profile.get_stream(rs.stream.color)
        depth_profile = active_profile.get_stream(rs.stream.depth)
        for index in range(frame_count):
            frames = pipeline.wait_for_frames(5000)
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise ValidationError(f"RealSense {serial} dropped color/depth frame {index}.")
            stem = f"{index:06d}"
            Image.fromarray(np.asanyarray(color_frame.get_data())).save(output / "rgb" / f"{stem}.png")
            Image.fromarray(np.asanyarray(depth_frame.get_data()).astype(np.uint16)).save(output / "depth" / f"{stem}.png")
            color_timestamp_ns = int(round(float(color_frame.get_timestamp()) * 1_000_000.0))
            depth_timestamp_ns = int(round(float(depth_frame.get_timestamp()) * 1_000_000.0))
            timestamp_samples.append((float(color_frame.get_timestamp()), float(depth_frame.get_timestamp())))
            timestamp_domain = str(color_frame.get_frame_timestamp_domain())
            records.append({
                "stem": stem,
                "rgb": f"rgb/{stem}.png",
                "depth": f"depth/{stem}.png",
                "rgb_timestamp_ns": color_timestamp_ns,
                "depth_timestamp_ns": depth_timestamp_ns,
                "rgb_depth_delta_ms": abs(color_timestamp_ns - depth_timestamp_ns) / 1_000_000.0,
                "rgb_frame_number": int(color_frame.get_frame_number()),
                "depth_frame_number": int(depth_frame.get_frame_number()),
            })
    except RuntimeError as exc:
        raise ValidationError(f"Could not capture RealSense {serial}: {exc}.") from exc
    finally:
        if started:
            pipeline.stop()

    if active_profile is None or color_profile is None or depth_profile is None or not timestamp_samples:
        raise ValidationError("RealSense capture ended without an active stream profile or frames.")
    camera = _device_info(active_profile.get_device(), rs)
    extrinsics = depth_profile.get_extrinsics_to(color_profile)
    last_color_ms, last_depth_ms = timestamp_samples[-1]
    deltas = [abs(color_ms - depth_ms) for color_ms, depth_ms in timestamp_samples]
    profile = {
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": camera,
        "streams": {
            "color": _intrinsics(color_profile),
            "depth": _intrinsics(depth_profile),
            "fps": fps,
        },
        "depth_to_color_extrinsics": {
            "rotation_row_major": [float(value) for value in extrinsics.rotation],
            "translation_m": [float(value) for value in extrinsics.translation],
        },
        "timestamp": {
            "color_ms": last_color_ms,
            "depth_ms": last_depth_ms,
            "delta_ms": abs(last_color_ms - last_depth_ms),
            "sample_count": len(deltas),
            "delta_ms_min": float(min(deltas)),
            "delta_ms_mean": float(np.mean(deltas)),
            "delta_ms_max": float(max(deltas)),
            "domain": timestamp_domain,
        },
        "registration": {
            "status": "registered" if align_depth_to_color else "unregistered",
            "method": "pyrealsense2.align" if align_depth_to_color else "none",
        },
    }
    metadata = {
        "schema_version": 2,
        "capture_stage": "raw_rgbd",
        "depth_scale": 1.0 / float(camera["depth_scale_m_per_raw_unit"]),
        "hardware": {"camera": camera},
        "capture": {
            "timestamp_unit": "ns",
            "rgb_depth_sync": {
                "mode": "unknown",
                "clock": "device",
                "max_delta_ms": max(record["rgb_depth_delta_ms"] for record in records),
            },
            "depth_to_color_registration": profile["registration"],
            "pose_source": {"type": "pending_robot_pose", "base_frame": "base_link"},
        },
    }
    dump_json(output / "camera_profile.json", profile)
    dump_json(output / "intrinsics.json", profile["streams"]["color"])
    dump_json(output / "camera_metadata.json", metadata)
    with (output / "realsense_frames.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    report = {
        "status": "captured",
        "frame_count": len(records),
        "serial": serial,
        "output": str(output),
        "max_rgb_depth_delta_ms": max(record["rgb_depth_delta_ms"] for record in records),
        "warmup_frames_discarded": warmup_frames,
        "pose_status": "pending_robot_pose_join",
    }
    dump_json(output / "realsense_capture_report.json", report)
    return report
