from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import ValidationError
from .io_utils import load_json
from .transforms import validate_transform


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class SequenceFrame:
    stem: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path | None
    pose_path: Path
    T_base_camera: np.ndarray
    valid_depth_ratio: float


@dataclass(frozen=True)
class ValidatedSequence:
    root: Path
    intrinsics: Intrinsics
    metadata: dict[str, Any]
    frames: tuple[SequenceFrame, ...]
    use_mask: bool = True

    @property
    def depth_scale(self) -> float:
        return float(self.metadata["depth_scale"])


def _file_map(
    directory: Path,
    extensions: set[str],
    label: str,
    errors: list[str],
    *,
    required: bool = True,
) -> dict[str, Path]:
    if not directory.is_dir():
        if required:
            errors.append(f"Missing {label} directory: {directory}. Create it and add one file per frame.")
        return {}
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            errors.append(
                f"Duplicate {label} stem '{path.stem}': {result[path.stem]} and {path}. Keep exactly one file."
            )
        else:
            result[path.stem] = path
    if required and not result:
        errors.append(f"No supported {label} files found in {directory}. Supported extensions: {sorted(extensions)}.")
    return result


def _parse_intrinsics(path: Path, errors: list[str]) -> Intrinsics | None:
    try:
        data = load_json(path)
    except ValidationError as exc:
        errors.append(str(exc))
        return None
    if not isinstance(data, dict):
        errors.append(f"{path} must contain a JSON object with width, height, fx, fy, cx, cy.")
        return None
    missing = [key for key in ("width", "height", "fx", "fy", "cx", "cy") if key not in data]
    if missing:
        errors.append(f"{path} is missing {missing}. Add all camera intrinsic fields.")
        return None
    try:
        values = {key: float(data[key]) for key in ("fx", "fy", "cx", "cy")}
        width = int(data["width"])
        height = int(data["height"])
    except (TypeError, ValueError) as exc:
        errors.append(f"{path} intrinsic fields must be numeric: {exc}.")
        return None
    if width <= 0 or height <= 0 or values["fx"] <= 0 or values["fy"] <= 0:
        errors.append(f"{path} width, height, fx and fy must be positive; recalibrate or correct the file.")
        return None
    if not all(np.isfinite(list(values.values()))):
        errors.append(f"{path} contains non-finite intrinsic values; replace them with calibration results.")
        return None
    return Intrinsics(width=width, height=height, **values)


def _parse_metadata(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = load_json(path)
    except ValidationError as exc:
        errors.append(str(exc))
        return None
    if not isinstance(data, dict):
        errors.append(f"{path} must contain a JSON object.")
        return None
    required = ("depth_scale", "object_id", "coordinate_frames")
    missing = [key for key in required if key not in data]
    if missing:
        errors.append(
            f"{path} is missing {missing}. Add depth units, object identity, and coordinate-frame descriptions."
        )
        return None
    try:
        depth_scale = float(data["depth_scale"])
    except (TypeError, ValueError):
        errors.append(f"{path} depth_scale must be a positive number of raw depth units per meter.")
        return None
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        errors.append(f"{path} depth_scale must be > 0; use e.g. 1000 for millimeter depth images.")
    if not isinstance(data["object_id"], str) or not data["object_id"].strip():
        errors.append(f"{path} object_id must be a non-empty string.")
    frames = data["coordinate_frames"]
    required_frames = {"base", "camera"}
    if not isinstance(frames, dict) or not required_frames.issubset(frames) or not all(
        isinstance(frames.get(name), str) and frames[name].strip() for name in required_frames
    ):
        errors.append(
            f"{path} coordinate_frames must describe non-empty 'base' and 'camera' frames; "
            "also document that poses are T_base_camera."
        )
    return data


def _read_image(path: Path, label: str, errors: list[str]) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            return np.asarray(image.copy())
    except (OSError, UnidentifiedImageError) as exc:
        errors.append(f"Cannot read {label} image {path}: {exc}. Re-export it in a supported image format.")
        return None


def validate_sequence(
    root: str | Path,
    *,
    min_valid_depth_ratio: float = 0.01,
    use_mask: bool = True,
) -> ValidatedSequence:
    """Validate an RGB-D sequence using an explicit mask contract.

    When ``use_mask`` is true, every frame must provide a mask and valid-depth
    ratios use mask pixels as the denominator. When false, masks are optional,
    ignored for depth validity, and ratios use all image pixels as the
    denominator.
    """
    root = Path(root)
    errors: list[str] = []
    if not isinstance(use_mask, bool):
        errors.append(f"use_mask must be true or false, got {use_mask!r}.")
    if not root.is_dir():
        raise ValidationError(f"Sequence directory does not exist: {root}. Pass the directory containing rgb/depth/masks/poses.")
    intrinsics = _parse_intrinsics(root / "intrinsics.json", errors)
    metadata = _parse_metadata(root / "metadata.json", errors)

    rgb = _file_map(root / "rgb", IMAGE_EXTENSIONS, "RGB", errors)
    depth = _file_map(root / "depth", IMAGE_EXTENSIONS, "depth", errors)
    masks = _file_map(root / "masks", IMAGE_EXTENSIONS, "mask", errors) if use_mask else {}
    poses = _file_map(root / "poses", {".json"}, "pose", errors)
    all_stems = set().union(rgb, depth, poses)
    if use_mask:
        all_stems |= set(masks)
    required_files = [("RGB", rgb), ("depth", depth), ("pose", poses)]
    if use_mask:
        required_files.append(("mask", masks))
    for stem in sorted(all_stems):
        missing = [label for label, files in required_files if stem not in files]
        if missing:
            errors.append(
                f"Frame '{stem}' is missing {missing}. Add files with the exact stem '{stem}' or remove the unmatched files."
            )

    frames: list[SequenceFrame] = []
    if not 0.0 <= min_valid_depth_ratio <= 1.0:
        errors.append(f"min_valid_depth_ratio must be within [0, 1], got {min_valid_depth_ratio}.")
    if intrinsics is not None and metadata is not None:
        complete_stems = set(rgb) & set(depth) & set(poses)
        if use_mask:
            complete_stems &= set(masks)
        for stem in sorted(complete_stems):
            rgb_image = _read_image(rgb[stem], "RGB", errors)
            depth_image = _read_image(depth[stem], "depth", errors)
            mask_path = masks.get(stem)
            mask_image = _read_image(mask_path, "mask", errors) if mask_path is not None else None
            expected = (intrinsics.height, intrinsics.width)
            for label, path, image in (
                ("RGB", rgb[stem], rgb_image), ("depth", depth[stem], depth_image),
                ("mask", mask_path, mask_image)
            ):
                if path is None:
                    continue
                if image is not None and image.shape[:2] != expected:
                    errors.append(
                        f"{label} image {path} has size {image.shape[1]}x{image.shape[0]}, expected "
                        f"{intrinsics.width}x{intrinsics.height} from {root / 'intrinsics.json'}. Resize/export consistently."
                    )
            if rgb_image is not None and (rgb_image.ndim != 3 or rgb_image.shape[2] not in (3, 4)):
                errors.append(f"RGB image {rgb[stem]} must have 3 or 4 channels, got shape {rgb_image.shape}.")
            if depth_image is not None and depth_image.ndim != 2:
                errors.append(f"Depth image {depth[stem]} must be single-channel, got shape {depth_image.shape}.")
            if mask_image is not None and mask_image.ndim != 2:
                errors.append(f"Mask image {masks[stem]} must be single-channel, got shape {mask_image.shape}.")

            valid_ratio = 0.0
            if depth_image is not None and depth_image.ndim == 2:
                valid = np.isfinite(depth_image) & (depth_image > 0)
                if use_mask and mask_image is not None and mask_image.ndim == 2 and mask_image.shape == depth_image.shape:
                    valid &= mask_image > 0
                    denominator = max(1, int(np.count_nonzero(mask_image)))
                else:
                    denominator = depth_image.size
                valid_ratio = float(np.count_nonzero(valid) / denominator)
                if valid_ratio < min_valid_depth_ratio:
                    errors.append(
                        f"Depth image {depth[stem]} has valid-depth ratio {valid_ratio:.3%}, below "
                        f"{min_valid_depth_ratio:.3%}. Check depth_scale, sensor range, and mask alignment."
                    )
            try:
                pose_data = load_json(poses[stem])
                if isinstance(pose_data, dict):
                    if "T_base_camera" not in pose_data:
                        raise ValidationError(
                            f"Pose file {poses[stem]} must contain key 'T_base_camera'; do not store an unnamed or inverse pose."
                        )
                    pose_data = pose_data["T_base_camera"]
                transform = validate_transform(pose_data, name=f"T_base_camera in {poses[stem]}")
            except ValidationError as exc:
                errors.append(str(exc))
                continue
            frames.append(SequenceFrame(stem, rgb[stem], depth[stem], mask_path, poses[stem], transform, valid_ratio))

    if errors:
        raise ValidationError("Sequence validation failed:\n- " + "\n- ".join(errors))
    assert intrinsics is not None and metadata is not None
    return ValidatedSequence(root, intrinsics, metadata, tuple(frames), use_mask=use_mask)


def sequence_summary(sequence: ValidatedSequence) -> dict[str, Any]:
    ratios = [frame.valid_depth_ratio for frame in sequence.frames]
    return {
        "input": str(sequence.root),
        "object_id": sequence.metadata["object_id"],
        "frame_count": len(sequence.frames),
        "depth_scale": sequence.depth_scale,
        "use_mask": sequence.use_mask,
        "valid_depth_ratio_denominator": "mask_pixels" if sequence.use_mask else "image_pixels",
        "mean_valid_depth_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "min_valid_depth_ratio": float(np.min(ratios)) if ratios else 0.0,
        "status": "valid",
    }
