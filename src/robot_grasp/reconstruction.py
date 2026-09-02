from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ReconstructionConfig
from .errors import RobotGraspError, ValidationError
from .io_utils import dump_json, dump_yaml, prepare_output_dir
from .open3d_utils import require_open3d
from .sequence import ValidatedSequence, validate_sequence
from .transforms import invert_transform


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _read_single_channel(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValidationError(f"Expected a single-channel image at {path}, got shape {array.shape}.")
    return np.asarray(array)


def _clean_mesh(mesh: Any, config: ReconstructionConfig) -> tuple[Any, dict[str, int]]:
    before = len(mesh.triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    removed_components = 0
    if len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_array = np.asarray(labels)
        counts_array = np.asarray(counts)
        largest = int(counts_array.max()) if len(counts_array) else 0
        threshold = max(config.min_component_triangles, int(np.ceil(largest * config.min_component_ratio)))
        keep_clusters = counts_array >= threshold
        if len(keep_clusters):
            keep_clusters[int(np.argmax(counts_array))] = True
            remove_mask = ~keep_clusters[labels_array]
            removed_components = int(np.count_nonzero(~keep_clusters))
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh, {"triangles_before_cleanup": before, "removed_small_components": removed_components}


def _bbox_report(geometry: Any) -> dict[str, list[float]]:
    bbox = geometry.get_axis_aligned_bounding_box()
    return {
        "min_m": np.asarray(bbox.min_bound, dtype=float).tolist(),
        "max_m": np.asarray(bbox.max_bound, dtype=float).tolist(),
        "extent_m": np.asarray(bbox.get_extent(), dtype=float).tolist(),
    }


def reconstruct_sequence(
    input_dir: str | Path,
    output_dir: str | Path,
    config: ReconstructionConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    sequence: ValidatedSequence = validate_sequence(
        input_dir, min_valid_depth_ratio=config.min_valid_depth_ratio
    )
    o3d = require_open3d("TSDF reconstruction")
    output = prepare_output_dir(output_dir, overwrite=overwrite)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        sequence.intrinsics.width,
        sequence.intrinsics.height,
        sequence.intrinsics.fx,
        sequence.intrinsics.fy,
        sequence.intrinsics.cx,
        sequence.intrinsics.cy,
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_length,
        sdf_trunc=config.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    fusion_ratios: list[float] = []
    depth_quantile_ranges: dict[str, list[float] | None] = {}
    for frame in sequence.frames:
        rgb = _read_rgb(frame.rgb_path)
        raw_depth = _read_single_channel(frame.depth_path).astype(np.float32)
        depth_m = raw_depth / sequence.depth_scale
        valid = np.isfinite(depth_m) & (depth_m >= config.depth_min) & (depth_m <= config.depth_max)
        if config.use_mask:
            valid &= _read_single_channel(frame.mask_path) > 0
        if np.any(valid) and (config.depth_quantile_low > 0.0 or config.depth_quantile_high < 1.0):
            low, high = np.quantile(
                depth_m[valid], [config.depth_quantile_low, config.depth_quantile_high]
            )
            valid &= (depth_m >= low) & (depth_m <= high)
            depth_quantile_ranges[frame.stem] = [float(low), float(high)]
        else:
            depth_quantile_ranges[frame.stem] = None
        depth_m[~valid] = 0.0
        fusion_ratios.append(float(np.count_nonzero(valid) / valid.size))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(np.ascontiguousarray(depth_m)),
            depth_scale=1.0,
            depth_trunc=config.depth_max,
            convert_rgb_to_intensity=False,
        )
        # Open3D integrate() expects world-to-camera, while sequence poses are T_base_camera.
        T_camera_base = invert_transform(frame.T_base_camera)
        volume.integrate(rgbd, intrinsic, T_camera_base)

    point_cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    if len(point_cloud.points) == 0 or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RobotGraspError(
            "TSDF fusion produced empty geometry. Check depth_scale, T_base_camera direction, masks, "
            "depth range, and whether the object is observed by multiple frames."
        )
    point_cloud.estimate_normals()
    mesh, cleanup = _clean_mesh(mesh, config)
    if len(mesh.triangles) == 0:
        raise RobotGraspError(
            "Mesh cleanup removed every triangle. Lower min_component_triangles/min_component_ratio "
            "or improve RGB-D coverage."
        )

    collision_mesh = copy.deepcopy(mesh)
    if len(collision_mesh.triangles) > config.collision_target_triangles:
        collision_mesh = collision_mesh.simplify_quadric_decimation(config.collision_target_triangles)
        collision_mesh.remove_degenerate_triangles()
        collision_mesh.remove_unreferenced_vertices()
    collision_mesh.compute_vertex_normals()

    outputs = {
        "point_cloud": output / "fused.ply",
        "mesh_high": output / "mesh_high.ply",
        "mesh_collision": output / "mesh_collision.obj",
    }
    writers = (
        (o3d.io.write_point_cloud, outputs["point_cloud"], point_cloud),
        (o3d.io.write_triangle_mesh, outputs["mesh_high"], mesh),
        (o3d.io.write_triangle_mesh, outputs["mesh_collision"], collision_mesh),
    )
    for writer, path, geometry in writers:
        if not writer(str(path), geometry):
            raise RobotGraspError(f"Open3D failed to write {path}. Check permissions and output format support.")

    resolved_config = config.to_dict() | {"depth_scale": sequence.depth_scale, "length_unit": "meter"}
    # This file can be passed back to --config without removing report-only metadata.
    dump_yaml(output / "reconstruction_config.yaml", config.to_dict())
    report = {
        "status": "completed",
        "input": str(sequence.root),
        "object_id": sequence.metadata["object_id"],
        "frame_count": len(sequence.frames),
        "valid_depth_ratio": {
            "per_frame": dict(zip((frame.stem for frame in sequence.frames), fusion_ratios)),
            "mean": float(np.mean(fusion_ratios)),
            "min": float(np.min(fusion_ratios)),
        },
        "depth_quantile_ranges_m": depth_quantile_ranges,
        "bounding_box": _bbox_report(mesh),
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "collision_vertex_count": len(collision_mesh.vertices),
        "collision_triangle_count": len(collision_mesh.triangles),
        "cleanup": cleanup,
        "config": resolved_config,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "hole_filling": False,
    }
    dump_json(output / "reconstruction_report.json", report)
    return report
