from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .errors import RobotGraspError, ValidationError
from .io_utils import dump_json, load_json, prepare_output_dir
from .open3d_utils import require_open3d
from .transforms import validate_transform


def load_named_transform(path: str | Path, key: str) -> Any:
    data = load_json(path)
    if isinstance(data, dict):
        if key not in data:
            raise ValidationError(f"Transform file {path} must contain key '{key}'.")
        data = data[key]
    return validate_transform(data, name=f"{key} in {path}")


def set_object_frame(
    mesh_path: str | Path,
    transform_path: str | Path,
    output_dir: str | Path,
    *,
    object_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    mesh_path = Path(mesh_path).resolve()
    if not mesh_path.is_file():
        raise ValidationError(f"Input mesh does not exist: {mesh_path}. Pass the reconstructed high-resolution mesh.")
    if not object_id.strip():
        raise ValidationError("object_id must be a non-empty string.")
    T_object_model = load_named_transform(transform_path, "T_object_model")
    o3d = require_open3d("object-frame mesh conversion")
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RobotGraspError(f"Open3D could not read a triangle mesh from {mesh_path}. Check the mesh format and contents.")

    original_copy = output / f"model_original{mesh_path.suffix.lower()}"
    transformed_path = output / "mesh_object.ply"
    shutil.copy2(mesh_path, original_copy)
    mesh.transform(T_object_model)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(transformed_path), mesh):
        raise RobotGraspError(f"Open3D failed to write transformed mesh to {transformed_path}.")
    frame_data = {
        "schema_version": 1,
        "object_id": object_id,
        "length_unit": "meter",
        "source_mesh": str(mesh_path),
        "preserved_mesh": str(original_copy),
        "object_mesh": str(transformed_path),
        "T_object_model": T_object_model.tolist(),
        "transform_convention": "T_dst_src transforms points from src coordinates into dst coordinates",
    }
    dump_json(output / "object_frame.json", frame_data)
    return frame_data
