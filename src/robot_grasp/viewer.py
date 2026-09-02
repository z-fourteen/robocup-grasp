from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import RobotGraspError, ValidationError
from .grasps import delete_candidate, load_grasps, save_grasps, validate_grasps
from .open3d_utils import require_open3d
from .transforms import validate_transform


def run_grasp_viewer(
    mesh_path: str | Path,
    grasps_path: str | Path,
    *,
    translation_step_m: float = 0.002,
    rotation_step_deg: float = 2.0,
) -> None:
    """Open a minimal keyboard-driven Open3D grasp annotation viewer."""
    if translation_step_m <= 0 or rotation_step_deg <= 0:
        raise ValidationError("Viewer translation and rotation steps must be positive.")
    o3d = require_open3d("grasp annotation viewer")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if mesh.is_empty():
        raise RobotGraspError(f"Could not read mesh {mesh_path}.")
    mesh.compute_vertex_normals()
    data = load_grasps(grasps_path)
    if not data["candidates"]:
        raise ValidationError("The viewer needs at least one candidate. Add one with 'grasps add' first.")

    state = {"index": 0, "axis": None, "dirty": False}
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="RoboCup grasp annotation")
    visualizer.add_geometry(mesh)
    object_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    visualizer.add_geometry(object_axis)

    def redraw(vis):
        if state["axis"] is not None:
            vis.remove_geometry(state["axis"], reset_bounding_box=False)
        candidate = data["candidates"][state["index"]]
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        axis.transform(np.asarray(candidate["T_object_grasp"]))
        state["axis"] = axis
        vis.add_geometry(axis, reset_bounding_box=False)
        vis.update_renderer()
        return False

    def mutate(translation=None, rotation_axis=None, sign=1.0):
        def callback(vis):
            candidate = data["candidates"][state["index"]]
            transform = np.asarray(candidate["T_object_grasp"], dtype=float)
            if translation is not None:
                transform[:3, 3] += sign * translation_step_m * np.asarray(translation)
            if rotation_axis is not None:
                angle = np.radians(sign * rotation_step_deg)
                delta = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray(rotation_axis) * angle)
                transform[:3, :3] = transform[:3, :3] @ delta
            candidate["T_object_grasp"] = validate_transform(transform, name="edited T_object_grasp").tolist()
            state["dirty"] = True
            return redraw(vis)
        return callback

    def switch(delta):
        def callback(vis):
            state["index"] = (state["index"] + delta) % len(data["candidates"])
            return redraw(vis)
        return callback

    def save_callback(_vis):
        validate_grasps(data, source=str(grasps_path))
        save_grasps(grasps_path, data)
        state["dirty"] = False
        return False

    def delete_callback(vis):
        candidate_id = data["candidates"][state["index"]]["id"]
        updated = delete_candidate(data, candidate_id)
        data.clear()
        data.update(updated)
        save_grasps(grasps_path, data)
        state["dirty"] = False
        if not data["candidates"]:
            visualizer.close()
            return False
        state["index"] %= len(data["candidates"])
        return redraw(vis)

    for key, vector in (("A", [-1, 0, 0]), ("D", [1, 0, 0]), ("S", [0, -1, 0]), ("W", [0, 1, 0]), ("Q", [0, 0, -1]), ("E", [0, 0, 1])):
        visualizer.register_key_callback(ord(key), mutate(translation=vector))
    for key, axis, sign in (("J", [1, 0, 0], -1), ("L", [1, 0, 0], 1), ("I", [0, 1, 0], 1), ("K", [0, 1, 0], -1), ("U", [0, 0, 1], -1), ("O", [0, 0, 1], 1)):
        visualizer.register_key_callback(ord(key), mutate(rotation_axis=axis, sign=sign))
    visualizer.register_key_callback(ord("N"), switch(1))
    visualizer.register_key_callback(ord("P"), switch(-1))
    visualizer.register_key_callback(ord("V"), save_callback)
    visualizer.register_key_callback(261, delete_callback)  # GLFW Delete key
    redraw(visualizer)
    visualizer.run()
    visualizer.destroy_window()
