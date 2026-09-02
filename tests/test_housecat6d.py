import pickle

import numpy as np
from PIL import Image

from robot_grasp.housecat6d import import_housecat6d_sequence
from robot_grasp.io_utils import load_json


def _write_matrix(path, matrix):
    np.savetxt(path, np.asarray(matrix, dtype=np.float64))


def _make_dataset(tmp_path):
    dataset = tmp_path / "housecat6d"
    scene = dataset / "val_scene1"
    for name in ("rgb", "depth", "depth_gt", "instance", "labels", "camera_pose", "obj_pose_final"):
        (scene / name).mkdir(parents=True, exist_ok=True)
    model_dir = dataset / "obj_models_small_size_final" / "cup"
    model_dir.mkdir(parents=True)
    (model_dir / "cup-test.obj").write_text("v 0 0 0\n", encoding="ascii")
    (scene / "meta.txt").write_text("4 4 cup-test\n", encoding="utf-8")
    _write_matrix(scene / "intrinsics.txt", [[100, 0, 2], [0, 100, 1.5], [0, 0, 1]])
    T_world_object = np.eye(4)
    T_world_object[:3, 3] = [0.2, 0.0, 0.5]
    _write_matrix(scene / "obj_pose_final" / "cup-test.txt", T_world_object)
    for frame, camera_x in (("000000", 0.0), ("000001", 0.1)):
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(scene / "rgb" / f"{frame}.png")
        Image.fromarray(np.full((4, 5), 500, dtype=np.uint16)).save(scene / "depth" / f"{frame}.png")
        Image.fromarray(np.full((4, 5), 510, dtype=np.uint16)).save(scene / "depth_gt" / f"{frame}.png")
        instances = np.full((4, 5), 255, dtype=np.uint8)
        instances[1:4, 1:5] = 4
        Image.fromarray(instances).save(scene / "instance" / f"{frame}.png")
        T_world_camera = np.eye(4)
        T_world_camera[0, 3] = camera_x
        _write_matrix(scene / "camera_pose" / f"{frame}.txt", T_world_camera)
        T_camera_object = np.linalg.inv(T_world_camera) @ T_world_object
        annotation = {
            "model_list": ["cup-test"],
            "instance_ids": [4],
            "rotations": T_camera_object[None, :3, :3],
            "translations": T_camera_object[None, :3, 3],
        }
        with (scene / "labels" / f"{frame}_label.pkl").open("wb") as stream:
            pickle.dump(annotation, stream, protocol=4)
    return dataset


def test_import_housecat6d_uses_object_centric_pose_and_binary_mask(tmp_path):
    dataset = _make_dataset(tmp_path)
    output = tmp_path / "sequence"
    report = import_housecat6d_sequence(
        dataset,
        output,
        scene_name="val_scene1",
        object_name="cup-test",
        depth_source="depth_gt",
        min_mask_pixels=1,
    )

    assert report["frame_count"] == 2
    assert report["coordinate_convention_check"]["max_translation_error_m"] < 1e-12
    metadata = load_json(output / "metadata.json")
    assert metadata["depth_scale"] == 1000.0
    assert metadata["source"]["depth_source"] == "depth_gt"
    pose = np.asarray(load_json(output / "poses" / "000001.json")["T_base_camera"])
    np.testing.assert_allclose(pose[:3, 3], [-0.1, 0.0, -0.5])
    mask = np.asarray(Image.open(output / "masks" / "000000.png"))
    assert set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) == 12


def test_import_housecat6d_supports_mask_erosion_and_frame_limit(tmp_path):
    dataset = _make_dataset(tmp_path)
    output = tmp_path / "sequence"
    report = import_housecat6d_sequence(
        dataset,
        output,
        scene_name="val_scene1",
        object_name="cup-test",
        mask_erosion_pixels=1,
        min_mask_pixels=1,
        max_frames=1,
    )

    assert report["frame_count"] == 1
    mask = np.asarray(Image.open(output / "masks" / "000000.png"))
    assert np.count_nonzero(mask) == 6
