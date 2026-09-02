import json

import numpy as np
from PIL import Image

from robot_grasp.bop import import_bop_sequence
from robot_grasp.io_utils import load_json


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_import_bop_converts_units_and_pose_direction(tmp_path):
    dataset = tmp_path / "lm"
    scene = dataset / "test" / "000001"
    for name in ("rgb", "depth", "mask_visib"):
        (scene / name).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(scene / "rgb" / "000000.png")
    Image.fromarray(np.full((3, 4), 500, dtype=np.uint16)).save(scene / "depth" / "000000.png")
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(scene / "mask_visib" / "000000_000000.png")
    write_json(scene / "scene_camera.json", {
        "0": {"cam_K": [100, 0, 2, 0, 100, 1.5, 0, 0, 1], "depth_scale": 0.1}
    })
    write_json(scene / "scene_gt.json", {
        "0": [{"obj_id": 1, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1], "cam_t_m2c": [100, 0, 500]}]
    })
    write_json(scene / "scene_gt_info.json", {"0": [{"visib_fract": 1.0}]})

    output = tmp_path / "sequence"
    report = import_bop_sequence(dataset, output, split="test", scene_id=1, object_id=1)
    assert report["frame_count"] == 1
    assert load_json(output / "metadata.json")["depth_scale"] == 10000.0
    pose = np.asarray(load_json(output / "poses" / "000000.json")["T_base_camera"])
    np.testing.assert_allclose(pose[:3, 3], [-0.1, 0.0, -0.5])
