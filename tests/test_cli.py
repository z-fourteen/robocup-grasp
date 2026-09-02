import json

from robot_grasp.cli import main


def test_compose_grasp_cli(tmp_path, capsys):
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    camera = tmp_path / "camera.json"
    object_pose = tmp_path / "object.json"
    grasps = tmp_path / "grasps.json"
    camera.write_text(json.dumps({"T_base_camera": identity}), encoding="utf-8")
    object_pose.write_text(json.dumps({"T_camera_object": identity}), encoding="utf-8")
    grasp = {
        "id": "g", "T_object_grasp": identity, "pregrasp_offset": [0, 0, -0.1],
        "gripper_width": 0.05, "approach_distance": 0.1, "priority": 1,
        "enabled": True, "symmetry_class": "none", "notes": "",
    }
    grasps.write_text(json.dumps({
        "schema_version": 1, "object_id": "obj", "length_unit": "meter",
        "transform_convention": "T_dst_src transforms points from src coordinates into dst coordinates",
        "candidates": [grasp],
    }), encoding="utf-8")
    assert main(["compose-grasp", "--camera-pose", str(camera), "--object-pose", str(object_pose), "--grasps", str(grasps)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["selected_grasp_id"] == "g"
    assert output["T_base_grasp"] == identity
