import json

import pytest

from robot_grasp.errors import ValidationError
from robot_grasp.zenoh_adapter import ZenohTopicSet, decode_json_payload


def test_zenoh_topic_set_matches_rpp_contract():
    topics = ZenohTopicSet("robot_group", "left")
    assert topics.joint_angle == "robot_group/slave_arm/left/Joint_angle"
    assert topics.relative_pose == "robot_group/slave_arm/left/Relative_pose"
    assert topics.gripper_status == "robot_group/slave_arm/left/Gripper_status"


def test_decode_zenoh_json_payload():
    assert decode_json_payload(json.dumps({"angle": [1, 2]}).encode()) == {"angle": [1, 2]}


def test_decode_zenoh_cdr_payload_has_actionable_error():
    with pytest.raises(ValidationError, match="not valid UTF-8 JSON"):
        decode_json_payload(b"\\x00\\x01cdr")
