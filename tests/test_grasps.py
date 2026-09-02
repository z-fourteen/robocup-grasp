import numpy as np
import pytest

from robot_grasp.errors import ValidationError
from robot_grasp.grasps import add_candidate, delete_candidate, empty_grasps, make_candidate, update_candidate


def test_grasp_crud_and_defaults():
    data = empty_grasps("cup")
    candidate = make_candidate("side", np.eye(4), gripper_width=0.06, priority=10)
    data = add_candidate(data, candidate)
    assert data["candidates"][0]["pregrasp_offset"] == [0.0, 0.0, -0.1]
    data = update_candidate(data, "side", {"enabled": False, "notes": "blocked"})
    assert data["candidates"][0]["enabled"] is False
    assert delete_candidate(data, "side")["candidates"] == []


def test_duplicate_grasp_id_rejected():
    candidate = make_candidate("same", np.eye(4))
    data = add_candidate(empty_grasps("cup"), candidate)
    with pytest.raises(ValidationError, match="already exists"):
        add_candidate(data, candidate)


def test_invalid_grasp_transform_rejected():
    transform = np.eye(4)
    transform[2, 2] = -1
    with pytest.raises(ValidationError, match="rotation is invalid"):
        make_candidate("reflection", transform)


def test_non_finite_grasp_length_rejected():
    with pytest.raises(ValidationError, match="NaN or infinity"):
        make_candidate("bad_width", np.eye(4), gripper_width=float("nan"))
