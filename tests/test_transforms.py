import numpy as np
import pytest

from robot_grasp.errors import ValidationError
from robot_grasp.transforms import (
    compose_base_grasp,
    compose_transform,
    invert_transform,
    transform_points,
    validate_transform,
)


def transform_z_90(translation=(0.0, 0.0, 0.0)):
    result = np.eye(4)
    result[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    result[:3, 3] = translation
    return result


def test_identity_transform():
    identity = np.eye(4)
    np.testing.assert_allclose(invert_transform(identity), identity)
    np.testing.assert_allclose(transform_points(identity, [[1, 2, 3]]), [[1, 2, 3]])


def test_known_translation_and_rotation():
    transform = transform_z_90((1.0, 2.0, 3.0))
    np.testing.assert_allclose(transform_points(transform, [[1.0, 0.0, 0.0]]), [[1.0, 3.0, 3.0]])


def test_inverse_round_trip():
    transform = transform_z_90((0.2, -0.4, 1.1))
    points = np.array([[0.1, 0.2, 0.3], [-1.0, 2.0, 0.0]])
    round_trip = transform_points(invert_transform(transform), transform_points(transform, points))
    np.testing.assert_allclose(round_trip, points, atol=1e-12)
    np.testing.assert_allclose(compose_transform(transform, invert_transform(transform)), np.eye(4), atol=1e-12)


def test_invalid_rotation_matrix_rejected():
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    with pytest.raises(ValidationError, match="rotation is invalid"):
        validate_transform(invalid, name="T_bad")


def test_direction_sensitive_base_grasp_composition():
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = [1.0, 0.25, 0.0]
    T_camera_object = transform_z_90((0.0, 2.0, 0.0))
    T_object_grasp = np.eye(4)
    T_object_grasp[:3, 3] = [0.3, 0.0, 0.0]

    T_base_grasp = compose_base_grasp(T_base_camera, T_camera_object, T_object_grasp)
    np.testing.assert_allclose(T_base_grasp[:3, 3], [1.0, 2.55, 0.0])
    wrong_order = T_object_grasp @ T_camera_object @ T_base_camera
    assert not np.allclose(T_base_grasp, wrong_order)
