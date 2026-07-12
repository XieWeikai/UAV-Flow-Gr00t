from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from utils.map2nav_vlnce.coordinates import C_HAB_TO_XNAV, habitat_poses_to_xnav


def test_habitat_basis_is_a_proper_right_handed_rotation() -> None:
    np.testing.assert_allclose(C_HAB_TO_XNAV @ C_HAB_TO_XNAV.T, np.eye(3), atol=0.0)
    assert np.linalg.det(C_HAB_TO_XNAV) == 1.0


def test_habitat_forward_maps_to_xnav_positive_x() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    rotations = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

    states = habitat_poses_to_xnav(positions, rotations)

    np.testing.assert_array_equal(states[0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(states[1, :3], [1.0, 0.0, 0.0], atol=1e-7)


def test_habitat_left_and_right_turns_map_to_signed_xnav_z_yaw() -> None:
    angles = np.deg2rad([0.0, 15.0, -15.0])
    rotations = Rotation.from_euler("y", angles).as_quat()
    positions = np.zeros((3, 3), dtype=np.float64)

    states = habitat_poses_to_xnav(positions, rotations)
    xnav_yaw = Rotation.from_quat(states[:, 3:]).as_euler("xyz", degrees=True)[:, 2]

    np.testing.assert_allclose(xnav_yaw, [0.0, 15.0, -15.0], atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(states[:, 3:], axis=1), 1.0, atol=1e-7)
    assert np.all(states[:, 6] >= 0.0)


def test_pose_is_expressed_in_the_first_agent_frame_before_axis_conversion() -> None:
    first = Rotation.from_euler("y", 90.0, degrees=True)
    positions = np.array([[2.0, 0.5, 3.0], [1.0, 0.5, 3.0]])
    rotations = np.stack([first.as_quat(), first.as_quat()])

    states = habitat_poses_to_xnav(positions, rotations)

    np.testing.assert_allclose(states[1, :3], [1.0, 0.0, 0.0], atol=1e-7)

