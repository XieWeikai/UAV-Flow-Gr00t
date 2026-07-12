"""Coordinate transforms for Map2Nav Habitat replay poses."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


C_HAB_TO_XNAV = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def habitat_poses_to_xnav(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> np.ndarray:
    """Convert Habitat world poses to xNav episode-start-relative poses.

    Habitat uses Y-up with agent forward along local -Z. xNav uses Z-up with
    +X forward and +Y left. Both input and output quaternions use XYZW order.
    """

    positions = np.asarray(positions, dtype=np.float64)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError(f"positions must have shape [N,3] with N>0, got {positions.shape}")
    if quaternions_xyzw.shape != (len(positions), 4):
        raise ValueError(
            "quaternions_xyzw must have shape [N,4] matching positions, "
            f"got {quaternions_xyzw.shape}"
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions_xyzw)):
        raise ValueError("poses contain non-finite values")
    norms = np.linalg.norm(quaternions_xyzw, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("poses contain a zero-norm quaternion")

    rotations = Rotation.from_quat(quaternions_xyzw)
    first_inv = rotations[0].inv()
    local_positions = first_inv.apply(positions - positions[0])
    xnav_positions = (C_HAB_TO_XNAV @ local_positions.T).T

    states = np.empty((len(positions), 7), dtype=np.float64)
    states[:, :3] = xnav_positions
    for index, rotation in enumerate(rotations):
        relative_hab = (first_inv * rotation).as_matrix()
        relative_xnav = C_HAB_TO_XNAV @ relative_hab @ C_HAB_TO_XNAV.T
        quaternion = Rotation.from_matrix(relative_xnav).as_quat()
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        states[index, 3:] = quaternion

    states[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return states.astype(np.float32)

