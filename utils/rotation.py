import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Iterable
import pathlib
import copy

indices = {
    "x": 0,
    "y": 1,
    "z": 2,
    "roll": 3,
    "pitch": 5,
    "yaw": 4,
}

# for coordinate system conversion
# UAV-Flow uses FRU (x-forward, y-right, z-up) (not sure, need to confirm, it can be NED)
sign = {
    "x": 1,
    "y": 1,
    "z": 1,
    "roll": 1,
    "pitch": 1,
    "yaw": 1,
}

def relative_pose(p1: np.array, p2: np.array, degree=False) -> np.array:
    """
    Compute the relative pose from p1 to p2.
    
    Args:
        p1 (np.array): Pose of P1 as [x, y, z, roll, pitch, yaw] (units: m, radians if degree=False)
        p2 (np.array): Pose of P2 as [x, y, z, roll, pitch, yaw] (units: m, radians if degree=False)

    Returns:
        np.array: Relative pose of P2 in P1's frame as [x, y, z, roll, pitch, yaw] (units: m, radians if degree=False)
    """
    # Extract positions and orientations
    pos1 = p1[:3]
    pos2 = p2[:3]
    orient1 = p1[[-1, -2, -3]]
    orient2 = p2[[-1, -2, -3]]

    delta_pos = pos2 - pos1

    # Compute relative orientation
    rot1 = R.from_euler('ZYX', orient1, degrees=degree)
    rot2 = R.from_euler('ZYX', orient2, degrees=degree)
    # relative position in p1's coordinate
    rot1_inv = rot1.inv()
    relative_pos = rot1_inv.apply(delta_pos)
    # relative_rot = rot2 * rot1_inv 
    relative_rot = rot1_inv * rot2
    # print(f"rot1_inv (as matrix): \n{rot1_inv.as_matrix()}")
    # print(f"rot2 (as matrix): \n{rot2.as_matrix()}")
    # print(f"rot1_inv * rot2 (as matrix): \n{(rot1_inv * rot2).as_matrix()}")
    # print(f"rot2 * rot1_inv (as matrix): \n{(rot2 * rot1_inv).as_matrix()}")
    relative_euler = relative_rot.as_euler('ZYX', degrees=degree)

    # Combine relative position and orientation
    relative_pose = np.concatenate([relative_pos, relative_euler[::-1]])
    return relative_pose

def relative_pose_given_axes(p1: np.array, p2: np.array, degree=False, axes: list[str] = ["x", "y", "z", "roll", "pitch", "yaw"]) -> np.array:
    """
    Compute the relative pose from p1 to p2 using UAV-Flow's method, considering only the specified axes.
    """
    
    # assert axes is subset of ["x", "y", "z", "roll", "pitch", "yaw"]
    assert set(axes).issubset(set(["x", "y", "z", "roll", "pitch", "yaw"])), "Invalid axes specified."
    # Create copies of p1 and p2 to avoid modifying the originals
    p1 = copy.deepcopy(p1)
    p2 = copy.deepcopy(p2)
    for i, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        if key not in axes:
            p1[i] = 0.0
            p2[i] = 0.0

    # Compute the relative pose using the modified poses
    return relative_pose(p1, p2, degree=degree)


def dict_to_array(p: dict) -> np.array:
    return np.array([p[key] * sign[key] for key in indices.keys()])

def array_to_dict(arr: np.array) -> dict:
    return {key: arr[i] * sign[key] for i, key in enumerate(indices.keys())}


# copy from UAV-Flow
# https://github.com/buaa-colalab/UAV-Flow/blob/0114801f585a29296be6d035d67401abeabd26d3/OpenVLA-UAV/prismatic/vla/datasets/uav_dataset.py#L167
def _transform_to_local_frame(current_pose: np.ndarray, next_pose: np.ndarray) -> np.ndarray:
        """
        Transform the next pose into the local frame of the current pose
        """
        # Extract position and yaw
        current_pos = current_pose[:3]
        current_yaw = current_pose[3]
        
        next_pos = next_pose[:3]
        next_yaw = next_pose[3]
        
        # Build 2D rotation matrix
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Compute relative position
        relative_pos = next_pos - current_pos
        local_pos = np.linalg.inv(R) @ relative_pos
        
        # Compute relative yaw
        relative_yaw = next_yaw - current_yaw
        relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([local_pos[0], local_pos[1], local_pos[2], relative_yaw])


def UAV_Flow_relative_pose(p1, p2):
    """
    Compute the relative pose from p1 to p2 using UAV-Flow's method.
    
    Args:
        p1 (dict): Pose of P1 with keys 'x','y','z','roll','pitch','yaw' (units: m, degrees)
        p2 (dict): Pose of P2 with the same keys (units: m, degrees)

    Returns:
        dict: Relative pose from P1 to P2
    """
    # Build 4D arrays [x, y, z, yaw_rad]; input yaw is in degrees
    p1_4d = np.array([p1['x'], p1['y'], p1['z'], np.deg2rad(p1['yaw'])])
    p2_4d = np.array([p2['x'], p2['y'], p2['z'], np.deg2rad(p2['yaw'])])

    # Transform to local frame; keep _transform_to_local_frame unchanged (expects radians and a stray first arg)
    local_frame = _transform_to_local_frame(p1_4d, p2_4d)

    # Extract local position and yaw
    local_pos = local_frame[:3]
    # Convert relative yaw back to degrees for output
    local_yaw = np.rad2deg(local_frame[3])

    # Construct relative pose
    relative_pose = {
        'x': local_pos[0],
        'y': local_pos[1],
        'z': local_pos[2],
        'roll': 0.0, # ignore roll/pitch for UAV-Flow
        'pitch': 0.0, # ignore roll/pitch for UAV-Flow
        'yaw': local_yaw
    }

    return relative_pose

def print_pose(pose, label="Pose"):
    print(f"{label}:")
    print(f"  x={pose['x']:.6f} m, y={pose['y']:.6f} m, z={pose['z']:.6f} m")
    print(f"  Roll={pose['roll']:.6f}°, Pitch={pose['pitch']:.6f}°, Yaw={pose['yaw']:.6f}°\n")


def test_relative_pose():
    THIS_DIR = pathlib.Path(__file__).parent
    example_path = THIS_DIR / "example_data.json"
    with open(example_path, "r") as f:
        data = json.load(f)
        
    absolute_poses = data["raw_logs"]
    
    length = len(absolute_poses)
    print(f"Total frames in example data: {length}\n")
    for i in range(length - 1):
        P1 = {key: absolute_poses[i][indices[key]] * sign[key] for key in indices}
        P2 = {key: absolute_poses[i + 1][indices[key]] * sign[key] for key in indices}
        print(f"--- Frame {i} to Frame {i+1} ---")
        print_pose(P1, label="P1 in World Frame")
        print_pose(P2, label="P2 in World Frame")

        relative_p2 = array_to_dict(relative_pose(dict_to_array(P1), dict_to_array(P2), degree=True))

        print_pose(relative_p2, label="P2 relative to P1 (calculated)")

        drone_relative_p2 = array_to_dict(relative_pose_given_axes(
            dict_to_array(P1), dict_to_array(P2), 
            axes=["x", "y", "z", "yaw"],
            degree=True
        ))
        print_pose(drone_relative_p2, label="P2 relative to P1 ignoring roll/pitch (calculated)")
        
        relative_p2_uavflow = UAV_Flow_relative_pose(P1, P2)
        print_pose(relative_p2_uavflow, label="P2 relative to P1 (UAV-Flow method)")
        
        assert np.allclose(
            [drone_relative_p2[k] for k in ["x", "y", "z", "yaw"]],
            [relative_p2_uavflow[k] for k in ["x", "y", "z", "yaw"]],
            atol=1e-6
        ), "Mismatch between calculated and UAV-Flow method!"
    
    # print a green success message
    print("All frames processed successfully! Calculated relative poses match UAV-Flow method.")
    
    relative_poses = data["preprocessed_logs"]
    P0 = {key: absolute_poses[0][indices[key]] * sign[key] for key in indices}
    for i in range(1, length):
        P = {key: absolute_poses[i][indices[key]] * sign[key] for key in indices}
        pose = relative_pose(dict_to_array(P0), dict_to_array(P), degree=True)
        ref_pose = {key: relative_poses[i][indices[key]] * sign[key] for key in indices}
        print(f"--- Frame {0} to Frame {i} ---")
        print_pose(P0, label="P0 in World Frame")
        print_pose(P, label="P1 in World Frame")
        print_pose(array_to_dict(pose), label="P1 relative to P0 (calculated)")
        print_pose(ref_pose, label="P1 relative to P0 (reference from preprocessed_logs)")

    # P1 = {key: absolute_poses[1][indices[key]] * sign[key] for key in indices}
    # print_pose(P1, label="P1 in World Frame")
    
    # P2 = {key: absolute_poses[2][indices[key]] * sign[key] for key in indices}
    # print_pose(P2, label="P2 in World Frame")

    # relative_p2 = array_to_dict(relative_pose(dict_to_array(P1), dict_to_array(P2), degree=True))

    # print_pose(relative_p2, label="P2 relative to P1 (calculated)")

    # drone_relative_p2 = array_to_dict(relative_pose_given_axes(
    #     dict_to_array(P1), dict_to_array(P2), 
    #     axes=["x", "y", "z", "yaw"],
    #     degree=True
    # ))
    # print_pose(drone_relative_p2, label="P2 relative to P1 ignoring roll/pitch (calculated)")
    
    # relative_p2_uavflow = UAV_Flow_relative_pose(P1, P2)
    # print_pose(relative_p2_uavflow, label="P2 relative to P1 (UAV-Flow method)")


def body_to_world_pose(p1: np.ndarray, p2: np.ndarray, degree: bool = False) -> np.ndarray:
    """
    Transform a pose given in the body (drone) frame into the world (origin) frame.

    Args:
        p1 (np.ndarray): Pose of the body frame origin in world as [x, y, z, roll, pitch, yaw]
        p2 (np.ndarray): Pose expressed in the body frame as [x, y, z, roll, pitch, yaw]
        degree (bool): Whether the roll/pitch/yaw values are in degrees. Defaults to False (radians).

    Returns:
        np.ndarray: Pose of p2 expressed in the world frame as [x, y, z, roll, pitch, yaw]

    Note:
        This uses the same Euler convention as the other helpers in this file: euler angles are
        interpreted as (yaw, pitch, roll) when passed to scipy Rotation with the 'ZYX' sequence.
    """
    # positions
    pos1 = p1[:3]
    pos2 = p2[:3]

    # orientations: convert from [roll, pitch, yaw] to [yaw, pitch, roll] for 'ZYX'
    orient1 = p1[[-1, -2, -3]]
    orient2 = p2[[-1, -2, -3]]

    # build rotations
    rot1 = R.from_euler('ZYX', orient1, degrees=degree)
    rot2 = R.from_euler('ZYX', orient2, degrees=degree)

    # transform position and compose rotations
    world_pos = rot1.apply(pos2) + pos1
    world_rot = rot1 * rot2
    world_euler = world_rot.as_euler('ZYX', degrees=degree)

    # return in [x, y, z, roll, pitch, yaw] order
    return np.concatenate([world_pos, world_euler[::-1]])


def random_pose(degree: bool = False) -> np.ndarray:
    """
    Generate a random pose vector [x, y, z, roll, pitch, yaw].

    Args:
        degree (bool): If True, return roll/pitch/yaw in degrees. If False, in radians.

    Returns:
        np.ndarray: Shape (6,) pose array in the same ordering used elsewhere in this file.

    Notes / assumptions:
        - x and y are sampled uniformly in [-10, 10] meters.
        - z is sampled uniformly in [0, 20] meters (assumes altitude >= 0).
        - roll, pitch, yaw are sampled uniformly in [-180, 180] degrees or the
          equivalent in radians when ``degree`` is False.
        These ranges are reasonable defaults for quick tests; pass a wrapper if you
        need different ranges.
    """
    x = float(np.random.uniform(-10.0, 10.0))
    y = float(np.random.uniform(-10.0, 10.0))
    z = float(np.random.uniform(0.0, 20.0))

    if degree:
        roll, pitch, yaw = np.random.uniform(-180.0, 180.0, size=3)
    else:
        # sample in degrees then convert to radians for a uniform angular distribution
        roll, pitch, yaw = np.deg2rad(np.random.uniform(-180.0, 180.0, size=3))

    return np.array([x, y, z, roll, pitch, yaw], dtype=float)


# Make sure you have this import at the top of your file
from scipy.spatial.transform import Rotation as R

def test_body_to_world_pose():
    """
    Corrected test function that properly compares 3D rotations.
    """
    np.random.seed(42)
    degrees = [True, False]
    for degree in degrees:
        for _ in range(10):
            p1 = random_pose(degree=degree)
            p2 = random_pose(degree=degree)

            # This part of your logic is correct
            relative = relative_pose(p1, p2, degree=degree)
            reconstructed_p2 = body_to_world_pose(p1, relative, degree=degree)

            # --- MODIFIED ASSERTION LOGIC ---

            # 1. Test the position part separately, which is straightforward.
            assert np.allclose(p2[:3], reconstructed_p2[:3], atol=1e-6), \
                "Position mismatch in body_to_world_pose test!"

            # 2. For orientation, convert Euler angles back to rotation objects and compare them.
            # this is because Euler angles can be non-unique and comparing them directly can be misleading.
            # Original orientation
            orient_p2 = p2[[-1, -2, -3]]
            rot_p2 = R.from_euler('ZYX', orient_p2, degrees=degree)

            # Reconstructed orientation
            orient_reconstructed = reconstructed_p2[[-1, -2, -3]]
            rot_reconstructed = R.from_euler('ZYX', orient_reconstructed, degrees=degree)

            # The best way to compare two rotations is to see if their difference
            # is the identity rotation. The "magnitude" of this difference rotation
            # is the angle (in radians) between them, which should be close to zero.
            angular_distance = (rot_p2 * rot_reconstructed.inv()).magnitude()
            
            assert np.isclose(angular_distance, 0.0, atol=1e-6), \
                "Orientation mismatch in body_to_world_pose test!"

    print("All body_to_world_pose tests passed! ✅")


if __name__ == '__main__':
    # p1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 90.0])
    # p2 = np.array([0.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    # print("Testing relative_pose:")
    # rel = relative_pose(p1, p2, degree=True)
    # print(f"Relative pose from p1 to p2: {rel}\n")
    # test_relative_pose()
    test_body_to_world_pose()

    