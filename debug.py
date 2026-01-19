from typing import Tuple
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation
from utils.coordinate import to_homogeneous, homogeneous_inv, relative_pose, body_to_world_pose

H = 270
W = 480

# parquet_path = "/home/ubuntu/xwk/projects/UAV-Flow-Gr00t/test_data/data/chunk-000/episode_000000.parquet"
parquet_path = "/data-25T/InternData-N1/hssd_d435i/102344280__d2d2d5fd7e/102344280/data/chunk-000/episode_000000.parquet"

df = pd.read_parquet(parquet_path)

def get_intrinsics(idx: int) -> np.ndarray:
    K = df["observation.camera_intrinsic"][idx]
    if K.shape == (9,): # flattened 3x3 matrix
        K = K.reshape((3,3))
    K = np.vstack(K).astype(np.float32)
    return K


def get_extrinsics(idx: int) -> np.ndarray:
    T_c_b = df["observation.camera_extrinsic"][idx]
    if T_c_b.shape == (16,): # flattened 4x4 matrix
        T_c_b = T_c_b.reshape((4,4))
    T_c_b = np.vstack(T_c_b).astype(np.float32)
    return T_c_b

def get_T_w_c(idx: int) -> np.ndarray:
    action = df["action"][idx]
    if action.shape == (16,): # flattened 4x4 matrix
        action = action.reshape((4,4))
    processed_action = np.vstack(action)
    return processed_action

def R(T: np.array) -> np.ndarray:
    return T[:3, :3]

def t(T: np.array) -> np.ndarray:
    return T[:3, 3]

def euler(T: np.ndarray) -> Tuple[float, float, float]:
    yaw, pitch, roll = Rotation.from_matrix(T[:3, :3]).as_euler('ZYX', degrees=True)
    return yaw, pitch, roll


def get_T_w_b(T_w_c: np.ndarray) -> tuple[np.ndarray, float]:
    yaw, pitch, roll = euler(T_w_c)
    assert abs(pitch) < 1e-3, f"Expected pitch to be ~ 0°, got {pitch}°"
    # pitch is always near 0°
    # we use OpenGL convention where +x is right, +y is up, +z is backwards (for camera)
    # the camera is not faceing horizontally
    # we want to roll the camera to be level with the horizon
    # so we fix this by rolling 90°
    ori_roll = roll
    roll = 90.0
    R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
    T_w_b = T_w_c.copy()
    T_w_b[:3, 0] = -R[:, 2]   # body +x = camera -z
    T_w_b[:3, 1] = -R[:, 0]   # body +y = camera -x
    T_w_b[:3, 2] = R[:, 1]    # body +z = camera +y
    return T_w_b, yaw, ori_roll

def project_camera_point(t: np.ndarray, K: np.ndarray, image_shape: tuple):
    """
    Project a 3D point in camera coordinates to 2D pixel coordinates.
    Xc: (X, Y, Z) in camera coordinates where +x is right, +y is up, +z is backwards
    K: Intrinsic camera matrix
    image_shape: (height, width)
    Returns: (u, v, depth) where (u, v) are pixel coordinates and depth is the depth value
    """

    H, W = image_shape
    x, y, z = t
    depth = -z

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = x * fx / depth + cx
    v = H - 1 - (y * fy / depth + cy)

    return u, v, depth

def compute_T_b_c_and_T_c_b()->tuple[np.array, np.array]:
    T_w_c = get_T_w_c(0)
    T_w_b, _, _ = get_T_w_b(T_w_c)
    T_b_w = homogeneous_inv(T_w_b)
    T_b_c = T_b_w @ T_w_c
    T_c_b = homogeneous_inv(T_b_c)
    return T_b_c, T_c_b

T_b_c, T_c_b = compute_T_b_c_and_T_c_b()
K = get_intrinsics(0)

if __name__ == "__main__":
    # body coordinate: +x forward, +y left, +z up
    # camera coordinate: +x right, +y up, +z backwards
    
    print("T_b_c:\n", T_b_c)
    print("T_c_b:\n", T_c_b)
    print("Intrinsic K:\n", K)
    print("\n\n")


    T_w_c0 = get_T_w_c(0)
    T_c0_w = homogeneous_inv(T_w_c0)
    for i in range(1, 30):
        T_w_ci = get_T_w_c(i)


        # T_c0_ci = T_c0_w @ T_w_ci
        
        # # ci camera在c0 camera的坐标系下的坐标
        # t_ci = t(T_c0_ci)
        # print(f"t_ci: {t_ci}")
        # u, v, depth = project_camera_point(t_ci, K, (H, W))
        # print(f"Projected pixel coordinates in c0 frame: u={u}, v={v}, depth={depth}")
        
        
        T_w_b0, _, _ = get_T_w_b(T_w_c0)
        T_w_bi, yaw, roll = get_T_w_b(T_w_ci)
        T_w_bi[2, 2] = 0.0  # set z to 0 (ground level)
        T_b0_w = homogeneous_inv(T_w_b0)
        T_b0_bi = T_b0_w @ T_w_bi
        t_bi = t(T_b0_bi)
        # print(f"t_bi: {t_bi}")

        t_ci = T_c_b @ to_homogeneous(t_bi)
        t_ci = t_ci[:3] / t_ci[3]
        # print(f"t_ci: {t_ci}")
        u, v, depth = project_camera_point(t_ci, K, (H, W))
        print(f"yaw: {yaw}°, roll: {roll}° - Projected pixel coordinates in c0 frame from body coord: u={u}, v={v}, depth={depth}")

    # for i in range(1):
    #     T_c_b = homogeneous_inv(get_extrinsics(i))
    #     T_w_c = get_T_w_c(i)
    #     T_w_b = T_w_c @ T_c_b
    #     yaw, pitch, roll = euler(T_w_c)
    #     print(f"Initial T_w_c Euler angles: yaw={yaw}°, pitch={pitch}°, roll={roll}°")
    #     yaw, pitch, roll = euler(T_w_b)
    #     print(f"Initial T_w_b Euler angles: yaw={yaw}°, pitch={pitch}°, roll={roll}°")

    #     T_w_c_, _, _ = get_T_w_b(T_w_c)
    #     T_w_b_, _, _ = get_T_w_b(T_w_b)
    #     T_b_c = homogeneous_inv(T_w_b_) @ T_w_c_

    #     print(f"t_c: {t(T_w_c)} t_b: {t(T_w_b)} t_ref: {t(T_b_c)}")

    
