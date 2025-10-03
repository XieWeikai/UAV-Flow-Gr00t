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
    Compute the relative pose from p1 to p2 using UAV-Flow's method.
    
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
    rot1 = R.from_euler('zyx', orient1, degrees=degree)
    rot2 = R.from_euler('zyx', orient2, degrees=degree)
    # relative position in p1's coordinate
    rot1_inv = rot1.inv()
    relative_pos = rot1_inv.apply(delta_pos)
    relative_rot = rot1_inv * rot2
    relative_euler = relative_rot.as_euler('zyx', degrees=degree)

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


def calculate_relative_pose(p1_coords, p2_coords):
    """
    计算 P2 相对于 P1 坐标系的位姿。

    参数:
    p1_coords (dict): P1 的坐标，包含 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
                      单位: m, °
    p2_coords (dict): P2 的坐标，同样包含 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
                      单位: m, °

    返回:
    dict: P2 相对于 P1 的坐标，包含 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
    """
    
    # --- 1. 提取并准备数据 ---
    
    # 提取 P1 的位置和姿态
    pos1 = np.array([p1_coords['x'], p1_coords['y'], p1_coords['z']])
    # scipy 的 Rotation 类默认使用 ZYX 顺序，即 yaw, pitch, roll
    # 注意：这里的 'zyx' 指的是内旋（intrinsic），等价于固定的 XYZ 轴（extrinsic）的外旋。
    # 这与我们 R = Rz * Ry * Rx 的定义是一致的。
    rot1 = R.from_euler('zyx', [p1_coords['yaw'], p1_coords['pitch'], p1_coords['roll']], degrees=True)

    # 提取 P2 的位置和姿态
    pos2 = np.array([p2_coords['x'], p2_coords['y'], p2_coords['z']])
    rot2 = R.from_euler('zyx', [p2_coords['yaw'], p2_coords['pitch'], p2_coords['roll']], degrees=True)
    
    # --- 2. 计算相对位置 ---
    
    # 在世界坐标系中计算从 P1 到 P2 的向量
    delta_pos_world = pos2 - pos1
    
    # 获取 P1 的逆旋转。这会将世界坐标系中的向量转换到 P1 的局部坐标系中。
    rot1_inv = rot1.inv()
    
    # 将 delta_pos_world 向量旋转到 P1 的坐标系中
    relative_pos = rot1_inv.apply(delta_pos_world)

    # --- 3. 计算相对姿态 ---
    
    # 相对旋转 = P1的逆旋转 * P2的旋转
    # R_rel = R1_inv * R2
    relative_rot = rot1_inv * rot2
    
    # 从相对旋转中提取欧拉角（yaw, pitch, roll），单位为度
    relative_euler = relative_rot.as_euler('zyx', degrees=True)
    relative_yaw, relative_pitch, relative_roll = relative_euler
    
    # --- 4. 格式化输出 ---
    
    relative_pose = {
        'x': relative_pos[0],
        'y': relative_pos[1],
        'z': relative_pos[2],
        'roll': relative_roll,
        'pitch': relative_pitch,
        'yaw': relative_yaw,
    }
    
    return relative_pose

def calculate_relative_pose_for_drone_control(
    p1_coords,
    p2_coords,
    orientation_axes: str | Iterable[str] | None = 'yaw',
):
    """
    Compute the relative pose for drone control (4-DoF simplified model).

    By default, the position transform only considers P1's yaw (heading),
    matching the prior behavior. You can choose any combination of roll/pitch/yaw
    to be applied to the XYZ transform by setting `orientation_axes`.

    Args:
        p1_coords (dict): Pose of P1 with keys 'x','y','z','roll','pitch','yaw' (units: m, degrees)
        p2_coords (dict): Pose of P2 with the same keys (units: m, degrees)
        orientation_axes (str | Iterable[str] | None): Which orientation components of P1
            to include when rotating the position vector from world into P1's body frame.
            Accepts:
              - strings: 'yaw', 'pitch', 'roll', any combo like 'zyx', 'zy', 'x', etc.
              - iterables: ['yaw','pitch'], ('roll',), {'z','y'}
              - None / '' / 'none' means no rotation (identity)
            Rotation order is intrinsic ZYX (yaw, then pitch, then roll), filtered by the
            selected components. For example, selecting {'yaw','pitch'} yields sequence 'zy'.

    Returns:
        dict: Relative pose of P2 in P1's frame with keys 'x','y','z','roll','pitch','yaw'.
    """
    if isinstance(orientation_axes, str):
        orientation_axes = [orientation_axes]
    # --- Normalize axis selection ---
    def _normalize_axes(sel: str | Iterable[str] | None) -> str:
        if sel is None:
            return ''
        if isinstance(sel, str):
            s = sel.strip().lower()
            if s in ('', 'none', 'identity', 'i'):
                return ''
            # Replace semantic names with axis letters
            s = (
                s.replace('yaw', 'z')
                 .replace('pitch', 'y')
                 .replace('roll', 'x')
            )
            # Keep only x/y/z letters
            letters = [ch for ch in s if ch in {'x','y','z'}]
            # Build sequence in intrinsic ZYX order
            ordered = ''.join([ax for ax in 'zyx' if ax in letters])
            return ordered
        # Iterable of names
        try:
            items = list(sel)
        except TypeError:
            raise ValueError("orientation_axes must be a string, an iterable of axis names, or None")
        mapped = []
        for item in items:
            name = str(item).strip().lower()
            if name in ('z', 'yaw'):
                mapped.append('z')
            elif name in ('y', 'pitch'):
                mapped.append('y')
            elif name in ('x', 'roll'):
                mapped.append('x')
            else:
                raise ValueError(f"Invalid axis in orientation_axes: {item}")
        # Remove duplicates, honor ZYX order
        unique = set(mapped)
        return ''.join([ax for ax in 'zyx' if ax in unique])

    axes_seq = _normalize_axes(orientation_axes)

    # --- 1. 提取数据 ---
    pos1 = np.array([p1_coords['x'], p1_coords['y'], p1_coords['z']])
    pos2 = np.array([p2_coords['x'], p2_coords['y'], p2_coords['z']])

    # --- 2. Compute relative position in the selected orientation frame ---
    
    # 在世界坐标系中计算从 P1 到 P2 的向量
    delta_pos_world = pos2 - pos1
    
    # Build rotation from P1 using selected axes (intrinsic ZYX order filtered by axes_seq)
    if axes_seq == '':
        rot1_sel_inv = R.identity()
    else:
        # Map P1 angles to ZYX order
        angles_map = {
            'z': p1_coords['yaw'],
            'y': p1_coords['pitch'],
            'x': p1_coords['roll'],
        }
        angles = [angles_map[ax] for ax in axes_seq]
        rot1_sel_inv = R.from_euler(axes_seq, angles, degrees=True).inv()
        angles_map = {
            'z': p2_coords['yaw'],
            'y': p2_coords['pitch'],
            'x': p2_coords['roll'],
        }
    
    # Rotate the world displacement vector into this selected-orientation frame
    relative_pos = rot1_sel_inv.apply(delta_pos_world)

    # --- 3. Compute relative orientation ---
    p1_pose = [
        p1_coords['yaw'] if 'yaw' in orientation_axes else 0.0,
        p1_coords['pitch'] if 'pitch' in orientation_axes else 0.0,
        p1_coords['roll'] if 'roll' in orientation_axes else 0.0
    ]
    
    p2_pose = [
        p2_coords['yaw'] if 'yaw' in orientation_axes else 0.0,
        p2_coords['pitch'] if 'pitch' in orientation_axes else 0.0,
        p2_coords['roll'] if 'roll' in orientation_axes else 0.0
    ]
    
    rot1 = R.from_euler('zyx', p1_pose, degrees=True)
    rot2 = R.from_euler('zyx', p2_pose, degrees=True)
    relative_rot = rot1.inv() * rot2
    relative_euler = relative_rot.as_euler('zyx', degrees=True)
    relative_yaw, relative_pitch, relative_roll = relative_euler
    
    # --- 4. 格式化输出 ---
    # print(f"relative_pos: {relative_pos}")
    relative_pos = relative_pos[0] if relative_pos.ndim > 1 else relative_pos
    relative_pose = {
        'x': relative_pos[0],
        'y': relative_pos[1],
        'z': relative_pos[2],
        'roll': relative_roll,
        'pitch': relative_pitch,
        'yaw': relative_yaw,
    }
    
    return relative_pose

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

# --- 示例 ---
if __name__ == '__main__':
    THIS_DIR = pathlib.Path(__file__).parent
    example_path = THIS_DIR / "example_data.json"
    with open(example_path, "r") as f:
        data = json.load(f)
        
    # 读取前两帧数据进行测试
    absolute_poses = data["raw_logs"]
    relative_poses = data["preprocessed_logs"]

    P1 = {key: absolute_poses[1][indices[key]] * sign[key] for key in indices}
    print_pose(P1, label="P1 in World Frame")
    
    P2 = {key: absolute_poses[2][indices[key]] * sign[key] for key in indices}
    print_pose(P2, label="P2 in World Frame")

    relative_p2 = array_to_dict(relative_pose(dict_to_array(P1), dict_to_array(P2), degree=True))

    print_pose(relative_p2, label="P2 relative to P1 (calculated)")

    drone_relative_p2 = array_to_dict(relative_pose_given_axes(
        dict_to_array(P1), dict_to_array(P2), 
        axes=["x", "y", "z", "yaw"],
        degree=True
    ))
    #calculate_relative_pose_for_drone_control(P1, P2, orientation_axes=['yaw'])
    print_pose(drone_relative_p2, label="P2 relative to P1 ignoring roll/pitch (calculated)")
    
    relative_p2_uavflow = UAV_Flow_relative_pose(P1, P2)
    print_pose(relative_p2_uavflow, label="P2 relative to P1 (UAV-Flow method)")
    
    # relative_p2_expected = {key: relative_poses[1][indices[key]] * sign[key] for key in indices}
    # print_pose(relative_p2_expected, label="P2 relative to P1 (expected)")

    