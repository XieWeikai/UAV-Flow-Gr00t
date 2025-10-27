from typing import Any, Dict, Optional, Dict, Tuple
from PIL import Image
from abc import ABC, abstractmethod
import numpy as np
import requests
import json
from io import BytesIO
import base64
import json_numpy
from scipy.spatial.transform import Rotation as R

json_numpy.patch()

# ====== Constants ======
IMG_INPUT_SIZE_OPENVLA: Tuple[int, int] = (224, 224)

def send_prediction_request(image: Image, proprio: np.ndarray, instr: str, server_url: str) -> Optional[Dict[str, Any]]:
    """Send a request to the inference service and return JSON response.

    Args:
        image: PIL image object, resized to 224x224 and sent as PNG.
        proprio: Vehicle state vector (np.ndarray), converted to list.
        instr: Text instruction.
        server_url: Inference service /predict endpoint URL.
    Returns:
        dict or None: Parsed JSON if successful, otherwise None on error.
    """
    proprio_list = proprio.tolist()
    img_io = BytesIO()
    if image.size != IMG_INPUT_SIZE_OPENVLA:
        image = image.resize(IMG_INPUT_SIZE_OPENVLA)
    image.save(img_io, format='PNG')
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    payload: Dict[str, Any] = {
        'image': img_base64,
        'proprio': proprio_list,
        'instr': instr
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(
        server_url,
        data=json.dumps(payload),
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

class PolicyClient(ABC):
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
    
    @abstractmethod
    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        pass
    
class OpenVLAClient(PolicyClient):
    def __init__(self, ip: str = "localhost", port: int = 5007):
        super().__init__(ip, port)
        self.server_url = f"http://{self.ip}:{self.port}/predict"
    
    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        image = observation.get('image')
        proprio = observation.get('proprio')
        instr = observation.get('instr')
        if image is None or proprio is None or instr is None:
            raise ValueError("Observation must contain 'image', 'proprio', and 'instr' keys")
        if not isinstance(image, Image.Image):
            raise TypeError("'image' must be a PIL Image")
        if not isinstance(proprio, np.ndarray):
            raise TypeError("'proprio' must be a numpy ndarray")
        if not isinstance(instr, str):
            raise TypeError("'instr' must be a string")
        
        response = send_prediction_request(image, proprio, instr, self.server_url)
        if response is None:
            raise RuntimeError("Failed to get valid response from server")
        # NOTE: yaw is in radians, different from Gr00tClient which uses degrees
        return response    
    
def get_action_from_server(data: dict[str, Any], ip: str = "127.0.0.1", port: int = 5555):
    keys = ["video.ego_view", "state.drone", "annotation.human.action.task_description"]
    obs = {key: data[key] for key in keys}
    response = requests.post(
        f"http://{ip}:{port}/act",
        json={"observation": obs},
    )
    action = response.json()
    return action


def to6d(p: np.array)-> np.array:
    """
    Convert a pose to 6-DOF format [x,y,z,roll,pitch,yaw].

    Args:
        p: Pose as length-4 ([x,y,z,yaw]) or length-6 ([x,y,z,roll,pitch,yaw]) array.
    Returns:
        np.array: Pose in length-6 format.
    """
    a = np.asarray(p)
    if a.size == 6:
        return a.astype(float)
    if a.size == 4:
        return np.array([a[0], a[1], a[2], 0.0, 0.0, a[3]], dtype=float)
    raise ValueError("Pose must have length 4 ([x,y,z,yaw]) or 6 ([x,y,z,roll,pitch,yaw])")

def to4d(p: np.array)-> np.array:
    """
    Convert a pose to 4-DOF format [x,y,z,yaw].

    Args:
        p: Pose as length-4 ([x,y,z,yaw]) or length-6 ([x,y,z,roll,pitch,yaw]) array.
    Returns:
        np.array: Pose in length-4 format.
    """
    a = np.asarray(p)
    if a.size == 4:
        return a.astype(float)
    if a.size == 6:
        return np.array([a[0], a[1], a[2], a[5]], dtype=float)
    raise ValueError("Pose must have length 4 ([x,y,z,yaw]) or 6 ([x,y,z,roll,pitch,yaw])")

def relative_pose(p1: np.array, p2: np.array, degree=False) -> np.array:
    """
    Compute the relative pose from p1 to p2.

    This function accepts p1 and p2 in either 6-element format
    ([x,y,z,roll,pitch,yaw]) or 4-element format ([x,y,z,yaw]). If a pose
    is provided with 4 elements, roll and pitch are assumed zero.

    The returned array matches the format of the provided p2: if p2 has 6
    elements, a 6-element relative pose [x,y,z,roll,pitch,yaw] is returned;
    if p2 has 4 elements, a 4-element relative pose [x,y,z,yaw] is returned.

    Args:
        p1: Pose of P1 as length-4 or length-6 array.
        p2: Pose of P2 as length-4 or length-6 array.
        degree: Whether angles are in degrees. Defaults to False (radians).

    Returns:
        np.array: Relative pose of P2 in P1's frame, length matches p2 (4 or 6).
    """
    p1_arr = np.asarray(p1)
    p2_arr = np.asarray(p2)
    p2_sz = p2_arr.size

    p1_6 = to6d(p1_arr)
    p2_6 = to6d(p2_arr)

    # Extract positions and orientations
    pos1 = p1_6[:3]
    pos2 = p2_6[:3]
    orient1 = p1_6[[-1, -2, -3]]
    orient2 = p2_6[[-1, -2, -3]]

    delta_pos = pos2 - pos1

    # Compute relative orientation
    rot1 = R.from_euler('ZYX', orient1, degrees=degree)
    rot2 = R.from_euler('ZYX', orient2, degrees=degree)
    # relative position in p1's coordinate
    rot1_inv = rot1.inv()
    relative_pos = rot1_inv.apply(delta_pos)
    relative_rot = rot1_inv * rot2
    relative_euler = relative_rot.as_euler('ZYX', degrees=degree)

    # Combine relative position and orientation
    rel_rpy = relative_euler[::-1]

    if p2_sz == 6:
        return np.concatenate([relative_pos, rel_rpy])
    else:
        # return [x,y,z,yaw]
        return np.array([relative_pos[0], relative_pos[1], relative_pos[2], rel_rpy[2]])

def body_to_world_pose(p1: np.ndarray, p2: np.ndarray, degree: bool = False) -> np.ndarray:
    """
    Transform a pose given in the body (drone) frame into the world (origin) frame.

    This merged function accepts either full 6-DOF poses [x, y, z, roll, pitch, yaw]
    or 4-element poses [x, y, z, yaw] (in which case roll and pitch are assumed 0.0).

    Args:
        p1: Pose of the body frame origin in world. Either length 6 ([x,y,z,roll,pitch,yaw])
            or length 4 ([x,y,z,yaw]). If length 4, roll and pitch are assumed 0.0.
        p2: Pose expressed in the body frame. Either length 6 or 4 (same convention as p1).
        degree: Whether the roll/pitch/yaw values are in degrees. Defaults to False (radians).

    Returns:
        np.ndarray: Array [x, y, z, yaw] giving the position and yaw of p2 in the world frame.

    Note:
        Internally the function uses the same Euler convention as before: scipy Rotation
        expects angles in the order (yaw, pitch, roll) for the 'ZYX' sequence.
    """
    # remember original sizes so we can return matching-length result
    p1_arr = np.asarray(p1)
    p2_arr = np.asarray(p2)
    sz = p2_arr.size

    p1_6 = to6d(p1_arr)
    p2_6 = to6d(p2_arr)

    # positions
    pos1 = p1_6[:3]
    pos2 = p2_6[:3]

    # orientations: convert from [roll, pitch, yaw] to [yaw, pitch, roll] for 'ZYX'
    orient1 = p1_6[[-1, -2, -3]]
    orient2 = p2_6[[-1, -2, -3]]

    # build rotations
    rot1 = R.from_euler('ZYX', orient1, degrees=degree)
    rot2 = R.from_euler('ZYX', orient2, degrees=degree)

    # transform position and compose rotations
    world_pos = rot1.apply(pos2) + pos1
    world_rot = rot1 * rot2
    world_euler = world_rot.as_euler('ZYX', degrees=degree)

    # prepare outputs
    # world_euler currently is [yaw, pitch, roll] due to 'ZYX' ordering; reverse to [roll,pitch,yaw]
    world_rpy = world_euler[::-1]

    if sz == 6:
        # return full 6-DOF [x, y, z, roll, pitch, yaw]
        return np.concatenate([world_pos, world_rpy])
    else:
        # return [x, y, z, yaw]
        world_yaw = world_rpy[2]
        return np.array([world_pos[0], world_pos[1], world_pos[2], world_yaw])
    

GR00T_IMAGE_SIZE = (256, 256)

class Gr00tClient(PolicyClient):
    def __init__(self, ip: str = "localhost", port: int = 5555, horizon=4):
        super().__init__(ip, port)
        self.horizon = horizon

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        first_image = observation.get('first_image')
        curr_image = observation.get('image')
        proprio = observation.get('proprio')
        instr = observation.get('instr')
        # Resize image to 256x256
        first_image = first_image.resize(GR00T_IMAGE_SIZE)
        curr_image = curr_image.resize(GR00T_IMAGE_SIZE)
        # concat first_image and curr_image to [2, H, W, 3]
        image_np = np.concatenate([np.array(first_image)[None, :], np.array(curr_image)[None, :]], axis=0)
        state = proprio[None, :]
        observation = {
            "video.ego_view": image_np,
            "state.drone": state,
            "annotation.human.action.task_description": [instr]
        }
        # print(f"video.ego_view shape: {observation['video.ego_view'].shape} dtype: {observation['video.ego_view'].dtype}")
        # print(f"state.drone shape: {observation['state.drone'].shape} dtype: {observation['state.drone'].dtype}")
        # print(f"annotation.human.action.task_description: {observation['annotation.human.action.task_description']}")
        
        delta_poses = get_action_from_server(observation, self.ip, self.port)["action.delta_pose"]
        abs_poses = []
        for delta in delta_poses:
            # previous relative_to_absolute returned [x,y,z,yaw]; our merged
            # body_to_world_pose now does the same when given 4-element poses
            proprio = body_to_world_pose(proprio, delta, degree=True)
            abs_poses.append(proprio.tolist())

        # print(f"len(delta_poses): {len(delta_poses)}")
        # print(f"len(abs_poses): {len(abs_poses)}")

        # NOTE: Gr00t actions tuple [x, y, z, yaw], yaw is in degrees.    UAV-Flow-OpenVLA use radians.
        return {
            "action": abs_poses[:self.horizon],
            "action_ori": delta_poses.tolist()[:self.horizon],
        }     

    
if __name__ == "__main__":
    open_vla_client = OpenVLAClient(ip="localhost", port=5007)
    gr00t_client = Gr00tClient(ip="localhost", port=5555)
    import os
    import pathlib
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image = Image.open(pathlib.Path(current_dir) / "test.jpg")
    proprio = np.array([0.0, 0.0, 0.0, 0.0])  # Dummy proprioceptive data
    instr = "Navigate to the target location."
    
    # OpenVLA action
    action = open_vla_client.get_action({
        'image': image,
        'proprio': proprio,
        'instr': instr
    })
    print("Action:", json.dumps(action, indent=2))
    
    # Gr00t action
    action = gr00t_client.get_action({
        'first_image': image,
        'image': image,
        'proprio': proprio,
        'instr': instr
    })
    print("Action:", json.dumps(action, indent=2))

