import cv2
from pathlib import Path
from typing import Callable, List, Dict, Union, Any
import numpy as np
import traceback
from PIL import Image
import logging
import pandas as pd
import json
from utils import Trajectories, Traj
from utils.coordinate import to_homogeneous, homogeneous_inv, relative_pose
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import multiprocessing

if multiprocessing.current_process().name == 'MainProcess':
    logging.warning("The VLN-N1 trajectory module is experimental and may contain bugs.")



class InternDataProcessor:
    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the processor with the root path of the dataset.
        e.g., 'InternData-n1-demo'
        """
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            logging.warning(f"Root path {self.root_path} does not exist.")

    # def get_trajectory_dirs(self) -> List[Path]:
    #     """
    #     Get a list of all trajectory directories (e.g., trajectory_5).
    #     """
    #     # Recursively find all directories starting with 'trajectory_'
    #     traj_dirs = list(self.root_path.rglob("trajectory_*"))
    #     # Filter to ensure they are directories
    #     traj_dirs = [d for d in traj_dirs if d.is_dir()]
    #     # Sort them for consistent order (optional but nice)
    #     traj_dirs.sort(key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else p.name)
    #     return traj_dirs

    # NOTE: It is not always 'trajectory_*', so we generalize the search
    def get_trajectory_dirs(self, limit: int = None) -> List[Path]:
        """
        Collect all directories that directly contain
        'data', 'meta', and 'videos' subdirectories.
        """
        traj_dirs = []

        # Added tqdm to show progress during directory scanning
        # for d in tqdm(self.root_path.rglob("*"), desc="Scanning directories"):
        #     if not d.is_dir():
        #         continue

        #     subdirs = {p.name for p in d.iterdir() if p.is_dir()}
        #     if {"data", "meta", "videos"}.issubset(subdirs):
        #         traj_dirs.append(d)
        #         if limit is not None and len(traj_dirs) >= limit:
        #             break
        for meta_dir in tqdm(self.root_path.rglob("meta"), desc="Scanning directories"):
            d = meta_dir.parent

            subdirs = {p.name for p in d.iterdir() if p.is_dir()}
            if {"data", "meta", "videos"}.issubset(subdirs):
                traj_dirs.append(d)
                if limit is not None and len(traj_dirs) >= limit:
                    break

        traj_dirs.sort(key=lambda p: str(p))

        return traj_dirs

    def get_episode_indices(self, trajectory_dir: Union[str, Path]) -> List[int]:
        """
        Quickly get list of episode indices from episodes.jsonl without full processing.
        """
        trajectory_dir = Path(trajectory_dir)
        episodes_path = trajectory_dir / "meta/episodes.jsonl"
        
        indices = []
        if not episodes_path.exists():
            return indices

        with open(episodes_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading episode indices", leave=False):
                line = line.strip()
                if not line: continue
                try:
                    ep_info = json.loads(line)
                    if 'episode_index' in ep_info:
                        indices.append(ep_info['episode_index'])
                except json.JSONDecodeError:
                    traceback.print_exc()
                    continue
        return indices

    def _collect_images(self, trajectory_dir: Path, folder_name: str, episode_index: int, is_single_episode: bool, extensions: List[str]) -> tuple[List[Path], Union[Path, None]]:
        target_dirs = list(trajectory_dir.rglob(folder_name))
        target_dir = target_dirs[0] if target_dirs else None
        
        if not target_dir or not target_dir.exists():
            return [], None

        prefix = f"episode_{episode_index:06d}_"
        images = []
        files = []

        # Try extensions in order
        for ext in extensions:
            files = list(target_dir.glob(f"{prefix}*{ext}"))
            if files:
                break
        
        if files:
            try:
                images = sorted(files, key=lambda x: int(x.stem.split('_')[-1]))
            except ValueError:
                traceback.print_exc()
                images = sorted(files)
        elif is_single_episode:
             # Fallback
            for ext in extensions:
                 files = list(target_dir.glob(f"*{ext}"))
                 if files:
                     break
            
            if files: 
                 try:
                    images = sorted(files, key=lambda x: int(x.stem))
                 except ValueError:
                    traceback.print_exc()
                    images = sorted(files)

        return images, target_dir

    def get_episodes_data(self, trajectory_dir: Union[str, Path]) -> List[Dict]:
        """
        Given a trajectory directory, return a list of dictionaries,
        each containing data for one episode found in episodes.jsonl.
        """
        trajectory_dir = Path(trajectory_dir)
        episodes_path = trajectory_dir / "meta/episodes.jsonl"
        
        episodes_data = []
        if not episodes_path.exists():
            logging.warning(f"Episodes metadata file not found: {episodes_path}")
            return episodes_data

        valid_ep_infos = []
        with open(episodes_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Parsing episodes JSON", leave=False)):
                line = line.strip()
                if not line: continue
                try:
                    ep_info = json.loads(line)
                    ep_info["episode_path"] = str(episodes_path)
                    ep_info["line_number"] = line_num + 1
                    if ep_info.get('episode_index') is not None:
                        valid_ep_infos.append(ep_info)
                except json.JSONDecodeError:
                    traceback.print_exc()
                    logging.warning(f"Failed to decode JSON line in {episodes_path}: {line}")
                    continue
        
        is_single_episode = (len(valid_ep_infos) == 1)

        for ep_info in tqdm(valid_ep_infos, desc="Processing episodes data", leave=False):
            episode_index = ep_info.get('episode_index')

            # 1. Find parquet file
            # Expected path: data/chunk-000/episode_{index:06d}.parquet
            parquet_name = f"episode_{episode_index:06d}.parquet"
            parquet_files = list(trajectory_dir.rglob(parquet_name))
            parquet_path = parquet_files[0] if parquet_files else None
            
            if not parquet_path:
                logging.warning(f"Parquet file {parquet_name} not found in {trajectory_dir}")
                continue

            # 2. Get RGB images
            images, images_dir = self._collect_images(trajectory_dir, "observation.images.rgb", episode_index, is_single_episode, ['.jpg', '.png'])
            
            if not images_dir:
                 logging.warning(f"Images directory not found in {trajectory_dir}")
                 continue

            if not images:
                 raise FileNotFoundError(f"No images found for episode {episode_index} in {images_dir}")

            # 3. Get Depth images
            depth_images, depth_images_dir = self._collect_images(trajectory_dir, "observation.images.depth", episode_index, is_single_episode, ['.png', '.jpg'])

            if not depth_images_dir:
                 logging.warning(f"Depth images directory not found in {trajectory_dir}")
                 continue

            if not depth_images:
                 raise FileNotFoundError(f"No depth images found for episode {episode_index} in {depth_images_dir}")

            episodes_data.append({
                "trajectory_dir": trajectory_dir,
                "parquet_path": parquet_path,
                "images_dir": images_dir,
                "images": images,
                "depth_images": depth_images, 
                "episode_info": ep_info
            })

        return episodes_data
        

from utils import Trajectories, Traj

def _is_valid_index_pair(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return False
    a, b = x[0], x[1]
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return a <= b

def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def validate_tasks(tasks: List[Dict]) -> bool:
    """
    Returns True iff tasks contains at least one valid item of either:
      1) sub item: {"sub_instruction": str, "revised_sub_instruction": str, "sub_indexes": [a,b]}
      2) sum item: {"sum_instruction": str, "sum_indexes": [a,b]}
    """
    if not isinstance(tasks, list) or len(tasks) == 0:
        return False

    for ins in tasks:
        if not isinstance(ins, dict):
            continue

        # Case 1: sub item
        if (
            (_is_nonempty_str(ins.get("sub_instruction")) or _is_nonempty_str(ins.get("revised_sub_instruction")) )
            and
            _is_valid_index_pair(ins.get("sub_indexes"))
        ):
            return True

        # Case 2: sum item
        if (
            _is_nonempty_str(ins.get("sum_instruction")) and
            _is_valid_index_pair(ins.get("sum_indexes"))
        ):
            return True
    return False

EDGE_PIX = 20     # 与边界距离阈值
PATCH_R = 5      # Patch 半径，用于遮挡判断

class Ignore(Exception):
    pass

class VLN_N1_Traj(Traj):
    
    def __init__(self, frames: dict, get_task_idx: Callable[[str], int], image_size: tuple[int, int] = (256, 256), filter_condition: dict = None):
        self.frames = frames
        self.image_size = image_size
        self.df = pd.read_parquet(frames["parquet_path"])
        self.actions = self.df["action"]
        self.K = self._get_intrinsics(0)

        # camera coordinate at the first frame
        T_w_c = self._get_action(0)

        _, _, roll = self.get_euler(T_w_c)
        roll_limit = 5.0
        if filter_condition and "roll_limit" in filter_condition:
            roll_limit = filter_condition["roll_limit"]

        if abs(90.0 - roll) > roll_limit:
            raise Ignore(f"Unexpected roll angle: {roll}°. Expected {90.0 - roll_limit}~{90.0 + roll_limit}°.")

        # camera coordinate after rolling z (-z is camera forward direction) to horizontal
        T_w_c_horizontal = self.roll_to_horizontal(T_w_c)

        # body coordinate: +x forward, +y left, +z up
        # camera coordinate: +x right, +y up, +z backwards
        T_w_b = T_w_c_horizontal.copy()
        T_w_b[:3, 0] = -T_w_c_horizontal[:3, 2]  # body +x = camera -z
        T_w_b[:3, 1] = -T_w_c_horizontal[:3, 0]   # body +y = camera -x
        T_w_b[:3, 2] = T_w_c_horizontal[:3, 1]    # body +z = camera +y

        T_b_w = homogeneous_inv(T_w_b)
        T_b_c = T_b_w @ T_w_c

        self.T_b_c = T_b_c
        

        
        if "episode_info" in frames:
            tasks = frames["episode_info"]["tasks"]
            episode_path = frames["episode_info"].get("episode_path", "unknown")
            line_number = frames["episode_info"].get("line_number", "unknown")
        else:
            # Fallback or error if episode_info is missing
            episodes_path = frames["trajectory_dir"] / "meta/episodes.jsonl"
            if not episodes_path.exists():
                raise FileNotFoundError(f"Episodes metadata file not found: {episodes_path}")
            
            with open(episodes_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tasks = data["tasks"]
            episode_path = str(episodes_path)
            line_number = 1

        # If tasks are strings (old format?), parse them. Otherwise assume dicts.
        if tasks and isinstance(tasks[0], str):
            tasks = [json.loads(j) for j in tasks]

        # keep track of sub_indexes for each task. useful for finding farthest visible frame
        sub_indexes = []
        for task in tasks:
            if "sub_indexes" in task:
                sub_indexes.append(task["sub_indexes"])
            if "sum_indexes" in task:
                sub_indexes.append(task["sum_indexes"])
        # sort by end index
        sub_indexes.sort(key=lambda x: x[1])
        self.sub_indexes = sub_indexes
            
        if not validate_tasks(tasks):
            raise ValueError(f"Invalid tasks found in episode at {episode_path}, line {line_number}")


        self.task = json.dumps(tasks)

        self.images = frames["images"]
        self.depth_images = frames["depth_images"]
        self.task_idx = get_task_idx(self.task)
        
        assert len(self.images) == len(self.df), "Number of images and dataframe rows must match."
        assert len(self.depth_images) == len(self.images), f"Number of depth images ({len(self.depth_images)}) must match RGB images ({len(self.images)})."

    @property
    def metadata(self) -> dict:
        return {
            "K": self.K.tolist(),
            "T_b_c": self.T_b_c.tolist(),
        }


    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def load_depth(depth_path: str) -> np.ndarray:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(depth_path)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth.astype(np.float32) * 0.0001

    @staticmethod
    def project_camera_point(Xc: np.ndarray, K: np.ndarray, image_shape: tuple):
        """
        Project a 3D point in camera coordinates to 2D pixel coordinates.
        Xc: (X, Y, Z) in camera coordinates where +x is right, +y is up, +z is backwards
        K: Intrinsic camera matrix
        image_shape: (height, width)
        Returns: (u, v, depth) where (u, v) are pixel coordinates and depth is the depth value
        """

        H, W = image_shape
        X, Y, Z = Xc
        depth = -Z
        if depth <= 0:
            return None  # 点在相机后侧或深度无效

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = X * fx / depth + cx
        v = H - 1 - (Y * fy / depth + cy)

        if u < 0 or u >= W or v < 0 or v >= H:
            return None  # 投影点在图像外
        return u, v, depth
    
    @staticmethod
    def is_collision_within_patch(depth, u, v, depth_proj, H, W):
        ui, vi = round(u), round(v)  # int(u), int(v)
        u0 = max(0, ui - PATCH_R)
        u1 = min(W - 1, ui + PATCH_R)
        v0 = max(0, vi - PATCH_R)
        v1 = min(H - 1, vi + PATCH_R)
        # print(f"ui: {ui}, vi: {vi}, u0: {u0}, u1: {u1}, v0: {v0}, v1: {v1} H: {H}, W: {W} depth shape: {depth.shape}")
        patch = depth[v0:v1+1, u0:u1+1]
        min_patch_depth = patch.min()
        return min_patch_depth <= depth_proj

    @staticmethod
    def is_near_edge(u, v, H, W, edge_pix=EDGE_PIX):
        return (u < edge_pix) or (u > W - edge_pix) or (v < edge_pix) or (v > H - edge_pix)
    
    @staticmethod
    def get_euler(T: np.ndarray) -> tuple[float, float, float]:
        R = T[:3, :3]
        yaw, pitch, roll = Rotation.from_matrix(R).as_euler('ZYX', degrees=True)
        return yaw, pitch, roll

    def roll_to_horizontal(self, T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        yaw, pitch, roll = self.get_euler(T)
        
        assert abs(pitch) < 1e-3, f"Expected pitch to be ~ 0°, got {pitch}°"
        
        # in this dataset, roll is always 60° while pitch is always 0°
        # we use OpenGL convention where +x is right, +y is up, +z is backwards (for camera)
        # we want to roll the camera to be level with the horizon
        # so we fix this by rolling 90°
        roll = 90.0
        R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
        # NOTE: preventing modifying the input T in-place
        T = T.copy()
        T[:3, :3] = R # now we have a horizontal camera frame
        return T
    
    def to_6d(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert a pose [x, y, z, yaw] to 6D representation.
        Here we use [x, y, z, roll, pitch, yaw] as a simple 6D representation. We just fill in roll and pitch as 0.
        """
        assert pose.shape == (4,), "Input pose must be of shape (4,)"
        x, y, z, yaw = pose
        roll = 0.0
        pitch = 0.0
        return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)
    
    def to_4d(self, pose_6d: np.ndarray) -> np.ndarray:
        """
        Convert a 6D pose [x, y, z, roll, pitch, yaw] back to 4D representation [x, y, z, yaw].
        """
        assert pose_6d.shape == (6,), "Input pose must be of shape (6,)"
        x, y, z, roll, pitch, yaw = pose_6d
        return np.array([x, y, z, yaw], dtype=np.float32)
    
    def _get_action(self, idx: int) -> np.ndarray:
        action = self.actions[idx]
        # print(f"action: {action} type: {type(action)}")
        if action.shape == (16,): # flattened 4x4 matrix
            action = action.reshape((4,4))
        # print(f"reshape action: {action} type: {type(action)}")
        processed_action = np.vstack(action)
        # print(f"processed_action: {processed_action} type: {type(processed_action)}")
        return processed_action

    def _get_intrinsics(self, idx: int) -> np.ndarray:
        K = self.df["observation.camera_intrinsic"][idx]
        if K.shape == (9,): # flattened 3x3 matrix
            K = K.reshape((3,3))
        K = np.vstack(K).astype(np.float32)
        return K

    def find_farthest_visible_frame(self, current_idx: int) -> tuple[int, np.ndarray]:
        """
        From the current frame index, find the farthest frame index
        that is still visible (not occluded) from the current frame.
        """
        W, H = self.image_size
        K = self.K
        T_w_c_current = self._get_action(current_idx)
        T_c_current_w = homogeneous_inv(T_w_c_current)

        # Pre-load depth image at current frame
        depth_img_path = self.depth_images[current_idx]
        depth_img = self.load_depth(depth_img_path)

        farthest_idx = current_idx
        farthest_T_w_c = T_w_c_current
        
        # Determine search range based on sub_indexes
        search_end = len(self.images) - 1
        for  start, end in self.sub_indexes:
            if start <= current_idx <= end:
                search_end = end
                break
        
        # Search backwards from search_end to current_idx for the first visible frame (ignoring edge constraints)
        # visible_idx = -1
        # for idx in range(search_end, current_idx, -1):
        #     T_w_c_target = self._get_action(idx)
        #     T_c_current_c_target = T_c_current_w @ T_w_c_target
        #     # Project target camera position into current camera frame
        #     p_c_target_in_current = T_c_current_c_target[:3, 3]
        #     proj = self.project_camera_point(p_c_target_in_current, K, [H, W])
            
        #     if proj is not None:
        #         u, v, depth_proj = proj
        #         # Check collision (occlusion)
        #         if not self.is_collision_within_patch(depth_img, u, v, depth_proj, H, W):
        #             visible_idx = idx
        #             break

        visible_idx = current_idx + 1
        for idx in range(current_idx + 1, search_end + 1):
            T_w_c_target = self._get_action(idx)
            T_c_current_c_target = T_c_current_w @ T_w_c_target
            # Project target camera position into current camera frame
            p_c_target_in_current = T_c_current_c_target[:3, 3]
            proj = self.project_camera_point(p_c_target_in_current, K, [H, W])
            
            if proj is None:
                break

            u, v, depth_proj = proj
            # Check collision (occlusion)
            if self.is_collision_within_patch(depth_img, u, v, depth_proj, H, W):
                break

            visible_idx = idx
        
        # If a visible frame is found, search backwards from there for the first frame that is not near the edge
        if visible_idx > current_idx and visible_idx <= search_end:
            for idx in range(visible_idx, current_idx, -1):
                T_w_c_target = self._get_action(idx)
                farthest_idx = idx
                farthest_T_w_c = T_w_c_target

                T_c_current_c_target = T_c_current_w @ T_w_c_target
                p_c_target_in_current = T_c_current_c_target[:3, 3]
                proj = self.project_camera_point(p_c_target_in_current, K, [H, W])
                
                if proj is not None:
                    u, v, depth_proj = proj
                    # Check edge constraint
                    if not self.is_near_edge(u, v, H, W):
                        break

        return farthest_idx, farthest_T_w_c


    def __iter__(self):
        T_w_c_base = self._get_action(0) # 4x4 homogeneous transformation matrix from c_base (first frame of trajectory) to world
        
        T_w_c_base = self.roll_to_horizontal(T_w_c_base) # fix the camera orientation
        
        # compute transformation from world to c_base
        T_c_base_w = homogeneous_inv(T_w_c_base)
        
        # get pose in first body frame coordinate
        def get_first_frame_pose(T_w_c):
            T_w_c = self.roll_to_horizontal(T_w_c)  # fix the camera orientation
            T_c_base_c = T_c_base_w @ T_w_c  # transformation from c (current frame) to c_base
            R_rel = Rotation.from_matrix(T_c_base_c[:3, :3])
            
            # get yaw_rel (yaw relative to c_base)
            # NOTE: The yaw I want is the yaw around the Y axis in YXZ convention
            yaw_rel, pitch_rel, roll_rel = R_rel.as_euler('YXZ', degrees=True)
            
            assert abs(pitch_rel) < 1e-3, f"Expected pitch to be ~ 0°, got {pitch_rel}°"
            assert abs(roll_rel) < 1e-3, f"Expected roll to be ~ 0°, got {roll_rel}°"
            
            p_w = T_w_c[:3, 3]  # current camera position in world frame in 3d space
            p_w = to_homogeneous(p_w) # make it homogeneous
            p_c_base = T_c_base_w @ p_w  # current camera position in c_base frame
            p_c_base = p_c_base[:3] / p_c_base[3]  # back to 3d space
            x, y, z = p_c_base
            
            # now we want to use a frame where +x is front, +y is left, +z is up
            pose = np.array([-z, -x, y, yaw_rel], dtype=np.float32)
            return pose
        
        def get_pose_at_index(idx):
            # compute pose in c_base frame
            T_w_c = self._get_action(idx)  # transformation from c (current frame) to world
            return get_first_frame_pose(T_w_c)
        
        for idx in range(len(self.images)):
            # ego_view
            img = Image.open(self.images[idx])
            img = img.resize(self.image_size)
            img = np.array(img)

            # ego_view.depth
            # depth_img = self.load_depth(self.depth_images[idx])
            
            pose = get_pose_at_index(idx)

            _, farthest_T_w_c = self.find_farthest_visible_frame(idx)
            farthest_pose = get_first_frame_pose(farthest_T_w_c)

            farthest_pose_rel = self.to_4d(
                relative_pose(
                    self.to_6d(pose), 
                    self.to_6d(farthest_pose),
                    degree=True
                )
            )
            
            if idx < len(self.images) - 1:
                next_pose = get_pose_at_index(idx + 1)
                # action is relative pose from current to next
                action = self.to_4d(
                    relative_pose(
                        self.to_6d(pose), 
                        self.to_6d(next_pose),
                        degree=True
                    )
                )
            else:
                action = np.zeros(4, dtype=np.float32)
            
            yield {
                "annotation.human.action.task_description": np.array([self.task_idx], dtype=np.int32),
                "observation.state": pose,
                "video.ego_view": img,
                # "observation.ego_view.depth": depth_img,

                # first 4 values are [dx, dy, dz, dyaw] to the next frame
                # last 4 values are relative pose of the farthest visible frame to the current frame
                "action": np.concatenate([action, farthest_pose_rel])
            }, self.task
            

class VLN_N1_Trajectories(Trajectories):
    FPS = 10
    
    ROBOT_TYPE = "UAV"
    
    FEATURES = {
        # The language instruction for the task.
        "annotation.human.action.task_description": {
            "dtype": "int32", # index of task
            "shape": (1,),
            "names": None,
        },
        # The drone's internal state (e.g., from an IMU or flight controller).
        "observation.state": {
            "dtype": "float32",
            "shape": (4,), # our current UAV can not provide full 6-DoF state, this is a placeholder that always has the value [0]
            "names": {
                "axes": ["x", "y", "z", "yaw"],
            },
        },
        # "observation.ego_view.depth": {
        #     "dtype": "uint16",
        #     "shape": (256, 256),
        #     "names": [
        #         "height",
        #         "width",
        #     ],
        # },
        # The primary video feed from the drone's ego-centric camera.
        "video.ego_view": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        },
        # The action command sent to the drone.
        # first 4 values are [dx, dy, dz, dyaw] to the next frame
        # last 4 values are relative pose of the farthest visible frame to the current frame
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": ["x", "y", "z", "yaw", "farthest_x", "farthest_y", "farthest_z", "farthest_yaw"],
            },
        },
    }
    
    INSTRUCTION_KEY = "annotation.human.action.task_description"
    
    @classmethod
    def get_features(cls, data_path: str) -> dict:
        """
        Dynamically determine features based on the dataset content.
        Specifically, it reads one image to determine the video resolution.
        """
        features = cls.FEATURES.copy()
        
        # Try to find an image to determine size
        processor = InternDataProcessor(data_path)
        traj_dirs = processor.get_trajectory_dirs(limit=6)
        
        if not traj_dirs:
            logging.warning(f"No trajectories found in {data_path}. Using default resolution (256, 256).")
            return features

        # Try to find a valid image in the first few trajectories
        found_image = False
        for traj_dir in traj_dirs[:5]: # Check first 5 trajectories
            episodes = processor.get_episodes_data(traj_dir)
            if episodes and episodes[0]["images"]:
                try:
                    with Image.open(episodes[0]["images"][0]) as img:
                        width, height = img.size
                        features["video.ego_view"]["shape"] = (height, width, 3)
                        features["video.ego_view"]["names"] = ["height", "width", "channels"]
                        # features["observation.ego_view.depth"]["shape"] = (height, width)
                        # features["observation.ego_view.depth"]["names"] = ["height", "width"]
                        logging.info(f"Determined video resolution from data: {width}x{height}")
                        found_image = True
                        break
                except Exception as e:
                    traceback.print_exc()
                    logging.warning(f"Failed to read image size from {episodes[0]['images'][0]}: {e}")
        
        if not found_image:
            logging.warning("Could not determine image size from data. Using default resolution (256, 256).")
            
        return features

    def __init__(self, data_path: str, get_task_idx: Callable[[str], int], features: dict = None, filter_condition: dict = None):
        self.data_path = Path(data_path)
        self.processor = InternDataProcessor(data_path)
        self.filter_condition = filter_condition
        
        # Use provided features or detect them
        if features is None:
            features = self.get_features(data_path)
        
        self.image_size = (features["video.ego_view"]["shape"][1], features["video.ego_view"]["shape"][0]) # (width, height)
        
        all_dirs = self.processor.get_trajectory_dirs()
        self.get_task_idx = get_task_idx
        
        self.progress_file = self.data_path / "processed_trajectories.txt"
        
        self.processed_ids = set()
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                self.processed_ids = set(line.strip() for line in f)
            logging.info(f"Loaded {len(self.processed_ids)} processed episodes from {self.progress_file}")
        
        # We keep all directories because we need to check individual episodes inside them
        self.trajectory_dirs = all_dirs
        logging.info(f"Found {len(self.trajectory_dirs)} trajectory directories.")

        # Pre-calculate total episodes to process for __len__
        self.total_episodes_to_process = 0
        for traj_dir in tqdm(self.trajectory_dirs, desc="Initializing trajectories"):
            try:
                rel_path = str(traj_dir.relative_to(self.data_path))
            except ValueError:
                rel_path = str(traj_dir)
            
            indices = self.processor.get_episode_indices(traj_dir)
            for idx in indices:
                current_id = f"{rel_path}/episode_{idx}"
                if current_id not in self.processed_ids:
                    self.total_episodes_to_process += 1
        
        logging.info(f"Total episodes to process: {self.total_episodes_to_process}")
    
    def __len__(self):
        return self.total_episodes_to_process
    
    def __iter__(self):
        previous_id = None
        
        for traj_dir in self.trajectory_dirs:
            try:
                rel_path = str(traj_dir.relative_to(self.data_path))
            except ValueError:
                rel_path = str(traj_dir)

            episodes = self.processor.get_episodes_data(traj_dir)
            for frames in episodes:
                ep_idx = frames["episode_info"]["episode_index"]
                current_id = f"{rel_path}/episode_{ep_idx}"
                
                if current_id in self.processed_ids:
                    continue
                
                # If we are about to yield a new one, mark the previous one as completed
                if previous_id is not None:
                    with open(self.progress_file, "a") as f:
                        f.write(f"{previous_id}\n")
                    self.processed_ids.add(previous_id)
                    previous_id = None
                
                try:
                    traj = VLN_N1_Traj(frames, self.get_task_idx, image_size=self.image_size, filter_condition=self.filter_condition)
                    yield traj
                    # If yield returns, it means the consumer has accepted the trajectory
                    # We set previous_id to mark it as done on the NEXT iteration
                    previous_id = current_id
                except Ignore as ig:
                    logging.info(f"Skipping episode {current_id}: {ig}")
                    continue
                except Exception as e:
                    traceback.print_exc()
                    logging.warning(f"Skipping episode {current_id} due to error: {e}")
                    continue
            
        # Mark the last trajectory as completed if we finished the loop
        if previous_id is not None:
            with open(self.progress_file, "a") as f:
                f.write(f"{previous_id}\n")
            self.processed_ids.add(previous_id)

    @property
    def schema(self) -> dict:
        return self.FEATURES
        

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Assuming the script is run from the project root or similar
    # Adjust the path as necessary for testing
    data_path = "InternData-n1-demo/hm3d_zed/00001-UVdNNRcVyV1"
    # data_path = "demo_data"
    root_dir = Path(data_path)
    
    trajectories = VLN_N1_Trajectories(root_dir, get_task_idx=lambda x: 0)
    
    for traj in trajectories:
        logging.info(f"Trajectory with {len(traj)} frames.")
        for frame, task in traj:
            print(f"Frame keys: {list(frame.keys())}")
         
