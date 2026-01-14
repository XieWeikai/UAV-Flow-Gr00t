from pathlib import Path
from typing import Callable, List, Dict, Union
import numpy as np
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
                    continue
        return indices

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

            # 2. Find images directory
            images_dirs = list(trajectory_dir.rglob("observation.images.rgb"))
            images_dir = images_dirs[0] if images_dirs else None

            # 3. Get all images for this episode
            images = []
            if images_dir and images_dir.exists():
                prefix = f"episode_{episode_index:06d}_"
                jpg_files = list(images_dir.glob(f"{prefix}*.jpg"))
                
                if jpg_files:
                    try:
                        # Sort by the frame index at the end
                        images = sorted(jpg_files, key=lambda x: int(x.stem.split('_')[-1]))
                    except ValueError:
                        images = sorted(jpg_files)
                elif is_single_episode:
                    # Fallback only if single episode
                    jpg_files = list(images_dir.glob("*.jpg"))
                    try:
                        images = sorted(jpg_files, key=lambda x: int(x.stem))
                    except ValueError:
                        images = sorted(jpg_files)
                    
                    if not images:
                         raise FileNotFoundError(f"No images found for single episode {episode_index} in {images_dir}")
                else:
                    raise FileNotFoundError(f"No images found with prefix {prefix} for episode {episode_index} in {images_dir} (multiple episodes present)")
            else:
                 logging.warning(f"Images directory not found in {trajectory_dir}")
                 continue

            episodes_data.append({
                "trajectory_dir": trajectory_dir,
                "parquet_path": parquet_path,
                "images_dir": images_dir,
                "images": images,
                "episode_info": ep_info
            })

        return episodes_data
        

from utils import Trajectories, Traj

def validate_tasks(tasks: List[Dict]) -> bool:
    instructions = tasks

    try:
        sum_instruction = None
        for i, ins in enumerate(instructions):
            if "sum_instruction" in ins:
                item = instructions.pop(i)
                sum_instruction = item["sum_instruction"]
                break
        
        selected_instruction = ""
        # 10% chance to select sum_instruction
        if sum_instruction is not None and np.random.rand() < 0.1:
            selected_instruction = sum_instruction
        else:
            found = False
            for ins in instructions:
                if ins["sub_indexes"][0] <= 1000 <= ins["sub_indexes"][1]:
                    if np.random.rand() < 0.5:
                        selected_instruction = ins["sub_instruction"]
                    else:
                        selected_instruction = ins["revised_sub_instruction"]
                    found = True
                    break
            if not found:
                if sum_instruction is not None:
                    selected_instruction = sum_instruction
                elif len(instructions) > 0:
                    selected_instruction = instructions[-1]["sub_instruction"]
                else:
                    # Fallback failed
                    logging.warning("validate_tasks: No instruction found and no fallback available.")
                    return False
                    
    except Exception as e:
        logging.warning(f"Error validating tasks: {e}")
        return False

    return True

class VLN_N1_Traj(Traj):
    
    def __init__(self, frames: dict, get_task_idx: Callable[[str], int], image_size: tuple[int, int] = (256, 256)):
        self.frames = frames
        self.image_size = image_size
        self.df = pd.read_parquet(frames["parquet_path"])
        self.actions = self.df['action']
        
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
            
        if not validate_tasks(tasks):
            logging.warning(f"Invalid tasks found in episode at {episode_path}, line {line_number}")

        self.task = json.dumps(tasks)

        self.images = frames["images"]
        self.task_idx = get_task_idx(self.task)
        
        assert len(self.images) == len(self.df), "Number of images and dataframe rows must match."

    def __len__(self):
        return len(self.images)
    
    
    def roll_to_horizontal(self, T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        yaw, pitch, roll = Rotation.from_matrix(R).as_euler('ZYX', degrees=True)
        
        assert abs(pitch) < 1e-3, f"Expected pitch to be ~ 0°, got {pitch}°"
        
        # in this dataset, roll is always 60° while pitch is always 0°
        # we use OpenGL convention where +x is right, +y is up, +z is backwards (for camera)
        # we want to roll the camera to be level with the horizon
        # so we fix this by rolling 90°
        roll = 90.0
        R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
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

    def __iter__(self):
        T_w_c_base = self._get_action(0) # 4x4 homogeneous transformation matrix from c_base (first frame of trajectory) to world
        
        T_w_c_base = self.roll_to_horizontal(T_w_c_base) # fix the camera orientation
        
        # compute transformation from world to c_base
        T_c_base_w = homogeneous_inv(T_w_c_base)
        
        def get_pose_at_index(idx):
             # compute pose in c_base frame
            T_w_c = self._get_action(idx)  # transformation from c (current frame) to world
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
        
        for idx in range(len(self.images)):
            # ego_view
            img = Image.open(self.images[idx])
            img = img.resize(self.image_size)
            img = np.array(img)
            
            pose = get_pose_at_index(idx)
            
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
                "action": action,
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
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": {
                "axes": ["x", "y", "z", "yaw"],
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
                        logging.info(f"Determined video resolution from data: {width}x{height}")
                        found_image = True
                        break
                except Exception as e:
                    logging.warning(f"Failed to read image size from {episodes[0]['images'][0]}: {e}")
        
        if not found_image:
            logging.warning("Could not determine image size from data. Using default resolution (256, 256).")
            
        return features

    def __init__(self, data_path: str, get_task_idx: Callable[[str], int], features: dict = None):
        self.data_path = Path(data_path)
        self.processor = InternDataProcessor(data_path)
        
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
                    traj = VLN_N1_Traj(frames, self.get_task_idx, image_size=self.image_size)
                    yield traj
                    # If yield returns, it means the consumer has accepted the trajectory
                    # We set previous_id to mark it as done on the NEXT iteration
                    previous_id = current_id
                except Exception as e:
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
         
