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

class InternDataProcessor:
    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the processor with the root path of the dataset.
        e.g., 'InternData-n1-demo'
        """
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            logging.warning(f"Root path {self.root_path} does not exist.")

    def get_trajectory_dirs(self) -> List[Path]:
        """
        Get a list of all trajectory directories (e.g., trajectory_5).
        """
        # Recursively find all directories starting with 'trajectory_'
        traj_dirs = list(self.root_path.rglob("trajectory_*"))
        # Filter to ensure they are directories
        traj_dirs = [d for d in traj_dirs if d.is_dir()]
        # Sort them for consistent order (optional but nice)
        traj_dirs.sort(key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else p.name)
        return traj_dirs

    def get_trajectory_data(self, trajectory_dir: Union[str, Path]) -> Dict:
        """
        Given a trajectory directory, return the paths to the parquet file,
        the images directory, and a list of image files.
        """
        trajectory_dir = Path(trajectory_dir)
        
        # 1. Find parquet file
        # Expected path: data/chunk-000/episode_000000.parquet
        # We search recursively for any .parquet file to be robust
        parquet_files = list(trajectory_dir.rglob("*.parquet"))
        parquet_path = parquet_files[0] if parquet_files else None

        # 2. Find images directory
        # Expected path: videos/chunk-000/observation.images.rgb/
        # We search for 'observation.images.rgb' directory
        images_dirs = list(trajectory_dir.rglob("observation.images.rgb"))
        images_dir = images_dirs[0] if images_dirs else None

        # 3. Get all images
        images = []
        if images_dir and images_dir.exists():
            # Get all .jpg files
            jpg_files = list(images_dir.glob("*.jpg"))
            # Sort by numeric filename (e.g., 0.jpg, 1.jpg, 10.jpg)
            # Assuming filenames are integers
            try:
                images = sorted(jpg_files, key=lambda x: int(x.stem))
            except ValueError:
                # Fallback to string sort if filenames are not purely numeric
                images = sorted(jpg_files)

        return {
            "trajectory_dir": trajectory_dir,
            "parquet_path": parquet_path,
            "images_dir": images_dir,
            "images": images
        }
        

from utils import Trajectories, Traj

class VLN_N1_Traj(Traj):
    IMAGE_SIZE = (256, 256)
    
    def __init__(self, frames: dict, get_task_idx: Callable[[str], int]):
        self.frames = frames
        self.df = pd.read_parquet(frames["parquet_path"])
        self.actions = self.df['action']
        episodes_path = frames["trajectory_dir"] / "meta/episodes.jsonl"
        with open(episodes_path, 'r', encoding='utf-8') as f:
            task = json.load(f)
        task = task["tasks"]
        tasks = [json.loads(j) for j in task]
        tasks_str = json.dumps(tasks)
        self.task = tasks_str
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
        
    
    def __iter__(self):
        T_w_c_base = np.vstack(self.actions[0]) # 4x4 homogeneous transformation matrix from c_base (first frame of trajectory) to world
        
        T_w_c_base = self.roll_to_horizontal(T_w_c_base) # fix the camera orientation
        
        # compute transformation from world to c_base
        T_c_base_w = homogeneous_inv(T_w_c_base)
        
        # last pose in c_base frame
        last_pose = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for idx in range(len(self.images)):
            # ego_view
            img = Image.open(self.images[idx])
            img = img.resize(self.IMAGE_SIZE)
            img = np.array(img)
            
            # compute pose in c_base frame
            T_w_c = np.vstack(self.actions[idx])  # transformation from c (current frame) to world
            T_w_c = self.roll_to_horizontal(T_w_c)  # fix the camera orientation
            T_c_base_c = T_c_base_w @ T_w_c  # transformation from c (current frame) to c_base
            R_rel = Rotation.from_matrix(T_c_base_c[:3, :3])
            
            # get yaw_rel (yaw relative to c_base)
            # NOTE: The yaw I want is the yaw around the Y axis in YXZ convention
            yaw_rel, pitch_rel, roll_rel = R_rel.as_euler('YXZ', degrees=True)
            
            # if abs(pitch_rel) > 1e-3 or abs(roll_rel) > 1e-3:
            #     logging.warning(f"path: {self.frames['trajectory_dir']}, frame idx: {idx}")
            #     logging.warning(f"Expected pitch and roll to be ~0°, got pitch: {pitch_rel}°, roll: {roll_rel}°")
            
            assert abs(pitch_rel) < 1e-3, f"Expected pitch to be ~ 0°, got {pitch_rel}°"
            assert abs(roll_rel) < 1e-3, f"Expected roll to be ~ 0°, got {roll_rel}°"
            
            p_w = T_w_c[:3, 3]  # current camera position in world frame in 3d space
            p_w = to_homogeneous(p_w) # make it homogeneous
            p_c_base = T_c_base_w @ p_w  # current camera position in c_base frame
            p_c_base = p_c_base[:3] / p_c_base[3]  # back to 3d space
            x, y, z = p_c_base
            
            # in camera base frame, +x is right, +y is up, +z is backwards
            # and we want to use a frame where +x is right, +y is forwards, +z is up
            pose = np.array([x, -z, y, yaw_rel], dtype=np.float32) # this is in our desired frame where +x is right, +y is forwards, +z is up
            
            # action is relative pose to last pose
            action = self.to_4d(
                relative_pose(
                    self.to_6d(last_pose), 
                    self.to_6d(pose)
                )
            )
            
            yield {
                "annotation.human.action.task_description": np.array([self.task_idx], dtype=np.int32),
                "observation.state": pose,
                "video.ego_view": img,
                "action": action,
            }, self.task
            
            last_pose = pose
            

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
    
    def __init__(self, data_path: str, get_task_idx: Callable[[str], int]):
        self.data_path = Path(data_path)
        self.processor = InternDataProcessor(data_path)
        all_dirs = self.processor.get_trajectory_dirs()
        self.get_task_idx = get_task_idx
        
        self.progress_file = self.data_path / "processed_trajectories.txt"
        
        self.processed_dirs = set()
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                self.processed_dirs = set(line.strip() for line in f)
            logging.info(f"Loaded {len(self.processed_dirs)} processed trajectories from {self.progress_file}")
            logging.info(f"total trajectories found: {len(all_dirs)}")
        
        # Filter out processed directories
        self.trajectory_dirs = []
        for d in all_dirs:
            try:
                rel_path = d.relative_to(self.data_path)
                if str(rel_path) not in self.processed_dirs:
                    self.trajectory_dirs.append(d)
            except ValueError:
                logging.warning(f"Path {d} is not relative to {self.data_path}. Processing it anyway.")
                self.trajectory_dirs.append(d)
        logging.info(f"{len(self.trajectory_dirs)} trajectories to process after filtering.")
    
    def __len__(self):
        return len(self.trajectory_dirs)
    
    def __iter__(self):
        previous_traj_rel = None
        
        for traj_dir in self.trajectory_dirs:
            try:
                current_traj_rel = str(traj_dir.relative_to(self.data_path))
            except ValueError:
                current_traj_rel = str(traj_dir)

            # If we are starting a new trajectory, mark the previous one as completed
            if previous_traj_rel is not None:
                with open(self.progress_file, "a") as f:
                    f.write(f"{previous_traj_rel}\n")
                self.processed_dirs.add(previous_traj_rel)
            
            frames = self.processor.get_trajectory_data(traj_dir)
            yield VLN_N1_Traj(frames, self.get_task_idx)
            
            previous_traj_rel = current_traj_rel
            
        # Mark the last trajectory as completed if we finished the loop
        if previous_traj_rel is not None:
            with open(self.progress_file, "a") as f:
                f.write(f"{previous_traj_rel}\n")
            self.processed_dirs.add(previous_traj_rel)
        

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
        
