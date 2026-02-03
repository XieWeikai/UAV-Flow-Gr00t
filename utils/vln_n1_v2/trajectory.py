import logging
import numpy as np
import json
from tqdm import tqdm
from typing import Callable, Iterable, Optional, Tuple, List, Dict
import traceback
from pathlib import Path
import pandas as pd
import open3d as o3d
from PIL import Image
import cv2
import scipy.ndimage

from utils import Traj, Trajectories
from scipy.spatial.transform import Rotation
from utils.coordinate import homogeneous_inv, get_poses, PointCloudESDF
from utils.obstacle import compute_collision_prob, compute_yaw_rate

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
        traceback.print_exc()
        logging.warning(f"Error validating tasks: {e}")
        return False

    return True


class Ignore(Exception):
    pass


class VLN_N1_V2_Traj(Traj):
    PATCH_SIZE = 11
    EDGE = 20
    COLLISION_THRESHOLD = 0.8
    SMOOTH_WINDOW = 10
    DT = 0.1 # 1 / 10 FPS
    
    _filter: Optional[Callable[["VLN_N1_V2_Traj"], None]] = None

    @classmethod
    def set_filter(cls, filter: Callable[["VLN_N1_V2_Traj"], None]):
        cls._filter = filter

    def __init__(self, parquet_path: Path, esdf: PointCloudESDF, images: List[Path], depth_images: List[Path], task: str, task_idx: int):
        self.df = pd.read_parquet(parquet_path)

        # intrinsic matrix 3x3
        K = self.df["observation.camera_intrinsic"][0]
        K = K.reshape((3,3))
        K = np.vstack(K).astype(np.float32)
        self.K = K

        # extrinsic matrices 4x4
        T_w_c = self.df["action"][0]
        T_w_c = T_w_c.reshape((4, 4))
        self.T_b_c, self.T_c_b, self.ori_roll = VLN_N1_V2_Traj._compute_T_b_c_and_T_c_b(T_w_c)

        # pointcloud
        self.esdf = esdf

        # images and other data
        self.images = images
        self.depth_images = depth_images # Added depth images
        
        # get H and W from the first image
        with Image.open(self.images[0]) as img:
            self.W, self.H = img.size  # (W, H)
        self.task = task
        self.task_idx = task_idx
        assert len(self.df) == len(self.images), "Number of images and frames must match."

        if VLN_N1_V2_Traj._filter is not None:
            VLN_N1_V2_Traj._filter(self)
        
        # preprocessing all data
        self.process_traj()

    def __getstate__(self):
        # Only preserve attributes needed for __iter__ and metadata
        return {
            "df": self.df,
            "images": self.images,
            "task": self.task,
            "task_idx": self.task_idx,
            "state": self.state,
            "actions": self.actions,
            "T_b_c": self.T_b_c,
            "K": self.K,
        }


    def __len__(self)-> int:
        return len(self.df)
    
    @staticmethod
    def load_depth(depth_path: str) -> np.ndarray:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth image from {depth_path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth.astype(np.float32) * 0.0001
    
    def find_farthest_visible_frame_vectorized(self, current_idx: int, min_depth_map: np.ndarray, T_w_c_all: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Vectorized implementation of searching for the farthest visible frame.
        """
        N = len(T_w_c_all)
        if current_idx >= N - 1:
            return current_idx, T_w_c_all[current_idx]

        # 1. Lookahead Candidates
        # Determine search end strictly. V1 used sub_indexes tasks logic.
        # Here we assume we can look up to the end of the trajectory.
        search_range = np.arange(current_idx + 1, N)
        T_w_targets = T_w_c_all[search_range] # [M, 4, 4]
        
        # 2. Transform to Current Camera Frame
        T_w_curr = T_w_c_all[current_idx]
        T_curr_w = homogeneous_inv(T_w_curr)
        # (4,4) @ (M, 4, 4) -> (M, 4, 4)
        T_curr_targets = np.matmul(T_curr_w, T_w_targets)
        
        # 3. Project to Image Plane
        # Position in current camera frame: T[:3, 3]
        P_c = T_curr_targets[:, :3, 3] # [M, 3] (X, Y, Z)
        
        # Camera Axes: +x right, +y up, +z backwards
        # Depth d = -Z
        X = P_c[:, 0]
        Y = P_c[:, 1]
        Z = P_c[:, 2]
        
        D = -Z 
        
        # 4. Filter Invalid Points
        # Min depth check
        valid_mask = D > 0.1 
        
        # Project
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        H, W = self.H, self.W
        
        # Note: V1 project_camera_point logic:
        # u = X * fx / depth + cx
        # v = H - 1 - (Y * fy / depth + cy)
        
        u = (X * fx / D) + cx
        v = (H - 1) - ((Y * fy / D) + cy)
        
        # Edge check
        in_bounds = (u >= self.EDGE) & (u <= W - self.EDGE) & (v >= self.EDGE) & (v <= H - self.EDGE)
        
        valid_mask = valid_mask & in_bounds
        
        # Optimization: If no points are valid in bounds/depth, return current
        if not np.any(valid_mask):
            return current_idx, T_w_c_all[current_idx]

        # 5. Visibility Check (Occlusion) using Min-Filter on Depth
        # min_depth_map is passed in now to avoid recomputing it inside
        
        # Sample safe depth at projection points
        u_int = np.clip(np.round(u), 0, W-1).astype(int)
        v_int = np.clip(np.round(v), 0, H-1).astype(int)
        
        measured_min_depths = min_depth_map[v_int, u_int]
        
        # Visibility Condition:
        # If min_depth_in_patch <= projected_depth, then there is an obstacle closer (or equal) -> occluded.
        # So we need min_depth_in_patch > projected_depth
        # Allow small margin? V1 uses strict comparison.
        is_visible = measured_min_depths > D # implicit margin if needed, or strict D
        
        final_mask = valid_mask & is_visible
        
        # 6. Find Farthest
        # Indices of 'search_range' that are true
        valid_indices = search_range[final_mask]
        
        if len(valid_indices) == 0:
            return current_idx, T_w_c_all[current_idx]
        
        farthest_idx = valid_indices.max()
        return farthest_idx, T_w_c_all[farthest_idx]
    
    def process_traj(self) -> None:
        # camera->world [N, 4, 4]
        T_w_c = np.vstack(self.df["action"].to_numpy()).reshape(-1, 4, 4)

        # body->world [N, 4, 4] x [4, 4] = [N, 4, 4]
        T_w_b = T_w_c @ self.T_c_b
        # [4, 4] first frame
        T_w_b0 = T_w_b[0]
        T_b0_w = homogeneous_inv(T_w_b0)
        # world -> body0 [N, 4, 4]
        # [4, 4] x [N, 4, 4] = [N, 4, 4]
        T_b0_b = T_b0_w @ T_w_b
        # poses in the body0 frame [N, 4]
        # each row: x, y, z (m), yaw (deg)
        state = get_poses(T_b0_b)
        
        poses_world = get_poses(T_w_b)
        
        # 1. Collision Probability
        probs = compute_collision_prob(poses_world, self.esdf, dt=self.DT)
        
        # 2. Yaw Rate (deg/s)
        # Using body frame yaw or world frame yaw? compute_yaw_rate expects yaw series.
        # poses_world[:, 3] is yaw in degrees.
        yaw_rates = compute_yaw_rate(poses_world[:, 3], dt=self.DT, smoothing_window=self.SMOOTH_WINDOW)
        
        # 3. Filter
        # If prob > threshold, use yaw_rate, else 0.0
        collision_yaw_rate = np.zeros_like(yaw_rates)
        mask = probs > self.COLLISION_THRESHOLD
        collision_yaw_rate[mask] = yaw_rates[mask]
        
        # Append to state [x, y, z, yaw, collision_yaw_rate]
        # state is [N, 4]
        self.state = np.hstack([state, collision_yaw_rate[:, None]]) # [N, 5]

        # [N - 1, 4, 4]
        T_b0__b_curr = T_b0_b[:-1]
        T_b0__b_next = T_b0_b[1:]
        # relative motion: [N - 1, 4, 4]
        relative_T = homogeneous_inv(T_b0__b_curr) @ T_b0__b_next
        relative_poses = get_poses(relative_T)
        
        # Calculate Farthest Visible Frames Loop
        farthest_indices = []
        
        for i in range(len(self.df) - 1): # Process up to N-1
            depth = self.load_depth(self.depth_images[i])
            min_depth_map = scipy.ndimage.minimum_filter(depth, size=self.PATCH_SIZE, mode='nearest')
            f_idx, _ = self.find_farthest_visible_frame_vectorized(i, min_depth_map, T_w_c)
            farthest_indices.append(f_idx)
            
        # Vectorized relative pose computation
        if farthest_indices:
            idx_array = np.array(farthest_indices)
            
            # Current and Farthest in World Frame
            T_w_c_curr = T_w_c[:-1]
            T_w_c_farthest = T_w_c[idx_array]
            
            # Transform to Body Frame (N-1, 4, 4)
            T_w_b_curr = T_w_c_curr @ self.T_c_b
            T_w_b_farthest = T_w_c_farthest @ self.T_c_b
            
            # Relative Pose: inv(Current) @ Farthest
            T_rel_farthest = homogeneous_inv(T_w_b_curr) @ T_w_b_farthest
            
            # Convert to [x, y, z, yaw]
            farthest_poses = get_poses(T_rel_farthest) # [N-1, 4]
        else:
            farthest_poses = np.zeros((0, 4), dtype=np.float32)

        # append zero for the last frame
        zero_pose = np.zeros((1, 4), dtype=relative_poses.dtype)
        
        relative_poses_full = relative_poses # [N-1, 4]
        
        # Actions size: 8
        # [dx, dy, dz, dyaw, f_dx, f_dy, f_dz, f_dyaw]
        actions_N_minus_1 = np.hstack([relative_poses_full, farthest_poses])
        
        # Last frame action (zeros)
        action_last = np.hstack([zero_pose, zero_pose])
        
        actions = np.vstack([actions_N_minus_1, action_last])
        self.actions = actions

    def __iter__(self) -> Iterable[Tuple[dict, str]]:
        # Yield data
        for i in range(len(self.images)):
             # Construct frame dict similar to V1
             frame = {
                 "annotation.human.action.task_description": np.array([self.task_idx], dtype=np.int32),
                 "observation.state": self.state[i].astype(np.float32),
                 "video.ego_view": np.array(Image.open(self.images[i]).convert("RGB")),
                 "action": self.actions[i].astype(np.float32)
             }
             yield frame, self.task


    @property
    def metadata(self) -> dict:
        return {
            "T_b_c": self.T_b_c,
            "K": self.K,
        }

    @staticmethod
    def _compute_T_b_c_and_T_c_b(T_w_c: np.ndarray)->tuple[np.array, np.array, float]:
        T_w_c = T_w_c.reshape((4,4))

        # camera +x right, +y up, +z backwards
        yaw, pitch, ori_roll = Rotation.from_matrix(T_w_c[:3, :3]).as_euler('ZYX', degrees=True)

        roll = 90.0 # force the camera to be level with the horizon
        R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
        T_w_b = T_w_c.copy()
        T_w_b[:3, 0] = -R[:, 2]   # body +x = camera -z
        T_w_b[:3, 1] = -R[:, 0]   # body +y = camera -x
        T_w_b[:3, 2] = R[:, 1]    # body +z = camera +y
        
        T_b_w = homogeneous_inv(T_w_b)
        T_b_c = T_b_w @ T_w_c
        T_c_b = homogeneous_inv(T_b_c)
        return T_b_c, T_c_b, ori_roll


class VLN_N1_V2_Trajectories(Trajectories):
    FPS: int = 10
    ROBOT_TYPE: str = "lerobot"
    INSTRUCTION_KEY: str = "annotation.human.action.task_description"


    FEATURES = {
        # The language instruction for the task.
        "annotation.human.action.task_description": {
            "dtype": "int32", # index of task
            "shape": (1,),
            "names": None,
        },
        # The drone's pose in the first frame of the trajectory.
        "observation.state": {
            "dtype": "float32",
            "shape": (5,),
            "names": {
                "axes": ["x", "y", "z", "yaw", "collision_avoidance_yaw_rate"],
            },
        },
        # The primary video feed from the drone's ego-centric camera.
        "video.ego_view": {
            "dtype": "video",
            "shape": (270, 480, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        },
        # The action command sent to the drone.
        # first 4 values are [dx, dy, dz, dyaw] to the next frame
        # last 4 values are goal pose of the to the current frame
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": ["x", "y", "z", "yaw", "farthest_x", "farthest_y", "farthest_z", "farthest_yaw"],
            },
        },
    }

    def __init__(self, data_path: str, get_task_idx: Callable[[str], int]):
        self.data_path = Path(data_path)
        self.get_task_idx = get_task_idx
        self.parquet_files = []
        
        # Scan for 'data' directories to find scenes
        # Structure: .../scene_id/data/chunk-XXX/episode_XXXXXX.parquet
        print(f"Scanning {self.data_path} for trajectories...")
        for data_dir in tqdm(self.data_path.rglob("data"), desc="Searching data dirs"):
            if not data_dir.is_dir():
                continue
            
            # Verify scene structure
            scene_dir = data_dir.parent
            if (scene_dir / "meta").exists() and (scene_dir / "videos").exists():
                for chunk_dir in data_dir.glob("chunk-*"):
                    if chunk_dir.is_dir():
                        self.parquet_files.extend(sorted(list(chunk_dir.glob("*.parquet"))))

        print(f"Found {len(self.parquet_files)} trajectories.")
        
    def __len__(self) -> int:
        return len(self.parquet_files)

    def __iter__(self) -> Iterable[Traj]:
        current_scene_dir = None
        current_tasks_map = {}
        current_esdf = None

        for parquet_path in self.parquet_files:
            try:
                # Path resolution
                chunk_dir = parquet_path.parent
                data_dir = chunk_dir.parent
                scene_dir = data_dir.parent
                
                chunk_name = chunk_dir.name
                episode_name = parquet_path.stem # episode_XXXXXX
                
                try:
                    episode_idx = int(episode_name.split('_')[1])
                except (IndexError, ValueError):
                    continue
                
                # Resource paths
                video_dir = scene_dir / "videos" / chunk_name
                rgb_dir = video_dir / "observation.images.rgb"
                depth_dir = video_dir / "observation.images.depth"
                pcd_path = scene_dir / "meta" / "pointcloud.ply"
                
                if not (rgb_dir.exists() and depth_dir.exists()):
                    continue

                # Update resources if scene changed
                if scene_dir != current_scene_dir:
                    current_scene_dir = scene_dir
                    
                    # 1. Update ESDF
                    if current_esdf is not None:
                        del current_esdf
                        import torch 
                        torch.cuda.empty_cache()
                    
                    # Assume pcd exists
                    pcd = o3d.io.read_point_cloud(str(pcd_path))
                    current_esdf = PointCloudESDF(pcd, device="cuda:0")

                    # 2. Update Metadata
                    current_tasks_map = {}
                    episodes_file = scene_dir / "meta" / "episodes.jsonl"
                    if episodes_file.exists():
                        with open(episodes_file, "r") as f:
                            for line in f:
                                try:
                                    item = json.loads(line)
                                    if "episode_index" in item:
                                        current_tasks_map[item["episode_index"]] = item.get("tasks", [])
                                except:
                                    continue
                
                # Instruction Selection
                tasks = current_tasks_map.get(episode_idx, [])
                instruction = ""
                
                if not validate_tasks(tasks):
                    logging.warning(f"Invalid tasks for trajectory: {parquet_path} tasks: {tasks}")
                    continue
                
                tasks_str = json.dumps(tasks, ensure_ascii=False, indent=2)
                instruction = tasks_str
                task_idx = self.get_task_idx(instruction)

                # Collect Images matches episode_XXXXXX
                prefix = f"{episode_name}_"
                
                # Glob is slightly expensive, but needed if we don't know the exact count/indices beforehand
                rgb_images = sorted(list(rgb_dir.glob(f"{prefix}*.jpg")))
                depth_images = sorted(list(depth_dir.glob(f"{prefix}*.png")))
                
                if not rgb_images:
                    continue
                
                yield VLN_N1_V2_Traj(
                    parquet_path=parquet_path,
                    esdf=current_esdf,
                    images=rgb_images,
                    depth_images=depth_images,
                    task=instruction,
                    task_idx=task_idx
                )
            except Ignore as e:
                logging.warning(f"Ignore trajectory: {parquet_path} for reason: {e}")
                continue

            except Exception:
                traceback.print_exc()
                print(f"parquet_path: {parquet_path}")
                print(f"pcd_path: {pcd_path}")
                print(f"rgb_dir: {rgb_dir}")
                for img in rgb_images:
                    print(f"  img: {img}")
                print(f"depth_dir: {depth_dir}")
                for dimg in depth_images:
                    print(f"  dimg: {dimg}")
                print(f"task: {instruction}")
                continue

    @property
    def schema(self) -> dict:
        return VLN_N1_V2_Trajectories.FEATURES



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Path to the generated data
    data_root = Path("VLN-N1-test_data")
    output_dir = Path("test/vln_n1_visual")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {data_root} for parquet files...")
    parquet_files = sorted(list(data_root.rglob("episode_*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {data_root}")
    else:
        for p_file in parquet_files:
            try:
                print(f"Processing {p_file.name}...")
                df = pd.read_parquet(p_file)
                
                # Stack the state column: [N, 5] -> x, y, z, yaw, collision_yaw_rate
                states = np.vstack(df["observation.state"].to_numpy())
                
                x = states[:, 0]
                y = states[:, 1]
                yaw_deg = states[:, 3]
                coll_rate = states[:, 4]
                
                # Convert yaw to radians for plotting arrows
                yaw_rad = np.deg2rad(yaw_deg)
                
                plt.figure(figsize=(10, 8))
                
                # Scatter plot colored by collision avoidance yaw rate (Magnitude)
                # Using absolute value to visually emphasize intensity of avoidance
                # Use a diverging map or just magnitude? 
                # User asked for "depth/shallow" (深浅), usually implies magnitude.
                sc = plt.scatter(x, y, c=np.abs(coll_rate), cmap='plasma', s=20, label='Avoidance Rate (Abs)')
                cb = plt.colorbar(sc)
                cb.set_label('Collision Yaw Rate Magnitude (deg/s)')
                
                # Plot Yaw direction with arrows
                # Downsample for clarity
                step = max(1, len(x) // 30)
                plt.quiver(x[::step], y[::step], np.cos(yaw_rad[::step]), np.sin(yaw_rad[::step]), 
                          color='black', alpha=0.6, scale=20, width=0.003, label='Yaw')
                
                # Start and End
                plt.plot(x[0], y[0], 'g^', markersize=10, label='Start')
                plt.plot(x[-1], y[-1], 'r*', markersize=10, label='End')
                
                plt.title(f"Trajectory: {p_file.stem}\nColored by Collision Avoidance Yaw Rate")
                plt.xlabel("X (m)")
                plt.ylabel("Y (m)")
                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                out_path = output_dir / f"{p_file.stem}.png"
                plt.savefig(out_path)
                plt.close()
                print(f"Saved visualization to {out_path}")
                
            except Exception as e:
                print(f"Failed to plot {p_file}: {e}")
                traceback.print_exc()

