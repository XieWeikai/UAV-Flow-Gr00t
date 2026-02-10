import logging
import numpy as np
import json
from tqdm import tqdm
from typing import Callable, Iterable, Optional, Tuple, List, Dict, Any
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

def _is_valid_index_pair(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return False
    a, b = x[0], x[1]
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return a <= b

def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def validate_tasks(tasks: List[Dict[str, Any]]) -> bool:
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
        # self.esdf = esdf # Do not store ESDF to avoid IPC issues

        # images and other data
        self.images = images
        self.depth_images = depth_images # Added depth images
        
        # get H and W from the first image
        with Image.open(self.images[0]) as img:
            self.W, self.H = img.size  # (W, H)
        self.task = task
        self.task_idx = task_idx
        assert len(self.df) == len(self.images), "Number of images and frames must match."

        # Parse tasks and extract sub_indexes
        try:
            tasks_list = json.loads(self.task)
            self.sub_indexes = []
            for t in tasks_list:
                if "sub_indexes" in t:
                    self.sub_indexes.append(t["sub_indexes"])
                if "sum_indexes" in t:
                    self.sub_indexes.append(t["sum_indexes"])
            # sort by end index
            self.sub_indexes.sort(key=lambda x: x[1])
        except Exception as e:
            # Fallback if parsing fails or structure is different
            print(f"Warning: Failed to parse tasks or extract sub_indexes in V2 trajectory: {e}")
            self.sub_indexes = []

        if VLN_N1_V2_Traj._filter is not None:
            VLN_N1_V2_Traj._filter(self)
        
        # preprocessing collision (requires ESDF)
        self._precompute_collision(esdf)


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
    
    def find_farthest_visible_frame_vectorized(self, current_idx: int, min_depth_map: np.ndarray, T_w_c_all: np.ndarray, search_end: int) -> tuple[int, np.ndarray]:
        N = len(T_w_c_all)
        # boundary check: if current_idx is already at or beyond search_end, return current_idx as the farthest visible frame (no movement)
        if current_idx >= N - 1 or search_end <= current_idx:
            return current_idx, T_w_c_all[current_idx]

        # 1. construct search range [current+1, search_end]
        search_range = np.arange(current_idx + 1, search_end + 1)
        T_w_targets = T_w_c_all[search_range] 
        
        # 2. projection calculation (keep formula exactly the same as V1)
        T_w_curr = T_w_c_all[current_idx]
        T_curr_w = homogeneous_inv(T_w_curr)
        T_curr_targets = np.matmul(T_curr_w, T_w_targets)
        
        P_c = T_curr_targets[:, :3, 3]
        X, Y, Z = P_c[:, 0], P_c[:, 1], P_c[:, 2]
        D = -Z 
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        H, W = self.H, self.W
        
        u = (X * fx / D) + cx
        v = (H - 1) - ((Y * fy / D) + cy)

        # 3. find last visible index from start to end (stop at the first index that is either out of bounds or occluded)
        #  A: out of bounds (must use float to exactly align with project_camera_point internal logic)
        is_out = (u < 0.0) | (u >= float(W)) | (v < 0.0) | (v >= float(H)) | (D <= 0.0)
        
        #  B: occlusion (only calculate for points not out of bounds)
        is_occ = np.zeros_like(is_out, dtype=bool)
        can_check_occ = ~is_out
        if np.any(can_check_occ):
            u_round = np.round(u[can_check_occ]).astype(int)
            v_round = np.round(v[can_check_occ]).astype(int)
            # in case that rounding pushes (479.9 -> 480) us out of bounds, we clip to valid range (this mimics the behavior of project_camera_point which would also fail if out of bounds)
            u_round = np.clip(u_round, 0, W - 1)
            v_round = np.clip(v_round, 0, H - 1)
            is_occ[can_check_occ] = (min_depth_map[v_round, u_round] <= D[can_check_occ])

        
        stop_mask = is_out | is_occ
        break_points = np.where(stop_mask)[0]
        
        if len(break_points) > 0:
            # If break occurs at index K, then the last successful frame is K-1
            last_success_rel_idx = break_points[0] - 1
            # Align with V1's initial value: visible_idx = current_idx + 1
            # If even the first frame fails, last_success_rel_idx = -1, max then points to search_range[0]
            visible_idx = search_range[max(0, last_success_rel_idx)]
        else:
            # All frames in the search range are visible
            visible_idx = search_end

        # 4. Simulate V1's second Backward Loop (handling edge fallback)
        # Goal: In the range [current+1, visible_idx], find the highest non-edge frame
        search_back_range = np.arange(current_idx + 1, visible_idx + 1)
        rel_indices = search_back_range - (current_idx + 1)
        
        u_back = u[rel_indices]
        v_back = v[rel_indices]
        
        # Condition C: Edge check (using float u, v)
        is_near_edge = (u_back < float(self.EDGE)) | (u_back > float(W - self.EDGE)) | \
                       (v_back < float(self.EDGE)) | (v_back > float(H - self.EDGE))
        
        not_near_edge = ~is_near_edge
        safe_indices = np.where(not_near_edge)[0]
        
        if len(safe_indices) > 0:
            # Take the last (highest) non-edge frame in the visible path
            farthest_idx = search_back_range[safe_indices[-1]]
        else:
            # If all are at the edge, V1 loop will always return current_idx + 1
            farthest_idx = current_idx + 1

        return int(farthest_idx), T_w_c_all[farthest_idx]
    
    def _precompute_collision(self, esdf: PointCloudESDF) -> None:
        # camera->world [N, 4, 4]
        T_w_c = np.vstack(self.df["action"].to_numpy()).reshape(-1, 4, 4)
        T_w_b = T_w_c @ self.T_c_b
        poses_world = get_poses(T_w_b)
        
        # 1. Collision Probability
        probs = compute_collision_prob(poses_world, esdf, dt=self.DT)
        
        # 2. Yaw Rate (deg/s)
        # Using body frame yaw or world frame yaw? compute_yaw_rate expects yaw series.
        # poses_world[:, 3] is yaw in degrees.
        yaw_rates = compute_yaw_rate(poses_world[:, 3], dt=self.DT, smoothing_window=self.SMOOTH_WINDOW)
        
        # 3. Filter
        # If prob > threshold, use yaw_rate, else 0.0
        self.collision_yaw_rate = np.zeros_like(yaw_rates)
        mask = probs > self.COLLISION_THRESHOLD
        self.collision_yaw_rate[mask] = yaw_rates[mask]

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
        
        # Append to state [x, y, z, yaw, collision_yaw_rate]
        # state is [N, 4]
        self.state = np.hstack([state, self.collision_yaw_rate[:, None]]) # [N, 5]

        # [N - 1, 4, 4]
        T_b0__b_curr = T_b0_b[:-1]
        T_b0__b_next = T_b0_b[1:]
        # relative motion: [N - 1, 4, 4]
        relative_T = homogeneous_inv(T_b0__b_curr) @ T_b0__b_next
        relative_poses = get_poses(relative_T)
        
        # Calculate Farthest Visible Frames Loop
        farthest_indices = []
        N = len(self.df)
        sub_idx_ptr = 0
        
        for i in range(len(self.df) - 1): # Process up to N-1
            depth = self.load_depth(self.depth_images[i])
            min_depth_map = scipy.ndimage.minimum_filter(depth, size=self.PATCH_SIZE, mode='nearest')
            
            # Determine search range based on sub_indexes
            # Optimized: Skip sub_indexes that have ended
            while sub_idx_ptr < len(self.sub_indexes):
                 _, end = self.sub_indexes[sub_idx_ptr]
                 if end < i:
                     sub_idx_ptr += 1
                 else:
                     break

            search_end = N - 1
            for k in range(sub_idx_ptr, len(self.sub_indexes)):
                start, end = self.sub_indexes[k]
                if start <= i <= end:
                    search_end = end
                    break
            
            f_idx, _ = self.find_farthest_visible_frame_vectorized(i, min_depth_map, T_w_c, search_end)
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
        # Ensure processing is done (lazy loading for workers)
        self.process_traj()
        
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
                        # import torch 
                        # torch.cuda.empty_cache()
                    
                    # Assume pcd exists
                    pcd = o3d.io.read_point_cloud(str(pcd_path))
                    current_esdf = PointCloudESDF(pcd,
                                                #    device="cuda:0"
                                                )

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

