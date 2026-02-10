import cv2
import sys
import os
import json
import numpy as np
import scipy.ndimage
import open3d as o3d
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.vln_n1.trajectory import VLN_N1_Traj, Ignore as IgnoreV1
from utils.vln_n1_v2.trajectory import VLN_N1_V2_Traj, Ignore as IgnoreV2
from utils.coordinate import PointCloudESDF, relative_pose, to_homogeneous, homogeneous_inv
from scipy.spatial.transform import Rotation

VISUAL_DIR = Path(__file__).parent / "vln_n1_correctness_visual"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

def visualize_discrepancy(img_path, frame_id, episode_id, scene_name, traj, current_idx, targets, output_dir):
    """
    targets: dict { 'V1': idx, 'V2': idx, 'Other': idx }
    """
    if not img_path.exists():
        return
    
    cv_img = cv2.imread(str(img_path))
    if cv_img is None:
        return
    
    W, H = traj.image_size

    # Helper for pose calculation (as in find_farthest_points.py)
    T_w_c_base = traj._get_action(0) # Use traj (V1) for consistency
    T_w_c_base = traj.roll_to_horizontal(T_w_c_base)
    T_c_base_w = homogeneous_inv(T_w_c_base)

    def get_first_frame_pose(T_w_c):
        T_w_c = traj.roll_to_horizontal(T_w_c)
        T_c_base_c = T_c_base_w @ T_w_c
        R_rel = Rotation.from_matrix(T_c_base_c[:3, :3])
        yaw_rel, _, _ = R_rel.as_euler('YXZ', degrees=True)
        
        p_w = T_w_c[:3, 3]
        p_w = to_homogeneous(p_w)
        p_c_base = T_c_base_w @ p_w
        p_c_base = p_c_base[:3] / p_c_base[3]
        x, y, z = p_c_base
        return np.array([-z, -x, y, yaw_rel], dtype=np.float32)

    T_w_c_current = traj._get_action(current_idx)
    current_pose = get_first_frame_pose(T_w_c_current)
    T_c_current_w = homogeneous_inv(T_w_c_current)

    colors = {
        'V1': (0, 255, 0),    # Green
        'V2': (255, 0, 0),    # Blue
        'Other': (0, 0, 255)  # Red
    }
    offsets = {
        'V1': 30,
        'V2': 50,
        'Other': 70
    }
    radii = {
        'V1': 9,
        'V2': 6,
        'Other': 3
    }

    # Draw info
    cv2.putText(cv_img, f"Frame: {frame_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    for label, target_idx in targets.items():
        if target_idx is None: 
            continue
            
        T_w_c_target = traj._get_action(target_idx)
        target_pose = get_first_frame_pose(T_w_c_target)
        
        rel_pose_4d = traj.to_4d(
            relative_pose(traj.to_6d(current_pose), traj.to_6d(target_pose), degree=True)
        )
        rx, ry, rz, ryaw = rel_pose_4d
        
        # Project point
        T_c_current_c_target = T_c_current_w @ T_w_c_target
        p_c = T_c_current_c_target[:3, 3]
        proj = traj.project_camera_point(p_c, traj.K, (H, W))
        
        text = f"{label} ID:{target_idx} ({rx:.2f}, {ry:.2f}, {rz:.2f}, {ryaw:.1f})"
        cv2.putText(cv_img, text, (10, offsets[label]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 1)

        if proj is not None:
            u, v, _ = proj
            cv2.circle(cv_img, (int(u), int(v)), radii[label], colors[label], 2)
            cv2.circle(cv_img, (int(u), int(v)), 1, colors[label], -1) # center dot

    filename = f"{scene_name}_{episode_id}_{frame_id}.jpg"
    cv2.imwrite(str(output_dir / filename), cv_img)



def setup_resources(parquet_path):
    parquet_path = Path(parquet_path).resolve()
    chunk_dir = parquet_path.parent  # chunk-XXX
    data_dir = chunk_dir.parent      # data
    scene_dir = data_dir.parent      # e.g., ffd98024-7200-429e-8b9a-1234a5937826
    
    chunk_name = chunk_dir.name
    episode_name = parquet_path.stem
    episode_idx = int(episode_name.split('_')[1])
    
    # Resource paths
    video_chunk_dir = scene_dir / "videos" / chunk_name
    rgb_dir = video_chunk_dir / "observation.images.rgb"
    depth_dir = video_chunk_dir / "observation.images.depth"
    pcd_path = scene_dir / "meta" / "pointcloud.ply"
    episodes_file = scene_dir / "meta" / "episodes.jsonl"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    if not depth_dir.exists():
         # Fallback for some formats where images are in videos/chunk-000 directly?
         # But assuming standard struct for now
         raise FileNotFoundError(f"Depth dir not found: {depth_dir}")

    # Load ESDF
    print(f"Loading PointCloud from {pcd_path}...")
    if pcd_path.exists():
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        esdf = PointCloudESDF(pcd, voxel_size=0.1)
    else:
        print("Warning: Pointcloud not found. Using dummy ESDF.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
        esdf = PointCloudESDF(pcd, voxel_size=0.1)

    # Load Task Info
    tasks = []
    episode_info = {}
    if episodes_file.exists():
        with open(episodes_file, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "episode_index" in item and item["episode_index"] == episode_idx:
                        tasks = item.get("tasks", [])
                        episode_info = item
                        break
                except:
                    continue
    
    # Collect Images
    prefix = f"{episode_name}_"
    # Find images starting with prefix
    rgb_images = sorted(list(rgb_dir.glob(f"{prefix}*.jpg")) + list(rgb_dir.glob(f"{prefix}*.png")))
    depth_images = sorted(list(depth_dir.glob(f"{prefix}*.png"))) # usually depth is png
    
    # Sort by index
    try:
        rgb_images.sort(key=lambda p: int(p.stem.split('_')[-1]))
        depth_images.sort(key=lambda p: int(p.stem.split('_')[-1]))
    except:
        print("Warning: could not sort images by index suffix.")

    # Validate lengths against Parquet
    df = pd.read_parquet(parquet_path)
    N = len(df)
    
    if len(rgb_images) != N:
         print(f"Warning: RGB image count ({len(rgb_images)}) != DF rows ({N}). Trimming/Checking.")
         if len(rgb_images) > N:
             rgb_images = rgb_images[:N]
    
    if len(depth_images) != N:
         print(f"Warning: Depth image count ({len(depth_images)}) != DF rows ({N}). Trimming/Checking.")
         if len(depth_images) > N:
             depth_images = depth_images[:N]
             
    # Ensure they match now
    rgb_images = rgb_images[:N]
    depth_images = depth_images[:N]

    return {
        "parquet_path": parquet_path,
        "esdf": esdf,
        "images": rgb_images,
        "depth_images": depth_images,
        "episode_info": episode_info,
        "tasks": tasks,
        "trajectory_dir": scene_dir # for V1
    }

################################################################################
# Other Implementation (from user snippet)
################################################################################
from scipy.spatial.transform import Rotation

EDGE_PIX = 20     # 与边界距离阈值
PATCH_R = 5       # Patch 半径，用于遮挡判断
FRAME_STEP = 1   # 当前帧步长：下一条样本从 idx_i + 1 开始
MIN_FRAME_GAP = 1 # 最小帧间隔：当前帧与groundtruth帧的最小距离
CAMERA_ROLL_THRES = 5.0 # 相机 roll 过滤阈值

def load_depth_other(depth_path: str) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32) * 0.0001

def world_to_camera_other(T_w_c: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    T_c_w = np.linalg.inv(T_w_c)
    Xc = T_c_w @ np.array([Xw[0], Xw[1], Xw[2], 1.0])
    return Xc[:3]

def project_camera_point_other(Xc: np.ndarray, K: np.ndarray, image_shape: tuple):
    H, W = image_shape
    X, Y, Z = Xc
    depth = -Z
    if depth <= 0:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = X * fx / depth + cx
    v = H - 1 - (Y * fy / depth + cy)
    return u, v, depth

def is_collision_within_patch_other(depth, u, v, depth_proj, H, W):
    ui, vi = round(u), round(v)  # int(u), int(v)
    u0 = max(0, ui - PATCH_R)
    u1 = min(W - 1, ui + PATCH_R)
    v0 = max(0, vi - PATCH_R)
    v1 = min(H - 1, vi + PATCH_R)
    patch = depth[v0:v1+1, u0:u1+1]
    return patch.min() <= depth_proj

def find_farthest_visible_frame_other(frame_i, actions, traj_world, depth, K, image_shape, search_end_idx=None):
    """
    输入：当前帧 index = frame_i
    输出：最远可见帧 index（考虑遮挡 Patch 和图像范围）
    """
    H, W = image_shape
    T_w_ci = actions[frame_i]

    last_valid_idx = frame_i # Default to current frame if no forward frame is visible

    end_idx = len(actions) if search_end_idx is None else search_end_idx + 1

    for idx in range(frame_i + 1, end_idx):
        # 3D -> 2D 投影
        Xw = traj_world[idx]
        Xc = world_to_camera_other(T_w_ci, Xw)
        proj = project_camera_point_other(Xc, K, image_shape)
        if proj is None:
            # Point behind camera or invalid depth
            break

        u, v, depth_proj = proj

        # 投影到当前帧图像外部则终止，上一帧为关键帧
        if not (0 <= u < W and 0 <= v < H):
            break

        # 如果patch内发生遮挡终止，上一帧为关键帧
        if is_collision_within_patch_other(depth, u, v, depth_proj, H, W):
            break

        last_valid_idx = idx

    return last_valid_idx

def is_near_edge(u, v, H, W, edge_pix=EDGE_PIX):
    return (u < edge_pix) or (u > W - edge_pix) or (v < edge_pix) or (v > H - edge_pix)

def adjust_frame_to_avoid_edge_other(frame_i, idx_end, actions, traj_world, K, image_shape):
    """
    根据最远可见点 idx_end，从其开始往前迭代，直到找到一个不靠边的投影。
    如果全部都靠边，则退回到 frame_i+1。
    """
    H, W = image_shape
    idx_new = idx_end
    while idx_new > frame_i:
        Xw = traj_world[idx_new]
        Xc = world_to_camera_other(actions[frame_i], Xw)
        proj = project_camera_point_other(Xc, K, image_shape)
        if proj is None:
            break
        u, v, _ = proj
        if not is_near_edge(u, v, H, W):
            return idx_new
        idx_new -= 1
    return frame_i + 1
    # return frame_i # NOTE: 和docstring不同，试一下直接退回到frame_i
################################################################################

import argparse
import random

def test_consistency(parquet_path, verbose=False, roll_limit=None) -> dict:
    try:
        resources = setup_resources(parquet_path)
    except Exception as e:
        return {"path": str(parquet_path), "status": "ERROR", "message": f"Setup failed: {e}"}
    
    # Determine image size from first image
    img0 = cv2.imread(str(resources["images"][0]))
    h, w = img0.shape[:2]

    # --- Initialize V1 ---
    if verbose: print("Initializing V1...")
    frames_v1 = {
        "trajectory_dir": resources["trajectory_dir"],
        "parquet_path": resources["parquet_path"],
        "images": resources["images"],
        "depth_images": resources["depth_images"],
        "episode_info": resources["episode_info"]
    }
    
    def get_task_idx(t): return 0
    
    # Filter for V1
    filter_cond = {}
    if roll_limit is not None:
        filter_cond["roll_limit"] = roll_limit

    try:
        traj_v1 = VLN_N1_Traj(frames_v1, get_task_idx, image_size=(w, h), filter_condition=filter_cond)
    except IgnoreV1 as e:
        return {"path": str(parquet_path), "status": "SKIPPED", "message": f"V1 Ignored: {e}"}
    except Exception as e:
        return {"path": str(parquet_path), "status": "ERROR", "message": f"V1 Init failed: {e}"}
    
    # --- Initialize V2 ---
    if verbose: print("Initializing V2...")
    
    # Filter for V2
    if roll_limit is not None:
        def roll_filter(traj: VLN_N1_V2_Traj):
            if abs(90.0 - traj.ori_roll) > roll_limit:
                raise IgnoreV2(f"Unexpected roll angle: {traj.ori_roll}°. Expected {90.0 - roll_limit}~{90.0 + roll_limit}°.")
        VLN_N1_V2_Traj.set_filter(roll_filter)
    else:
        VLN_N1_V2_Traj.set_filter(None)

    task_str = json.dumps(resources["tasks"])
    try:
        traj_v2 = VLN_N1_V2_Traj(
            resources["parquet_path"], 
            resources["esdf"], 
            resources["images"], 
            resources["depth_images"], 
            task_str, 
            0
        )
    except IgnoreV2 as e:
        return {"path": str(parquet_path), "status": "SKIPPED", "message": f"V2 Ignored: {e}"}
    except Exception as e:
        return {"path": str(parquet_path), "status": "ERROR", "message": f"V2 Init failed: {e}"}

    # --- Compare ---
    if verbose: print(f"Comparing {len(traj_v1)} frames...")
    
    T_w_c_all = np.vstack(traj_v2.df["action"].to_numpy()).reshape(-1, 4, 4)
    traj_world_other = T_w_c_all[:, :3, 3]
    K_other = traj_v2.K
    
    N = len(traj_v1)
    discrepancies = []
    
    for i in range(N - 1):
        # V1
        v1_idx, _ = traj_v1.find_farthest_visible_frame(i)
        
        # V2
        depth = traj_v2.load_depth(traj_v2.depth_images[i])
        min_depth_map = scipy.ndimage.minimum_filter(depth, size=traj_v2.PATCH_SIZE, mode='nearest')
        
        # Calculate search_end for V2
        search_end = N - 1
        # V2 internal logic for sub_indexes
        for start, end in traj_v2.sub_indexes:
            if start <= i <= end:
                search_end = end
                break
        
        v2_idx, _ = traj_v2.find_farthest_visible_frame_vectorized(i, min_depth_map, T_w_c_all, search_end)
        v2_idx = int(v2_idx)
        # Other
        # Note: Other impl uses simple load_depth logic, we can reuse depth loaded by V2
        # Use full search range (or up to search_end to match V1/V2 semantics)
        idx_end_other = find_farthest_visible_frame_other(i, T_w_c_all, traj_world_other, depth, K_other, (h, w), search_end_idx=search_end)
        other_idx = adjust_frame_to_avoid_edge_other(i, idx_end_other, T_w_c_all, traj_world_other, K_other, (h, w))
        
        mismatch_v1_v2 = abs(v1_idx - v2_idx) > 1
        mismatch_v1_other = abs(v1_idx - other_idx) > 1
        
        if mismatch_v1_v2 or mismatch_v1_other:
            discrepancies.append((i, v1_idx, v2_idx, other_idx))
            
            # Visualization
            scene_name = resources["trajectory_dir"].name
            episode_id = resources["parquet_path"].stem.split('_')[1]
            img_path = resources["images"][i]
            
            visualize_discrepancy(
                img_path, i, episode_id, scene_name,
                traj_v1, i, 
                {'V1': v1_idx, 'V2': v2_idx, 'Other': other_idx},
                VISUAL_DIR
            )

    if not discrepancies:
        return {"path": str(parquet_path), "status": "SUCCESS", "message": "Matched perfectly (V1==V2==Other)"}
    else:
        msg = f"Found {len(discrepancies)} discrepancies. "
        msg += f"First 5: " + ", ".join([f"Frame {d[0]}: V1={d[1]}, V2={d[2]}, Other={d[3]}" for d in discrepancies[:5]])
        return {"path": str(parquet_path), "status": "FAILURE", "message": msg}

def main():
    parser = argparse.ArgumentParser(description="Test VLN V1 vs V2 Consistency")
    parser.add_argument("--root", type=str, default="test_data", help="Root directory to search for parquets")
    parser.add_argument("--num", type=int, default=1, help="Number of random episodes to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--roll_limit", type=float, default=None, help="Roll limit for filtering trajectories")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    random.seed(args.seed)
    root_path = Path(args.root)
    
    print(f"Searching for parquet files in {root_path}...")
    parquets = list(root_path.rglob("*.parquet"))
    print(f"Found {len(parquets)} parquet files.")
    
    if not parquets:
        return
        
    if len(parquets) > args.num:
        selected_parquets = random.sample(parquets, args.num)
    else:
        selected_parquets = parquets
        
    results = []
    for p in tqdm(selected_parquets, desc="Testing Episodes"):
        if args.verbose:
            print(f"\nTesting: {p}")
        res = test_consistency(p, verbose=args.verbose, roll_limit=args.roll_limit)
        results.append(res)
        if args.verbose:
            print(f"Result: {res['status']} - {res['message']}")

    # Report
    print("\n" + "="*50)
    print("TEST REPORT")
    print("="*50)
    
    success_count = 0
    failure_count = 0
    error_count = 0
    skipped_count = 0
    
    for r in results:
        status = r["status"]
        if status == "SUCCESS":
            success_count += 1
            print(f"[PASS] {r['path']}")
        elif status == "SKIPPED":
            skipped_count += 1
            print(f"[SKIP] {r['path']}")
            print(f"       Reason: {r['message']}")
        else:
            if status == "FAILURE":
                failure_count += 1
                level = "[FAIL]" 
            else:
                error_count += 1
                level = "[ERR ]"
            
            print(f"{level} {r['path']}")
            print(f"       Reason: {r['message']}")
            
    print("-" * 50)
    print(f"Total: {len(results)} | Success: {success_count} | Fail: {failure_count} | Error: {error_count} | Skip: {skipped_count}")
    
    if failure_count > 0 or error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
