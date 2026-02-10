
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.coordinate import homogeneous_inv, get_poses, PointCloudESDF
from utils.obstacle import compute_collision_prob, compute_yaw_rate

# ==========================================
#              CONFIGURATION
# ==========================================
# 目录路径 (DIR) - Supports str or List[str]
# SEARCH_DIR =  "/data-10T/InternData-N1/3dfront_d435i" # "/data-10T/InternData-N1/hm3d_d435i"
# SEARCH_DIR = "/data-10T/InternData-N1/hm3d_zed/00024-XNoaAZwsWKk__3ba4ce86c1/00024-XNoaAZwsWKk"
# SEARCH_DIR = "/data-10T/InternData-N1/hm3d_zed/00003-NtVbfPCkBFy__93366f7175/00003-NtVbfPCkBFy"
# SEARCH_DIR = "/data-10T/InternData-N1/3dfront_d435i/0e563bef-ab48-4a9a-94c1-3018575fa499__999a5d0e93"
SEARCH_DIR = [
    "/data-10T/InternData-N1/3dfront_d435i",
    "/data-10T/InternData-N1/3dfront_zed",
    "/data-10T/InternData-N1/hm3d_d435i",
    "/data-10T/InternData-N1/hm3d_zed",
    "/data-10T/InternData-N1/replica_d435i",
    "/data-10T/InternData-N1/replica_zed",
    "/data-10T/InternData-N1/hssd_d435i",
    "/data-10T/InternData-N1/hssd_zed",
    "/data-10T/InternData-N1/matterport3d_d435i",
    "/data-10T/InternData-N1/matterport3d_zed",
    "/data-10T/InternData-N1/gibson_d435i",
    "/data-10T/InternData-N1/gibson_zed",
]

# 采样轨迹数量 (N)
N_SAMPLES = 3000

# 平滑窗口大小 (SMOOTH_WINDOW)
SMOOTH_WINDOW = 10

# 碰撞概率阈值 (COLLISION_THRESHOLD)
COLLISION_THRESHOLD = 0.8

# 统计模式：百分位 (YAW_RATE_PERCENTILE), e.g. 0.25 represents top 25%
YAW_RATE_PERCENTILE = 0.25 

# 可视化模式：角速度阈值 (VIS_YAW_RATE_THRESHOLD)
VIS_YAW_RATE_THRESHOLD = 18.96 # deg/s

# 运行模式 (MODE): "STAT" or "VIS"
MODE = "STAT"
# MODE = "VIS"

# 输出目录
OUTPUT_DIR = Path("test/obstacle_stat_output")


# ==========================================
#           HELPER FUNCTIONS
# ==========================================

def compute_T_b_c_and_T_c_b(first_pose_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Body-to-Camera and Camera-to-Body transforms assuming specific camera setup.
    Copied from vln_n1_obstacle.py
    """
    T_w_c = first_pose_matrix.reshape((4,4))

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

def find_pointcloud_path(parquet_path: Path) -> Path:
    """
    Locate the pointcloud.ply file relative to the parquet file.
    """
    parts = parquet_path.parts
    if 'data' in parts:
        data_idx = parts.index('data')
        # scene root is one level above 'data'
        scene_root = Path(*parts[:data_idx])
        pdc_file = scene_root / "meta" / "pointcloud.ply"
    else:
        # Fallback
        pdc_file = parquet_path.parent.parent.parent / "meta" / "pointcloud.ply"
    return pdc_file

def plot_trajectory_2d_highlighted(poses, obstacles, highlight_mask, save_path, title=None):
    x = poses[:, 0]
    y = poses[:, 1]
    
    plt.figure(figsize=(10, 8))
    
    # Obstacles
    if obstacles is not None:
        plt.scatter(obstacles[:, 0], obstacles[:, 1], c='gray', s=1, alpha=0.1, label='Obstacles')
    
    # Full trajectory
    plt.plot(x, y, color='blue', alpha=0.4, linewidth=1, zorder=1, label='Trajectory')
    
    # Highlighted segments
    if np.any(highlight_mask):
        plt.scatter(x[highlight_mask], y[highlight_mask], c='red', s=15, alpha=1.0, zorder=3, label='High Risk + High Turn')
    
    # Start/End
    plt.scatter(x[0], y[0], c='green', s=80, marker='^', zorder=4, label='Start')
    plt.scatter(x[-1], y[-1], c='purple', s=80, marker='*', zorder=4, label='End')
    
    plt.title(title or "Trajectory Visualization")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_analysis():
    # 0. Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    search_dirs = [SEARCH_DIR] if isinstance(SEARCH_DIR, str) else SEARCH_DIR
    all_files = []
    
    print(f"[{MODE}] Searching for parquet files...")
    for s_dir in search_dirs:
        s_path = Path(s_dir)
        if not s_path.exists():
            print(f"  [WARN] Path does not exist: {s_path}")
            continue
        found = list(s_path.rglob("*.parquet"))
        print(f"  Found {len(found)} files in {s_path}")
        all_files.extend(found)
        
    print(f"Total Found {len(all_files)} files.")
    
    if len(all_files) == 0:
        print("No files found. Exiting.")
        return

    # Random Sampling
    selected_files = random.sample(all_files, min(len(all_files), N_SAMPLES))
    print(f"Selected {len(selected_files)} files for processing.")

    # Caches
    pcd_cache = {}
    
    # Data containers for STAT mode
    high_risk_yaw_rates = []  # List to store yaw rates of high risk points
    
    fps = 10
    dt = 1.0 / fps

    for idx, parquet_path in enumerate(tqdm(selected_files, desc="Processing Trajectories")):
        try:
            # 1. Load Data
            df = pd.read_parquet(parquet_path)
            pdc_file = find_pointcloud_path(parquet_path)
            
            if not pdc_file.exists():
                continue
                
            scene_key = str(pdc_file.parent.parent) # Just used for caching key
            
            # 2. Load ESDF and Visual PCD
            if scene_key not in pcd_cache:
                o3d_pcd = o3d.io.read_point_cloud(str(pdc_file))
                esdf = PointCloudESDF(o3d_pcd, device="cuda:0", voxel_size=None)
                pcd_cache[scene_key] = {
                    "esdf": esdf,
                    "pcd": o3d_pcd,
                    "scene_name": pdc_file.parent.parent.name
                }
            
            scene_data = pcd_cache[scene_key]
            pcd = scene_data["esdf"]
            vis_pcd = scene_data["pcd"]
            
            # 3. Compute Body Poses
            # [N, 4, 4]
            T_w_c = np.vstack(df["action"].to_numpy()).reshape(-1, 4, 4)
            if len(T_w_c) < 5: continue
            
            T_b_c, T_c_b, _ = compute_T_b_c_and_T_c_b(T_w_c[0])
            T_w_b = T_w_c @ T_c_b
            poses_body = get_poses(T_w_b) # [N, 4] (x,y,z,yaw_deg)
            
            # 4. Valid check
            # Avoid overly long trajectories if necessary, or just process all
            
            # 5. Compute Metrics
            # collision prob: 0~1
            probs = compute_collision_prob(poses_body, pcd, dt=dt)
            # yaw rates: deg/s, smoothed
            yaw_rates = compute_yaw_rate(poses_body[:, 3], dt, smoothing_window=SMOOTH_WINDOW)
            
            if MODE == "STAT":
                # Filter points with High Collision Probability
                mask = probs > COLLISION_THRESHOLD
                if np.any(mask):
                    selected_rates = yaw_rates[mask]
                    high_risk_yaw_rates.extend(selected_rates.tolist())
                    
            elif MODE == "VIS":
                # Condition: High Prob AND High Yaw Rate
                mask = (probs > COLLISION_THRESHOLD) & (yaw_rates > VIS_YAW_RATE_THRESHOLD)
                
                if np.any(mask):
                    # Only visualize if we found interesting points
                    points, _ = pcd.query(poses_body[:, :3], 100) # Get local obstacles for context
                    obstacles = points.reshape(-1, 3)
                    
                    save_name = f"{parquet_path.stem}_vis.png"
                    title = f"Risk>{COLLISION_THRESHOLD} & YawRate>{VIS_YAW_RATE_THRESHOLD}deg/s"
                    plot_trajectory_2d_highlighted(
                        poses_body, 
                        obstacles, 
                        mask, 
                        OUTPUT_DIR / save_name,
                        title=title
                    )

                    # --- 3D Point Cloud Visualization ---
                    # Colors: Blue (default) -> Red (Highlight)
                    colors = np.zeros((len(poses_body), 3))
                    colors[:] = [0, 0, 1]  # Blue
                    colors[mask] = [1, 0, 0] # Red
                    
                    traj_pcd = o3d.geometry.PointCloud()
                    traj_pcd.points = o3d.utility.Vector3dVector(poses_body[:, :3])
                    traj_pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Start Point (Green Sphere)
                    start_pos = poses_body[0, :3]
                    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
                    start_sphere.translate(start_pos)
                    start_sphere.paint_uniform_color([0, 1, 0]) # Green
                    start_pcd = start_sphere.sample_points_uniformly(number_of_points=200)

                    # End Point (Purple Sphere to match 2D)
                    end_pos = poses_body[-1, :3]
                    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
                    end_sphere.translate(end_pos)
                    end_sphere.paint_uniform_color([0.5, 0, 0.5]) # Purple
                    end_pcd = end_sphere.sample_points_uniformly(number_of_points=200)

                    # Accumulate
                    vis_pcd += traj_pcd
                    vis_pcd += start_pcd
                    vis_pcd += end_pcd

        except Exception as e:
            # print(f"Error processing {parquet_path}: {e}")
            pass

    # ==========================================
    #           REPORTING (STAT MODE)
    # ==========================================
    if MODE == "VIS":
        print("\n" + "="*40)
        print("Saving Visualized Point Clouds...")
        for scene_key, data in pcd_cache.items():
            # Only save if we actually modified it (though checking modification is hard, 
            # we assume if it's in cache and we ran VIS mode, we might want it. 
            # But really we only care if we added trajectories. 
            # Since we modify 'pcd' object in place, we can just save all cached scenes.)
            scene_name = data["scene_name"]
            vis_pcd = data["pcd"]
            save_path = OUTPUT_DIR / f"{scene_name}_vis.ply"
            print(f"  Saving {save_path} ...")
            o3d.io.write_point_cloud(str(save_path), vis_pcd)
            
    if MODE == "STAT":
        print("\n" + "="*40)
        print("          STATISTICS REPORT")
        print("="*40)
        print(f"Total Trajectories Processed: {len(selected_files)}")
        print(f"Smooth Window: {SMOOTH_WINDOW}")
        print(f"Collision Threshold: {COLLISION_THRESHOLD}")
        
        total_high_risk_points = len(high_risk_yaw_rates)
        print(f"\nTotal points with Risk > {COLLISION_THRESHOLD}: {total_high_risk_points}")
        
        if total_high_risk_points > 0:
            arr = np.array(high_risk_yaw_rates)
            
            # 统计前 YAW_RATE_PERCENTILE 的角速度
            # 如果 YAW_RATE_PERCENTILE = 0.25 (Top 25%), 即寻找 75th percentile
            q_target = 1.0 - YAW_RATE_PERCENTILE
            threshold_val = np.quantile(arr, q_target)
            
            mean_val = np.mean(arr)
            max_val = np.max(arr)
            
            print(f"Yaw Rate Statistics for High Risk Points (deg/s):")
            print(f"  Mean:       {mean_val:.2f}")
            print(f"  Max:        {max_val:.2f}")
            print(f"  Median:     {np.median(arr):.2f}")
            print(f"  Top {YAW_RATE_PERCENTILE*100}% Threshold: {threshold_val:.2f}")
            
            # 也可以计算前 25% 的平均值
            top_block = arr[arr >= threshold_val]
            print(f"  Mean of Top {YAW_RATE_PERCENTILE*100}%: {np.mean(top_block):.2f}")

            # Plot Probability Density Distribution
            try:
                plt.figure(figsize=(10, 6))
                
                # Histogram (Density)
                n, bins, patches = plt.hist(arr, bins=50, density=True, alpha=0.6, color='b', label='Histogram')
                
                # KDE (Kernel Density Estimation)
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(arr)
                x_grid = np.linspace(arr.min(), arr.max(), 500)
                plt.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='KDE')
                
                # Vertical lines for stats
                plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
                plt.axvline(threshold_val, color='g', linestyle='dashed', linewidth=1, label=f'Top {YAW_RATE_PERCENTILE*100}%: {threshold_val:.2f}')
                
                plt.title(f"Yaw Rate Distribution for Risk > {COLLISION_THRESHOLD}\n(N={total_high_risk_points})")
                plt.xlabel("Yaw Rate (deg/s)")
                plt.ylabel("Probability Density")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                dist_path = OUTPUT_DIR / "high_risk_yaw_rates_dist.png"
                plt.savefig(dist_path)
                plt.close()
                print(f"Saved yaw rate distribution plot to: {dist_path}")
                
            except Exception as e:
                print(f"Failed to plot distribution: {e}")
            
        else:
            print("No high risk points found.")
        print("="*40)

if __name__ == "__main__":
    run_analysis()
