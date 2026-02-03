from pathlib import Path
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation
from utils.coordinate import homogeneous_inv, get_poses, PointCloudESDF
from utils.obstacle import compute_avoidance_scores

import matplotlib.pyplot as plt
import open3d as o3d

def plot_trajectory_with_obstacles(poses, points, scores=None, save_path="vln_n1_obstacle_trajectory.png"):
    x = poses[:, 0]
    y = poses[:, 1]
    yaw = poses[:, 3]

    ox = points[:, 0]
    oy = points[:, 1]

    plt.figure(figsize=(10, 8))
    
    # Plot obstacles
    plt.scatter(ox, oy, c='gray', s=1, alpha=0.1, label='Obstacles')

    # Plot trajectory
    if scores is not None:
        # Plot with scores
        sc = plt.scatter(x, y, c=scores, cmap='jet', label='Trajectory (Score)', s=5, alpha=0.8, zorder=2, vmin=0, vmax=1)
        plt.colorbar(sc, label='Avoidance Score')
    else:
        plt.plot(x, y, color='blue', alpha=0.5, linewidth=1, zorder=1)
        plt.scatter(x, y, c='blue', label='Trajectory', s=2, alpha=0.8, zorder=2)

    # Start/End
    plt.scatter(x[0], y[0], color='green', label='Start', s=100, zorder=5, edgecolors='black')
    plt.scatter(x[-1], y[-1], color='red', label='End', s=100, zorder=5, edgecolors='black')

    # Quiver
    yaw_rad = np.deg2rad(yaw)
    u = np.cos(yaw_rad)
    v = np.sin(yaw_rad)
    step = max(1, len(x) // 20)
    plt.quiver(x[::step], y[::step], u[::step], v[::step], angles='xy', scale_units='xy', scale=2, color='orange', alpha=0.8, label='Orientation', width=0.003)

    plt.title("VLN-N1 Obstacle Course Trajectory with Obstacles (2D)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

# parquet_path = Path("/data-25T/InternData-N1/exp2/VLN-N1-hm3d_d435i/data/chunk-003/episode_003012.parquet")
# scene_path = parquet_path.parent.parent.parent
# pdc_file = scene_path / "meta" / "pointcloud.ply"

# df = pd.read_parquet(parquet_path)
df = None  # Placeholder for actual DataFrame loading


def compute_T_b_c_and_T_c_b(first_pose_matrix: np.ndarray)->tuple[np.array, np.array]:
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

if __name__ == "__main__":
    import random

    # search_path = "/data-10T/InternData-N1/hm3d_d435i"
    search_path = "/data-10T/InternData-N1/hm3d_d435i/00075-u3zrj4Nojev__059a3d5940/00075-u3zrj4Nojev"
    print(f"Searching for parquet files in {search_path}...")
    all_files = list(Path(search_path).rglob("*.parquet"))
    print(f"Found {len(all_files)} files.")

    SAMPLE_LIMIT = 10

    if len(all_files) > SAMPLE_LIMIT:
        selected_files = random.sample(all_files, SAMPLE_LIMIT)
    else:
        selected_files = all_files
    
    print(f"Selected {len(selected_files)} files for processing.")

    output_dir = Path("test/obstacle_visual")
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd_cache = {}
    results = []
    all_point_scores = []  # To store every single point score for global stats

    fps = 10
    dt = 1.0 / fps

    # selected_files = [Path("/data-10T/InternData-N1/hm3d_d435i/00723-hWDDQnSDMXb__3764edf97b/00723-hWDDQnSDMXb/data/chunk-000/episode_000065.parquet")]
    for idx, parquet_path in enumerate(selected_files):
        print(f"[{idx+1}/{len(selected_files)}] Processing {parquet_path.name}...")
        try:
            # 1. Load Data
            df = pd.read_parquet(parquet_path)
            
            # Scene Path logic
            # /data/.../chunk-000/episode_000000.parquet
            # scene base: /data/.../
            # Pointclouds are often in ../../../meta/pointcloud.ply relative to the parquet if inside data/chunk
            # Let's verify robustness. 
            # Looking at provided path: .../00001-UVdNNRcVyV1/data/chunk-000/episode_000000.parquet
            # Pointcloud: .../00001-UVdNNRcVyV1/meta/pointcloud.ply
            
            # Go up until we find 'data' then go parent
            parts = parquet_path.parts
            if 'data' in parts:
                data_idx = parts.index('data')
                # scene root is one level above 'data' e.g. .../00001-UV.../
                scene_root = Path(*parts[:data_idx])
                pdc_file = scene_root / "meta" / "pointcloud.ply"
            else:
                # Fallback: assume standard structure 3 levels up
                scene_root = parquet_path.parent.parent.parent
                pdc_file = scene_root / "meta" / "pointcloud.ply"
            
            if not pdc_file.exists():
                print(f"  [WARN] Pointcloud not found at {pdc_file}, skipping.")
                continue

            # 2. Load PointCloud (ESDF)
            # Cache key is the scene root directory name (or full path to be safe)
            scene_key = str(scene_root)
            
            if scene_key not in pcd_cache:
                print(f"  Loading new scene: {scene_key}")
                # Load raw O3D point cloud for visualization
                o3d_pcd = o3d.io.read_point_cloud(str(pdc_file))
                
                # Create ESDF using the loaded point cloud
                esdf = PointCloudESDF(o3d_pcd, device="cuda:0")
                
                pcd_cache[scene_key] = {
                    "esdf": esdf,
                    "pcd": o3d_pcd,
                    "scene_name": scene_root.name
                }
            
            scene_data = pcd_cache[scene_key]
            pcd = scene_data["esdf"]
            vis_pcd = scene_data["pcd"]

            # 3. Compute Poses
            # [N, 4, 4]
            T_w_c = np.vstack(df["action"].to_numpy()).reshape(-1, 4, 4)
            
            # Compute T_c_b for this episode (or use standard if all same)
            # Assuming first frame defines alignment
            T_b_c, T_c_b, _ = compute_T_b_c_and_T_c_b(T_w_c[0])
            
            T_w_b = T_w_c @ T_c_b
            poses_body = get_poses(T_w_b)

            # 4. Compute Scores
            scores = compute_avoidance_scores(poses_body, pcd, dt=dt)
            max_score = scores.max()
            mean_score = scores.mean()
            
            # Store point-wise scores
            all_point_scores.extend(scores.tolist())

            # 5. Add Trajectory to Point Cloud
            # Create color map (Blue to Red)
            cmap = plt.get_cmap("jet")
            colors = cmap(scores)[:, :3]  # Get RGB, ignore alpha
            
            traj_pcd = o3d.geometry.PointCloud()
            traj_pcd.points = o3d.utility.Vector3dVector(poses_body[:, :3])
            traj_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Start Point (Green Sphere)
            start_pos = poses_body[0, :3]
            start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            start_sphere.translate(start_pos)
            start_sphere.paint_uniform_color([0, 1, 0]) # Green
            start_pcd = start_sphere.sample_points_uniformly(number_of_points=200)

            # End Point (Red Sphere)
            end_pos = poses_body[-1, :3]
            end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            end_sphere.translate(end_pos)
            end_sphere.paint_uniform_color([1, 0, 0]) # Red
            end_pcd = end_sphere.sample_points_uniformly(number_of_points=200)

            # Add trajectory and markers to the scene point cloud
            vis_pcd += traj_pcd
            vis_pcd += start_pcd
            vis_pcd += end_pcd

            results.append({
                "path": str(parquet_path),
                "max_score": max_score,
                "mean_score": mean_score,
                "length": len(scores)
            })

            # 6. Plot 2D (visualize only subset of obstacles for speed)
            points, _ = pcd.query(poses_body[:, :3], 100)
            points = points.reshape(-1, 3)
            # print(points.shape)
            
            # Extract scene id (first part before - or _) and episode id
            try:
                # scene_root is like .../00001-UVdNNRcVyV1
                scene_id = scene_root.name.split('-')[0].split('_')[0]
                # parquet_path.stem is like episode_000000
                episode_id = parquet_path.stem.split('_')[-1]
                save_name = f"{scene_id}_{episode_id}.png"
            except Exception:
                # Fallback if any naming convention fails
                save_name = f"{parquet_path.stem}_{idx}.png"

            plot_path = output_dir / save_name
            plot_trajectory_with_obstacles(poses_body, points, scores=scores, save_path=str(plot_path))
        
        except Exception as e:
            print(f"  [ERROR] Failed to process {parquet_path}: {e}")
            import traceback
            traceback.print_exc()

    # 7. Save updated point clouds
    print("\nSaving updated point clouds with trajectories...")
    for scene_key, data in pcd_cache.items():
        scene_name = data["scene_name"]
        vis_pcd = data["pcd"]
        save_path = output_dir / f"{scene_name}.ply"
        print(f"  Saving {save_path} ...")
        o3d.io.write_point_cloud(str(save_path), vis_pcd)

    # 8. Report
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)

    # 6.1 Point-wise Statistics
    if all_point_scores:
        points_arr = np.array(all_point_scores)
        print("GLOBAL POINT-WISE STATISTICS (Avoidance Score):")
        print(f"  Count:    {len(points_arr)}")
        print(f"  Max:      {points_arr.max():.6f}")
        print(f"  Min:      {points_arr.min():.6f}")
        print(f"  Mean:     {points_arr.mean():.6f}")
        print(f"  Variance: {points_arr.var():.6f}")
        print(f"  Q99 (99%):{np.percentile(points_arr, 99):.6f}")
        print(f"  Q01 (1%): {np.percentile(points_arr, 1):.6f}")
        print("-" * 50)
    else:
        print("No scores computed.")

    # Sort by max score descending
    sorted_results = sorted(results, key=lambda x: x['max_score'], reverse=True)
    
    print("Trajectories sorted by Max Avoidance Score (Risk):")
    for i, res in enumerate(sorted_results):
        print(f"{i+1}. Max: {res['max_score']:.4f} | Mean: {res['mean_score']:.4f}")
        print(f"   File: {res['path']}")
    
    # Save full report
    report_df = pd.DataFrame(sorted_results)
    report_df.to_csv(output_dir / "avoidance_report.csv", index=False)
    print(f"\nFull report saved to {output_dir / 'avoidance_report.csv'}")


