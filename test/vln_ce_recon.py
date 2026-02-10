import re
import open3d as o3d
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_depth(depth_path: str, 
               mode: str="scale", 
               scale: float=0.001, 
               min_depth: float=0.0,
               max_depth: float=10.0) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load depth image from {depth_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    assert mode in ["scale", "linear"], f"Unsupported mode: {mode}"
    
    if mode == "scale":
        depth = depth.astype(np.float32) * scale
    
    if mode == "linear":
        depth = depth.astype(np.float32)
        depth = min_depth + (max_depth - min_depth) * (depth / 65535.0)
    
    return depth

def habitat_pinhole_K(width: int, height: int, hfov_deg: float, center_mode: str = "pixel"):
    """
    center_mode:
      - "pixel": cx=(W-1)/2, cy=(H-1)/2   （推荐：像素中心坐标系 u∈[0,W-1]）
      - "plane": cx=W/2,     cy=H/2
    """
    W, H = width, height
    hfov = np.deg2rad(hfov_deg)

    fx = (W / 2.0) / np.tan(hfov / 2.0)
    # vfov derived from aspect ratio
    vfov = 2.0 * np.arctan((H / W) * np.tan(hfov / 2.0))
    fy = (H / 2.0) / np.tan(vfov / 2.0)

    if center_mode == "pixel":
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
    elif center_mode == "plane":
        cx = W / 2.0
        cy = H / 2.0
    else:
        raise ValueError("center_mode must be 'pixel' or 'plane'")

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, (fx, fy, cx, cy), vfov

# habitat sensor config example
# rgb_sensor:             
#     width: 640             
#     height: 480             
#     hfov: 79

HFOV = 79.0
HEIGHT = 480
WIDTH = 640

K, (fx, fy, cx, cy), VFOV = habitat_pinhole_K(
    width=WIDTH,
    height=HEIGHT,
    hfov_deg=HFOV,
    center_mode="pixel"
)

def image_to_pointcloud(rgb:np.ndarray, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, T_w_c: np.ndarray, max_depth: float = 4.0) -> o3d.geometry.PointCloud:
    """
    depth: [H, W] depth in meters
    T_w_c: [4, 4] camera to world
    return: points [N, 3]
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    z = depth.flatten()
    valid_mask = (z > 0) & (z <= max_depth)
    
    z = z[valid_mask]
    x = (uu.flatten()[valid_mask] - cx) * z / fx
    y = (vv.flatten()[valid_mask] - cy) * z / fy

    points = np.stack([x, y, z], axis=1)  # [N, 3]
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # [N, 4]
    # [4, 4] @ [4, N] -> [4, N]
    # transpose to [N, 4]
    points_w_h = (T_w_c @ points_h.T).T  # [N, 4]
    
    rgb_flat = rgb.reshape(-1, 3)
    colors = rgb_flat[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_w_h[:, :3])
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float64)) / 255.0)
    return pcd

if __name__ == "__main__":
    parquet_path = Path("/data-10T/InternData-N1/rxr/1LXtFkjw3qL__95f93db5b8/1LXtFkjw3qL/data/chunk-000/episode_000000.parquet")
    df = pd.read_parquet(parquet_path)
    pose_col = df["pose.125cm_0deg"].to_numpy()
    # print(f"pose_col: {pose_col}")

    # [N, 4, 4]
    # np.stack on array of objects (rows) only gives [N, 4], we need explicit stack for inner dimensions
    T_w_c = np.stack([np.stack(p) for p in pose_col])
    print(f"T_w_c shape: {T_w_c.shape}")

    def frame_index(p: Path):
        # 取文件名里最后一段数字（按你的命名可调整）
        m = re.findall(r"\d+", p.stem)
        return int(m[-1]) if m else -1
    
    episode_name = parquet_path.stem
    prefix = f"{episode_name}_"
    scene_dir = parquet_path.parent.parent.parent
    depth_images = (scene_dir / "videos" /"chunk-000" / "observation.images.depth.125cm_0deg").glob(f"{prefix}*.png")
    depth_images = sorted(depth_images, key=frame_index)
    rgb_images = (scene_dir / "videos" /"chunk-000" / "observation.images.rgb.125cm_0deg").glob(f"{prefix}*.jpg")
    rgb_images = sorted(rgb_images, key=frame_index)


    # for depth_path, rgb_path in zip(depth_images, rgb_images):
    #     print(f"Depth: {depth_path}, RGB: {rgb_path}")


    global_pcd = o3d.geometry.PointCloud()
    START_FRAME = 0
    IMAGE_LIMIT = 100
    for frame_id in tqdm(range(START_FRAME, min(START_FRAME + IMAGE_LIMIT, len(depth_images)))):
        depth = load_depth(depth_images[frame_id], mode="scale", scale=0.001)
        rgb_path = rgb_images[frame_id]
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        pcd = image_to_pointcloud(
            rgb=rgb,
            depth=depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            T_w_c=T_w_c[frame_id],
        )
        global_pcd += pcd
        global_pcd = global_pcd.voxel_down_sample(voxel_size=0.005)

    o3d.io.write_point_cloud(f"{episode_name}_pointcloud.ply", global_pcd)
    
