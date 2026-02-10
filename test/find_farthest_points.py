import argparse
import sys
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from scipy.spatial.transform import Rotation

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.vln_n1.trajectory import VLN_N1_Traj
from utils.coordinate import relative_pose, to_homogeneous, homogeneous_inv

def collect_images_simple(trajectory_dir: Path, episode_index: int, folder_name: str, extensions: list[str]):
    # simplified logic to find images matching VLN_N1 pattern
    # Recursively find the folder
    target_dirs = list(trajectory_dir.rglob(folder_name))
    target_dir = target_dirs[0] if target_dirs else None
    
    if not target_dir or not target_dir.exists():
        return []

    prefix = f"episode_{episode_index:06d}_"
    files = []
    
    # Try extensions in order
    for ext in extensions:
        files = list(target_dir.glob(f"{prefix}*{ext}"))
        if files:
            break
    
    # Sort files
    try:
        images = sorted(files, key=lambda x: int(x.stem.split('_')[-1]))
    except ValueError:
        images = sorted(files)
        
    return images

def main():
    parser = argparse.ArgumentParser(description="Find farthest points in trajectory")
    parser.add_argument("parquet_path", type=str, help="Path to the parquet file")
    args = parser.parse_args()
    
    parquet_path = Path(args.parquet_path).resolve()
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        return

    # Infer trajectory directory (ancestor containing 'meta' folder)
    trajectory_dir = parquet_path.parent
    while len(trajectory_dir.parts) > 1:
        if (trajectory_dir / "meta").exists():
            break
        trajectory_dir = trajectory_dir.parent
    
    if not (trajectory_dir / "meta").exists():
        print(f"Error: Could not locate trajectory root directory from {parquet_path}")
        return
        
    print(f"Trajectory directory: {trajectory_dir}")

    # Extract episode index from parquet filename: episode_000000.parquet
    try:
        start_idx = parquet_path.stem.find('_') + 1
        episode_index = int(parquet_path.stem[start_idx:])
    except ValueError:
        print(f"Error: Could not parse episode index from filename {parquet_path.name}")
        return
        
    print(f"Episode Index: {episode_index}")

    # Collect images
    images = collect_images_simple(trajectory_dir, episode_index, "observation.images.rgb", ['.jpg', '.png'])
    depth_images = collect_images_simple(trajectory_dir, episode_index, "observation.images.depth", ['.png', '.jpg'])
    
    if not images:
        print("Error: No RGB images found.")
        return
    if not depth_images:
        print("Error: No depth images found.")
        return

    print(f"Found {len(images)} images.")

    # Load episode info for tasks
    episodes_path = trajectory_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"Error: episodes.jsonl not found at {episodes_path}")
        return

    episode_info = None
    with open(episodes_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('episode_index') == episode_index:
                    episode_info = data
                    break
            except json.JSONDecodeError:
                continue
    
    if not episode_info:
        print(f"Warning: Episode info for index {episode_index} not found in episodes.jsonl. Using default info.")
        # Create minimal dummy info if not found, though tasks are needed for sub_indexes
        episode_info = {
            "episode_index": episode_index,
            "tasks": [{"sub_instruction": "dummy", "sub_indexes": [0, len(images)-1]}]
        }

    # Prepare frames dict for VLN_N1_Traj
    frames = {
        "parquet_path": parquet_path,
        "images": images,
        "depth_images": depth_images,
        "episode_info": episode_info,
        "trajectory_dir": trajectory_dir
    }

    # Determine original image size from the first image
    image_size = (256, 256)
    if images:
        tmp_img = cv2.imread(str(images[0]))
        if tmp_img is not None:
            # shape is H, W, C
            image_size = (tmp_img.shape[1], tmp_img.shape[0])
            print(f"Detected image size: {image_size}")

    # Initialize Traj
    # We provide a dummy get_task_idx since we don't need task embeddings here
    try:
        traj = VLN_N1_Traj(frames, get_task_idx=lambda x: 0, image_size=image_size,
                           filter_condition={"roll_limit": 40.0})
    except Exception as e:
        print(f"Error initializing VLN_N1_Traj: {e}")
        return

    # Output directory
    output_dir = project_root / "test" / "farthest_point_visual"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving output to {output_dir}")

    # Video writer setup
    video_path = output_dir / "visualization.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, traj.image_size)

    # Helper function to compute pose in first frame coordinate (c_base) adapted from VLN_N1_Traj.__iter__
    T_w_c_base = traj._get_action(0)
    T_w_c_base = traj.roll_to_horizontal(T_w_c_base)
    T_c_base_w = homogeneous_inv(T_w_c_base)

    def get_first_frame_pose(T_w_c):
        T_w_c = traj.roll_to_horizontal(T_w_c)
        T_c_base_c = T_c_base_w @ T_w_c
        R_rel = Rotation.from_matrix(T_c_base_c[:3, :3])
        
        # YXZ convention
        yaw_rel, _, _ = R_rel.as_euler('YXZ', degrees=True)
        
        p_w = T_w_c[:3, 3]
        p_w = to_homogeneous(p_w)
        p_c_base = T_c_base_w @ p_w
        p_c_base = p_c_base[:3] / p_c_base[3]
        x, y, z = p_c_base
        
        # +x front, +y left, +z up
        pose = np.array([-z, -x, y, yaw_rel], dtype=np.float32)
        return pose

    # Loop through frames
    for idx in range(len(traj)):
        # Current pose
        T_w_c_current = traj._get_action(idx)
        current_pose = get_first_frame_pose(T_w_c_current)

        # Find farthest visible frame
        farthest_idx, farthest_T_w_c = traj.find_farthest_visible_frame(idx)
        
        # Check condition: if not found or just next/current frame
        # User defined: "if not found (i.e. farthest point is current point + 1)"
        has_farthest = farthest_idx > idx + 1
        
        # Load image
        img_path = images[idx]
        if not img_path.exists():
            print(f"Warning: Image file missing {img_path}")
            continue
            
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue

        # Resize to match traj size if needed, though loaded images might differ from traj.image_size
        # traj operations (project_camera_point) depend on traj.image_size.
        W, H = traj.image_size
        # Since we initialized traj with image_size from the images, W and H should match cv_img
        
        # Verify size (optional warning)
        if (cv_img.shape[1], cv_img.shape[0]) != (W, H):
             print(f"Warning: Image size {cv_img.shape[:2]} does not match initialized trajectory size ({H}, {W})")

        label_text = "no farthest point"
        color = (0, 0, 255) # Red

        if has_farthest:
            farthest_pose = get_first_frame_pose(farthest_T_w_c)
            
            # Calculate relative pose
            rel_pose_4d = traj.to_4d(
                relative_pose(traj.to_6d(current_pose), traj.to_6d(farthest_pose), degree=True)
            )
            rx, ry, rz, ryaw = rel_pose_4d
            label_text = f"ID:{farthest_idx} x:{rx:.2f} y:{ry:.2f} z:{rz:.2f} yaw:{ryaw:.2f}"
            
            # Project point
            T_c_current_w = homogeneous_inv(T_w_c_current)
            T_c_current_c_target = T_c_current_w @ farthest_T_w_c
            p_c_target_in_current = T_c_current_c_target[:3, 3]
            
            # project_camera_point expects image_shape (H, W)
            proj = traj.project_camera_point(p_c_target_in_current, traj.K, (H, W))
            
            if proj is not None:
                u, v, _ = proj
                cv2.circle(cv_img, (int(u), int(v)), 5, (0, 255, 0), -1) # Green dot
                color = (0, 255, 0)
        
        # Draw label
        cv2.putText(cv_img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save
        out_file = output_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(out_file), cv_img)

        # Write to video
        # video_writer.write(cv_img)
        
        if idx % 10 == 0:
            print(f"Processed frame {idx}/{len(traj)}")

    video_writer.release()
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()
