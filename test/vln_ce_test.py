"""
OpenGL camera: +x right, +y up, +z backwards
OpenCV/Habitat camera: +x right, +y down, +z forward
Our body coord: +x forward, +y left, +z up
"""

import numpy as np
import pandas as pd
import glob
import os
import random

from scipy.spatial.transform import Rotation
from utils.coordinate import homogeneous_inv
from utils.draw import plot_2d_trajectory_with_yaw, animate_trajectory_with_goals


T_b_c = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

T_c_b = homogeneous_inv(T_b_c)

def get_poses(T: np.array)->np.array:
    """
    Get poses in [x, y, z, yaw] format
    use coordinate system where +x is forward, +y is left, +z is up
    0 yaw is facing +x direction
    90 yaw is facing +y direction
    yaw is between -180 to 180 degrees
    pitch and roll should be close to zero
    args:
        T: [N, 4, 4]
    returns:
        poses: [N, 4] in [x, y, z, yaw] format
    """
    yaw, pitch, roll = Rotation.from_matrix(T[:, :3, :3]).as_euler('ZYX', degrees=True).T
    positions = T[:, :3, 3]

    # assert np.all(np.abs(pitch) < 1e-3), "pitch is not close to zero"
    # assert np.all(np.abs(roll) < 1e-3), "roll is not close to zero"
    
    # [x, y, z, yaw]
    poses = np.concatenate([
        positions, yaw[:, None]], axis=1)
    return poses


def process_parquet(parquet_file, output_dir, idx):
    try:
        df = pd.read_parquet(parquet_file)
        
        # [N, 4, 4]
        T_w_c = df["pose.125cm_0deg"].to_numpy()

        # [N, 4, 4]
        T_w_b = T_w_c @ T_c_b

        # [4, 4] use the first frame as body frame
        T_w_body = T_w_b[0]
        T_body_w = homogeneous_inv(T_w_body)

        # [4, 4] x [N, 4, 4] = [N, 4, 4]
        T_body_b = np.einsum('ij,njk->nik', T_body_w, T_w_b)
        poses_body = get_poses(T_body_b)

        # [N-1, 4, 4]
        T_body__b_current = T_body_b[:-1]
        T_b_current__body = homogeneous_inv(T_body__b_current)
        # [N-1, 4, 4]
        T_body__b_next = T_body_b[1:]
        T_b_current__b_next = T_b_current__body @ T_body__b_next
        action_deltas = get_poses(T_b_current__b_next)
        print(f"Action deltas:\n{action_deltas[:3]}")

        # Prepare output paths
        basename = os.path.basename(parquet_file).replace('.parquet', '')
        # Handle cases where basename might be same (e.g. episode_000.parquet in different chunks)
        # Using idx to ensure uniqueness
        unique_name = f"{idx:03d}_{basename}"
        
        static_path = os.path.join(output_dir, f"{unique_name}_traj.png")
        anim_path = os.path.join(output_dir, f"{unique_name}.gif")

        # Static plot
        plot_2d_trajectory_with_yaw(
            poses_body, 
            save_path=static_path,
            title=f"Trajectory: {basename} (Body Frame)"
        )

        # Animation
        goal_frame_idx = df["relative_goal_frame_id.125cm_45deg"].to_numpy()
        animate_trajectory_with_goals(
            poses_body, 
            goal_frame_idx, 
            save_path=anim_path,
            title=f"Goal Nav: {basename}"
        )

        # -1: placeholer 0: stop 1: forward 2: left 3: right
        actions = df["action"].to_numpy()
        refined_goal_frame_idx = goal_frame_idx.copy()
        for i in range(len(refined_goal_frame_idx)):
            if refined_goal_frame_idx[i] != -1:
                continue
            
            # If current goal is -1, search forward for next 'forward' action (action == 1)
            # Starting search from i+1
            found_goal_idx = -1
            for j in range(i + 1, len(actions)):
                if actions[j] == 1:
                    found_goal_idx = j
                    break
            
            if found_goal_idx != -1:
                refined_goal_frame_idx[i] = found_goal_idx - i # Store relative index
        
        refined_anim_path = os.path.join(output_dir, f"{unique_name}_refined.gif")
        animate_trajectory_with_goals(
            poses_body, 
            refined_goal_frame_idx, 
            save_path=refined_anim_path,
            title=f"Goal Nav Refined: {basename}"
        )

    except Exception as e:
        print(f"Error processing {parquet_file}: {e}")


if __name__ == "__main__":
    # root_dir = "/data-10T/InternData-N1/r2r"
    # output_dir = "test/vln_ce_visual"
    # os.makedirs(output_dir, exist_ok=True)

    # print(f"Searching for parquet files in {root_dir} ...")
    
    # # More efficient finding if we know the structure depth, but recursive is safest given variations
    # files = glob.glob(os.path.join(root_dir, "**", "*.parquet"), recursive=True)
    
    # print(f"Found {len(files)} files.")
    
    # if not files:
    #     print("No files found, exiting.")
    #     exit()

    # sample_size = 1
    # if len(files) > sample_size:
    #     sampled_files = random.sample(files, sample_size)
    # else:
    #     sampled_files = files
        
    # print(f"Sampling {len(sampled_files)} files for processing...")
    
    # for i, f in enumerate(sampled_files):
    #     print(f"[{i+1}/{len(sampled_files)}] Processing {f}...")
    #     process_parquet(f, output_dir, i)
        
    # print(f"All done! Results saved to {output_dir}")

    process_parquet("/data-10T/InternData-N1/r2r/1LXtFkjw3qL__95f93db5b8/1LXtFkjw3qL/data/chunk-000/episode_000000.parquet", "test/vln_ce_visual", 0)


