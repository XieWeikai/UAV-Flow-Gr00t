#!/usr/bin/env python

import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import io
import argparse

# Import the main class from the lerobot library
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Define the features (schema) for our drone dataset.
# This structure tells the dataset writer what kind of data to expect.
DROID_FEATURES = {
    # The language instruction for the task.
    "annotation.human.action.task_description": {
        "dtype": "int32", # index of task
        "shape": (1,),
        "names": None,
    },
    # The drone's internal state (e.g., from an IMU or flight controller).
    "observation.state": {
        "dtype": "float32",
        "shape": (1,), # our current UAV can not provide full 6-DoF state, this is a placeholder that always has the value [0]
        "names": {
            "axes": ["ignore"],
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

def parse_args():
    """Parse CLI arguments for configuring dataset generation."""
    parser = argparse.ArgumentParser(description="Generate a LeRobot dataset from UAVFlow trajectories.")
    parser.add_argument("--repo-id", type=str, default="UAVFlowLeRobot", help="Output dataset repo/directory name.")
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second for the dataset.")
    parser.add_argument("--data-path", type=str, default="./UAVFlow", help="Path to the source UAVFlow data directory.")
    parser.add_argument("--robot-type", type=str, default="tello", help="Robot type identifier for LeRobot dataset.")
    return parser.parse_args()


def main():
    """
    Main function to generate a dummy LeRobot dataset.
    """
    # --- 1. Setup Paths and Parameters ---
    args = parse_args()
    # Defer heavy or optional imports until after parsing args so `--help` works without full deps
    from utils.rotation import relative_pose_given_axes
    from utils.trajectory import MultiParquetTrajectoryProcessor

    # Define the repository ID, which will also be the name of the output directory.
    repo_id = args.repo_id
    # Set the root path where the dataset will be saved.
    root_path = Path("./") / repo_id
    fps = args.fps
    robot_type = args.robot_type
    if fps <= 0:
        raise ValueError("fps must be a positive integer")
    
    # Clean up the directory if it exists from a previous run.
    if root_path.exists():
        print(f"Removing existing directory: {root_path}")
        shutil.rmtree(root_path)

    # --- 2. Initialize the LeRobotDataset object ---
    
    # Use LeRobotDataset.create to set up an empty dataset with our desired schema.
    print(f"Initializing dataset at: {root_path}")
    tds = LeRobotDataset.create(
        repo_id=repo_id,
        root=root_path,
        fps=fps,
        robot_type=robot_type,
        features=DROID_FEATURES
    )

    # --- 3. Generate and Add Trajectories ---
    
    uav_flow_path = args.data_path
    uav_flow_dataset = MultiParquetTrajectoryProcessor.from_dir(uav_flow_path)
    
    POSE_INDICES={"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 5, "yaw": 4}
    
    task_id = 0
    for _, traj_images, log in uav_flow_dataset:
        for instruction_key in ["instruction", "instruction_unified"]:
            task = log[instruction_key]
            
            last_timestamp = 0.0
            added_frames = 0
            for frame_idx, img in traj_images:
                curr_frame_timestamp = log['raw_logs'][frame_idx][-1]
                if frame_idx > 0:
                    delta_time = curr_frame_timestamp - last_timestamp
                    last_timestamp = curr_frame_timestamp
                    assert delta_time - 1.0 / fps < 1e-1, f"Frame interval {delta_time} does not match fps {fps}"
                else:
                    last_timestamp = curr_frame_timestamp
                    
                # state
                # just a placeholder
                state = np.array([0.0], dtype=np.float32)

                # observation
                image = Image.open(io.BytesIO(img))
                # Optionally, you can resize or process the image here
                # image = image.resize((256, 256))
                ego_view = np.array(image)
                
                # action
                raw_logs = log['raw_logs']
                last_frame_idx = max(0, frame_idx - 1)
                last_pose = [raw_logs[last_frame_idx][i] for i in POSE_INDICES.values()]
                curr_pose = [raw_logs[frame_idx][i] for i in POSE_INDICES.values()]
                
                action = relative_pose_given_axes(
                    np.array(last_pose, dtype=np.float32), np.array(curr_pose, dtype=np.float32),
                    axes=["x", "y", "z", "yaw"],
                    degree=True
                )
                
                action = action[[0, 1, 2, 5]].astype(np.float32) # x,y,z,yaw

                # Add the processed data to the dataset
                tds.add_frame({
                    "annotation.human.action.task_description": np.array([task_id], dtype=np.int32),
                    "observation.state": state,
                    "video.ego_view": ego_view,
                    "action": action,
                }, task)
                added_frames += 1
                
            if added_frames > 0:
                task_id += 1
                
                print(f"saved trajectory with {frame_idx+1} frames, task: {task}")
                tds.save_episode()
            n = 2
            if task_id >= n:
                print(f"Demo finished; processed {n} trajectories.")
                return


if __name__ == "__main__":
    main()
    