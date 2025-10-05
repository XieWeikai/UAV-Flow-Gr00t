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
    parser.add_argument("--max-trajectories", type=int, default=1000, help="Maximum number of trajectories to process.")
    return parser.parse_args()

def get_task_idx(ds: LeRobotDataset, task: str, episode_idx: int=None) -> int:
    """Get the index of a task, adding it if it doesn't exist."""
    task_index = ds.meta.get_task_index(task)
    if task_index is None:
        ds.meta.add_task(task)
        task_index = ds.meta.get_task_index(task)
    return task_index


import signal

def main():
    FPS_NOT_MATCH_TIMES = 0

    args = parse_args()
    from utils.rotation import relative_pose_given_axes
    from utils.trajectory import MultiParquetTrajectoryProcessor

    repo_id = args.repo_id
    root_path = Path("./") / repo_id
    fps = args.fps
    robot_type = args.robot_type
    if fps <= 0:
        raise ValueError("fps must be a positive integer")

    if root_path.exists():
        print(f"Removing existing directory: {root_path}")
        shutil.rmtree(root_path)

    print(f"Initializing dataset at: {root_path}")
    tds = LeRobotDataset.create(
        repo_id=repo_id,
        root=root_path,
        fps=fps,
        robot_type=robot_type,
        features=DROID_FEATURES
    )

    uav_flow_path = args.data_path
    uav_flow_dataset = MultiParquetTrajectoryProcessor.from_dir(uav_flow_path)

    POSE_INDICES={"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 5, "yaw": 4}

    episode_idx = 0
    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        print("\nKeyboardInterrupt received. Will save current episode before exiting...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        for _, traj_images, log in uav_flow_dataset:
            for instruction_key in ["instruction", "instruction_unified"]:
                task = log[instruction_key]

                last_timestamp = 0.0
                added_frames = 0
                for frame_idx, img in traj_images:
                    if interrupted:
                        break
                    curr_frame_timestamp = log['raw_logs'][frame_idx][-1]
                    if frame_idx > 0:
                        delta_time = curr_frame_timestamp - last_timestamp
                        last_timestamp = curr_frame_timestamp
                        if delta_time - 1.0 / fps > 1e-1:
                            print(f"Frame interval {delta_time} does not match fps {fps}")
                            FPS_NOT_MATCH_TIMES += 1
                    else:
                        last_timestamp = curr_frame_timestamp

                    state = np.array([0.0], dtype=np.float32)

                    image = Image.open(io.BytesIO(img))
                    ego_view = np.array(image)

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

                    task_idx = get_task_idx(tds, task, episode_idx)
                    tds.add_frame({
                        "annotation.human.action.task_description": np.array([task_idx], dtype=np.int32),
                        "observation.state": state,
                        "video.ego_view": ego_view,
                        "action": action,
                    }, task)
                    added_frames += 1

                if added_frames > 0:
                    episode_idx += 1
                    print(f"saved trajectory with {frame_idx+1} frames, task: {task}")
                    tds.save_episode()
                n = args.max_trajectories
                if episode_idx >= n or interrupted:
                    if interrupted:
                        print("Gracefully saved episode after interruption.")
                    print(f"Demo finished; processed {episode_idx} trajectories.")
                    return
        print(f"All done! Total trajectories processed: {episode_idx}")
        print(f"Total times fps mismatch detected: {FPS_NOT_MATCH_TIMES}")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught in main loop. Saving current episode if needed...")
        tds.save_episode()
        print("Episode saved. Exiting.")


if __name__ == "__main__":
    main()

    