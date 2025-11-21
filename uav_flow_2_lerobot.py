#!/usr/bin/env python

import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import io
from utils import get_task_idx
import argparse

# Global counters for reporting (will be printed in the __main__ block)
TRAIN_COUNT = 0
EVAL_COUNT = 0
TOTAL_FRAMES_TRAIN = 0
TOTAL_FRAMES_EVAL = 0
TOTAL_SECONDS_TRAIN = 0.0
TOTAL_SECONDS_EVAL = 0.0
FPS_NOT_MATCH_TIMES = 0


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
        "shape": (4,), # our current UAV can not provide full 6-DoF state, this is a placeholder that always has the value [0]
        "names": {
            "axes": ["x", "y", "z", "yaw"],
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
    parser.add_argument(
        "--train-trajectories",
        type=str,
        default="1000",
        help="Number of trajectories for the training split. Use 'inf' to process all available trajectories.",
    )
    parser.add_argument(
        "--eval-trajectories",
        type=str,
        default="200",
        help="Number of trajectories for the eval split. Use 'inf' to process all available trajectories.",
    )
    return parser.parse_args()


def parse_trajectories_arg(val: str):
    """Parse trajectory count CLI arg. Accepts integer strings or 'inf' to mean unlimited."""
    if val is None:
        return 0
    v = str(val).strip().lower()
    if v in ("inf", "infty", "infinite", "+inf", "all"):
        return float("inf")
    try:
        n = int(float(v))
        return max(0, n)
    except Exception:
        raise ValueError(f"Invalid trajectory count: {val}")




import signal

def main():
    args = parse_args()
    # use module-level globals for reporting
    global TRAIN_COUNT, EVAL_COUNT, TOTAL_FRAMES_TRAIN, TOTAL_FRAMES_EVAL, TOTAL_SECONDS_TRAIN, TOTAL_SECONDS_EVAL, FPS_NOT_MATCH_TIMES
    from utils.coordinate import relative_pose_given_axes
    from utils.uavflow.trajectory import MultiParquetTrajectoryProcessor

    repo_id = args.repo_id
    # Create two datasets: one for train and one for eval
    repo_id_train = f"{repo_id}-train"
    repo_id_eval = f"{repo_id}-eval"
    root_path_train = Path("./") / repo_id_train
    root_path_eval = Path("./") / repo_id_eval
    fps = args.fps
    robot_type = args.robot_type
    if fps <= 0:
        raise ValueError("fps must be a positive integer")

    # Clean previous outputs if they exist
    if root_path_train.exists():
        print(f"Removing existing directory: {root_path_train}")
        shutil.rmtree(root_path_train)
    if root_path_eval.exists():
        print(f"Removing existing directory: {root_path_eval}")
        shutil.rmtree(root_path_eval)

    print(f"Initializing train dataset at: {root_path_train}")
    tds_train = LeRobotDataset.create(
        repo_id=repo_id_train,
        root=root_path_train,
        fps=fps,
        robot_type=robot_type,
        features=DROID_FEATURES
    )

    print(f"Initializing eval dataset at: {root_path_eval}")
    tds_eval = LeRobotDataset.create(
        repo_id=repo_id_eval,
        root=root_path_eval,
        fps=fps,
        robot_type=robot_type,
        features=DROID_FEATURES
    )

    uav_flow_path = args.data_path
    uav_flow_dataset = MultiParquetTrajectoryProcessor.from_dir(uav_flow_path)

    POSE_INDICES={"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 5, "yaw": 4}

    train_target = parse_trajectories_arg(args.train_trajectories)
    eval_target = parse_trajectories_arg(args.eval_trajectories)

    # local aliases (write-through to globals)
    train_count = TRAIN_COUNT
    eval_count = EVAL_COUNT
    total_frames_train = TOTAL_FRAMES_TRAIN
    total_frames_eval = TOTAL_FRAMES_EVAL
    total_seconds_train = TOTAL_SECONDS_TRAIN
    total_seconds_eval = TOTAL_SECONDS_EVAL

    # Track which dataset we're currently writing into (for graceful interrupts)
    current_tds = None
    episode_idx = 0  # total episodes processed across both splits (for logging only)
    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        print("\nKeyboardInterrupt received. Will save current episode before exiting...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)

    done = False
    try:
        for _, traj_images, log in uav_flow_dataset:
            for instruction_key in ["instruction", "instruction_unified"]:
                # Decide which split to write to next: save eval first, then train
                if eval_count < eval_target:
                    current_tds = tds_eval
                    split_name = "eval"
                elif train_count < train_target:
                    current_tds = tds_train
                    split_name = "train"
                else:
                    # Both targets satisfied; finish processing loop and exit main so reporting happens in __main__
                    print(f"Reached targets: processed train={train_count}, eval={eval_count}. Stopping further processing.")
                    done = True
                    break

                task = log[instruction_key]

                last_timestamp = 0.0
                added_frames = 0
                first_frame_pose = [log['raw_logs'][0][i] for i in POSE_INDICES.values()]
                first_frame_pose = np.array(first_frame_pose, dtype=np.float32)
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


                    image = Image.open(io.BytesIO(img))
                    ego_view = np.array(image)

                    raw_logs = log['raw_logs']
                    next_frame_idx = min(frame_idx + 1, len(raw_logs) - 1)
                    next_pose = [raw_logs[next_frame_idx][i] for i in POSE_INDICES.values()]
                    curr_pose = [raw_logs[frame_idx][i] for i in POSE_INDICES.values()]
                    
                    state = relative_pose_given_axes(
                        first_frame_pose, np.array(curr_pose, dtype=np.float32),
                        axes=["x", "y", "z", "yaw"],
                        degree=True
                    )
                    
                    state = state[[0, 1, 2, 5]].astype(np.float32) # x,y,z,yaw

                    action = relative_pose_given_axes(
                        np.array(curr_pose, dtype=np.float32), np.array(next_pose, dtype=np.float32),
                        axes=["x", "y", "z", "yaw"],
                        degree=True
                    )

                    action = action[[0, 1, 2, 5]].astype(np.float32) # x,y,z,yaw

                    task_idx = get_task_idx(current_tds, task)
                    current_tds.add_frame({
                        "annotation.human.action.task_description": np.array([task_idx], dtype=np.int32),
                        "observation.state": state,
                        "video.ego_view": ego_view,
                        "action": action,
                    }, task)
                    added_frames += 1

                if added_frames > 0:
                    # compute trajectory duration using raw_logs first and last frame timestamps
                    raw_logs = log['raw_logs']
                    try:
                        first_ts = float(raw_logs[0][-1])
                        last_ts = float(raw_logs[frame_idx][-1])
                    except Exception:
                        # fallback if indexing fails; set duration 0
                        first_ts = 0.0
                        last_ts = 0.0
                    duration = max(0.0, last_ts - first_ts)

                    episode_idx += 1
                    if split_name == "train":
                        train_count += 1
                        total_frames_train += added_frames
                        total_seconds_train += duration
                    else:
                        eval_count += 1
                        total_frames_eval += added_frames
                        total_seconds_eval += duration

                    print(f"[{split_name}] saved trajectory with {added_frames} frames, task: {task}")
                    current_tds.save_episode()

                if interrupted:
                    print("Gracefully saved episode after interruption.")
                    print(f"Processed so far -> train={train_count}, eval={eval_count} trajectories.")
                    # stop processing and return to let __main__ report
                    done = True
                    break

            if done or interrupted:
                break
        # update module-level globals from local counters before exiting main
        TRAIN_COUNT = train_count
        EVAL_COUNT = eval_count
        TOTAL_FRAMES_TRAIN = total_frames_train
        TOTAL_FRAMES_EVAL = total_frames_eval
        TOTAL_SECONDS_TRAIN = total_seconds_train
        TOTAL_SECONDS_EVAL = total_seconds_eval
        # FPS_NOT_MATCH_TIMES is updated in place
        return
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught in main loop. Saving current episode if needed...")
        if current_tds is not None:
            current_tds.save_episode()
        print("Episode saved. Exiting.")
        # update globals from locals (if locals exist in scope)
        try:
            TRAIN_COUNT = train_count
            EVAL_COUNT = eval_count
            TOTAL_FRAMES_TRAIN = total_frames_train
            TOTAL_FRAMES_EVAL = total_frames_eval
            TOTAL_SECONDS_TRAIN = total_seconds_train
            TOTAL_SECONDS_EVAL = total_seconds_eval
        except Exception:
            pass
        return


if __name__ == "__main__":
    main()

    # Print final report using module-level globals
    def _fmt(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s - h*3600 - m*60
        return f"{h}h {m}m {sec:.2f}s"

    print("--- Final report ---")
    print(f"Total times fps mismatch detected: {FPS_NOT_MATCH_TIMES}")
    print(f"Eval -> trajectories: {EVAL_COUNT}, frames: {TOTAL_FRAMES_EVAL}, total duration: {_fmt(TOTAL_SECONDS_EVAL)}")
    print(f"Train -> trajectories: {TRAIN_COUNT}, frames: {TOTAL_FRAMES_TRAIN}, total duration: {_fmt(TOTAL_SECONDS_TRAIN)}")

    