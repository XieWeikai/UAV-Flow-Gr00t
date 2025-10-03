#!/usr/bin/env python

import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import io

from utils.trajectory import MultiParquetTrajectoryProcessor

# Import the main class from the lerobot library
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Define the features (schema) for our drone dataset.
# This structure tells the dataset writer what kind of data to expect.
DROID_FEATURES = {
    # The language instruction for the task.
    "annotation.human.action.task_description": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    # The drone's internal state (e.g., from an IMU or flight controller).
    "observation.state.drone": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["x", "y", "z", "roll", "pitch", "yaw"],
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
    "action.relative_position": {
        "dtype": "float32",
        "shape": (4,),
        "names": {
            "axes": ["x", "y", "z", "yaw"],
        },
    },
}

def main():
    """
    Main function to generate a dummy LeRobot dataset.
    """
    # --- 1. Setup Paths and Parameters ---

    # Define the repository ID, which will also be the name of the output directory.
    repo_id = "my-awesome-drone-dataset"
    # Set the root path where the dataset will be saved.
    root_path = Path("./") / repo_id
    fps = 5
    
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
        robot_type="tello",
        features=DROID_FEATURES
    )

    # --- 3. Generate and Add Trajectories ---
    
    uav_flow_path = "./UAVFlow"
    uav_flow_dataset = MultiParquetTrajectoryProcessor.from_dir(uav_flow_path)
    
    for _, traj_images, log in uav_flow_dataset:
        for instruction_key in ["instruction", "instruction_unified"]:
            task = log[instruction_key]
            first_frame_timestamp = log['raw_logs'][0][-1]
            
            for frame_idx, img in traj_images:
                frame_timestamp = log['raw_logs'][frame_idx][-1]
                relative_time = frame_timestamp - first_frame_timestamp
                # state
                state = np.array(log["preprocessed_logs"][frame_idx], dtype=np.float32)

                # observation
                image = Image.open(io.BytesIO(img))
                # Optionally, you can resize or process the image here
                # image = image.resize((256, 256))
                ego_view = np.array(image)
                
                # action
                # NOTE: 这个action需要特别处理
                # 此处的action是每一帧相对于起始位置（trajectory中的第一帧）的相对位置
                # 但实际希望的action是这一帧之后要怎么飞，即这一帧之后的位置和现在的相对位置
                # 由于yaw也变了，所以x, y也不能直接相减，要以飞机当前视角来变换相对位置
                # 这个相对位置根据当前帧的不同会有不同的变化，不是固定的
                # 只能在data transform中计算，Gr00t中有相关transform
                # 此处只能保留绝对位置，参考EmbodimentTag.OXE_DROID的data config
                # 参考state_action.py中的RotationTransform
                idx = [0, 1, 2, -1]
                action = np.array([log["preprocessed_logs"][i] for i in idx], dtype=np.float32)

                # Add the processed data to the dataset
                tds.add_frame({
                    "annotation.human.action.task_description": task,
                    "observation.state.drone": state,
                    "video.ego_view": ego_view,
                    "action.relative_position": action,
                })


if __name__ == "__main__":
    main()
    