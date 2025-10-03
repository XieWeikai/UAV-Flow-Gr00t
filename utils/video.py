#!/usr/bin/env python

import numpy as np
from pathlib import Path
import shutil

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

    # --- 3. Generate and Add Dummy Trajectories ---

    num_trajectories = 4
    num_frames_per_trajectory = 10

    # The main loop iterates to create each trajectory (episode).
    timestamp = 0.0
    for episode_idx in range(num_trajectories):
        print(f"\nGenerating trajectory {episode_idx + 1}/{num_trajectories}...")
        
        # A dummy task description, unique for each episode.
        task_description = f"Dummy task number {episode_idx}"
        
        # The inner loop creates each frame within a single trajectory.
        for frame_idx in range(num_frames_per_trajectory):
            # Generate dummy data for each feature.
            
            # State: 6 random float values.
            dummy_state = np.random.randn(6).astype(np.float32)
            
            # Image: A 256x256 RGB image. We create a simple gradient.
            r, g, b = np.mgrid[0:256, 0:256, 0:1]
            r = np.full((256, 256, 1), fill_value=frame_idx * 10, dtype=np.uint8) # Red channel changes with frame
            g = np.full((256, 256, 1), fill_value=episode_idx * 50, dtype=np.uint8) # Green channel changes with episode
            b = np.zeros((256, 256, 1), dtype=np.uint8)
            dummy_image = np.concatenate([r, g, b], axis=2)

            # Action: 4 random float values.
            dummy_action = np.random.randn(4).astype(np.float32)
            
            # Assemble the frame dictionary. The keys MUST match DROID_FEATURES.
            frame = {
                "annotation.human.action.task_description": task_description,
                "observation.state.drone": dummy_state,
                "video.ego_view": dummy_image,
                "action.relative_position": dummy_action,
            }
            
            # frame["task"] = task_description
            # frame["observation.state"] = dummy_image
            # frame["action"] = dummy_action
            
            # Add the generated frame to the dataset's internal buffer.
            tds.add_frame(frame, task_description, timestamp)
            # Increment the timestamp for the next frame.
            timestamp += 1.0 / fps
            
        # After all frames for one episode are added, save the episode to disk.
        # This writes the .parquet and .mp4 files for this specific episode.
        print(f"Saving trajectory {episode_idx + 1}...")
        tds.save_episode()

    print(f"\n✅ Successfully created dataset with {num_trajectories} trajectories.")
    print(f"   Check the output files in the '{root_path}' directory.")


if __name__ == "__main__":
    main()
    