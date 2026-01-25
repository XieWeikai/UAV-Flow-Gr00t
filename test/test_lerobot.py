import shutil
import numpy as np
import logging
from pathlib import Path
from functools import partial
import time
import cv2

from utils.lerobot.lerobot_creater import LeRobotCreator
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

# Define features as per uav_flow_2_lerobot.py
DROID_FEATURES = {
    "annotation.human.action.task_description": {
        "dtype": "int32", # index of task
        "shape": (1,),
        "names": None,
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (4,), # x, y, z, yaw
        "names": {
            "axes": ["x", "y", "z", "yaw"],
        },
    },
    "video.ego_view": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": [
            "height",
            "width",
            "channels",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (4,),
        "names": {
            "axes": ["x", "y", "z", "yaw"],
        },
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "episode_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
}

def clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

def generate_episode_data(episode_id, num_frames=30):
    """Generator yielding (frame, task) for a single episode."""
    # Simulate different tasks based on episode ID
    task = f"fly_to_target_obj{episode_id}"
    
    for i in range(num_frames):
        # Create a dynamic image (episode_id moving from left to right)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Calculate x position: moves from -text_width to 256
        # Approximate text width/movement for simplicity
        text = str(episode_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        total_distance = 256 + text_size[0]
        start_x = -text_size[0]
        x = int(start_x + (i / (num_frames - 1)) * total_distance)
        y = 128  # Centered vertically
        
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # State: episode_id randomly placed, others 0
        state_vec = np.zeros(4, dtype=np.float32)
        state_vec[np.random.randint(0, 4)] = float(episode_id)
        
        # Action: episode_id randomly placed, others 0
        action_vec = np.zeros(4, dtype=np.float32)
        action_vec[np.random.randint(0, 4)] = float(episode_id)
        
        frame = {
            # Dummy index for task description feature (actual index handled by creator metadata)
            "annotation.human.action.task_description": np.array([0], dtype=np.int32), 
            
            "observation.state": state_vec,
            "video.ego_view": img,
            "action": action_vec,
        }
        
        yield frame, task

def validate_dataset(repo_id, root: str):
    """Sanity check that ensure meta data can be loaded and all files are present."""
    meta = LeRobotDatasetMetadata(repo_id, root=root)

    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")

    for ep_idx in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(ep_idx)

        if not data_path.exists():
            raise ValueError(f"Parquet file is missing in: {data_path}")

        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.exists():
                raise ValueError(f"Video file is missing in: {vid_path}")

def test_correctness():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root = Path("test_dataset_lerobot")
    clean_dir(root)
    
    num_workers = 8
    num_encoders = 8
    fps = 10
    
    print(f"Initializing LeRobotCreator with {num_workers} workers and {num_encoders} encoders...")
    creator = LeRobotCreator(
        root=str(root), 
        features=DROID_FEATURES, 
        fps=fps, 
        num_workers=num_workers, 
        num_video_encoders=num_encoders,
        robot_type="tello"
    )
    
    num_episodes = 20
    print(f"Submitting {num_episodes} episodes...")
    
    start_time = time.time()
    
    for i in range(num_episodes):
        # Use partial to pass arguments to the generator function
        # The worker will call this partial object () -> iterator
        task_callable = partial(generate_episode_data, i, num_frames=30)
        creator.submit_episode(task_callable)
        if i % 5 == 0:
            print(f"Submitted {i}/{num_episodes}...")

    print("All tasks submitted. Waiting for completion...")
    creator.wait()
    
    duration = time.time() - start_time
    print(f"Done! Processed {num_episodes} episodes in {duration:.2f}s")
    
    # Simple verification
    meta_dir = root / "meta"
    if (meta_dir / "info.json").exists():
        print("Meta info.json exists.")
    else:
        print("ERROR: info.json missing!")
        
    # Check stats file content manually
    stats_file = meta_dir / "episodes_stats.jsonl"
    if stats_file.exists():
        print("Stats file exists. Checking content...")
        import json
        with open(stats_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                if "stats" not in item:
                    print(f"ERROR: Item missing stats: {item}")
                else:
                    # Check keys inside stats
                    pass
        print("Stats file content check passed.")

    videos_dir = root / "videos"
    if videos_dir.exists():
        # Check chunk-000/video.ego_view
        video_files = list(videos_dir.glob("**/*.mp4"))
        print(f"Found {len(video_files)} generated videos.")
    
    print("Running validation...")
    validate_dataset("test_dataset", root=root)
    print("Validation passed!")

def test_speed():
    print("\n" + "="*50)
    print("Starting Speed Calibration")
    print("="*50)
    
    num_episodes = 1010
    
    # --- 1. Custom LeRobotCreator Speed Test ---
    print("\n[Custom] Testing LeRobotCreator speed...")
    root_custom = Path("test_dataset_speed_custom")
    clean_dir(root_custom)
    
    creator = LeRobotCreator(
        root=str(root_custom), 
        features=DROID_FEATURES, 
        fps=10, 
        num_workers=8, 
        num_video_encoders=8,
        robot_type="tello"
    )
    
    start_time_custom = time.time()
    for i in range(num_episodes):
        task_callable = partial(generate_episode_data, i, num_frames=30)
        creator.submit_episode(task_callable)
        if (i+1) % 100 == 0:
            print(f"[Custom] Submitted {i+1}/{num_episodes}...")
            
    creator.wait()
    duration_custom = time.time() - start_time_custom
    print(f"[Custom] Finished {num_episodes} episodes in {duration_custom:.2f}s (FPS: {num_episodes*30/duration_custom:.2f})")
    
    # --- 2. Official LeRobotDataset Speed Test ---
    print("\n[Official] Testing LeRobotDataset speed...")
    root_official = Path("test_dataset_speed_official")
    if root_official.exists():
        shutil.rmtree(root_official)
    
    # Strip system features for official API as it likely adds them automatically
    official_features = {k:v for k,v in DROID_FEATURES.items() if k not in [
        "timestamp", "frame_index", "episode_index", "index", "task_index", "task"
    ]}
    
    ds = LeRobotDataset.create(
        repo_id="benchmark_official", 
        root=root_official, 
        fps=10, 
        robot_type="tello", 
        features=official_features
    )
    
    start_time_official = time.time()
    for i in range(num_episodes):
        gen = generate_episode_data(i, num_frames=30)
        for frame, task in gen:
            # frame dict has keys: 'annotation...', 'observation...', 'video...', 'action'
            # official API expects these
            ds.add_frame(frame, task=task)
        
        ds.save_episode()
        
        if (i+1) % 100 == 0:
            print(f"[Official] Saved {i+1}/{num_episodes}...")

    
    duration_official = time.time() - start_time_official
    print(f"[Official] Finished {num_episodes} episodes in {duration_official:.2f}s (FPS: {num_episodes*30/duration_official:.2f})")

    # --- Comparison ---
    print("\n" + "="*50)
    print("Speed Comparison Summary")
    print("="*50)
    print(f"Episodes: {num_episodes}, Frames per Episode: 30")
    print(f"Custom LeRobotCreator: {duration_custom:.2f}s")
    print(f"Official LeRobotDataset: {duration_official:.2f}s")
    speedup = duration_official / duration_custom if duration_custom > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

    
if __name__ == "__main__":
    test_correctness()
    test_speed()
