import sys
import os
import time
import json
import logging
import pandas as pd
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

from utils.vln_n1.trajectory import VLN_N1_Traj
from utils.vln_n1_v2.trajectory import VLN_N1_V2_Traj, Ignore
from utils.coordinate import PointCloudESDF

# Configuration
# PARQUET_PATH = "/data-10T/InternData-N1/hm3d_d435i/00005-yPKGKBCyYx8__d2af861fa6/00005-yPKGKBCyYx8/data/chunk-000/episode_000000.parquet"
# PARQUET_PATH = "/data-10T/InternData-N1/3dfront_d435i/0a7770c3-fc94-42a5-a955-474b9b471ab0__3353a4f7f7/0a7770c3-fc94-42a5-a955-474b9b471ab0/data/chunk-000/episode_000000.parquet"
# PARQUET_PATH = "/data-10T/InternData-N1/3dfront_d435i/0ed3cea7-f8c0-4f91-87b8-9414ec60a713__62496c921e/0ed3cea7-f8c0-4f91-87b8-9414ec60a713/data/chunk-000/episode_000000.parquet"
# PARQUET_PATH = "/home/ubuntu/xwk/projects/UAV-Flow-Gr00t/test_data/ffd98024-7200-429e-8b9a-1234a5937826__d6567fa4ea/ffd98024-7200-429e-8b9a-1234a5937826/data/chunk-000/episode_000001.parquet"

PARQUET_PATH = "/data-25T/InternData-N1/hm3d_d435i/00001-UVdNNRcVyV1__8b94c4e2ac/00001-UVdNNRcVyV1/data/chunk-000/episode_000000.parquet"
ITERATIONS = 1
ROLL_LIMIT = 45.0

def setup_resources(parquet_path):
    parquet_path = Path(parquet_path).resolve()
    chunk_dir = parquet_path.parent  # chunk-XXX
    data_dir = chunk_dir.parent      # data
    scene_dir = data_dir.parent      # VLN-N1-3dfront_d435i (scene_id in older structure, or dataset root here)
    
    # In this dataset structure, scene_dir seems to be the dataset root equivalent for single scene?
    # Or is it struct: dataset/scene/data/chunk/episode?
    # Based on listing:
    # VLN-N1-3dfront_d435i/
    #   data/chunk-000/
    #   meta/
    #   videos/chunk-000/
    
    # So "scene_dir" here effectively VLN-N1-3dfront_d435i
    
    chunk_name = chunk_dir.name
    episode_name = parquet_path.stem
    episode_idx = int(episode_name.split('_')[1])
    
    # Resource paths
    video_chunk_dir = scene_dir / "videos" / chunk_name
    rgb_dir = video_chunk_dir / "observation.images.rgb"
    depth_dir = video_chunk_dir / "observation.images.depth"
    pcd_path = scene_dir / "meta" / "pointcloud.ply"
    episodes_file = scene_dir / "meta" / "episodes.jsonl"
    
    print(f"Loading resources for {parquet_path}...")
    print(f"  RGB Dir: {rgb_dir}")
    print(f"  PCD Path: {pcd_path}")
    
    if not pcd_path.exists():
        raise FileNotFoundError(f"PCD not found at {pcd_path}")
    
    # Load ESDF (Heavy operation, done once)
    print("  Computing ESDF...")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    esdf = PointCloudESDF(pcd)
    
    # Load Task Info
    print("  Loading Task Info...")
    tasks = []
    if episodes_file.exists():
        with open(episodes_file, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "episode_index" in item and item["episode_index"] == episode_idx:
                        tasks = item.get("tasks", [])
                        break
                except:
                    continue
    else:
        print(f"Warning: Episodes file not found at {episodes_file}")
    
    task_str = json.dumps(tasks)
    
    # Collect Images
    print("  Collecting Images...")
    prefix = f"{episode_name}_"
    rgb_images = sorted(list(rgb_dir.glob(f"{prefix}*.jpg")))
    depth_images = sorted(list(depth_dir.glob(f"{prefix}*.png")))
    
    if not rgb_images:
        raise ValueError("No images found!")
        
    return {
        "parquet_path": parquet_path,
        "esdf": esdf,
        "images": rgb_images,
        "depth_images": depth_images,
        "task_str": task_str,
        "task_idx": 0, # Dummy
        "tasks_list": tasks, # For V1 frame dict
        "scene_dir": scene_dir # needed?
    }

def profile_v1(resources, iterations):
    print(f"\nProfiling V1 (Original) for {iterations} iterations...")
    
    # Prepare V1 input
    frames = {
        "parquet_path": resources["parquet_path"],
        "images": resources["images"],
        "depth_images": resources["depth_images"],
        # V1 parses tasks from episode_info or meta/episodes.jsonl
        # To avoid re-reading file, we can inject episode_info if supported, 
        # or it reads from file. V1 reads from file if episode_info not present.
        # Let's provide episode_info to speed up/avoid IO inside loop if possible 
        # but V1 __init__ logic:
        # if "episode_info" in frames: ... else: read file
        "episode_info": {
            "tasks": resources["tasks_list"],
            "episode_path": "dummy",
            "line_number": 0
        },
        "trajectory_dir": resources["scene_dir"] # fallback
    }
    
    def get_task_idx_fn(x): return 0
    
    total_times = []
    init_times = []
    iter_times = []

    for _ in tqdm(range(iterations)):
        t0 = time.time()
        
        # Init
        traj = VLN_N1_Traj(frames, get_task_idx=get_task_idx_fn, filter_condition={"roll_limit": ROLL_LIMIT})
        
        t1 = time.time()

        # Iterate to trigger processing (lazy or in iter)
        # V1 calculates farther visible frame inside __iter__ or uses precalc?
        # V1 find_farthest_visible_frame is called inside __iter__ loop.
        for f, t in traj:
            # print(f"{f['action'][4:]}")
            pass
            
        t2 = time.time()

        total_times.append(t2 - t0)
        init_times.append(t1 - t0)
        iter_times.append(t2 - t1)
        
    avg_total = np.mean(total_times)
    std_total = np.std(total_times)

    avg_init = np.mean(init_times)
    avg_iter = np.mean(iter_times)

    print(f"V1 Result Total: {avg_total:.4f} s/traj +/- {std_total:.4f}")
    print(f"V1 Split -> Init: {avg_init:.4f} s, Iter: {avg_iter:.4f} s")
    return avg_total

def profile_v2(resources, iterations):
    print(f"\nProfiling V2 (Vectorized) for {iterations} iterations...")

    # Define filter
    def roll_filter(traj: VLN_N1_V2_Traj):
        if abs(90.0 - traj.ori_roll) > ROLL_LIMIT:
            raise Ignore(f"Unexpected roll angle: {traj.ori_roll}°. Expected {90.0 - ROLL_LIMIT}~{90.0 + ROLL_LIMIT}°.")
    
    VLN_N1_V2_Traj.set_filter(roll_filter)
    
    total_times = []
    init_times = []
    iter_times = []

    for _ in tqdm(range(iterations)):
        t0 = time.time()
        
        # Init (lightweight, only precomputes collision using ESDF)
        traj = VLN_N1_V2_Traj(
            parquet_path=resources["parquet_path"],
            esdf=resources["esdf"],
            images=resources["images"],
            depth_images=resources["depth_images"],
            task=resources["task_str"],
            task_idx=resources["task_idx"]
        )
        
        t1 = time.time()
        
        # Iteration triggers process_traj and data generation
        for f, t in traj:
            # print(f"{f['action'][4:]}")
            pass
            
        t2 = time.time()
        
        total_times.append(t2 - t0)
        init_times.append(t1 - t0)
        iter_times.append(t2 - t1)
        
    avg_total = np.mean(total_times)
    std_total = np.std(total_times)
    
    avg_init = np.mean(init_times)
    avg_iter = np.mean(iter_times)

    print(f"V2 Result Total: {avg_total:.4f} s/traj +/- {std_total:.4f}")
    print(f"V2 Split -> Init: {avg_init:.4f} s, Iter: {avg_iter:.4f} s")
    
    return avg_total


if __name__ == "__main__":
    try:
        resources = setup_resources(PARQUET_PATH)
        
        print("start profiling V2\n" + "="*40)
        t2 = profile_v2(resources, ITERATIONS)
        print("\n" + "="*40)
        print("start profiling V1\n" + "="*40)
        t1 = profile_v1(resources, ITERATIONS)
        
        speedup = t1 / t2 if t2 > 0 else 0
        print(f"\nSummary:")
        print(f"V1 (Avg): {t1:.4f} s")
        print(f"V2 (Avg): {t2:.4f} s")
        print(f"Speedup: {speedup:.2f}x")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
