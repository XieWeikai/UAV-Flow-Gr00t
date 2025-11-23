import logging
import time
import shutil
import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from utils import get_task_idx, Traj
from utils.vln_n1 import VLN_N1_Trajectories
from functools import partial
from argparse import ArgumentParser

parser = ArgumentParser(description="Port VLN-N1 dataset to LeRobotDataset format")
parser.add_argument("--raw_dir", type=str, default="InternData-n1-demo", help="Path to the raw VLN-N1 dataset directory")
parser.add_argument("--output_dir", type=str, default=".", help="Path to the output LeRobotDataset directory")
args = parser.parse_args()
    

def port(
    raw_dir: str,
    repo_id: str,
    root: str,
    traj_cls: Traj,
    *args, **kwargs
):
    """Port raw dataset to LeRobotDataset format."""
    logging.info(f"Porting raw dataset from {raw_dir} to LeRobotDataset repo {repo_id}")

    if root and Path(root).exists():
        logging.info(f"Loading existing dataset from {root}")
        lerobot_dataset = LeRobotDataset(repo_id, root=root)
    else:
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            robot_type=traj_cls.ROBOT_TYPE,
            fps=traj_cls.FPS,
            features=traj_cls.FEATURES,
        )
    trajectories = traj_cls(raw_dir, get_task_idx=partial(get_task_idx, lerobot_dataset))

    start_time = time.time()
    num_episodes = len(trajectories)
    logging.info(f"Number of episodes {num_episodes}")

    for episode_index, episode in enumerate(trajectories, start=1):
        elapsed_time = time.time() - start_time
        logging.info(f"{episode_index} / {num_episodes} episodes processed (after {elapsed_time:.3f} seconds)")

        for frame, task in episode:
            lerobot_dataset.add_frame(frame, task=task)

        lerobot_dataset.save_episode()
        logging.info("Save_episode")



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


def main():
    raw_dir = Path(args.raw_dir)
    folder_name = raw_dir.name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / f"VLN-N1-{folder_name}"
    port(
        raw_dir=raw_dir,
        repo_id=f"VLN-N1-{folder_name}",
        root=root,
        traj_cls=VLN_N1_Trajectories,
    )
    
    validate_dataset(f"VLN-N1-{folder_name}", root=root)

if __name__ == "__main__":
    main()

