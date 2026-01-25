import logging
import warnings
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from argparse import ArgumentParser
from pathlib import Path

# 0. Pre-parse raw_dir for log filename
temp_parser = ArgumentParser(add_help=False)
temp_parser.add_argument("--raw_dir", type=str, default="InternData-n1-demo")
known_args, _ = temp_parser.parse_known_args()
raw_dir_name = Path(known_args.raw_dir).name
# Clean up the name for filename safety
raw_dir_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in raw_dir_name)

# 1. root logger：最低 INFO
root = logging.getLogger()
root.setLevel(logging.INFO)

# 2. 控制台 handler（INFO+）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s"
)
console_handler.setFormatter(console_formatter)

# 3. 文件 handler（WARNING+）
# 使用包含时间戳和 PID 的独立日志文件名，避免多进程冲突
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pid = os.getpid()
log_filename = f"warnings_{raw_dir_name}_{timestamp}_{pid}.log"

file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=50 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",   # 强烈建议
)
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s"
)
file_handler.setFormatter(file_formatter)

# 4. 挂 handler（避免重复挂）
root.handlers.clear()
root.addHandler(console_handler)
root.addHandler(file_handler)

# 5. 捕获 warnings.warn
logging.captureWarnings(True)

import time
import json
from pathlib import Path


from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from utils.lerobot.lerobot_creater import LeRobotCreator
from utils import Trajectories
from utils.vln_n1 import VLN_N1_Trajectories
from argparse import ArgumentParser

parser = ArgumentParser(description="Port VLN-N1 dataset to LeRobotDataset format")
parser.add_argument("--raw_dir", type=str, default="InternData-n1-demo", help="Path to the raw VLN-N1 dataset directory")
parser.add_argument("--output_dir", type=str, default=".", help="Path to the output LeRobotDataset directory")
parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec to use for encoding")
parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for image writing")
args = parser.parse_args()


def port(
    raw_dir: str,
    repo_id: str,
    root: str,
    traj_cls: type[Trajectories],
    num_processes: int,
    codec: str = "h264",
    *args, **kwargs
):
    """Port raw dataset to LeRobotDataset format."""
    logging.info(f"Porting raw dataset from {raw_dir} to LeRobotDataset repo {repo_id}")
    
    # Determine features dynamically
    features = traj_cls.get_features(raw_dir)

    logging.info(f"Creating LeRobotCreator at {root}")
    creator = LeRobotCreator(
        root=str(root),
        robot_type=traj_cls.ROBOT_TYPE,
        fps=traj_cls.FPS,
        features=features,
        num_workers=max(1, num_processes),
        num_video_encoders=max(1, num_processes),
        codec=codec,
        has_extras=True,
    )
    
    def get_task_idx(task: str) -> int:
        return creator.add_task(task)

    trajectories = traj_cls(raw_dir, get_task_idx=get_task_idx, features=features)

    start_time = time.time()
    num_episodes = len(trajectories)
    logging.info(f"Number of episodes {num_episodes}")

    for episode_index, episode in enumerate(trajectories):
        creator.submit_episode(episode)
        
        elapsed_time = time.time() - start_time
        if (episode_index + 1) % 10 == 0:
            logging.info(f"Submitted {episode_index + 1} / {num_episodes} episodes (elapsed {elapsed_time:.3f} s)")

    logging.info("All episodes submitted. Waiting for completion...")
    creator.wait()
    logging.info("LeRobotCreator finished.")
    

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
        num_processes=args.num_processes,
        codec=args.codec,
    )
    
    validate_dataset(f"VLN-N1-{folder_name}", root=root)

if __name__ == "__main__":
    main()

