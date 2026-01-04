import logging
import time
from pathlib import Path


from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from utils import Trajectories, get_task_idx
from utils.vln_n1 import VLN_N1_Trajectories
from functools import partial
from argparse import ArgumentParser

from utils.video import use_encoding

logging.basicConfig(
    level=logging.INFO,      # 把门槛降到 INFO
    format='%(asctime)s %(levelname)s %(message)s'
)

parser = ArgumentParser(description="Port VLN-N1 dataset to LeRobotDataset format")
parser.add_argument("--raw_dir", type=str, default="InternData-n1-demo", help="Path to the raw VLN-N1 dataset directory")
parser.add_argument("--output_dir", type=str, default=".", help="Path to the output LeRobotDataset directory")
parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec to use for encoding")
parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for image writing")
parser.add_argument("--num_processes", type=int, default=0, help="Number of processes for image writing")
parser.add_argument("--batch_size", type=int, default=50, help="Batch size for video encoding")
args = parser.parse_args()

use_encoding(args.codec)
    

def port(
    raw_dir: str,
    repo_id: str,
    root: str,
    traj_cls: type[Trajectories],
    num_threads: int,
    num_processes: int,
    batch_size: int,
    *args, **kwargs
):
    """Port raw dataset to LeRobotDataset format."""
    logging.info(f"Porting raw dataset from {raw_dir} to LeRobotDataset repo {repo_id}")
    
    # Determine features dynamically
    features = traj_cls.get_features(raw_dir)

    if root and Path(root).exists():
        logging.info(f"Loading existing dataset from {root}")
        lerobot_dataset = LeRobotDataset(repo_id, root=root, batch_encoding_size=batch_size)
        lerobot_dataset.start_image_writer(num_processes=num_processes, num_threads=num_threads)
    else:
        logging.info(f"Creating new dataset at {root}")
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            robot_type=traj_cls.ROBOT_TYPE,
            fps=traj_cls.FPS,
            features=features,
            image_writer_processes=num_processes,
            image_writer_threads=num_threads,
            batch_encoding_size=batch_size,
        )
    
    trajectories = traj_cls(raw_dir, get_task_idx=partial(get_task_idx, lerobot_dataset), features=features)

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

    # Manually flush any remaining videos when batch_encoding_size > 1.
    if lerobot_dataset.batch_encoding_size > 1 and lerobot_dataset.episodes_since_last_encoding > 0:
        start_ep = lerobot_dataset.num_episodes - lerobot_dataset.episodes_since_last_encoding
        logging.info(
            f"Batch encoding remaining {lerobot_dataset.episodes_since_last_encoding} episodes "
            f"[{start_ep}, {lerobot_dataset.num_episodes})"
        )
        lerobot_dataset.batch_encode_videos(start_episode=start_ep, end_episode=lerobot_dataset.num_episodes)
        lerobot_dataset.episodes_since_last_encoding = 0

    # Ensure asynchronous image writer workers finish before validation.
    lerobot_dataset.stop_image_writer()
    
    



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
        num_threads=args.num_threads,
        num_processes=args.num_processes,
        batch_size=args.batch_size,
    )
    
    validate_dataset(f"VLN-N1-{folder_name}", root=root)

if __name__ == "__main__":
    main()

