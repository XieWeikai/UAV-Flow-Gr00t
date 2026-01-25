import multiprocessing as mp
import shutil
import time
import queue # For access to queue.Empty if needed, though mp.Queue handles it.
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Callable, Iterator, Optional, Union, Tuple, List
import numpy as np
import datasets
from PIL import Image
import torch
import cv2

# Local imports
from .lerobot_metadata import LeRobotMetadata
from .video_utils import encode_video_frames
from .compute_stats import compute_episode_stats
from .image_writer import AsyncImageWriter

# --- Protocol Constants ---
CMD_ALLOCATE_EPISODE = "allocate_episode"
CMD_ADD_TASK = "add_task"
CMD_APPEND_EPISODE = "append_episode"
CMD_APPEND_STATS = "append_stats"
CMD_APPEND_EXTRAS = "append_extras"
CMD_UPDATE_GLOBAL = "update_global"
CMD_STOP = "stop"

# --- Helper Functions ---
def process_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3: # CHW -> HWC
             img = np.transpose(img, (1, 2, 0))
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
             img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
    return img

class MetadataClient:
    """Methods to communicate with the Metadata Process via Queue."""
    def __init__(self, req_queue: mp.Queue, resp_queue: mp.Queue, rank: int):
        self.req_queue = req_queue
        # Use the provided queue for responses (created in main process, passed via inheritance)
        self.resp_queue = resp_queue 
        self.rank = rank

    def allocate_episode_index(self) -> int:
        # Pass rank instead of queue object
        self.req_queue.put((CMD_ALLOCATE_EPISODE, None, self.rank))
        return self.resp_queue.get()

    def add_task(self, task: str) -> int:
        self.req_queue.put((CMD_ADD_TASK, task, self.rank))
        return self.resp_queue.get()

    def append_episode(self, ep_dict: Dict):
        self.req_queue.put((CMD_APPEND_EPISODE, ep_dict, None))

    def append_episode_stats(self, stats: Dict):
        self.req_queue.put((CMD_APPEND_STATS, stats, None))

    def append_episode_extras(self, extras: Dict):
        self.req_queue.put((CMD_APPEND_EXTRAS, extras, None))

    def update_global_stats(self, n_frames: int, n_videos: int):
        self.req_queue.put((CMD_UPDATE_GLOBAL, (n_frames, n_videos), None))


class WorkerEpisodeBuilder:
    """Manages state for building a single episode within a Worker Process."""
    def __init__(self, root: Path, meta_client: MetadataClient, features: Dict, fps: int, video_queue: mp.Queue, codec: str = "h264", pix_fmt: str = "yuv420p", has_extras: bool = False, extra_metadata: Dict = None):
        self.root = root
        self.meta = meta_client
        self.features = features
        self.fps = fps
        self.video_queue = video_queue
        self.codec = codec
        self.pix_fmt = pix_fmt
        self.has_extras = has_extras
        self.extra_metadata = extra_metadata or {}
        
        # 1. Allocate Episode ID
        self.episode_index = self.meta.allocate_episode_index()
        
        # 2. Initialize Buffers
        self.buffer = {k: [] for k, v in features.items() if v["dtype"] not in ["image", "video"]}
        self.buffer["frame_index"] = []
        self.buffer["timestamp"] = []
        self.buffer["task"] = [] 
        
        self.image_keys = [k for k, v in features.items() if v["dtype"] in ["image", "video"]]
        self.image_buffer = {k: [] for k in self.image_keys} 

        self.chunk_size = 1000
        self.chunk = self.episode_index // self.chunk_size
        
        # 3. Prepare Temp Directories
        self.temp_image_dirs = {}
        for key in self.image_keys:
             # Use atomic mkdir or ignore exist error
             p = self.root / "videos" / f"chunk-{self.chunk:03d}" / key / f"episode_{self.episode_index:06d}_temp"
             try:
                p.mkdir(parents=True, exist_ok=True)
             except FileExistsError:
                pass
             self.temp_image_dirs[key] = p
             
        self.frame_count = 0
        self.tasks_set = set()
        
        # 4. Image Writer (Threaded inside this process)
        # Assuming AsyncImageWriter works locally. 
        # Since we are in a worker process, we create a new thread pool here.
        self.image_writer = AsyncImageWriter(num_threads=4)

    def add_frame(self, frame: Dict, task: str):
        self.tasks_set.add(task)
        self.buffer["task"].append(task)
        
        # Save Images
        for key in self.image_keys:
            if key in frame:
                img = process_image(frame[key])
                img_path = self.temp_image_dirs[key] / f"frame_{self.frame_count:06d}.png"
                self.image_writer.save_image(img, img_path)
                self.image_buffer[key].append(str(img_path))
            else:
                self.image_writer.stop()
                raise ValueError(f"Image key {key} missing in frame data")
        
        # Save Data
        for key in self.buffer:
            if key in ["frame_index", "timestamp", "task"]: continue
            if key in frame:
                val = frame[key]
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                self.buffer[key].append(val)
        
        self.buffer["frame_index"].append(self.frame_count)
        self.buffer["timestamp"].append(self.frame_count / self.fps)
        self.frame_count += 1

    def finalize(self):
        if self.frame_count == 0:
            return

        # Ensure all images are written
        self.image_writer.wait_until_done()
        # AsyncImageWriter.stop() might kill executor? 
        # We need to make sure we don't kill it if we reuse it? 
        # But we create one per Builder (per Episode). So it's fine.
        # Actually image_writer shutdown handling needs care.
        # Assuming wait_until_done() is sufficient.
        
        # Resolve Tasks via Metadata Process
        unique_tasks = list(self.tasks_set)
        task_map = {}
        for t in unique_tasks:
            idx = self.meta.add_task(t)
            task_map[t] = idx
            
        # Prepare Data
        data = {k: np.array(v) for k,v in self.buffer.items() if k != "task"}
        data["index"] = np.arange(self.frame_count)
        data["episode_index"] = np.full(self.frame_count, self.episode_index)
        data["task_index"] = np.array([task_map[t] for t in self.buffer["task"]])
        
        # Write Parquet
        chunk_dir = self.root / "data" / f"chunk-{self.chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / f"episode_{self.episode_index:06d}.parquet"
        
        ds = datasets.Dataset.from_dict(data)
        ds.to_parquet(parquet_path)
        
        # Compute Stats
        if compute_episode_stats:
            stats_buffer = {}
            for key in self.features:
                if key in data:
                    stats_buffer[key] = data[key]
                elif key in self.image_keys:
                    stats_buffer[key] = self.image_buffer[key]
            
            try:
                ep_stats = compute_episode_stats(stats_buffer, self.features)
                self.meta.append_episode_stats({
                    "episode_index": self.episode_index,
                    "stats": ep_stats
                })
            except Exception as e:
                logging.error(f"Stats computation failed for ep {self.episode_index}: {e}")
        
        # Submit Video Encoding Jobs
        for key in self.image_keys:
            temp_dir = self.temp_image_dirs[key]
            out_dir = self.root / "videos" / f"chunk-{self.chunk:03d}" / key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"episode_{self.episode_index:06d}.mp4"
            
            self.video_queue.put((str(temp_dir), str(out_path), self.fps, self.codec, self.pix_fmt))

        # Update Meta
        self.meta.append_episode({
            "episode_index": self.episode_index,
            "tasks": unique_tasks,
            "length": self.frame_count
        })

        if self.has_extras:
            self.extra_metadata["episode_index"] = self.episode_index
            self.meta.append_episode_extras(self.extra_metadata)
            
        self.meta.update_global_stats(self.frame_count, len(self.image_keys))


# --- Service Entry Points ---

def metadata_service(root: Path, req_queue: mp.Queue, reply_queues: List[mp.Queue]):
    """Entry point for the Metadata Manager Process."""
    try:
        meta = LeRobotMetadata(root)
        while True:
            try:
                msg = req_queue.get()
            except (EOFError, BrokenPipeError):
                break
                
            cmd, args, rank = msg
            
            if cmd == CMD_STOP:
                meta.flush()
                break
            
            res = None
            try:
                if cmd == CMD_ALLOCATE_EPISODE:
                    res = meta.allocate_episode_index()
                elif cmd == CMD_ADD_TASK:
                    res = meta.add_task(args)
                elif cmd == CMD_APPEND_EXTRAS:
                    meta.append_episode_extras(args)
                elif cmd == CMD_APPEND_EPISODE:
                    meta.append_episode(args)
                elif cmd == CMD_APPEND_STATS:
                    meta.append_episode_stats(args)
                elif cmd == CMD_UPDATE_GLOBAL:
                    meta.update_global_stats(args[0], args[1])
            except Exception as e:
                logging.error(f"Metadata Service Error: {e}")
                traceback.print_exc()
            
            if rank is not None:
                # Reply to the specific worker rank
                reply_queues[rank].put(res)
                
    except Exception as e:
        logging.critical(f"Metadata Process Failed: {e}")
        traceback.print_exc()

def video_encoder_service(video_queue: mp.JoinableQueue):
    """Entry point for Video Encoder Processes."""
    while True:
        try:
            msg = video_queue.get()
        except (EOFError, BrokenPipeError):
            break
            
        if msg == CMD_STOP:
            video_queue.task_done()
            break
        
        try:
            temp_dir, out_path, fps, vcodec, pix_fmt = msg
            encode_video_frames(
                video_path=out_path,
                imgs_dir=temp_dir,
                fps=fps,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
                overwrite=True
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logging.error(f"Video Encoder Failed ({out_path}): {e}")
            traceback.print_exc()
        finally:
            video_queue.task_done()

def worker_service(task_queue: mp.JoinableQueue, meta_req_queue: mp.Queue, resp_queue: mp.Queue, video_queue: mp.JoinableQueue, root: Path, features: Dict, fps: int, rank: int, codec: str = "h264", pix_fmt: str = "yuv420p", has_extras: bool = False):
    """Entry point for Worker Processes."""
    meta_client = MetadataClient(meta_req_queue, resp_queue, rank)
    
    while True:
        try:
            item = task_queue.get()
        except (EOFError, BrokenPipeError):
            break
            
        if item == CMD_STOP:
            task_queue.task_done()
            break
            
        try:
            # Handle incoming task (create iterator)
            extra_metadata = {}
            if callable(item):
                iterator = item()
            else:
                iterator = item
                if has_extras and hasattr(item, "metadata"):
                    extra_metadata = item.metadata
                
            builder = WorkerEpisodeBuilder(root, meta_client, features, fps, video_queue, codec=codec, pix_fmt=pix_fmt, has_extras=has_extras, extra_metadata=extra_metadata)
            
            for element in iterator:
                # Unpack tuple (frame, task) or dict
                if isinstance(element, tuple) and len(element) == 2:
                    frame, task = element
                elif isinstance(element, dict) and "task" in element:
                    task = element["task"]
                    frame = element
                else:
                    # Fallback if structure is unknown; assume element is frame and task is missing?
                    # Raising error is safer.
                    # Or check if last task exists?
                    logging.warning(f"Unexpected item in iterator: {type(element)}. Expecting (frame, task) or frame_dict['task'].")
                    continue
                    
                builder.add_frame(frame, task)
            
            builder.finalize()
            
        except Exception as e:
            logging.error(f"Worker Task Failed: {e}")
            traceback.print_exc()
        finally:
            task_queue.task_done()


# --- Main API ---

class LeRobotCreator:
    def __init__(
        self, 
        root: str, 
        robot_type: str = None, 
        fps: int = 30, 
        features: Dict = None,
        num_workers: int = 4,
        num_video_encoders: int = 2,
        codec: str = "h264",
        pix_fmt: str = "yuv420p",
        has_extras: bool = False
    ):
        self.root = Path(root)
        self.fps = fps
        self.num_workers = num_workers
        self.num_video_encoders = num_video_encoders
        self.codec = codec
        self.pix_fmt = pix_fmt
        self.has_extras = has_extras
        
        # 1. Initialize Global Info (Safe single-process op before forking)
        # this will create meta dir if not exists and init info.json
        if features:
            LeRobotMetadata(self.root).init_info(features, fps, robot_type, codec=codec, pix_fmt=pix_fmt)
        
        # 2. Create Communication Channels
        self.task_queue = mp.JoinableQueue(maxsize=num_workers * 2) 
        self.meta_req_queue = mp.Queue()
        self.video_queue = mp.JoinableQueue()
        
        # Create dedicated reply queues for each worker
        # and the num_workers + 1 for creator if needed
        self.reply_queues = [mp.Queue() for _ in range(num_workers + 1)]
        self.meta_client = MetadataClient(self.meta_req_queue, self.reply_queues[-1], rank=num_workers) # Creator uses last queue
        
        # 3. Start Metadata Process
        # We pass the list of reply queues to metadata service
        self.meta_process = mp.Process(
            target=metadata_service, 
            args=(self.root, self.meta_req_queue, self.reply_queues),
            daemon=True
        )
        self.meta_process.start()
        
        # 4. Start Video Encoders
        self.encoders = []
        for _ in range(num_video_encoders):
            p = mp.Process(
                target=video_encoder_service, 
                args=(self.video_queue,),
                daemon=True
            )
            p.start()
            self.encoders.append(p)
            
        # 5. Start Workers
        self.workers = []
        for i in range(num_workers):
            # Pass the SPECIFIC reply queue for this worker, and its rank
            p = mp.Process(
                target=worker_service, 
                args=(self.task_queue, self.meta_req_queue, self.reply_queues[i], self.video_queue, self.root, features, fps, i, codec, pix_fmt, has_extras),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    def add_task(self, task: str) -> int:
        """Adds a new task to the metadata and returns its index."""
        return self.meta_client.add_task(task)

    def submit_episode(self, episode_iterator: Union[Iterator, Callable[[], Iterator]]):
        """
        Submits an episode for processing. 
        Args:
            episode_iterator: An iterator (or generator function) that yields (frame, task) tuples or dictionaries containing 'task'.
        Blocks if all workers are busy (queue full).
        """
        self.task_queue.put(episode_iterator)

    def wait(self):
        """
        Waits for all submitted episodes to be processed and encoded, then shuts down.
        """
        # 1. Wait for all submitted tasks to be picked up and processed by workers
        self.task_queue.join()
        
        # 2. Stop Workers
        for _ in range(self.num_workers):
            self.task_queue.put(CMD_STOP)
        
        # Wait for workers to finish current items and exit
        for p in self.workers:
            p.join()
            
        # 3. Wait for all video encoding jobs to finish
        self.video_queue.join()
        
        # 4. Stop Encoders
        for _ in range(self.num_video_encoders):
            self.video_queue.put(CMD_STOP)
            
        for p in self.encoders:
            p.join()
            
        # 5. Stop Metadata
        self.meta_req_queue.put((CMD_STOP, None, None))
        self.meta_process.join()
