import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class LeRobotMetadata:
    """
    Manages dataset metadata.
    
    Key Features:
    - Buffers episodes, stats, tasks, and INFO updates to minimize I/O.
    - Uses in-memory counters for ID allocation, syncing to `info.json` during flush.
    
    NOTE: Not thread-safe or process-safe. Expected to be used by a single dedicated metadata process.
    """
    def __init__(self, root: str | Path, buffer_size: int = 7500):
        self.root = Path(root)
        self.meta_dir = self.root / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        
        self.info_path = self.meta_dir / "info.json"
        self.tasks_path = self.meta_dir / "tasks.jsonl"
        self.episodes_path = self.meta_dir / "episodes.jsonl"
        self.stats_path = self.meta_dir / "episodes_stats.jsonl"
        self.extras_path = self.meta_dir / "episodes_extras.jsonl"
        
        # Buffer configuration
        self.buffer_size = buffer_size
        self.op_count = 0
        
        # Data Buffers
        self.episode_buffer = []
        self.stats_buffer = []
        self.task_buffer = []
        self.extras_buffer = []
        
        # In-memory Task Map
        self.tasks_map: Dict[str, int] = {} # task_str -> task_index
        
        # Load info.json into memory
        self.info_content = self._read_json_safe(self.info_path) or {}
        
        # Initial Load
        self._load_tasks_map()

    def _read_json_safe(self, path: Path, default=None):
        """Reads JSON."""
        if not path.exists():
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default

    def _write_json_safe(self, path: Path, data: Dict):
        """Writes JSON."""
        with open(path, "w") as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
            f.flush()
            os.fsync(f.fileno())

    def _load_tasks_map(self):
        """Loads existing tasks from disk to memory."""
        if self.tasks_path.exists():
            try:
                with open(self.tasks_path, "r") as f:
                    for line in f:
                        if line.strip():
                            t = json.loads(line)
                            self.tasks_map[t["task"]] = t["task_index"]
            except Exception:
                pass

    def init_info(self, features: Dict, fps: int, robot_type: str = None, codec: str = "h264", pix_fmt: str = "yuv420p"):
        """One-time initialization of info.json."""
        if not self.info_path.exists():
            # Deep copy features to avoid mutating the original dict which might be used elsewhere
            import copy
            features = copy.deepcopy(features)
            
            # 1. Add video info
            for key, feat in features.items():
                if feat["dtype"] == "video":
                    # Assume shape is (h, w, c) or (c, h, w)? 
                    # Looking at provided info.json: "shape": [270, 480, 3] -> (H, W, C)
                    # "names": ["height", "width", "channels"] confirms H, W, C.
                    if "shape" in feat and len(feat["shape"]) == 3:
                        h, w, c = feat["shape"]
                        if "info" not in feat or feat["info"] is None:
                            feat["info"] = {
                                "video.height": h,
                                "video.width": w,
                                "video.codec": codec,
                                "video.pix_fmt": pix_fmt,
                                "video.is_depth_map": False,
                                "video.fps": fps,
                                "video.channels": c,
                                "has_audio": False
                            }

            # 2. Add standard fields if missing
            standard_fields = {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None}
            }
            
            for k, v in standard_fields.items():
                if k not in features:
                    features[k] = v

            info = {
                "codebase_version": "v2.1",
                "robot_type": robot_type,
                "fps": fps,
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 0,
                "total_videos": 0,
                "total_chunks": 0,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": features,
                "splits": {}
            }
            self._write_json_safe(self.info_path, info)
            self.info_content = info
        elif not self.info_content:
             self.info_content = self._read_json_safe(self.info_path) or {}

    def _increment_counter(self, counter_key: str) -> int:
        """
        Increment of a global counter in-memory. 
        Updates are flushed to disk periodically.
        """
        if not self.info_content:
             # Try reload just in case init happened externally or delayed
             self.info_content = self._read_json_safe(self.info_path)
             if not self.info_content:
                  raise RuntimeError("info.json not initialized")

        idx = self.info_content.get(counter_key, 0)
        self.info_content[counter_key] = idx + 1
        
        if counter_key == "total_episodes":
            total = self.info_content[counter_key]
            if "splits" not in self.info_content: self.info_content["splits"] = {}
            self.info_content["splits"]["train"] = f"0:{total}"
            
            chunks_size = self.info_content.get("chunks_size", 1000)
            if total > 0:
                self.info_content["total_chunks"] = (total - 1) // chunks_size + 1
            else: 
                self.info_content["total_chunks"] = 0
        
        # Defer write to flush()
        return idx

    def allocate_episode_index(self) -> int:
        """ Allocates the next available episode index. """
        return self._increment_counter("total_episodes")

    def add_task(self, task: str) -> int:
        """ Returns task index. Uses memory cache. """
        if task in self.tasks_map:
            return self.tasks_map[task]
        
        # New Task: Allocate ID
        new_idx = self._increment_counter("total_tasks")
        
        self.tasks_map[task] = new_idx
        self.task_buffer.append({"task_index": new_idx, "task": task})
        self._check_flush_condition()
        return new_idx

    def append_episode(self, episode_dict: Dict):
        """Buffers episode metadata."""
        self.episode_buffer.append(episode_dict)
        self._check_flush_condition()

    def append_episode_stats(self, stats_dict: Dict):
        """Buffers episode stats."""
        self.stats_buffer.append(stats_dict)

    def append_episode_extras(self, extras_dict: Dict):
        """Buffers episode extras."""
        self.extras_buffer.append(extras_dict)

    def update_global_stats(self, num_frames: int, num_videos: int):
        if self.info_content:
             self.info_content["total_frames"] = self.info_content.get("total_frames", 0) + num_frames
             self.info_content["total_videos"] = self.info_content.get("total_videos", 0) + num_videos

    def _check_flush_condition(self):
        self.op_count += 1
        if self.op_count >= self.buffer_size:
            self._flush_internal()

    def _flush_internal(self):
        # Swap buffers
        eps = self.episode_buffer
        stats = self.stats_buffer
        tasks = self.task_buffer
        extras = self.extras_buffer
        
        # Sort buffers (ascending as requested)
        if tasks:
            tasks.sort(key=lambda x: x.get("task_index", -1), reverse=False)
        if eps:
            eps.sort(key=lambda x: x.get("episode_index", -1), reverse=False)
        if stats:
             stats.sort(key=lambda x: x.get("episode_index", -1), reverse=False)
        if extras:
             extras.sort(key=lambda x: x.get("episode_index", -1), reverse=False)
        
        self.episode_buffer = []
        self.stats_buffer = []
        self.task_buffer = []
        self.extras_buffer = []
        self.op_count = 0
        
        # 1. Write Tasks
        if tasks:
            with open(self.tasks_path, "a") as f:
                for item in tasks:
                    f.write(json.dumps(item) + "\n")

        # 2. Write Episodes
        if eps:
            with open(self.episodes_path, "a") as f:
                for item in eps:
                    f.write(json.dumps(item, cls=NumpyEncoder) + "\n")

        # 3. Write Stats
        if stats:
            with open(self.stats_path, "a") as f:
                for item in stats:
                    f.write(json.dumps(item, cls=NumpyEncoder) + "\n")

        # 4. Write Extras
        if extras:
            with open(self.extras_path, "a") as f:
                for item in extras:
                    f.write(json.dumps(item, cls=NumpyEncoder) + "\n")
        
        # 5. Update Info Stats
        if self.info_content:
             self._write_json_safe(self.info_path, self.info_content)

    def flush(self):
        """Manual flush."""
        self._flush_internal()

    def __del__(self):
        try:
            self.flush()
        except:
            pass
