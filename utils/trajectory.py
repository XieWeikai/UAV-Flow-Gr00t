import pyarrow.parquet as pq
import json
from PIL import Image
import io
from typing import Iterator, Tuple, Dict, List, Any, Iterable
import glob

#default batch size for reading parquet files
DEFAULT_BATCH_SIZE = 1000

# Type alias for clarity
# It represents one trajectory: (trajectory_id, image-bytes iterator, log data)
TrajectoryData = Tuple[str, Iterator[Tuple[int, bytes]], Dict[str, Any]]

class TrajectoryProcessor:
    """
    A class to stream-read and process trajectory data from a Parquet file.

    This class is iterable; each iteration yields one complete trajectory.
    It correctly handles trajectories that span multiple read batches.

    Usage:
        processor = TrajectoryProcessor('path/to/file.parquet')
        for traj_id, images_iterator, log_data in processor:
            # Process each trajectory here
            ...
    """
    def __init__(self, file_path: str, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initialize the processor.
        
        Args:
            file_path (str): Path to the Parquet file.
            batch_size (int): Number of rows to read per batch.
        """
        if not file_path:
            raise ValueError("File path cannot be empty")
        self.file_path = file_path
        self.batch_size = batch_size
        
        # --- Internal state ---
        self._current_traj_id: str | None = None
        self._current_images: List[Tuple[int, bytes]] = []
        self._current_log: Dict[str, Any] | None = None

    def __iter__(self) -> Iterator[TrajectoryData]:
        """
        Implement the iterator protocol to process trajectory by trajectory.
        
        Yields:
            A tuple (traj_id, images_iterator, log_data), where:
            - traj_id (str): Unique ID of the trajectory.
            - images_iterator (Iterator[Tuple[int, bytes]]): Iterator of image bytes for each frame and it corresponding frame index.
            - log_data (Dict): Parsed JSON log data for the trajectory.
        """
        try:
            parquet_file = pq.ParquetFile(self.file_path)
            
            # Read data from the Parquet file in batches
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df_chunk = batch.to_pandas()
                
                # Process rows in the current batch
                for _, row in df_chunk.iterrows():
                    row_id = row['id']
                    
                    # Check if this is a new trajectory ID
                    if self._current_traj_id is not None and row_id != self._current_traj_id:
                        # ID changed -> previous trajectory ended
                        # Yield the collected data for the previous trajectory
                        yield (self._current_traj_id, iter(self._current_images), self._current_log)
                        
                        # Reset state to collect the new trajectory
                        self._current_traj_id = None
                        self._current_images = []
                        self._current_log = None

                    # First trajectory or start of a new one
                    if self._current_traj_id is None:
                        self._current_traj_id = row_id
                        # Log is invariant within a trajectory; parse once
                        self._current_log = json.loads(row['log'])
                    
                    # trajectory log are invariant within a trajectory; assert it is the same
                    assert self._current_log == json.loads(row['log']), "Inconsistent log data within a trajectory" 
                    
                    # Append current row's image bytes to the list
                    self._current_images.append((row['frame_idx'], row['image']['bytes']))
            
            # --- After the loop ---
            # File is done; if there's a last trajectory buffered, yield it
            if self._current_traj_id is not None:
                yield (self._current_traj_id, iter(self._current_images), self._current_log)

        except Exception as e:
            print(f"Error processing file {self.file_path}: {e}")
            raise # Re-raise so the caller is aware


class MultiParquetTrajectoryProcessor:
    """
    Iterate trajectories across multiple Parquet files by delegating to TrajectoryProcessor.

    Files are processed sequentially in the order provided. Trajectories are assumed not to
    span across files; this class does not merge trajectories across file boundaries.

    Example:
        multi = MultiParquetTrajectoryProcessor([
            'UAVFlow/train-00053-of-00054.parquet',
            'UAVFlow/train-00054-of-00054.parquet',
        ], batch_size=200)
        for traj_id, images_iter, log in multi:
            ...
    """

    def __init__(self, file_paths: Iterable[str], batch_size: int = DEFAULT_BATCH_SIZE):
        if file_paths is None:
            raise ValueError("file_paths cannot be None")
        self.file_paths: List[str] = list(file_paths)
        if not self.file_paths:
            raise ValueError("file_paths cannot be empty")
        self.batch_size = batch_size

    @classmethod
    def from_glob(cls, pattern: str, batch_size: int = DEFAULT_BATCH_SIZE) -> "MultiParquetTrajectoryProcessor":
        """Create a processor from a glob pattern, sorted lexicographically."""
        paths = sorted(glob.glob(pattern))
        return cls(paths, batch_size=batch_size)
    
    @classmethod
    def from_dir(cls, dir_path: str, batch_size: int = DEFAULT_BATCH_SIZE) -> "MultiParquetTrajectoryProcessor":
        """Create a processor from all Parquet files in a directory, sorted lexicographically."""
        pattern = f"{dir_path.rstrip('/')}/**/*.parquet"
        paths = sorted(glob.glob(pattern, recursive=True))
        return cls(paths, batch_size=batch_size)

    def __iter__(self) -> Iterator[TrajectoryData]:
        for path in self.file_paths:
            # Delegate to TrajectoryProcessor for each file
            for traj in TrajectoryProcessor(path, batch_size=self.batch_size):
                yield traj


if __name__ == "__main__":
    # Your Parquet file path
    file_path = 'UAVFlow/train-00053-of-00054.parquet'

    # Initialize the trajectory processor
    processor = TrajectoryProcessor(file_path, batch_size=200)

    print("Start iterating through each trajectory...")
    
    # Limit to the first n trajectories for demonstration
    n = 2
    trajectory_count = 0

    # 'processor' is an iterator; each loop returns a complete trajectory
    for traj_id, images_iterator, log_data in processor:
        trajectory_count += 1
        print("-" * 50)
        print(f"Processing trajectory #{trajectory_count}")
        print(f"  Trajectory ID: {traj_id}")
        
        # 'log_data' is already a parsed Python dictionary
        raw_logs = log_data.get('raw_logs', [])
        print(f"  Log info: {len(raw_logs)} records")
        print(f"    First log entry: {raw_logs[0] if raw_logs else 'N/A'}")

        frame_count = 0
        # 'images_iterator' yields image bytes; we can iterate over it
        for frame_idx, image_bytes in images_iterator:
            # Process each frame here (e.g., feed into a model, save locally, etc.)
            frame_count += 1
            
            if frame_count >= 1:
                last_time_stamp = log_data['raw_logs'][frame_idx - 1][-1]
                curr_time_stamp = log_data['raw_logs'][frame_idx][-1]
                print(f"    Frame {frame_idx}: time delta = {curr_time_stamp - last_time_stamp} seconds")

            # For demo, only process the first frame of each trajectory
            if frame_count == 1:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    print(f"  > First frame: format={image.format}, size={image.size}")
                    # image.save(f"{traj_id}_frame_{frame_count}.jpg") # Uncomment to save the image
                except Exception as img_e:
                    print(f"  > Error processing image: {img_e}")
        
        print(f"  This trajectory has {frame_count} frame(s)")

        # For a quick demo, stop after processing the first n trajectories
        if trajectory_count >= n:
            print("-" * 50)
            print(f"Demo finished; processed {n} trajectories.")
            break
        

