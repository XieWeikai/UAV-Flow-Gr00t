from lerobot.datasets.lerobot_dataset import LeRobotDataset

def get_task_idx(ds: LeRobotDataset, task: str) -> int:
    """Get the index of a task, adding it if it doesn't exist."""
    task_index = ds.meta.get_task_index(task)
    if task_index is None:
        ds.meta.add_task(task)
        task_index = ds.meta.get_task_index(task)
    return task_index

class Traj:
    """
    Single trajectory, iterable to output each frame.
    """
    def __init__(self, frames):
        raise NotImplementedError("Traj is an abstract class and cannot be instantiated directly.")

    def __len__(self):
        raise NotImplementedError("Traj is an abstract class and cannot be instantiated directly.")

    def __iter__(self):
        raise NotImplementedError("Traj is an abstract class and cannot be instantiated directly.")

class Trajectories:
    """
    abstract class representing a collection of trajectories.
    """
    FPS: int = None
    ROBOT_TYPE: str = None
    FEATURES: dict = None
    INSTRUCTION_KEY: str = None
    
    def __init__(self, data_path: str):
        raise NotImplementedError("Trajectories is an abstract class and cannot be instantiated directly.")

    def __len__(self):
        raise NotImplementedError("Trajectories is an abstract class and cannot be instantiated directly.")

    def __iter__(self):
        raise NotImplementedError("Trajectories is an abstract class and cannot be instantiated directly.")
