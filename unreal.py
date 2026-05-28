from __future__ import annotations

"""Unreal Go2 episode -> LeRobot v2.1 转换入口。

典型用法：
    .\\.venv\\Scripts\\python.exe unreal.py ^
        --raw_dir C:/Data/Saved/scene_0002/szt/episode_000000 ^
        --output_dir ./tmp ^
        --dataset_name unreal_go2_test ^
        --num_processes 1

`--raw_dir` 可传三种层级：
    1. UE OutputRoot，例如 C:/Data/Saved
    2. 某个 scene/user 目录，例如 C:/Data/Saved/scene_0002/szt
    3. 单个 episode 目录，例如 C:/Data/Saved/scene_0002/szt/episode_000000

脚本会递归查找 `episode_meta.json`，只转换 `status == "completed"` 且存在
`frames.jsonl` 的 episode。默认导出 front/rear/left/right 四路 RGB；可用
`--camera_keys front` 或 `--camera_keys front,left,right` 选择子集。

输入 episode 需要包含：
    episode_meta.json
    frames.jsonl
    rgb/<camera>.mp4 或 rgb/<camera>/<00000>.png 序列
    task_info.csv（可选）

输出 LeRobot 数据集包含：
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_extras.jsonl
    data/chunk-000/episode_*.parquet
    videos/chunk-000/video.<camera>/episode_*.mp4

每帧 parquet 字段：
    annotation.human.action.task_description
    observation.state  # [tx, ty, tz, qx, qy, qz, qw], 单位 m, 四元数 xyzw
    action             # 当前复制 observation.state

坐标约定：
    UE 输入：位置 cm，机体系/相机系均为 +X 前、+Y 右、+Z 上。
    输出：位置 m，机体系 +X 前、+Y 左、+Z 上；相机系为 OpenCV +X 右、+Y 下、+Z 前。
    trajectory 的 world 坐标系固定为第一帧机体坐标系，因此第一帧 state 应接近
    [0, 0, 0, 0, 0, 0, 1]。

外参处理：
    `video.<camera>.body_from_camera` 是 episode 级 metadata。因为 UE 每帧都写
    `pose` 和 `camera_pose_<camera>`，本脚本会逐帧反算 T_body<-camera，并严格检查
    一个 episode 内外参是否固定。容差由 `--extrinsic_tolerance_translation_m` 和
    `--extrinsic_tolerance_rotation_deg` 控制。
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from utils.coordinate import homogeneous_inv
from utils.lerobot.lerobot_creater import LeRobotCreator
from utils.rgb_pose_dataset import select_video_pixel_format, transform_to_pose_vector

TASK_DESCRIPTION_KEY = "annotation.human.action.task_description"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
POSE_AXES = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
DEFAULT_CAMERA_KEYS = ("front", "rear", "left", "right")

# UE 录制使用 +X 前、+Y 右、+Z 上；目标机体系要求 +Y 为左，因此只需要翻转 Y 轴。
UE_TO_TARGET = np.diag([1.0, -1.0, 1.0]).astype(np.float32)

# UE 相机局部轴为 +X 前、+Y 右、+Z 上；OpenCV 相机轴为 +X 右、+Y 下、+Z 前。
# 该矩阵把 OpenCV 相机坐标中的点转换到 UE 相机坐标。
UE_CAMERA_FROM_OPENCV = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Unreal Go2 recording episodes to LeRobot v2.1 format.")
    parser.add_argument("--raw_dir", type=str, required=True, help="UE OutputRoot, scene/user dir, or one episode_* dir.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory used to store the exported dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="LeRobot dataset directory name.")
    parser.add_argument("--camera_keys", type=str, default=",".join(DEFAULT_CAMERA_KEYS), help="Comma-separated cameras to export.")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of writer worker processes.")
    parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec.")
    parser.add_argument("--pix_fmt", type=str, default="auto", choices=["auto", "yuv420p", "yuv444p"], help="Video pixel format.")
    parser.add_argument("--extrinsic_tolerance_translation_m", type=float, default=1e-4)
    parser.add_argument("--extrinsic_tolerance_rotation_deg", type=float, default=0.1)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def scan_episode_dirs(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {root}")
    if root.is_file():
        raise ValueError(f"raw_dir must be a directory: {root}")

    if (root / "episode_meta.json").exists():
        return [root]

    return sorted(path.parent for path in root.rglob("episode_meta.json"))


def parse_camera_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise ValueError("camera_keys must not be empty.")
    return keys


def build_features(image_size: tuple[int, int], camera_keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    height, width = image_size
    features: dict[str, dict[str, Any]] = {
        TASK_DESCRIPTION_KEY: {"dtype": "int32", "shape": (1,), "names": None},
        STATE_KEY: {"dtype": "float32", "shape": (7,), "names": {"axes": POSE_AXES}},
    }
    for camera_key in camera_keys:
        features[f"video.{camera_key}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    features[ACTION_KEY] = {"dtype": "float32", "shape": (7,), "names": {"axes": POSE_AXES}}
    return features


def unreal_pose_to_target_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    """将 UE 的 [cm, Roll/Pitch/Yaw] 位姿转换到目标机体系坐标约定下的 SE(3)。"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Unreal pose must have shape (6,), got {pose.shape}")

    location_m = pose[:3] / 100.0
    roll, pitch, yaw = pose[3:]
    rotation_ue = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (UE_TO_TARGET @ rotation_ue @ UE_TO_TARGET).astype(np.float32)
    transform[:3, 3] = (UE_TO_TARGET @ location_m).astype(np.float32)
    return transform


def unreal_camera_pose_to_target_opencv_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    """将 UE 相机 world pose 转为目标 world 下的 OpenCV 相机坐标系 pose。"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Unreal camera pose must have shape (6,), got {pose.shape}")

    location_m = pose[:3] / 100.0
    roll, pitch, yaw = pose[3:]
    rotation_ue = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (UE_TO_TARGET @ rotation_ue @ UE_CAMERA_FROM_OPENCV).astype(np.float32)
    transform[:3, 3] = (UE_TO_TARGET @ location_m).astype(np.float32)
    return transform


def body_from_camera_for_frame(frame: dict[str, Any], camera_key: str) -> np.ndarray:
    # 外参定义为 T_body<-camera，即把 OpenCV 相机坐标中的点变换到目标机体系。
    body_transform = unreal_pose_to_target_transform(frame["pose"])
    camera_transform = unreal_camera_pose_to_target_opencv_transform(frame[f"camera_pose_{camera_key}"])
    return (homogeneous_inv(body_transform) @ camera_transform).astype(np.float32)


def intrinsic_4(frame_or_meta: dict[str, Any], camera_key: str) -> list[float]:
    """从 UE 写出的 3x3 K 展平数组中提取 [fx, fy, cx, cy]。"""
    key = f"K_{camera_key}"
    if key not in frame_or_meta:
        raise ValueError(f"Missing {key}")
    matrix = frame_or_meta[key]
    if len(matrix) != 9:
        raise ValueError(f"{key} must contain 9 values, got {len(matrix)}")
    return [float(matrix[0]), float(matrix[4]), float(matrix[2]), float(matrix[5])]


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3])
    return float(np.degrees(delta.magnitude()))


def validate_fixed_extrinsics(
    episode_dir: Path,
    frames: list[dict[str, Any]],
    camera_keys: list[str],
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict[str, np.ndarray]:
    """严格校验相机安装外参在一个 episode 内保持不变。"""
    if not frames:
        raise ValueError(f"No frames in {episode_dir}")

    baseline = {camera: body_from_camera_for_frame(frames[0], camera) for camera in camera_keys}
    max_translation: dict[str, float] = {camera: 0.0 for camera in camera_keys}
    max_rotation: dict[str, float] = {camera: 0.0 for camera in camera_keys}

    for frame in frames[1:]:
        for camera in camera_keys:
            current = body_from_camera_for_frame(frame, camera)
            trans_delta = float(np.linalg.norm(current[:3, 3] - baseline[camera][:3, 3]))
            rot_delta = rotation_delta_deg(baseline[camera], current)
            max_translation[camera] = max(max_translation[camera], trans_delta)
            max_rotation[camera] = max(max_rotation[camera], rot_delta)

    violations = [
        f"{camera}: translation={max_translation[camera]:.6g}m rotation={max_rotation[camera]:.6g}deg"
        for camera in camera_keys
        if max_translation[camera] > translation_tolerance_m or max_rotation[camera] > rotation_tolerance_deg
    ]
    if violations:
        raise ValueError(f"Dynamic body_from_camera in {episode_dir}: " + "; ".join(violations))

    return baseline


def load_task_info(episode_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    """读取 UE 写出的子任务分段；LeRobot 当前只使用第一个非空 name 作为整段 task。"""
    path = episode_dir / "task_info.csv"
    if not path.exists():
        return "", []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        first_line = file.readline()
        if not first_line.startswith("sep="):
            file.seek(0)
        reader = csv.DictReader(file)
        for row in reader:
            if not row:
                continue
            parsed = dict(row)
            for key in ("subtask_index", "start_frame", "end_frame"):
                if parsed.get(key) not in (None, ""):
                    parsed[key] = int(parsed[key])
            rows.append(parsed)

    task = next((str(row.get("name", "")).strip() for row in rows if str(row.get("name", "")).strip()), "")
    return task, rows


def infer_source_ids(episode_dir: Path) -> tuple[str, str]:
    user_id = episode_dir.parent.name if episode_dir.parent else ""
    scene_id = episode_dir.parent.parent.name if episode_dir.parent and episode_dir.parent.parent else ""
    return scene_id, user_id


@dataclass
class CameraImageSource:
    video_path: Path | None
    image_paths: list[Path] | None
    frame_count: int

    @classmethod
    def from_episode(cls, episode_dir: Path, meta: dict[str, Any], camera_key: str, frame_count: int) -> "CameraImageSource":
        # 优先使用 episode_meta 中记录的视频路径；路径失效时回退到 episode 内的相对 mp4/PNG 序列。
        video_path: Path | None = None
        rgb_video_paths = meta.get("rgb_video_paths") or {}
        video_candidates = []
        if camera_key in rgb_video_paths:
            video_candidates.append(Path(rgb_video_paths[camera_key]))
        video_candidates.append(episode_dir / "rgb" / f"{camera_key}.mp4")

        for candidate in video_candidates:
            if candidate.exists():
                video_path = candidate
                capture = cv2.VideoCapture(str(video_path))
                try:
                    encoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
                finally:
                    capture.release()
                if encoded_count > 0 and encoded_count != frame_count:
                    raise ValueError(
                        f"RGB video frame count mismatch for {episode_dir} camera {camera_key}: "
                        f"expected {frame_count}, got {encoded_count}"
                    )
                break

        if video_path is not None:
            return cls(video_path=video_path, image_paths=None, frame_count=frame_count)

        image_dir = episode_dir / "rgb" / camera_key
        image_paths = sorted(image_dir.glob("*.png"))
        if len(image_paths) != frame_count:
            raise ValueError(
                f"RGB frame count mismatch for {episode_dir} camera {camera_key}: "
                f"expected {frame_count}, got {len(image_paths)}"
            )
        return cls(video_path=None, image_paths=image_paths, frame_count=frame_count)

    def iter_rgb(self):
        if self.video_path is not None:
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                raise ValueError(f"Failed to open video: {self.video_path}")
            try:
                for index in range(self.frame_count):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Video ended early at frame {index}: {self.video_path}")
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                capture.release()
            return

        assert self.image_paths is not None
        for path in self.image_paths:
            with Image.open(path) as image:
                yield np.asarray(image.convert("RGB"))


class UnrealEpisode:
    def __init__(
        self,
        episode_dir: Path,
        meta: dict[str, Any],
        frames: list[dict[str, Any]],
        camera_keys: list[str],
        task: str,
        task_idx: int,
        task_info: list[dict[str, Any]],
        body_from_camera: dict[str, np.ndarray],
    ):
        self.episode_dir = episode_dir
        self.meta = meta
        self.frames = frames
        self.camera_keys = camera_keys
        self.task = task
        self.task_idx = task_idx
        self.task_info = task_info
        self.body_from_camera = body_from_camera
        self.image_sources = {
            camera: CameraImageSource.from_episode(episode_dir, meta, camera, len(frames))
            for camera in camera_keys
        }

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def metadata(self) -> dict[str, Any]:
        scene_id, user_id = infer_source_ids(self.episode_dir)
        metadata: dict[str, Any] = {
            "source_episode_path": str(self.episode_dir),
            "scene_id": scene_id,
            "user_id": user_id,
            "original_episode_index": int(self.meta.get("episode_index", -1)),
            "map_name": self.meta.get("map_name", ""),
            "frame_count": len(self.frames),
            "task_info": self.task_info,
        }
        for camera in self.camera_keys:
            video_key = f"video.{camera}"
            metadata[f"{video_key}.K"] = intrinsic_4(self.frames[0], camera)
            metadata[f"{video_key}.body_from_camera"] = self.body_from_camera[camera]
        return metadata

    def __iter__(self):
        image_iters = {camera: self.image_sources[camera].iter_rgb() for camera in self.camera_keys}
        first_body_inv: np.ndarray | None = None

        for frame in self.frames:
            world_from_body = unreal_pose_to_target_transform(frame["pose"])
            if first_body_inv is None:
                # 按数据规范，trajectory 的 world 取第一帧机体坐标系。
                first_body_inv = homogeneous_inv(world_from_body)
            local_pose = transform_to_pose_vector((first_body_inv @ world_from_body).astype(np.float32))

            item: dict[str, Any] = {
                TASK_DESCRIPTION_KEY: np.array([self.task_idx], dtype=np.int32),
                STATE_KEY: local_pose,
                ACTION_KEY: local_pose.copy(),
            }
            for camera in self.camera_keys:
                item[f"video.{camera}"] = next(image_iters[camera])
            yield item, self.task


class UnrealEpisodeCollection:
    ROBOT_TYPE = "go2"
    INSTRUCTION_KEY = TASK_DESCRIPTION_KEY

    def __init__(
        self,
        raw_dir: str | Path,
        camera_keys: list[str],
        get_task_idx,
        translation_tolerance_m: float,
        rotation_tolerance_deg: float,
    ):
        self.raw_dir = Path(raw_dir)
        self.camera_keys = camera_keys
        self.get_task_idx = get_task_idx
        self.translation_tolerance_m = translation_tolerance_m
        self.rotation_tolerance_deg = rotation_tolerance_deg
        self.episodes = self._load_episodes()

        if not self.episodes:
            raise ValueError(f"No completed Unreal episodes found under {self.raw_dir}")

        first_meta, first_frames = self.episodes[0][1], self.episodes[0][2]
        self.fps = int(round(float(first_meta["sample_rate_hz"])))
        self.image_size = (int(first_meta["capture_height"]), int(first_meta["capture_width"]))
        self.FEATURES = build_features(self.image_size, self.camera_keys)

        for episode_dir, meta, frames, _, _, _ in self.episodes:
            fps = int(round(float(meta["sample_rate_hz"])))
            image_size = (int(meta["capture_height"]), int(meta["capture_width"]))
            if fps != self.fps:
                raise ValueError(f"FPS mismatch in {episode_dir}: expected {self.fps}, got {fps}")
            if image_size != self.image_size:
                raise ValueError(f"Resolution mismatch in {episode_dir}: expected {self.image_size}, got {image_size}")
            if len(frames) != int(meta.get("frame_count", len(frames))):
                raise ValueError(f"frame_count mismatch in {episode_dir}")

    def _load_episodes(self):
        loaded = []
        for episode_dir in scan_episode_dirs(self.raw_dir):
            meta_path = episode_dir / "episode_meta.json"
            frames_path = episode_dir / "frames.jsonl"
            if not frames_path.exists():
                continue
            meta = load_json(meta_path)
            if meta.get("status") != "completed":
                continue

            missing = [camera for camera in self.camera_keys if camera not in (meta.get("camera_names") or [])]
            if missing:
                raise ValueError(f"{episode_dir} is missing cameras in episode_meta.json: {missing}")

            frames = load_jsonl(frames_path)
            for frame in frames:
                for camera in self.camera_keys:
                    if f"camera_pose_{camera}" not in frame or f"K_{camera}" not in frame:
                        raise ValueError(f"{episode_dir} frame {frame.get('frame_index')} missing camera fields for {camera}")

            task, task_info = load_task_info(episode_dir)
            body_from_camera = validate_fixed_extrinsics(
                episode_dir,
                frames,
                self.camera_keys,
                self.translation_tolerance_m,
                self.rotation_tolerance_deg,
            )
            loaded.append((episode_dir, meta, frames, task, task_info, body_from_camera))

        return loaded

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        for episode_dir, meta, frames, task, task_info, body_from_camera in self.episodes:
            task_idx = self.get_task_idx(task)
            yield UnrealEpisode(episode_dir, meta, frames, self.camera_keys, task, task_idx, task_info, body_from_camera)


def validate_lerobot_dataset(repo_id: str, root: str | Path):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")
    for episode_index in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(episode_index)
        if not data_path.exists():
            raise ValueError(f"Parquet file is missing: {data_path}")
        for video_key in meta.video_keys:
            video_path = meta.root / meta.get_video_file_path(episode_index, video_key)
            if not video_path.exists():
                raise ValueError(f"Video file is missing: {video_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or raw_dir.name
    root = output_dir / dataset_name
    camera_keys = parse_camera_keys(args.camera_keys)

    collection = UnrealEpisodeCollection(
        raw_dir=raw_dir,
        camera_keys=camera_keys,
        get_task_idx=lambda _task: 0,
        translation_tolerance_m=args.extrinsic_tolerance_translation_m,
        rotation_tolerance_deg=args.extrinsic_tolerance_rotation_deg,
    )
    resolved_pix_fmt = select_video_pixel_format(collection.image_size, codec=args.codec, pix_fmt=args.pix_fmt)

    creator = LeRobotCreator(
        root=str(root),
        robot_type=UnrealEpisodeCollection.ROBOT_TYPE,
        fps=collection.fps,
        features=collection.FEATURES,
        num_workers=max(1, args.num_processes),
        num_video_encoders=max(1, int(max(1, args.num_processes) * 1.75)),
        codec=args.codec,
        pix_fmt=resolved_pix_fmt,
        has_extras=True,
    )
    collection.get_task_idx = creator.add_task

    start_time = time.time()
    for episode_index, episode in enumerate(collection, start=1):
        creator.submit_episode(episode)
        logging.info("Submitted episode %s / %s: %s", episode_index, len(collection), episode.episode_dir)

    creator.wait()
    validate_lerobot_dataset(repo_id=dataset_name, root=root)
    logging.info("Done! %d episodes in %.2fs -> %s", len(collection), time.time() - start_time, root)


if __name__ == "__main__":
    main()
