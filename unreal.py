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
如果输入根目录混有旧格式或不完整 episode，可加 `--skip_invalid_episodes`
跳过不兼容条目。转换报告会写入 `meta/unreal_conversion_report.json`，其中记录
已准备提交、实际成功落盘、失败或跳过的 episode。
如果同一个 raw_dir 中混有多套 fps/分辨率，可加 `--split_by_schema`，脚本会按
`sample_rate_hz + capture_height + capture_width` 拆成多个 LeRobot 数据集，例如：
    output/3d_simu_ue_pointnav_fps10_720x1280
    output/3d_simu_ue_pointnav_fps30_480x640
拆分模式下还会额外写出总报告：
    output/3d_simu_ue_pointnav_conversion_report.json
如果旧数据存在“frames.jsonl 比 episode_meta.frame_count 多 1 行”的半帧问题，可加
`--trim_extra_tail_frame`。脚本只会在最后一行 frame_index 正好等于 frame_count 时，
在内存中裁掉最后一行，不会修改原始 episode 文件。

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
from datetime import datetime, timezone
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
    parser.add_argument(
        "--skip_invalid_episodes",
        action="store_true",
        help="Skip incompatible episodes and record them in meta/unreal_conversion_report.json.",
    )
    parser.add_argument(
        "--split_by_schema",
        action="store_true",
        help="Export one LeRobot dataset per fps/resolution schema.",
    )
    parser.add_argument(
        "--trim_extra_tail_frame",
        action="store_true",
        help="Trim one extra tail frame from frames.jsonl when it is exactly meta.frame_count + 1.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default)
        file.write("\n")


def load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def episode_schema(meta: dict[str, Any]) -> tuple[int, tuple[int, int]]:
    fps = int(round(float(meta["sample_rate_hz"])))
    image_size = (int(meta["capture_height"]), int(meta["capture_width"]))
    return fps, image_size


def schema_suffix(schema: tuple[int, tuple[int, int]]) -> str:
    fps, (height, width) = schema
    return f"fps{fps}_{height}x{width}"


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
            "task": self.task,
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
        skip_invalid_episodes: bool = False,
        target_schema: tuple[int, tuple[int, int]] | None = None,
        keep_all_schemas: bool = False,
        trim_extra_tail_frame: bool = False,
        initial_episodes: list[tuple] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
        initial_repairs: list[dict[str, Any]] | None = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.camera_keys = camera_keys
        self.get_task_idx = get_task_idx
        self.translation_tolerance_m = translation_tolerance_m
        self.rotation_tolerance_deg = rotation_tolerance_deg
        self.skip_invalid_episodes = skip_invalid_episodes
        self.target_schema = target_schema
        self.keep_all_schemas = keep_all_schemas
        self.trim_extra_tail_frame = trim_extra_tail_frame
        self.failed_episodes: list[dict[str, Any]] = list(initial_failures or [])
        self.repaired_episodes: list[dict[str, Any]] = list(initial_repairs or [])
        self.prepared_episodes: list[dict[str, Any]] = []
        self.successful_episodes: list[dict[str, Any]] = []
        self.schema_groups: dict[str, dict[str, Any]] = {}
        self.schema_valid_episodes: list[tuple] = []
        self.episodes = list(initial_episodes) if initial_episodes is not None else self._load_episodes()

        if not self.episodes:
            if self.skip_invalid_episodes:
                self.fps = 0
                self.image_size = (0, 0)
                self.FEATURES = {}
                return
            raise ValueError(f"No completed Unreal episodes found under {self.raw_dir}")

        schema_candidates: list[tuple[tuple[int, tuple[int, int]], tuple]] = []
        for episode in self.episodes:
            episode_dir, meta, frames, _, _, _ = episode
            try:
                frames = self._repair_frames_if_needed(episode_dir, meta, frames)
                episode = (episode_dir, meta, frames, episode[3], episode[4], episode[5])
                schema_candidates.append((episode_schema(meta), episode))
            except Exception as exc:
                self._record_failure(episode_dir, "schema_validation", exc)

        self.schema_valid_episodes = [episode for _, episode in schema_candidates]
        self.schema_groups = {}
        for schema, _ in schema_candidates:
            key = schema_suffix(schema)
            if key not in self.schema_groups:
                fps, image_size = schema
                self.schema_groups[key] = {
                    "schema_key": key,
                    "fps": fps,
                    "image_size": image_size,
                    "num_episodes": 0,
                }
            self.schema_groups[key]["num_episodes"] += 1

        if not schema_candidates:
            if self.skip_invalid_episodes:
                self.fps = 0
                self.image_size = (0, 0)
                self.FEATURES = {}
                return
            raise ValueError(f"No schema-compatible Unreal episodes found under {self.raw_dir}")

        selected_schema = self.target_schema or schema_candidates[0][0]
        self.fps, self.image_size = selected_schema
        self.FEATURES = build_features(self.image_size, self.camera_keys)

        compatible = []
        for schema, episode in schema_candidates:
            episode_dir = episode[0]
            if self.keep_all_schemas or schema == selected_schema:
                compatible.append(episode)
                continue
            self._record_failure(
                episode_dir,
                "schema_validation",
                ValueError(
                    f"Schema mismatch: expected {schema_suffix(selected_schema)}, got {schema_suffix(schema)}"
                ),
            )

        self.episodes = compatible
        if not self.episodes:
            if self.skip_invalid_episodes:
                self.fps = 0
                self.image_size = (0, 0)
                self.FEATURES = {}
                return
            raise ValueError(f"No compatible Unreal episodes found under {self.raw_dir}")

    def _load_episodes(self):
        loaded = []
        episode_dirs = scan_episode_dirs(self.raw_dir)
        logging.info("Found %d episode_meta.json files under %s", len(episode_dirs), self.raw_dir)

        for index, episode_dir in enumerate(episode_dirs, start=1):
            logging.info("Scanning episode %d / %d: %s", index, len(episode_dirs), episode_dir)
            meta_path = episode_dir / "episode_meta.json"
            frames_path = episode_dir / "frames.jsonl"
            try:
                if not frames_path.exists():
                    raise ValueError("missing frames.jsonl")
                meta = load_json(meta_path)
                if meta.get("status") != "completed":
                    reason = f"episode status is {meta.get('status')!r}, expected 'completed'"
                    self.failed_episodes.append(
                        {
                            "source_episode_path": str(episode_dir),
                            "stage": "episode_status",
                            "error": reason,
                        }
                    )
                    logging.info("Skipping non-completed episode at %s: %s", episode_dir, reason)
                    continue

                missing = [camera for camera in self.camera_keys if camera not in (meta.get("camera_names") or [])]
                if missing:
                    raise ValueError(f"missing cameras in episode_meta.json: {missing}")

                frames = load_jsonl(frames_path)
                for frame in frames:
                    for camera in self.camera_keys:
                        if f"camera_pose_{camera}" not in frame or f"K_{camera}" not in frame:
                            raise ValueError(f"frame {frame.get('frame_index')} missing camera fields for {camera}")

                task, task_info = load_task_info(episode_dir)
                logging.info("Validating fixed camera extrinsics for %s", episode_dir)
                body_from_camera = validate_fixed_extrinsics(
                    episode_dir,
                    frames,
                    self.camera_keys,
                    self.translation_tolerance_m,
                    self.rotation_tolerance_deg,
                )
                loaded.append((episode_dir, meta, frames, task, task_info, body_from_camera))
                logging.info("Accepted episode %s with %d frames", episode_dir, len(frames))
            except Exception as exc:
                self._record_failure(episode_dir, "episode_scan", exc)

        return loaded

    def _record_failure(self, episode_dir: Path, stage: str, error: Exception):
        failure = {
            "source_episode_path": str(episode_dir),
            "stage": stage,
            "error": str(error),
        }
        self.failed_episodes.append(failure)
        if self.skip_invalid_episodes:
            logging.warning("Skipping invalid episode at %s during %s: %s", episode_dir, stage, error)
            return
        raise error

    def _repair_frames_if_needed(self, episode_dir: Path, meta: dict[str, Any], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expected = int(meta.get("frame_count", len(frames)))
        if len(frames) == expected:
            return frames

        if self.trim_extra_tail_frame and len(frames) == expected + 1:
            last_frame_index = frames[-1].get("frame_index")
            if last_frame_index == expected:
                repair = {
                    "source_episode_path": str(episode_dir),
                    "stage": "frame_count_repair",
                    "action": "trimmed_extra_tail_frame",
                    "meta_frame_count": expected,
                    "original_frame_lines": len(frames),
                    "used_frame_count": expected,
                    "trimmed_frame_index": last_frame_index,
                }
                self.repaired_episodes.append(repair)
                logging.warning(
                    "Trimming one extra tail frame in %s: meta.frame_count=%d frames.jsonl=%d",
                    episode_dir,
                    expected,
                    len(frames),
                )
                return frames[:expected]

        raise ValueError(f"frame_count mismatch: meta={meta.get('frame_count')} frames={len(frames)}")

    def for_schema(self, schema: tuple[int, tuple[int, int]]) -> "UnrealEpisodeCollection":
        return UnrealEpisodeCollection(
            raw_dir=self.raw_dir,
            camera_keys=self.camera_keys,
            get_task_idx=self.get_task_idx,
            translation_tolerance_m=self.translation_tolerance_m,
            rotation_tolerance_deg=self.rotation_tolerance_deg,
            skip_invalid_episodes=True,
            target_schema=schema,
            trim_extra_tail_frame=self.trim_extra_tail_frame,
            initial_episodes=self.schema_valid_episodes,
            initial_failures=self.failed_episodes,
            initial_repairs=self.repaired_episodes,
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        for episode_dir, meta, frames, task, task_info, body_from_camera in self.episodes:
            task_idx = self.get_task_idx(task)
            try:
                episode = UnrealEpisode(episode_dir, meta, frames, self.camera_keys, task, task_idx, task_info, body_from_camera)
            except Exception as exc:
                self._record_failure(episode_dir, "episode_prepare", exc)
                continue

            self.prepared_episodes.append(
                {
                    "source_episode_path": str(episode_dir),
                    "original_episode_index": int(meta.get("episode_index", -1)),
                    "frame_count": len(frames),
                    "task": task,
                }
            )
            yield episode

    def sync_successful_episodes_from_output(self, root: Path):
        """从实际写出的 extras metadata 回读成功 episode，避免把仅提交到队列的任务误报为成功。"""
        extras_path = root / "meta" / "episodes_extras.jsonl"
        extras = load_jsonl_dicts(extras_path)
        if not extras:
            logging.warning("No episodes_extras.jsonl found or no extras written at %s", extras_path)
            self.successful_episodes = []
        else:
            self.successful_episodes = [
                {
                    "source_episode_path": str(item.get("source_episode_path", "")),
                    "episode_index": item.get("episode_index"),
                    "original_episode_index": item.get("original_episode_index"),
                    "frame_count": item.get("frame_count"),
                    "task": item.get("task", ""),
                }
                for item in extras
            ]

        success_sources = {item["source_episode_path"] for item in self.successful_episodes if item["source_episode_path"]}
        existing_failures = {
            (item.get("source_episode_path"), item.get("stage"))
            for item in self.failed_episodes
        }
        for item in self.prepared_episodes:
            source_path = item["source_episode_path"]
            key = (source_path, "output_validation")
            if source_path not in success_sources and key not in existing_failures:
                self.failed_episodes.append(
                    {
                        "source_episode_path": source_path,
                        "stage": "output_validation",
                        "error": "episode was submitted but no episodes_extras entry was written",
                    }
                )
                existing_failures.add(key)

    def build_report(self, root: Path, started_at: str, completed_at: str | None, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "raw_dir": str(self.raw_dir),
            "output_root": str(root),
            "camera_keys": self.camera_keys,
            "selected_schema": {
                "fps": self.fps,
                "image_size": self.image_size,
                "schema_key": schema_suffix((self.fps, self.image_size)) if self.fps else "",
            },
            "schema_groups": list(self.schema_groups.values()),
            "num_prepared": len(self.prepared_episodes),
            "num_successful": len(self.successful_episodes),
            "num_failed": len(self.failed_episodes),
            "num_repaired": len(self.repaired_episodes),
            "prepared_episodes": self.prepared_episodes,
            "successful_episodes": self.successful_episodes,
            "failed_episodes": self.failed_episodes,
            "repaired_episodes": self.repaired_episodes,
        }


def write_conversion_report(root: Path, report: dict[str, Any]):
    report_path = root / "meta" / "unreal_conversion_report.json"
    write_json(report_path, report)
    logging.info(
        "Wrote conversion report: %s (successful=%s failed=%s)",
        report_path,
        report.get("num_successful"),
        report.get("num_failed"),
    )


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


def run_conversion(collection: UnrealEpisodeCollection, root: Path, dataset_name: str, args, started_at: str) -> dict[str, Any]:
    if len(collection) == 0:
        completed_at = utc_now_iso()
        report = collection.build_report(root, started_at, completed_at, "no_valid_episodes")
        write_conversion_report(root, report)
        raise ValueError(f"No compatible Unreal episodes found under {collection.raw_dir}")

    logging.info(
        "Prepared %d compatible episodes for %s; skipped/failed during scan: %d",
        len(collection),
        root,
        len(collection.failed_episodes),
    )
    resolved_pix_fmt = select_video_pixel_format(collection.image_size, codec=args.codec, pix_fmt=args.pix_fmt)
    logging.info("Using fps=%s image_size=%s pix_fmt=%s", collection.fps, collection.image_size, resolved_pix_fmt)

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
    status = "failed"
    try:
        for episode_index, episode in enumerate(collection, start=1):
            logging.info("Submitting episode %s / %s: %s", episode_index, len(collection), episode.episode_dir)
            creator.submit_episode(episode)

        logging.info("Waiting for worker processes and video encoders to finish")
        creator.wait()
        logging.info("Reading written episode metadata from %s", root / "meta" / "episodes_extras.jsonl")
        collection.sync_successful_episodes_from_output(root)
        logging.info("Validating generated LeRobot dataset at %s", root)
        validate_lerobot_dataset(repo_id=dataset_name, root=root)
        status = "completed"
    finally:
        if status != "completed":
            collection.sync_successful_episodes_from_output(root)
        completed_at = utc_now_iso()
        report = collection.build_report(root, started_at, completed_at, status)
        write_conversion_report(root, report)

    logging.info("Done! %d episodes in %.2fs -> %s", len(collection.successful_episodes), time.time() - start_time, root)
    return collection.build_report(root, started_at, utc_now_iso(), status)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or raw_dir.name
    root = output_dir / dataset_name
    camera_keys = parse_camera_keys(args.camera_keys)
    started_at = utc_now_iso()
    logging.info("Starting Unreal conversion raw_dir=%s output=%s dataset=%s cameras=%s", raw_dir, root, dataset_name, camera_keys)

    collection = UnrealEpisodeCollection(
        raw_dir=raw_dir,
        camera_keys=camera_keys,
        get_task_idx=lambda _task: 0,
        translation_tolerance_m=args.extrinsic_tolerance_translation_m,
        rotation_tolerance_deg=args.extrinsic_tolerance_rotation_deg,
        skip_invalid_episodes=args.skip_invalid_episodes,
        keep_all_schemas=args.split_by_schema,
        trim_extra_tail_frame=args.trim_extra_tail_frame,
    )

    if not args.split_by_schema:
        run_conversion(collection, root, dataset_name, args, started_at)
        return

    if not collection.schema_groups:
        completed_at = utc_now_iso()
        report = collection.build_report(root, started_at, completed_at, "no_valid_episodes")
        write_conversion_report(root, report)
        raise ValueError(f"No schema-compatible Unreal episodes found under {raw_dir}")

    logging.info("Splitting by schema into %d LeRobot datasets", len(collection.schema_groups))
    group_reports: list[dict[str, Any]] = []
    group_errors: list[dict[str, str]] = []
    for group in sorted(collection.schema_groups.values(), key=lambda item: item["schema_key"]):
        schema = (int(group["fps"]), tuple(group["image_size"]))
        group_dataset_name = f"{dataset_name}_{group['schema_key']}"
        group_root = output_dir / group_dataset_name
        logging.info(
            "Converting schema %s -> %s (%d episodes)",
            group["schema_key"],
            group_root,
            group["num_episodes"],
        )
        group_collection = collection.for_schema(schema)
        try:
            group_reports.append(run_conversion(group_collection, group_root, group_dataset_name, args, started_at))
        except Exception as exc:
            logging.exception("Schema conversion failed for %s", group["schema_key"])
            group_errors.append({"schema_key": group["schema_key"], "error": str(exc)})

    completed_at = utc_now_iso()
    top_report = {
        "status": "completed" if not group_errors else "completed_with_errors",
        "started_at": started_at,
        "completed_at": completed_at,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "dataset_name": dataset_name,
        "camera_keys": camera_keys,
        "schema_groups": list(collection.schema_groups.values()),
        "group_reports": group_reports,
        "group_errors": group_errors,
        "scan_failures": collection.failed_episodes,
        "repaired_episodes": collection.repaired_episodes,
    }
    top_report_path = output_dir / f"{dataset_name}_conversion_report.json"
    write_json(top_report_path, top_report)
    logging.info("Wrote split conversion report: %s", top_report_path)

    if group_errors and not group_reports:
        raise RuntimeError(f"All schema conversions failed; see {top_report_path}")


if __name__ == "__main__":
    main()
