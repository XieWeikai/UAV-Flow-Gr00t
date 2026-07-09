from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


RGB_VIEW_MAP = {
    "front": "front",
    "left": "left",
    "right": "right",
    "back": "rear",
}

MAP_ASSET_KEYS = (
    "traversibility",
    "traversibility_overlay",
    "floorplan",
    "floorplan_overlay",
    "floorplan_detail",
    "floorplan_detail_overlay",
)

POSE_AXES = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
DEFAULT_TASK = (
    "Navigate through the indoor scene using the current multi-view RGB observations "
    "and the provided floor maps."
)
PARQUET_COLUMNS = [
    "annotation.human.action.task_description",
    "observation.state",
    "action",
    "frame_index",
    "timestamp",
    "index",
    "episode_index",
    "task_index",
    "extra.world_position",
    "extra.world_rotation_xyzw",
    "extra.floorplan_xy",
    "extra.discrete_action_to_next_id",
    "extra.cot",
]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: int


def convert_dataset(
    input_root: str | Path,
    output_root: str | Path,
    dataset_name: str,
    split: str,
    max_episodes: int | None = None,
    chunk_size: int = 1000,
    overwrite: bool = False,
) -> Path:
    input_root = Path(input_root)
    output_root = Path(output_root)
    split_root = input_root / split
    episodes_root = split_root / "episodes" / split
    if not episodes_root.exists():
        raise FileNotFoundError(f"Replay episodes directory not found: {episodes_root}")

    dataset_root = output_root / split
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output split already exists: {dataset_root}")
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    source_episodes = sorted(p for p in episodes_root.iterdir() if p.is_dir())
    if max_episodes is not None:
        source_episodes = source_episodes[:max_episodes]
    if not source_episodes:
        raise ValueError(f"No replay episodes found in {episodes_root}")

    first_video = source_episodes[0] / "front.mp4"
    video_info = _read_video_info(first_video)
    features = _build_features(video_info)
    task = DEFAULT_TASK
    task_index = 0

    episode_rows: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    episode_extras: list[dict[str, Any]] = []
    total_frames = 0

    iterator = tqdm(source_episodes, desc=f"Converting {dataset_name}/{split}", unit="episode")
    for episode_index, source_dir in enumerate(iterator):
        result = _convert_episode(
            split_root=split_root,
            source_dir=source_dir,
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            split=split,
            episode_index=episode_index,
            task=task,
            task_index=task_index,
            global_frame_start=total_frames,
            chunk_size=chunk_size,
            video_info=video_info,
        )
        episode_rows.append({"episode_index": episode_index, "tasks": [task], "length": result["length"]})
        episode_stats.append({"episode_index": episode_index, "stats": result["stats"]})
        episode_extras.append(result["extras"])
        total_frames += result["length"]

    _write_metadata(
        dataset_root=dataset_root,
        features=features,
        fps=video_info.fps,
        chunk_size=chunk_size,
        total_episodes=len(source_episodes),
        total_frames=total_frames,
        total_videos=len(source_episodes) * len(RGB_VIEW_MAP),
        task=task,
        task_index=task_index,
        episodes=episode_rows,
        episode_stats=episode_stats,
        episode_extras=episode_extras,
    )
    return dataset_root


def _convert_episode(
    *,
    split_root: Path,
    source_dir: Path,
    dataset_root: Path,
    dataset_name: str,
    split: str,
    episode_index: int,
    task: str,
    task_index: int,
    global_frame_start: int,
    chunk_size: int,
    video_info: VideoInfo,
) -> dict[str, Any]:
    episode = _read_json(source_dir / "episode.json")
    steps = _read_jsonl(source_dir / "steps.jsonl")
    if not steps:
        raise ValueError(f"Empty steps.jsonl: {source_dir}")

    chunk = episode_index // chunk_size
    episode_file = f"episode_{episode_index:06d}"
    data_dir = dataset_root / "data" / f"chunk-{chunk:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)

    _copy_rgb_videos(source_dir, dataset_root, chunk, episode_file)
    map_result = _write_map_assets(split_root, source_dir, dataset_root, chunk, episode_file, episode, steps)

    rows, stats_inputs = _build_episode_rows(
        steps=steps,
        episode_index=episode_index,
        task_index=task_index,
        task=task,
        fps=video_info.fps,
        global_frame_start=global_frame_start,
    )
    pd.DataFrame(rows, columns=PARQUET_COLUMNS).to_parquet(data_dir / f"{episode_file}.parquet", index=False)

    extras = {
        "episode_index": episode_index,
        "dataset_name": dataset_name,
        "split": split,
        "trajectory_id": str(episode.get("trajectory_id", "")),
        "scene_key": str(episode.get("scene_key", "")),
        "source_episode_ids": [str(x) for x in episode.get("episode_ids", [])],
        "source_episode_dir": str(source_dir.relative_to(split_root)),
        "instructions": _extract_instructions(episode),
        "map_size": [map_result["height"], map_result["width"]],
        "map_scale": map_result["scale"],
        "map_assets": map_result["assets"],
    }
    return {
        "length": len(rows),
        "stats": _episode_stats(stats_inputs),
        "extras": extras,
    }


def _build_episode_rows(
    *,
    steps: list[dict[str, Any]],
    episode_index: int,
    task_index: int,
    task: str,
    fps: int,
    global_frame_start: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    positions = np.asarray([step["position"] for step in steps], dtype=np.float64)
    rotations = np.asarray([step["rotation"] for step in steps], dtype=np.float64)
    states = _relative_poses(positions, rotations)

    world_positions = positions.astype(np.float32)
    world_rotations = np.asarray([_normalize_quat(q) for q in rotations], dtype=np.float32)
    floorplan_xy = []
    discrete_ids = []

    rows = []
    for idx, step in enumerate(steps):
        _assert_aligned_xy(step, idx)
        floorplan_xy_value = [int(step["floorplan_xy"][0]), int(step["floorplan_xy"][1])]
        floorplan_xy.append(floorplan_xy_value)
        discrete_id = int(step.get("discrete_action_to_next_id", 0))
        discrete_ids.append(discrete_id)

        rows.append(
            {
                "annotation.human.action.task_description": [int(task_index)],
                "observation.state": states[idx].astype(np.float32).tolist(),
                "action": states[idx].astype(np.float32).tolist(),
                "frame_index": int(idx),
                "timestamp": float(idx / fps),
                "index": int(global_frame_start + idx),
                "episode_index": int(episode_index),
                "task_index": int(task_index),
                "extra.world_position": world_positions[idx].tolist(),
                "extra.world_rotation_xyzw": world_rotations[idx].tolist(),
                "extra.floorplan_xy": floorplan_xy_value,
                "extra.discrete_action_to_next_id": discrete_id,
                "extra.cot": "",
            }
        )

    return rows, {
        "observation.state": states.astype(np.float32),
        "action": states.astype(np.float32),
        "extra.world_position": world_positions,
        "extra.world_rotation_xyzw": world_rotations,
        "extra.floorplan_xy": np.asarray(floorplan_xy, dtype=np.int64),
        "extra.discrete_action_to_next_id": np.asarray(discrete_ids, dtype=np.int64).reshape(-1, 1),
    }


def _write_map_assets(
    split_root: Path,
    source_dir: Path,
    dataset_root: Path,
    chunk: int,
    episode_file: str,
    episode: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = _map_sources(split_root, episode)
    output_rel_paths = {
        key: Path("maps") / f"chunk-{chunk:03d}" / key / f"{episode_file}.png"
        for key in MAP_ASSET_KEYS
    }

    with Image.open(sources["floorplan"]) as base_image:
        src_width, src_height = base_image.size

    for key, source_path in sources.items():
        with Image.open(source_path) as image:
            if image.size != (src_width, src_height):
                raise ValueError(
                    f"Map asset size mismatch for {source_dir}: {key} has {image.size}, "
                    f"expected {(src_width, src_height)}"
                )
        output_path = dataset_root / output_rel_paths[key]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)

    for idx, step in enumerate(steps):
        _assert_aligned_xy(step, idx)

    return {
        "width": src_width,
        "height": src_height,
        "scale": 1.0,
        "assets": {key: str(path) for key, path in output_rel_paths.items()},
    }


def _map_sources(split_root: Path, episode: dict[str, Any]) -> dict[str, Path]:
    scene_maps = episode["scene_map_paths"]
    graph_rel = scene_maps["graph_floor"]["graph"]

    # floor_level_id is per-step in current replay output, so infer from overlay level name if
    # episode-level metadata does not contain it.
    overlay_paths = [str(p) for p in episode.get("overlay_paths", [])]
    layout_overlay = _select_overlay(overlay_paths, "layout")
    detail_overlay = _select_overlay(overlay_paths, "detail")
    graph_overlay = _select_overlay(overlay_paths, "graph")
    level = _level_from_overlay(layout_overlay)
    levels = scene_maps["levels"]
    if level not in levels:
        raise KeyError(f"Floor level {level!r} not found in scene_map_paths levels")

    rel_sources = {
        "traversibility": graph_rel,
        "traversibility_overlay": graph_overlay,
        "floorplan": levels[level]["layout"],
        "floorplan_overlay": layout_overlay,
        "floorplan_detail": levels[level]["detail"],
        "floorplan_detail_overlay": detail_overlay,
    }
    return {key: split_root / value for key, value in rel_sources.items()}


def _select_overlay(paths: list[str], token: str) -> str:
    matches = [p for p in paths if token in Path(p).name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one overlay containing {token!r}, found {matches}")
    return matches[0]


def _level_from_overlay(path: str) -> str:
    stem = Path(path).stem
    marker = "level_"
    if marker not in stem:
        raise ValueError(f"Cannot infer floor level from overlay path: {path}")
    return stem.split(marker, 1)[1]


def _copy_rgb_videos(source_dir: Path, dataset_root: Path, chunk: int, episode_file: str) -> None:
    for source_view, output_view in RGB_VIEW_MAP.items():
        source_path = source_dir / f"{source_view}.mp4"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing replay RGB video: {source_path}")
        output_path = dataset_root / "videos" / f"chunk-{chunk:03d}" / f"video.{output_view}" / f"{episode_file}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)


def _read_video_info(path: Path) -> VideoInfo:
    if not path.exists():
        raise FileNotFoundError(f"Missing video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 10
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions for {path}: {width}x{height}")
    return VideoInfo(width=width, height=height, fps=fps)


def _relative_poses(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    q0_inv = _quat_conjugate(_normalize_quat(rotations[0]))
    p0 = positions[0]
    states = []
    for idx, (position, quat) in enumerate(zip(positions, rotations)):
        rel_pos = _quat_rotate(q0_inv, position - p0)
        rel_quat = _quat_multiply(q0_inv, _normalize_quat(quat))
        rel_quat = _normalize_quat(rel_quat)
        if rel_quat[3] < 0:
            rel_quat = -rel_quat
        if idx == 0:
            rel_pos = np.zeros(3, dtype=np.float64)
            rel_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        states.append(np.concatenate([rel_pos, rel_quat]))
    return np.asarray(states, dtype=np.float32)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm quaternion")
    return q / norm


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    v_quat = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return _quat_multiply(_quat_multiply(q, v_quat), _quat_conjugate(q))[:3]


def _assert_aligned_xy(step: dict[str, Any], idx: int) -> None:
    map_xy = step.get("map_xy")
    graph_xy = step.get("graph_xy")
    floorplan_xy = step.get("floorplan_xy")
    if map_xy != graph_xy or map_xy != floorplan_xy:
        raise ValueError(
            f"Unaligned xy at step {idx}: map_xy={map_xy}, graph_xy={graph_xy}, "
            f"floorplan_xy={floorplan_xy}"
        )


def _episode_stats(values: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    stats = {}
    for key, array in values.items():
        arr = np.asarray(array)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        stats[key] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def _build_features(video_info: VideoInfo) -> dict[str, Any]:
    video_feature = {
        "dtype": "video",
        "shape": [video_info.height, video_info.width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": video_info.height,
            "video.width": video_info.width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": video_info.fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    features: dict[str, Any] = {
        "annotation.human.action.task_description": {"dtype": "int32", "shape": [1], "names": None},
        "observation.state": {"dtype": "float32", "shape": [7], "names": {"axes": POSE_AXES}},
        "video.front": video_feature,
        "video.left": video_feature,
        "video.right": video_feature,
        "video.rear": video_feature,
        "action": {"dtype": "float32", "shape": [7], "names": {"axes": POSE_AXES}},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "extra.world_position": {"dtype": "float32", "shape": [3], "names": {"axes": ["x", "y", "z"]}},
        "extra.world_rotation_xyzw": {"dtype": "float32", "shape": [4], "names": {"axes": ["qx", "qy", "qz", "qw"]}},
        "extra.floorplan_xy": {"dtype": "int64", "shape": [2], "names": {"axes": ["x", "y"]}},
        "extra.discrete_action_to_next_id": {"dtype": "int64", "shape": [1], "names": None},
        "extra.cot": {"dtype": "string", "shape": [1], "names": None},
    }
    return features


def _write_metadata(
    *,
    dataset_root: Path,
    features: dict[str, Any],
    fps: int,
    chunk_size: int,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    task: str,
    task_index: int,
    episodes: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    episode_extras: list[dict[str, Any]],
) -> None:
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = math.ceil(total_episodes / chunk_size)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "xnav_indoor",
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": chunk_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "splits": {"train": f"0:{total_episodes}"},
    }
    _write_json(meta_dir / "info.json", info)
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": task_index, "task": task}])
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)
    _write_jsonl(meta_dir / "episodes_extras.jsonl", episode_extras)
    _write_json(meta_dir / "modality.json", _build_modality())


def _build_modality() -> dict[str, Any]:
    return {
        "state": {
            "drone": {
                "start": 0,
                "end": 7,
                "absolute": False,
                "rotation_type": "quaternion",
                "original_key": "observation.state",
            }
        },
        "action": {
            "pose": {
                "start": 0,
                "end": 7,
                "absolute": False,
                "rotation_type": "quaternion",
                "original_key": "action",
            }
        },
        "video": {
            "front": {"original_key": "video.front"},
            "left": {"original_key": "video.left"},
            "right": {"original_key": "video.right"},
            "rear": {"original_key": "video.rear"},
        },
        "annotation": {"human.action.task_description": {}},
    }


def _extract_instructions(episode: dict[str, Any]) -> list[str]:
    instructions = []
    for item in episode.get("instructions", []):
        if isinstance(item, str):
            instructions.append(item)
        elif isinstance(item, dict):
            text = item.get("instruction") or item.get("text") or item.get("instruction_text")
            if text is not None:
                instructions.append(str(text))
    return instructions


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
