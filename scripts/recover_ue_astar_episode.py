"""Recover one interrupted Unreal A* episode in an otherwise complete dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from utils.lerobot.video_utils import encode_video_frames

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class RecoveryAudit:
    episode_index: int
    frame_count: int
    task_index: int
    task: str
    cameras: tuple[str, ...]
    parquet_path: Path
    temp_frames: dict[str, tuple[Path, ...]]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def _indices(rows: list[dict], key: str, path: Path) -> set[int]:
    values = [int(row[key]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {key} values in {path}")
    return set(values)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stage_json(path: Path, value: dict) -> Path:
    staged = path.with_name(f".{path.name}.recovery")
    staged.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return staged


def _stage_jsonl(path: Path, rows: list[dict]) -> Path:
    staged = path.with_name(f".{path.name}.recovery")
    staged.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in rows),
        encoding="utf-8",
    )
    return staged


def audit_incomplete_episode(dataset_root: str | Path, episode_index: int) -> RecoveryAudit:
    root = Path(dataset_root)
    meta = root / "meta"
    info = _read_json(meta / "info.json")
    total_episodes = int(info["total_episodes"])
    if not 0 <= episode_index < total_episodes:
        raise ValueError(f"episode index {episode_index} is outside [0, {total_episodes})")

    expected = set(range(total_episodes))
    for name in ("episodes.jsonl", "episodes_extras.jsonl", "episodes_stats.jsonl"):
        path = meta / name
        missing = expected - _indices(_read_jsonl(path), "episode_index", path)
        if missing != {episode_index}:
            raise ValueError(f"expected only episode {episode_index} to be missing from {path}, got {sorted(missing)}")

    tasks_path = meta / "tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    if len(task_by_index) != len(tasks):
        raise ValueError(f"duplicate task_index values in {tasks_path}")

    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(parquet_path, columns=["episode_index", "task_index", "frame_index"])
    frame_count = table.num_rows
    if frame_count <= 0:
        raise ValueError(f"empty parquet: {parquet_path}")
    episode_values = set(table.column("episode_index").to_pylist())
    task_values = set(table.column("task_index").to_pylist())
    frame_values = table.column("frame_index").to_pylist()
    if episode_values != {episode_index}:
        raise ValueError(f"parquet episode_index mismatch: {sorted(episode_values)}")
    if len(task_values) != 1:
        raise ValueError(f"expected one task_index in {parquet_path}, got {sorted(task_values)}")
    task_index = int(next(iter(task_values)))
    if task_index not in task_by_index:
        raise ValueError(f"task_index {task_index} is absent from {tasks_path}")
    if frame_values != list(range(frame_count)):
        raise ValueError(f"non-contiguous frame_index values in {parquet_path}")

    cameras = tuple(
        key.removeprefix("video.")
        for key, spec in info["features"].items()
        if key.startswith("video.") and spec.get("dtype") == "video"
    )
    if not cameras:
        raise ValueError("dataset has no video cameras")
    temp_frames: dict[str, tuple[Path, ...]] = {}
    for camera in cameras:
        temp = root / "videos" / f"chunk-{chunk:03d}" / f"video.{camera}" / f"episode_{episode_index:06d}_temp"
        frames = tuple(sorted(temp.glob("frame_*.png")))
        expected_names = [f"frame_{index:06d}.png" for index in range(frame_count)]
        if [path.name for path in frames] != expected_names:
            raise ValueError(f"incomplete temporary frames for video.{camera}: {len(frames)} / {frame_count}")
        temp_frames[camera] = frames

    return RecoveryAudit(
        episode_index=episode_index,
        frame_count=frame_count,
        task_index=task_index,
        task=task_by_index[task_index],
        cameras=cameras,
        parquet_path=parquet_path,
        temp_frames=temp_frames,
    )


def encode_missing_videos(dataset_root: str | Path, audit: RecoveryAudit) -> tuple[Path, ...]:
    root = Path(dataset_root)
    info = _read_json(root / "meta" / "info.json")
    fps = int(info["fps"])
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = audit.episode_index // chunk_size
    outputs: list[Path] = []

    for camera in audit.cameras:
        logging.info("Encoding recovery video for video.%s", camera)
        feature = info["features"][f"video.{camera}"]
        video_info = feature.get("info") or {}
        codec = str(video_info.get("video.codec", "h264"))
        pix_fmt = str(video_info.get("video.pix_fmt", "yuv420p"))
        output_dir = root / "videos" / f"chunk-{chunk:03d}" / f"video.{camera}"
        output = output_dir / f"episode_{audit.episode_index:06d}.mp4"
        staged = output_dir / f"episode_{audit.episode_index:06d}.recovery.mp4"
        staged.unlink(missing_ok=True)
        encode_video_frames(
            imgs_dir=audit.temp_frames[camera][0].parent,
            video_path=staged,
            fps=fps,
            vcodec=codec,
            pix_fmt=pix_fmt,
            overwrite=True,
        )
        with av.open(str(staged), "r") as container:
            frame_count = sum(1 for _ in container.decode(video=0))
        if frame_count != audit.frame_count:
            staged.unlink(missing_ok=True)
            raise ValueError(
                f"encoded frame count mismatch for video.{camera}: "
                f"{frame_count} != {audit.frame_count}"
            )
        os.replace(staged, output)
        outputs.append(output)

    return tuple(outputs)


def compute_recovery_stats(dataset_root: str | Path, audit: RecoveryAudit) -> dict[str, dict[str, list]]:
    root = Path(dataset_root)
    info = _read_json(root / "meta" / "info.json")
    parquet_schema = pq.read_schema(audit.parquet_path)
    numeric_keys = [
        key
        for key, spec in info["features"].items()
        if key != "timestamp"
        and spec.get("dtype") in {"float32", "float64"}
        and key in parquet_schema.names
    ]
    table = pq.read_table(audit.parquet_path, columns=numeric_keys)
    stats: dict[str, dict[str, list]] = {}

    for key in numeric_keys:
        dtype = np.dtype(info["features"][key]["dtype"])
        values = np.asarray(table.column(key).to_pylist(), dtype=dtype)
        if values.ndim == 1:
            values = values[:, None]
        stats[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [int(values.shape[0])],
        }

    for camera in audit.cameras:
        logging.info("Computing streaming recovery stats for video.%s", camera)
        minimum = np.full(3, np.inf, dtype=np.float64)
        maximum = np.full(3, -np.inf, dtype=np.float64)
        total = np.zeros(3, dtype=np.float64)
        square_total = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        image_shape: tuple[int, int, int] | None = None
        for path in audit.temp_frames[camera]:
            with Image.open(path) as image:
                values = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
            if image_shape is None:
                image_shape = values.shape
            elif values.shape != image_shape:
                raise ValueError(f"inconsistent temporary image shape for video.{camera}: {path}")
            minimum = np.minimum(minimum, values.min(axis=(0, 1)))
            maximum = np.maximum(maximum, values.max(axis=(0, 1)))
            total += values.sum(axis=(0, 1))
            square_total += np.square(values).sum(axis=(0, 1))
            pixel_count += values.shape[0] * values.shape[1]
        mean = total / pixel_count
        variance = np.maximum(square_total / pixel_count - np.square(mean), 0.0)
        stats[f"video.{camera}"] = {
            "min": minimum.tolist(),
            "max": maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [audit.frame_count],
        }

    return stats


def repair_metadata(
    dataset_root: str | Path,
    audit: RecoveryAudit,
    *,
    extras: dict,
    stats: dict,
) -> None:
    root = Path(dataset_root)
    meta = root / "meta"
    info_path = meta / "info.json"
    info = _read_json(info_path)

    paths = {
        "episodes": meta / "episodes.jsonl",
        "extras": meta / "episodes_extras.jsonl",
        "stats": meta / "episodes_stats.jsonl",
        "tasks": meta / "tasks.jsonl",
    }
    episodes = _read_jsonl(paths["episodes"])
    extras_rows = _read_jsonl(paths["extras"])
    stats_rows = _read_jsonl(paths["stats"])
    tasks = _read_jsonl(paths["tasks"])

    for name, rows in (("episodes", episodes), ("extras", extras_rows), ("stats", stats_rows)):
        indices = _indices(rows, "episode_index", paths[name])
        if audit.episode_index in indices:
            raise ValueError(f"episode {audit.episode_index} already exists in {paths[name]}")

    episodes.append(
        {
            "episode_index": audit.episode_index,
            "tasks": [audit.task],
            "length": audit.frame_count,
        }
    )
    extras_rows.append({**extras, "episode_index": audit.episode_index})
    stats_rows.append({"episode_index": audit.episode_index, "stats": stats})
    for rows in (episodes, extras_rows, stats_rows):
        rows.sort(key=lambda row: int(row["episode_index"]))

    expected = list(range(len(episodes)))
    if [int(row["episode_index"]) for row in episodes] != expected:
        raise ValueError("repaired episode indices are not contiguous")
    if [int(row["episode_index"]) for row in extras_rows] != expected:
        raise ValueError("repaired extras indices are not contiguous")
    if [int(row["episode_index"]) for row in stats_rows] != expected:
        raise ValueError("repaired stats indices are not contiguous")

    task_indices = _indices(tasks, "task_index", paths["tasks"])
    if audit.task_index not in task_indices:
        raise ValueError(f"task {audit.task_index} is absent from {paths['tasks']}")
    chunks_size = int(info.get("chunks_size", 1000))
    total_episodes = len(episodes)
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": sum(int(row["length"]) for row in episodes),
            "total_tasks": len(tasks),
            "total_videos": total_episodes * len(audit.cameras),
            "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
            "splits": {"train": f"0:{total_episodes}"},
        }
    )

    staged = {
        paths["episodes"]: _stage_jsonl(paths["episodes"], episodes),
        paths["extras"]: _stage_jsonl(paths["extras"], extras_rows),
        paths["stats"]: _stage_jsonl(paths["stats"], stats_rows),
        info_path: _stage_json(info_path, info),
    }
    for path in (paths["episodes"], paths["extras"], paths["stats"], info_path):
        os.replace(staged[path], path)


def load_recovery_source(source_episode: str | Path, audit: RecoveryAudit):
    from ue_astar import AStarEpisodeCollection

    def get_task_index(task: str) -> int:
        if task != audit.task:
            raise ValueError("source VLN instruction does not match task metadata")
        return audit.task_index

    collection = AStarEpisodeCollection(
        raw_dir=source_episode,
        camera_keys=list(audit.cameras),
        get_task_idx=get_task_index,
        translation_tolerance_m=1e-4,
        rotation_tolerance_deg=0.1,
        skip_invalid_episodes=False,
        trim_extra_tail_frame=True,
        instruction_type="vln",
    )
    episodes = list(collection)
    if len(episodes) != 1:
        raise ValueError(f"expected exactly one source episode, got {len(episodes)}")
    episode = episodes[0]
    if len(episode) != audit.frame_count:
        raise ValueError(f"source frame count mismatch: {len(episode)} != {audit.frame_count}")
    if episode.task != audit.task or episode.task_idx != audit.task_index:
        raise ValueError("source task does not match the incomplete parquet")
    return episode


def verify_recovered_dataset(dataset_root: str | Path, episode_index: int) -> dict[str, Any]:
    from unreal import validate_lerobot_dataset

    root = Path(dataset_root)
    meta = root / "meta"
    info = _read_json(meta / "info.json")
    total_episodes = int(info["total_episodes"])
    expected = set(range(total_episodes))
    rows_by_name = {
        name: _read_jsonl(meta / name)
        for name in ("episodes.jsonl", "episodes_extras.jsonl", "episodes_stats.jsonl")
    }
    for name, rows in rows_by_name.items():
        indices = _indices(rows, "episode_index", meta / name)
        if indices != expected or len(rows) != total_episodes:
            raise ValueError(f"incomplete recovered metadata: {meta / name}")

    episode_row = next(
        row for row in rows_by_name["episodes.jsonl"] if int(row["episode_index"]) == episode_index
    )
    frame_count = int(episode_row["length"])
    chunks_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunks_size
    decoded_frames: dict[str, int] = {}
    camera_keys = [
        key
        for key, spec in info["features"].items()
        if key.startswith("video.") and spec.get("dtype") == "video"
    ]
    for key in camera_keys:
        video = root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
        with av.open(str(video), "r") as container:
            decoded_frames[key] = sum(1 for _ in container.decode(video=0))
        if decoded_frames[key] != frame_count:
            raise ValueError(f"decoded frame count mismatch for {video}: {decoded_frames[key]} != {frame_count}")

    validate_lerobot_dataset(repo_id=root.name, root=root)
    return {
        "total_episodes": total_episodes,
        "total_frames": int(info["total_frames"]),
        "total_videos": int(info["total_videos"]),
        "episode_index": episode_index,
        "episode_frames": frame_count,
        "decoded_frames": decoded_frames,
    }


def recover_episode(
    dataset_root: str | Path,
    source_episode: str | Path,
    episode_index: int,
    *,
    commit: bool,
) -> dict[str, Any]:
    from ue_astar import write_astar_dataset_sidecars

    root = Path(dataset_root)
    audit = audit_incomplete_episode(root, episode_index)
    episode = load_recovery_source(source_episode, audit)
    report: dict[str, Any] = {
        "status": "dry_run" if not commit else "in_progress",
        "dataset_root": str(root),
        "episode_index": audit.episode_index,
        "frame_count": audit.frame_count,
        "task_index": audit.task_index,
        "source_episode_path": str(episode.episode_dir),
        "cameras": list(audit.cameras),
        "temporary_frames_preserved": True,
    }
    if not commit:
        return report

    stats = compute_recovery_stats(root, audit)
    videos = encode_missing_videos(root, audit)
    repair_metadata(root, audit, extras=episode.metadata, stats=stats)
    report["videos"] = [str(path) for path in videos]
    report["sidecars"] = write_astar_dataset_sidecars(root, list(audit.cameras), include_depth=False)
    if report["sidecars"]["map_sidecars"]["missing"]:
        raise ValueError("map sidecar recovery completed with missing source files")
    report["verification"] = verify_recovered_dataset(root, episode_index)
    report["status"] = "completed"
    report_path = root / "meta" / f"ue_astar_episode_{episode_index:06d}_recovery.json"
    os.replace(_stage_json(report_path, report), report_path)
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-episode", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Write videos, metadata, and A* sidecars. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    report = recover_episode(
        args.dataset_root,
        args.source_episode,
        args.episode_index,
        commit=args.commit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
