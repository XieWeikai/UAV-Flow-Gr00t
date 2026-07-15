from __future__ import annotations

"""Unreal A* episode -> LeRobot v2.1 conversion entry.

Typical input root is a worker collection data directory such as:

    /mnt/glx/pointnav_collect/worker_01/data

Passing the worker root remains supported for compatibility; in that case the
converter scans only its `data/` child. It writes one standard LeRobot dataset
at `--output_dir`. It keeps the standard Unreal training fields
(`observation.state`, `action`, RGB videos and episode extras) and adds generic
map-grounding fields:

    observation.map.grid_cell   # [x, y, layer], int32, invalid [-1, -1, -1]
    observation.map.pixel_4096  # [x, y], int32, invalid [-1, -1]

`--instruction_type` is required. `pointnav` preserves the task_info.csv task,
`vln` keeps only episodes with a non-empty VLN instruction, and `objectnav`
keeps only episodes with a non-empty ObjectNav instruction. ObjectNav tasks are
JSON strings containing `task` and `target_category`; VLN tasks are plain text.

Map assets are copied as dataset sidecars:

    maps/chunk-000/planned_path_map/episode_000000.png
    maps/chunk-000/actual_path_map/episode_000000.png
    maps/chunk-000/path_comparison_map/episode_000000.png
    maps/chunk-000/astar_plan/episode_000000.json

Depth PNG sidecars are optional and disabled by default. Pass `--include_depth`
to copy them under `images/chunk-000/observation.depth.<camera>/...`.
"""

import argparse
import json
import logging
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from unreal import (
    ACTION_KEY,
    DEFAULT_CAMERA_KEYS,
    STATE_KEY,
    TASK_DESCRIPTION_KEY,
    CameraImageSource,
    UnrealEpisode,
    UnrealEpisodeCollection,
    build_features,
    group_episodes_by_schema,
    homogeneous_inv,
    infer_source_ids,
    intrinsic_4,
    intrinsic_matrix,
    json_default,
    load_json,
    load_jsonl,
    load_jsonl_dicts,
    load_task_info,
    parse_camera_keys,
    schema_suffix,
    select_video_pixel_format,
    transform_to_pose_vector,
    unreal_pose_to_target_transform,
    utc_now_iso,
    validate_fixed_extrinsics,
    validate_lerobot_dataset,
    write_json,
)

GRID_CELL_KEY = "observation.map.grid_cell"
MAP_PIXEL_4096_KEY = "observation.map.pixel_4096"
MAP_LONG_EDGE_PX = 4096
INVALID_GRID_CELL = np.array([-1, -1, -1], dtype=np.int32)
INVALID_PIXEL = np.array([-1, -1], dtype=np.int32)

MAP_SIDECARS = {
    "planned_path_map": "planned_path_map.png",
    "actual_path_map": "actual_path_map.png",
    "path_comparison_map": "path_comparison_map.png",
    "astar_plan": "astar_plan_path.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Unreal A* collection episodes to LeRobot v2.1 format.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Collection data directory (for example .../worker_01/data), worker root, scene/run directory, or one episode_* directory.",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Directory used to store the exported LeRobot dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset/repo name used for validation metadata. Defaults to output_dir name.",
    )
    parser.add_argument(
        "--instruction_type",
        type=str,
        required=True,
        choices=["pointnav", "vln", "objectnav"],
        help="Task source: legacy task_info.csv, VLN instruction, or ObjectNav instruction.",
    )
    parser.add_argument("--camera_keys", type=str, default=",".join(DEFAULT_CAMERA_KEYS), help="Comma-separated cameras to export.")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of writer worker processes.")
    parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec.")
    parser.add_argument("--pix_fmt", type=str, default="auto", choices=["auto", "yuv420p", "yuv444p"], help="Video pixel format.")
    parser.add_argument("--extrinsic_tolerance_translation_m", type=float, default=1e-4)
    parser.add_argument("--extrinsic_tolerance_rotation_deg", type=float, default=0.1)
    parser.add_argument(
        "--skip_invalid_episodes",
        action="store_true",
        help="Skip incompatible episodes and record them in meta/ue_astar_conversion_report.json.",
    )
    parser.add_argument(
        "--split_by_schema",
        action="store_true",
        help="Export one LeRobot dataset per fps/resolution schema. This never splits by scene.",
    )
    parser.add_argument(
        "--include_depth",
        action="store_true",
        help="Copy raw depth PNG sidecars into images/. Disabled by default.",
    )
    parser.add_argument(
        "--trim_extra_tail_frame",
        action="store_true",
        help="Trim one extra tail frame from frames.jsonl when it is exactly meta.frame_count + 1.",
    )
    return parser.parse_args()


def write_jsonl_dicts(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=json_default))
            file.write("\n")


def resolve_astar_episode_search_root(raw_dir: str | Path) -> Path:
    """Resolve the subtree that should be scanned for episode directories."""

    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {root}")
    if root.is_file():
        raise ValueError(f"raw_dir must be a directory: {root}")

    if (root / "episode_meta.json").exists():
        return root

    if root.name == "data":
        return root

    return root / "data" if (root / "data").is_dir() else root


def scan_astar_episode_dirs(raw_dir: str | Path) -> list[Path]:
    """Find episode directories.

    Preferred input is the collection `data/` directory. The worker root also
    remains accepted; because it contains traversability/task bookkeeping, only
    its `data/` child is scanned when present.
    """

    search_root = resolve_astar_episode_search_root(raw_dir)
    if (search_root / "episode_meta.json").exists():
        return [search_root]
    return sorted(path.parent for path in search_root.rglob("episode_meta.json"))


def _clean_instruction_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def load_navigation_instruction(episode_dir: Path) -> dict[str, Any] | None:
    """Load and normalize the optional navigation instruction annotation."""

    path = episode_dir / "instruction.json"
    if not path.exists():
        return None

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"instruction.json must contain an object: {path}")

    vln = payload.get("vln") if isinstance(payload.get("vln"), dict) else {}
    objectnav = payload.get("objectnav") if isinstance(payload.get("objectnav"), dict) else {}
    return {
        "has_quality_issue": payload.get("has_quality_issue") is True,
        "quality_reason": _clean_instruction_text(payload.get("quality_reason")),
        "vln_instruction": _clean_instruction_text(vln.get("instruction")),
        "objectnav_instruction": _clean_instruction_text(objectnav.get("instruction")),
        "objectnav_target_category": _clean_instruction_text(objectnav.get("target_category")),
    }


def build_instruction_task(annotation: dict[str, Any], instruction_type: str) -> str | None:
    """Build the LeRobot task string for one instruction mode."""

    if instruction_type == "vln":
        return annotation.get("vln_instruction") or None
    if instruction_type == "objectnav":
        instruction = annotation.get("objectnav_instruction") or ""
        if not instruction:
            return None
        return json.dumps(
            {
                "task": instruction,
                "target_category": annotation.get("objectnav_target_category") or "",
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    raise ValueError(f"Unsupported instruction_type for instruction task: {instruction_type}")


def build_astar_features(image_size: tuple[int, int], camera_keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    features = build_features(image_size, camera_keys)
    features[GRID_CELL_KEY] = {
        "dtype": "int32",
        "shape": (3,),
        "names": {"axes": ["x", "y", "layer"]},
    }
    features[MAP_PIXEL_4096_KEY] = {
        "dtype": "int32",
        "shape": (2,),
        "names": {"axes": ["x", "y"]},
    }
    return features


def graph_dimensions(graph: dict[str, Any]) -> tuple[int, int]:
    candidates = (
        ("width", "height"),
        ("grid_width", "grid_height"),
        ("cols", "rows"),
        ("num_cols", "num_rows"),
    )
    for width_key, height_key in candidates:
        if width_key in graph and height_key in graph:
            width = int(graph[width_key])
            height = int(graph[height_key])
            if width <= 0 or height <= 0:
                raise ValueError(f"graph dimensions must be positive, got width={width} height={height}")
            return width, height
    raise ValueError(f"graph is missing width/height fields: keys={sorted(graph)}")


def compute_map_size_4096(graph: dict[str, Any], long_edge_px: int = MAP_LONG_EDGE_PX) -> list[int]:
    width, height = graph_dimensions(graph)
    scale = float(long_edge_px) / float(max(width, height))
    return [int(round(width * scale)), int(round(height * scale))]


def normalize_grid_cell(value: Any) -> list[int] | None:
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            return None
        return [int(value["x"]), int(value["y"]), int(value.get("layer", 0))]

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) < 2:
            return None
        layer = parts[2] if len(parts) >= 3 else 0
        return [int(parts[0]), int(parts[1]), int(layer)]

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        layer = value[2] if len(value) >= 3 else 0
        return [int(value[0]), int(value[1]), int(layer)]

    return None


def compute_map_pixel_4096(
    grid_cell: dict[str, Any] | list[int] | tuple[int, ...] | str,
    graph: dict[str, Any],
    long_edge_px: int = MAP_LONG_EDGE_PX,
) -> tuple[list[int], list[int]]:
    cell = normalize_grid_cell(grid_cell)
    if cell is None:
        raise ValueError(f"Invalid grid cell: {grid_cell!r}")

    width, height = graph_dimensions(graph)
    x, y, _layer = cell
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError(f"grid cell out of graph bounds: cell={cell} graph_width={width} graph_height={height}")

    scale = float(long_edge_px) / float(max(width, height))
    map_size = [int(round(width * scale)), int(round(height * scale))]
    pixel = [int(round((x + 0.5) * scale)), int(round((y + 0.5) * scale))]
    return pixel, map_size


def build_map_frame_fields(frame: dict[str, Any], graph: dict[str, Any]) -> dict[str, np.ndarray]:
    raw_valid = frame.get("pointnav_grid_cell_valid", frame.get("grid_cell_valid", True))
    cell = frame.get("pointnav_grid_cell", frame.get("grid_cell"))
    if not raw_valid or cell in (None, ""):
        return {
            GRID_CELL_KEY: INVALID_GRID_CELL.copy(),
            MAP_PIXEL_4096_KEY: INVALID_PIXEL.copy(),
        }

    try:
        grid_cell = normalize_grid_cell(cell)
        if grid_cell is None:
            raise ValueError(f"invalid grid cell: {cell!r}")
        pixel, _size = compute_map_pixel_4096(grid_cell, graph)
    except Exception:
        return {
            GRID_CELL_KEY: INVALID_GRID_CELL.copy(),
            MAP_PIXEL_4096_KEY: INVALID_PIXEL.copy(),
        }

    return {
        GRID_CELL_KEY: np.asarray(grid_cell, dtype=np.int32),
        MAP_PIXEL_4096_KEY: np.asarray(pixel, dtype=np.int32),
    }


def infer_collection_root(episode_dir: Path) -> Path | None:
    for parent in episode_dir.parents:
        if parent.name == "data":
            return parent.parent
    return None


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def nested_task_payload(pointnav_payload: dict[str, Any]) -> dict[str, Any]:
    task = pointnav_payload.get("task")
    return task if isinstance(task, dict) else {}


def first_present(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return None


def extract_astar_value(context: dict[str, Any], key: str):
    astar = context.get("astar") or {}
    pointnav_task = nested_task_payload(context.get("pointnav") or {})
    return first_present(astar.get(key), pointnav_task.get(key), context.get(key))


def load_graph_from_traversability(episode_dir: Path, scene_id: str, graph_id: str | None) -> dict[str, Any]:
    collection_root = infer_collection_root(episode_dir)
    if collection_root is None or not graph_id:
        return {}

    search_root = collection_root / "traversability" / scene_id
    if not search_root.exists():
        return {}

    candidates = [
        search_root / graph_id / "graph.json",
        search_root / f"{graph_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_json(candidate)

    for candidate in search_root.rglob("*.json"):
        try:
            payload = load_json(candidate)
        except Exception:
            continue
        if payload.get("graph_id") == graph_id:
            return payload
    return {}


def normalize_astar_context(context: dict[str, Any]) -> dict[str, Any]:
    if "astar" in context or "pointnav" in context:
        astar = context.get("astar") or {}
        pointnav = context.get("pointnav") or {}
        graph = context.get("graph") or astar.get("graph") or {}
        return {"astar": astar, "pointnav": pointnav, "graph": graph}

    graph = context.get("graph") or {}
    return {"astar": context, "pointnav": {}, "graph": graph}


def load_astar_context(episode_dir: Path) -> dict[str, Any]:
    astar_path = episode_dir / "astar_plan_path.json"
    if not astar_path.exists():
        raise ValueError("missing astar_plan_path.json")

    missing_maps = [filename for filename in MAP_SIDECARS.values() if not (episode_dir / filename).exists()]
    if missing_maps:
        raise ValueError(f"missing A* sidecar files: {missing_maps}")

    pointnav = load_optional_json(episode_dir / "pointnav.json")
    astar = load_json(astar_path)
    scene_id, _run_id = infer_source_ids(episode_dir)
    graph_id = first_present(astar.get("graph_id"), nested_task_payload(pointnav).get("graph_id"))
    graph = astar.get("graph") or load_graph_from_traversability(episode_dir, scene_id, graph_id)
    if not graph:
        raise ValueError("missing graph metadata in astar_plan_path.json or traversability")
    graph_dimensions(graph)

    return {
        "pointnav": pointnav,
        "astar": astar,
        "graph": graph,
    }


def intrinsic_4_from_frame_or_meta(frame: dict[str, Any], meta: dict[str, Any], camera_key: str) -> list[float]:
    source = frame if f"K_{camera_key}" in frame else meta
    return intrinsic_4(source, camera_key)


def intrinsic_matrix_from_frame_or_meta(frame: dict[str, Any], meta: dict[str, Any], camera_key: str) -> list[list[float]]:
    source = frame if f"K_{camera_key}" in frame else meta
    return intrinsic_matrix(source, camera_key)


class AStarEpisode(UnrealEpisode):
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
        astar_context: dict[str, Any],
        instruction_type: str = "pointnav",
    ):
        self.episode_dir = episode_dir
        self.meta = meta
        self.frames = frames
        self.camera_keys = camera_keys
        self.task = task
        self.task_idx = task_idx
        self.task_info = task_info
        self.instruction_type = instruction_type
        self.body_from_camera = body_from_camera
        self.astar_context = normalize_astar_context(astar_context)
        self.graph = self.astar_context["graph"]
        graph_dimensions(self.graph)
        self.image_sources = {
            camera: CameraImageSource.from_episode(episode_dir, meta, camera, len(frames))
            for camera in camera_keys
        }

    @property
    def metadata(self) -> dict[str, Any]:
        scene_id, run_id = infer_source_ids(self.episode_dir)
        map_width, map_height = compute_map_size_4096(self.graph)
        metadata_task = self.task
        if self.instruction_type == "objectnav":
            metadata_task = _clean_instruction_text(json.loads(self.task).get("task"))
        metadata: dict[str, Any] = {
            "source_episode_path": str(self.episode_dir),
            "scene_id": scene_id,
            "user_id": run_id,
            "run_id": run_id,
            "original_episode_index": int(self.meta.get("episode_index", -1)),
            "map_name": self.meta.get("map_name", ""),
            "frame_count": len(self.frames),
            "fps": int(round(float(self.meta.get("sample_rate_hz", 0)))),
            "capture_width": int(self.meta.get("capture_width", 0)),
            "capture_height": int(self.meta.get("capture_height", 0)),
            "camera_keys": self.camera_keys,
            "task": metadata_task,
            "task_info": self.task_info,
            "task.task_uid": extract_astar_value(self.astar_context, "task_uid"),
            "task.path_uid": extract_astar_value(self.astar_context, "path_uid"),
            "task.graph_id": extract_astar_value(self.astar_context, "graph_id"),
            "task.start_cell": extract_astar_value(self.astar_context, "start_cell"),
            "task.goal_cell": extract_astar_value(self.astar_context, "goal_cell"),
            "astar.graph": self.graph,
            "astar.path_distance_cm": extract_astar_value(self.astar_context, "path_distance_cm"),
            "astar.straight_distance_cm": extract_astar_value(self.astar_context, "straight_distance_cm"),
            "astar.turn_count": extract_astar_value(self.astar_context, "turn_count"),
            "maps.long_edge_px": MAP_LONG_EDGE_PX,
            "maps.width_4096": map_width,
            "maps.height_4096": map_height,
        }
        for camera in self.camera_keys:
            video_key = f"video.{camera}"
            metadata[f"{video_key}.K"] = intrinsic_4_from_frame_or_meta(self.frames[0], self.meta, camera)
            metadata[f"{video_key}.body_from_camera"] = self.body_from_camera[camera]
            metadata[f"K_{camera}"] = intrinsic_matrix_from_frame_or_meta(self.frames[0], self.meta, camera)
            metadata[f"Extrinsic_{camera}"] = self.body_from_camera[camera]
        return metadata

    def __iter__(self):
        image_iters = {camera: self.image_sources[camera].iter_rgb() for camera in self.camera_keys}
        first_body_inv: np.ndarray | None = None

        for frame in self.frames:
            world_from_body = unreal_pose_to_target_transform(frame["pose"])
            if first_body_inv is None:
                first_body_inv = homogeneous_inv(world_from_body)
            local_pose = transform_to_pose_vector((first_body_inv @ world_from_body).astype(np.float32))

            item: dict[str, Any] = {
                TASK_DESCRIPTION_KEY: np.array([self.task_idx], dtype=np.int32),
                STATE_KEY: local_pose,
                ACTION_KEY: local_pose.copy(),
            }
            item.update(build_map_frame_fields(frame, self.graph))
            for camera in self.camera_keys:
                item[f"video.{camera}"] = next(image_iters[camera])
            yield item, self.task


class AStarEpisodeCollection(UnrealEpisodeCollection):
    ROBOT_TYPE = "go2"
    INSTRUCTION_KEY = TASK_DESCRIPTION_KEY

    def __init__(self, *args, **kwargs):
        self.instruction_type = kwargs.pop("instruction_type", "pointnav")
        if self.instruction_type not in {"pointnav", "vln", "objectnav"}:
            raise ValueError(f"Unsupported instruction_type: {self.instruction_type}")
        super().__init__(*args, **kwargs)
        if self.episodes:
            self.FEATURES = build_astar_features(self.image_size, self.camera_keys)

    def _record_instruction_exclusion(
        self,
        episode_dir: Path,
        reason: str,
        *,
        quality_reason: str = "",
    ):
        exclusion = {
            "source_episode_path": str(episode_dir),
            "stage": "instruction_filter",
            "reason": reason,
        }
        if quality_reason:
            exclusion["quality_reason"] = quality_reason
        self.excluded_episodes.append(exclusion)
        logging.info("Excluding A* episode during instruction filtering (%s): %s", reason, episode_dir)

    def build_report(self, root: Path, started_at: str, completed_at: str | None, status: str) -> dict[str, Any]:
        report = super().build_report(root, started_at, completed_at, status)
        report["instruction_type"] = self.instruction_type
        report["instruction_filter_counts"] = dict(
            sorted(
                Counter(
                    item["reason"]
                    for item in self.excluded_episodes
                    if item.get("stage") == "instruction_filter"
                ).items()
            )
        )
        return report

    def _load_episodes(self):
        loaded = []
        episode_dirs = scan_astar_episode_dirs(self.raw_dir)
        logging.info("Found %d episode_meta.json files under %s", len(episode_dirs), self.raw_dir)

        for index, episode_dir in enumerate(episode_dirs, start=1):
            logging.info("Scanning A* episode %d / %d: %s", index, len(episode_dirs), episode_dir)
            meta_path = episode_dir / "episode_meta.json"
            frames_path = episode_dir / "frames.jsonl"
            try:
                selected_task: str | None = None
                if self.instruction_type != "pointnav":
                    annotation = load_navigation_instruction(episode_dir)
                    if annotation is None:
                        self._record_instruction_exclusion(episode_dir, "missing_instruction_file")
                        continue
                    if annotation["has_quality_issue"]:
                        self._record_instruction_exclusion(
                            episode_dir,
                            "instruction_quality_issue",
                            quality_reason=annotation["quality_reason"],
                        )
                        continue
                    selected_task = build_instruction_task(annotation, self.instruction_type)
                    if selected_task is None:
                        self._record_instruction_exclusion(
                            episode_dir,
                            f"missing_{self.instruction_type}_instruction",
                        )
                        continue

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
                        if f"camera_pose_{camera}" not in frame:
                            raise ValueError(f"frame {frame.get('frame_index')} missing camera pose for {camera}")
                        if f"K_{camera}" not in frame and f"K_{camera}" not in meta:
                            raise ValueError(f"missing K_{camera} in frame {frame.get('frame_index')} and episode_meta.json")

                astar_context = load_astar_context(episode_dir)
                pointnav_task, task_info = load_task_info(episode_dir)
                task = pointnav_task if self.instruction_type == "pointnav" else selected_task
                logging.info("Validating fixed camera extrinsics for %s", episode_dir)
                body_from_camera = validate_fixed_extrinsics(
                    episode_dir,
                    frames,
                    self.camera_keys,
                    self.translation_tolerance_m,
                    self.rotation_tolerance_deg,
                )
                episode_context = {
                    "body_from_camera": body_from_camera,
                    "astar_context": astar_context,
                }
                loaded.append((episode_dir, meta, frames, task, task_info, episode_context))
                logging.info("Accepted A* episode %s with %d frames", episode_dir, len(frames))
            except Exception as exc:
                self._record_failure(episode_dir, "episode_scan", exc)

        return loaded

    def for_schema(self, schema: tuple[int, tuple[int, int]]) -> "AStarEpisodeCollection":
        return AStarEpisodeCollection(
            raw_dir=self.raw_dir,
            camera_keys=self.camera_keys,
            get_task_idx=self.get_task_idx,
            translation_tolerance_m=self.translation_tolerance_m,
            rotation_tolerance_deg=self.rotation_tolerance_deg,
            skip_invalid_episodes=True,
            target_schema=schema,
            trim_extra_tail_frame=self.trim_extra_tail_frame,
            instruction_type=self.instruction_type,
            initial_episodes=self.schema_valid_episodes,
            initial_failures=self.failed_episodes,
            initial_repairs=self.repaired_episodes,
            initial_exclusions=self.excluded_episodes,
        )

    def for_episodes(
        self,
        episodes: list[tuple],
        *,
        target_schema: tuple[int, tuple[int, int]] | None = None,
    ) -> "AStarEpisodeCollection":
        return AStarEpisodeCollection(
            raw_dir=self.raw_dir,
            camera_keys=self.camera_keys,
            get_task_idx=self.get_task_idx,
            translation_tolerance_m=self.translation_tolerance_m,
            rotation_tolerance_deg=self.rotation_tolerance_deg,
            skip_invalid_episodes=True,
            target_schema=target_schema,
            trim_extra_tail_frame=self.trim_extra_tail_frame,
            instruction_type=self.instruction_type,
            initial_episodes=episodes,
            initial_failures=[],
            initial_repairs=[],
            initial_exclusions=self.excluded_episodes,
        )

    def __iter__(self):
        for episode_dir, meta, frames, task, task_info, episode_context in self.episodes:
            task_idx = self.get_task_idx(task)
            try:
                episode = AStarEpisode(
                    episode_dir=episode_dir,
                    meta=meta,
                    frames=frames,
                    camera_keys=self.camera_keys,
                    task=task,
                    task_idx=task_idx,
                    task_info=task_info,
                    body_from_camera=episode_context["body_from_camera"],
                    astar_context=episode_context["astar_context"],
                    instruction_type=self.instruction_type,
                )
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


def copy_depth_sidecars_limited(root: Path, camera_keys: list[str]) -> dict[str, Any]:
    extras_path = root / "meta" / "episodes_extras.jsonl"
    extras = load_jsonl_dicts(extras_path)
    report: dict[str, Any] = {
        "status": "completed",
        "source": str(extras_path),
        "output_root": str(root / "images"),
        "num_episodes": len(extras),
        "num_copied_files": 0,
        "missing": [],
    }
    if not extras:
        report["status"] = "skipped"
        report["reason"] = "missing_or_empty_episodes_extras_jsonl"
        return report

    for item in extras:
        source_episode_path = item.get("source_episode_path")
        episode_index = item.get("episode_index")
        if source_episode_path in (None, "") or episode_index is None:
            report["missing"].append(
                {
                    "source_episode_path": source_episode_path or "",
                    "episode_index": episode_index,
                    "reason": "missing_source_or_episode_index",
                }
            )
            continue

        episode_dir = Path(str(source_episode_path))
        frame_count = int(item.get("frame_count") or 0)
        chunk = int(episode_index) // 1000
        for camera in camera_keys:
            source_dir = episode_dir / "depth" / camera
            if not source_dir.exists():
                report["missing"].append(
                    {
                        "source_episode_path": str(episode_dir),
                        "episode_index": int(episode_index),
                        "camera": camera,
                        "reason": "missing_depth_dir",
                    }
                )
                continue

            depth_paths = sorted(source_dir.glob("*.png"))
            if not depth_paths:
                report["missing"].append(
                    {
                        "source_episode_path": str(episode_dir),
                        "episode_index": int(episode_index),
                        "camera": camera,
                        "reason": "empty_depth_dir",
                    }
                )
                continue

            used_paths = depth_paths[:frame_count] if frame_count > 0 else depth_paths
            if frame_count > 0 and len(depth_paths) < frame_count:
                report["missing"].append(
                    {
                        "source_episode_path": str(episode_dir),
                        "episode_index": int(episode_index),
                        "camera": camera,
                        "reason": "depth_frame_count_too_short",
                        "expected": frame_count,
                        "actual": len(depth_paths),
                    }
                )

            target_dir = (
                root
                / "images"
                / f"chunk-{chunk:03d}"
                / f"observation.depth.{camera}"
                / f"episode_{int(episode_index):06d}"
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, source_path in enumerate(used_paths):
                shutil.copy2(source_path, target_dir / f"{frame_index:05d}.png")
                report["num_copied_files"] += 1

    if report["missing"]:
        report["status"] = "completed_with_missing_depth"
    return report


def copy_astar_map_sidecars(root: Path) -> dict[str, Any]:
    extras_path = root / "meta" / "episodes_extras.jsonl"
    extras = load_jsonl_dicts(extras_path)
    report: dict[str, Any] = {
        "status": "completed",
        "source": str(extras_path),
        "output_root": str(root / "maps"),
        "num_episodes": len(extras),
        "num_copied_files": 0,
        "missing": [],
    }
    if not extras:
        report["status"] = "skipped"
        report["reason"] = "missing_or_empty_episodes_extras_jsonl"
        return report

    for item in extras:
        source_episode_path = item.get("source_episode_path")
        episode_index = item.get("episode_index")
        if source_episode_path in (None, "") or episode_index is None:
            report["missing"].append(
                {
                    "source_episode_path": source_episode_path or "",
                    "episode_index": episode_index,
                    "reason": "missing_source_or_episode_index",
                }
            )
            continue

        episode_dir = Path(str(source_episode_path))
        chunk = int(episode_index) // 1000
        for key, filename in MAP_SIDECARS.items():
            source_path = episode_dir / filename
            extension = ".json" if filename.endswith(".json") else ".png"
            relative_target = f"maps/chunk-{chunk:03d}/{key}/episode_{int(episode_index):06d}{extension}"
            target_path = root / relative_target
            if not source_path.exists():
                report["missing"].append(
                    {
                        "source_episode_path": str(episode_dir),
                        "episode_index": int(episode_index),
                        "map_key": key,
                        "reason": "missing_map_sidecar",
                        "source": str(source_path),
                    }
                )
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            item[f"maps.{key}"] = relative_target
            report["num_copied_files"] += 1

    write_jsonl_dicts(extras_path, extras)
    if report["missing"]:
        report["status"] = "completed_with_missing_maps"
    return report


def write_modality_json(root: Path, camera_keys: list[str]) -> dict[str, Any]:
    path = root / "meta" / "modality.json"
    payload: dict[str, Any] = {
        "state": {
            "drone": {"start": 0, "end": 7, "original_key": STATE_KEY},
        },
        "action": {
            "state": {"start": 0, "end": 7, "absolute": True, "original_key": ACTION_KEY},
        },
        "video": {
            camera: {"original_key": f"video.{camera}"}
            for camera in camera_keys
        },
        "annotation": {
            TASK_DESCRIPTION_KEY: {"original_key": TASK_DESCRIPTION_KEY},
        },
        "extra": [
            GRID_CELL_KEY,
            MAP_PIXEL_4096_KEY,
        ],
    }
    write_json(path, payload)
    return {
        "status": "completed",
        "output": str(path),
    }


def skipped_depth_sidecars_report(root: Path) -> dict[str, Any]:
    extras_path = root / "meta" / "episodes_extras.jsonl"
    extras = load_jsonl_dicts(extras_path)
    return {
        "status": "skipped",
        "reason": "include_depth_not_enabled",
        "source": str(extras_path),
        "output_root": str(root / "images"),
        "num_episodes": len(extras),
        "num_copied_files": 0,
        "missing": [],
    }


def write_astar_dataset_sidecars(root: Path, camera_keys: list[str], include_depth: bool = False) -> dict[str, Any]:
    depth_report = copy_depth_sidecars_limited(root, camera_keys) if include_depth else skipped_depth_sidecars_report(root)
    map_report = copy_astar_map_sidecars(root)
    modality_report = write_modality_json(root, camera_keys)
    return {
        "depth_sidecars": depth_report,
        "map_sidecars": map_report,
        "modality": modality_report,
    }


def write_astar_scene_sidecars(root: Path, camera_keys: list[str]) -> dict[str, Any]:
    return write_astar_dataset_sidecars(root, camera_keys, include_depth=True)


def build_output_groups(collection: AStarEpisodeCollection, output_dir: Path, split_by_schema: bool) -> list[dict[str, Any]]:
    if not split_by_schema:
        return [
            {
                "root": output_dir,
                "dataset_name": output_dir.name or "ue_astar",
                "schema": (collection.fps, collection.image_size),
                "schema_key": "",
                "episodes": list(collection.episodes),
            }
        ]

    groups: list[dict[str, Any]] = []
    for schema, episodes in sorted(
        group_episodes_by_schema(collection.schema_valid_episodes).items(),
        key=lambda item: schema_suffix(item[0]),
    ):
        schema_key = schema_suffix(schema)
        groups.append(
            {
                "root": output_dir / schema_key,
                "dataset_name": schema_key,
                "schema": schema,
                "schema_key": schema_key,
                "episodes": episodes,
            }
        )
    return groups


def write_astar_conversion_report(root: Path, report: dict[str, Any]):
    report_path = root / "meta" / "ue_astar_conversion_report.json"
    write_json(report_path, report)
    logging.info(
        "Wrote conversion report: %s (successful=%s failed=%s)",
        report_path,
        report.get("num_successful"),
        report.get("num_failed"),
    )


def run_conversion(collection: AStarEpisodeCollection, root: Path, dataset_name: str, args, started_at: str) -> dict[str, Any]:
    if len(collection) == 0:
        completed_at = utc_now_iso()
        report = collection.build_report(root, started_at, completed_at, "no_valid_episodes")
        write_astar_conversion_report(root, report)
        raise ValueError(f"No compatible A* episodes found under {collection.raw_dir}")

    logging.info(
        "Prepared %d compatible A* episodes for %s; skipped/failed during scan: %d",
        len(collection),
        root,
        len(collection.failed_episodes),
    )
    resolved_pix_fmt = select_video_pixel_format(collection.image_size, codec=args.codec, pix_fmt=args.pix_fmt)
    logging.info("Using fps=%s image_size=%s pix_fmt=%s", collection.fps, collection.image_size, resolved_pix_fmt)

    from utils.lerobot.lerobot_creater import LeRobotCreator

    creator = LeRobotCreator(
        root=str(root),
        robot_type=AStarEpisodeCollection.ROBOT_TYPE,
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
    sidecar_report: dict[str, Any] = {}
    try:
        for episode_index, episode in enumerate(collection, start=1):
            logging.info("Submitting A* episode %s / %s: %s", episode_index, len(collection), episode.episode_dir)
            creator.submit_episode(episode)

        logging.info("Waiting for worker processes and video encoders to finish")
        creator.wait()
        logging.info("Reading written episode metadata from %s", root / "meta" / "episodes_extras.jsonl")
        collection.sync_successful_episodes_from_output(root)
        sidecar_report = write_astar_dataset_sidecars(root, collection.camera_keys, include_depth=args.include_depth)
        logging.info("Validating generated LeRobot dataset at %s", root)
        validate_lerobot_dataset(repo_id=dataset_name, root=root)
        status = "completed"
    finally:
        if status != "completed":
            collection.sync_successful_episodes_from_output(root)
            if not sidecar_report:
                sidecar_report = write_astar_dataset_sidecars(root, collection.camera_keys, include_depth=args.include_depth)
        completed_at = utc_now_iso()
        report = collection.build_report(root, started_at, completed_at, status)
        report["sidecars"] = sidecar_report
        write_astar_conversion_report(root, report)

    logging.info("Done! %d episodes in %.2fs -> %s", len(collection.successful_episodes), time.time() - start_time, root)
    report = collection.build_report(root, started_at, utc_now_iso(), status)
    report["sidecars"] = sidecar_report
    return report


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_keys = parse_camera_keys(args.camera_keys)
    started_at = utc_now_iso()
    logging.info(
        "Starting A* conversion raw_dir=%s output_dir=%s instruction_type=%s cameras=%s include_depth=%s",
        raw_dir,
        output_dir,
        args.instruction_type,
        camera_keys,
        args.include_depth,
    )

    collection = AStarEpisodeCollection(
        raw_dir=raw_dir,
        camera_keys=camera_keys,
        get_task_idx=lambda _task: 0,
        translation_tolerance_m=args.extrinsic_tolerance_translation_m,
        rotation_tolerance_deg=args.extrinsic_tolerance_rotation_deg,
        skip_invalid_episodes=args.skip_invalid_episodes,
        keep_all_schemas=args.split_by_schema,
        trim_extra_tail_frame=args.trim_extra_tail_frame,
        instruction_type=args.instruction_type,
    )

    if not collection.schema_valid_episodes:
        completed_at = utc_now_iso()
        report = collection.build_report(output_dir, started_at, completed_at, "no_valid_episodes")
        write_astar_conversion_report(output_dir, report)
        raise ValueError(f"No schema-compatible A* episodes found under {raw_dir}")

    output_groups = build_output_groups(collection, output_dir, args.split_by_schema)
    group_reports: list[dict[str, Any]] = []
    group_errors: list[dict[str, str]] = []
    for group in output_groups:
        schema = group["schema"]
        schema_key = group["schema_key"]
        dataset_root = group["root"]
        dataset_name = args.dataset_name or group["dataset_name"]
        dataset_episodes = group["episodes"]
        logging.info(
            "Converting dataset schema=%s -> %s (%d episodes)",
            schema_key or "default",
            dataset_root,
            len(dataset_episodes),
        )
        dataset_collection = collection if not args.split_by_schema else collection.for_episodes(dataset_episodes, target_schema=schema)
        try:
            report = run_conversion(dataset_collection, dataset_root, dataset_name, args, started_at)
            report["schema_key"] = schema_key
            group_reports.append(report)
        except Exception as exc:
            logging.exception("A* dataset conversion failed for schema=%s", schema_key or "default")
            group_errors.append({"schema_key": schema_key, "error": str(exc)})

    completed_at = utc_now_iso()
    top_report = {
        "status": "completed" if not group_errors else "completed_with_errors",
        "started_at": started_at,
        "completed_at": completed_at,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "camera_keys": camera_keys,
        "instruction_type": args.instruction_type,
        "schema_groups": list(collection.schema_groups.values()),
        "group_reports": group_reports,
        "group_errors": group_errors,
        "scan_failures": collection.failed_episodes,
        "repaired_episodes": collection.repaired_episodes,
    }
    if args.split_by_schema:
        top_report_path = output_dir / "ue_astar_conversion_report.json"
        write_json(top_report_path, top_report)
        logging.info("Wrote conversion report: %s", top_report_path)

    if group_errors and not group_reports:
        raise RuntimeError(f"All A* dataset conversions failed under {output_dir}")


if __name__ == "__main__":
    main()
