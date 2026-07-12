"""Full source-to-output validation for Map2Nav VLN-CE datasets."""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from .assets import project_world_positions, resolve_map_bundle
from .converter import _read_jsonl, _scan_source
from .coordinates import habitat_poses_to_xnav
from .filtering import classify_floor_levels
from .schema import (
    DEFAULT_TASK,
    MAP_ASSET_KEYS,
    PARQUET_COLUMNS,
    RGB_VIEW_MAP,
    SCHEMA_VERSION,
    VideoInfo,
    build_features,
    build_modality,
    numeric_stats,
    parquet_schema,
)


EXPECTED_FULL_COUNTS = {
    ("r2r", "train"): {"episodes": 3390, "frames": 201379, "skipped": 213},
    ("r2r", "val_seen"): {"episodes": 248, "frames": 15114, "skipped": 11},
    ("r2r", "val_unseen"): {"episodes": 544, "frames": 31395, "skipped": 69},
    ("rxr_guide", "train"): {"episodes": 7327, "frames": 659415, "skipped": 956},
    ("rxr_guide", "val_seen"): {"episodes": 838, "frames": 74943, "skipped": 105},
    ("rxr_guide", "val_unseen"): {"episodes": 1010, "frames": 82360, "skipped": 180},
}


class DatasetValidationError(RuntimeError):
    """Raised when converted data differs from its replay source contract."""


def validate_split(
    *,
    input_root: str | Path,
    dataset_root: str | Path,
    dataset_name: str,
    split: str,
    hash_sample_size: int = 32,
    decode_video_sample_size: int = 32,
    enforce_expected_full_counts: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    split_root = input_root / split
    meta_root = dataset_root / "meta"
    info = _read_json(meta_root / "info.json")
    conversion = _read_json(meta_root / "conversion_report.json")
    modality = _read_json(meta_root / "modality.json")
    episodes = _read_jsonl(meta_root / "episodes.jsonl")
    stats = _read_jsonl(meta_root / "episodes_stats.jsonl")
    extras = _read_jsonl(meta_root / "episodes_extras.jsonl")
    skipped_rows = _read_jsonl(meta_root / "skipped_episodes.jsonl")
    tasks = _read_jsonl(meta_root / "tasks.jsonl")
    candidates = _scan_source(split_root, dataset_name=dataset_name, split=split)
    accepted_source = [candidate for candidate in candidates if candidate.eligibility.accepted]
    multi_floor = [candidate for candidate in candidates if not candidate.eligibility.accepted]

    accepted_count = int(conversion.get("accepted", -1))
    _require(conversion.get("schema_version") == SCHEMA_VERSION, "conversion schema_version mismatch")
    _require(conversion.get("errors") == 0, "conversion report contains errors")
    _require(conversion.get("source_manifest_total") == len(candidates), "source manifest count mismatch")
    _require(conversion.get("eligible_single_floor") == len(accepted_source), "eligible count mismatch")
    _require(conversion.get("skipped_multi_floor") == len(multi_floor), "multi-floor count mismatch")
    _require(accepted_count == len(episodes) == len(stats) == len(extras), "metadata row counts differ")
    _require(info.get("schema_version") == SCHEMA_VERSION, "info schema_version mismatch")
    _require(info.get("robot_type") == "map2nav_vlnce", "info robot_type mismatch")
    _require(info.get("total_episodes") == accepted_count, "info total_episodes mismatch")
    _require(info.get("total_videos") == accepted_count * 4, "info total_videos mismatch")
    _require(info.get("splits") == {split: f"0:{accepted_count}"}, "info splits mismatch")
    _require(tasks == [{"task_index": 0, "task": DEFAULT_TASK}], "tasks.jsonl contract mismatch")
    _require(modality == build_modality(), "modality.json contract mismatch")
    _require(bool(extras), "converted split has no episode extras")
    first_video = extras[0].get("video")
    _require(isinstance(first_video, dict), "episode video metadata is missing")
    expected_features = build_features(
        VideoInfo(
            width=int(first_video["width"]),
            height=int(first_video["height"]),
            fps=int(first_video["fps"]),
            hfov=float(first_video["hfov"]),
        )
    )
    _require(info.get("features") == expected_features, "info feature schema mismatch")
    _require(info.get("fps") == int(first_video["fps"]), "info fps mismatch")
    _require(
        "extra.floor_level_id" not in info["features"]
        and "extra.target_index" not in info["features"],
        "forbidden floor_level_id/target_index feature",
    )
    expected_skip_rows = [
        {
            "source_manifest_index": candidate.manifest_index,
            "source_episode_dir": candidate.episode_dir_relative,
            "trajectory_id": candidate.trajectory_id,
            "scene_key": candidate.scene_key,
            "reason": "multi_floor",
            "visited_levels": list(candidate.eligibility.visited_levels),
        }
        for candidate in multi_floor
    ]
    _require(skipped_rows == expected_skip_rows, "skipped_episodes.jsonl differs from source")
    selected_source = accepted_source[:accepted_count]
    _require(len(selected_source) == accepted_count, "output has more episodes than eligible source")

    if enforce_expected_full_counts:
        expected = EXPECTED_FULL_COUNTS[(dataset_name, split)]
        _require(conversion.get("complete_source_conversion") is True, "split is a limited conversion")
        _require(accepted_count == expected["episodes"], "accepted episode count differs from audit")
        _require(len(multi_floor) == expected["skipped"], "skipped episode count differs from audit")

    hash_indices = set(_sample_indices(accepted_count, hash_sample_size, f"hash:{dataset_name}:{split}"))
    decode_indices = set(
        _sample_indices(accepted_count, decode_video_sample_size, f"decode:{dataset_name}:{split}")
    )
    sha_samples: list[dict[str, Any]] = []
    global_frame_start = 0
    max_projection_error = 0
    validated_videos = 0
    validated_maps = 0
    chunk_size = int(info["chunks_size"])
    iterator = tqdm(
        zip(selected_source, episodes, stats, extras, strict=True),
        total=accepted_count,
        desc=f"Validate {dataset_name}/{split}",
        unit="episode",
    )
    for episode_index, (source_candidate, episode_row, stats_row, extra) in enumerate(iterator):
        _require(episode_row.get("episode_index") == episode_index, "non-contiguous episodes.jsonl")
        _require(stats_row.get("episode_index") == episode_index, "non-contiguous episode stats")
        _require(extra.get("episode_index") == episode_index, "non-contiguous episode extras")
        _require(
            extra.get("source_episode_dir") == source_candidate.episode_dir_relative,
            f"source episode order mismatch at output episode {episode_index}",
        )
        source_episode = _read_json(source_candidate.episode_dir / "episode.json")
        steps = _read_jsonl(source_candidate.episode_dir / "steps.jsonl")
        eligibility = classify_floor_levels(steps)
        _require(
            eligibility.accepted and eligibility.source_level_id is not None,
            f"output episode {episode_index} is not single-floor",
        )
        bundle = resolve_map_bundle(split_root, source_episode, eligibility.source_level_id)
        length = len(steps)
        _require(episode_row.get("length") == length, f"episode length mismatch at {episode_index}")
        _require(extra.get("schema_version") == SCHEMA_VERSION, "episode schema_version mismatch")
        _require(extra.get("trajectory_id") == source_episode.get("trajectory_id"), "trajectory mismatch")
        _require(extra.get("scene_key") == source_episode.get("scene_key"), "scene mismatch")
        _require(extra.get("instructions") == source_episode.get("instructions"), "instructions mismatch")
        _require(
            extra.get("source_episode_ids")
            == [str(value) for value in source_episode.get("episode_ids", [])],
            "source episode ids mismatch",
        )
        _require(extra.get("map_projection") == bundle.projection, "map projection metadata mismatch")
        _require(extra.get("map_size") == [bundle.height, bundle.width], "map size metadata mismatch")

        chunk = episode_index // chunk_size
        name = f"episode_{episode_index:06d}"
        parquet = dataset_root / "data" / f"chunk-{chunk:03d}" / f"{name}.parquet"
        _require(parquet.is_file() and parquet.stat().st_size > 0, f"missing parquet: {parquet}")
        frame = pd.read_parquet(parquet)
        _require(frame.columns.tolist() == PARQUET_COLUMNS, f"parquet columns mismatch: {parquet}")
        _require(len(frame) == length, f"parquet length mismatch: {parquet}")
        _require(
            all(value == "" for value in frame["extra.cot"].tolist()),
            f"extra.cot must be an empty string in phase one: {parquet}",
        )
        _require(pq.read_schema(parquet).equals(parquet_schema()), f"Arrow schema mismatch: {parquet}")
        _require(
            "extra.floor_level_id" not in frame and "extra.target_index" not in frame,
            f"forbidden fields in parquet: {parquet}",
        )

        source_positions = np.asarray([step["position"] for step in steps], dtype=np.float32)
        source_rotations = np.asarray([step["rotation"] for step in steps], dtype=np.float32)
        output_positions = np.stack(frame["extra.habitat_world_position"].to_numpy()).astype(
            np.float32
        )
        output_rotations = np.stack(
            frame["extra.habitat_world_rotation_xyzw"].to_numpy()
        ).astype(np.float32)
        _require(
            np.array_equal(output_positions, source_positions),
            f"raw Habitat position changed: {parquet}",
        )
        _require(
            np.array_equal(output_rotations, source_rotations),
            f"raw Habitat rotation changed: {parquet}",
        )
        expected_states = habitat_poses_to_xnav(source_positions, source_rotations)
        output_states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        output_actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        _require(
            np.allclose(output_states, expected_states, atol=1e-6, rtol=0.0),
            f"xNav pose conversion mismatch: {parquet}",
        )
        _require(np.array_equal(output_states, output_actions), f"action mirror mismatch: {parquet}")
        _require(
            np.array_equal(output_states[0], [0, 0, 0, 0, 0, 0, 1]),
            f"first xNav pose is not identity: {parquet}",
        )
        _require(
            np.allclose(np.linalg.norm(output_states[:, 3:], axis=1), 1.0, atol=1e-6),
            f"non-unit xNav quaternion: {parquet}",
        )
        _require(np.all(output_states[:, 6] >= 0), f"non-canonical xNav quaternion sign: {parquet}")

        output_pixels = np.stack(frame["extra.floorplan_xy"].to_numpy()).astype(np.int32)
        source_pixels = np.asarray([step["floorplan_xy"] for step in steps], dtype=np.int32)
        _require(np.array_equal(output_pixels, source_pixels), f"floorplan_xy changed: {parquet}")
        projected = project_world_positions(source_positions, bundle.projection, rounded=True)
        projection_error = int(np.abs(projected - source_pixels).max(initial=0))
        max_projection_error = max(max_projection_error, projection_error)
        _require(projection_error <= 1, f"projection error >1px: {parquet}")

        discrete = np.stack(frame["extra.discrete_action_to_next_id"].to_numpy()).reshape(-1)
        source_discrete = np.asarray(
            [step["discrete_action_to_next_id"] for step in steps], dtype=np.int32
        )
        _require(np.array_equal(discrete, source_discrete), f"discrete actions changed: {parquet}")
        _require(int(discrete[-1]) == 0, f"terminal action is not STOP: {parquet}")
        expected_stats = numeric_stats(
            {
                "observation.state": output_states,
                "action": output_actions,
                "extra.habitat_world_position": output_positions,
                "extra.habitat_world_rotation_xyzw": output_rotations,
                "extra.floorplan_xy": output_pixels,
                "extra.discrete_action_to_next_id": discrete.reshape(-1, 1),
            }
        )
        _require(
            _episode_stats_equal(stats_row.get("stats"), expected_stats),
            f"episode stats mismatch at output episode {episode_index}",
        )
        _require(
            np.array_equal(frame["frame_index"].to_numpy(), np.arange(length)),
            f"frame_index mismatch: {parquet}",
        )
        _require(
            np.array_equal(
                frame["index"].to_numpy(),
                np.arange(global_frame_start, global_frame_start + length),
            ),
            f"global index mismatch: {parquet}",
        )
        _require(
            np.all(frame["episode_index"].to_numpy() == episode_index),
            f"episode_index mismatch: {parquet}",
        )
        _require(np.all(frame["task_index"].to_numpy() == 0), f"task_index mismatch: {parquet}")

        map_assets = extra.get("map_assets")
        _require(
            isinstance(map_assets, dict) and tuple(map_assets) == MAP_ASSET_KEYS,
            f"map asset inventory mismatch at episode {episode_index}",
        )
        for key in MAP_ASSET_KEYS:
            target = dataset_root / map_assets[key]
            source = bundle.sources[key]
            _validate_copied_file(source, target, label=f"map {key}")
            try:
                with Image.open(target) as image:
                    image.verify()
                with Image.open(target) as image:
                    _require(
                        image.size == (bundle.width, bundle.height),
                        f"map dimensions mismatch: {target}",
                    )
            except DatasetValidationError:
                raise
            except Exception as exc:
                raise DatasetValidationError(f"cannot decode map: {target}") from exc
            validated_maps += 1
            if episode_index in hash_indices:
                sha_samples.append(_hash_pair(source, target, episode_index, f"map.{key}"))

        for source_view, output_view in RGB_VIEW_MAP.items():
            source = source_candidate.episode_dir / f"{source_view}.mp4"
            target = (
                dataset_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / f"video.{output_view}"
                / f"{name}.mp4"
            )
            _validate_copied_file(source, target, label=f"video.{output_view}")
            _validate_video_header(
                target,
                expected_frames=length,
                expected_width=int(extra["video"]["width"]),
                expected_height=int(extra["video"]["height"]),
                expected_fps=float(extra["video"]["fps"]),
            )
            if episode_index in decode_indices:
                _decode_video_samples(target, length)
            if episode_index in hash_indices:
                sha_samples.append(_hash_pair(source, target, episode_index, f"video.{output_view}"))
            validated_videos += 1
        global_frame_start += length

    _require(info.get("total_frames") == global_frame_start, "info total_frames mismatch")
    _require(conversion.get("accepted_frames") == global_frame_start, "report frame count mismatch")
    _require(
        len(list(dataset_root.glob("data/chunk-*/episode_*.parquet"))) == accepted_count,
        "unexpected parquet file count",
    )
    _require(
        len(list(dataset_root.glob("videos/chunk-*/video.*/episode_*.mp4")))
        == accepted_count * 4,
        "unexpected video file count",
    )
    _require(
        len(list(dataset_root.glob("maps/chunk-*/episode_*/*.png"))) == accepted_count * 6,
        "unexpected map file count",
    )
    if enforce_expected_full_counts:
        expected = EXPECTED_FULL_COUNTS[(dataset_name, split)]
        _require(global_frame_start == expected["frames"], "accepted frame count differs from audit")

    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "split": split,
        "source_manifest_total": len(candidates),
        "accepted_episodes": accepted_count,
        "skipped_multi_floor": len(multi_floor),
        "validated_frames": global_frame_start,
        "validated_videos": validated_videos,
        "validated_maps": validated_maps,
        "max_projection_error_px": max_projection_error,
        "hash_sample_episodes": sorted(hash_indices),
        "decoded_video_sample_episodes": sorted(decode_indices),
        "sha256_samples": sha_samples,
    }


def validate_delivery(
    *,
    processed_root: str | Path,
    r2r_input_root: str | Path,
    rxr_input_root: str | Path,
    hash_sample_size: int = 32,
    decode_video_sample_size: int = 32,
) -> dict[str, Any]:
    processed_root = Path(processed_root).resolve()
    split_reports: list[dict[str, Any]] = []
    for dataset_name, input_root in (
        ("r2r", Path(r2r_input_root)),
        ("rxr_guide", Path(rxr_input_root)),
    ):
        for split in ("train", "val_seen", "val_unseen"):
            split_reports.append(
                validate_split(
                    input_root=input_root,
                    dataset_root=processed_root / dataset_name / split,
                    dataset_name=dataset_name,
                    split=split,
                    hash_sample_size=hash_sample_size,
                    decode_video_sample_size=decode_video_sample_size,
                    enforce_expected_full_counts=True,
                )
            )
    report = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "processed_root": str(processed_root),
        "totals": {
            "source_manifest_total": sum(row["source_manifest_total"] for row in split_reports),
            "accepted_episodes": sum(row["accepted_episodes"] for row in split_reports),
            "skipped_multi_floor": sum(row["skipped_multi_floor"] for row in split_reports),
            "validated_frames": sum(row["validated_frames"] for row in split_reports),
            "validated_videos": sum(row["validated_videos"] for row in split_reports),
            "validated_maps": sum(row["validated_maps"] for row in split_reports),
        },
        "splits": split_reports,
    }
    _write_json(processed_root / "stage1_delivery_report.json", report)
    (processed_root / "stage1_delivery_report.html").write_text(
        _delivery_html(report), encoding="utf-8"
    )
    return report


def _validate_copied_file(source: Path, target: Path, *, label: str) -> None:
    _require(source.is_file() and source.stat().st_size > 0, f"missing source {label}: {source}")
    _require(target.is_file() and target.stat().st_size > 0, f"missing output {label}: {target}")
    _require(source.stat().st_size == target.stat().st_size, f"copied size mismatch: {target}")
    _require(not os.path.samefile(source, target), f"{label} is hardlinked instead of copied: {target}")


def _validate_video_header(
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        _require(capture.isOpened(), f"cannot open video: {path}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    _require((width, height) == (expected_width, expected_height), f"video size mismatch: {path}")
    _require(frames == expected_frames, f"video frame count mismatch: {path}")
    _require(abs(fps - expected_fps) <= 0.05, f"video fps mismatch: {path}")


def _decode_video_samples(path: Path, frame_count: int) -> None:
    indices = sorted({0, frame_count // 2, frame_count - 1})
    capture = cv2.VideoCapture(str(path))
    try:
        _require(capture.isOpened(), f"cannot open sampled video: {path}")
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            _require(ok and frame is not None and frame.size > 0, f"cannot decode {path} frame {index}")
    finally:
        capture.release()


def _hash_pair(source: Path, target: Path, episode_index: int, key: str) -> dict[str, Any]:
    source_hash = _sha256(source)
    target_hash = _sha256(target)
    _require(source_hash == target_hash, f"SHA-256 mismatch: {target}")
    return {
        "episode_index": episode_index,
        "key": key,
        "source": str(source),
        "target": str(target),
        "sha256": source_hash,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sample_indices(total: int, count: int, seed: str) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), count))


def _episode_stats_equal(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, expected_metrics in expected.items():
        actual_metrics = actual.get(key)
        if not isinstance(actual_metrics, dict) or set(actual_metrics) != set(expected_metrics):
            return False
        for metric, expected_value in expected_metrics.items():
            actual_value = np.asarray(actual_metrics[metric])
            expected_array = np.asarray(expected_value)
            if actual_value.shape != expected_array.shape:
                return False
            if metric == "count":
                if not np.array_equal(actual_value, expected_array):
                    return False
            elif not np.allclose(actual_value, expected_array, atol=1e-8, rtol=1e-8):
                return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetValidationError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetValidationError(message)


def _delivery_html(report: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['dataset_name'])}</td>"
        f"<td>{html.escape(row['split'])}</td>"
        f"<td>{row['source_manifest_total']}</td>"
        f"<td>{row['accepted_episodes']}</td>"
        f"<td>{row['skipped_multi_floor']}</td>"
        f"<td>{row['validated_frames']}</td>"
        f"<td>{row['validated_videos']}</td>"
        f"<td>{row['validated_maps']}</td>"
        f"<td>{row['max_projection_error_px']}</td>"
        "</tr>"
        for row in report["splits"]
    )
    totals = report["totals"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Map2Nav VLN-CE 阶段一交付报告</title>
<style>body{{font:15px/1.55 system-ui;margin:32px;color:#1f2328}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d0d7de;padding:8px;text-align:right}}th:first-child,td:first-child,
th:nth-child(2),td:nth-child(2){{text-align:left}}.ok{{color:#1a7f37;font-weight:700}}</style>
</head><body><h1>Map2Nav VLN-CE 阶段一交付报告</h1>
<p class="ok">状态：{html.escape(report['status'])}</p>
<p>Schema：<code>{html.escape(report['schema_version'])}</code>；
输出：<code>{html.escape(report['processed_root'])}</code></p>
<table><thead><tr><th>数据集</th><th>Split</th><th>源轨迹</th><th>输出</th>
<th>跨楼层过滤</th><th>帧</th><th>视频</th><th>地图</th><th>最大投影误差(px)</th>
</tr></thead><tbody>{rows}</tbody></table>
<p>总计：源轨迹 {totals['source_manifest_total']}，输出 episode {totals['accepted_episodes']}，
跨楼层过滤 {totals['skipped_multi_floor']}，帧 {totals['validated_frames']}，
视频 {totals['validated_videos']}，地图 {totals['validated_maps']}。</p>
</body></html>
"""
