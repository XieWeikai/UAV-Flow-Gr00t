import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from utils.map2nav_indoor.trajectory import _write_map_assets


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path("/mnt/glx/data/map2nav/r2r_replay_4_view_aligned")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_map_assets_are_copied_without_resize(tmp_path: Path):
    split_root = tmp_path / "input" / "train"
    source_dir = split_root / "episodes" / "train" / "mp3d_TestScene_traj_1"
    scene_root = split_root / "scenes" / "TestScene"
    graph_root = scene_root / "graph_floor_0p000"
    source_size = (321, 123)

    _write_rgb(scene_root / "level_0" / "layout.png", source_size, (10, 20, 30))
    _write_rgb(scene_root / "level_0" / "detail.png", source_size, (40, 50, 60))
    _write_rgb(graph_root / "graph.png", source_size, (70, 80, 90))
    _write_rgb(source_dir / "overlays" / "trajectory_on_layout_level_0.png", source_size, (100, 110, 120))
    _write_rgb(source_dir / "overlays" / "trajectory_on_detail_level_0.png", source_size, (130, 140, 150))
    _write_rgb(source_dir / "overlays" / "trajectory_on_graph_floor_0p000.png", source_size, (160, 170, 180))

    episode = {
        "scene_map_paths": {
            "levels": {
                "0": {
                    "layout": "scenes/TestScene/level_0/layout.png",
                    "detail": "scenes/TestScene/level_0/detail.png",
                }
            },
            "graph_floor": {"graph": "scenes/TestScene/graph_floor_0p000/graph.png"},
        },
        "overlay_paths": [
            "episodes/train/mp3d_TestScene_traj_1/overlays/trajectory_on_layout_level_0.png",
            "episodes/train/mp3d_TestScene_traj_1/overlays/trajectory_on_detail_level_0.png",
            "episodes/train/mp3d_TestScene_traj_1/overlays/trajectory_on_graph_floor_0p000.png",
        ],
    }
    steps = [{"map_xy": [17, 23], "graph_xy": [17, 23], "floorplan_xy": [17, 23]}]

    result = _write_map_assets(
        split_root=split_root,
        source_dir=source_dir,
        dataset_root=tmp_path / "output",
        chunk=0,
        episode_file="episode_000000",
        episode=episode,
        steps=steps,
    )

    assert result["scale"] == 1.0
    assert (result["width"], result["height"]) == source_size
    for rel_path in result["assets"].values():
        with Image.open(tmp_path / "output" / rel_path) as image:
            assert image.size == source_size


def test_map2nav_indoor_conversion_writes_rgb_videos_and_map_assets(tmp_path: Path):
    episodes_root = RAW_ROOT / "train" / "episodes" / "train"
    if not episodes_root.exists():
        pytest.skip(f"Map2Nav replay sample missing: {episodes_root}")

    output_root = tmp_path / "xnav_indoor_smoke" / "r2r"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "map2nav_indoor.py"),
        "--input-root",
        str(RAW_ROOT),
        "--output-root",
        str(output_root),
        "--dataset-name",
        "r2r",
        "--split",
        "train",
        "--max-episodes",
        "3",
        "--chunk-size",
        "3",
        "--overwrite",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Map2Nav indoor conversion failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    dataset_root = output_root / "train"
    meta_dir = dataset_root / "meta"
    data_file = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    extras_file = meta_dir / "episodes_extras.jsonl"
    info_file = meta_dir / "info.json"

    assert data_file.exists()
    assert extras_file.exists()
    assert info_file.exists()

    with info_file.open("r", encoding="utf-8") as f:
        info = json.load(f)
    assert info["total_episodes"] == 3
    video_keys = {
        key
        for key, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    }
    assert video_keys == {"video.front", "video.left", "video.right", "video.rear"}
    assert not any(key.startswith("video.floorplan") for key in info["features"])
    assert not any(key.startswith("video.traversibility") for key in info["features"])

    for key in ["video.front", "video.left", "video.right", "video.rear"]:
        assert (dataset_root / "videos" / "chunk-000" / key / "episode_000000.mp4").exists()

    extras = _read_jsonl(extras_file)
    first_extra = extras[0]
    assert first_extra["dataset_name"] == "r2r"
    assert first_extra["split"] == "train"
    assert first_extra["map_scale"] == 1.0
    assert len(first_extra["instructions"]) >= 1
    assert set(first_extra["map_assets"]) == {
        "traversibility",
        "traversibility_overlay",
        "floorplan",
        "floorplan_overlay",
        "floorplan_detail",
        "floorplan_detail_overlay",
    }

    source_episode = RAW_ROOT / "train" / first_extra["source_episode_dir"]
    source_steps = _read_jsonl(source_episode / "steps.jsonl")
    df = pd.read_parquet(data_file)
    assert df.columns[:4].tolist() == [
        "annotation.human.action.task_description",
        "observation.state",
        "action",
        "frame_index",
    ]
    assert len(df) == len(source_steps)
    assert np.asarray(df.iloc[0]["annotation.human.action.task_description"]).tolist() == [0]
    assert np.allclose(
        np.asarray(df.iloc[0]["observation.state"], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        np.stack(df["observation.state"].to_numpy()),
        np.stack(df["action"].to_numpy()),
        atol=1e-6,
    )

    first_xy = source_steps[0]["floorplan_xy"]
    assert list(df.iloc[0]["extra.floorplan_xy"]) == first_xy

    source_episode_json = json.loads((source_episode / "episode.json").read_text())
    floor_level = str(source_steps[0]["floor_level_id"])
    source_floorplan = RAW_ROOT / "train" / source_episode_json["scene_map_paths"]["levels"][floor_level]["layout"]
    with Image.open(source_floorplan) as image:
        source_map_width, source_map_height = image.size

    map_height, map_width = first_extra["map_size"]
    assert (map_width, map_height) == (source_map_width, source_map_height)
    for rel_path in first_extra["map_assets"].values():
        image_path = dataset_root / rel_path
        assert image_path.exists()
        with Image.open(image_path) as image:
            assert image.size == (map_width, map_height)
