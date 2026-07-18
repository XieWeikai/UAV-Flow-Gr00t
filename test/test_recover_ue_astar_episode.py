from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

from scripts.recover_ue_astar_episode import (
    audit_incomplete_episode,
    compute_recovery_stats,
    encode_missing_videos,
    repair_metadata,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_incomplete_dataset(root: Path) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "fps": 10,
        "total_episodes": 3,
        "total_frames": 2,
        "total_tasks": 3,
        "total_videos": 4,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:3"},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "video.front": {"dtype": "video", "shape": [2, 2, 3]},
            "video.rear": {"dtype": "video", "shape": [2, 2, 3]},
            "action": {"dtype": "float32", "shape": [2]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": 0, "task": "first"},
            {"task_index": 1, "task": "missing"},
            {"task_index": 2, "task": "last"},
        ],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["first"], "length": 1},
            {"episode_index": 2, "tasks": ["last"], "length": 1},
        ],
    )
    _write_jsonl(
        meta / "episodes_extras.jsonl",
        [
            {"episode_index": 0, "source_episode_path": "/source/first"},
            {"episode_index": 2, "source_episode_path": "/source/last"},
        ],
    )
    _write_jsonl(
        meta / "episodes_stats.jsonl",
        [
            {"episode_index": 0, "stats": {}},
            {"episode_index": 2, "stats": {}},
        ],
    )

    parquet = root / "data" / "chunk-000" / "episode_000001.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [1, 1],
            "task_index": [1, 1],
            "frame_index": [0, 1],
            "observation.state": [
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([2.0, 3.0], dtype=np.float32),
            ],
            "action": [
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([2.0, 3.0], dtype=np.float32),
            ],
            "timestamp": [0.0, 0.1],
        }
    ).to_parquet(parquet, index=False)

    for camera in ("front", "rear"):
        temp = root / "videos" / "chunk-000" / f"video.{camera}" / "episode_000001_temp"
        temp.mkdir(parents=True)
        Image.fromarray(np.full((2, 2, 3), 32, dtype=np.uint8)).save(temp / "frame_000000.png")
        Image.fromarray(np.full((2, 2, 3), 96, dtype=np.uint8)).save(temp / "frame_000001.png")


def test_audit_accepts_one_metadata_hole_with_complete_temp_frames(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_incomplete_dataset(root)

    audit = audit_incomplete_episode(root, episode_index=1)

    assert audit.episode_index == 1
    assert audit.frame_count == 2
    assert audit.task_index == 1
    assert audit.task == "missing"
    assert audit.cameras == ("front", "rear")


def test_encode_missing_videos_preserves_temp_frames(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_incomplete_dataset(root)
    audit = audit_incomplete_episode(root, episode_index=1)

    outputs = encode_missing_videos(root, audit)

    assert len(outputs) == 2
    for path in outputs:
        assert path.is_file()
        with av.open(str(path), "r") as container:
            assert sum(1 for _ in container.decode(video=0)) == 2
    assert all(path.is_file() for path in audit.temp_frames["front"])
    assert all(path.is_file() for path in audit.temp_frames["rear"])


def test_repair_metadata_fills_hole_and_rebuilds_counters(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_incomplete_dataset(root)
    audit = audit_incomplete_episode(root, episode_index=1)

    repair_metadata(
        root,
        audit,
        extras={"source_episode_path": "/source/missing", "frame_count": 2},
        stats={"observation.state": {"count": [2]}},
    )

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    extras = [json.loads(line) for line in (root / "meta" / "episodes_extras.jsonl").read_text().splitlines()]
    stats = [json.loads(line) for line in (root / "meta" / "episodes_stats.jsonl").read_text().splitlines()]

    assert [row["episode_index"] for row in episodes] == [0, 1, 2]
    assert [row["episode_index"] for row in extras] == [0, 1, 2]
    assert [row["episode_index"] for row in stats] == [0, 1, 2]
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 4
    assert info["total_videos"] == 6
    assert info["total_tasks"] == 3
    assert info["total_chunks"] == 1
    assert info["splits"] == {"train": "0:3"}


def test_compute_recovery_stats_matches_episode_stat_contract(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_incomplete_dataset(root)
    audit = audit_incomplete_episode(root, episode_index=1)

    stats = compute_recovery_stats(root, audit)

    assert set(stats) == {"observation.state", "video.front", "video.rear", "action"}
    for key in ("observation.state", "action"):
        assert stats[key]["min"] == [0.0, 1.0]
        assert stats[key]["max"] == [2.0, 3.0]
        assert stats[key]["mean"] == [1.0, 2.0]
        assert stats[key]["std"] == [1.0, 1.0]
        assert stats[key]["count"] == [2]
    expected_video = {
        "min": np.full(3, 32 / 255),
        "max": np.full(3, 96 / 255),
        "mean": np.full(3, 64 / 255),
        "std": np.full(3, 32 / 255),
    }
    for camera in ("front", "rear"):
        for stat_name, expected in expected_video.items():
            np.testing.assert_allclose(stats[f"video.{camera}"][stat_name], expected, rtol=1e-6)
        assert stats[f"video.{camera}"]["count"] == [2]
