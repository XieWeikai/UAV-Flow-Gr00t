from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
import unittest
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from tracking import (
    ACTION_KEY,
    CONTROLLER_DT,
    EXPECTED_FULL_COUNTS,
    FPS,
    STATE_KEY,
    EpisodePlan,
    assign_root_indices,
    convert_dataset,
    integrate_nominal_poses,
    split_inventory,
    validate_output_dataset,
)

WORK_ROOT = Path("/data4/glx/tracking/work/tests")


def make_row(
    current: str,
    instruction: str,
    command: list[float],
    *,
    collision: bool = False,
) -> dict:
    return {
        "images": [],
        "current": current,
        "instruction": instruction,
        "trajectory": [[0.0, 0.0, 0.0]],
        "actions": [command],
        "collision": collision,
        "target_distance": 1.0,
    }


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def jpeg_bytes(value: int, size: tuple[int, int] = (384, 384)) -> bytes:
    output = io.BytesIO()
    image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(image).save(output, format="JPEG", quality=90)
    return output.getvalue()


def write_synthetic_archives(root: Path) -> tuple[Path, Path]:
    jsonl_archive = root / "raw" / "archives" / "jsonl" / "seed_101.tar"
    frames_dir = root / "raw" / "archives" / "frames" / "seed_101"
    jsonl_archive.parent.mkdir(parents=True)
    frames_dir.mkdir(parents=True)

    episodes = {
        ("source_a", "2", "Follow A"): [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ("source_a", "10", "Follow B"): [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        ("source_b", "1", "Follow C"): [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
    }
    with tarfile.open(jsonl_archive, mode="w") as json_tar:
        by_source: dict[str, list[tuple[str, str, list[list[float]]]]] = {}
        for (source, episode, instruction), commands in episodes.items():
            by_source.setdefault(source, []).append((episode, instruction, commands))
            rows = [
                make_row(
                    f"frames/seed_101/{source}/{episode}/frame_{index + 1:05d}.jpg",
                    instruction,
                    command,
                    collision=index == 0,
                )
                for index, command in enumerate(commands)
            ]
            payload = b"".join(
                (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
                for row in rows
            )
            add_bytes(json_tar, f"seed_101/{source}/{episode}.jsonl", payload)
        for source, source_episodes in by_source.items():
            with tarfile.open(frames_dir / f"{source}.tar", mode="w") as frame_tar:
                for episode, _, commands in source_episodes:
                    for index in range(len(commands) + 2):
                        add_bytes(
                            frame_tar,
                            f"{source}/{episode}/frame_{index + 1:05d}.jpg",
                            jpeg_bytes(20 + index * 20),
                        )
    return jsonl_archive, frames_dir


class TrackingConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        temp_dir = WORK_ROOT / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for key in ("TMPDIR", "TEMP", "TMP"):
            os.environ[key] = str(temp_dir)

    def setUp(self) -> None:
        test_name = self.id().rsplit(".", 1)[-1]
        self.root = WORK_ROOT / test_name
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_integrate_nominal_poses_is_pre_action_and_clipped(self):
        rows = [
            make_row("unused", "task", [1.0, 0.0, 0.0]),
            make_row("unused", "task", [0.0, 1.0, 0.0]),
            make_row("unused", "task", [0.0, 0.0, 1.0]),
            make_row("unused", "task", [100.0, 100.0, 100.0]),
        ]
        poses = integrate_nominal_poses(rows)
        self.assertEqual(poses.shape, (4, 7))
        np.testing.assert_allclose(poses[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(poses[1, :3], [15 * CONTROLLER_DT, 0, 0], atol=1e-7)
        np.testing.assert_allclose(
            poses[2, :3],
            [15 * CONTROLLER_DT, 10 * CONTROLLER_DT, 0],
            atol=1e-7,
        )
        self.assertGreater(poses[3, 5], 0.0)
        self.assertEqual(len(poses), len(rows))
        # The last extreme command is deliberately not integrated: there is no frame N+1.
        np.testing.assert_allclose(poses[-1], poses[3], atol=0.0)

    def test_integrate_negative_lateral_and_yaw(self):
        rows = [
            make_row("unused", "task", [0.0, -1.0, 0.0]),
            make_row("unused", "task", [0.0, 0.0, -1.0]),
            make_row("unused", "task", [0.0, 0.0, 0.0]),
        ]
        poses = integrate_nominal_poses(rows)
        self.assertLess(poses[1, 1], 0.0)
        self.assertLess(poses[2, 5], 0.0)

    def test_split_and_root_indices_are_deterministic(self):
        plans = [
            EpisodePlan("seed_101", "b", "10", 10, "seed_101/b/10.jsonl", "A", 9),
            EpisodePlan("seed_101", "a", "2", 2, "seed_101/a/2.jsonl", "A", 5),
            EpisodePlan("seed_101", "a", "3", 3, "seed_101/a/3.jsonl", "A", 7),
            EpisodePlan("seed_101", "c", "1", 1, "seed_101/c/1.jsonl", "B", 4),
            EpisodePlan("seed_101", "d", "1", 1, "seed_101/d/1.jsonl", "B", 8),
        ]
        split_once = assign_root_indices(split_inventory(plans, val_unseen_instruction_count=1))
        split_twice = assign_root_indices(split_inventory(list(reversed(plans)), val_unseen_instruction_count=1))
        signature = lambda values: sorted(
            (
                p.canonical_key,
                p.split,
                p.episode_index,
                p.task_index,
                p.global_frame_start,
            )
            for p in values
        )
        self.assertEqual(signature(split_once), signature(split_twice))
        unseen = {p.instruction for p in split_once if p.split == "val_unseen"}
        self.assertEqual(len(unseen), 1)
        for instruction in unseen:
            self.assertTrue(all(p.split == "val_unseen" for p in split_once if p.instruction == instruction))

    def test_synthetic_end_to_end(self):
        jsonl_archive, frames_dir = write_synthetic_archives(self.root)
        output = self.root / "processed"
        result = convert_dataset(
            jsonl_archive=jsonl_archive,
            frames_dir=frames_dir,
            output_dir=output,
            work_dir=self.root / "work",
            workers=2,
            max_episodes=3,
            allow_partial=True,
            strict_full_inventory=False,
        )
        self.assertTrue(result["partial"])
        summary = validate_output_dataset(output, decode_videos=True)
        self.assertEqual(sum(item["episodes"] for item in summary.values()), 3)
        self.assertEqual(sum(item["frames"] for item in summary.values()), 7)

        parquet_files = sorted(output.rglob("episode_*.parquet"))
        video_files = sorted(output.rglob("episode_*.mp4"))
        self.assertEqual(len(parquet_files), 3)
        self.assertEqual(len(video_files), 3)
        table = pq.read_table(parquet_files[0])
        state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
        action = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
        np.testing.assert_array_equal(state, action)
        with av.open(str(video_files[0])) as container:
            self.assertEqual(container.streams.video[0].codec_context.name, "h264")
            self.assertEqual(container.streams.video[0].codec_context.format.name, "yuv420p")
            self.assertAlmostEqual(float(container.streams.video[0].average_rate), FPS)

        extras_files = sorted(output.rglob("episodes_extras.jsonl"))
        self.assertTrue(extras_files)
        for path in extras_files:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertIsNone(row["video.front.K"])
                self.assertIsNone(row["K_front"])
                self.assertGreater(row["physical_frame_count"], row["frame_count"])

        self.assertFalse(list(output.rglob("pair_tasks.jsonl")))
        self.assertFalse(list(output.rglob("conversion_context.json")))
        self.assertFalse(list(output.rglob("conversion_report.json")))
        for root in [path.parent.parent for path in output.rglob("meta/info.json")]:
            self.assertTrue((root / "depth").is_dir())
            self.assertTrue((root / "maps").is_dir())
            for empty_view in ("video.rear", "video.left", "video.right"):
                dirs = list((root / "videos").glob(f"chunk-*/{empty_view}"))
                self.assertTrue(dirs)
                self.assertTrue(all(not any(directory.iterdir()) for directory in dirs))

    def test_existing_output_requires_overwrite(self):
        jsonl_archive, frames_dir = write_synthetic_archives(self.root)
        output = self.root / "processed"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            convert_dataset(
                jsonl_archive=jsonl_archive,
                frames_dir=frames_dir,
                output_dir=output,
                work_dir=self.root / "work",
                max_episodes=1,
                allow_partial=True,
            )

    def test_filters_require_partial_mode(self):
        jsonl_archive, frames_dir = write_synthetic_archives(self.root)
        with self.assertRaisesRegex(ValueError, "filters require --allow-partial"):
            convert_dataset(
                jsonl_archive=jsonl_archive,
                frames_dir=frames_dir,
                output_dir=self.root / "processed",
                work_dir=self.root / "work",
                max_episodes=1,
                strict_full_inventory=False,
            )


if __name__ == "__main__":
    unittest.main()
