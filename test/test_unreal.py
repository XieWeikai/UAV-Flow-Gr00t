from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET_ENGINE = True
except ImportError:
    HAS_PARQUET_ENGINE = False

from unreal import (
    ACTION_KEY,
    STATE_KEY,
    TASK_DESCRIPTION_KEY,
    UnrealEpisode,
    UnrealEpisodeCollection,
    body_from_camera_for_frame,
    build_features,
    copy_depth_sidecars,
    group_episodes_by_scene,
    group_episodes_by_schema,
    intrinsic_matrix,
    intrinsic_4,
    scan_episode_dirs,
    validate_fixed_extrinsics,
    write_episode_extras_parquet,
)


def make_frame(frame_index, body_x_cm, camera_x_cm):
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "timestamp_wall_sec": float(frame_index),
        "timestamp_sim_sec": float(frame_index),
        "pose": [body_x_cm, 0.0, 0.0, 0.0, 0.0, 0.0],
        "view_mode": "first_person",
        "camera_pose_front": [camera_x_cm, 0.0, 0.0, 0.0, 0.0, 0.0],
        "K_front": [100.0, 0.0, 2.0, 0.0, 110.0, 3.0, 0.0, 0.0, 1.0],
    }


def write_episode(
    root: Path,
    frames: list[dict],
    scene_id: str = "scene_0001",
    user_id: str = "user_0001",
    episode_id: str = "episode_000000",
    width: int = 4,
    height: int = 3,
    fps: int = 10,
    meta_frame_count: int | None = None,
    write_depth: bool = False,
):
    episode_dir = root / scene_id / user_id / episode_id
    (episode_dir / "rgb" / "front").mkdir(parents=True)
    if write_depth:
        (episode_dir / "depth" / "front").mkdir(parents=True)

    meta = {
        "status": "completed",
        "episode_index": 0,
        "scene_id": scene_id,
        "map_name": "Entry",
        "capture_width": width,
        "capture_height": height,
        "sample_rate_hz": fps,
        "frame_count": len(frames) if meta_frame_count is None else meta_frame_count,
        "camera_names": ["front"],
    }
    (episode_dir / "episode_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with (episode_dir / "frames.jsonl").open("w", encoding="utf-8") as file:
        for frame in frames:
            file.write(json.dumps(frame) + "\n")

    for index in range(len(frames)):
        image = np.full((height, width, 3), index + 1, dtype=np.uint8)
        Image.fromarray(image).save(episode_dir / "rgb" / "front" / f"{index:05d}.png")
        if write_depth:
            depth = np.full((height, width), index + 1, dtype=np.uint16)
            Image.fromarray(depth).save(episode_dir / "depth" / "front" / f"{index:05d}.png")

    return episode_dir, meta


class UnrealConversionTests(unittest.TestCase):
    def test_scan_episode_dirs_accepts_root_and_episode_dir(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            episode_dir, _ = write_episode(root, [make_frame(0, 0.0, 100.0)])

            self.assertEqual(scan_episode_dirs(root), [episode_dir])
            self.assertEqual(scan_episode_dirs(episode_dir), [episode_dir])

    def test_intrinsic_4_extracts_fx_fy_cx_cy(self):
        frame = make_frame(0, 0.0, 100.0)
        self.assertEqual(intrinsic_4(frame, "front"), [100.0, 110.0, 2.0, 3.0])
        self.assertEqual(intrinsic_matrix(frame, "front"), [[100.0, 0.0, 2.0], [0.0, 110.0, 3.0], [0.0, 0.0, 1.0]])

    def test_body_from_camera_is_fixed_when_body_and_camera_move_together(self):
        frames = [
            make_frame(0, 0.0, 100.0),
            make_frame(1, 100.0, 200.0),
        ]
        baseline = validate_fixed_extrinsics(Path("episode_000000"), frames, ["front"], 1e-4, 0.1)

        expected = body_from_camera_for_frame(frames[0], "front")
        np.testing.assert_allclose(baseline["front"], expected, atol=1e-6)

    def test_body_from_camera_change_fails(self):
        frames = [
            make_frame(0, 0.0, 100.0),
            make_frame(1, 100.0, 201.0),
        ]
        with self.assertRaisesRegex(ValueError, "Dynamic body_from_camera"):
            validate_fixed_extrinsics(Path("episode_000000"), frames, ["front"], 1e-4, 0.1)

    def test_unreal_episode_outputs_local_7d_state(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            frames = [
                make_frame(0, 100.0, 200.0),
                make_frame(1, 200.0, 300.0),
            ]
            episode_dir, meta = write_episode(root, frames)
            body_from_camera = validate_fixed_extrinsics(episode_dir, frames, ["front"], 1e-4, 0.1)
            episode = UnrealEpisode(
                episode_dir=episode_dir,
                meta=meta,
                frames=frames,
                camera_keys=["front"],
                task="",
                task_idx=0,
                task_info=[],
                body_from_camera=body_from_camera,
            )

            emitted = [frame for frame, _task in episode]
            self.assertEqual(emitted[0][TASK_DESCRIPTION_KEY].tolist(), [0])
            self.assertEqual(emitted[0]["video.front"].shape, (3, 4, 3))
            np.testing.assert_allclose(emitted[0][STATE_KEY], np.array([0, 0, 0, 0, 0, 0, 1]), atol=1e-6)
            np.testing.assert_allclose(emitted[1][STATE_KEY][:3], np.array([1, 0, 0]), atol=1e-6)
            np.testing.assert_allclose(emitted[1][ACTION_KEY], emitted[1][STATE_KEY], atol=1e-6)

            metadata = episode.metadata
            self.assertEqual(metadata["K_front"], [[100.0, 0.0, 2.0], [0.0, 110.0, 3.0], [0.0, 0.0, 1.0]])
            self.assertIn("Extrinsic_front", metadata)
            self.assertEqual(metadata["fps"], 10)
            self.assertEqual(metadata["capture_width"], 4)
            self.assertEqual(metadata["capture_height"], 3)

    def test_build_features_adds_all_requested_camera_keys(self):
        features = build_features((3, 4), ["front", "rear"])
        self.assertEqual(features["video.front"]["shape"], (3, 4, 3))
        self.assertEqual(features["video.rear"]["shape"], (3, 4, 3))
        self.assertEqual(features[STATE_KEY]["shape"], (7,))

    def test_collection_can_skip_invalid_episode(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            write_episode(root, [make_frame(0, 0.0, 100.0)])

            invalid_dir = root / "scene_0001" / "user_0002" / "episode_000000"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "episode_meta.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "episode_index": 0,
                        "capture_width": 4,
                        "capture_height": 3,
                        "sample_rate_hz": 10,
                        "frame_count": 1,
                        "camera_names": [],
                    }
                ),
                encoding="utf-8",
            )
            (invalid_dir / "frames.jsonl").write_text(json.dumps(make_frame(0, 0.0, 100.0)) + "\n", encoding="utf-8")

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
            )

            self.assertEqual(len(collection), 1)
            self.assertEqual(len(collection.failed_episodes), 1)
            self.assertIn("missing cameras", collection.failed_episodes[0]["error"])

    def test_collection_can_split_mixed_schemas(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            write_episode(root, [make_frame(0, 0.0, 100.0)], user_id="user_0001", fps=10, width=4, height=3)
            write_episode(
                root,
                [make_frame(0, 0.0, 100.0)],
                user_id="user_0002",
                episode_id="episode_000001",
                fps=30,
                width=8,
                height=6,
            )

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                keep_all_schemas=True,
            )
            self.assertEqual(set(collection.schema_groups), {"fps10_3x4", "fps30_6x8"})

            split_collection = collection.for_schema((30, (6, 8)))
            self.assertEqual(len(split_collection), 1)
            self.assertEqual(split_collection.fps, 30)
            self.assertEqual(split_collection.image_size, (6, 8))
            self.assertEqual(len(split_collection.failed_episodes), 0)
            self.assertEqual(len(split_collection.excluded_episodes), 1)
            self.assertEqual(split_collection.excluded_episodes[0]["reason"], "other_schema")

    def test_group_episodes_by_scene_and_schema(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            ep_a, _ = write_episode(root, [make_frame(0, 0.0, 100.0)], scene_id="scene_a", fps=10, width=4, height=3)
            ep_b, _ = write_episode(root, [make_frame(0, 0.0, 100.0)], scene_id="scene_b", fps=30, width=8, height=6)

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                keep_all_schemas=True,
            )

            by_scene = group_episodes_by_scene(collection.schema_valid_episodes)
            self.assertEqual(set(by_scene), {"scene_a", "scene_b"})
            self.assertEqual(by_scene["scene_a"][0][0], ep_a)
            self.assertEqual(by_scene["scene_b"][0][0], ep_b)

            by_schema = group_episodes_by_schema(collection.schema_valid_episodes)
            self.assertEqual(set(by_schema), {(10, (3, 4)), (30, (6, 8))})

    def test_collection_can_trim_one_extra_tail_frame(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            frames = [make_frame(0, 0.0, 100.0), make_frame(1, 100.0, 200.0)]
            write_episode(root, frames, meta_frame_count=1)

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                trim_extra_tail_frame=True,
            )

            self.assertEqual(len(collection), 1)
            self.assertEqual(len(collection.episodes[0][2]), 1)
            self.assertEqual(len(collection.repaired_episodes), 1)
            self.assertEqual(collection.repaired_episodes[0]["action"], "trimmed_extra_tail_frame")

    def test_write_episode_extras_parquet(self):
        if not HAS_PARQUET_ENGINE:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory(prefix="unreal_sidecar_") as tmp:
            root = Path(tmp)
            meta_dir = root / "meta"
            meta_dir.mkdir()
            extras = {
                "episode_index": 0,
                "scene_id": "scene_0001",
                "K_front": [[100.0, 0.0, 2.0], [0.0, 110.0, 3.0], [0.0, 0.0, 1.0]],
                "Extrinsic_front": np.eye(4).tolist(),
                "task_info": [{"name": "go"}],
            }
            (meta_dir / "episodes_extras.jsonl").write_text(json.dumps(extras) + "\n", encoding="utf-8")

            report = write_episode_extras_parquet(root)

            self.assertEqual(report["status"], "completed")
            output_path = root / "episodes_extras.parquet"
            self.assertTrue(output_path.exists())
            df = pd.read_parquet(output_path)
            self.assertEqual(df.loc[0, "episode_index"], 0)
            self.assertEqual(df.loc[0, "scene_id"], "scene_0001")
            self.assertIn("100.0", df.loc[0, "K_front"])

    def test_copy_depth_sidecars(self):
        with tempfile.TemporaryDirectory(prefix="unreal_sidecar_") as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            episode_dir, _ = write_episode(raw_root, [make_frame(0, 0.0, 100.0), make_frame(1, 100.0, 200.0)], write_depth=True)
            meta_dir = root / "meta"
            meta_dir.mkdir()
            extras = {"episode_index": 3, "source_episode_path": str(episode_dir)}
            (meta_dir / "episodes_extras.jsonl").write_text(json.dumps(extras) + "\n", encoding="utf-8")

            report = copy_depth_sidecars(root, ["front"])

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["num_copied_files"], 2)
            depth_dir = root / "images" / "chunk-000" / "observation.depth.front" / "episode_000003"
            self.assertTrue((depth_dir / "00000.png").exists())
            self.assertTrue((depth_dir / "00001.png").exists())


if __name__ == "__main__":
    unittest.main()
