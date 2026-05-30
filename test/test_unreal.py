import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from unreal import (
    ACTION_KEY,
    STATE_KEY,
    TASK_DESCRIPTION_KEY,
    UnrealEpisode,
    UnrealEpisodeCollection,
    body_from_camera_for_frame,
    build_features,
    intrinsic_4,
    scan_episode_dirs,
    validate_fixed_extrinsics,
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
    user_id: str = "user_0001",
    episode_id: str = "episode_000000",
    width: int = 4,
    height: int = 3,
    fps: int = 10,
):
    episode_dir = root / "scene_0001" / user_id / episode_id
    (episode_dir / "rgb" / "front").mkdir(parents=True)

    meta = {
        "status": "completed",
        "episode_index": 0,
        "map_name": "Entry",
        "capture_width": width,
        "capture_height": height,
        "sample_rate_hz": fps,
        "frame_count": len(frames),
        "camera_names": ["front"],
    }
    (episode_dir / "episode_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with (episode_dir / "frames.jsonl").open("w", encoding="utf-8") as file:
        for frame in frames:
            file.write(json.dumps(frame) + "\n")

    for index in range(len(frames)):
        image = np.full((height, width, 3), index + 1, dtype=np.uint8)
        Image.fromarray(image).save(episode_dir / "rgb" / "front" / f"{index:05d}.png")

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


if __name__ == "__main__":
    unittest.main()
