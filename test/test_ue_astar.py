from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ue_astar import (
    GRID_CELL_KEY,
    MAP_PIXEL_4096_KEY,
    AStarEpisode,
    AStarEpisodeCollection,
    build_instruction_task,
    build_astar_features,
    build_map_frame_fields,
    build_output_groups,
    compute_map_pixel_4096,
    copy_astar_map_sidecars,
    copy_depth_sidecars_limited,
    load_navigation_instruction,
    scan_astar_episode_dirs,
    write_astar_dataset_sidecars,
)
from unreal import ACTION_KEY, STATE_KEY, TASK_DESCRIPTION_KEY, validate_fixed_extrinsics


def make_frame(frame_index, body_x_cm, camera_x_cm, grid_cell=None, valid=True):
    if grid_cell is None:
        grid_cell = {"x": 41, "y": 13, "layer": 0}
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "timestamp_wall_sec": float(frame_index),
        "timestamp_sim_sec": float(frame_index),
        "pose": [body_x_cm, 0.0, 0.0, 0.0, 0.0, 0.0],
        "view_mode": "first_person",
        "camera_pose_front": [camera_x_cm, 0.0, 0.0, 0.0, 0.0, 0.0],
        "K_front": [100.0, 0.0, 2.0, 0.0, 110.0, 3.0, 0.0, 0.0, 1.0],
        "pointnav_grid_cell_valid": valid,
        "pointnav_grid_cell": grid_cell,
        "pointnav_grid_world_cm": [100.0, 200.0, 0.0],
    }


def write_astar_episode(
    root: Path,
    frames: list[dict],
    scene_id: str = "FloorPlan12_physics",
    run_id: str = "20260708T123518Z_seed1337_n1_slot_00",
    episode_id: str = "episode_000000",
    width: int = 4,
    height: int = 3,
    fps: int = 10,
    meta_frame_count: int | None = None,
    write_depth: bool = False,
    graph: dict | None = None,
):
    graph = graph or {"width": 57, "height": 38, "cell_size_cm": 50.0}
    graph_id = "graph_default"
    episode_dir = root / "data" / scene_id / run_id / episode_id
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

    for index in range(meta["frame_count"]):
        image = np.full((height, width, 3), index + 1, dtype=np.uint8)
        Image.fromarray(image).save(episode_dir / "rgb" / "front" / f"{index:05d}.png")
    if write_depth:
        for index in range(len(frames)):
            depth = np.full((height, width), index + 1, dtype=np.uint16)
            Image.fromarray(depth).save(episode_dir / "depth" / "front" / f"{index:05d}.png")

    task_info = "name,start_frame,end_frame\nastar,0,0\n"
    (episode_dir / "task_info.csv").write_text(task_info, encoding="utf-8")

    pointnav_payload = {
        "task": {
            "task_uid": "task-000",
            "path_uid": "path-000",
            "graph_id": graph_id,
            "start_cell": {"x": 1, "y": 2, "layer": 0},
            "goal_cell": {"x": 10, "y": 12, "layer": 0},
            "path_distance_cm": 1234.5,
            "straight_distance_cm": 900.0,
            "turn_count": 3,
        },
        "result": {"status": "success"},
    }
    (episode_dir / "pointnav.json").write_text(json.dumps(pointnav_payload), encoding="utf-8")

    astar_payload = {
        "task_uid": "task-000",
        "path_uid": "path-000",
        "graph_id": graph_id,
        "graph": graph,
        "start_cell": {"x": 1, "y": 2, "layer": 0},
        "goal_cell": {"x": 10, "y": 12, "layer": 0},
        "path_distance_cm": 1234.5,
        "straight_distance_cm": 900.0,
        "turn_count": 3,
        "raw_path_grid": [{"x": 1, "y": 2, "layer": 0}, {"x": 10, "y": 12, "layer": 0}],
        "simplified_path_grid": [{"x": 1, "y": 2, "layer": 0}, {"x": 10, "y": 12, "layer": 0}],
    }
    (episode_dir / "astar_plan_path.json").write_text(json.dumps(astar_payload), encoding="utf-8")

    for filename in ("planned_path_map.png", "actual_path_map.png", "path_comparison_map.png"):
        Image.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8)).save(episode_dir / filename)

    traversability_dir = root / "traversability" / scene_id / graph_id
    traversability_dir.mkdir(parents=True)
    (traversability_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")

    return episode_dir, meta, astar_payload


class UEAStarConversionTests(unittest.TestCase):
    def test_load_navigation_instruction_normalizes_nested_fields(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_instruction_") as tmp:
            episode_dir = Path(tmp)
            payload = {
                "has_quality_issue": False,
                "quality_reason": None,
                "vln": {"instruction": "  Walk to the desk.  "},
                "objectnav": {
                    "instruction": "  Find the chair.  ",
                    "target_category": "  chair  ",
                },
            }
            (episode_dir / "instruction.json").write_text(json.dumps(payload), encoding="utf-8")

            annotation = load_navigation_instruction(episode_dir)

            self.assertEqual(
                annotation,
                {
                    "has_quality_issue": False,
                    "quality_reason": "",
                    "vln_instruction": "Walk to the desk.",
                    "objectnav_instruction": "Find the chair.",
                    "objectnav_target_category": "chair",
                },
            )

    def test_load_navigation_instruction_returns_none_when_file_is_missing(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_instruction_") as tmp:
            self.assertIsNone(load_navigation_instruction(Path(tmp)))

    def test_build_instruction_task_uses_mode_specific_schema(self):
        annotation = {
            "vln_instruction": "Walk to the desk.",
            "objectnav_instruction": "Find the chair.",
            "objectnav_target_category": "chair",
        }

        self.assertEqual(build_instruction_task(annotation, "vln"), "Walk to the desk.")
        objectnav_task = build_instruction_task(annotation, "objectnav")
        self.assertEqual(
            json.loads(objectnav_task),
            {"task": "Find the chair.", "target_category": "chair"},
        )
        self.assertNotIn("vln_instruction", objectnav_task)
        self.assertNotIn("objectnav_instruction", objectnav_task)

    def test_build_instruction_task_returns_none_for_empty_selected_instruction(self):
        annotation = {
            "vln_instruction": "",
            "objectnav_instruction": "",
            "objectnav_target_category": "chair",
        }

        self.assertIsNone(build_instruction_task(annotation, "vln"))
        self.assertIsNone(build_instruction_task(annotation, "objectnav"))

    def test_quality_issue_is_excluded_before_frames_are_loaded(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_instruction_") as tmp:
            root = Path(tmp)
            episode_dir = root / "data" / "scene" / "run" / "episode_000000"
            episode_dir.mkdir(parents=True)
            (episode_dir / "episode_meta.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            (episode_dir / "instruction.json").write_text(
                json.dumps({"has_quality_issue": True, "quality_reason": "blurred"}),
                encoding="utf-8",
            )

            collection = AStarEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                instruction_type="vln",
            )

            self.assertEqual(collection.failed_episodes, [])
            self.assertEqual(len(collection.excluded_episodes), 1)
            self.assertEqual(collection.excluded_episodes[0]["reason"], "instruction_quality_issue")
            self.assertEqual(collection.excluded_episodes[0]["quality_reason"], "blurred")

    def test_instruction_report_includes_mode_and_filter_counts(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_instruction_") as tmp:
            root = Path(tmp)
            episode_dir = root / "data" / "scene" / "run" / "episode_000000"
            episode_dir.mkdir(parents=True)
            (episode_dir / "episode_meta.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )

            collection = AStarEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                instruction_type="objectnav",
            )
            report = collection.build_report(root, "start", "end", "no_valid_episodes")

            self.assertEqual(report["instruction_type"], "objectnav")
            self.assertEqual(report["instruction_filter_counts"], {"missing_instruction_file": 1})

    def test_scan_astar_episode_dirs_reads_data_tree_only(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_episode_") as tmp:
            root = Path(tmp)
            episode_dir, _, _ = write_astar_episode(root, [make_frame(0, 0.0, 100.0)])

            for path in (
                root / "runs" / "bad" / "episode_meta.json",
                root / "tasks" / "bad" / "episode_meta.json",
                root / "traversability" / "bad" / "episode_meta.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

            self.assertEqual(scan_astar_episode_dirs(root), [episode_dir])
            self.assertEqual(scan_astar_episode_dirs(root / "data"), [episode_dir])
            self.assertEqual(scan_astar_episode_dirs(episode_dir), [episode_dir])

    def test_compute_map_pixel_4096_matches_collection_examples(self):
        pixel, size = compute_map_pixel_4096({"x": 41, "y": 13, "layer": 0}, {"width": 57, "height": 38})
        self.assertEqual(pixel, [2982, 970])
        self.assertEqual(size, [4096, 2731])

        pixel, size = compute_map_pixel_4096({"x": 14, "y": 29, "layer": 0}, {"width": 38, "height": 46})
        self.assertEqual(pixel, [1291, 2627])
        self.assertEqual(size, [3384, 4096])

    def test_build_astar_features_adds_map_fields_without_valid_flags(self):
        features = build_astar_features((3, 4), ["front"])

        self.assertEqual(features["video.front"]["shape"], (3, 4, 3))
        self.assertEqual(features[STATE_KEY]["shape"], (7,))
        self.assertEqual(features[GRID_CELL_KEY]["shape"], (3,))
        self.assertEqual(features[MAP_PIXEL_4096_KEY]["shape"], (2,))
        self.assertFalse(any("valid" in key for key in features))

    def test_build_map_frame_fields_uses_placeholders_for_missing_or_invalid_cells(self):
        graph = {"width": 57, "height": 38}
        fields = build_map_frame_fields(make_frame(0, 0.0, 100.0), graph)
        self.assertEqual(fields[GRID_CELL_KEY].tolist(), [41, 13, 0])
        self.assertEqual(fields[MAP_PIXEL_4096_KEY].tolist(), [2982, 970])
        self.assertFalse(any("valid" in key for key in fields))

        fields = build_map_frame_fields(make_frame(1, 0.0, 100.0, grid_cell=None, valid=False), graph)
        self.assertEqual(fields[GRID_CELL_KEY].tolist(), [-1, -1, -1])
        self.assertEqual(fields[MAP_PIXEL_4096_KEY].tolist(), [-1, -1])

    def test_astar_episode_outputs_local_state_and_map_fields(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_episode_") as tmp:
            root = Path(tmp)
            frames = [
                make_frame(0, 100.0, 200.0),
                make_frame(1, 200.0, 300.0, {"x": 42, "y": 13, "layer": 0}),
            ]
            episode_dir, meta, astar_context = write_astar_episode(root, frames)
            body_from_camera = validate_fixed_extrinsics(episode_dir, frames, ["front"], 1e-4, 0.1)
            episode = AStarEpisode(
                episode_dir=episode_dir,
                meta=meta,
                frames=frames,
                camera_keys=["front"],
                task="astar",
                task_idx=0,
                task_info=[],
                body_from_camera=body_from_camera,
                astar_context=astar_context,
            )

            emitted = [frame for frame, _task in episode]
            self.assertEqual(emitted[0][TASK_DESCRIPTION_KEY].tolist(), [0])
            np.testing.assert_allclose(emitted[0][STATE_KEY], np.array([0, 0, 0, 0, 0, 0, 1]), atol=1e-6)
            np.testing.assert_allclose(emitted[1][ACTION_KEY], emitted[1][STATE_KEY], atol=1e-6)
            self.assertEqual(emitted[0][GRID_CELL_KEY].tolist(), [41, 13, 0])
            self.assertEqual(emitted[0][MAP_PIXEL_4096_KEY].tolist(), [2982, 970])

            metadata = episode.metadata
            self.assertEqual(metadata["task.task_uid"], "task-000")
            self.assertEqual(metadata["task.path_uid"], "path-000")
            self.assertEqual(metadata["task.graph_id"], "graph_default")
            self.assertEqual(metadata["astar.turn_count"], 3)
            self.assertEqual(metadata["maps.long_edge_px"], 4096)
            self.assertEqual(metadata["maps.width_4096"], 4096)
            self.assertEqual(metadata["maps.height_4096"], 2731)

            objectnav_task = json.dumps(
                {"task": "Find the chair.", "target_category": "chair"},
                separators=(",", ":"),
                sort_keys=True,
            )
            objectnav_episode = AStarEpisode(
                episode_dir=episode_dir,
                meta=meta,
                frames=frames,
                camera_keys=["front"],
                task=objectnav_task,
                task_idx=0,
                task_info=[],
                body_from_camera=body_from_camera,
                astar_context=astar_context,
                instruction_type="objectnav",
            )
            self.assertEqual(objectnav_episode.metadata["task"], "Find the chair.")
            self.assertNotIn("target_category", objectnav_episode.metadata["task"])

    def test_collection_can_trim_one_extra_tail_frame(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_episode_") as tmp:
            root = Path(tmp)
            frames = [make_frame(0, 0.0, 100.0), make_frame(1, 100.0, 200.0)]
            write_astar_episode(root, frames, meta_frame_count=1)

            collection = AStarEpisodeCollection(
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

    def test_build_output_groups_uses_single_dataset_root_without_scene_subdirs(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_episode_") as tmp:
            root = Path(tmp)
            write_astar_episode(root, [make_frame(0, 0.0, 100.0)], scene_id="FloorPlan12_physics")
            write_astar_episode(root, [make_frame(0, 0.0, 100.0)], scene_id="FloorPlan15_physics")

            collection = AStarEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
            )
            output_dir = root / "output"

            groups = build_output_groups(collection, output_dir, split_by_schema=False)

            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["root"], output_dir)
            self.assertEqual(groups[0]["dataset_name"], output_dir.name)
            self.assertEqual(len(groups[0]["episodes"]), 2)
            self.assertNotIn("FloorPlan12_physics", str(groups[0]["root"]))

    def test_copy_astar_map_sidecars_updates_extras_without_manifest(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_sidecar_") as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            episode_dir, _, _ = write_astar_episode(raw_root, [make_frame(0, 0.0, 100.0)])
            meta_dir = root / "meta"
            meta_dir.mkdir()
            extras = {"episode_index": 3, "source_episode_path": str(episode_dir), "frame_count": 1}
            (meta_dir / "episodes_extras.jsonl").write_text(json.dumps(extras) + "\n", encoding="utf-8")

            report = copy_astar_map_sidecars(root)

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["num_copied_files"], 4)
            self.assertFalse((root / "pointnav_maps_manifest.json").exists())

            expected_png = root / "maps" / "chunk-000" / "planned_path_map" / "episode_000003.png"
            expected_json = root / "maps" / "chunk-000" / "astar_plan" / "episode_000003.json"
            self.assertTrue(expected_png.exists())
            self.assertTrue(expected_json.exists())

            rewritten = [json.loads(line) for line in (meta_dir / "episodes_extras.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rewritten[0]["maps.planned_path_map"], "maps/chunk-000/planned_path_map/episode_000003.png")
            self.assertEqual(rewritten[0]["maps.astar_plan"], "maps/chunk-000/astar_plan/episode_000003.json")

    def test_copy_depth_sidecars_limited_uses_frame_count(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_sidecar_") as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            episode_dir, _, _ = write_astar_episode(
                raw_root,
                [make_frame(0, 0.0, 100.0), make_frame(1, 100.0, 200.0), make_frame(2, 200.0, 300.0)],
                meta_frame_count=2,
                write_depth=True,
            )
            meta_dir = root / "meta"
            meta_dir.mkdir()
            extras = {"episode_index": 3, "source_episode_path": str(episode_dir), "frame_count": 2}
            (meta_dir / "episodes_extras.jsonl").write_text(json.dumps(extras) + "\n", encoding="utf-8")

            report = copy_depth_sidecars_limited(root, ["front"])

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["num_copied_files"], 2)
            depth_dir = root / "images" / "chunk-000" / "observation.depth.front" / "episode_000003"
            self.assertTrue((depth_dir / "00000.png").exists())
            self.assertTrue((depth_dir / "00001.png").exists())
            self.assertFalse((depth_dir / "00002.png").exists())

    def test_write_astar_dataset_sidecars_skips_depth_and_extras_parquet_by_default(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_sidecar_") as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            episode_dir, _, _ = write_astar_episode(raw_root, [make_frame(0, 0.0, 100.0)], write_depth=True)
            meta_dir = root / "meta"
            meta_dir.mkdir()
            extras = {"episode_index": 0, "source_episode_path": str(episode_dir), "frame_count": 1}
            (meta_dir / "episodes_extras.jsonl").write_text(json.dumps(extras) + "\n", encoding="utf-8")

            report = write_astar_dataset_sidecars(root, ["front"])

            self.assertEqual(report["depth_sidecars"]["status"], "skipped")
            self.assertFalse((root / "images").exists())
            self.assertFalse((root / "episodes_extras.parquet").exists())
            self.assertTrue((root / "maps" / "chunk-000" / "planned_path_map" / "episode_000000.png").exists())
            self.assertTrue((root / "meta" / "modality.json").exists())

    def test_write_astar_dataset_sidecars_can_copy_depth_when_requested(self):
        with tempfile.TemporaryDirectory(prefix="ue_astar_sidecar_") as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            episode_dir, _, _ = write_astar_episode(raw_root, [make_frame(0, 0.0, 100.0)], write_depth=True)
            meta_dir = root / "meta"
            meta_dir.mkdir()
            extras = {"episode_index": 0, "source_episode_path": str(episode_dir), "frame_count": 1}
            (meta_dir / "episodes_extras.jsonl").write_text(json.dumps(extras) + "\n", encoding="utf-8")

            report = write_astar_dataset_sidecars(root, ["front"], include_depth=True)

            self.assertEqual(report["depth_sidecars"]["status"], "completed")
            depth_dir = root / "images" / "chunk-000" / "observation.depth.front" / "episode_000000"
            self.assertTrue((depth_dir / "00000.png").exists())


if __name__ == "__main__":
    unittest.main()
