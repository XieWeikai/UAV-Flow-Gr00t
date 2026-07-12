from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from test.map2nav_vlnce_testdata import (
    create_replay_source,
    read_jsonl,
    write_json,
    write_jsonl,
)
from utils.map2nav_vlnce import convert_dataset
from utils.map2nav_vlnce.validator import DatasetValidationError, validate_split


def test_validator_checks_complete_source_to_output_contract(tmp_path: Path) -> None:
    source = create_replay_source(tmp_path / "source")
    dataset = convert_dataset(source, tmp_path / "processed" / "r2r", "r2r", "train")

    report = validate_split(
        input_root=source,
        dataset_root=dataset,
        dataset_name="r2r",
        split="train",
        hash_sample_size=2,
        decode_video_sample_size=2,
    )

    assert report["status"] == "passed"
    assert report["source_manifest_total"] == 3
    assert report["accepted_episodes"] == 2
    assert report["skipped_multi_floor"] == 1
    assert report["validated_frames"] == 4
    assert report["validated_videos"] == 8
    assert report["validated_maps"] == 12
    assert len(report["sha256_samples"]) == 20
    assert report["max_projection_error_px"] == 0


def test_validator_rejects_nonempty_phase_one_cot(tmp_path: Path) -> None:
    source = create_replay_source(tmp_path / "source")
    dataset = convert_dataset(
        source, tmp_path / "processed" / "r2r", "r2r", "train", max_episodes=1
    )
    parquet = dataset / "data" / "chunk-000" / "episode_000000.parquet"
    frame = pd.read_parquet(parquet)
    frame.loc[0, "extra.cot"] = "not phase-one data"
    frame.to_parquet(parquet, index=False)

    with pytest.raises(DatasetValidationError, match="extra.cot"):
        validate_split(
            input_root=source,
            dataset_root=dataset,
            dataset_name="r2r",
            split="train",
            hash_sample_size=0,
            decode_video_sample_size=0,
        )


def test_validator_rejects_corrupt_episode_stats(tmp_path: Path) -> None:
    source = create_replay_source(tmp_path / "source")
    dataset = convert_dataset(
        source, tmp_path / "processed" / "r2r", "r2r", "train", max_episodes=1
    )
    stats_path = dataset / "meta" / "episodes_stats.jsonl"
    rows = read_jsonl(stats_path)
    rows[0]["stats"]["observation.state"]["count"] = [999]
    write_jsonl(stats_path, rows)

    with pytest.raises(DatasetValidationError, match="episode stats"):
        validate_split(
            input_root=source,
            dataset_root=dataset,
            dataset_name="r2r",
            split="train",
            hash_sample_size=0,
            decode_video_sample_size=0,
        )


def test_validator_rejects_info_feature_drift(tmp_path: Path) -> None:
    source = create_replay_source(tmp_path / "source")
    dataset = convert_dataset(
        source, tmp_path / "processed" / "r2r", "r2r", "train", max_episodes=1
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["extra.cot"]["dtype"] = "int32"
    write_json(info_path, info)

    with pytest.raises(DatasetValidationError, match="feature schema"):
        validate_split(
            input_root=source,
            dataset_root=dataset,
            dataset_name="r2r",
            split="train",
            hash_sample_size=0,
            decode_video_sample_size=0,
        )
