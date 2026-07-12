from __future__ import annotations

import pytest

from utils.map2nav_vlnce.filtering import SourceSchemaError, classify_floor_levels


def test_single_floor_episode_is_accepted() -> None:
    result = classify_floor_levels([{"floor_level_id": 2}, {"floor_level_id": 2}])

    assert result.accepted is True
    assert result.source_level_id == 2
    assert result.visited_levels == (2,)
    assert result.reason is None


def test_multi_floor_episode_is_a_deterministic_skip() -> None:
    result = classify_floor_levels([{"floor_level_id": 1}, {"floor_level_id": 0}])

    assert result.accepted is False
    assert result.source_level_id is None
    assert result.visited_levels == (0, 1)
    assert result.reason == "multi_floor"


@pytest.mark.parametrize(
    "steps",
    [[], [{"step_index": 0}], [{"floor_level_id": None}]],
)
def test_missing_floor_information_fails_closed(steps: list[dict]) -> None:
    with pytest.raises(SourceSchemaError):
        classify_floor_levels(steps)
