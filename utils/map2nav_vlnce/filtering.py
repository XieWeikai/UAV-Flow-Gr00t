"""Eligibility checks for single-floor Map2Nav replay episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SourceSchemaError(ValueError):
    """Raised when replay source data cannot satisfy the stable-data contract."""


@dataclass(frozen=True)
class FloorEligibility:
    accepted: bool
    source_level_id: int | None
    visited_levels: tuple[int, ...]
    reason: str | None


def classify_floor_levels(steps: list[dict[str, Any]]) -> FloorEligibility:
    """Accept only episodes whose complete step sequence stays on one floor."""

    if not steps:
        raise SourceSchemaError("steps.jsonl is empty")

    levels: set[int] = set()
    for index, step in enumerate(steps):
        if "floor_level_id" not in step or step["floor_level_id"] is None:
            raise SourceSchemaError(f"step {index} is missing floor_level_id")
        value = step["floor_level_id"]
        if isinstance(value, bool):
            raise SourceSchemaError(f"step {index} has invalid floor_level_id={value!r}")
        try:
            level = int(value)
        except (TypeError, ValueError) as exc:
            raise SourceSchemaError(
                f"step {index} has invalid floor_level_id={value!r}"
            ) from exc
        if isinstance(value, float) and not value.is_integer():
            raise SourceSchemaError(f"step {index} has non-integral floor_level_id={value!r}")
        levels.add(level)

    visited = tuple(sorted(levels))
    if len(visited) == 1:
        return FloorEligibility(True, visited[0], visited, None)
    return FloorEligibility(False, None, visited, "multi_floor")

