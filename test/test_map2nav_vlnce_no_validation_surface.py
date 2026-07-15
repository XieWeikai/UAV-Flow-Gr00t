from pathlib import Path


def test_map2nav_vlnce_has_no_post_conversion_validator_surface() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert not (project_root / "validate_map2nav_vlnce.py").exists()
    assert not (project_root / "utils" / "map2nav_vlnce" / "validator.py").exists()
