from __future__ import annotations

from pathlib import Path

from skills.primitives.object_manipulation_skills import PlaceObjectSkill


def load_skill(path: str | Path | None = None) -> PlaceObjectSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return PlaceObjectSkill(config_path)
