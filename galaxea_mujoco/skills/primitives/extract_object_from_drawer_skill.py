from __future__ import annotations

from pathlib import Path

from skills.primitives.drawer_door_skills import ExtractObjectFromDrawerSkill


def load_skill(path: str | Path | None = None) -> ExtractObjectFromDrawerSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return ExtractObjectFromDrawerSkill(config_path)
