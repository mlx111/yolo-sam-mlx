from __future__ import annotations

from pathlib import Path

from skills.primitives.drawer_door_skills import PullDrawerSkill


def load_skill(path: str | Path | None = None) -> PullDrawerSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return PullDrawerSkill(config_path)
