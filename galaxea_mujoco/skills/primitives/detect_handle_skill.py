from __future__ import annotations

from pathlib import Path

from skills.primitives.drawer_door_skills import DetectHandleSkill


def load_skill(path: str | Path | None = None) -> DetectHandleSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return DetectHandleSkill(config_path)
