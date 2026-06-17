from __future__ import annotations

from pathlib import Path

from skills.primitives.drawer_door_skills import RedetectHandleSkill


def load_skill(path: str | Path | None = None) -> RedetectHandleSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return RedetectHandleSkill(config_path)
