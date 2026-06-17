from __future__ import annotations

from pathlib import Path

from skills.primitives.whole_body_positioning_skills import PreGraspSafePostureSkill


def load_skill(path: str | Path | None = None) -> PreGraspSafePostureSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return PreGraspSafePostureSkill(config_path)
