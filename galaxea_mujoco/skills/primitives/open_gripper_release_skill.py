from __future__ import annotations

from pathlib import Path

from skills.primitives.object_manipulation_skills import OpenGripperReleaseSkill


def load_skill(path: str | Path | None = None) -> OpenGripperReleaseSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return OpenGripperReleaseSkill(config_path)
