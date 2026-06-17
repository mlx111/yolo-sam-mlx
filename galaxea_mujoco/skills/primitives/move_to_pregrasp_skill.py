from __future__ import annotations

from pathlib import Path

from skills.primitives.object_manipulation_skills import MoveToPregraspSkill


def load_skill(path: str | Path | None = None) -> MoveToPregraspSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return MoveToPregraspSkill(config_path)
