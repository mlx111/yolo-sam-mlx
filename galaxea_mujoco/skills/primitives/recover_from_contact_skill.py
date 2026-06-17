from __future__ import annotations

from pathlib import Path

from skills.primitives.recovery_skills import RecoverFromContactSkill


def load_skill(path: str | Path | None = None) -> RecoverFromContactSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return RecoverFromContactSkill(config_path)
