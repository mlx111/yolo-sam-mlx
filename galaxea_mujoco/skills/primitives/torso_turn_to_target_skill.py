from __future__ import annotations

from pathlib import Path

from skills.primitives.whole_body_positioning_skills import TorsoTurnToTargetSkill


def load_skill(path: str | Path | None = None) -> TorsoTurnToTargetSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return TorsoTurnToTargetSkill(config_path)
