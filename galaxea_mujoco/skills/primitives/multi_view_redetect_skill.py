from __future__ import annotations

from pathlib import Path

from skills.primitives.perception_skills import MultiViewRedetectSkill


def load_skill(path: str | Path | None = None) -> MultiViewRedetectSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return MultiViewRedetectSkill(config_path)
