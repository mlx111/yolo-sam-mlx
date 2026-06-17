from __future__ import annotations

from pathlib import Path

from skills.primitives.perception_skills import ClassifyTargetObjectSkill


def load_skill(path: str | Path | None = None) -> ClassifyTargetObjectSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return ClassifyTargetObjectSkill(config_path)
