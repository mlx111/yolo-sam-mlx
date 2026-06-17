from __future__ import annotations

from pathlib import Path

from skills.primitives.base_navigation_skills import BaseReplanPathSkill


def load_skill(path: str | Path | None = None) -> BaseReplanPathSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return BaseReplanPathSkill(config_path)
