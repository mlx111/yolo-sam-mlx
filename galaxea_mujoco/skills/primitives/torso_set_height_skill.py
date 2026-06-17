from __future__ import annotations

from pathlib import Path

from skills.primitives.whole_body_positioning_skills import TorsoSetHeightSkill


def load_skill(path: str | Path | None = None) -> TorsoSetHeightSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return TorsoSetHeightSkill(config_path)
