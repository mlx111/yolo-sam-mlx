from __future__ import annotations

from pathlib import Path

from skills.base.arm_move_skill import R1ProArmMoveSkill


class R1ProLeftArmMoveSkill(R1ProArmMoveSkill):
    """Left-arm TCP move skill."""


def load_skill(path: str | Path | None = None) -> R1ProLeftArmMoveSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    skill = R1ProLeftArmMoveSkill.from_json(config_path)
    if skill.config.side != "left":
        raise ValueError(f"Expected a left-arm skill config, got side={skill.config.side!r}")
    return skill
