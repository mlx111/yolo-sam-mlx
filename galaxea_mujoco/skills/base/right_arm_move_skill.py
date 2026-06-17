from __future__ import annotations

from pathlib import Path

from skills.base.arm_move_skill import R1ProArmMoveSkill


class R1ProRightArmMoveSkill(R1ProArmMoveSkill):
    """Right-arm TCP move skill."""


def load_skill(path: str | Path | None = None) -> R1ProRightArmMoveSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    skill = R1ProRightArmMoveSkill.from_json(config_path)
    if skill.config.side != "right":
        raise ValueError(f"Expected a right-arm skill config, got side={skill.config.side!r}")
    return skill
