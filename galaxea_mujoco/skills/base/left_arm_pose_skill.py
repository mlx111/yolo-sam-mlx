from __future__ import annotations

from pathlib import Path

from skills.base.arm_pose_skill import R1ProArmPoseSkill


class R1ProLeftArmPoseSkill(R1ProArmPoseSkill):
    """Left-arm 6D pose skill."""


def load_skill(path: str | Path | None = None) -> R1ProLeftArmPoseSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    skill = R1ProLeftArmPoseSkill.from_json(config_path)
    if skill.config.side != "left":
        raise ValueError(f"Expected a left-arm skill config, got side={skill.config.side!r}")
    return skill
