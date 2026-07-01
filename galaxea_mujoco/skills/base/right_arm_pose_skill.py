from __future__ import annotations

from pathlib import Path

from skills.base.arm_pose_skill import R1ProArmPoseSkill


class R1ProRightArmPoseSkill(R1ProArmPoseSkill):
    """Right-arm 6D pose skill."""


def load_skill(path: str | Path | None = None) -> R1ProRightArmPoseSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    skill = R1ProRightArmPoseSkill.from_json(config_path)
    if skill.config.side != "right":
        raise ValueError(f"Expected a right-arm skill config, got side={skill.config.side!r}")
    return skill
