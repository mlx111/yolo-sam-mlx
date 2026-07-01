from __future__ import annotations

from typing import Any

from skills import recovery_steps
from skills.context import FR5SkillContext
from skills.primitives._factory import FunctionBackedFR5Skill


def _open(experiment: Any, params: dict[str, Any]) -> None:
    recovery_steps.gripper_action(experiment, {**params, "state": 0})


class OpenGripperSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("open_gripper", _open)


def load_skill() -> OpenGripperSkill:
    return OpenGripperSkill()
