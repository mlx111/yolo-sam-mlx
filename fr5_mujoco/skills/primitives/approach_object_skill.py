from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class ApproachObjectSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("approach_object", recovery_steps.move_grasp)


def load_skill() -> ApproachObjectSkill:
    return ApproachObjectSkill()
