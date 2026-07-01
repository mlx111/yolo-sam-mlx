from __future__ import annotations

from skills.primitives._factory import FunctionBackedUr5eSkill
from skills import recovery_steps


class ApproachObjectSkill(FunctionBackedUr5eSkill):
    def __init__(self) -> None:
        super().__init__("approach_object", recovery_steps.move_grasp)


def load_skill() -> ApproachObjectSkill:
    return ApproachObjectSkill()

