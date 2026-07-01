from __future__ import annotations

from skills.primitives._factory import FunctionBackedUr5eSkill
from skills import recovery_steps


class LiftSkill(FunctionBackedUr5eSkill):
    def __init__(self) -> None:
        super().__init__("lift", recovery_steps.vertical_grasp)


def load_skill() -> LiftSkill:
    return LiftSkill()

