from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class LiftSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("lift", recovery_steps.vertical_grasp)


def load_skill() -> LiftSkill:
    return LiftSkill()
