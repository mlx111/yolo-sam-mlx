from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class GoHomeSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("go_home", recovery_steps.execute_init)


def load_skill() -> GoHomeSkill:
    return GoHomeSkill()
