from __future__ import annotations

from skills.primitives._factory import FunctionBackedUr5eSkill
from skills import recovery_steps


class GoHomeSkill(FunctionBackedUr5eSkill):
    def __init__(self) -> None:
        super().__init__("go_home", recovery_steps.execute_init)


def load_skill() -> GoHomeSkill:
    return GoHomeSkill()

