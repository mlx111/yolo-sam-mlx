from __future__ import annotations

from skills.primitives._factory import FunctionBackedUr5eSkill
from skills import recovery_steps


class MoveToPregraspSkill(FunctionBackedUr5eSkill):
    def __init__(self) -> None:
        super().__init__("move_to_pregrasp", recovery_steps.move_pregrasp)


def load_skill() -> MoveToPregraspSkill:
    return MoveToPregraspSkill()

