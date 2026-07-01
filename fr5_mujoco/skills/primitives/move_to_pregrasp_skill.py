from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class MoveToPregraspSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("move_to_pregrasp", recovery_steps.move_pregrasp)


def load_skill() -> MoveToPregraspSkill:
    return MoveToPregraspSkill()
