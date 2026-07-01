from __future__ import annotations

from skills import recovery_steps
from skills.primitives._factory import FunctionBackedFR5Skill


class MoveLiftedObjectToSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        super().__init__("move_lifted_object_to", recovery_steps.move_lifted_object_to)


def load_skill() -> MoveLiftedObjectToSkill:
    return MoveLiftedObjectToSkill()
