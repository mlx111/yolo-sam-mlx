from __future__ import annotations

from skills.primitives._factory import FunctionBackedUr5eSkill
from skills import recovery_steps


class MoveLiftedObjectToSkill(FunctionBackedUr5eSkill):
    def __init__(self) -> None:
        super().__init__("move_lifted_object_to", recovery_steps.move_lifted_object_to)


def load_skill() -> MoveLiftedObjectToSkill:
    return MoveLiftedObjectToSkill()
