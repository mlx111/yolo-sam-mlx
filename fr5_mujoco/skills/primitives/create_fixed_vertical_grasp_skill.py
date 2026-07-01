from __future__ import annotations

from typing import Any

from skills import recovery_steps
from skills.context import FR5SkillContext
from skills.primitives._factory import FunctionBackedFR5Skill


class CreateFixedVerticalGraspSkill(FunctionBackedFR5Skill):
    def __init__(self) -> None:
        self.name = "create_fixed_vertical_grasp"
        self._default_pregrasp_height = 0.08

    def execute_recovery_action(self, runtime: FR5SkillContext | Any, params: dict[str, Any] | None = None) -> Any:
        context = self._context(runtime, self._default_pregrasp_height)
        return recovery_steps.create_grasp(context.experiment, params if isinstance(params, dict) else {}, context.default_pregrasp_height)


def load_skill() -> CreateFixedVerticalGraspSkill:
    return CreateFixedVerticalGraspSkill()
