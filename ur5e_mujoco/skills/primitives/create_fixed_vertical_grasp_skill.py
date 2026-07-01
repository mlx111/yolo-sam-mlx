from __future__ import annotations

from typing import Any

from skills.base.experiment_skill import Ur5eExperimentSkill
from skills.context import Ur5eSkillContext
from skills import recovery_steps


class CreateFixedVerticalGraspSkill(Ur5eExperimentSkill):
    name = "create_fixed_vertical_grasp"

    def execute_recovery_action(self, runtime: Ur5eSkillContext | Any, params: dict[str, Any] | None = None) -> Any:
        context = self._context(runtime)
        return recovery_steps.create_grasp(context.experiment, params if isinstance(params, dict) else {}, context.default_pregrasp_height)


def load_skill() -> CreateFixedVerticalGraspSkill:
    return CreateFixedVerticalGraspSkill()

