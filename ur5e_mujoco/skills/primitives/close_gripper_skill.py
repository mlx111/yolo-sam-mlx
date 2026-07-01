from __future__ import annotations

from typing import Any

from skills.base.experiment_skill import Ur5eExperimentSkill
from skills.context import Ur5eSkillContext
from skills import recovery_steps


class CloseGripperSkill(Ur5eExperimentSkill):
    name = "close_gripper"

    def execute_recovery_action(self, runtime: Ur5eSkillContext | Any, params: dict[str, Any] | None = None) -> Any:
        merged = dict(params or {})
        merged["state"] = 1
        return recovery_steps.gripper_action(self._context(runtime).experiment, merged)


def load_skill() -> CloseGripperSkill:
    return CloseGripperSkill()

