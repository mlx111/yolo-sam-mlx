from __future__ import annotations

from typing import Any, Callable

from skills.base.experiment_skill import FR5ExperimentSkill
from skills.context import FR5SkillContext


class FunctionBackedFR5Skill(FR5ExperimentSkill):
    def __init__(self, name: str, handler: Callable[[Any, dict[str, Any]], Any], *, default_pregrasp_height: float = 0.08) -> None:
        self.name = name
        self._handler = handler
        self._default_pregrasp_height = float(default_pregrasp_height)

    def execute_recovery_action(self, runtime: FR5SkillContext | Any, params: dict[str, Any] | None = None) -> Any:
        context = self._context(runtime, self._default_pregrasp_height)
        return self._handler(context.experiment, params if isinstance(params, dict) else {})
