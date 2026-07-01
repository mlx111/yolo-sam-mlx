"""Shared helpers for small UR5e primitive skill modules."""

from __future__ import annotations

from typing import Any, Callable

from skills.base.experiment_skill import Ur5eExperimentSkill
from skills.context import Ur5eSkillContext


class FunctionBackedUr5eSkill(Ur5eExperimentSkill):
    def __init__(self, name: str, handler: Callable[[Any, dict[str, Any]], Any], *, default_pregrasp_height: float = 0.127) -> None:
        self.name = name
        self._handler = handler
        self._default_pregrasp_height = float(default_pregrasp_height)

    def execute_recovery_action(self, runtime: Ur5eSkillContext | Any, params: dict[str, Any] | None = None) -> Any:
        context = self._context(runtime, self._default_pregrasp_height)
        return self._handler(context.experiment, params if isinstance(params, dict) else {})

