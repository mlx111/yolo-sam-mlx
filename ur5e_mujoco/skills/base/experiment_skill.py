"""Base class for UR5e skills that execute through ExperimentV4."""

from __future__ import annotations

from typing import Any

from skills.context import Ur5eSkillContext


class Ur5eExperimentSkill:
    name = ""

    def _context(self, runtime: Any, default_pregrasp_height: float = 0.127) -> Ur5eSkillContext:
        if isinstance(runtime, Ur5eSkillContext):
            return runtime
        return Ur5eSkillContext(runtime, default_pregrasp_height=default_pregrasp_height)

