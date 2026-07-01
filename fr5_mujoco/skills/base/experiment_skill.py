from __future__ import annotations

from typing import Any

from skills.context import FR5SkillContext


class FR5ExperimentSkill:
    name: str

    def _context(self, runtime: FR5SkillContext | Any, default_pregrasp_height: float = 0.08) -> FR5SkillContext:
        return runtime if isinstance(runtime, FR5SkillContext) else FR5SkillContext(runtime, default_pregrasp_height=default_pregrasp_height)

    def execute_recovery_action(self, runtime: FR5SkillContext | Any, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError
