from __future__ import annotations

from typing import Any


class FR5SkillContext:
    """Thin adapter around an FR5MotionRuntime-like object."""

    def __init__(self, runtime: Any, *, default_pregrasp_height: float = 0.08) -> None:
        self.experiment = runtime
        self.default_pregrasp_height = float(default_pregrasp_height)

    @property
    def model(self) -> Any:
        return self.experiment.model

    @property
    def data(self) -> Any:
        return self.experiment.data

    @property
    def metrics(self) -> dict[str, Any]:
        return self.experiment.metrics

    def record_basic_skill(
        self,
        skill: str,
        success: bool = True,
        reason: str = "ok",
        **extra: Any,
    ) -> None:
        if hasattr(self.experiment, "_record_basic_skill"):
            self.experiment._record_basic_skill(skill, success, reason, **extra)
