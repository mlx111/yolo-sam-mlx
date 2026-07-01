"""Runtime context passed to UR5e skills.

The current UR5e experiment owns the MuJoCo model/data, robot helpers,
perception state, metrics, and memory hooks.  This context gives the new
Galaxea-style skill layer a stable surface without moving all of that logic
into the skills package.
"""

from __future__ import annotations

from typing import Any


class Ur5eSkillContext:
    """Thin adapter around an ExperimentV4-like object."""

    def __init__(self, experiment: Any, *, default_pregrasp_height: float = 0.127) -> None:
        self.experiment = experiment
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

    def record_basic_skill(self, skill: str, success: bool = True, reason: str = "ok", **extra: Any) -> None:
        if hasattr(self.experiment, "_record_basic_skill"):
            self.experiment._record_basic_skill(skill, success, reason, **extra)

