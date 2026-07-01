"""Real-robot execution boundary for field atomic validated plans.

This module intentionally does not implement any vendor SDK calls. It defines
the adapter interface and a guarded executor boundary so validated plans can be
checked before being handed to a site-specific robot driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .robot_plan_executor import RobotPlanExecutionReport, SkillExecutor, execute_validated_robot_plan
from .schema import GALAXEA_R1PRO_TORSO_SKILLS, canonical_skill_action, utc_now


class FieldAtomicRobotAdapter(Protocol):
    adapter_name: str

    def can_execute(self, action: str) -> bool:
        ...

    def execute(self, action: str, parameters: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class NotConfiguredFieldAtomicRobotAdapter:
    """Safe placeholder that never moves hardware."""

    adapter_name: str = "not_configured_field_atomic_robot_adapter"

    def can_execute(self, action: str) -> bool:
        return canonical_skill_action(action) in GALAXEA_R1PRO_TORSO_SKILLS

    def execute(self, action: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "status": "real_adapter_not_configured",
            "message": (
                "No real robot SDK adapter is configured; hardware command was not sent. "
                "This boundary only accepts Galaxea R1Pro torso atomic skills."
            ),
            "action": action,
            "canonical_action": canonical_skill_action(action),
            "parameters": dict(parameters),
        }


def execute_field_atomic_validated_plan_on_robot(
    plan: dict[str, Any],
    adapter: SkillExecutor | None = None,
    *,
    stop_on_failure: bool = True,
) -> RobotPlanExecutionReport:
    """Execute a validated field atomic plan through a real robot adapter."""

    selected_adapter = adapter or NotConfiguredFieldAtomicRobotAdapter()
    report = execute_validated_robot_plan(
        plan,
        selected_adapter,
        mode="real_robot",
        stop_on_failure=stop_on_failure,
    )
    report.safety_notes.append("field atomic real execution requires a site-specific SDK adapter and external emergency-stop supervision")
    if isinstance(selected_adapter, NotConfiguredFieldAtomicRobotAdapter):
        report.finished_at = utc_now()
        report.success = False
        if report.status == "executed":
            report.status = "real_adapter_not_configured"
    return report
