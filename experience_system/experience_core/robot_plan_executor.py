"""Validated robot-plan execution boundary and dry-run executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .schema import utc_now


class SkillExecutor(Protocol):
    def can_execute(self, action: str) -> bool:
        ...

    def execute(self, action: str, parameters: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class DryRunSkillExecutor:
    allowed_skills: set[str]
    executor_name: str = "dry_run_skill_executor"

    def can_execute(self, action: str) -> bool:
        return action in self.allowed_skills

    def execute(self, action: str, parameters: dict[str, Any]) -> dict[str, Any]:
        if not self.can_execute(action):
            return {
                "success": False,
                "status": "unsupported_skill",
                "message": f"unsupported skill: {action}",
            }
        return {
            "success": True,
            "status": "dry_run_pass",
            "message": "validated action is supported by dry-run registry",
            "parameters": dict(parameters),
        }


@dataclass
class RobotPlanExecutionReport:
    schema_version: str = "validated_robot_plan_execution_report_v1"
    plan_id: str = ""
    executor_name: str = ""
    mode: str = "dry_run"
    started_at: str = ""
    finished_at: str = ""
    success: bool = False
    status: str = ""
    step_reports: list[dict[str, Any]] = field(default_factory=list)
    unsupported_actions: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "executor_name": self.executor_name,
            "mode": self.mode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "status": self.status,
            "step_reports": self.step_reports,
            "unsupported_actions": self.unsupported_actions,
            "safety_notes": self.safety_notes,
            "validation": self.validation,
        }


def default_r1pro_skill_registry() -> set[str]:
    return {
        "move_base_relative",
        "set_torso_posture",
        "head_camera_rgbd_save",
        "head_camera_grounded_sam2_pose",
        "plan_cartesian_trajectory",
        "move_to_pregrasp",
        "approach_object",
        "close_gripper",
        "open_gripper",
        "lift",
        "lower_held_object",
        "transport_to_detected_target",
        "frame_alignment_debug",
    }


def validate_robot_plan_for_executor(plan: dict[str, Any], executor: SkillExecutor) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {"schema_status": "fail", "reason": "plan is not a JSON object", "unsupported_actions": []}
    if plan.get("schema_version") != "validated_robot_plan_v1":
        return {"schema_status": "fail", "reason": "schema_version must be validated_robot_plan_v1", "unsupported_actions": []}
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return {"schema_status": "fail", "reason": "steps must be a non-empty list", "unsupported_actions": []}
    unsupported: list[str] = []
    malformed: list[int] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not str(step.get("action") or ""):
            malformed.append(index)
            continue
        action = str(step.get("action"))
        if not executor.can_execute(action):
            unsupported.append(action)
    validation_payload = plan.get("validation") if isinstance(plan.get("validation"), dict) else {}
    sandbox_status = str(validation_payload.get("sandbox_status") or "").lower()
    decision = str(validation_payload.get("decision") or "").lower()
    task_success = bool(validation_payload.get("task_success", False))
    sandbox_gate_status = "pass"
    sandbox_gate_reasons: list[str] = []
    if decision != "accept":
        sandbox_gate_status = "fail"
        sandbox_gate_reasons.append(f"decision={decision or 'missing'}")
    if sandbox_status in {"", "not_run", "block", "reject"}:
        sandbox_gate_status = "fail"
        sandbox_gate_reasons.append(f"sandbox_status={sandbox_status or 'missing'}")
    if not task_success:
        sandbox_gate_status = "fail"
        sandbox_gate_reasons.append("task_success=false")
    return {
        "schema_status": "pass" if not malformed else "fail",
        "executor_status": "pass" if not unsupported and not malformed else "fail",
        "sandbox_gate_status": sandbox_gate_status,
        "sandbox_gate_reasons": sandbox_gate_reasons,
        "step_count": len(steps),
        "malformed_step_indices": malformed,
        "unsupported_actions": sorted(set(unsupported)),
    }


def execute_validated_robot_plan(
    plan: dict[str, Any],
    executor: SkillExecutor,
    *,
    mode: str = "dry_run",
    stop_on_failure: bool = True,
) -> RobotPlanExecutionReport:
    started = utc_now()
    validation = validate_robot_plan_for_executor(plan, executor)
    step_reports: list[dict[str, Any]] = []
    unsupported = list(validation.get("unsupported_actions") or [])
    executor_name = str(getattr(executor, "executor_name", executor.__class__.__name__))

    if (
        validation.get("schema_status") != "pass"
        or validation.get("executor_status") != "pass"
        or validation.get("sandbox_gate_status") != "pass"
    ):
        return RobotPlanExecutionReport(
            plan_id=str(plan.get("plan_id") or "") if isinstance(plan, dict) else "",
            executor_name=executor_name,
            mode=mode,
            started_at=started,
            finished_at=utc_now(),
            success=False,
            status="validation_failed",
            step_reports=[],
            unsupported_actions=unsupported,
            safety_notes=["plan was not sent to robot executor because schema/executor/sandbox gate validation failed"],
            validation=validation,
        )

    success = True
    status = "executed"
    for index, step in enumerate(plan.get("steps") or []):
        action = str(step.get("action") or "")
        parameters = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        result = executor.execute(action, parameters)
        report = {
            "index": index,
            "action": action,
            "stage": str(step.get("stage") or ""),
            "parameters": parameters,
            "success": bool(result.get("success")),
            "status": str(result.get("status") or ""),
            "message": str(result.get("message") or ""),
            "result": result,
        }
        step_reports.append(report)
        if not report["success"]:
            success = False
            status = "step_failed"
            if stop_on_failure:
                break

    return RobotPlanExecutionReport(
        plan_id=str(plan.get("plan_id") or ""),
        executor_name=executor_name,
        mode=mode,
        started_at=started,
        finished_at=utc_now(),
        success=success,
        status=status,
        step_reports=step_reports,
        unsupported_actions=unsupported,
        safety_notes=[
            "dry_run mode validates dispatch only; it does not move the real robot"
            if mode == "dry_run"
            else "real execution mode must be guarded by robot-side safety checks"
        ],
        validation=validation,
    )
