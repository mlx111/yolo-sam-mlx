"""Write sandbox-validated recovery plan rollouts back into experience memory."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .library import ExperienceLibrary
from .schema import CriticResult, ExperienceEntry, MemoryGate, ObjectState, RobotState, SkillTraceItem, build_retrieval_key, utc_now


RECOVERY_PARAMETER_KEYS = {
    "lateral_offset",
    "forward_offset",
    "yaw_delta",
    "height_level",
    "safe_pregrasp_distance",
    "pregrasp_distance",
    "grasp_offset_z",
    "approach_velocity_limit",
    "approach_segment_count",
    "approach_force_scale",
    "retry_lift_dz",
    "lift_tolerance",
}

INTERNAL_CONTROL_PARAMETER_KEYS = {
    "steps",
    "settle_steps",
    "max_joint_step",
    "fail_threshold",
    "success_threshold",
    "pregrasp_success_threshold",
    "direct_qpos",
    "stabilize",
    "lock_posture",
    "orientation_threshold",
}


def _public_parameters(params: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in params.items()
        if not str(key).startswith("_") and str(key) not in INTERNAL_CONTROL_PARAMETER_KEYS
    }


def extract_plan_recovery_parameters(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return parameterized plan steps that should become future priors."""

    extracted: list[dict[str, Any]] = []
    for index, step in enumerate(plan.get("steps") or []):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or step.get("skill") or "")
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        kept = {str(key): params[key] for key in RECOVERY_PARAMETER_KEYS if key in params}
        if not kept:
            continue
        extracted.append({
            "step_index": index,
            "action": action,
            "parameters": kept,
            "reason": str(step.get("reason") or ""),
        })
    return extracted


def writeback_sandbox_rollout(
    library: ExperienceLibrary,
    *,
    sandbox_report: dict[str, Any],
    recovery_plan: dict[str, Any],
    plan_index: int = -1,
    selected_by_search: bool = False,
    source_tool: str = "",
    merge_duplicates: bool = False,
) -> dict[str, Any]:
    """Persist one sandbox rollout as an experience entry using write policy."""

    raw_entry = sandbox_report.get("experience_entry") if isinstance(sandbox_report.get("experience_entry"), dict) else {}
    if not raw_entry:
        return {
            "decision": "skip",
            "write": False,
            "reason": "sandbox_report_missing_experience_entry",
            "plan_id": str(recovery_plan.get("plan_id") or ""),
            "plan_index": int(plan_index),
        }

    entry = ExperienceEntry(**raw_entry)
    parameter_steps = extract_plan_recovery_parameters(recovery_plan)
    decision = str(sandbox_report.get("decision") or "")
    critic_status = str(sandbox_report.get("critic_status") or "")
    task_success = bool(sandbox_report.get("task_success", sandbox_report.get("success", False)))
    accepted = decision == "accept" and critic_status == "pass" and task_success
    primary_reason = _primary_failure_reason(sandbox_report)
    plan_id = str(recovery_plan.get("plan_id") or sandbox_report.get("candidate_id") or entry.experience_id)

    entry.experience_id = _writeback_id(entry.experience_id, plan_id, plan_index, sandbox_report)
    entry.source = "simulation"
    entry.backend = entry.backend or "mujoco"
    entry.validation_status = "sandbox_validated" if accepted else "sandbox_rejected"
    entry.updated_at = utc_now()
    entry.result.update({
        "success": bool(sandbox_report.get("success", False)),
        "task_success": task_success,
        "sandbox_decision": decision,
        "sandbox_critic_status": critic_status,
        "sandbox_score": float(sandbox_report.get("sandbox_score") or 0.0),
    })
    entry.failure_taxonomy = dict(entry.failure_taxonomy or {})
    if not accepted:
        entry.failure_taxonomy["failure_type"] = primary_reason or entry.failure_taxonomy.get("failure_type") or "sandbox_rejected"

    entry.execution_feedback = dict(entry.execution_feedback or {})
    entry.execution_feedback["recovery_plan"] = _compact_plan(recovery_plan)
    entry.execution_feedback["recovery_parameters"] = parameter_steps
    entry.execution_feedback["sandbox_writeback"] = {
        "source_tool": source_tool,
        "plan_id": plan_id,
        "plan_index": int(plan_index),
        "selected_by_search": bool(selected_by_search),
        "accepted": bool(accepted),
        "decision": decision,
        "critic_status": critic_status,
        "critic_risk_score": float(sandbox_report.get("critic_risk_score") or 0.0),
        "sandbox_score": float(sandbox_report.get("sandbox_score") or 0.0),
        "failure_diagnosis": sandbox_report.get("failure_diagnosis") if isinstance(sandbox_report.get("failure_diagnosis"), dict) else {},
        "critic_flags": list(sandbox_report.get("critic_flags") or []),
        "failed_skills": list(sandbox_report.get("failed_skills") or []),
    }

    entry.memory_tags = dict(entry.memory_tags or {})
    entry.memory_tags["memory_type"] = "sandbox_parameter_experience"
    entry.memory_tags["memory_role"] = "parameter_success_prior" if accepted else "parameter_failure_case"
    entry.memory_tags["sandbox_outcome_type"] = "success" if accepted else "failure"
    entry.memory_tags["parameterized_recovery"] = bool(parameter_steps)
    entry.memory_tags["sandbox_writeback"] = True

    entry.metadata = dict(entry.metadata or {})
    entry.metadata["parameter_search_writeback"] = True
    entry.metadata["sandbox_writeback_source_tool"] = source_tool
    entry.metadata["sandbox_writeback_plan_id"] = plan_id
    entry.metadata["sandbox_writeback_plan_index"] = int(plan_index)
    entry.metadata["sandbox_writeback_selected_by_search"] = bool(selected_by_search)
    entry.metadata["sandbox_writeback_recovery_parameters"] = parameter_steps
    entry.metadata["sandbox_writeback_primary_reason"] = primary_reason
    entry.metadata["sandbox_writeback_outcome_type"] = "success" if accepted else "failure"

    _append_plan_parameter_trace(entry, parameter_steps)
    entry.retrieval_key = build_retrieval_key(entry)
    write_policy = library.add_with_policy(entry, merge_duplicates=merge_duplicates)
    return {
        **write_policy,
        "plan_id": plan_id,
        "plan_index": int(plan_index),
        "selected_by_search": bool(selected_by_search),
        "memory_role": entry.memory_tags.get("memory_role"),
        "sandbox_outcome_type": entry.memory_tags.get("sandbox_outcome_type"),
        "parameter_step_count": len(parameter_steps),
        "primary_failure_reason": primary_reason,
    }


def writeback_sandbox_reports(
    library: ExperienceLibrary,
    candidate_reports: list[dict[str, Any]],
    *,
    selected_plan_index: int = -1,
    source_tool: str = "",
    merge_duplicates: bool = False,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for item in candidate_reports:
        if bool(item.get("sandbox_skipped", False)):
            plan = item.get("recovery_plan") if isinstance(item.get("recovery_plan"), dict) else {}
            sandbox = item.get("sandbox_result") if isinstance(item.get("sandbox_result"), dict) else {}
            validation = (
                item.get("plan_semantic_validation")
                if isinstance(item.get("plan_semantic_validation"), dict)
                else sandbox.get("plan_semantic_validation") if isinstance(sandbox.get("plan_semantic_validation"), dict) else {}
            )
            decisions.append(writeback_semantic_plan_failure(
                library,
                sandbox_report=sandbox,
                recovery_plan=plan,
                semantic_validation=validation,
                plan_index=int(item.get("plan_index", -1)),
                selected_by_search=int(item.get("plan_index", -1)) == int(selected_plan_index),
                source_tool=source_tool,
                merge_duplicates=merge_duplicates,
            ))
            continue
        plan = item.get("recovery_plan") if isinstance(item.get("recovery_plan"), dict) else {}
        sandbox = item.get("sandbox_result") if isinstance(item.get("sandbox_result"), dict) else {}
        decisions.append(writeback_sandbox_rollout(
            library,
            sandbox_report=sandbox,
            recovery_plan=plan,
            plan_index=int(item.get("plan_index", -1)),
            selected_by_search=int(item.get("plan_index", -1)) == int(selected_plan_index),
            source_tool=source_tool,
            merge_duplicates=merge_duplicates,
        ))
    return _summary(decisions)


def writeback_semantic_plan_failure(
    library: ExperienceLibrary,
    *,
    sandbox_report: dict[str, Any],
    recovery_plan: dict[str, Any],
    semantic_validation: dict[str, Any],
    plan_index: int = -1,
    selected_by_search: bool = False,
    source_tool: str = "",
    merge_duplicates: bool = False,
) -> dict[str, Any]:
    """Persist a pre-sandbox semantic validation failure as planner memory."""

    plan_id = str(recovery_plan.get("plan_id") or sandbox_report.get("candidate_id") or "")
    scenario = str(recovery_plan.get("scenario") or sandbox_report.get("scenario") or "")
    condition = str(recovery_plan.get("condition") or sandbox_report.get("condition") or "")
    object_class = _object_class_for_scenario(scenario)
    issues = [item for item in semantic_validation.get("issues") or [] if isinstance(item, dict)]
    primary_issue = issues[0] if issues else {}
    failure_type = str(primary_issue.get("code") or "semantic_validation_failed")
    failed_action = str(primary_issue.get("action") or "")
    missing_requires = list(primary_issue.get("missing_requires") or [])
    steps = [
        step for step in recovery_plan.get("steps") or []
        if isinstance(step, dict)
    ]
    entry = ExperienceEntry(
        experience_id=_semantic_failure_id(plan_id, plan_index, semantic_validation, recovery_plan),
        source="planner",
        backend="semantic_validator",
        validation_status="semantic_rejected",
        robot=RobotState(robot_type="mobile_dual_arm", backend="semantic_validator"),
        scenario={"scenario_id": scenario},
        condition={"condition_id": condition},
        object_state=ObjectState(object_class=object_class),
        task={
            "name": "recovery_plan_semantic_validation",
            "stage": "sandbox_rewrite",
        },
        skill_sequence=[
            SkillTraceItem(
                name=str(step.get("action") or ""),
                primitive_type="llm_recovery_plan_step",
                phase="semantic_validation",
                inputs={"parameters": _public_parameters(step.get("parameters") if isinstance(step.get("parameters"), dict) else {})},
                outputs={"semantic_validation": semantic_validation},
                success=False,
                message=str(step.get("reason") or ""),
                raw={"plan_step_index": index, "plan_step": step},
            )
            for index, step in enumerate(steps)
        ],
        result={
            "success": False,
            "task_success": False,
            "sandbox_skipped": True,
            "semantic_validation_status": str(semantic_validation.get("status") or ""),
            "semantic_fatal_count": int(semantic_validation.get("fatal_count") or 0),
            "semantic_warning_count": int(semantic_validation.get("warning_count") or 0),
            "failed_action": failed_action,
            "missing_requires": missing_requires,
        },
        execution_feedback={
            "recovery_plan": _compact_plan(recovery_plan),
            "semantic_validation": semantic_validation,
            "llm_critic": {
                "failure_type": failure_type,
                "failure_stage": "semantic_validation",
                "root_cause": str(primary_issue.get("message") or "pre-sandbox semantic validation failed"),
                "corrective_direction": "补齐前置条件后再进入 sandbox rollout。",
                "missing_phases": ["precondition repair", "sandbox rerun"],
                "failed_predicates": [str(primary_issue.get("message") or "")] if primary_issue.get("message") else [],
                "memory_lesson": str(primary_issue.get("message") or ""),
                "failure_evidence": {
                    "semantic_validation": semantic_validation,
                    "sandbox_skipped": True,
                },
            },
            "semantic_failure": {
                "failure_type": failure_type,
                "failed_action": failed_action,
                "missing_requires": missing_requires,
                "failed_action_parameters": primary_issue.get("parameters") if isinstance(primary_issue.get("parameters"), dict) else {},
                "memory_lesson": str(primary_issue.get("message") or ""),
                "issues": issues,
            },
            "sandbox_writeback": {
                "source_tool": source_tool,
                "plan_id": plan_id,
                "plan_index": int(plan_index),
                "selected_by_search": bool(selected_by_search),
                "sandbox_skipped": True,
                "reason": "pre_sandbox_semantic_validation_failed",
            },
        },
        memory_tags={
            "memory_type": "semantic_plan_failure",
            "memory_role": "semantic_plan_failure",
            "sandbox_skipped": True,
            "planner_failure": True,
        },
        memory_gate=MemoryGate(
            failure_score=1.0,
            surprise_score=0.6,
            recovery_utility_score=0.7,
            write_score=1.0,
            write_decision="write",
            trigger_events=["semantic_validation_failed", failure_type],
            explanation={
                "failed_action": failed_action,
                "missing_requires": missing_requires,
            },
        ),
        critic_result=CriticResult(
            overall_status="block",
            critic_risk_score=1.0,
            rule_flags=[
                {
                    "rule": failure_type,
                    "stage": "semantic_validation",
                    "severity": "block",
                    "evidence": str(primary_issue.get("message") or "semantic validation failed"),
                    "action": failed_action,
                    "missing_requires": missing_requires,
                }
            ],
            feedback_for_rewrite=str(primary_issue.get("message") or "Fix missing preconditions before sandbox rollout."),
            evidence={
                "semantic_validation": semantic_validation,
                "sandbox_skipped": True,
            },
        ),
        failure_taxonomy={
            "failure_type": failure_type,
            "failed_action": failed_action,
            "missing_requires": missing_requires,
            "failure_stage": "semantic_validation",
            "failure_reason": str(primary_issue.get("message") or "pre-sandbox semantic validation failed"),
        },
        metadata={
            "semantic_plan_failure_writeback": True,
            "sandbox_writeback_source_tool": source_tool,
            "sandbox_writeback_plan_id": plan_id,
            "sandbox_writeback_plan_index": int(plan_index),
            "sandbox_writeback_selected_by_search": bool(selected_by_search),
        },
    )
    entry.retrieval_key = build_retrieval_key(entry)
    write_policy = library.add_with_policy(entry, merge_duplicates=merge_duplicates)
    return {
        **write_policy,
        "plan_id": plan_id,
        "plan_index": int(plan_index),
        "selected_by_search": bool(selected_by_search),
        "memory_role": entry.memory_tags.get("memory_role"),
        "sandbox_outcome_type": "semantic_failure",
        "parameter_step_count": 0,
        "primary_failure_reason": failure_type,
        "sandbox_skipped": True,
        "failed_action": failed_action,
        "missing_requires": missing_requires,
    }


def _append_plan_parameter_trace(entry: ExperienceEntry, parameter_steps: list[dict[str, Any]]) -> None:
    existing = {(item.name, json.dumps(item.raw.get("parameters", {}), sort_keys=True, ensure_ascii=False)) for item in entry.skill_sequence}
    for step in parameter_steps:
        action = str(step.get("action") or "")
        params = _public_parameters(step.get("parameters") if isinstance(step.get("parameters"), dict) else {})
        key = (action, json.dumps(params, sort_keys=True, ensure_ascii=False))
        if not action or key in existing:
            continue
        entry.skill_sequence.append(SkillTraceItem(
            name=action,
            primitive_type="llm_parameterized_recovery",
            phase="sandbox_plan_parameter",
            inputs={"parameters": params},
            outputs={"parameters": params},
            success=bool(entry.result.get("success", entry.result.get("task_success", False))),
            message="LLM-proposed recovery parameters evaluated by sandbox",
            raw={"parameters": params, "plan_step_index": step.get("step_index")},
        ))


def _primary_failure_reason(sandbox_report: dict[str, Any]) -> str:
    diagnosis = sandbox_report.get("failure_diagnosis") if isinstance(sandbox_report.get("failure_diagnosis"), dict) else {}
    return str(
        diagnosis.get("primary_reason")
        or sandbox_report.get("failure_reason")
        or (sandbox_report.get("critic_flags") or [""])[0]
        or ""
    )


def _compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_id": str(plan.get("plan_id") or ""),
        "scenario": str(plan.get("scenario") or ""),
        "condition": str(plan.get("condition") or ""),
        "candidate_steps": list(plan.get("candidate_steps") or []),
        "steps": list(plan.get("steps") or []),
        "confidence": plan.get("confidence"),
        "evidence_ids": list(plan.get("evidence_ids") or []),
    }


def _writeback_id(base_id: str, plan_id: str, plan_index: int, sandbox_report: dict[str, Any]) -> str:
    payload = json.dumps({
        "base_id": base_id,
        "plan_id": plan_id,
        "plan_index": plan_index,
        "decision": sandbox_report.get("decision"),
        "score": sandbox_report.get("sandbox_score"),
        "failure": _primary_failure_reason(sandbox_report),
    }, sort_keys=True, ensure_ascii=False)
    return f"exp_sandbox_param_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _semantic_failure_id(
    plan_id: str,
    plan_index: int,
    semantic_validation: dict[str, Any],
    recovery_plan: dict[str, Any],
) -> str:
    payload = json.dumps({
        "plan_id": plan_id,
        "plan_index": plan_index,
        "issues": semantic_validation.get("issues") or [],
        "steps": recovery_plan.get("steps") or [],
    }, sort_keys=True, ensure_ascii=False)
    return f"exp_semantic_plan_failure_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _object_class_for_scenario(scenario: str) -> str:
    scenario = scenario.upper()
    if scenario == "G3":
        return "sortable_object"
    if scenario == "G4":
        return "large_object"
    return "unknown_object"


def _summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    write_count = sum(1 for item in decisions if bool(item.get("write")))
    merge_count = sum(1 for item in decisions if item.get("decision") == "merge")
    parameter_evidence_count = sum(1 for item in decisions if int(item.get("parameter_step_count") or 0) > 0)
    return {
        "schema_version": "sandbox_writeback_report_v1",
        "attempted_write_count": len(decisions),
        "write_count": write_count,
        "merge_count": merge_count,
        "parameter_evidence_count": parameter_evidence_count,
        "success_prior_write_count": sum(1 for item in decisions if item.get("memory_role") == "parameter_success_prior" and bool(item.get("write"))),
        "failure_case_write_count": sum(1 for item in decisions if item.get("memory_role") == "parameter_failure_case" and bool(item.get("write"))),
        "success_prior_attempt_count": sum(1 for item in decisions if item.get("memory_role") == "parameter_success_prior"),
        "failure_case_attempt_count": sum(1 for item in decisions if item.get("memory_role") == "parameter_failure_case"),
        "semantic_failure_write_count": sum(1 for item in decisions if item.get("memory_role") == "semantic_plan_failure" and bool(item.get("write"))),
        "semantic_failure_attempt_count": sum(1 for item in decisions if item.get("memory_role") == "semantic_plan_failure"),
        "decisions": decisions,
    }
