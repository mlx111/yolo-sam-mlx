"""Normalize critic outputs into memory_v3_plus critic_result fields."""

from __future__ import annotations

from typing import Any


_COLLISION_RULES = {
    "unsafe_collision",
    "obstacle_collision",
}
_JOINT_RULES = {
    "joint_limit",
    "unsafe_joint",
}
_GRIPPER_CONTACT_RULES = {
    "no_contact_detected",
    "contact_lost_during_lift",
    "contact_gained_during_lift_unexpected",
    "pinch_too_wide",
    "grasp_not_secured",
    "apple_not_tracked",
}
_END_EFFECTOR_RULES = {
    "object_not_lifted",
    "insufficient_z_change",
    "perception_pos_inconsistency",
    "perception_offset_exceeds_grasp_range",
}


def _rule_names(rule_flags: list[dict[str, Any]]) -> set[str]:
    return {
        str(flag.get("rule") or "")
        for flag in rule_flags
        if isinstance(flag, dict) and flag.get("rule")
    }


def _flag_summary(rule_flags: list[dict[str, Any]], wanted: set[str]) -> dict[str, Any]:
    matched = [
        flag
        for flag in rule_flags
        if isinstance(flag, dict) and str(flag.get("rule") or "") in wanted
    ]
    return {
        "status": "warn" if matched else "pass",
        "flag_count": len(matched),
        "rules": [str(flag.get("rule") or "") for flag in matched],
        "evidence": [
            str(flag.get("evidence") or flag.get("description_cn") or "")[:200]
            for flag in matched[:3]
        ],
    }


def build_critic_result(
    *,
    rule_result: dict[str, Any] | None = None,
    llm_result: dict[str, Any] | None = None,
    is_failure: bool = False,
) -> dict[str, Any]:
    """Merge deterministic and LLM critics into CriticResultInfo-compatible dict."""

    rule_result = rule_result if isinstance(rule_result, dict) else {}
    llm_result = llm_result if isinstance(llm_result, dict) else {}
    rule_flags = [
        flag
        for flag in (rule_result.get("rule_flags") or [])
        if isinstance(flag, dict)
    ]
    names = _rule_names(rule_flags)
    llm_enabled = bool(llm_result.get("enabled")) and not bool(llm_result.get("error"))
    llm_failure_type = str(llm_result.get("failure_type") or "")
    llm_root_cause = str(llm_result.get("root_cause") or "")
    llm_stage = str(llm_result.get("failure_stage") or "")

    if rule_flags:
        overall_status = "warn"
    elif is_failure:
        overall_status = "unknown"
    else:
        overall_status = "pass"

    if names & {"plan_blocked_invalid", "invalid_skill_steps_in_plan", "plan_blocked_by_failed_history"}:
        overall_status = "block"
    if llm_result.get("critic_warning"):
        overall_status = "warn"

    risk_score = 0.0
    if is_failure:
        risk_score += 0.25
    risk_score += min(len(rule_flags) * 0.12, 0.55)
    if llm_enabled and (llm_failure_type or llm_root_cause):
        risk_score += 0.15
    if overall_status == "block":
        risk_score = max(risk_score, 0.8)
    elif overall_status == "warn":
        risk_score = max(risk_score, 0.45)
    risk_score = min(risk_score, 1.0)

    feedback_parts: list[str] = []
    if llm_failure_type:
        feedback_parts.append(f"failure_type={llm_failure_type}")
    if llm_stage:
        feedback_parts.append(f"failure_stage={llm_stage}")
    if llm_root_cause:
        feedback_parts.append(f"root_cause={llm_root_cause}")
    if llm_result.get("corrective_direction"):
        feedback_parts.append(f"corrective_direction={llm_result.get('corrective_direction')}")
    if llm_result.get("missing_phases"):
        feedback_parts.append("missing_phases=" + ", ".join(str(x) for x in llm_result.get("missing_phases") or []))

    return {
        "overall_status": overall_status,
        "critic_risk_score": round(risk_score, 4),
        "collision": _flag_summary(rule_flags, _COLLISION_RULES),
        "joint": _flag_summary(rule_flags, _JOINT_RULES),
        "gripper_contact": _flag_summary(rule_flags, _GRIPPER_CONTACT_RULES),
        "end_effector_pose": _flag_summary(rule_flags, _END_EFFECTOR_RULES),
        "rule_flags": rule_flags,
        "feedback_for_rewrite": "；".join(feedback_parts)[:600],
    }
