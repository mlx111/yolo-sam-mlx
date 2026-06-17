"""Rule-based critic for universal robot experience entries."""

from __future__ import annotations

import math
import re
from typing import Any

from .failure_taxonomy import standardize_failure_taxonomy
from .schema import CriticResult, ExperienceEntry, build_retrieval_key


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _bool_contact(value: Any) -> bool:
    if isinstance(value, dict):
        return any(bool(item) for item in value.values())
    return bool(value)


def _message_numbers(message: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)", message):
        numeric = _as_float(value)
        if numeric is not None:
            found[key] = numeric
    return found


def _rule(rule: str, stage: str, severity: str, evidence: str, **extra: Any) -> dict[str, Any]:
    return {
        "rule": rule,
        "stage": stage,
        "severity": severity,
        "evidence": evidence,
        **{key: value for key, value in extra.items() if value not in (None, "", [], {})},
    }


def build_critic_result(
    *,
    rule_result: dict[str, Any] | None = None,
    llm_result: dict[str, Any] | None = None,
    is_failure: bool = False,
) -> dict[str, Any]:
    """Merge external rule/LLM critic outputs into CriticResult-compatible dict."""

    rule_result = rule_result if isinstance(rule_result, dict) else {}
    llm_result = llm_result if isinstance(llm_result, dict) else {}
    rule_flags = [flag for flag in (rule_result.get("rule_flags") or []) if isinstance(flag, dict)]
    names = {str(flag.get("rule") or "") for flag in rule_flags}

    overall_status = "pass"
    if rule_flags:
        overall_status = "warn"
    if is_failure and not rule_flags:
        overall_status = "unknown"
    if names & {
        "plan_blocked_invalid",
        "invalid_skill_steps_in_plan",
        "plan_blocked_by_failed_history",
        "sim_success_real_fail",
        "collision_risk",
        "joint_limit_risk",
    }:
        overall_status = "block"
    if llm_result.get("critic_warning"):
        overall_status = "warn"

    risk = 0.25 if is_failure else 0.0
    risk += min(len(rule_flags) * 0.12, 0.55)
    if llm_result.get("enabled") and not llm_result.get("error") and (
        llm_result.get("failure_type") or llm_result.get("root_cause")
    ):
        risk += 0.15
    if overall_status == "block":
        risk = max(risk, 0.85)
    elif overall_status == "warn":
        risk = max(risk, 0.45)

    feedback_parts: list[str] = []
    for key in ("failure_type", "failure_stage", "root_cause", "corrective_direction"):
        if llm_result.get(key):
            feedback_parts.append(f"{key}={llm_result[key]}")
    if llm_result.get("missing_phases"):
        feedback_parts.append("missing_phases=" + ",".join(str(item) for item in llm_result.get("missing_phases") or []))

    return {
        "overall_status": overall_status,
        "critic_risk_score": round(_clamp01(risk), 4),
        "rule_flags": rule_flags,
        "feedback_for_rewrite": "; ".join(feedback_parts)[:600],
        "evidence": {
            "rule_count": len(rule_flags),
            "external_rule_enabled": bool(rule_result.get("enabled", bool(rule_flags))),
            "llm_enabled": bool(llm_result.get("enabled")),
        },
    }


def critique_experience(entry: ExperienceEntry, *, thresholds: dict[str, float] | None = None) -> CriticResult:
    """Generate a deterministic critic result from one universal experience."""

    thresholds = thresholds or {}
    flags: list[dict[str, Any]] = []
    result_success = bool(entry.result.get("success", entry.result.get("recovery_success", False)))
    metrics = entry.execution_feedback.get("metrics") if isinstance(entry.execution_feedback.get("metrics"), dict) else {}

    flags.extend(_object_lift_flags(entry, metrics, thresholds))
    flags.extend(_place_zone_flags(entry, thresholds))
    flags.extend(_gripper_contact_flags(entry))
    flags.extend(_contact_stability_flags(entry, metrics, thresholds))
    flags.extend(_dual_arm_flags(entry, thresholds))
    flags.extend(_motion_level_flags(entry, metrics, thresholds))
    flags.extend(_safety_flags(entry))
    flags.extend(_sim_real_gap_flags(entry, thresholds))
    flags.extend(_failure_reason_flags(entry))

    raw = build_critic_result(rule_result={"enabled": True, "rule_flags": flags}, is_failure=not result_success)
    return CriticResult(**raw)


def apply_critic(entry: ExperienceEntry, *, thresholds: dict[str, float] | None = None) -> ExperienceEntry:
    """Attach critic_result to entry and rebuild its retrieval key."""

    entry.critic_result = critique_experience(entry, thresholds=thresholds)
    standardize_failure_taxonomy(entry)
    entry.retrieval_key = build_retrieval_key(entry)
    return entry


def _object_lift_flags(entry: ExperienceEntry, metrics: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    min_lift = float(thresholds.get("min_object_lift", 0.05))
    lift = _as_float(metrics.get("object_lift") or entry.execution_feedback.get("object_lift"))
    if lift is None:
        for value in entry.object_state.objects.values():
            if not isinstance(value, dict):
                continue
            start = value.get("start_position") or value.get("start_pos")
            final = value.get("final_position") or value.get("final_pos")
            if isinstance(start, list) and isinstance(final, list) and len(start) >= 3 and len(final) >= 3:
                lift = _as_float(final[2] - start[2])
                break
    if lift is None:
        return []
    if lift < min_lift and any("lift" in item.name for item in entry.skill_sequence):
        return [_rule("object_not_lifted", "lift", "warn", f"object_lift={lift:.4f} < min_object_lift={min_lift:.4f}", value=lift)]
    return []


def _place_zone_flags(entry: ExperienceEntry, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    max_xy = float(thresholds.get("max_place_xy_error", 0.05))
    max_z = float(thresholds.get("max_place_z_error", 0.08))
    flags: list[dict[str, Any]] = []
    for item in entry.skill_sequence:
        if "verify_place" not in item.name:
            continue
        numbers = _message_numbers(item.message)
        xy_error = _as_float(item.outputs.get("xy_error") or numbers.get("xy_error"))
        z_error = _as_float(item.outputs.get("z_error") or numbers.get("z_error"))
        if xy_error is not None and xy_error > max_xy:
            flags.append(_rule("place_xy_error_high", "place", "warn", f"xy_error={xy_error:.4f} > max_place_xy_error={max_xy:.4f}", value=xy_error))
        if z_error is not None and z_error > max_z:
            flags.append(_rule("place_z_error_high", "place", "warn", f"z_error={z_error:.4f} > max_place_z_error={max_z:.4f}", value=z_error))
    if any("place" in item.name for item in entry.skill_sequence) and entry.result.get("success") is False:
        reason = str(entry.result.get("failure_reason") or "")
        if "place" in reason.lower() or "zone" in reason.lower():
            flags.append(_rule("place_zone_miss", "place", "warn", reason or "place-related failure"))
    return flags


def _gripper_contact_flags(entry: ExperienceEntry) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    contact_state = entry.sensor_summary.contact_state or {}
    if contact_state and not _bool_contact(contact_state) and any("gripper" in item.name or "grasp" in item.name for item in entry.skill_sequence):
        flags.append(_rule("gripper_contact_missing", "grasp", "warn", "contact_state has no active contact"))

    feedback = entry.execution_feedback
    close = feedback.get("contact_after_close") if isinstance(feedback.get("contact_after_close"), dict) else {}
    lift = feedback.get("contact_after_lift") if isinstance(feedback.get("contact_after_lift"), dict) else {}
    if close and not _bool_contact(close):
        flags.append(_rule("no_contact_detected", "gripper_close", "warn", "contact_after_close has no active contact"))
    if close and lift and _bool_contact(close) and not _bool_contact(lift):
        flags.append(_rule("contact_lost_during_lift", "lift", "warn", "contact present after close but absent after lift"))
    return flags


def _contact_stability_flags(entry: ExperienceEntry, metrics: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    stability = metrics.get("contact_stability") if isinstance(metrics.get("contact_stability"), dict) else {}
    if not stability:
        stability = entry.execution_feedback.get("contact_stability") if isinstance(entry.execution_feedback.get("contact_stability"), dict) else {}
    if not stability:
        return []

    flags: list[dict[str, Any]] = []
    contact_after_close = bool(stability.get("contact_after_close", False))
    contact_ratio = _as_float(stability.get("contact_during_lift_ratio"))
    contact_lost_step = _as_float(stability.get("contact_lost_step"))
    slip_distance = _as_float(stability.get("object_lift_slip_distance") or stability.get("object_slip_distance"))
    force_proxy = _as_float(stability.get("wrist_force_proxy"))
    grasp_score = _as_float(stability.get("grasp_stability_score"))

    min_contact_ratio = float(thresholds.get("min_contact_during_lift_ratio", 0.5))
    max_slip = float(thresholds.get("max_object_slip_distance", 0.10))
    max_force_proxy = float(thresholds.get("max_wrist_force_proxy_warn", 25.0))
    min_grasp_score = float(thresholds.get("min_grasp_stability_score", 0.45))

    if not contact_after_close:
        flags.append(_rule("contact_missing_after_close", "gripper_close", "warn", "sandbox proxy found no contact after close"))
    if contact_ratio is not None and contact_ratio < min_contact_ratio:
        flags.append(_rule("contact_during_lift_low", "lift", "warn", f"contact_during_lift_ratio={contact_ratio:.4f} < {min_contact_ratio:.4f}", value=contact_ratio))
    if contact_lost_step is not None and contact_lost_step >= 0:
        flags.append(_rule("contact_lost_during_transport", "transport", "warn", f"contact_lost_step={contact_lost_step:.0f}", value=contact_lost_step))
    if slip_distance is not None and slip_distance > max_slip:
        flags.append(_rule("slip_risk_high", "transport", "warn", f"object_lift_slip_distance={slip_distance:.4f} > {max_slip:.4f}", value=slip_distance))
    if force_proxy is not None and force_proxy > max_force_proxy:
        flags.append(_rule("wrist_force_proxy_high", "contact", "warn", f"wrist_force_proxy={force_proxy:.4f} > {max_force_proxy:.4f}", value=force_proxy))
    if grasp_score is not None and grasp_score < min_grasp_score:
        flags.append(_rule("grasp_stability_low", "grasp", "warn", f"grasp_stability_score={grasp_score:.4f} < {min_grasp_score:.4f}", value=grasp_score))
    return flags


def _dual_arm_flags(entry: ExperienceEntry, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    max_height = float(thresholds.get("max_dual_arm_height_mismatch", 0.02))
    candidates = [
        entry.execution_feedback.get("dual_arm_height_mismatch"),
        entry.execution_feedback.get("height_mismatch"),
    ]
    for item in entry.skill_sequence:
        if "level" in item.name or "dual_arm" in item.name:
            candidates.append(item.outputs.get("height_mismatch"))
            candidates.append(_message_numbers(item.message).get("height_mismatch"))
    for candidate in candidates:
        value = _as_float(candidate)
        if value is not None and value > max_height:
            return [_rule("dual_arm_height_mismatch", "dual_arm", "warn", f"height_mismatch={value:.4f} > max={max_height:.4f}", value=value)]
    return []


def _motion_level_flags(entry: ExperienceEntry, metrics: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    motion = metrics.get("motion_critic") if isinstance(metrics.get("motion_critic"), dict) else {}
    if not motion:
        return []

    flags: list[dict[str, Any]] = []
    max_joint_delta = _as_float(motion.get("max_joint_delta"))
    max_joint_speed = _as_float(motion.get("max_joint_speed_proxy"))
    min_joint_margin = _as_float(motion.get("min_joint_limit_margin"))
    max_contact_delta = _as_float(motion.get("max_contact_count_delta"))
    max_tcp_delta = _as_float(motion.get("max_tcp_delta"))
    max_workspace_radius = _as_float(motion.get("max_workspace_radius"))

    joint_delta_warn = float(thresholds.get("max_joint_delta_warn", 1.5))
    joint_speed_warn = float(thresholds.get("max_joint_speed_warn", 1.5))
    joint_margin_block = float(thresholds.get("min_joint_limit_margin_block", -0.001))
    contact_delta_warn = float(thresholds.get("max_contact_count_delta_warn", 100.0))
    tcp_delta_warn = float(thresholds.get("max_tcp_delta_warn", 0.7))
    workspace_radius_warn = float(thresholds.get("max_workspace_radius_warn", 1.05))
    min_transport_segments = int(thresholds.get("min_transport_segments", 2))

    if min_joint_margin is not None and min_joint_margin < joint_margin_block:
        flags.append(_rule("joint_limit_risk", "motion", "block", f"min_joint_limit_margin={min_joint_margin:.4f} < {joint_margin_block:.4f}", value=min_joint_margin))
    if max_joint_speed is not None and max_joint_speed > joint_speed_warn:
        flags.append(_rule("joint_speed_risk", "motion", "warn", f"max_joint_speed_proxy={max_joint_speed:.4f} > {joint_speed_warn:.4f}", value=max_joint_speed))
    if max_joint_delta is not None and max_joint_delta > joint_delta_warn:
        flags.append(_rule("joint_step_risk", "motion", "warn", f"max_joint_delta={max_joint_delta:.4f} > {joint_delta_warn:.4f}", value=max_joint_delta))
    if max_contact_delta is not None and max_contact_delta > contact_delta_warn:
        flags.append(_rule("collision_risk", "motion", "block", f"max_contact_count_delta={max_contact_delta:.0f} > {contact_delta_warn:.0f}", value=max_contact_delta))
    if max_tcp_delta is not None and max_tcp_delta > tcp_delta_warn:
        flags.append(_rule("end_effector_pose_risk", "motion", "warn", f"max_tcp_delta={max_tcp_delta:.4f} > {tcp_delta_warn:.4f}", value=max_tcp_delta))
    if max_workspace_radius is not None and max_workspace_radius > workspace_radius_warn:
        flags.append(_rule("workspace_usage_risk", "motion", "warn", f"max_workspace_radius={max_workspace_radius:.4f} > {workspace_radius_warn:.4f}", value=max_workspace_radius))
    for item in entry.skill_sequence:
        if "segmented_transport" not in item.name:
            continue
        segment_count = _as_float(item.outputs.get("segment_count"))
        if segment_count is not None and segment_count < min_transport_segments:
            flags.append(_rule("joint_speed_risk", item.name, "warn", f"segment_count={segment_count:.0f} < min_transport_segments={min_transport_segments}", value=segment_count))
    return flags


def _safety_flags(entry: ExperienceEntry) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    values = [entry.execution_feedback, entry.spatial_state, entry.result]
    for source in values:
        if not isinstance(source, dict):
            continue
        if source.get("collision") or source.get("collision_detected") or source.get("collision_risk"):
            flags.append(_rule("collision_risk", "safety", "block", "collision risk/detection flag present"))
        if source.get("joint_limit") or source.get("joint_limit_violation") or source.get("joint_limit_risk"):
            flags.append(_rule("joint_limit_risk", "safety", "block", "joint limit risk/detection flag present"))
    for item in entry.skill_sequence:
        for safety_flag in item.safety_flags:
            if not isinstance(safety_flag, dict):
                continue
            name = str(safety_flag.get("rule") or safety_flag.get("type") or safety_flag.get("name") or "")
            if "collision" in name:
                flags.append(_rule("collision_risk", item.name, "block", str(safety_flag)))
            if "joint" in name and "limit" in name:
                flags.append(_rule("joint_limit_risk", item.name, "block", str(safety_flag)))
    return flags


def _sim_real_gap_flags(entry: ExperienceEntry, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    high_gap = float(thresholds.get("high_sim_real_gap", 0.65))
    flags: list[dict[str, Any]] = []
    if entry.sim_real_gap.gap_score >= high_gap:
        flags.append(_rule("sim_real_gap_high", "sim_real", "warn", f"gap_score={entry.sim_real_gap.gap_score:.4f} >= {high_gap:.4f}", value=entry.sim_real_gap.gap_score))
    outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
    if outcome_type == "sim_success_real_fail":
        flags.append(_rule("sim_success_real_fail", "sim_real", "block", "simulation success paired with real failure"))
    return flags


def _failure_reason_flags(entry: ExperienceEntry) -> list[dict[str, Any]]:
    reason = str(entry.result.get("failure_reason") or "").lower()
    if not reason:
        return []
    if "gripper" in reason or "grasp" in reason:
        return [_rule("gripper_failure_reason", "grasp", "warn", str(entry.result.get("failure_reason") or ""))]
    if "collision" in reason:
        return [_rule("collision_risk", "safety", "block", str(entry.result.get("failure_reason") or ""))]
    if "joint" in reason and "limit" in reason:
        return [_rule("joint_limit_risk", "safety", "block", str(entry.result.get("failure_reason") or ""))]
    return [_rule("task_failure_reason", "task", "warn", str(entry.result.get("failure_reason") or ""))]
