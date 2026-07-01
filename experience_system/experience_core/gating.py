"""Experience write-gating signals for dual-source memory.

The gate is
deterministic and side-effect free; callers still decide whether to save.
"""

from __future__ import annotations

from typing import Any


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _has_contact(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(value.get("left_contact") or value.get("right_contact") or value.get("contact"))


def _event(trigger_events: list[str], name: str) -> None:
    if name not in trigger_events:
        trigger_events.append(name)


def _score_from_numeric_gap(sim_real_gap: dict[str, Any] | None) -> float:
    if not isinstance(sim_real_gap, dict):
        return 0.0
    for key in ("gap_score", "uncertainty"):
        if key in sim_real_gap:
            return _clamp01(sim_real_gap.get(key))
    return 0.0


def _write_decision(write_score: float) -> str:
    if write_score >= 0.75:
        return "high_value"
    if write_score >= 0.50:
        return "ltm_candidate"
    if write_score >= 0.20:
        return "stm_only"
    return "raw_log"


def compute_memory_gate(
    metrics: dict[str, Any],
    *,
    task_success: bool | None = None,
    validation_status: str = "",
    sim_real_gap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute deterministic write-value signals for one episode."""

    metrics = metrics if isinstance(metrics, dict) else {}
    trigger_events: list[str] = []

    if task_success is None and metrics.get("task_success") is not None:
        task_success = bool(metrics.get("task_success"))

    condition_injection = metrics.get("condition_injection") or {}
    anomaly_detected = bool(metrics.get("anomaly_detected"))
    if anomaly_detected:
        _event(trigger_events, "anomaly_detected")
    if condition_injection:
        _event(trigger_events, "condition_injection")

    anomaly_score = 0.0
    if anomaly_detected:
        anomaly_score = 0.75
    if condition_injection:
        anomaly_score = max(anomaly_score, 0.85)
    if metrics.get("condition_id") or metrics.get("failure_family"):
        anomaly_score = max(anomaly_score, 0.35)

    recovery_blocked = bool(metrics.get("recovery_blocked_by_invalid_plan"))
    invalid_plan_count = int(metrics.get("invalid_plan_count") or 0)
    if recovery_blocked:
        _event(trigger_events, "recovery_blocked_by_invalid_plan")
    if invalid_plan_count > 0:
        _event(trigger_events, "invalid_plan")

    virtual_validation_success = metrics.get("virtual_validation_success")
    if virtual_validation_success is False:
        _event(trigger_events, "virtual_validation_failed")
    elif virtual_validation_success is True:
        _event(trigger_events, "virtual_validation_passed")

    failure_score = 0.0
    if task_success is False:
        failure_score = 0.65
        _event(trigger_events, "task_failed")
    elif recovery_blocked or virtual_validation_success is False or invalid_plan_count > 0:
        failure_score = 0.75
    elif validation_status in {"failed", "failure"}:
        failure_score = 0.8

    if metrics.get("failed_plan_blocker_matches"):
        _event(trigger_events, "failed_plan_blocker_match")
        failure_score = max(failure_score, 0.7)
    if metrics.get("repeated_failure_detected"):
        _event(trigger_events, "repeated_failure_pattern")
        failure_score = max(failure_score, 0.7)

    sim_real_gap_score = _score_from_numeric_gap(sim_real_gap)
    if sim_real_gap_score >= 0.67:
        _event(trigger_events, "sim_real_gap_high")
    elif sim_real_gap_score >= 0.34:
        _event(trigger_events, "sim_real_gap_medium")

    executed_steps = metrics.get("executed_recovery_steps") or []
    llm_steps = metrics.get("llm_recovery_steps") or []
    recovery_plan = metrics.get("recovery_plan") or {}
    if isinstance(recovery_plan, dict):
        plan_steps = recovery_plan.get("steps") or []
    else:
        plan_steps = []
    step_count = len(executed_steps or llm_steps or plan_steps or [])
    task_criteria = metrics.get("task_success_criteria") or {}
    has_structured_evidence = bool(task_criteria or metrics.get("skill_results"))

    if task_success is True:
        _event(trigger_events, "task_success")
        recovery_utility_score = 0.70
        if virtual_validation_success is True:
            recovery_utility_score = 0.85
        recovery_utility_score = 1.0
        if step_count:
            recovery_utility_score = max(recovery_utility_score, 0.75)
    elif failure_score > 0.0 and has_structured_evidence:
        recovery_utility_score = 0.45
    elif step_count:
        recovery_utility_score = 0.30
    else:
        recovery_utility_score = 0.0

    surprise_score = 0.0
    if recovery_blocked or invalid_plan_count > 0:
        surprise_score = max(surprise_score, 0.85)
    if virtual_validation_success is False:
        surprise_score = max(surprise_score, 0.75)
    if metrics.get("failed_plan_blocker_matches") or metrics.get("repeated_failure_detected"):
        surprise_score = max(surprise_score, 0.70)
    if anomaly_detected and task_success is False:
        surprise_score = max(surprise_score, 0.60)

    contact_close = metrics.get("contact_after_close") or {}
    contact_lift = metrics.get("contact_after_lift") or {}
    if _has_contact(contact_close) != _has_contact(contact_lift):
        _event(trigger_events, "contact_state_changed")
        surprise_score = max(surprise_score, 0.50)

    z_change = task_criteria.get("z_change") if isinstance(task_criteria, dict) else None
    try:
        if abs(float(z_change)) >= 0.05:
            _event(trigger_events, "large_z_change")
            surprise_score = max(surprise_score, 0.35)
    except (TypeError, ValueError):
        pass

    weights = {
        "anomaly": 0.25,
        "failure": 0.25,
        "sim_real_gap": 0.25,
        "recovery_utility": 0.15,
        "surprise": 0.10,
    }
    write_score = _clamp01(
        weights["anomaly"] * anomaly_score
        + weights["failure"] * failure_score
        + weights["sim_real_gap"] * sim_real_gap_score
        + weights["recovery_utility"] * recovery_utility_score
        + weights["surprise"] * surprise_score
    )

    return {
        "anomaly_score": round(anomaly_score, 4),
        "failure_score": round(failure_score, 4),
        "sim_real_gap_score": round(sim_real_gap_score, 4),
        "recovery_utility_score": round(recovery_utility_score, 4),
        "surprise_score": round(surprise_score, 4),
        "write_score": round(write_score, 4),
        "write_decision": _write_decision(write_score),
        "trigger_events": trigger_events,
        "explanation": {
            "weights": weights,
            "validation_status": validation_status,
            "step_count": step_count,
            "has_structured_evidence": has_structured_evidence,
            "task_success": task_success,
        },
    }
