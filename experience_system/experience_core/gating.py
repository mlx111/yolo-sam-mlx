"""Deterministic write-gating for universal experience entries."""

from __future__ import annotations

from typing import Any


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _decision(write_score: float) -> str:
    if write_score >= 0.75:
        return "high_value"
    if write_score >= 0.50:
        return "ltm_candidate"
    if write_score >= 0.20:
        return "stm_only"
    return "raw_log"


def _gap_score(sim_real_gap: dict[str, Any] | None) -> float:
    gap = sim_real_gap if isinstance(sim_real_gap, dict) else {}
    return _clamp01(gap.get("gap_score") or gap.get("uncertainty") or 0.0)


def _validation_signal(validation_status: str) -> float:
    status = str(validation_status or "")
    if status in {"real_validated"}:
        return 1.0
    if status in {"real_executed", "pseudo_real_executed", "simulation_validated", "sandbox_validated"}:
        return 0.65
    if status in {"failed", "failure"}:
        return 0.50
    return 0.0


def compute_memory_gate(
    metrics: dict[str, Any],
    *,
    recovery_success: bool,
    task_success: bool,
    validation_status: str = "",
    sim_real_gap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return universal write-value signals for one episode."""

    metrics = metrics if isinstance(metrics, dict) else {}
    trigger_events: list[str] = []

    anomaly_score = 0.0
    condition_id = str(metrics.get("condition_id") or "")
    if condition_id and condition_id != "clean":
        anomaly_score = 0.80
        trigger_events.append("condition_injection")
    if metrics.get("anomaly_detected"):
        anomaly_score = max(anomaly_score, 0.85)
        trigger_events.append("anomaly_detected")

    failure_score = 0.0
    if not recovery_success:
        failure_score = 1.0
        trigger_events.append("recovery_failed")
    elif not task_success:
        failure_score = 0.65
        trigger_events.append("task_failed")

    sim_real_gap_score = _gap_score(sim_real_gap)
    if sim_real_gap_score >= 0.67:
        trigger_events.append("sim_real_gap_high")
    elif sim_real_gap_score >= 0.34:
        trigger_events.append("sim_real_gap_medium")

    recovery_utility_score = 0.0
    if recovery_success:
        recovery_utility_score = 0.75
        if task_success:
            recovery_utility_score = 1.0
            trigger_events.append("task_success")
        else:
            trigger_events.append("recovery_success")
    elif metrics.get("skill_trace"):
        recovery_utility_score = 0.45

    surprise_score = 0.0
    if metrics.get("selected_alternate_place"):
        surprise_score = max(surprise_score, 0.45)
        trigger_events.append("alternate_place_selected")
    if metrics.get("attached_object"):
        trigger_events.append("object_attached")
    if metrics.get("critic_warnings"):
        surprise_score = max(surprise_score, 0.70)
        trigger_events.append("critic_warning")

    validation_score = _validation_signal(validation_status)
    if validation_score >= 0.65:
        trigger_events.append("validated_execution")

    weights = {
        "anomaly": 0.22,
        "failure": 0.23,
        "sim_real_gap": 0.22,
        "recovery_utility": 0.15,
        "surprise": 0.08,
        "validation": 0.10,
    }
    write_score = _clamp01(
        weights["anomaly"] * anomaly_score
        + weights["failure"] * failure_score
        + weights["sim_real_gap"] * sim_real_gap_score
        + weights["recovery_utility"] * recovery_utility_score
        + weights["surprise"] * surprise_score
        + weights["validation"] * validation_score
    )

    return {
        "anomaly_score": round(anomaly_score, 4),
        "failure_score": round(failure_score, 4),
        "sim_real_gap_score": round(sim_real_gap_score, 4),
        "recovery_utility_score": round(recovery_utility_score, 4),
        "surprise_score": round(surprise_score, 4),
        "write_score": round(write_score, 4),
        "write_decision": _decision(write_score),
        "trigger_events": trigger_events,
        "explanation": {
            "weights": weights,
            "recovery_success": bool(recovery_success),
            "task_success": bool(task_success),
            "condition_id": condition_id,
            "validation_status": str(validation_status or ""),
            "validation_score": round(validation_score, 4),
        },
    }
