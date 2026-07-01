"""Experience-backed priors for LLM recovery-skill parameter generation."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from .schema import ExperienceEntry


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


def _success(entry: ExperienceEntry) -> bool:
    return bool(entry.result.get("task_success", entry.result.get("success", entry.result.get("task_success", False))))


def _failure_diagnosis(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    metrics = feedback.get("metrics") if isinstance(feedback.get("metrics"), dict) else {}
    diagnosis = metrics.get("failure_diagnosis") if isinstance(metrics.get("failure_diagnosis"), dict) else {}
    return diagnosis


def _candidate_parameters_from_entry(entry: ExperienceEntry) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    def add_params(source: dict[str, Any]) -> None:
        params = {str(key): source[key] for key in RECOVERY_PARAMETER_KEYS if key in source}
        if not params:
            return
        key = tuple(sorted((str(item_key), str(item_value)) for item_key, item_value in params.items()))
        if key in seen:
            return
        seen.add(key)
        values.append(params)

    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    for raw_step in feedback.get("recovery_parameters") or []:
        if not isinstance(raw_step, dict):
            continue
        source = raw_step.get("parameters") if isinstance(raw_step.get("parameters"), dict) else raw_step
        add_params(source)
    for item in entry.skill_sequence:
        raw = item.raw if isinstance(item.raw, dict) else {}
        outputs = item.outputs if isinstance(item.outputs, dict) else {}
        for source in (raw.get("parameters"), raw.get("params"), outputs.get("parameters"), outputs.get("params"), raw, outputs):
            if not isinstance(source, dict):
                continue
            add_params(source)
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    legacy = metadata.get("legacy_task_chain_result") if isinstance(metadata.get("legacy_task_chain_result"), dict) else {}
    for raw_step in legacy.get("skill_trace") or []:
        if not isinstance(raw_step, dict):
            continue
        add_params(raw_step)
    return values


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize_values(items: list[Any]) -> dict[str, Any]:
    nums = [value for value in (_numeric(item) for item in items) if value is not None]
    if nums:
        return {
            "count": len(nums),
            "median": round(float(median(nums)), 6),
            "min": round(float(min(nums)), 6),
            "max": round(float(max(nums)), 6),
        }
    strings = [str(item) for item in items if str(item)]
    if strings:
        counts: dict[str, int] = {}
        for item in strings:
            counts[item] = counts.get(item, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return {"count": len(strings), "top_values": [{"value": key, "count": count} for key, count in ranked[:4]]}
    return {"count": 0}


def _default_parameter_candidates(primary_reason: str) -> list[dict[str, Any]]:
    if primary_reason in {"joint_limit_violation", "actuator_tracking_error"}:
        return [
            {"reposition_base_for_reach": {"lateral_offset": -0.04, "forward_offset": -0.04}},
            {"reposition_base_for_reach": {"lateral_offset": 0.04, "forward_offset": -0.04}},
            {"adjust_torso_for_reach": {"height_level": "mid"}},
            {"move_to_pregrasp": {"pregrasp_distance": 0.04}},
            {"approach_object": {"approach_velocity_limit": 0.2, "approach_segment_count": 10}},
        ]
    if primary_reason in {"contact_lost", "slip_risk_high", "object_not_lifted"}:
        return [
            {"approach_object": {"approach_velocity_limit": 0.18, "approach_segment_count": 12}},
            {"move_to_pregrasp": {"pregrasp_distance": 0.04}},
            {"left_vertical_lift": {"lift_tolerance": 0.04}},
        ]
    return [
        {"reposition_base_for_reach": {"lateral_offset": -0.04, "forward_offset": -0.04}},
        {"adjust_torso_for_reach": {"height_level": "mid"}},
        {"approach_object": {"approach_velocity_limit": 0.2, "approach_segment_count": 10}},
    ]


def build_recovery_parameter_priors(
    entries: list[ExperienceEntry],
    *,
    scenario: str = "",
    condition: str = "",
    primary_reason: str = "",
    limit: int = 8,
) -> dict[str, Any]:
    """Summarize experience-backed parameter priors for planner_input."""

    matched: list[ExperienceEntry] = []
    for entry in entries:
        if scenario and entry.scenario_id and entry.scenario_id != scenario:
            continue
        if condition and entry.condition_id and entry.condition_id != condition:
            continue
        matched.append(entry)

    success_values: dict[str, list[Any]] = defaultdict(list)
    failure_values: dict[str, list[Any]] = defaultdict(list)
    evidence_ids: list[str] = []
    failure_reasons: dict[str, int] = {}
    for entry in matched[: max(limit * 4, limit)]:
        params_list = _candidate_parameters_from_entry(entry)
        if not params_list:
            diagnosis = _failure_diagnosis(entry)
            reason = str(diagnosis.get("primary_reason") or entry.failure_taxonomy.get("failure_type") or "")
            if reason:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue
        evidence_ids.append(entry.experience_id)
        target = success_values if _success(entry) else failure_values
        for params in params_list:
            for key, value in params.items():
                target[key].append(value)

    inferred_reason = primary_reason or (sorted(failure_reasons, key=failure_reasons.get, reverse=True)[0] if failure_reasons else "")
    return {
        "schema_version": "recovery_parameter_priors_v1",
        "matched_entry_count": len(matched),
        "parameter_evidence_count": len(evidence_ids),
        "evidence_ids": evidence_ids[:limit],
        "primary_failure_reason": inferred_reason,
        "recommended_from_success": {key: _summarize_values(values) for key, values in success_values.items()},
        "avoid_from_failure": {key: _summarize_values(values) for key, values in failure_values.items()},
        "default_parameter_candidates": _default_parameter_candidates(inferred_reason),
        "usage": "Use these as bounded priors for recovery-skill parameters; sandbox critic remains the final selector.",
    }
