"""Sandbox calibration from universal sim-real gap memories."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from .schema import ExperienceEntry, SandboxCalibration, build_retrieval_key, utc_now


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(float(value), hi))


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _vector(value: Any) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def _stable_id(group_key: tuple[str, str, str, str], source_gap_ids: list[str]) -> str:
    payload = "|".join(group_key) + "|" + "|".join(sorted(source_gap_ids))
    return "cal_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def calibration_group_key(entry: ExperienceEntry) -> tuple[str, str, str, str]:
    return (
        entry.robot.robot_type,
        entry.scenario_id,
        entry.condition_id,
        entry.object_state.object_class,
    )


def group_gap_entries(entries: list[ExperienceEntry]) -> dict[tuple[str, str, str, str], list[ExperienceEntry]]:
    groups: dict[tuple[str, str, str, str], list[ExperienceEntry]] = {}
    for entry in entries:
        if entry.sim_real_pair.get("validation_status") != "paired":
            continue
        if not entry.sim_real_gap.gap_id:
            continue
        groups.setdefault(calibration_group_key(entry), []).append(entry)
    return groups


def _gap_weight(entry: ExperienceEntry) -> float:
    gap_score = _clamp(entry.sim_real_gap.gap_score, 0.0, 1.0)
    uncertainty = _clamp(entry.sim_real_gap.uncertainty, 0.0, 1.0)
    pair_score = _clamp(_numeric(entry.sim_real_pair.get("pair_score"), 0.5), 0.0, 1.0)
    return _clamp(0.50 * gap_score + 0.30 * (1.0 - uncertainty) + 0.20 * pair_score, 0.0, 1.0)


def _pose_delta(entry: ExperienceEntry) -> list[float] | None:
    pose_gap = entry.sim_real_gap.pose_gap if isinstance(entry.sim_real_gap.pose_gap, dict) else {}
    sim_pos = _vector(pose_gap.get("sim_object_position") or pose_gap.get("sim_observed_pos"))
    real_pos = _vector(pose_gap.get("real_object_position") or pose_gap.get("real_observed_pos"))
    if not sim_pos or not real_pos:
        return None
    n = min(3, len(sim_pos), len(real_pos))
    if n <= 0:
        return None
    delta = [real_pos[i] - sim_pos[i] for i in range(n)]
    while len(delta) < 3:
        delta.append(0.0)
    return delta


def compute_sandbox_calibration(group_entries: list[ExperienceEntry], *, group_key: tuple[str, str, str, str] | None = None) -> SandboxCalibration:
    source_gap_ids: list[str] = []
    evidence: list[dict[str, Any]] = []
    weighted_pose = [0.0, 0.0, 0.0]
    total_pose_weight = 0.0
    total_weight = 0.0
    contact_mismatch_weight = 0.0
    sim_success_real_fail_weight = 0.0
    matched_success_weight = 0.0
    pose_error_weighted = 0.0
    seen_gap_ids: set[str] = set()

    for entry in group_entries:
        if not entry.sim_real_gap.gap_id:
            continue
        if entry.sim_real_gap.gap_id in seen_gap_ids:
            continue
        seen_gap_ids.add(entry.sim_real_gap.gap_id)
        weight = _gap_weight(entry)
        if weight <= 0.0:
            continue
        total_weight += weight
        source_gap_ids.append(entry.sim_real_gap.gap_id)

        delta = _pose_delta(entry)
        if delta is not None:
            for index in range(3):
                weighted_pose[index] += delta[index] * weight
            total_pose_weight += weight

        pose_gap = entry.sim_real_gap.pose_gap if isinstance(entry.sim_real_gap.pose_gap, dict) else {}
        pose_error = _numeric(pose_gap.get("object_pose_error"), 0.0)
        pose_error_weighted += pose_error * weight

        contact_gap = entry.sim_real_gap.contact_gap if isinstance(entry.sim_real_gap.contact_gap, dict) else {}
        if contact_gap.get("contact_mismatch"):
            contact_mismatch_weight += weight

        outcome_type = str((entry.sim_real_gap.outcome_gap or {}).get("type") or "")
        if outcome_type == "sim_success_real_fail":
            sim_success_real_fail_weight += weight
        elif outcome_type == "matched_success":
            matched_success_weight += weight

        evidence.append({
            "experience_id": entry.experience_id,
            "source": entry.source,
            "gap_id": entry.sim_real_gap.gap_id,
            "gap_score": entry.sim_real_gap.gap_score,
            "uncertainty": entry.sim_real_gap.uncertainty,
            "pair_score": entry.sim_real_pair.get("pair_score", 0.0),
            "weight": round(weight, 4),
            "outcome_gap_type": outcome_type,
            "contact_mismatch": bool(contact_gap.get("contact_mismatch")),
            "object_pose_error": pose_error,
        })

    if not source_gap_ids or total_weight <= 0.0:
        return SandboxCalibration(details={"reason": "no_gap_memory"})

    pose_bias = [x / total_pose_weight for x in weighted_pose] if total_pose_weight > 0.0 else [0.0, 0.0, 0.0]
    pose_bias = [_clamp(value, -0.04, 0.04) for value in pose_bias]
    avg_pose_error = pose_error_weighted / total_weight
    perception_noise_bias = [_clamp(avg_pose_error, 0.0, 0.08)] * 3

    contact_mismatch_rate = contact_mismatch_weight / total_weight
    sim_fail_rate = sim_success_real_fail_weight / total_weight
    matched_success_rate = matched_success_weight / total_weight
    contact_success_bias = _clamp(0.20 * matched_success_rate - 0.80 * contact_mismatch_rate - 0.70 * sim_fail_rate, -1.0, 1.0)
    slip_risk_bias = _clamp(0.70 * sim_fail_rate + 0.50 * contact_mismatch_rate, 0.0, 1.0)
    confidence = _clamp((total_weight / max(len(evidence), 1)) * min(len(source_gap_ids), 5) / 5.0, 0.0, 1.0)
    key = group_key or calibration_group_key(group_entries[0])

    return SandboxCalibration(
        calibration_id=_stable_id(key, source_gap_ids),
        source_gap_ids=source_gap_ids,
        object_pose_bias=[round(value, 6) for value in pose_bias],
        perception_noise_bias=[round(value, 6) for value in perception_noise_bias],
        actuation_delay_bias=0.0,
        contact_success_bias=round(contact_success_bias, 4),
        slip_risk_bias=round(slip_risk_bias, 4),
        calibration_confidence=round(confidence, 4),
        details={
            "method": "universal_grouped_gap_weighted_average",
            "created_at": utc_now(),
            "group_key": {
                "robot_type": key[0],
                "scenario_id": key[1],
                "condition_id": key[2],
                "object_class": key[3],
            },
            "source_count": len(evidence),
            "total_weight": round(total_weight, 4),
            "contact_mismatch_rate": round(contact_mismatch_rate, 4),
            "sim_success_real_fail_rate": round(sim_fail_rate, 4),
            "matched_success_rate": round(matched_success_rate, 4),
            "avg_object_pose_error": round(avg_pose_error, 6),
            "evidence": evidence,
        },
    )


def compute_group_calibrations(entries: list[ExperienceEntry]) -> dict[tuple[str, str, str, str], SandboxCalibration]:
    groups = group_gap_entries(entries)
    return {
        key: compute_sandbox_calibration(group_entries, group_key=key)
        for key, group_entries in groups.items()
    }


def apply_sandbox_calibration(entries: list[ExperienceEntry]) -> list[ExperienceEntry]:
    calibrations = compute_group_calibrations(entries)
    updated: list[ExperienceEntry] = []
    for entry in entries:
        calibration = calibrations.get(calibration_group_key(entry))
        if calibration is None or not calibration.calibration_id:
            updated.append(entry)
            continue
        new_entry = replace(entry)
        new_entry.sandbox_calibration = calibration
        new_entry.retrieval_key = build_retrieval_key(new_entry)
        updated.append(new_entry)
    return updated


def apply_calibration_to_position(position: Any, calibration: SandboxCalibration | dict[str, Any] | None) -> list[float]:
    pos = _vector(position)
    if pos is None:
        raise ValueError("position must be a numeric vector")
    while len(pos) < 3:
        pos.append(0.0)
    if isinstance(calibration, SandboxCalibration):
        bias = calibration.object_pose_bias
    elif isinstance(calibration, dict):
        bias = calibration.get("object_pose_bias") or []
    else:
        bias = []
    bias_vector = _vector(bias) or [0.0, 0.0, 0.0]
    while len(bias_vector) < 3:
        bias_vector.append(0.0)
    return [float(pos[i] + bias_vector[i]) for i in range(3)]


def calibrated_attach_distance(base_distance: float, calibration: SandboxCalibration | dict[str, Any] | None) -> float:
    """Wrapper1-compatible attach-distance calibration helper.

    Galaxea physical grasping code does not use fake attachment. This helper is
    kept for memory/sandbox compatibility with wrapper1 reports.
    """

    if isinstance(calibration, SandboxCalibration):
        details = calibration.details if isinstance(calibration.details, dict) else {}
        contact_success_bias = calibration.contact_success_bias
        slip_risk_bias = calibration.slip_risk_bias
    elif isinstance(calibration, dict):
        details = calibration.get("details") if isinstance(calibration.get("details"), dict) else {}
        contact_success_bias = _numeric(calibration.get("contact_success_bias"), 0.0)
        slip_risk_bias = _numeric(calibration.get("slip_risk_bias"), 0.0)
    else:
        details = {}
        contact_success_bias = 0.0
        slip_risk_bias = 0.0
    gripper_gap = _numeric(details.get("avg_gripper_closure_gap"), 0.0)
    adjusted = float(base_distance)
    if gripper_gap > 0.0:
        adjusted += min(gripper_gap * 0.5, 0.02)
    if contact_success_bias < 0.0 or slip_risk_bias > 0.0:
        adjusted -= min(0.01 * abs(contact_success_bias) + 0.01 * slip_risk_bias, 0.015)
    return _clamp(adjusted, 0.025, 0.065)
