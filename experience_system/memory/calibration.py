"""Rule-based sandbox calibration from sim-real gap memories."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _asdict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {}


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


def _stable_id(source_gap_ids: list[str]) -> str:
    payload = "|".join(sorted(source_gap_ids)) if source_gap_ids else _utc_now()
    return "cal_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _entry_gap(entry: Any) -> dict[str, Any]:
    return _asdict(getattr(entry, "sim_real_gap", {}) or {})


def _entry_score(score: Any) -> float:
    return _clamp(_numeric(score, 1.0), 0.0, 5.0)


def _gap_weight(gap: dict[str, Any], retrieval_score: float) -> float:
    gap_score = _clamp(_numeric(gap.get("gap_score"), 0.0), 0.0, 1.0)
    uncertainty = _clamp(_numeric(gap.get("uncertainty"), 0.0), 0.0, 1.0)
    retrieval = _clamp(retrieval_score / 2.0, 0.0, 1.0)
    return _clamp(0.45 * gap_score + 0.35 * (1.0 - uncertainty) + 0.20 * retrieval, 0.0, 1.0)


def _pose_delta(gap: dict[str, Any]) -> list[float] | None:
    pose_gap = gap.get("pose_gap") if isinstance(gap.get("pose_gap"), dict) else {}
    sim_pos = _vector(pose_gap.get("sim_observed_pos"))
    real_pos = _vector(pose_gap.get("real_observed_pos"))
    if not sim_pos or not real_pos:
        return None
    n = min(3, len(sim_pos), len(real_pos))
    if n <= 0:
        return None
    delta = [real_pos[i] - sim_pos[i] for i in range(n)]
    while len(delta) < 3:
        delta.append(0.0)
    return delta


def compute_sandbox_calibration(
    experiences: list[tuple[Any, float]],
    *,
    max_gap_memories: int = 5,
) -> dict[str, Any]:
    """Build SandboxCalibrationInfo-compatible dict from retrieved gap memories."""

    weighted_pose = [0.0, 0.0, 0.0]
    weighted_perception = [0.0, 0.0, 0.0]
    total_weight = 0.0
    source_gap_ids: list[str] = []
    evidence: list[dict[str, Any]] = []
    contact_mismatch_weight = 0.0
    sim_success_real_fail_weight = 0.0
    gripper_gap_weighted = 0.0
    gripper_gap_weight = 0.0

    for entry, score in (experiences or [])[:max_gap_memories]:
        gap = _entry_gap(entry)
        if not gap or not gap.get("gap_id"):
            continue
        weight = _gap_weight(gap, _entry_score(score))
        if weight <= 0.0:
            continue
        source_gap_ids.append(str(gap.get("gap_id")))

        delta = _pose_delta(gap)
        if delta is not None:
            for i in range(3):
                weighted_pose[i] += delta[i] * weight
                weighted_perception[i] += delta[i] * weight
            total_weight += weight

        contact_gap = gap.get("contact_gap") if isinstance(gap.get("contact_gap"), dict) else {}
        if contact_gap.get("contact_mismatch"):
            contact_mismatch_weight += weight

        outcome_gap = gap.get("outcome_gap") if isinstance(gap.get("outcome_gap"), dict) else {}
        if outcome_gap.get("type") == "sim_success_real_fail":
            sim_success_real_fail_weight += weight

        actuation_gap = gap.get("actuation_gap") if isinstance(gap.get("actuation_gap"), dict) else {}
        gripper_closure_gap = _numeric(actuation_gap.get("gripper_closure_gap"), 0.0)
        if gripper_closure_gap > 0.0:
            gripper_gap_weighted += gripper_closure_gap * weight
            gripper_gap_weight += weight

        evidence.append(
            {
                "experience_id": getattr(entry, "experience_id", ""),
                "gap_id": gap.get("gap_id", ""),
                "gap_score": gap.get("gap_score", 0.0),
                "uncertainty": gap.get("uncertainty", 0.0),
                "retrieval_score": score,
                "weight": round(weight, 4),
                "outcome_gap_type": outcome_gap.get("type", ""),
                "contact_mismatch": bool(contact_gap.get("contact_mismatch")),
            }
        )

    if not source_gap_ids:
        return {
            "calibration_id": "",
            "source_gap_ids": [],
            "object_pose_bias": [],
            "gripper_delay_bias": 0.0,
            "slip_risk_bias": 0.0,
            "contact_success_bias": 0.0,
            "perception_noise_bias": [],
            "applied_to_candidate": False,
            "calibration_confidence": 0.0,
            "details": {"reason": "no_gap_memory"},
        }

    pose_bias = [x / total_weight for x in weighted_pose] if total_weight > 0.0 else [0.0, 0.0, 0.0]
    perception_bias = [x / total_weight for x in weighted_perception] if total_weight > 0.0 else [0.0, 0.0, 0.0]
    # Keep first-pass pose corrections conservative: max 4cm per axis.
    pose_bias = [_clamp(x, -0.04, 0.04) for x in pose_bias]
    perception_bias = [_clamp(x, -0.04, 0.04) for x in perception_bias]

    gripper_closure_gap = gripper_gap_weighted / gripper_gap_weight if gripper_gap_weight > 0.0 else 0.0
    slip_risk_bias = _clamp(sim_success_real_fail_weight / max(sum(item["weight"] for item in evidence), 1e-6), 0.0, 1.0)
    contact_success_bias = -_clamp(contact_mismatch_weight / max(sum(item["weight"] for item in evidence), 1e-6), 0.0, 1.0)
    confidence = _clamp(sum(item["weight"] for item in evidence) / max(len(evidence), 1), 0.0, 1.0)

    return {
        "calibration_id": _stable_id(source_gap_ids),
        "source_gap_ids": source_gap_ids,
        "object_pose_bias": [round(x, 6) for x in pose_bias],
        "gripper_delay_bias": 0.0,
        "slip_risk_bias": round(slip_risk_bias, 4),
        "contact_success_bias": round(contact_success_bias, 4),
        "perception_noise_bias": [round(x, 6) for x in perception_bias],
        "applied_to_candidate": True,
        "calibration_confidence": round(confidence, 4),
        "details": {
            "method": "rule_gap_weighted_average",
            "source_count": len(evidence),
            "avg_gripper_closure_gap": round(gripper_closure_gap, 6),
            "evidence": evidence,
        },
    }


def apply_calibration_to_position(position: Any, calibration: dict[str, Any] | None) -> list[float]:
    """Apply object pose bias to a 3D position."""

    pos = _vector(position)
    if pos is None:
        raise ValueError("position must be a numeric vector")
    while len(pos) < 3:
        pos.append(0.0)
    calibration = calibration if isinstance(calibration, dict) else {}
    bias = _vector(calibration.get("object_pose_bias")) or [0.0, 0.0, 0.0]
    while len(bias) < 3:
        bias.append(0.0)
    return [float(pos[i] + bias[i]) for i in range(3)]


def calibrated_attach_distance(base_distance: float, calibration: dict[str, Any] | None) -> float:
    """Convert gap-derived gripper/contact risk into a conservative attach gate."""

    calibration = calibration if isinstance(calibration, dict) else {}
    details = calibration.get("details") if isinstance(calibration.get("details"), dict) else {}
    gripper_gap = _numeric(details.get("avg_gripper_closure_gap"), 0.0)
    contact_success_bias = _numeric(calibration.get("contact_success_bias"), 0.0)
    slip_risk_bias = _numeric(calibration.get("slip_risk_bias"), 0.0)
    adjusted = float(base_distance)
    if gripper_gap > 0.0:
        adjusted += min(gripper_gap * 0.5, 0.02)
    if contact_success_bias < 0.0 or slip_risk_bias > 0.0:
        adjusted -= min(0.01 * abs(contact_success_bias) + 0.01 * slip_risk_bias, 0.015)
    return _clamp(adjusted, 0.025, 0.065)
