"""Pair simulation/real experiences and compute first-pass sim-real gaps."""

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


def _entry_id(entry: Any) -> str:
    return str(getattr(entry, "experience_id", "") or "")


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _result_success(entry: Any) -> bool:
    result = getattr(entry, "result", None)
    if result is None:
        return False
    return bool(getattr(result, "success", None) if getattr(result, "success", None) is not None else getattr(result, "recovery_success", False))


def _feedback(entry: Any) -> dict[str, Any]:
    return _asdict(getattr(entry, "execution_feedback", {}) or {})


def _sensor_summary(entry: Any) -> dict[str, Any]:
    return _asdict(getattr(entry, "sensor_summary", {}) or {})


def _retrieval_key(entry: Any) -> dict[str, Any]:
    return getattr(entry, "retrieval_key", {}) if isinstance(getattr(entry, "retrieval_key", {}), dict) else {}


def _contact_pattern(entry: Any) -> str:
    rk = _retrieval_key(entry)
    if rk.get("contact_pattern"):
        return str(rk.get("contact_pattern"))
    feedback = _feedback(entry)
    close = feedback.get("contact_after_close") or {}
    lift = feedback.get("contact_after_lift") or {}

    def has_contact(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return bool(value.get("left_contact") or value.get("right_contact") or value.get("contact"))

    close_ok = has_contact(close)
    lift_ok = has_contact(lift)
    if close_ok and lift_ok:
        return "contact_close_and_lift"
    if close_ok:
        return "contact_after_close_only"
    if lift_ok:
        return "contact_after_lift_only"
    return "no_contact"


def _plan_signature(entry: Any) -> str:
    rk = _retrieval_key(entry)
    return str(rk.get("plan_signature") or getattr(entry, "plan_signature", "") or "")


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    out: list[float] = []
    for item in value:
        numeric = _numeric(item)
        if numeric is None:
            return None
        out.append(numeric)
    return out


def _l2(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right:
        return None
    n = min(len(left), len(right))
    if n <= 0:
        return None
    return sum((left[i] - right[i]) ** 2 for i in range(n)) ** 0.5


def _observed_position(entry: Any) -> list[float] | None:
    feedback = _feedback(entry)
    pos = _vector(feedback.get("observed_pos"))
    if pos is not None:
        return pos
    perception = _asdict(getattr(entry, "perception", {}) or {})
    after = perception.get("after_anomaly") if isinstance(perception.get("after_anomaly"), dict) else {}
    return _vector(after.get("object_pos"))


def _perceived_position(entry: Any) -> list[float] | None:
    perception = _asdict(getattr(entry, "perception", {}) or {})
    after = perception.get("after_anomaly") if isinstance(perception.get("after_anomaly"), dict) else {}
    return _vector(after.get("object_pos"))


def _pinch_distance(entry: Any) -> float | None:
    criteria = (_feedback(entry).get("recovery_success_criteria") or {})
    if isinstance(criteria, dict):
        value = _numeric(criteria.get("pinch_distance"))
        if value is not None:
            return value
    gripper_state = _sensor_summary(entry).get("gripper_state") or {}
    if isinstance(gripper_state, dict):
        for key in ("pinch_distance", "gripper_width", "width"):
            value = _numeric(gripper_state.get(key))
            if value is not None:
                return value
    return None


def _joint_positions(entry: Any) -> list[float] | None:
    return _vector(_sensor_summary(entry).get("joint_positions"))


def _gripper_state_scalar(entry: Any) -> float | None:
    gripper_state = _sensor_summary(entry).get("gripper_state") or {}
    if not isinstance(gripper_state, dict):
        return None
    for key in ("gripper_width", "width", "pinch_distance"):
        value = _numeric(gripper_state.get(key))
        if value is not None:
            return value
    driver_joints = gripper_state.get("driver_joints") or {}
    if not isinstance(driver_joints, dict):
        return None
    positions: list[float] = []
    for state in driver_joints.values():
        if isinstance(state, dict):
            value = _numeric(state.get("position"))
            if value is not None:
                positions.append(value)
    if not positions:
        return None
    return sum(abs(value) for value in positions) / len(positions)


def _z_after_recovery(entry: Any) -> float | None:
    return _numeric(_feedback(entry).get("apple_z_after_recovery"))


def _normalize_distance(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    if scale <= 0:
        return 0.0
    return max(0.0, min(float(value) / scale, 1.0))


def pair_score(sim_entry: Any, real_entry: Any) -> float:
    """Score whether two entries are comparable sim/real experiences."""

    if getattr(sim_entry, "scenario_id", "") != getattr(real_entry, "scenario_id", ""):
        return 0.0

    score = 0.25
    if getattr(sim_entry, "condition_id", "") and getattr(sim_entry, "condition_id", "") == getattr(real_entry, "condition_id", ""):
        score += 0.30
    if _plan_signature(sim_entry) and _plan_signature(sim_entry) == _plan_signature(real_entry):
        score += 0.25
    if _contact_pattern(sim_entry) == _contact_pattern(real_entry):
        score += 0.10
    if getattr(sim_entry, "task", None) and getattr(real_entry, "task", None):
        sim_task = _asdict(getattr(sim_entry, "task"))
        real_task = _asdict(getattr(real_entry, "task"))
        if sim_task.get("stage") and sim_task.get("stage") == real_task.get("stage"):
            score += 0.10
    return round(min(score, 1.0), 4)


def pair_sim_real_experiences(
    sim_entries: list[Any],
    real_entries: list[Any],
    *,
    min_pair_score: float = 0.55,
) -> list[dict[str, Any]]:
    """Return best scored sim/real pairs, one real entry per pair."""

    pairs: list[dict[str, Any]] = []
    used_real_ids: set[str] = set()
    candidates: list[tuple[float, Any, Any]] = []
    for sim_entry in sim_entries:
        for real_entry in real_entries:
            score = pair_score(sim_entry, real_entry)
            if score >= min_pair_score:
                candidates.append((score, sim_entry, real_entry))
    candidates.sort(key=lambda item: (-item[0], _entry_id(item[1]), _entry_id(item[2])))

    for score, sim_entry, real_entry in candidates:
        rid = _entry_id(real_entry)
        if rid in used_real_ids:
            continue
        used_real_ids.add(rid)
        paired_by = "scenario_id"
        if getattr(sim_entry, "condition_id", "") == getattr(real_entry, "condition_id", ""):
            paired_by = "condition_id"
        if _plan_signature(sim_entry) and _plan_signature(sim_entry) == _plan_signature(real_entry):
            paired_by = "plan_signature"
        pairs.append(
            {
                "pair_id": _stable_id("pair", _entry_id(sim_entry), _entry_id(real_entry)),
                "sim_experience_id": _entry_id(sim_entry),
                "real_experience_id": rid,
                "paired_by": paired_by,
                "pair_score": score,
                "gap_score": 0.0,
                "validation_status": "paired",
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
            }
        )
    return pairs


def compute_sim_real_gap(sim_entry: Any, real_entry: Any) -> dict[str, Any]:
    """Compute interpretable first-pass gap fields for a sim/real pair."""

    sim_success = _result_success(sim_entry)
    real_success = _result_success(real_entry)
    if sim_success and real_success:
        outcome_type = "matched_success"
    elif (not sim_success) and (not real_success):
        outcome_type = "matched_failure"
    elif sim_success and not real_success:
        outcome_type = "sim_success_real_fail"
    else:
        outcome_type = "sim_fail_real_success"

    sim_observed = _observed_position(sim_entry)
    real_observed = _observed_position(real_entry)
    pose_error = _l2(sim_observed, real_observed)
    sim_perceived = _perceived_position(sim_entry)
    real_perceived = _perceived_position(real_entry)
    perception_pose_error = _l2(sim_perceived, real_perceived)
    z_gap = None
    sim_z = _z_after_recovery(sim_entry)
    real_z = _z_after_recovery(real_entry)
    if sim_z is not None and real_z is not None:
        z_gap = abs(sim_z - real_z)

    sim_contact = _contact_pattern(sim_entry)
    real_contact = _contact_pattern(real_entry)
    contact_mismatch = sim_contact != real_contact

    sim_pinch = _pinch_distance(sim_entry)
    real_pinch = _pinch_distance(real_entry)
    gripper_closure_gap = None
    if sim_pinch is not None and real_pinch is not None:
        gripper_closure_gap = abs(sim_pinch - real_pinch)

    sim_joint_positions = _joint_positions(sim_entry)
    real_joint_positions = _joint_positions(real_entry)
    joint_position_error = _l2(sim_joint_positions, real_joint_positions)
    sim_gripper_state = _gripper_state_scalar(sim_entry)
    real_gripper_state = _gripper_state_scalar(real_entry)
    gripper_state_gap = None
    if sim_gripper_state is not None and real_gripper_state is not None:
        gripper_state_gap = abs(sim_gripper_state - real_gripper_state)

    outcome_gap_score = 0.0 if sim_success == real_success else 1.0
    pose_score = max(_normalize_distance(pose_error, 0.10), _normalize_distance(z_gap, 0.08))
    perception_score = _normalize_distance(perception_pose_error, 0.08)
    contact_score = 1.0 if contact_mismatch else 0.0
    actuation_score = _normalize_distance(gripper_closure_gap, 0.05)
    gap_score = (
        0.35 * outcome_gap_score
        + 0.25 * pose_score
        + 0.20 * contact_score
        + 0.10 * perception_score
        + 0.10 * actuation_score
    )

    missing_evidence = sum(
        value is None
        for value in (pose_error, perception_pose_error, z_gap, gripper_closure_gap)
    )
    uncertainty = min(0.15 + missing_evidence * 0.12 + (0.20 if contact_mismatch else 0.0), 1.0)

    return {
        "gap_id": _stable_id("gap", _entry_id(sim_entry), _entry_id(real_entry)),
        "gap_score": round(gap_score, 4),
        "pose_gap": {
            "object_pose_error": pose_error,
            "z_after_recovery_gap": z_gap,
            "sim_observed_pos": sim_observed,
            "real_observed_pos": real_observed,
        },
        "contact_gap": {
            "contact_mismatch": contact_mismatch,
            "sim_contact_pattern": sim_contact,
            "real_contact_pattern": real_contact,
        },
        "outcome_gap": {
            "type": outcome_type,
            "sim_success": sim_success,
            "real_success": real_success,
            "outcome_gap_score": outcome_gap_score,
        },
        "perception_gap": {
            "pose_estimation_gap": perception_pose_error,
            "sim_perceived_pos": sim_perceived,
            "real_perceived_pos": real_perceived,
        },
        "actuation_gap": {
            "gripper_closure_gap": gripper_closure_gap,
            "sim_pinch_distance": sim_pinch,
            "real_pinch_distance": real_pinch,
        },
        "robot_state_gap": {
            "joint_position_error": joint_position_error,
            "gripper_state_gap": gripper_state_gap,
            "sim_joint_positions": sim_joint_positions,
            "real_joint_positions": real_joint_positions,
            "sim_gripper_state": sim_gripper_state,
            "real_gripper_state": real_gripper_state,
        },
        "scene_reconstruction_gap": {},
        "uncertainty": round(uncertainty, 4),
        "evidence": {
            "pair_score": pair_score(sim_entry, real_entry),
            "gap_components": {
                "outcome": outcome_gap_score,
                "pose": round(pose_score, 4),
                "contact": contact_score,
                "perception": round(perception_score, 4),
                "actuation": round(actuation_score, 4),
                "robot_state_observed": joint_position_error is not None or gripper_state_gap is not None,
            },
        },
    }


def apply_pair_and_gap(sim_entry: Any, real_entry: Any, pair: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute pair/gap and return dictionaries ready for MemoryV3Entry fields."""

    pair = dict(pair or {})
    if not pair:
        score = pair_score(sim_entry, real_entry)
        pair = {
            "pair_id": _stable_id("pair", _entry_id(sim_entry), _entry_id(real_entry)),
            "sim_experience_id": _entry_id(sim_entry),
            "real_experience_id": _entry_id(real_entry),
            "paired_by": "condition_id" if getattr(sim_entry, "condition_id", "") == getattr(real_entry, "condition_id", "") else "scenario_id",
            "pair_score": score,
            "validation_status": "paired",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
    gap = compute_sim_real_gap(sim_entry, real_entry)
    pair["gap_score"] = gap["gap_score"]
    if isinstance(gap.get("evidence"), dict):
        gap["evidence"]["pair_score"] = pair.get("pair_score", gap["evidence"].get("pair_score", 0.0))
    return pair, gap
