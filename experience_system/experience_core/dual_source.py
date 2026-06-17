"""Sim-real pairing and gap computation for universal experience entries."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from .failure_taxonomy import standardize_failure_taxonomy
from .schema import ExperienceEntry, SimRealGap, build_retrieval_key, utc_now


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _success(entry: ExperienceEntry) -> bool:
    return bool(entry.result.get("success", entry.result.get("recovery_success", False)))


def _plan_signature(entry: ExperienceEntry) -> str:
    return str(entry.retrieval_key.get("plan_signature") or "->".join(item.name for item in entry.skill_sequence))


def _contact_pattern(entry: ExperienceEntry) -> str:
    value = str(entry.spatial_state.get("contact_pattern") or entry.embodiment.get("attach_mode") or "")
    if value:
        return value
    feedback = entry.execution_feedback
    close = feedback.get("contact_after_close") if isinstance(feedback.get("contact_after_close"), dict) else {}
    lift = feedback.get("contact_after_lift") if isinstance(feedback.get("contact_after_lift"), dict) else {}

    def has_contact(payload: dict[str, Any]) -> bool:
        return bool(payload.get("left_contact") or payload.get("right_contact") or payload.get("contact"))

    close_ok = has_contact(close)
    lift_ok = has_contact(lift)
    if close_ok and lift_ok:
        return "contact_close_and_lift"
    if close_ok:
        return "contact_after_close_only"
    if lift_ok:
        return "contact_after_lift_only"
    return "unknown"


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def _l2(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right:
        return None
    n = min(len(left), len(right))
    if n <= 0:
        return None
    return sum((left[i] - right[i]) ** 2 for i in range(n)) ** 0.5


def _target_position(entry: ExperienceEntry) -> list[float] | None:
    target = entry.object_state.target_object
    obj = entry.object_state.objects.get(target, {}) if isinstance(entry.object_state.objects, dict) else {}
    if isinstance(obj, dict):
        for key in ("final_position", "observed_position", "position"):
            pos = _vector(obj.get(key))
            if pos is not None:
                return pos
    pos = _vector(entry.execution_feedback.get("observed_pos"))
    if pos is not None:
        return pos
    final_objects = entry.state_after.get("objects") if isinstance(entry.state_after.get("objects"), dict) else {}
    if target and isinstance(final_objects.get(target), dict):
        return _vector(final_objects[target].get("position"))
    return None


def _normalize_gap(value: float | None, scale: float) -> float:
    if value is None or scale <= 0:
        return 0.0
    return max(0.0, min(float(value) / scale, 1.0))


def pair_score(sim_entry: ExperienceEntry, real_entry: ExperienceEntry) -> float:
    """Return whether two entries are comparable sim/real experiences."""

    if sim_entry.source == "real" or real_entry.source not in {"real", "pseudo_real"}:
        return 0.0
    if sim_entry.scenario_id != real_entry.scenario_id:
        return 0.0
    if sim_entry.condition_id and real_entry.condition_id and sim_entry.condition_id != real_entry.condition_id:
        return 0.0

    score = 0.25
    if sim_entry.condition_id and sim_entry.condition_id == real_entry.condition_id:
        score += 0.25
    if sim_entry.robot.robot_type == real_entry.robot.robot_type:
        score += 0.15
    if sim_entry.task.get("stage") and sim_entry.task.get("stage") == real_entry.task.get("stage"):
        score += 0.10
    if _plan_signature(sim_entry) and _plan_signature(sim_entry) == _plan_signature(real_entry):
        score += 0.15
    if _contact_pattern(sim_entry) == _contact_pattern(real_entry):
        score += 0.05
    if sim_entry.object_state.object_class and sim_entry.object_state.object_class == real_entry.object_state.object_class:
        score += 0.05
    return round(min(score, 1.0), 4)


def pair_sim_real_experiences(
    entries: list[ExperienceEntry],
    *,
    min_pair_score: float = 0.55,
) -> list[dict[str, Any]]:
    sim_entries = [entry for entry in entries if entry.source == "simulation"]
    real_entries = [entry for entry in entries if entry.source in {"real", "pseudo_real"}]
    candidates: list[tuple[float, ExperienceEntry, ExperienceEntry]] = []
    for sim_entry in sim_entries:
        for real_entry in real_entries:
            score = pair_score(sim_entry, real_entry)
            if score >= min_pair_score:
                candidates.append((score, sim_entry, real_entry))
    candidates.sort(key=lambda item: (-item[0], item[1].experience_id, item[2].experience_id))

    pairs: list[dict[str, Any]] = []
    used_sim: set[str] = set()
    used_real: set[str] = set()
    for score, sim_entry, real_entry in candidates:
        if sim_entry.experience_id in used_sim:
            continue
        if real_entry.experience_id in used_real:
            continue
        used_sim.add(sim_entry.experience_id)
        used_real.add(real_entry.experience_id)
        paired_by = "scenario_id"
        if sim_entry.condition_id == real_entry.condition_id:
            paired_by = "condition_id"
        if _plan_signature(sim_entry) and _plan_signature(sim_entry) == _plan_signature(real_entry):
            paired_by = "plan_signature"
        pairs.append({
            "pair_id": _stable_id("pair", sim_entry.experience_id, real_entry.experience_id),
            "sim_experience_id": sim_entry.experience_id,
            "real_experience_id": real_entry.experience_id,
            "paired_by": paired_by,
            "pair_score": score,
            "gap_score": 0.0,
            "validation_status": "paired",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        })
    return pairs


def compute_sim_real_gap(sim_entry: ExperienceEntry, real_entry: ExperienceEntry) -> SimRealGap:
    sim_success = _success(sim_entry)
    real_success = _success(real_entry)
    if sim_success and real_success:
        outcome_type = "matched_success"
        outcome_gap_score = 0.0
    elif (not sim_success) and (not real_success):
        outcome_type = "matched_failure"
        outcome_gap_score = 0.15
    elif sim_success and not real_success:
        outcome_type = "sim_success_real_fail"
        outcome_gap_score = 1.0
    else:
        outcome_type = "sim_fail_real_success"
        outcome_gap_score = 0.55

    sim_pos = _target_position(sim_entry)
    real_pos = _target_position(real_entry)
    object_pose_error = _l2(sim_pos, real_pos)
    pose_score = _normalize_gap(object_pose_error, 0.20)
    contact_mismatch = _contact_pattern(sim_entry) != _contact_pattern(real_entry)
    contact_score = 0.35 if contact_mismatch else 0.0

    gap_score = max(outcome_gap_score, min(1.0, 0.45 * pose_score + contact_score))
    missing_evidence = 0
    if sim_pos is None or real_pos is None:
        missing_evidence += 1
    uncertainty = min(1.0, 0.15 + 0.25 * missing_evidence + (0.35 if outcome_type == "sim_success_real_fail" else 0.0))

    return SimRealGap(
        gap_id=_stable_id("gap", sim_entry.experience_id, real_entry.experience_id),
        gap_score=round(gap_score, 4),
        uncertainty=round(uncertainty, 4),
        outcome_gap={
            "type": outcome_type,
            "sim_success": sim_success,
            "real_success": real_success,
            "outcome_gap_score": round(outcome_gap_score, 4),
        },
        pose_gap={
            "object_pose_error": None if object_pose_error is None else round(object_pose_error, 6),
            "sim_object_position": sim_pos,
            "real_object_position": real_pos,
        },
        contact_gap={
            "sim_contact_pattern": _contact_pattern(sim_entry),
            "real_contact_pattern": _contact_pattern(real_entry),
            "contact_mismatch": contact_mismatch,
        },
        evidence={
            "sim_experience_id": sim_entry.experience_id,
            "real_experience_id": real_entry.experience_id,
            "pair_score": pair_score(sim_entry, real_entry),
        },
    )


def apply_pair_and_gap(entries: list[ExperienceEntry], pairs: list[dict[str, Any]]) -> list[ExperienceEntry]:
    by_id = {entry.experience_id: entry for entry in entries}
    updated: list[ExperienceEntry] = []
    pair_by_entry: dict[str, dict[str, Any]] = {}
    gap_by_entry: dict[str, SimRealGap] = {}

    for pair in pairs:
        sim_entry = by_id.get(str(pair.get("sim_experience_id")))
        real_entry = by_id.get(str(pair.get("real_experience_id")))
        if sim_entry is None or real_entry is None:
            continue
        gap = compute_sim_real_gap(sim_entry, real_entry)
        pair = dict(pair)
        pair["gap_score"] = gap.gap_score
        pair_by_entry[sim_entry.experience_id] = pair
        pair_by_entry[real_entry.experience_id] = pair
        gap_by_entry[sim_entry.experience_id] = gap
        gap_by_entry[real_entry.experience_id] = gap

    for entry in entries:
        if entry.experience_id in pair_by_entry:
            new_entry = replace(entry)
            new_entry.sim_real_pair = pair_by_entry[entry.experience_id]
            new_entry.sim_real_gap = gap_by_entry[entry.experience_id]
            tags = dict(new_entry.memory_tags)
            tags["memory_role"] = "sim_real_gap_memory"
            new_entry.memory_tags = tags
            standardize_failure_taxonomy(new_entry)
            new_entry.retrieval_key = build_retrieval_key(new_entry)
            updated.append(new_entry)
        else:
            updated.append(entry)
    return updated
