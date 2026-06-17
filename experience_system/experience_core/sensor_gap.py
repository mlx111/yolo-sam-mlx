"""Conservative sim-real gap extraction from real-format sensor evidence."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from .schema import ExperienceEntry, SimRealGap, build_retrieval_key


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _l2(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right:
        return None
    n = min(len(left), len(right))
    if n <= 0:
        return None
    return sum((left[i] - right[i]) ** 2 for i in range(n)) ** 0.5


def _nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _observed_object_position(entry: ExperienceEntry) -> list[float] | None:
    visual = entry.sensor_evidence.visual_observation or {}
    for key in (
        "observed_object_position",
        "detected_object_position",
        "object_position",
        "target_position",
    ):
        pos = _vector(visual.get(key))
        if pos is not None:
            return pos
    for section_key in ("object_pose", "detected_object_pose", "target_pose"):
        section = visual.get(section_key)
        if isinstance(section, dict):
            pos = _vector(section.get("position") or section.get("translation") or section.get("xyz"))
            if pos is not None:
                return pos
    summary = entry.sensor_evidence.summary or {}
    return _vector(summary.get("observed_object_position") or summary.get("detected_object_position"))


def _reference_object_position(entry: ExperienceEntry) -> list[float] | None:
    target = entry.object_state.target_object
    if target and isinstance(entry.object_state.objects, dict):
        item = entry.object_state.objects.get(target)
        if isinstance(item, dict):
            for key in ("sim_position", "expected_position", "commanded_position", "position"):
                pos = _vector(item.get(key))
                if pos is not None:
                    return pos
    for key in ("expected_object_position", "commanded_object_position", "sim_object_position"):
        pos = _vector(entry.state_before.get(key) or entry.state_after.get(key) or entry.execution_feedback.get(key))
        if pos is not None:
            return pos
    return None


def _timestamp_delay(entry: ExperienceEntry) -> float | None:
    summary = entry.sensor_evidence.summary or {}
    for key in ("timestamp_delay_s", "sensor_delay_s", "latency_s", "time_offset_s"):
        if summary.get(key) is not None:
            return abs(_float(summary.get(key)))
    sensor_summary = entry.sensor_summary.timestamps if isinstance(entry.sensor_summary.timestamps, dict) else {}
    for key in ("timestamp_delay_s", "sensor_delay_s", "latency_s", "time_offset_s"):
        if sensor_summary.get(key) is not None:
            return abs(_float(sensor_summary.get(key)))
    return None


def derive_sensor_sim_real_gap(entry: ExperienceEntry) -> SimRealGap:
    """Derive a conservative gap summary from sensor evidence on one entry."""

    modalities = {str(item) for item in entry.sensor_evidence.modalities or [] if item}
    summary = dict(entry.sensor_evidence.summary or {})

    observed_pos = _observed_object_position(entry)
    reference_pos = _reference_object_position(entry)
    pose_error = _l2(reference_pos, observed_pos)
    pose_score = _clamp01((pose_error or 0.0) / 0.20)
    perception_gap: dict[str, Any] = {}
    if "rgbd" in modalities or "rgb" in modalities:
        perception_gap = {
            "source": "sensor_evidence.visual_observation",
            "has_visual_evidence": True,
            "observed_object_position": observed_pos,
            "reference_object_position": reference_pos,
            "object_pose_error": None if pose_error is None else round(pose_error, 6),
            "perception_gap_score": round(pose_score, 4),
            "confidence": 0.65 if pose_error is not None else 0.35,
        }

    max_force = _float(summary.get("max_wrist_force_norm"))
    wrist = entry.sensor_evidence.wrist_force_observation or {}
    if max_force <= 0.0:
        max_force = max(
            _float(wrist.get("force_norm")),
            _float(wrist.get("max_force_norm")),
            _float(wrist.get("peak_force_norm")),
            _float(_nested_get(wrist, "left", "force_norm")),
            _float(_nested_get(wrist, "right", "force_norm")),
        )
    contact_score = 0.0
    if "wrist_force" in modalities:
        contact_score = _clamp01(max_force / 8.0)
        if not bool(entry.result.get("success", entry.result.get("recovery_success", False))) and max_force >= 2.0:
            contact_score = max(contact_score, 0.55)
    contact_gap: dict[str, Any] = {}
    if "wrist_force" in modalities:
        contact_gap = {
            "source": "sensor_evidence.wrist_force_observation",
            "max_wrist_force_norm": round(max_force, 4),
            "contact_gap_score": round(contact_score, 4),
            "contact_risk_signal": max_force >= 2.0,
            "confidence": 0.70 if max_force > 0.0 else 0.40,
        }

    lidar = entry.sensor_evidence.lidar_observation or {}
    nearest = summary.get("nearest_obstacle_distance")
    if nearest is None:
        nearest = lidar.get("nearest_obstacle_distance")
    scene_score = 0.0
    if nearest is not None:
        distance = _float(nearest, 999.0)
        scene_score = _clamp01((0.75 - distance) / 0.75) if distance < 0.75 else 0.0
    scene_gap: dict[str, Any] = {}
    if "lidar" in modalities:
        scene_gap = {
            "source": "sensor_evidence.lidar_observation",
            "nearest_obstacle_distance": nearest,
            "ray_count": summary.get("lidar_ray_count", lidar.get("ray_count")),
            "scene_reconstruction_gap_score": round(scene_score, 4),
            "near_obstacle_signal": nearest is not None and _float(nearest, 999.0) <= 0.50,
            "confidence": 0.65 if nearest is not None else 0.35,
        }

    delay = _timestamp_delay(entry)
    timing_score = _clamp01((delay or 0.0) / 0.50)
    timing_gap = {}
    if delay is not None:
        timing_gap = {
            "source": "sensor_timestamps",
            "timestamp_delay_s": round(delay, 4),
            "timing_gap_score": round(timing_score, 4),
            "confidence": 0.50,
        }

    component_scores = [
        pose_score if perception_gap else 0.0,
        contact_score if contact_gap else 0.0,
        scene_score if scene_gap else 0.0,
        timing_score if timing_gap else 0.0,
    ]
    evidence_count = sum(1 for score in component_scores if score > 0.0)
    modality_count = len(modalities)
    gap_score = max(component_scores) if component_scores else 0.0
    if evidence_count >= 2:
        gap_score = max(gap_score, min(1.0, sum(component_scores) / max(len(component_scores), 1)))
    uncertainty = _clamp01(0.70 - 0.10 * modality_count - 0.08 * evidence_count)

    if gap_score >= 0.50:
        outcome_type = "sensor_gap_high"
    elif gap_score > 0.0:
        outcome_type = "sensor_gap_observed"
    else:
        outcome_type = "sensor_gap_no_strong_signal"

    return SimRealGap(
        gap_id=_stable_id("sensor_gap", entry.experience_id, ",".join(sorted(modalities))),
        gap_score=round(gap_score, 4),
        uncertainty=round(uncertainty, 4),
        outcome_gap={
            "type": outcome_type,
            "source": "sensor_evidence",
            "sensor_gap_score": round(gap_score, 4),
        },
        pose_gap={
            "object_pose_error": None if pose_error is None else round(pose_error, 6),
            "reference_object_position": reference_pos,
            "observed_object_position": observed_pos,
        },
        contact_gap=contact_gap,
        perception_gap=perception_gap,
        timing_gap=timing_gap,
        scene_reconstruction_gap=scene_gap,
        evidence={
            "method": "sensor_derived_gap_v1",
            "experience_id": entry.experience_id,
            "source": entry.source,
            "modalities": sorted(modalities),
            "component_scores": {
                "perception": round(pose_score, 4) if perception_gap else 0.0,
                "contact": round(contact_score, 4) if contact_gap else 0.0,
                "scene_reconstruction": round(scene_score, 4) if scene_gap else 0.0,
                "timing": round(timing_score, 4) if timing_gap else 0.0,
            },
        },
    )


def attach_sensor_sim_real_gap(entry: ExperienceEntry, *, overwrite: bool = False) -> ExperienceEntry:
    """Return an entry with sensor-derived gap fields when appropriate."""

    if entry.source not in {"real", "pseudo_real"}:
        return entry
    if entry.sim_real_gap.gap_id and not overwrite:
        return entry
    if not entry.sensor_evidence.modalities:
        return entry
    updated = replace(entry)
    updated.sim_real_gap = derive_sensor_sim_real_gap(updated)
    tags = dict(updated.memory_tags or {})
    tags.setdefault("sensor_gap_method", "sensor_derived_gap_v1")
    updated.memory_tags = tags
    updated.retrieval_key = build_retrieval_key(updated)
    return updated


def apply_sensor_sim_real_gaps(entries: list[ExperienceEntry], *, overwrite: bool = False) -> list[ExperienceEntry]:
    """Attach sensor-derived gaps to all eligible entries."""

    return [attach_sensor_sim_real_gap(entry, overwrite=overwrite) for entry in entries]
