"""Sensor-evidence quality signals for real-format memory entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ExperienceEntry


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


def _episode_root(entry: ExperienceEntry) -> Path | None:
    root = entry.raw_refs.get("episode_dir") or entry.real_episode_ref.get("episode_dir")
    if not root:
        return None
    return Path(str(root))


def _path_exists(path_value: Any, root: Path | None) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    path = Path(path_value)
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.exists()


def _iter_sensor_refs(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if not isinstance(value, dict):
        return refs
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, str) and item and (key.endswith("_path") or key.endswith("_dir") or key in {"rgb", "depth", "scan", "log"}):
            refs.append((name, item))
        elif isinstance(item, dict):
            refs.extend(_iter_sensor_refs(item, name))
    return refs


def sensor_quality_report(entry: ExperienceEntry, *, check_refs: bool = False) -> dict[str, Any]:
    """Return bounded quality signals for stored sensor evidence.

    The report is deterministic and conservative. It records engineering
    quality signals, not learned sensor reliability.
    """

    evidence = entry.sensor_evidence
    modalities = {str(item) for item in evidence.modalities or [] if item}
    summary = dict(evidence.summary or {})
    root = _episode_root(entry)
    trigger_events: list[str] = []

    has_rgb = "rgb" in modalities or "rgbd" in modalities
    has_rgbd = "rgbd" in modalities
    has_lidar = "lidar" in modalities
    has_wrist_force = "wrist_force" in modalities

    visual_score = 0.0
    visual = evidence.visual_observation or {}
    if has_rgbd:
        visual_score += 0.25
        trigger_events.append("sensor_rgbd_available")
    elif has_rgb:
        visual_score += 0.15
        trigger_events.append("sensor_rgb_available")
    if isinstance(visual, dict) and (visual.get("rgb_path") or visual.get("image_path")):
        visual_score += 0.04
    if isinstance(visual, dict) and (visual.get("depth_path") or visual.get("depth_image_path")):
        visual_score += 0.04

    retrieval_keyframes = [
        frame for frame in entry.keyframes
        if isinstance(frame, dict) and frame.get("used_for_retrieval", True) and frame.get("image_path")
    ]
    keyframe_score = min(0.18, 0.06 * len(retrieval_keyframes))
    if retrieval_keyframes:
        trigger_events.append("sensor_keyframes_available")

    lidar_score = 0.0
    lidar_risk_signal_score = 0.0
    if has_lidar:
        lidar_score += 0.16
        trigger_events.append("sensor_lidar_available")
        ray_count = summary.get("lidar_ray_count")
        if ray_count is None and isinstance(evidence.lidar_observation, dict):
            ray_count = evidence.lidar_observation.get("ray_count")
        if _float(ray_count) > 0:
            lidar_score += 0.04
        nearest = summary.get("nearest_obstacle_distance")
        if nearest is None and isinstance(evidence.lidar_observation, dict):
            nearest = evidence.lidar_observation.get("nearest_obstacle_distance")
        if nearest is not None and _float(nearest, 999.0) <= 0.50:
            lidar_risk_signal_score = 0.12
            trigger_events.append("sensor_lidar_near_obstacle")

    wrist_score = 0.0
    force_risk_signal_score = 0.0
    max_force = _float(summary.get("max_wrist_force_norm"))
    if has_wrist_force:
        wrist_score += 0.16
        trigger_events.append("sensor_wrist_force_available")
        if max_force > 0.0:
            wrist_score += 0.04
        if max_force >= 2.0:
            force_risk_signal_score = 0.12
            trigger_events.append("sensor_wrist_force_contact_signal")

    completeness_score = _clamp01(visual_score + keyframe_score + lidar_score + wrist_score)
    risk_signal_score = _clamp01(lidar_risk_signal_score + force_risk_signal_score)

    sensor_refs = []
    sensor_refs.extend(_iter_sensor_refs(evidence.visual_observation, "visual_observation"))
    sensor_refs.extend(_iter_sensor_refs(evidence.lidar_observation, "lidar_observation"))
    sensor_refs.extend(_iter_sensor_refs(evidence.wrist_force_observation, "wrist_force_observation"))
    sensor_refs.extend(
        (f"sensor_evidence_refs.{key}", str(value))
        for key, value in (evidence.evidence_refs or {}).items()
        if isinstance(value, str) and (str(key).endswith("_path") or str(key) in {"rgb", "depth", "scan", "log"})
    )

    missing_refs: list[dict[str, str]] = []
    if check_refs:
        seen: set[tuple[str, str]] = set()
        for name, path in sensor_refs:
            key = (name, path)
            if key in seen:
                continue
            seen.add(key)
            if not _path_exists(path, root):
                missing_refs.append({"field": name, "path": path})
        if missing_refs:
            trigger_events.append("sensor_missing_reference")

    missing_ref_penalty = min(0.30, 0.06 * len(missing_refs))
    sensor_evidence_score = _clamp01(0.75 * completeness_score + 0.25 * risk_signal_score - missing_ref_penalty)

    return {
        "sensor_evidence_score": round(sensor_evidence_score, 4),
        "sensor_completeness_score": round(completeness_score, 4),
        "sensor_risk_signal_score": round(risk_signal_score, 4),
        "sensor_missing_ref_count": len(missing_refs),
        "sensor_ref_count": len(sensor_refs),
        "sensor_modalities": sorted(modalities),
        "has_rgb": has_rgb,
        "has_rgbd": has_rgbd,
        "has_lidar": has_lidar,
        "has_wrist_force": has_wrist_force,
        "max_wrist_force_norm": round(max_force, 4),
        "lidar_nearest_obstacle_distance": summary.get("nearest_obstacle_distance"),
        "missing_refs": missing_refs,
        "trigger_events": trigger_events,
    }


def enrich_memory_gate_with_sensor_quality(
    entry: ExperienceEntry,
    *,
    check_refs: bool = False,
) -> ExperienceEntry:
    """Attach sensor-quality signals to ``entry.memory_gate.explanation``."""

    report = sensor_quality_report(entry, check_refs=check_refs)
    explanation = dict(entry.memory_gate.explanation or {})
    explanation["sensor_quality"] = report
    entry.memory_gate.explanation = explanation
    existing_events = list(entry.memory_gate.trigger_events or [])
    for event in report.get("trigger_events", []):
        if event not in existing_events:
            existing_events.append(event)
    entry.memory_gate.trigger_events = existing_events
    return entry
