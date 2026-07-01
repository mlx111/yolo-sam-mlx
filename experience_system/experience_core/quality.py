"""Quality validation for universal experience entries and real episodes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .schema import ExperienceEntry


REQUIRED_ENTRY_FIELDS = (
    "scenario_id",
    "condition_id",
    "robot_type",
    "backend",
    "skill_sequence",
    "result",
    "object_class",
)


def missing_entry_fields(entry: ExperienceEntry) -> list[str]:
    missing = []
    if not entry.scenario_id:
        missing.append("scenario_id")
    if not entry.condition_id:
        missing.append("condition_id")
    if not entry.robot.robot_type:
        missing.append("robot_type")
    if not entry.backend:
        missing.append("backend")
    if not entry.skill_sequence:
        missing.append("skill_sequence")
    if not entry.result:
        missing.append("result")
    if not entry.object_state.object_class:
        missing.append("object_class")
    return missing


def validate_experience_entry(entry: ExperienceEntry, *, check_refs: bool = False) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for field in missing_entry_fields(entry):
        issues.append({"severity": "error", "code": "missing_required_field", "field": field})
    for index, skill in enumerate(entry.skill_sequence):
        if not skill.name:
            issues.append({"severity": "error", "code": "empty_skill_name", "index": index})
    if "success" not in entry.result and "task_success" not in entry.result:
        issues.append({"severity": "error", "code": "missing_result_success"})
    if check_refs:
        for ref_name, ref_value in _iter_path_refs(entry.raw_refs):
            if not Path(ref_value).exists():
                issues.append({"severity": "warning", "code": "missing_raw_ref", "field": ref_name, "path": ref_value})
        for index, frame in enumerate(entry.keyframes):
            image_path = str(frame.get("image_path") or "")
            if image_path and not Path(image_path).exists():
                issues.append({"severity": "warning", "code": "missing_keyframe", "index": index, "path": image_path})
    return issues


def validate_experience_library(entries: list[ExperienceEntry], *, check_refs: bool = False) -> dict[str, Any]:
    duplicate_counts = Counter(entry.experience_id for entry in entries)
    duplicate_ids = {key: value for key, value in duplicate_counts.items() if key and value > 1}
    entries_report = []
    error_count = 0
    warning_count = 0
    for entry in entries:
        issues = validate_experience_entry(entry, check_refs=check_refs)
        if entry.experience_id in duplicate_ids:
            issues.append({"severity": "error", "code": "duplicate_experience_id", "experience_id": entry.experience_id})
        error_count += sum(1 for issue in issues if issue.get("severity") == "error")
        warning_count += sum(1 for issue in issues if issue.get("severity") == "warning")
        if issues:
            entries_report.append({
                "experience_id": entry.experience_id,
                "scenario_id": entry.scenario_id,
                "condition_id": entry.condition_id,
                "issues": issues,
            })
    return {
        "entry_count": len(entries),
        "error_count": error_count,
        "warning_count": warning_count,
        "duplicate_experience_ids": duplicate_ids,
        "entries": entries_report,
        "passed": error_count == 0,
    }


def validate_raw_real_episode(raw_episode: dict[str, Any], *, root: str | Path | None = None, check_refs: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    scenario_id = _first_string(raw_episode, "scenario_id", "scene_id", section=("scenario", "scenario_id"))
    condition_id = _first_string(raw_episode, "condition_id", "anomaly_id", section=("condition", "condition_id"))
    robot_type = _first_string(raw_episode, "robot_type", section=("robot", "robot_type"))
    object_class = _first_string(raw_episode, "object_class", "target_class", section=("object_state", "object_class"))
    skills = raw_episode.get("skill_sequence") or raw_episode.get("executed_recovery_steps") or raw_episode.get("recovery_steps") or []
    result = raw_episode.get("result") if isinstance(raw_episode.get("result"), dict) else raw_episode

    if not scenario_id:
        issues.append({"severity": "error", "code": "missing_required_field", "field": "scenario_id"})
    if not condition_id:
        issues.append({"severity": "error", "code": "missing_required_field", "field": "condition_id"})
    if not robot_type:
        issues.append({"severity": "error", "code": "missing_required_field", "field": "robot_type"})
    if not object_class:
        issues.append({"severity": "error", "code": "missing_required_field", "field": "object_class"})
    if not isinstance(skills, list) or not skills:
        issues.append({"severity": "error", "code": "missing_required_field", "field": "skill_sequence"})
    elif not all(_skill_name(item) for item in skills if isinstance(item, dict)):
        issues.append({"severity": "error", "code": "empty_skill_name"})
    if not isinstance(result, dict) or not any(key in result for key in ("success", "task_success")):
        issues.append({"severity": "error", "code": "missing_result_success"})

    if check_refs:
        root_path = Path(root) if root else None
        for index, frame in enumerate(raw_episode.get("keyframes") or []):
            if not isinstance(frame, dict) or not frame.get("image_path"):
                continue
            path = Path(str(frame["image_path"]))
            if not path.is_absolute() and root_path is not None:
                path = root_path / path
            if not path.exists():
                issues.append({"severity": "warning", "code": "missing_keyframe", "index": index, "path": str(path)})
        refs = raw_episode.get("real_episode_ref") if isinstance(raw_episode.get("real_episode_ref"), dict) else {}
        for key in ("hdf5_path", "video_dir", "keyframe_dir", "robot_log_path"):
            if not refs.get(key):
                continue
            path = Path(str(refs[key]))
            if not path.is_absolute() and root_path is not None:
                path = root_path / path
            if not path.exists():
                issues.append({"severity": "warning", "code": "missing_real_episode_ref", "field": key, "path": str(path)})
        for section_name in ("visual_observation", "rgbd_observation", "lidar_observation", "wrist_force_observation", "force_torque_observation"):
            section = raw_episode.get(section_name)
            if not isinstance(section, dict):
                continue
            for ref_key, ref_value in _sensor_path_refs(section):
                path = Path(ref_value)
                if not path.is_absolute() and root_path is not None:
                    path = root_path / path
                if not path.exists():
                    issues.append({
                        "severity": "warning",
                        "code": "missing_sensor_ref",
                        "section": section_name,
                        "field": ref_key,
                        "path": str(path),
                    })

    return {
        "episode_id": str(raw_episode.get("episode_id") or raw_episode.get("experience_id") or ""),
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "error_count": sum(1 for issue in issues if issue.get("severity") == "error"),
        "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "passed": not any(issue.get("severity") == "error" for issue in issues),
        "issues": issues,
    }


def _iter_path_refs(raw_refs: dict[str, Any]) -> list[tuple[str, str]]:
    refs = []
    for key, value in raw_refs.items():
        if not isinstance(value, str) or not value:
            continue
        if key.endswith("_dir") or key.endswith("_path") or key in {"episode_dir", "hdf5_path", "video_dir", "robot_log_path"}:
            refs.append((key, value))
    return refs


def _first_string(raw: dict[str, Any], *keys: str, section: tuple[str, str] | None = None) -> str:
    for key in keys:
        if raw.get(key):
            return str(raw[key])
    if section and isinstance(raw.get(section[0]), dict):
        return str(raw[section[0]].get(section[1]) or "")
    return ""


def _sensor_path_refs(section: dict[str, Any]) -> list[tuple[str, str]]:
    refs = []
    for key, value in section.items():
        if isinstance(value, str) and value and (key.endswith("_path") or key.endswith("_dir") or key in {"rgb", "depth", "scan", "log"}):
            refs.append((key, value))
        if isinstance(value, dict):
            for child_key, child_value in _sensor_path_refs(value):
                refs.append((f"{key}.{child_key}", child_value))
    return refs


def _skill_name(raw: dict[str, Any]) -> str:
    return str(raw.get("name") or raw.get("skill") or raw.get("action") or raw.get("type") or "")
