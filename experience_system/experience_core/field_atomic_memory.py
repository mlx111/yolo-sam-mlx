"""Field atomic experience retrieval and parameter-prior summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any

from .schema import ExperienceEntry


def is_field_atomic_entry(entry: ExperienceEntry) -> bool:
    return (
        entry.memory_tags.get("memory_type") == "field_atomic_experience"
        or bool(entry.metadata.get("field_atomic"))
        or str(entry.memory_tags.get("memory_role") or "").startswith("field_atomic_")
    )


def field_atomic_action(entry: ExperienceEntry) -> str:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    return str(feedback.get("field_atomic_action") or entry.metadata.get("field_atomic_action") or (entry.skill_sequence[0].name if entry.skill_sequence else ""))


def field_atomic_parameters(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    params = feedback.get("field_atomic_parameters")
    if isinstance(params, dict):
        return dict(params)
    for item in entry.skill_sequence:
        raw = item.raw if isinstance(item.raw, dict) else {}
        params = raw.get("parameters")
        if isinstance(params, dict):
            return dict(params)
    return {}


def field_atomic_success(entry: ExperienceEntry) -> bool:
    role = str(entry.memory_tags.get("memory_role") or "")
    if role == "field_atomic_success":
        return True
    if role == "field_atomic_failure":
        return False
    return bool(entry.result.get("success", False))


def build_field_atomic_parameter_priors(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    action: str = "",
    limit: int = 12,
) -> dict[str, Any]:
    matched: list[ExperienceEntry] = []
    for entry in entries:
        if not is_field_atomic_entry(entry):
            continue
        if scenario_id and entry.scenario_id and entry.scenario_id != scenario_id:
            continue
        if condition_id and entry.condition_id and entry.condition_id != condition_id:
            continue
        if action and field_atomic_action(entry) != action:
            continue
        matched.append(entry)

    by_action: dict[str, dict[str, Any]] = {}
    action_counts = Counter(field_atomic_action(entry) for entry in matched)
    role_counts = Counter(str(entry.memory_tags.get("memory_role") or "") for entry in matched)
    for action_name in sorted(action_counts):
        action_entries = [entry for entry in matched if field_atomic_action(entry) == action_name]
        success_entries = [entry for entry in action_entries if field_atomic_success(entry)]
        failure_entries = [entry for entry in action_entries if not field_atomic_success(entry)]
        by_action[action_name] = {
            "entry_count": len(action_entries),
            "success_count": len(success_entries),
            "failure_count": len(failure_entries),
            "success_rate": round(len(success_entries) / len(action_entries), 4) if action_entries else 0.0,
            "recommended_from_success": _summarize_parameter_entries(success_entries),
            "avoid_from_failure": _summarize_parameter_entries(failure_entries),
            "success_evidence_ids": [entry.experience_id for entry in success_entries[:limit]],
            "failure_evidence_ids": [entry.experience_id for entry in failure_entries[:limit]],
        }

    return {
        "schema_version": "field_atomic_parameter_priors_v1",
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "action_filter": action,
        "field_atomic_entry_count": len(matched),
        "field_atomic_success_count": sum(1 for entry in matched if field_atomic_success(entry)),
        "field_atomic_failure_count": sum(1 for entry in matched if not field_atomic_success(entry)),
        "memory_role_distribution": dict(role_counts),
        "action_distribution": dict(action_counts),
        "by_action": by_action,
        "evidence_ids": [entry.experience_id for entry in matched[:limit]],
        "usage": "Use recommended_from_success as parameter examples and avoid_from_failure as failure priors for future field_atomic plans.",
    }


def build_field_atomic_planner_input(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    goal: str = "",
    limit: int = 12,
) -> dict[str, Any]:
    priors = build_field_atomic_parameter_priors(
        entries,
        scenario_id=scenario_id,
        condition_id=condition_id,
        limit=limit,
    )
    recent = []
    for entry in reversed(entries):
        if not is_field_atomic_entry(entry):
            continue
        if scenario_id and entry.scenario_id and entry.scenario_id != scenario_id:
            continue
        if condition_id and entry.condition_id and entry.condition_id != condition_id:
            continue
        recent.append({
            "experience_id": entry.experience_id,
            "memory_role": str(entry.memory_tags.get("memory_role") or ""),
            "action": field_atomic_action(entry),
            "parameters": field_atomic_parameters(entry),
            "success": field_atomic_success(entry),
            "status": entry.result.get("field_atomic_status", ""),
        })
        if len(recent) >= limit:
            break
    return {
        "schema_version": "field_atomic_planner_input_v2",
        "goal": goal,
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "field_atomic_memory_count": priors["field_atomic_entry_count"],
        "field_atomic_parameter_priors": priors,
        "recent_field_atomic_experiences": recent,
        "usage": "Prefer successful parameter ranges and avoid repeated failed parameter patterns.",
    }


def _summarize_parameter_entries(entries: list[ExperienceEntry]) -> dict[str, Any]:
    values: dict[str, list[Any]] = defaultdict(list)
    evidence: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        params = field_atomic_parameters(entry)
        for key, value in params.items():
            values[str(key)].append(value)
            evidence[str(key)].append(entry.experience_id)
    return {
        key: {
            **_summarize_values(items),
            "evidence_ids": evidence[key][:6],
        }
        for key, items in sorted(values.items())
    }


def _summarize_values(items: list[Any]) -> dict[str, Any]:
    nums = [float(item) for item in items if isinstance(item, (int, float)) and not isinstance(item, bool)]
    if nums:
        return {
            "count": len(nums),
            "median": round(float(median(nums)), 6),
            "min": round(float(min(nums)), 6),
            "max": round(float(max(nums)), 6),
        }
    lists = [item for item in items if isinstance(item, list) and item]
    if lists:
        dims = min(len(item) for item in lists)
        return {
            "count": len(lists),
            "median": [round(float(median([float(item[index]) for item in lists if isinstance(item[index], (int, float))])), 6) for index in range(dims)],
        }
    texts = [str(item) for item in items]
    counts = Counter(texts)
    return {
        "count": len(texts),
        "top_values": [{"value": value, "count": count} for value, count in counts.most_common(5)],
    }
