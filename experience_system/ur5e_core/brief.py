"""Prompt brief helpers for UR5e experience memories."""

from __future__ import annotations

from typing import Any

from experience_core import ExperienceEntry
from experience_core.schema import canonical_skill_action


def _skill_sequence(entry: ExperienceEntry) -> list[str]:
    return [canonical_skill_action(item.name) for item in entry.skill_sequence if item.name]


def build_ur5e_experience_brief(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    return {
        "experience_id": entry.experience_id,
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "memory_role": str(entry.memory_tags.get("memory_role") or ""),
        "memory_type": str(entry.memory_tags.get("memory_type") or ""),
        "skill_sequence": _skill_sequence(entry),
        "task_name": str(entry.task.get("name") or ""),
        "object_class": entry.object_state.object_class,
        "target_object": entry.object_state.target_object,
        "plan_signature": str(key.get("plan_signature") or ""),
        "text_summary": str(entry.metadata.get("text_summary") or feedback.get("llm_failure_summary") or ""),
        "failure_type": str(entry.failure_taxonomy.get("failure_type") or ""),
        "failure_stage": str(entry.failure_taxonomy.get("failure_stage") or ""),
        "failure_reason": str(entry.failure_taxonomy.get("failure_reason") or ""),
        "success": bool(entry.result.get("success", False)),
    }


def build_ur5e_planning_prompt_context(entry: ExperienceEntry) -> dict[str, Any]:
    brief = build_ur5e_experience_brief(entry)
    return {
        "schema_version": "ur5e_planning_prompt_context_v1",
        "experience": brief,
        "robot": {
            "robot_id": entry.robot.robot_id,
            "robot_type": entry.robot.robot_type,
            "backend": entry.backend,
        },
        "observations": {
            "sensor_modalities": list(entry.sensor_summary.sensor_modalities or []),
            "state_before": dict(entry.state_before or {}),
            "state_after": dict(entry.state_after or {}),
        },
        "result": dict(entry.result or {}),
    }
