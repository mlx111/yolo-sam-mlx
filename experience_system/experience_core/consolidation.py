"""Consolidate duplicate low-risk universal experience entries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .failure_taxonomy import is_actionable_failure_type
from .schema import ExperienceEntry, build_retrieval_key, utc_now


def plan_signature(entry: ExperienceEntry) -> str:
    return str(entry.retrieval_key.get("plan_signature") or "->".join(item.name for item in entry.skill_sequence if item.name))


def consolidation_key(entry: ExperienceEntry) -> tuple[str, str, str, str, str, bool, str, str]:
    return (
        entry.robot.robot_type,
        entry.scenario_id,
        entry.condition_id,
        entry.object_state.object_class,
        plan_signature(entry),
        bool(entry.result.get("success", entry.result.get("task_success", False))),
        str(entry.failure_taxonomy.get("failure_type") or ""),
        entry.source,
    )


def should_consolidate(entry: ExperienceEntry) -> tuple[bool, str]:
    if entry.source in {"real", "pseudo_real"}:
        return False, "preserve_real_or_pseudo_real"
    if not bool(entry.result.get("success", entry.result.get("task_success", False))):
        return False, "preserve_failure"
    if entry.memory_tags.get("memory_role") == "sim_real_gap_memory":
        return False, "preserve_gap_memory"
    if entry.sim_real_gap.gap_id or entry.sim_real_gap.outcome_gap.get("type"):
        return False, "preserve_sim_real_gap"
    if entry.critic_result.overall_status == "block":
        return False, "preserve_blocked_critic"
    if is_actionable_failure_type(entry.failure_taxonomy.get("failure_type")):
        return False, "preserve_failure_taxonomy"
    return True, "low_risk_success"


def consolidate_experiences(entries: list[ExperienceEntry]) -> tuple[list[ExperienceEntry], dict[str, Any]]:
    preserved: list[ExperienceEntry] = []
    groups: dict[tuple[str, str, str, str, str, bool, str, str], list[ExperienceEntry]] = {}
    decisions: list[dict[str, Any]] = []

    for entry in entries:
        ok, reason = should_consolidate(entry)
        if not ok:
            preserved.append(entry)
            decisions.append({"experience_id": entry.experience_id, "decision": "preserve", "reason": reason})
            continue
        groups.setdefault(consolidation_key(entry), []).append(entry)

    consolidated = list(preserved)
    merged_groups = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        if len(group) == 1:
            entry = group[0]
            consolidated.append(entry)
            decisions.append({"experience_id": entry.experience_id, "decision": "preserve", "reason": "unique_low_risk_success"})
            continue
        representative = _choose_representative(group)
        merged_ids = [entry.experience_id for entry in group if entry.experience_id != representative.experience_id]
        new_entry = replace(representative)
        metadata = dict(new_entry.metadata or {})
        metadata["support_count"] = int(metadata.get("support_count", 1)) + len(merged_ids)
        metadata["consolidated_experience_ids"] = sorted(set(metadata.get("consolidated_experience_ids", []) + merged_ids))
        metadata["consolidation_key"] = {
            "robot_type": key[0],
            "scenario_id": key[1],
            "condition_id": key[2],
            "object_class": key[3],
            "plan_signature": key[4],
            "success": key[5],
            "failure_type": key[6],
            "source": key[7],
        }
        new_entry.metadata = metadata
        new_entry.updated_at = utc_now()
        new_entry.retrieval_key = build_retrieval_key(new_entry)
        consolidated.append(new_entry)
        merged_groups.append({
            "representative_id": new_entry.experience_id,
            "merged_ids": merged_ids,
            "support_count": metadata["support_count"],
            "key": metadata["consolidation_key"],
        })
        decisions.append({
            "experience_id": new_entry.experience_id,
            "decision": "representative",
            "reason": "merged_low_risk_success",
            "merged_count": len(merged_ids),
        })
        for merged_id in merged_ids:
            decisions.append({"experience_id": merged_id, "decision": "merged", "reason": "duplicate_low_risk_success", "representative_id": new_entry.experience_id})

    report = {
        "input_count": len(entries),
        "output_count": len(consolidated),
        "removed_count": len(entries) - len(consolidated),
        "merged_group_count": len(merged_groups),
        "merged_groups": merged_groups,
        "decisions": decisions,
    }
    return consolidated, report


def _choose_representative(entries: list[ExperienceEntry]) -> ExperienceEntry:
    def rank(entry: ExperienceEntry) -> tuple[float, str]:
        return (float(entry.memory_gate.write_score or 0.0), entry.experience_id)

    return max(entries, key=rank)
