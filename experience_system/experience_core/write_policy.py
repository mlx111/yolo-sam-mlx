"""Write-time policy for universal experience memory entries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .consolidation import consolidation_key, should_consolidate
from .failure_taxonomy import is_actionable_failure_type
from .quality import missing_entry_fields
from .schema import ExperienceEntry, build_retrieval_key, utc_now


def should_write_entry(
    entry: ExperienceEntry,
    existing_entries: list[ExperienceEntry],
    *,
    strict_quality: bool = True,
    merge_duplicates: bool = True,
) -> dict[str, Any]:
    """Return a deterministic write decision for one candidate entry."""

    missing = missing_entry_fields(entry)
    if strict_quality and missing:
        return _decision(entry, "reject", "missing_required_fields", write=False, missing_fields=missing)

    force_reason = _force_write_reason(entry)
    if force_reason:
        return _decision(entry, "write", force_reason, write=True, missing_fields=missing)

    can_consolidate, consolidate_reason = should_consolidate(entry)
    if can_consolidate and merge_duplicates:
        duplicate = _find_duplicate_low_risk_success(entry, existing_entries)
        if duplicate is not None:
            return _decision(
                entry,
                "merge",
                "duplicate_low_risk_success",
                write=False,
                target_experience_id=duplicate.experience_id,
                missing_fields=missing,
                consolidation_reason=consolidate_reason,
            )

    write_score = float(entry.memory_gate.write_score or 0.0)
    if can_consolidate and write_score < 0.20:
        return _decision(
            entry,
            "skip",
            "low_value_success",
            write=False,
            missing_fields=missing,
            consolidation_reason=consolidate_reason,
        )

    return _decision(entry, "write", "accepted", write=True, missing_fields=missing)


def apply_write_decision(
    entries: list[ExperienceEntry],
    entry: ExperienceEntry,
    decision: dict[str, Any],
) -> tuple[list[ExperienceEntry], ExperienceEntry | None]:
    """Apply a write-policy decision and return updated entries plus written/merged entry."""

    action = str(decision.get("decision") or "")
    if action == "write":
        return _upsert(entries, entry), entry
    if action == "merge":
        target_id = str(decision.get("target_experience_id") or "")
        return _merge_support(entries, entry, target_id), _find_by_id(entries, target_id)
    return entries, None


def _force_write_reason(entry: ExperienceEntry) -> str:
    if entry.source in {"real", "pseudo_real"}:
        if not bool(entry.result.get("success", entry.result.get("task_success", False))):
            return "preserve_real_or_pseudo_real_failure"
        return "preserve_real_or_pseudo_real"
    if entry.memory_tags.get("memory_role") == "sim_real_gap_memory":
        return "preserve_gap_memory"
    if entry.memory_tags.get("memory_role") in {"parameter_success_prior", "parameter_failure_case"}:
        return "preserve_sandbox_parameter_experience"
    if entry.memory_tags.get("memory_role") == "semantic_plan_failure":
        return "preserve_semantic_plan_failure"
    if entry.memory_tags.get("memory_role") in {"field_atomic_success", "field_atomic_failure"}:
        return "preserve_field_atomic_experience"
    if entry.sim_real_gap.gap_id or entry.sim_real_gap.outcome_gap.get("type"):
        return "preserve_sim_real_gap"
    if entry.critic_result.overall_status == "block":
        return "preserve_blocked_critic"
    if is_actionable_failure_type(entry.failure_taxonomy.get("failure_type")):
        return "preserve_failure_taxonomy"
    if not bool(entry.result.get("success", entry.result.get("task_success", False))):
        return "preserve_failure"
    return ""


def _find_duplicate_low_risk_success(entry: ExperienceEntry, existing_entries: list[ExperienceEntry]) -> ExperienceEntry | None:
    key = consolidation_key(entry)
    for existing in existing_entries:
        ok, _ = should_consolidate(existing)
        if ok and consolidation_key(existing) == key:
            return existing
    return None


def _upsert(entries: list[ExperienceEntry], entry: ExperienceEntry) -> list[ExperienceEntry]:
    updated = list(entries)
    for index, existing in enumerate(updated):
        if existing.experience_id == entry.experience_id:
            entry.updated_at = utc_now()
            updated[index] = entry
            return updated
    updated.append(entry)
    return updated


def _find_by_id(entries: list[ExperienceEntry], experience_id: str) -> ExperienceEntry | None:
    for entry in entries:
        if entry.experience_id == experience_id:
            return entry
    return None


def _merge_support(entries: list[ExperienceEntry], entry: ExperienceEntry, target_id: str) -> list[ExperienceEntry]:
    updated = list(entries)
    for index, existing in enumerate(updated):
        if existing.experience_id != target_id:
            continue
        merged = replace(existing)
        metadata = dict(merged.metadata or {})
        metadata["support_count"] = int(metadata.get("support_count", 1)) + 1
        ids = set(metadata.get("write_policy_merged_experience_ids", []))
        ids.add(entry.experience_id)
        metadata["write_policy_merged_experience_ids"] = sorted(ids)
        metadata["last_write_policy_merge_at"] = utc_now()
        merged.metadata = metadata
        merged.updated_at = utc_now()
        merged.retrieval_key = build_retrieval_key(merged)
        updated[index] = merged
        return updated
    return _upsert(updated, entry)


def _decision(entry: ExperienceEntry, decision: str, reason: str, *, write: bool, **extra: Any) -> dict[str, Any]:
    payload = {
        "candidate_experience_id": entry.experience_id,
        "decision": decision,
        "reason": reason,
        "write": bool(write),
        "source": entry.source,
        "backend": entry.backend,
        "robot_type": entry.robot.robot_type,
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "success": bool(entry.result.get("success", entry.result.get("task_success", False))),
        "write_score": float(entry.memory_gate.write_score or 0.0),
    }
    payload.update({key: value for key, value in extra.items() if value not in ("", None, [], {})})
    return payload
