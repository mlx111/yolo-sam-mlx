"""STM/LTM lifecycle management for universal experience entries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .failure_taxonomy import is_actionable_failure_type
from .schema import ExperienceEntry, build_retrieval_key, utc_now

STM = "stm"
LTM = "ltm"


def memory_tier(entry: ExperienceEntry) -> str:
    tags = entry.memory_tags if isinstance(entry.memory_tags, dict) else {}
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    lifecycle = metadata.get("memory_lifecycle") if isinstance(metadata.get("memory_lifecycle"), dict) else {}
    tier = str(tags.get("memory_tier") or lifecycle.get("tier") or STM).lower()
    return LTM if tier == LTM else STM


def set_memory_tier(entry: ExperienceEntry, tier: str, *, reason: str, retrieval_count: int = 0) -> None:
    tier = LTM if str(tier).lower() == LTM else STM
    entry.memory_tags = dict(entry.memory_tags or {})
    entry.memory_tags["memory_tier"] = tier
    metadata = dict(entry.metadata or {})
    lifecycle = dict(metadata.get("memory_lifecycle") or {})
    history = list(lifecycle.get("promotion_history") or [])
    previous = str(lifecycle.get("tier") or entry.memory_tags.get("memory_tier") or STM)
    lifecycle.update({
        "tier": tier,
        "updated_at": utc_now(),
        "retrieval_count": int(retrieval_count),
    })
    if previous != tier or not history:
        history.append({
            "action": f"set_{tier}",
            "from": previous,
            "to": tier,
            "reason": reason,
            "retrieval_count": int(retrieval_count),
            "timestamp": utc_now(),
        })
    lifecycle["promotion_history"] = history
    metadata["memory_lifecycle"] = lifecycle
    entry.metadata = metadata
    entry.updated_at = utc_now()
    entry.retrieval_key = build_retrieval_key(entry)


def increment_retrieval_count(entry: ExperienceEntry, amount: int = 1) -> int:
    metadata = dict(entry.metadata or {})
    lifecycle = dict(metadata.get("memory_lifecycle") or {})
    count = int(lifecycle.get("retrieval_count") or 0) + int(amount)
    lifecycle["retrieval_count"] = count
    lifecycle.setdefault("tier", memory_tier(entry))
    lifecycle["last_retrieved_at"] = utc_now()
    metadata["memory_lifecycle"] = lifecycle
    entry.metadata = metadata
    return count


def retrieval_count(entry: ExperienceEntry) -> int:
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    lifecycle = metadata.get("memory_lifecycle") if isinstance(metadata.get("memory_lifecycle"), dict) else {}
    return int(lifecycle.get("retrieval_count") or 0)


def initialize_memory_lifecycle(entries: list[ExperienceEntry]) -> dict[str, Any]:
    report = {"initialized": 0, "ltm": 0, "stm": 0}
    for entry in entries:
        tier = memory_tier(entry)
        metadata = dict(entry.metadata or {})
        lifecycle = dict(metadata.get("memory_lifecycle") or {})
        if not lifecycle:
            lifecycle = {
                "tier": tier,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
                "retrieval_count": 0,
                "promotion_history": [],
            }
            metadata["memory_lifecycle"] = lifecycle
            entry.metadata = metadata
            report["initialized"] += 1
        entry.memory_tags = dict(entry.memory_tags or {})
        entry.memory_tags["memory_tier"] = str(lifecycle.get("tier") or tier)
        if memory_tier(entry) == LTM:
            report["ltm"] += 1
        else:
            report["stm"] += 1
    return report


def should_promote_to_ltm(
    entry: ExperienceEntry,
    *,
    min_retrieval_count: int = 3,
    min_write_score: float = 0.65,
    promote_real: bool = True,
    promote_failures: bool = True,
    promote_validated_success: bool = True,
) -> tuple[bool, str]:
    if memory_tier(entry) == LTM:
        return False, "already_ltm"
    success = bool(entry.result.get("success", entry.result.get("recovery_success", False)))
    failure_type = str(entry.failure_taxonomy.get("failure_type") or "")
    if promote_real and entry.source in {"real", "pseudo_real"}:
        return True, "real_or_pseudo_real_source"
    if promote_failures and (not success or is_actionable_failure_type(failure_type)):
        return True, "failure_or_failure_taxonomy"
    if promote_validated_success and entry.validation_status in {"real_validated", "real_executed", "simulation_validated", "sandbox_validated"} and success:
        return True, "validated_success"
    if float(entry.memory_gate.write_score or 0.0) >= min_write_score:
        return True, "high_write_score"
    if success and retrieval_count(entry) >= min_retrieval_count:
        return True, "frequently_retrieved_success"
    return False, "not_promotion_eligible"


def consolidate_memory_lifecycle(
    entries: list[ExperienceEntry],
    *,
    stm_capacity: int = 30,
    min_retrieval_count: int = 3,
    min_write_score: float = 0.65,
    promote_real: bool = True,
    promote_failures: bool = True,
    promote_validated_success: bool = True,
    evict_batch_size: int = 5,
) -> tuple[list[ExperienceEntry], dict[str, Any]]:
    """Promote valuable STM entries and evict low-value STM overflow."""

    updated = [replace(entry) for entry in entries]
    init_report = initialize_memory_lifecycle(updated)
    report: dict[str, Any] = {
        "input_count": len(entries),
        "stm_capacity": int(stm_capacity),
        "initialized": init_report["initialized"],
        "promoted": [],
        "evicted": [],
        "preserved": [],
        "decisions": [],
    }

    for entry in updated:
        ok, reason = should_promote_to_ltm(
            entry,
            min_retrieval_count=min_retrieval_count,
            min_write_score=min_write_score,
            promote_real=promote_real,
            promote_failures=promote_failures,
            promote_validated_success=promote_validated_success,
        )
        if ok:
            set_memory_tier(entry, LTM, reason=reason, retrieval_count=retrieval_count(entry))
            report["promoted"].append(entry.experience_id)
            report["decisions"].append({"experience_id": entry.experience_id, "decision": "promote", "reason": reason})
        else:
            report["decisions"].append({"experience_id": entry.experience_id, "decision": "keep", "reason": reason, "tier": memory_tier(entry)})

    stm_entries = [entry for entry in updated if memory_tier(entry) == STM]
    over_capacity = len(stm_entries) - int(stm_capacity)
    evict_ids: set[str] = set()
    if over_capacity > 0:
        candidates = sorted(stm_entries, key=_eviction_key)
        for entry in candidates[: min(over_capacity, int(evict_batch_size))]:
            evict_ids.add(entry.experience_id)
        report["evicted"] = sorted(evict_ids)
        report["decisions"].extend(
            {"experience_id": entry_id, "decision": "evict", "reason": "stm_capacity_overflow"}
            for entry_id in sorted(evict_ids)
        )

    output = [entry for entry in updated if entry.experience_id not in evict_ids]
    report["output_count"] = len(output)
    report["removed_count"] = len(entries) - len(output)
    report["stm_count"] = sum(1 for entry in output if memory_tier(entry) == STM)
    report["ltm_count"] = sum(1 for entry in output if memory_tier(entry) == LTM)
    return output, report


def _eviction_key(entry: ExperienceEntry) -> tuple[int, int, float, str]:
    success = bool(entry.result.get("success", entry.result.get("recovery_success", False)))
    write_score = float(entry.memory_gate.write_score or 0.0)
    protected_failure = 1 if not success or is_actionable_failure_type(entry.failure_taxonomy.get("failure_type")) else 0
    return (protected_failure, retrieval_count(entry), write_score, entry.created_at)
