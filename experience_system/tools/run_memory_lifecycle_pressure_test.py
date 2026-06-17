"""Build and evaluate an STM/LTM pressure dataset for universal memory lifecycle."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import (
    ExperienceEntry,
    ExperienceLibrary,
    MemoryGate,
    consolidate_memory_lifecycle,
    is_actionable_failure_type,
    set_memory_tier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic STM/LTM pressure test from an existing universal memory library.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--low-value-copies", type=int, default=6)
    parser.add_argument("--hot-success-copies", type=int, default=2)
    parser.add_argument("--hot-retrieval-count", type=int, default=5)
    parser.add_argument("--stm-capacity", type=int, default=10)
    parser.add_argument("--min-retrieval-count", type=int, default=3)
    parser.add_argument("--min-write-score", type=float, default=0.65)
    parser.add_argument("--evict-batch-size", type=int, default=999)
    return parser.parse_args()


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = path if path.is_absolute() else ROOT / path
    return json.loads(source.read_text(encoding="utf-8"))


def _arg_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None and name in {"input", "output_dir"}:
        return value
    if name not in {"input", "output_dir"} and value != default:
        return value
    return config.get(name, value if value is not None else default)


def _success(entry: ExperienceEntry) -> bool:
    return bool(entry.result.get("success", entry.result.get("recovery_success", False)))


def _failure_like(entry: ExperienceEntry) -> bool:
    return (not _success(entry)) or is_actionable_failure_type(entry.failure_taxonomy.get("failure_type"))


def _clone(entry: ExperienceEntry, experience_id: str) -> ExperienceEntry:
    payload = copy.deepcopy(entry.to_dict())
    payload["experience_id"] = experience_id
    payload["created_at"] = ""
    payload["updated_at"] = ""
    payload["metadata"] = dict(payload.get("metadata") or {})
    payload["metadata"].pop("memory_lifecycle", None)
    payload["memory_tags"] = dict(payload.get("memory_tags") or {})
    payload["memory_tags"]["memory_tier"] = "stm"
    return ExperienceEntry(**payload)


def _make_low_value_success(entry: ExperienceEntry, experience_id: str) -> ExperienceEntry:
    clone = _clone(entry, experience_id)
    clone.source = "simulation"
    clone.backend = "mujoco_pressure"
    clone.validation_status = "simulation_only"
    clone.failure_taxonomy = {}
    clone.sim_real_pair = {}
    clone.sim_real_gap.gap_id = ""
    clone.sim_real_gap.gap_score = 0.0
    clone.sim_real_gap.uncertainty = 0.0
    clone.sim_real_gap.outcome_gap = {}
    clone.memory_gate = MemoryGate(write_score=0.05, write_decision="skip", explanation={"pressure_test": "low_value_duplicate_success"})
    clone.memory_tags = {
        "memory_type": "episodic",
        "memory_scope": "pressure_test",
        "memory_role": "low_value_duplicate_success",
        "memory_tier": "stm",
    }
    clone.metadata = {
        "pressure_test_source_id": entry.experience_id,
        "pressure_test_class": "low_value_duplicate_success",
    }
    return clone


def _make_hot_success(entry: ExperienceEntry, experience_id: str, retrieval_count: int) -> ExperienceEntry:
    clone = _make_low_value_success(entry, experience_id)
    clone.memory_gate.write_score = 0.35
    clone.memory_gate.write_decision = "write"
    clone.memory_tags["memory_role"] = "frequently_retrieved_success"
    clone.metadata["pressure_test_class"] = "frequently_retrieved_success"
    set_memory_tier(clone, "stm", reason="pressure_test_seed", retrieval_count=retrieval_count)
    return clone


def build_pressure_entries(
    entries: list[ExperienceEntry],
    *,
    low_value_copies: int,
    hot_success_copies: int,
    hot_retrieval_count: int,
) -> tuple[list[ExperienceEntry], dict[str, Any]]:
    failure_entries = [entry for entry in entries if _failure_like(entry)]
    clean_success_entries = [entry for entry in entries if _success(entry) and not _failure_like(entry)]
    if not clean_success_entries:
        raise ValueError("input library has no clean success entries to duplicate")

    pressure_entries: list[ExperienceEntry] = []
    generated = Counter()

    for index, entry in enumerate(failure_entries):
        clone = _clone(entry, f"pressure_protected_failure_{index:03d}_{entry.experience_id}")
        clone.metadata["pressure_test_source_id"] = entry.experience_id
        clone.metadata["pressure_test_class"] = "protected_failure_or_gap"
        pressure_entries.append(clone)
        generated["protected_failure_or_gap"] += 1

    for index, entry in enumerate(clean_success_entries[: max(hot_success_copies, 0)]):
        pressure_entries.append(_make_hot_success(entry, f"pressure_hot_success_{index:03d}_{entry.experience_id}", hot_retrieval_count))
        generated["frequently_retrieved_success"] += 1

    for copy_index in range(max(low_value_copies, 0)):
        for entry_index, entry in enumerate(clean_success_entries):
            pressure_entries.append(_make_low_value_success(entry, f"pressure_low_success_{copy_index:03d}_{entry_index:03d}_{entry.experience_id}"))
            generated["low_value_duplicate_success"] += 1

    return pressure_entries, {
        "source_entry_count": len(entries),
        "source_failure_like_count": len(failure_entries),
        "source_clean_success_count": len(clean_success_entries),
        "generated_distribution": dict(generated),
        "generated_count": len(pressure_entries),
    }


def _class_of(entry: ExperienceEntry) -> str:
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    return str(metadata.get("pressure_test_class") or "unknown")


def evaluate_pressure_result(before: list[ExperienceEntry], after: list[ExperienceEntry], lifecycle_report: dict[str, Any]) -> dict[str, Any]:
    before_by_id = {entry.experience_id: entry for entry in before}
    after_ids = {entry.experience_id for entry in after}
    evicted_ids = set(lifecycle_report.get("evicted") or [])
    promoted_ids = set(lifecycle_report.get("promoted") or [])

    evicted_by_class = Counter(_class_of(before_by_id[eid]) for eid in evicted_ids if eid in before_by_id)
    promoted_by_class = Counter(_class_of(before_by_id[eid]) for eid in promoted_ids if eid in before_by_id)
    retained_by_class = Counter(_class_of(entry) for entry in after)
    before_by_class = Counter(_class_of(entry) for entry in before)

    protected_before = [entry for entry in before if _class_of(entry) == "protected_failure_or_gap"]
    protected_retained = [entry for entry in protected_before if entry.experience_id in after_ids]
    low_before = [entry for entry in before if _class_of(entry) == "low_value_duplicate_success"]
    low_evicted = [entry for entry in low_before if entry.experience_id in evicted_ids]
    hot_before = [entry for entry in before if _class_of(entry) == "frequently_retrieved_success"]
    hot_promoted = [entry for entry in hot_before if entry.experience_id in promoted_ids]

    return {
        "before_count": len(before),
        "after_count": len(after),
        "removed_count": len(before) - len(after),
        "before_by_class": dict(before_by_class),
        "retained_by_class": dict(retained_by_class),
        "evicted_by_class": dict(evicted_by_class),
        "promoted_by_class": dict(promoted_by_class),
        "protected_retention_rate": round(len(protected_retained) / len(protected_before), 4) if protected_before else 1.0,
        "low_value_eviction_rate": round(len(low_evicted) / len(low_before), 4) if low_before else 0.0,
        "hot_success_promotion_rate": round(len(hot_promoted) / len(hot_before), 4) if hot_before else 0.0,
        "passed": (
            len(protected_retained) == len(protected_before)
            and (not low_before or len(low_evicted) > 0)
            and (not hot_before or len(hot_promoted) == len(hot_before))
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    input_path = _arg_value(args, config, "input", None)
    if input_path is None:
        raise ValueError("--input or config.input is required")
    input_path = Path(str(input_path))
    input_path = input_path if input_path.is_absolute() else ROOT / input_path
    output_value = _arg_value(args, config, "output_dir", ROOT / "results/memory/lifecycle_pressure_test_v1")
    output_dir = Path(str(output_value))
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    low_value_copies = int(_arg_value(args, config, "low_value_copies", 6))
    hot_success_copies = int(_arg_value(args, config, "hot_success_copies", 2))
    hot_retrieval_count = int(_arg_value(args, config, "hot_retrieval_count", 5))
    stm_capacity = int(_arg_value(args, config, "stm_capacity", 10))
    min_retrieval_count = int(_arg_value(args, config, "min_retrieval_count", 3))
    min_write_score = float(_arg_value(args, config, "min_write_score", 0.65))
    evict_batch_size = int(_arg_value(args, config, "evict_batch_size", 999))
    output_dir.mkdir(parents=True, exist_ok=True)

    source_library = ExperienceLibrary.load(input_path)
    pressure_entries, generation_report = build_pressure_entries(
        source_library.entries,
        low_value_copies=low_value_copies,
        hot_success_copies=hot_success_copies,
        hot_retrieval_count=hot_retrieval_count,
    )
    pressure_library = ExperienceLibrary(pressure_entries)
    pressure_input_path = output_dir / "pressure_input_library.json"
    pressure_library.save(pressure_input_path)

    after_entries, lifecycle_report = consolidate_memory_lifecycle(
        pressure_entries,
        stm_capacity=stm_capacity,
        min_retrieval_count=min_retrieval_count,
        min_write_score=min_write_score,
        promote_real=True,
        promote_failures=True,
        promote_validated_success=False,
        evict_batch_size=args.evict_batch_size,
    )
    output_library = ExperienceLibrary(after_entries)
    pressure_output_path = output_dir / "pressure_output_library.json"
    output_library.save(pressure_output_path)

    evaluation = evaluate_pressure_result(pressure_entries, after_entries, lifecycle_report)
    report = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "config": {
            "low_value_copies": low_value_copies,
            "hot_success_copies": hot_success_copies,
            "hot_retrieval_count": hot_retrieval_count,
            "stm_capacity": stm_capacity,
            "min_retrieval_count": min_retrieval_count,
            "min_write_score": min_write_score,
            "evict_batch_size": evict_batch_size,
        },
        "generation": generation_report,
        "lifecycle": lifecycle_report,
        "evaluation": evaluation,
        "artifacts": {
            "pressure_input_library": str(pressure_input_path),
            "pressure_output_library": str(pressure_output_path),
            "report": str(output_dir / "pressure_report.json"),
        },
    }
    _write_json(output_dir / "memory_lifecycle_report.json", lifecycle_report)
    _write_json(output_dir / "pressure_report.json", report)
    print(json.dumps({
        "passed": evaluation["passed"],
        "before_count": evaluation["before_count"],
        "after_count": evaluation["after_count"],
        "removed_count": evaluation["removed_count"],
        "protected_retention_rate": evaluation["protected_retention_rate"],
        "low_value_eviction_rate": evaluation["low_value_eviction_rate"],
        "hot_success_promotion_rate": evaluation["hot_success_promotion_rate"],
        "report": str(output_dir / "pressure_report.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
