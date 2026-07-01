"""Build a memory-type coverage report for the universal experience library."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


MEMORY_TYPES = (
    "temporal_memory",
    "spatial_memory",
    "episodic_memory",
    "semantic_memory",
    "perceptual_memory",
    "sim_real_gap_memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize temporal/spatial/episodic/semantic/perceptual/sim-real memory coverage."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/memory/galaxea_field_atomic_experience_library.json"),
        help="Universal experience library JSON.",
    )
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=5)
    return parser.parse_args()


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    return [entry for entry in raw_entries if isinstance(entry, dict)]


def _nonempty(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, dict):
        return any(_nonempty(item) for item in value.values())
    if isinstance(value, list):
        return any(_nonempty(item) for item in value)
    return True


def _get(entry: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = entry
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part, default)
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return value


def _count_if(entries: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int:
    return sum(1 for entry in entries if predicate(entry))


def _has_any(entry: dict[str, Any], paths: tuple[str, ...]) -> bool:
    return any(_nonempty(_get(entry, path)) for path in paths)


def _skill_sequence(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in entry.get("skill_sequence", []) if isinstance(item, dict)]


def _has_temporal(entry: dict[str, Any]) -> bool:
    skills = _skill_sequence(entry)
    return bool(skills) or _has_any(
        entry,
        (
            "action_trace",
            "observation_trace",
            "sensor_summary.timestamps",
            "key_slices",
        ),
    )


def _has_spatial(entry: dict[str, Any]) -> bool:
    return _has_any(
        entry,
        (
            "object_state.objects",
            "object_state.spatial_relations",
            "object_state.support_relations",
            "object_state.occupancy",
            "spatial_state",
            "state_before",
            "state_after",
            "result.object_lift",
            "result.place_error",
            "execution_feedback.place_error",
            "sim_real_gap.pose_gap",
        ),
    )


def _has_episodic(entry: dict[str, Any]) -> bool:
    return _has_any(entry, ("scenario", "condition", "task", "result")) and _has_any(
        entry,
        ("source", "backend", "validation_status", "experience_id"),
    )


def _has_semantic(entry: dict[str, Any]) -> bool:
    return _has_any(
        entry,
        (
            "anomaly",
            "failure_taxonomy",
            "memory_tags",
            "retrieval_key",
            "critic_result.rule_flags",
            "critic_result.feedback_for_rewrite",
        ),
    )


def _has_perceptual(entry: dict[str, Any]) -> bool:
    return _has_any(
        entry,
        (
            "keyframes",
            "sensor_summary.sensor_modalities",
            "sensor_summary.raw_refs",
            "sensor_summary.force_torque",
            "sensor_evidence.modalities",
            "sensor_evidence.visual_observation",
            "sensor_evidence.lidar_observation",
            "sensor_evidence.wrist_force_observation",
            "sensor_evidence.evidence_refs",
        ),
    )


def _has_sim_real_gap(entry: dict[str, Any]) -> bool:
    return _has_any(
        entry,
        (
            "sim_real_pair",
            "sim_real_gap.gap_id",
            "sim_real_gap.outcome_gap",
            "sim_real_gap.pose_gap",
            "sim_real_gap.contact_gap",
            "sim_real_gap.perception_gap",
            "sandbox_calibration.calibration_id",
            "sandbox_calibration.source_gap_ids",
            "sandbox_calibration.object_pose_bias",
        ),
    )


PREDICATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "temporal_memory": _has_temporal,
    "spatial_memory": _has_spatial,
    "episodic_memory": _has_episodic,
    "semantic_memory": _has_semantic,
    "perceptual_memory": _has_perceptual,
    "sim_real_gap_memory": _has_sim_real_gap,
}


EVIDENCE_PATHS: dict[str, tuple[str, ...]] = {
    "temporal_memory": (
        "skill_sequence",
        "action_trace",
        "observation_trace",
        "sensor_summary.timestamps",
        "key_slices",
    ),
    "spatial_memory": (
        "object_state.objects",
        "object_state.spatial_relations",
        "object_state.support_relations",
        "object_state.occupancy",
        "spatial_state",
        "state_before",
        "state_after",
        "result.object_lift",
        "result.place_error",
        "sim_real_gap.pose_gap",
    ),
    "episodic_memory": (
        "scenario",
        "condition",
        "task",
        "result",
        "source",
        "backend",
        "validation_status",
    ),
    "semantic_memory": (
        "anomaly",
        "failure_taxonomy",
        "memory_tags",
        "retrieval_key",
        "critic_result.rule_flags",
        "critic_result.feedback_for_rewrite",
    ),
    "perceptual_memory": (
        "keyframes",
        "sensor_summary.sensor_modalities",
        "sensor_summary.raw_refs",
        "sensor_evidence.modalities",
        "sensor_evidence.visual_observation",
        "sensor_evidence.lidar_observation",
        "sensor_evidence.wrist_force_observation",
    ),
    "sim_real_gap_memory": (
        "sim_real_pair",
        "sim_real_gap.gap_id",
        "sim_real_gap.outcome_gap",
        "sim_real_gap.pose_gap",
        "sim_real_gap.contact_gap",
        "sandbox_calibration.calibration_id",
        "sandbox_calibration.source_gap_ids",
        "sandbox_calibration.object_pose_bias",
    ),
}


def _entry_id(entry: dict[str, Any]) -> str:
    return str(entry.get("experience_id") or "")


def _coverage_fields(entry: dict[str, Any], memory_type: str) -> list[str]:
    return [path for path in EVIDENCE_PATHS[memory_type] if _nonempty(_get(entry, path))]


def _entry_profile(entry: dict[str, Any]) -> dict[str, Any]:
    covered = [memory_type for memory_type in MEMORY_TYPES if PREDICATES[memory_type](entry)]
    evidence_fields = {
        memory_type: _coverage_fields(entry, memory_type)
        for memory_type in covered
    }
    return {
        "experience_id": _entry_id(entry),
        "source": entry.get("source", ""),
        "scenario_id": _get(entry, "scenario.scenario_id", ""),
        "condition_id": _get(entry, "condition.condition_id", ""),
        "result_success": _get(entry, "result.success", None),
        "covered_memory_types": covered,
        "covered_memory_type_count": len(covered),
        "evidence_fields": evidence_fields,
    }


def _counter(entries: list[dict[str, Any]], path: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for entry in entries:
        value = _get(entry, path, "")
        if isinstance(value, list):
            for item in value:
                counter[str(item)] += 1
        elif isinstance(value, dict):
            for key in value:
                counter[str(key)] += 1
        else:
            counter[str(value)] += 1
    return dict(counter)


def build_report(entries: list[dict[str, Any]], *, input_path: Path, example_limit: int) -> dict[str, Any]:
    entry_count = len(entries)
    profiles = [_entry_profile(entry) for entry in entries]
    coverage: dict[str, Any] = {}
    for memory_type in MEMORY_TYPES:
        covered_entries = [entry for entry in entries if PREDICATES[memory_type](entry)]
        evidence_field_counts = {
            path: _count_if(entries, lambda entry, path=path: _nonempty(_get(entry, path)))
            for path in EVIDENCE_PATHS[memory_type]
        }
        coverage[memory_type] = {
            "covered_entry_count": len(covered_entries),
            "coverage_rate": round(len(covered_entries) / entry_count, 4) if entry_count else 0.0,
            "evidence_field_counts": evidence_field_counts,
            "example_experience_ids": [_entry_id(entry) for entry in covered_entries[:example_limit]],
            "missing_entry_count": entry_count - len(covered_entries),
        }

    type_count_distribution = Counter(str(profile["covered_memory_type_count"]) for profile in profiles)
    all_type_entries = [
        profile["experience_id"]
        for profile in profiles
        if profile["covered_memory_type_count"] == len(MEMORY_TYPES)
    ]
    report = {
        "schema_version": "memory_type_coverage_report_v1",
        "input": str(input_path),
        "entry_count": entry_count,
        "memory_types": list(MEMORY_TYPES),
        "summary": {
            "covered_memory_type_count_avg": round(
                sum(profile["covered_memory_type_count"] for profile in profiles) / entry_count, 4
            )
            if entry_count
            else 0.0,
            "entries_covering_all_memory_types": len(all_type_entries),
            "entries_covering_all_memory_types_rate": round(len(all_type_entries) / entry_count, 4)
            if entry_count
            else 0.0,
            "covered_memory_type_count_distribution": dict(type_count_distribution),
            "source_distribution": _counter(entries, "source"),
            "scenario_distribution": _counter(entries, "scenario.scenario_id"),
            "memory_tag_type_distribution": _counter(entries, "memory_tags.memory_type"),
            "memory_role_distribution": _counter(entries, "memory_tags.memory_role"),
        },
        "coverage": coverage,
        "entry_profiles": profiles,
        "paper_wording": {
            "safe_claim": (
                "The experience library stores multi-type robot memories, with temporal, spatial, "
                "episodic, semantic, perceptual, and sim-real gap evidence represented by explicit fields."
            ),
            "avoid_claim": (
                "Do not claim broad RoboMME-scale coverage or real-robot sensor validation from this report; "
                "the report measures field-level coverage in the current library."
            ),
        },
    }
    return report


def _md_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Memory Type Coverage Report",
        "",
        "This report measures field-level evidence for six robot memory types in the universal experience library.",
        "",
        "## Summary",
        "",
        f"- Input: `{report['input']}`",
        f"- Entry count: {report['entry_count']}",
        f"- Average covered memory types per entry: {report['summary']['covered_memory_type_count_avg']}",
        f"- Entries covering all six types: {report['summary']['entries_covering_all_memory_types']} "
        f"({report['summary']['entries_covering_all_memory_types_rate']})",
        f"- Source distribution: `{json.dumps(report['summary']['source_distribution'], ensure_ascii=False, sort_keys=True)}`",
        f"- Memory role distribution: `{json.dumps(report['summary']['memory_role_distribution'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Coverage By Memory Type",
        "",
        _md_row(["Memory type", "Covered entries", "Coverage rate", "Main evidence fields", "Example ids"]),
        _md_row(["---", "---", "---", "---", "---"]),
    ]
    for memory_type in MEMORY_TYPES:
        item = report["coverage"][memory_type]
        nonzero_fields = [
            f"{path}={count}"
            for path, count in item["evidence_field_counts"].items()
            if count
        ]
        lines.append(
            _md_row(
                [
                    memory_type,
                    item["covered_entry_count"],
                    item["coverage_rate"],
                    "; ".join(nonzero_fields) or "n/a",
                    ", ".join(item["example_experience_ids"]),
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Paper Wording Boundary",
            "",
            f"- Safe claim: {report['paper_wording']['safe_claim']}",
            f"- Avoid claim: {report['paper_wording']['avoid_claim']}",
            "",
            "## Entry Profiles",
            "",
            _md_row(["Experience id", "Source", "Scenario", "Condition", "Success", "Covered memory types"]),
            _md_row(["---", "---", "---", "---", "---", "---"]),
        ]
    )
    for profile in report["entry_profiles"]:
        lines.append(
            _md_row(
                [
                    profile["experience_id"],
                    profile["source"],
                    profile["scenario_id"],
                    profile["condition_id"],
                    profile["result_success"],
                    ", ".join(profile["covered_memory_types"]),
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entries = _load_entries(args.input)
    report = build_report(entries, input_path=args.input, example_limit=args.example_limit)
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "save_json": str(args.save_json),
                "save_md": str(args.save_md),
                "entry_count": report["entry_count"],
                "covered_memory_type_count_avg": report["summary"]["covered_memory_type_count_avg"],
                "entries_covering_all_memory_types": report["summary"]["entries_covering_all_memory_types"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
