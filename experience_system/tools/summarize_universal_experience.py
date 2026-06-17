"""Summarize a universal experience memory library."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, missing_entry_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize universal experience memory contents and quality signals.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _counter_dict(counter: Counter[str], *, top_k: int | None = None) -> dict[str, int]:
    items = counter.most_common(top_k) if top_k else counter.most_common()
    return {key: value for key, value in items}


def _bucket(value: float) -> str:
    if value <= 0.0:
        return "0"
    if value < 0.25:
        return "0-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return "0.75-1.0"


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _missing_fields(entry: Any) -> list[str]:
    return missing_entry_fields(entry)


def build_summary(library: ExperienceLibrary, *, top_k: int = 10) -> dict[str, Any]:
    source = Counter()
    backend = Counter()
    robot_type = Counter()
    scenario = Counter()
    condition = Counter()
    scenario_condition = Counter()
    memory_partition = Counter()
    validation_status = Counter()
    memory_role = Counter()
    memory_type = Counter()
    write_decision = Counter()
    critic_status = Counter()
    gap_type = Counter()
    sensor_gap_count = 0
    sensor_gap_type = Counter()
    pair_status = Counter()
    calibration_group = Counter()
    object_class = Counter()
    skill_names = Counter()
    rule_flags = Counter()
    failure_type = Counter()
    standard_failure_type = Counter()
    gate_scores: list[float] = []
    gap_scores: list[float] = []
    gap_uncertainties: list[float] = []
    critic_risks: list[float] = []
    calibration_confidences: list[float] = []
    duplicate_ids = Counter()
    missing_by_entry: dict[str, list[str]] = {}
    success_count = 0
    paired_ids: set[str] = set()
    gap_ids: set[str] = set()
    calibration_ids: set[str] = set()
    keyframe_count = 0
    real_ref_count = 0

    for entry in library.entries:
        duplicate_ids[entry.experience_id] += 1
        success_count += int(bool(entry.result.get("success", False)))
        source[entry.source or ""] += 1
        backend[entry.backend or ""] += 1
        robot_type[entry.robot.robot_type or ""] += 1
        scenario[entry.scenario_id or ""] += 1
        condition[entry.condition_id or ""] += 1
        scenario_condition[f"{entry.scenario_id}/{entry.condition_id}"] += 1
        memory_partition[entry.memory_partition or ""] += 1
        validation_status[entry.validation_status or ""] += 1
        memory_role[str(entry.memory_tags.get("memory_role") or "")] += 1
        memory_type[str(entry.memory_tags.get("memory_type") or "")] += 1
        write_decision[entry.memory_gate.write_decision or ""] += 1
        critic_status[entry.critic_result.overall_status or ""] += 1
        object_class[entry.object_state.object_class or ""] += 1
        failure_type[str(entry.failure_taxonomy.get("failure_type") or "")] += 1
        standard_failure_type[str(entry.failure_taxonomy.get("standard_failure_type") or entry.failure_taxonomy.get("failure_type") or "")] += 1
        keyframe_count += len(entry.keyframes)
        real_ref_count += int(bool(entry.real_episode_ref))

        if entry.memory_gate.write_score:
            gate_scores.append(float(entry.memory_gate.write_score))
        if entry.critic_result.critic_risk_score:
            critic_risks.append(float(entry.critic_result.critic_risk_score))
        if entry.sim_real_pair.get("validation_status"):
            pair_status[str(entry.sim_real_pair.get("validation_status"))] += 1
            paired_ids.add(str(entry.sim_real_pair.get("pair_id") or ""))
        if entry.sim_real_gap.gap_id:
            gap_ids.add(entry.sim_real_gap.gap_id)
            gap_type[str(entry.sim_real_gap.outcome_gap.get("type") or "")] += 1
            gap_scores.append(float(entry.sim_real_gap.gap_score))
            gap_uncertainties.append(float(entry.sim_real_gap.uncertainty))
            if str(entry.sim_real_gap.evidence.get("method") or "") == "sensor_derived_gap_v1":
                sensor_gap_count += 1
                sensor_gap_type[str(entry.sim_real_gap.outcome_gap.get("type") or "")] += 1
        if entry.sandbox_calibration.calibration_id:
            calibration_ids.add(entry.sandbox_calibration.calibration_id)
            calibration_confidences.append(float(entry.sandbox_calibration.calibration_confidence))
            group = entry.sandbox_calibration.details.get("group_key") if isinstance(entry.sandbox_calibration.details, dict) else {}
            if isinstance(group, dict):
                calibration_group[
                    f"{group.get('robot_type', '')}/{group.get('scenario_id', '')}/{group.get('condition_id', '')}/{group.get('object_class', '')}"
                ] += 1
        for skill in entry.skill_sequence:
            if skill.name:
                skill_names[skill.name] += 1
        for flag in entry.critic_result.rule_flags:
            if isinstance(flag, dict) and flag.get("rule"):
                rule_flags[str(flag["rule"])] += 1

        missing = _missing_fields(entry)
        if missing:
            missing_by_entry[entry.experience_id] = missing

    duplicate_id_counts = {key: value for key, value in duplicate_ids.items() if value > 1}
    entry_count = len(library.entries)
    return {
        "entry_count": entry_count,
        "success_count": success_count,
        "failure_count": entry_count - success_count,
        "success_rate": round(success_count / entry_count, 4) if entry_count else 0.0,
        "source_distribution": _counter_dict(source),
        "backend_distribution": _counter_dict(backend),
        "robot_type_distribution": _counter_dict(robot_type),
        "scenario_distribution": _counter_dict(scenario),
        "condition_distribution": _counter_dict(condition),
        "scenario_condition_distribution": _counter_dict(scenario_condition),
        "object_class_distribution": _counter_dict(object_class),
        "memory_partition_distribution": _counter_dict(memory_partition),
        "validation_status_distribution": _counter_dict(validation_status),
        "memory_role_distribution": _counter_dict(memory_role),
        "memory_type_distribution": _counter_dict(memory_type),
        "write_decision_distribution": _counter_dict(write_decision),
        "critic_status_distribution": _counter_dict(critic_status),
        "gap_type_distribution": _counter_dict(gap_type),
        "sensor_gap_type_distribution": _counter_dict(sensor_gap_type),
        "pair_status_distribution": _counter_dict(pair_status),
        "calibration_group_distribution": _counter_dict(calibration_group),
        "failure_type_distribution": _counter_dict(failure_type),
        "standard_failure_type_distribution": _counter_dict(standard_failure_type),
        "top_skills": _counter_dict(skill_names, top_k=top_k),
        "top_critic_rules": _counter_dict(rule_flags, top_k=top_k),
        "score_summary": {
            "memory_gate_write_score_avg": _avg(gate_scores),
            "critic_risk_score_avg": _avg(critic_risks),
            "gap_score_avg": _avg(gap_scores),
            "gap_uncertainty_avg": _avg(gap_uncertainties),
            "calibration_confidence_avg": _avg(calibration_confidences),
        },
        "gap_score_buckets": _counter_dict(Counter(_bucket(value) for value in gap_scores)),
        "critic_risk_buckets": _counter_dict(Counter(_bucket(value) for value in critic_risks)),
        "coverage": {
            "pair_count": len({item for item in paired_ids if item}),
            "gap_count": len(gap_ids),
            "sensor_gap_count": sensor_gap_count,
            "calibration_count": len(calibration_ids),
            "keyframe_count": keyframe_count,
            "entries_with_real_episode_ref": real_ref_count,
            "entries_with_missing_required_fields": len(missing_by_entry),
        },
        "quality_issues": {
            "duplicate_experience_ids": duplicate_id_counts,
            "missing_required_fields": missing_by_entry,
        },
    }


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    report = {
        "input": str(args.input),
        "summary": build_summary(library, top_k=args.top_k),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
