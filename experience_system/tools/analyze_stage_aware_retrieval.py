"""Analyze stage-aware retrieval behavior for the experience-memory library."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, RetrievalQuery
from source.run_r1pro_memory_policy_smoke import candidates_for_scenario, object_class_for_scenario


STAGE_POLICIES = {
    "anomaly_identification": {
        "description": "retrieve condition-level evidence and visual/perceptual context",
        "include_failed": True,
        "risk_aware": False,
        "memory_partition": "",
        "memory_role": "",
        "critic_status": "",
        "gap_type": "",
    },
    "candidate_generation": {
        "description": "retrieve successful or validated episodes as positive recovery examples",
        "include_failed": False,
        "risk_aware": False,
        "memory_partition": "validated_memory",
        "memory_role": "",
        "critic_status": "",
        "gap_type": "",
    },
    "candidate_ranking": {
        "description": "retrieve risky failure/gap memories for candidate scoring",
        "include_failed": True,
        "risk_aware": True,
        "memory_partition": "",
        "memory_role": "sim_real_gap_memory",
        "critic_status": "",
        "gap_type": "",
    },
    "sandbox_rewrite": {
        "description": "retrieve critic-warning/block cases for rewrite constraints",
        "include_failed": True,
        "risk_aware": True,
        "memory_partition": "",
        "memory_role": "",
        "critic_status": "block",
        "gap_type": "",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stage-aware retrieval diversity and specificity.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def _candidate_steps(scenario: str, candidate_id: str) -> list[str]:
    candidates = candidates_for_scenario(scenario, include_risky=True)
    if candidate_id:
        for candidate in candidates:
            if candidate.candidate_id == candidate_id:
                return list(candidate.steps)
        raise ValueError(f"unknown candidate_id for {scenario}: {candidate_id}")
    default_id = f"{scenario.lower()}_default"
    for candidate in candidates:
        if candidate.candidate_id == default_id:
            return list(candidate.steps)
    return list(candidates[0].steps) if candidates else []


def _match_summary(match: Any) -> dict[str, Any]:
    entry = match.entry
    return {
        "experience_id": entry.experience_id,
        "score": match.score,
        "source": entry.source,
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "task_stage": str(entry.task.get("stage") or ""),
        "success": bool(entry.result.get("success", False)),
        "memory_partition": entry.memory_partition,
        "memory_role": str(entry.memory_tags.get("memory_role") or ""),
        "memory_type": str(entry.memory_tags.get("memory_type") or ""),
        "critic_status": entry.critic_result.overall_status,
        "gap_type": str(entry.sim_real_gap.outcome_gap.get("type") or ""),
        "failure_type": str(entry.failure_taxonomy.get("failure_type") or entry.failure_taxonomy.get("standard_failure_type") or ""),
        "explanation": match.explanation,
    }


def _counter(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key) or "") for item in items).items()))


def _stage_query(
    *,
    stage: str,
    policy: dict[str, Any],
    scenario: str,
    condition: str,
    object_class: str,
    candidate_steps: list[str],
    top_k: int,
) -> RetrievalQuery:
    return RetrievalQuery(
        scenario_id=scenario,
        condition_id=condition,
        robot_type="mobile_dual_arm",
        object_class=object_class,
        task_stage="task_chain",
        memory_partition=str(policy.get("memory_partition") or ""),
        memory_role=str(policy.get("memory_role") or ""),
        critic_status=str(policy.get("critic_status") or ""),
        gap_type=str(policy.get("gap_type") or ""),
        skill_sequence=candidate_steps if stage in {"candidate_ranking", "sandbox_rewrite"} else [],
        include_failed=bool(policy.get("include_failed", True)),
        risk_aware=bool(policy.get("risk_aware", False)),
        update_retrieval_stats=False,
        top_k=top_k,
    )


def _specificity(matches_by_stage: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids_by_stage = {stage: {item["experience_id"] for item in matches} for stage, matches in matches_by_stage.items()}
    all_ids = set().union(*ids_by_stage.values()) if ids_by_stage else set()
    overlap_pairs: dict[str, float] = {}
    stages = list(ids_by_stage)
    for i, left in enumerate(stages):
        for right in stages[i + 1 :]:
            union = ids_by_stage[left] | ids_by_stage[right]
            inter = ids_by_stage[left] & ids_by_stage[right]
            overlap_pairs[f"{left}__{right}"] = round(len(inter) / len(union), 4) if union else 0.0
    mean_overlap = round(sum(overlap_pairs.values()) / len(overlap_pairs), 4) if overlap_pairs else 0.0
    unique_stage_hits = {
        stage: sorted(ids - set().union(*(other_ids for other, other_ids in ids_by_stage.items() if other != stage)))
        for stage, ids in ids_by_stage.items()
    }
    return {
        "unique_retrieved_count": len(all_ids),
        "stage_overlap": overlap_pairs,
        "mean_stage_overlap": mean_overlap,
        "stage_specificity_score": round(1.0 - mean_overlap, 4),
        "unique_stage_hits": unique_stage_hits,
    }


def build_report(
    library: ExperienceLibrary,
    *,
    scenario: str,
    condition: str,
    candidate_id: str,
    top_k: int,
) -> dict[str, Any]:
    object_class = object_class_for_scenario(scenario)
    candidate_steps = _candidate_steps(scenario, candidate_id)
    stages: list[dict[str, Any]] = []
    matches_by_stage: dict[str, list[dict[str, Any]]] = {}

    for stage, policy in STAGE_POLICIES.items():
        query = _stage_query(
            stage=stage,
            policy=policy,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            candidate_steps=candidate_steps,
            top_k=top_k,
        )
        matches = [_match_summary(match) for match in library.query_structured(query)]
        matches_by_stage[stage] = matches
        stages.append({
            "stage": stage,
            "description": policy["description"],
            "query": {
                "scenario_id": query.scenario_id,
                "condition_id": query.condition_id,
                "robot_type": query.robot_type,
                "object_class": query.object_class,
                "task_stage": query.task_stage,
                "memory_partition": query.memory_partition,
                "memory_role": query.memory_role,
                "critic_status": query.critic_status,
                "include_failed": query.include_failed,
                "risk_aware": query.risk_aware,
                "skill_count": len(query.skill_sequence),
                "top_k": query.top_k,
            },
            "match_count": len(matches),
            "memory_role_distribution": _counter(matches, "memory_role"),
            "memory_partition_distribution": _counter(matches, "memory_partition"),
            "critic_status_distribution": _counter(matches, "critic_status"),
            "gap_type_distribution": _counter(matches, "gap_type"),
            "failure_type_distribution": _counter(matches, "failure_type"),
            "matches": matches,
        })

    return {
        "schema_version": "stage_aware_retrieval_report_v1",
        "scenario": scenario,
        "condition": condition,
        "candidate_id": candidate_id or f"{scenario.lower()}_default",
        "candidate_steps": candidate_steps,
        "top_k": top_k,
        "stage_count": len(stages),
        "stages": stages,
        "specificity": _specificity(matches_by_stage),
    }


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    specificity = report.get("specificity") or {}
    for stage in report.get("stages") or []:
        rows.append({
            "scenario": report["scenario"],
            "condition": report["condition"],
            "candidate_id": report["candidate_id"],
            "stage": stage["stage"],
            "match_count": stage["match_count"],
            "memory_roles": json.dumps(stage["memory_role_distribution"], ensure_ascii=False),
            "memory_partitions": json.dumps(stage["memory_partition_distribution"], ensure_ascii=False),
            "critic_statuses": json.dumps(stage["critic_status_distribution"], ensure_ascii=False),
            "gap_types": json.dumps(stage["gap_type_distribution"], ensure_ascii=False),
            "stage_specificity_score": specificity.get("stage_specificity_score", 0.0),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    report = build_report(
        library,
        scenario=args.scenario,
        condition=args.condition,
        candidate_id=args.candidate_id,
        top_k=args.top_k,
    )
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_csv is not None:
        _write_csv(args.save_csv, _csv_rows(report))
    print(json.dumps({
        "stage_count": report["stage_count"],
        "stage_specificity_score": report["specificity"]["stage_specificity_score"],
        "mean_stage_overlap": report["specificity"]["mean_stage_overlap"],
        "save": str(args.save) if args.save else "",
        "save_csv": str(args.save_csv) if args.save_csv else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
