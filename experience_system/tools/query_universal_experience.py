"""Query universal experience memory with structured filters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, RetrievalQuery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured query for universal experience memory.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--condition-id", default="")
    parser.add_argument("--robot-type", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--task-stage", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--memory-role", default="")
    parser.add_argument("--memory-type", default="")
    parser.add_argument("--memory-partition", default="")
    parser.add_argument("--failure-type", default="")
    parser.add_argument("--critic-status", default="")
    parser.add_argument("--gap-type", default="")
    parser.add_argument("--object-class", default="")
    parser.add_argument("--target-object", default="")
    parser.add_argument("--plan-signature", default="")
    parser.add_argument("--skill", action="append", default=[])
    parser.add_argument("--required-sensor-modality", action="append", default=[])
    parser.add_argument("--preferred-sensor-modality", action="append", default=[])
    parser.add_argument("--prefer-real-sensor-evidence", action="store_true")
    parser.add_argument("--wrist-force-min", type=float, default=None)
    parser.add_argument("--wrist-force-max", type=float, default=None)
    parser.add_argument("--nearest-obstacle-min", type=float, default=None)
    parser.add_argument("--nearest-obstacle-max", type=float, default=None)
    parser.add_argument("--sensor-evidence-weight", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-failed", action="store_true", default=True)
    parser.add_argument("--exclude-failed", action="store_false", dest="include_failed")
    parser.add_argument("--risk-aware", action="store_true")
    parser.add_argument("--ltm-weight", type=float, default=0.05)
    parser.add_argument("--no-update-retrieval-stats", action="store_true")
    parser.add_argument("--save-updated", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    wrist_force_range = None
    if args.wrist_force_min is not None or args.wrist_force_max is not None:
        wrist_force_range = (
            float("-inf") if args.wrist_force_min is None else args.wrist_force_min,
            float("inf") if args.wrist_force_max is None else args.wrist_force_max,
        )
    nearest_obstacle_range = None
    if args.nearest_obstacle_min is not None or args.nearest_obstacle_max is not None:
        nearest_obstacle_range = (
            float("-inf") if args.nearest_obstacle_min is None else args.nearest_obstacle_min,
            float("inf") if args.nearest_obstacle_max is None else args.nearest_obstacle_max,
        )
    query = RetrievalQuery(
        scenario_id=args.scenario_id,
        condition_id=args.condition_id,
        robot_type=args.robot_type,
        backend=args.backend,
        task_stage=args.task_stage,
        source=args.source,
        memory_role=args.memory_role,
        memory_type=args.memory_type,
        memory_partition=args.memory_partition,
        failure_type=args.failure_type,
        critic_status=args.critic_status,
        gap_type=args.gap_type,
        object_class=args.object_class,
        target_object=args.target_object,
        plan_signature=args.plan_signature,
        skill_sequence=args.skill,
        required_sensor_modalities=args.required_sensor_modality,
        preferred_sensor_modalities=args.preferred_sensor_modality,
        prefer_real_sensor_evidence=args.prefer_real_sensor_evidence,
        wrist_force_norm_range=wrist_force_range,
        nearest_obstacle_distance_range=nearest_obstacle_range,
        sensor_evidence_weight=args.sensor_evidence_weight,
        include_failed=args.include_failed,
        risk_aware=args.risk_aware,
        ltm_weight=args.ltm_weight,
        update_retrieval_stats=not args.no_update_retrieval_stats,
        top_k=args.top_k,
    )
    matches = library.query_structured(query)
    report = {
        "input": str(args.input),
        "match_count": len(matches),
        "matches": [
            {
                "experience_id": match.entry.experience_id,
                "score": match.score,
                "source": match.entry.source,
                "scenario_id": match.entry.scenario_id,
                "condition_id": match.entry.condition_id,
                "robot_type": match.entry.robot.robot_type,
                "backend": match.entry.backend,
                "success": bool(match.entry.result.get("success", False)),
                "memory_partition": match.entry.memory_partition,
                "memory_tier": match.entry.memory_tags.get("memory_tier", ""),
                "retrieval_count": ((match.entry.metadata.get("memory_lifecycle") or {}).get("retrieval_count") if isinstance(match.entry.metadata, dict) else 0),
                "memory_role": match.entry.memory_tags.get("memory_role", ""),
                "critic_status": match.entry.critic_result.overall_status,
                "gap_type": match.entry.sim_real_gap.outcome_gap.get("type", ""),
                "failure_type": match.entry.failure_taxonomy.get("failure_type", ""),
                "raw_failure_type": match.entry.failure_taxonomy.get("raw_failure_type", ""),
                "standard_failure_type": match.entry.failure_taxonomy.get("standard_failure_type", ""),
                "explanation": match.explanation,
            }
            for match in matches
        ],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_updated is not None:
        library.save(args.save_updated)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
