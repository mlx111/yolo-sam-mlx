"""Demonstrate closed-loop memory writeback and retrieval change."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import R1ProMujocoAdapter
from experience_core import ExperienceLibrary, load_lesson_library, load_policy_risk_calibration
from source.legacy_r1pro.candidate_sandbox import evaluate_candidate_in_sandbox, fuse_memory_and_sandbox, select_sandbox_calibration, summarize_sandbox_fusion
from source.legacy_r1pro.run_r1pro_memory_policy_smoke import (
    adjust_candidate_with_lessons,
    candidates_for_scenario,
    evaluate_candidate,
    object_class_for_scenario,
    sandbox_selection_rank,
    select_candidate,
    select_sandbox_candidate,
    selection_rank,
)
from source.legacy_r1pro.run_r1pro_task_chain import run_task_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a closed-loop writeback demo without mutating the source library.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--policy-calibration", type=Path, default=None)
    parser.add_argument("--lesson-lib", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--execute-on", choices=["accept", "review", "always"], default="review")
    parser.add_argument("--include-risky-candidates", action="store_true")
    parser.add_argument("--use-sandbox-rollout", action="store_true")
    parser.add_argument("--use-sandbox-calibration", action="store_true")
    parser.add_argument("--sandbox-weight", type=float, default=0.45)
    parser.add_argument("--lesson-weight", type=float, default=0.08)
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def _match_ids(report: dict[str, Any]) -> list[str]:
    retrieval = report.get("retrieval") or {}
    return [str(item.get("experience_id")) for item in retrieval.get("matches") or [] if item.get("experience_id")]


def _score(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return report.get("fused_score") or report.get("candidate_score") or {}


def _candidate_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    score = _score(report)
    return {
        "candidate_id": report.get("candidate_id", ""),
        "decision": score.get("decision", ""),
        "score": score.get("combined_score", score.get("candidate_score", 0.0)),
        "risk_score": (report.get("candidate_score") or {}).get("risk_score", score.get("memory_risk_score", 0.0)),
        "match_count": (report.get("retrieval") or {}).get("match_count", 0),
        "retrieved_ids": _match_ids(report),
    }


def _policy_eval(
    library: ExperienceLibrary,
    *,
    scenario: str,
    condition: str,
    top_k: int,
    execute_on: str,
    policy_calibration: dict[str, Any] | None,
    lessons: list[dict[str, Any]],
    lesson_weight: float,
    include_risky_candidates: bool,
    use_sandbox_rollout: bool,
    use_sandbox_calibration: bool,
    sandbox_weight: float,
    control_mode: str,
) -> dict[str, Any]:
    object_class = object_class_for_scenario(scenario)
    reports = [
        evaluate_candidate(
            library,
            candidate,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            top_k=top_k,
            risk_aware=True,
            policy_calibration=policy_calibration,
        )
        for candidate in candidates_for_scenario(scenario, include_risky=include_risky_candidates)
    ]
    lesson_reports = []
    if lessons:
        for report in reports:
            lesson_reports.append({
                "candidate_id": report["candidate_id"],
                **adjust_candidate_with_lessons(
                    report,
                    lessons,
                    scenario=scenario,
                    condition=condition,
                    lesson_weight=lesson_weight,
                ),
            })

    selected_before_sandbox = select_candidate(reports, execute_on) or max(reports, key=selection_rank)
    selected = selected_before_sandbox
    sandbox_summary: dict[str, Any] = {}
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=scenario,
        condition=condition,
        object_class=object_class,
    ) if use_sandbox_calibration else None
    if use_sandbox_rollout:
        for report in reports:
            sandbox = evaluate_candidate_in_sandbox(
                scenario=scenario,
                condition=condition,
                candidate_id=str(report["candidate_id"]),
                control_mode=control_mode,
                sandbox_calibration=sandbox_calibration,
            )
            report["sandbox"] = sandbox
            report["fused_score"] = fuse_memory_and_sandbox(report, sandbox, sandbox_weight=sandbox_weight)
        reports = sorted(reports, key=sandbox_selection_rank, reverse=True)
        selected = select_sandbox_candidate(reports, execute_on) or reports[0]
        sandbox_summary = summarize_sandbox_fusion(reports)
    else:
        reports = sorted(reports, key=selection_rank, reverse=True)

    return {
        "selected": selected,
        "selected_summary": _candidate_summary(selected),
        "candidate_ranking": [_candidate_summary(item) for item in reports],
        "lesson_reports": lesson_reports,
        "sandbox_summary": sandbox_summary,
        "sandbox_calibration": sandbox_calibration or {},
    }


def _retrieval_delta(before: dict[str, Any], after: dict[str, Any], written_id: str) -> dict[str, Any]:
    before_ids = set(before.get("selected_summary", {}).get("retrieved_ids") or [])
    after_ids = set(after.get("selected_summary", {}).get("retrieved_ids") or [])
    union = before_ids | after_ids
    overlap = before_ids & after_ids
    return {
        "retrieval_overlap_before_after": round(len(overlap) / len(union), 4) if union else 0.0,
        "new_retrieved_ids": sorted(after_ids - before_ids),
        "dropped_retrieved_ids": sorted(before_ids - after_ids),
        "new_memory_retrieved": bool(written_id and written_id in after_ids),
    }


def main() -> None:
    args = parse_args()
    source_library = ExperienceLibrary.load(args.universal_experience_lib)
    library = ExperienceLibrary(deepcopy(source_library.entries))
    policy_calibration = load_policy_risk_calibration(args.policy_calibration) if args.policy_calibration else None
    lessons = load_lesson_library(args.lesson_lib) if args.lesson_lib else []

    before = _policy_eval(
        library,
        scenario=args.scenario,
        condition=args.condition,
        top_k=args.top_k,
        execute_on=args.execute_on,
        policy_calibration=policy_calibration,
        lessons=lessons,
        lesson_weight=args.lesson_weight,
        include_risky_candidates=args.include_risky_candidates,
        use_sandbox_rollout=args.use_sandbox_rollout,
        use_sandbox_calibration=args.use_sandbox_calibration,
        sandbox_weight=args.sandbox_weight,
        control_mode=args.control_mode,
    )
    selected_id = str(before["selected"].get("candidate_id") or "")
    result = run_task_chain(args.scenario, args.condition, args.control_mode, candidate_id=selected_id)
    entry = R1ProMujocoAdapter().normalize_episode(result)
    entry.metadata["writeback_demo"] = True
    entry.metadata["selected_candidate_id"] = selected_id
    entry.metadata["selected_candidate_score"] = _score(before["selected"])
    entry.memory_tags["memory_role"] = "writeback_demo_executed"
    entry.retrieval_key["memory_role"] = "writeback_demo_executed"
    write_policy = library.add_with_policy(entry)
    written_experience_id = str(write_policy.get("stored_experience_id") or "")

    after = _policy_eval(
        library,
        scenario=args.scenario,
        condition=args.condition,
        top_k=args.top_k,
        execute_on=args.execute_on,
        policy_calibration=policy_calibration,
        lessons=lessons,
        lesson_weight=args.lesson_weight,
        include_risky_candidates=args.include_risky_candidates,
        use_sandbox_rollout=args.use_sandbox_rollout,
        use_sandbox_calibration=args.use_sandbox_calibration,
        sandbox_weight=args.sandbox_weight,
        control_mode=args.control_mode,
    )
    before_score = float(before["selected_summary"].get("score") or 0.0)
    after_score = float(after["selected_summary"].get("score") or 0.0)
    before_risk = float(before["selected_summary"].get("risk_score") or 0.0)
    after_risk = float(after["selected_summary"].get("risk_score") or 0.0)
    report = {
        "schema_version": "memory_writeback_demo_v1",
        "source_library": str(args.universal_experience_lib),
        "scenario": args.scenario,
        "condition": args.condition,
        "control_mode": args.control_mode,
        "library_entry_count_before": len(source_library.entries),
        "library_entry_count_after": len(library.entries),
        "entry_count_delta": len(library.entries) - len(source_library.entries),
        "written_experience_id": written_experience_id,
        "write_policy": write_policy,
        "execution_success": bool(result.success),
        "execution_result": result.to_dict(),
        "before": {
            "selected": before["selected_summary"],
            "candidate_ranking": before["candidate_ranking"],
            "sandbox_summary": before["sandbox_summary"],
        },
        "after": {
            "selected": after["selected_summary"],
            "candidate_ranking": after["candidate_ranking"],
            "sandbox_summary": after["sandbox_summary"],
        },
        "selected_candidate_changed": before["selected_summary"].get("candidate_id") != after["selected_summary"].get("candidate_id"),
        "score_delta_after_writeback": round(after_score - before_score, 4),
        "risk_delta_after_writeback": round(after_risk - before_risk, 4),
        **_retrieval_delta(before, after, written_experience_id),
    }
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "written_experience_id": written_experience_id,
        "write_decision": write_policy.get("decision", ""),
        "entry_count_delta": report["entry_count_delta"],
        "selected_candidate_changed": report["selected_candidate_changed"],
        "score_delta_after_writeback": report["score_delta_after_writeback"],
        "risk_delta_after_writeback": report["risk_delta_after_writeback"],
        "new_memory_retrieved": report["new_memory_retrieved"],
        "retrieval_overlap_before_after": report["retrieval_overlap_before_after"],
        "save": str(args.save) if args.save else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
