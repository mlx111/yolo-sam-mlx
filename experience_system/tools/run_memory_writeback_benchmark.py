"""Run repeated closed-loop memory writeback rounds without mutating the source library."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import R1ProMujocoAdapter
from experience_core import (
    ExperienceLibrary,
    apply_stage_score_adjustment,
    load_lesson_library,
    load_policy_risk_calibration,
    run_stage_retrieval,
    summarize_stage_retrieval,
)
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
    parser = argparse.ArgumentParser(description="Run repeated closed-loop writeback benchmark on an in-memory library copy.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--policy-calibration", type=Path, default=None)
    parser.add_argument("--lesson-lib", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--execute-on", choices=["accept", "review", "always"], default="review")
    parser.add_argument("--include-risky-candidates", action="store_true")
    parser.add_argument("--use-stage-retrieval", action="store_true")
    parser.add_argument("--stage-top-k", type=int, default=None)
    parser.add_argument("--stage-support-weight", type=float, default=0.08)
    parser.add_argument("--stage-risk-weight", type=float, default=0.12)
    parser.add_argument("--use-sandbox-rollout", action="store_true")
    parser.add_argument("--use-sandbox-calibration", action="store_true")
    parser.add_argument("--sandbox-weight", type=float, default=0.45)
    parser.add_argument("--lesson-weight", type=float, default=0.08)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--save-updated-library", type=Path, default=None, help="optional path to persist the benchmark-mutated library copy")
    return parser.parse_args()


def _match_ids(report: dict[str, Any]) -> list[str]:
    retrieval = report.get("retrieval") or {}
    return [str(item.get("experience_id")) for item in retrieval.get("matches") or [] if item.get("experience_id")]


def _score(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return report.get("fused_score") or report.get("candidate_score") or {}


def _score_value(report: dict[str, Any] | None) -> float:
    score = _score(report)
    return float(score.get("combined_score", score.get("candidate_score", 0.0)) or 0.0)


def _risk_value(report: dict[str, Any] | None) -> float:
    if not report:
        return 0.0
    score = report.get("candidate_score") or {}
    fused = report.get("fused_score") or {}
    return float(score.get("risk_score", fused.get("memory_risk_score", 0.0)) or 0.0)


def _candidate_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    score = _score(report)
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "decision": str(score.get("decision") or ""),
        "score": round(_score_value(report), 4),
        "risk_score": round(_risk_value(report), 4),
        "match_count": int((report.get("retrieval") or {}).get("match_count") or 0),
        "retrieved_ids": _match_ids(report),
        "stage_support_score": (report.get("stage_retrieval") or {}).get("stage_support_score", 0.0),
        "stage_risk_score": (report.get("stage_retrieval") or {}).get("stage_risk_score", 0.0),
        "sandbox_score": (report.get("sandbox") or {}).get("sandbox_score", None),
        "critic_status": (report.get("sandbox") or {}).get("critic_status", ""),
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
    use_stage_retrieval: bool,
    stage_top_k: int | None,
    stage_support_weight: float,
    stage_risk_weight: float,
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
    if use_stage_retrieval:
        for report in reports:
            stage_report = run_stage_retrieval(
                library,
                scenario=scenario,
                condition=condition,
                object_class=object_class,
                candidate_id=str(report["candidate_id"]),
                candidate_steps=list(report["candidate_steps"]),
                top_k=stage_top_k,
            )
            apply_stage_score_adjustment(
                report,
                stage_report,
                support_weight=stage_support_weight,
                risk_weight=stage_risk_weight,
            )
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
        "selected_before_sandbox": _candidate_summary(selected_before_sandbox),
        "candidate_ranking": [_candidate_summary(item) for item in reports],
        "lesson_reports": lesson_reports,
        "stage_retrieval_summary": summarize_stage_retrieval(reports) if use_stage_retrieval else {},
        "sandbox_summary": sandbox_summary,
        "sandbox_calibration": sandbox_calibration or {},
    }


def _mean_round(rounds: list[dict[str, Any]], key: str) -> float:
    values = [float(item.get(key) or 0.0) for item in rounds]
    return round(mean(values), 4) if values else 0.0


def _rate(rounds: list[dict[str, Any]], key: str) -> float:
    if not rounds:
        return 0.0
    return round(sum(1 for item in rounds if item.get(key)) / len(rounds), 4)


def main() -> None:
    args = parse_args()
    source_library = ExperienceLibrary.load(args.universal_experience_lib)
    library = ExperienceLibrary(deepcopy(source_library.entries))
    policy_calibration = load_policy_risk_calibration(args.policy_calibration) if args.policy_calibration else None
    lessons = load_lesson_library(args.lesson_lib) if args.lesson_lib else []

    round_reports: list[dict[str, Any]] = []
    rounds = max(int(args.rounds), 0)
    for round_index in range(rounds):
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
            use_stage_retrieval=args.use_stage_retrieval,
            stage_top_k=args.stage_top_k,
            stage_support_weight=args.stage_support_weight,
            stage_risk_weight=args.stage_risk_weight,
            use_sandbox_rollout=args.use_sandbox_rollout,
            use_sandbox_calibration=args.use_sandbox_calibration,
            sandbox_weight=args.sandbox_weight,
            control_mode=args.control_mode,
        )
        selected_id = str(before["selected"].get("candidate_id") or "")
        result = run_task_chain(args.scenario, args.condition, args.control_mode, candidate_id=selected_id)
        entry = R1ProMujocoAdapter().normalize_episode(result)
        entry.metadata["writeback_benchmark"] = True
        entry.metadata["writeback_round_index"] = round_index
        entry.metadata["selected_candidate_id"] = selected_id
        entry.metadata["selected_candidate_score"] = _score(before["selected"])
        entry.memory_tags["memory_role"] = "writeback_benchmark_executed"
        entry.retrieval_key["memory_role"] = "writeback_benchmark_executed"
        entry_count_before_write = len(library.entries)
        write_policy = library.add_with_policy(entry)
        entry_count_after_write = len(library.entries)
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
            use_stage_retrieval=args.use_stage_retrieval,
            stage_top_k=args.stage_top_k,
            stage_support_weight=args.stage_support_weight,
            stage_risk_weight=args.stage_risk_weight,
            use_sandbox_rollout=args.use_sandbox_rollout,
            use_sandbox_calibration=args.use_sandbox_calibration,
            sandbox_weight=args.sandbox_weight,
            control_mode=args.control_mode,
        )
        before_score = float(before["selected_summary"].get("score") or 0.0)
        after_score = float(after["selected_summary"].get("score") or 0.0)
        before_risk = float(before["selected_summary"].get("risk_score") or 0.0)
        after_risk = float(after["selected_summary"].get("risk_score") or 0.0)
        round_report = {
            "round_index": round_index,
            "library_entry_count_before_round": entry_count_before_write,
            "library_entry_count_after_round": entry_count_after_write,
            "entry_count_delta": entry_count_after_write - entry_count_before_write,
            "written_experience_id": written_experience_id,
            "write_policy": write_policy,
            "write_decision": str(write_policy.get("decision") or ""),
            "write_count": int(bool(written_experience_id)),
            "execution_success": bool(result.success),
            "task_success": bool(result.task_success),
            "selected_candidate_changed": before["selected_summary"].get("candidate_id") != after["selected_summary"].get("candidate_id"),
            "selected_candidate_confirmed": before["selected_summary"].get("candidate_id") == after["selected_summary"].get("candidate_id"),
            "score_delta_after_writeback": round(after_score - before_score, 4),
            "risk_delta_after_writeback": round(after_risk - before_risk, 4),
            "before": {
                "selected": before["selected_summary"],
                "selected_before_sandbox": before["selected_before_sandbox"],
                "candidate_ranking": before["candidate_ranking"],
                "stage_retrieval_summary": before["stage_retrieval_summary"],
                "sandbox_summary": before["sandbox_summary"],
            },
            "after": {
                "selected": after["selected_summary"],
                "selected_before_sandbox": after["selected_before_sandbox"],
                "candidate_ranking": after["candidate_ranking"],
                "stage_retrieval_summary": after["stage_retrieval_summary"],
                "sandbox_summary": after["sandbox_summary"],
            },
            **_retrieval_delta(before, after, written_experience_id),
        }
        round_reports.append(round_report)

    write_count = sum(int(item.get("write_count") or 0) for item in round_reports)
    report = {
        "schema_version": "memory_writeback_benchmark_v1",
        "source_library": str(args.universal_experience_lib),
        "scenario": args.scenario,
        "condition": args.condition,
        "control_mode": args.control_mode,
        "round_count": len(round_reports),
        "library_entry_count_before": len(source_library.entries),
        "library_entry_count_after": len(library.entries),
        "entry_count_delta": len(library.entries) - len(source_library.entries),
        "write_count": write_count,
        "write_rate": round(write_count / len(round_reports), 4) if round_reports else 0.0,
        "execution_success_rate": _rate(round_reports, "execution_success"),
        "task_success_rate": _rate(round_reports, "task_success"),
        "new_memory_retrieval_rate": _rate(round_reports, "new_memory_retrieved"),
        "selected_candidate_change_rate": _rate(round_reports, "selected_candidate_changed"),
        "selected_candidate_confirmation_rate": _rate(round_reports, "selected_candidate_confirmed"),
        "score_delta_after_writeback_avg": _mean_round(round_reports, "score_delta_after_writeback"),
        "risk_delta_after_writeback_avg": _mean_round(round_reports, "risk_delta_after_writeback"),
        "retrieval_overlap_before_after_avg": _mean_round(round_reports, "retrieval_overlap_before_after"),
        "include_risky_candidates": args.include_risky_candidates,
        "stage_retrieval_enabled": args.use_stage_retrieval,
        "sandbox_rollout_enabled": args.use_sandbox_rollout,
        "sandbox_calibration_enabled": args.use_sandbox_calibration,
        "rounds": round_reports,
    }
    if args.save_updated_library is not None:
        library.save(args.save_updated_library)
        report["updated_library_saved_to"] = str(args.save_updated_library)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "round_count": report["round_count"],
        "entry_count_delta": report["entry_count_delta"],
        "write_count": report["write_count"],
        "new_memory_retrieval_rate": report["new_memory_retrieval_rate"],
        "selected_candidate_change_rate": report["selected_candidate_change_rate"],
        "selected_candidate_confirmation_rate": report["selected_candidate_confirmation_rate"],
        "score_delta_after_writeback_avg": report["score_delta_after_writeback_avg"],
        "risk_delta_after_writeback_avg": report["risk_delta_after_writeback_avg"],
        "save": str(args.save) if args.save else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
