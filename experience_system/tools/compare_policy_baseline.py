"""Compare default baseline candidates against memory-selected candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, load_policy_risk_calibration
from source.run_r1pro_memory_policy_smoke import (
    CandidatePlan,
    candidates_for_scenario,
    evaluate_candidate,
    load_visual_scores,
    object_class_for_scenario,
    select_candidate,
    selection_rank,
)
from source.run_r1pro_task_chain import run_task_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare default baseline policy against memory-selected policy.")
    parser.add_argument("--scenario", choices=["G3"], action="append", default=[])
    parser.add_argument(
        "--condition",
        choices=["clean", "place_occupied", "grasp_miss", "grasp_slip", "transport_collision", "dual_arm_mismatch"],
        action="append",
        default=[],
    )
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--policy-calibration", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--execute-on", choices=["accept", "review", "always"], default="review")
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--visual-index-dir", type=Path, default=None)
    parser.add_argument("--query-image", type=Path, action="append", default=[])
    parser.add_argument("--visual-top-k", type=int, default=10)
    parser.add_argument("--visual-weight", type=float, default=0.12)
    parser.add_argument("--execute", action="store_true", help="execute baseline and memory-selected candidates")
    parser.add_argument("--save-updated-library", action="store_true", help="persist retrieval_count/lifecycle updates after comparison retrieval")
    return parser.parse_args()


def default_candidate_for_scenario(scenario: str) -> CandidatePlan:
    for candidate in candidates_for_scenario(scenario):
        if candidate.candidate_id == f"{scenario.lower()}_default":
            return candidate
    raise ValueError(f"No default candidate for scenario: {scenario}")


def build_policy_comparison(
    library: ExperienceLibrary,
    *,
    scenario: str,
    condition: str,
    policy_calibration: dict[str, Any] | None = None,
    top_k: int = 5,
    execute_on: str = "review",
    control_mode: str = "physical",
    execute: bool = False,
    visual_scores: dict[str, float] | None = None,
    visual_weight: float = 0.12,
) -> dict[str, Any]:
    object_class = object_class_for_scenario(scenario)
    candidates = [
        evaluate_candidate(
            library,
            candidate,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            top_k=top_k,
            risk_aware=True,
            policy_calibration=policy_calibration,
            visual_scores=visual_scores,
            visual_weight=visual_weight,
        )
        for candidate in candidates_for_scenario(scenario)
    ]
    ranked = sorted(candidates, key=selection_rank, reverse=True)
    selected = select_candidate(candidates, execute_on)
    baseline_plan = default_candidate_for_scenario(scenario)
    baseline = next(item for item in candidates if item["candidate_id"] == baseline_plan.candidate_id)
    memory = selected or (ranked[0] if ranked else None)

    baseline_result = _maybe_execute(scenario, condition, control_mode, baseline["candidate_id"], execute)
    memory_result = _maybe_execute(scenario, condition, control_mode, str(memory["candidate_id"]), execute) if memory else None
    memory_score = memory["candidate_score"] if memory else {}
    baseline_score = baseline["candidate_score"]
    return {
        "scenario": scenario,
        "condition": condition,
        "baseline_candidate": _candidate_summary(baseline),
        "memory_selected_candidate": _candidate_summary(memory) if memory else None,
        "candidate_changed": bool(memory and baseline["candidate_id"] != memory["candidate_id"]),
        "score_delta": round(float(memory_score.get("candidate_score", 0.0)) - float(baseline_score.get("candidate_score", 0.0)), 4),
        "risk_delta": round(float(memory_score.get("risk_score", 0.0)) - float(baseline_score.get("risk_score", 0.0)), 4),
        "memory_decision": memory_score.get("decision", "") if memory else "",
        "baseline_executed": bool(baseline_result),
        "memory_executed": bool(memory_result),
        "baseline_success": baseline_result.get("success") if baseline_result else None,
        "memory_success": memory_result.get("success") if memory_result else None,
        "skill_sequence_diff": _skill_sequence_diff(baseline["candidate_steps"], memory["candidate_steps"] if memory else []),
        "baseline_risk_evidence": _risk_evidence(baseline),
        "memory_risk_evidence": _risk_evidence(memory) if memory else [],
        "visual_scores": visual_scores or {},
        "visual_weight": visual_weight,
        "candidate_ranking": [_candidate_summary(item) for item in ranked],
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    score = candidate["candidate_score"]
    return {
        "candidate_id": candidate["candidate_id"],
        "decision": score["decision"],
        "candidate_score": score["candidate_score"],
        "risk_score": score["risk_score"],
        "support_score": score.get("support_score", 0.0),
        "match_count": candidate["retrieval"]["match_count"],
    }


def _maybe_execute(scenario: str, condition: str, control_mode: str, candidate_id: str, execute: bool) -> dict[str, Any] | None:
    if not execute:
        return None
    result = run_task_chain(scenario, condition, control_mode, candidate_id=candidate_id)
    return result.to_dict()


def _skill_sequence_diff(baseline_steps: list[str], memory_steps: list[str]) -> dict[str, Any]:
    baseline_set = set(baseline_steps)
    memory_set = set(memory_steps)
    first_difference = -1
    for index, (left, right) in enumerate(zip(baseline_steps, memory_steps)):
        if left != right:
            first_difference = index
            break
    if first_difference < 0 and len(baseline_steps) != len(memory_steps):
        first_difference = min(len(baseline_steps), len(memory_steps))
    return {
        "first_difference_index": first_difference,
        "baseline_only": [step for step in baseline_steps if step not in memory_set],
        "memory_only": [step for step in memory_steps if step not in baseline_set],
        "baseline_length": len(baseline_steps),
        "memory_length": len(memory_steps),
    }


def _risk_evidence(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = candidate["candidate_score"].get("evidence") or []
    risky = []
    for item in evidence:
        adjustment = item.get("adjustment") or {}
        transfer = item.get("risk_transfer") or {}
        risk = max(float(adjustment.get("critic_risk", 0.0)), float(adjustment.get("gap_uncertainty", 0.0)))
        if risk <= 0.0 and item.get("success") is True:
            continue
        risky.append({
            "experience_id": item.get("experience_id", ""),
            "scenario_id": item.get("scenario_id", ""),
            "condition_id": item.get("condition_id", ""),
            "source": item.get("source", ""),
            "success": item.get("success"),
            "retrieval_score": item.get("retrieval_score", 0.0),
            "action_overlap": item.get("action_overlap", 0.0),
            "risk_transfer_weight": transfer.get("risk_transfer_weight", 0.0),
            "critic_risk": adjustment.get("critic_risk", 0.0),
            "gap_uncertainty": adjustment.get("gap_uncertainty", 0.0),
        })
    return risky


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    policy_calibration = load_policy_risk_calibration(args.policy_calibration) if args.policy_calibration else None
    visual_scores = load_visual_scores(args.visual_index_dir, list(args.query_image or []), top_k=args.visual_top_k)
    scenarios = args.scenario or ["G3"]
    conditions = args.condition or ["clean", "place_occupied"]
    comparisons = [
        build_policy_comparison(
            library,
            scenario=scenario,
            condition=condition,
            policy_calibration=policy_calibration,
            top_k=args.top_k,
            execute_on=args.execute_on,
            control_mode=args.control_mode,
            execute=args.execute,
            visual_scores=visual_scores,
            visual_weight=args.visual_weight,
        )
        for scenario in scenarios
        for condition in conditions
    ]
    report = {
        "experience_library": str(args.universal_experience_lib),
        "policy_calibration": str(args.policy_calibration) if args.policy_calibration else "",
        "execute": args.execute,
        "visual_index_dir": str(args.visual_index_dir) if args.visual_index_dir else "",
        "query_images": [str(path) for path in args.query_image],
        "visual_score_count": len(visual_scores),
        "comparison_count": len(comparisons),
        "changed_count": sum(1 for item in comparisons if item["candidate_changed"]),
        "comparisons": comparisons,
        "updated_library_saved": bool(args.save_updated_library),
    }
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "comparison_count": report["comparison_count"],
        "changed_count": report["changed_count"],
        "updated_library_saved": report["updated_library_saved"],
        "save": str(args.save) if args.save else "",
    }, ensure_ascii=False))
    if args.save_updated_library:
        library.save(args.universal_experience_lib)


if __name__ == "__main__":
    main()
