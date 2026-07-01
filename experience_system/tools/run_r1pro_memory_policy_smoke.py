"""Run an R1Pro task through universal memory retrieval and candidate scoring."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import R1ProMujocoAdapter
from experience_core import (
    ExperienceLibrary,
    RetrievalQuery,
    VisualRetrievalIndex,
    TextSemanticRetrievalIndex,
    adjust_candidate_with_lessons,
    apply_stage_planner_guidance,
    apply_stage_score_adjustment,
    build_stage_planner_context,
    build_validated_robot_plan,
    load_policy_risk_calibration,
    load_lesson_library,
    matches_to_tuples,
    invoke_recovery_plan_llm,
    normalize_recovery_plan,
    recovery_plan_prompt,
    recovery_plan_to_candidate,
    run_stage_retrieval,
    semantic_query_text,
    summarize_stage_planner_contexts,
    score_candidate_plan,
    summarize_stage_retrieval,
)
from source.legacy_r1pro.candidate_sandbox import evaluate_candidate_in_sandbox, fuse_memory_and_sandbox, select_sandbox_calibration, summarize_sandbox_fusion
from source.legacy_r1pro.run_r1pro_task_chain import run_task_chain


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    description: str
    steps: list[str]
    executable: bool = True
    planner_generated: bool = False
    planner_source_id: str = ""
    planner_reason: str = ""


G3_DEFAULT = [
    "detect_multiple_objects",
    "select_correct_object",
    "move_to_pregrasp",
    "approach_object",
    "left_gripper_close",
    "left_vertical_lift",
    "detect_place_occupancy",
    "choose_alternate_place",
    "place_object",
    "open_gripper_release",
    "verify_place_zone",
]

G3_PLACE_FIRST = [
    "detect_multiple_objects",
    "select_correct_object",
    "detect_place_occupancy",
    "choose_alternate_place",
    "move_to_pregrasp",
    "approach_object",
    "left_gripper_close",
    "left_vertical_lift",
    "place_object",
    "open_gripper_release",
    "verify_place_zone",
]

G3_CAUTIOUS_PLACE = [
    "detect_multiple_objects",
    "select_correct_object",
    "move_to_pregrasp",
    "approach_object",
    "left_gripper_close",
    "verify_grasp",
    "left_vertical_lift",
    "detect_place_occupancy",
    "choose_alternate_place",
    "place_object",
    "open_gripper_release",
    "verify_place_zone",
]

G3_REACH_RECOVERY = [
    "detect_multiple_objects",
    "select_correct_object",
    "reposition_base_for_reach",
    "adjust_torso_for_reach",
    "move_to_pregrasp",
    "approach_object",
    "left_gripper_close",
    "verify_grasp",
    "left_vertical_lift",
    "detect_place_occupancy",
    "choose_alternate_place",
    "place_object",
    "open_gripper_release",
    "verify_place_zone",
]

def candidates_for_scenario(scenario: str, *, include_risky: bool = False) -> list[CandidatePlan]:
    scenario = scenario.upper()
    if scenario == "G3":
        return [
            CandidatePlan("g3_default", "current executable G3 sorting chain", list(G3_DEFAULT), executable=True),
            CandidatePlan("g3_place_first", "check place occupancy before grasp/lift", list(G3_PLACE_FIRST), executable=True),
            CandidatePlan("g3_cautious_place", "add grasp verification before lift", list(G3_CAUTIOUS_PLACE), executable=True),
            CandidatePlan("g3_reach_recovery", "recover from joint-limit or reach failure before grasp/lift", list(G3_REACH_RECOVERY), executable=False),
        ]
    raise ValueError(f"Unsupported scenario: {scenario}")


def _steps_key(steps: list[str]) -> tuple[str, ...]:
    return tuple(str(item) for item in steps)


def _candidate_catalog(scenario: str, *, include_risky: bool) -> dict[tuple[str, ...], CandidatePlan]:
    return {_steps_key(candidate.steps): candidate for candidate in candidates_for_scenario(scenario, include_risky=include_risky)}


def _replace_skill(steps: list[str], source: str, target: str) -> list[str]:
    return [target if step == source else step for step in steps]


def _ensure_skill_before(steps: list[str], skill: str, anchor: str) -> list[str]:
    items = [step for step in steps if step != skill]
    try:
        index = items.index(anchor)
    except ValueError:
        return list(items)
    items.insert(index, skill)
    return items


def _move_skills_before(steps: list[str], skills: list[str], anchor: str) -> list[str]:
    ordered = [skill for skill in skills if skill]
    items = [step for step in steps if step not in ordered]
    try:
        index = items.index(anchor)
    except ValueError:
        return list(steps)
    for offset, skill in enumerate(ordered):
        items.insert(index + offset, skill)
    return items


def _materialize_planner_candidate(
    scenario: str,
    *,
    base_candidate: CandidatePlan,
    steps: list[str],
    reason: str,
    include_risky: bool,
) -> CandidatePlan | None:
    if _steps_key(steps) == _steps_key(base_candidate.steps):
        return None
    matched = _candidate_catalog(scenario, include_risky=include_risky).get(_steps_key(steps))
    if matched is not None:
        return CandidatePlan(
            candidate_id=matched.candidate_id,
            description=matched.description,
            steps=list(matched.steps),
            executable=matched.executable,
            planner_generated=True,
            planner_source_id=base_candidate.candidate_id,
            planner_reason=reason,
        )
    derived_id = f"{base_candidate.candidate_id}__{reason}"
    return CandidatePlan(
        candidate_id=derived_id,
        description=f"{base_candidate.description} [planner:{reason}]",
        steps=list(steps),
        executable=False,
        planner_generated=True,
        planner_source_id=base_candidate.candidate_id,
        planner_reason=reason,
    )


def derive_candidates_from_planner_input(
    scenario: str,
    candidate: CandidatePlan,
    context: dict[str, Any],
    *,
    include_risky: bool,
) -> list[CandidatePlan]:
    planner_input = context.get("planner_input") or {}
    generation = planner_input.get("generation_guidance") or {}
    ranking = planner_input.get("ranking_guidance") or {}
    rewrite = planner_input.get("rewrite_guidance") or {}
    recommended = {str(item) for item in generation.get("recommended_skills") or [] if str(item)}
    watched = {str(item) for item in rewrite.get("watched_skills") or [] if str(item)}
    risk_score = max(
        float(ranking.get("risk_score") or 0.0),
        float(rewrite.get("rewrite_risk_score") or 0.0),
        float(ranking.get("max_retrieved_risk") or 0.0),
    )
    steps = list(candidate.steps)
    derived: list[CandidatePlan] = []

    if scenario.upper() == "G3":
        if (
            "left_vertical_lift" in steps
            and (
                "verify_grasp" in recommended
                or (risk_score >= 0.45 and watched & {"approach_object", "left_gripper_close"})
            )
        ):
            variant = _ensure_skill_before(steps, "verify_grasp", "left_vertical_lift")
            item = _materialize_planner_candidate(
                scenario,
                base_candidate=candidate,
                steps=variant,
                reason="rewrite_verify_grasp",
                include_risky=include_risky,
            )
            if item is not None:
                derived.append(item)
        if (
            {"detect_place_occupancy", "choose_alternate_place"} <= recommended
            and "move_to_pregrasp" in steps
            and risk_score >= 0.35
        ):
            variant = _move_skills_before(
                steps,
                ["detect_place_occupancy", "choose_alternate_place"],
                "move_to_pregrasp",
            )
            item = _materialize_planner_candidate(
                scenario,
                base_candidate=candidate,
                steps=variant,
                reason="rewrite_place_first",
                include_risky=include_risky,
            )
            if item is not None:
                derived.append(item)

    deduped: list[CandidatePlan] = []
    seen: set[tuple[str, ...]] = set()
    for item in derived:
        key = _steps_key(item.steps)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def object_class_for_scenario(scenario: str) -> str:
    scenario = scenario.upper()
    if scenario == "G3":
        return "sortable_object"
    return "object"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R1Pro memory-policy smoke using universal experience memory.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--save", type=Path, default=None, help="policy smoke report JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--risk-aware", action="store_true", default=True)
    parser.add_argument("--execute-on", choices=["accept", "review", "always"], default="review")
    parser.add_argument("--policy-calibration", type=Path, default=None, help="optional policy risk-transfer calibration JSON")
    parser.add_argument("--visual-index-dir", type=Path, default=None, help="optional CLIP/FAISS visual index directory")
    parser.add_argument("--query-image", type=Path, action="append", default=[], help="query keyframe image for visual retrieval")
    parser.add_argument("--visual-top-k", type=int, default=10)
    parser.add_argument("--visual-weight", type=float, default=0.12)
    parser.add_argument("--use-text-semantic-retrieval", action="store_true", help="use TF-IDF + FAISS text-semantic retrieval as an auxiliary score")
    parser.add_argument("--semantic-backend", choices=["auto", "faiss", "token_overlap"], default="auto")
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--semantic-weight", type=float, default=0.10)
    parser.add_argument("--write-experience", action="store_true", help="write the executed result back to the universal library")
    parser.add_argument("--save-updated-library", action="store_true", help="persist retrieval_count/lifecycle updates after policy retrieval")
    parser.add_argument("--use-sandbox-rollout", action="store_true", help="roll out every candidate in sandbox before selecting")
    parser.add_argument("--sandbox-weight", type=float, default=0.45)
    parser.add_argument("--sandbox-keyframe-dir", type=Path, default=None)
    parser.add_argument("--use-sandbox-calibration", action="store_true", help="consume gap-derived sandbox calibration during sandbox rollout scoring")
    parser.add_argument("--include-risky-candidates", action="store_true", help="include ablation-only risky candidates for sandbox critic validation")
    parser.add_argument("--lesson-lib", type=Path, default=None, help="optional LLM lesson library JSON")
    parser.add_argument("--lesson-weight", type=float, default=0.08)
    parser.add_argument("--use-stage-retrieval", action="store_true", help="apply stage-specific retrieval evidence to candidate ranking")
    parser.add_argument("--stage-top-k", type=int, default=None, help="override per-stage retrieval top-k")
    parser.add_argument("--stage-support-weight", type=float, default=0.08)
    parser.add_argument("--stage-risk-weight", type=float, default=0.12)
    parser.add_argument("--render-stage-context", action="store_true", help="render compact stage-specific planner context for each candidate")
    parser.add_argument("--use-stage-planner-guidance", action="store_true", help="use rendered stage planner context to adjust candidate score/risk before selection")
    parser.add_argument("--stage-planner-guidance-weight", type=float, default=0.10)
    parser.add_argument("--use-stage-planner-candidate-rewrite", action="store_true", help="derive planner-input candidate rewrites/generation variants before final ranking")
    parser.add_argument("--generate-recovery-plan", action="store_true", help="generate a structured LLM recovery plan for the selected candidate")
    parser.add_argument("--recovery-plan-provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--recovery-plan-model", default="")
    parser.add_argument("--recovery-plan-output", type=Path, default=None, help="optional path to save validated robot plan JSON")
    return parser.parse_args()


def should_execute(decision: str, execute_on: str) -> bool:
    if execute_on == "always":
        return True
    if execute_on == "accept":
        return decision == "accept"
    if execute_on == "review":
        return decision in {"accept", "review"}
    return False


def match_report(matches: Any) -> list[dict[str, Any]]:
    return [
        {
            "experience_id": match.entry.experience_id,
            "score": match.score,
            "source": match.entry.source,
            "scenario_id": match.entry.scenario_id,
            "condition_id": match.entry.condition_id,
            "robot_type": match.entry.robot.robot_type,
            "success": bool(match.entry.result.get("success", False)),
            "memory_partition": match.entry.memory_partition,
            "memory_role": match.entry.memory_tags.get("memory_role", ""),
            "critic_status": match.entry.critic_result.overall_status,
            "gap_type": match.entry.sim_real_gap.outcome_gap.get("type", ""),
            "calibration_id": match.entry.sandbox_calibration.calibration_id,
            "explanation": match.explanation,
        }
        for match in matches
    ]


def merge_matches(primary: Any, secondary: Any) -> list[Any]:
    merged = []
    seen = set()
    for match in list(primary) + list(secondary):
        eid = match.entry.experience_id
        if eid in seen:
            continue
        seen.add(eid)
        merged.append(match)
    return merged


def evaluate_candidate(
    library: ExperienceLibrary,
    candidate: CandidatePlan,
    *,
    scenario: str,
    condition: str,
    object_class: str,
    top_k: int,
    risk_aware: bool,
    policy_calibration: dict[str, Any] | None,
    visual_scores: dict[str, float] | None = None,
    visual_weight: float = 0.12,
    semantic_scores: dict[str, float] | None = None,
    semantic_weight: float = 0.10,
    semantic_backend: str = "",
) -> dict[str, Any]:
    visual_scores = visual_scores or {}
    semantic_scores = semantic_scores or {}
    query = RetrievalQuery(
        scenario_id=scenario,
        condition_id=condition,
        robot_type="mobile_dual_arm",
        object_class=object_class,
        skill_sequence=candidate.steps,
        risk_aware=risk_aware,
        visual_scores=visual_scores,
        visual_weight=visual_weight,
        semantic_scores=semantic_scores,
        semantic_weight=semantic_weight,
        top_k=top_k,
    )
    support_matches = library.query_structured(query)
    risk_query = RetrievalQuery(
        scenario_id=scenario,
        condition_id=condition,
        robot_type="mobile_dual_arm",
        object_class=object_class,
        memory_role="sim_real_gap_memory",
        skill_sequence=candidate.steps,
        risk_aware=False,
        visual_scores=visual_scores,
        visual_weight=visual_weight,
        semantic_scores=semantic_scores,
        semantic_weight=semantic_weight,
        top_k=top_k,
    )
    risk_matches = library.query_structured(risk_query)
    matches = merge_matches(support_matches, risk_matches)
    score = score_candidate_plan(
        candidate.steps,
        matches_to_tuples(matches),
        query_context={
            "scenario_id": scenario,
            "condition_id": condition,
            "robot_type": "mobile_dual_arm",
            "object_class": object_class,
            "task_stage": "task_chain",
        },
        policy_calibration=policy_calibration,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "description": candidate.description,
        "executable": candidate.executable,
        "candidate_steps": candidate.steps,
        "retrieval": {
            "support_query": {
                "scenario_id": query.scenario_id,
                "condition_id": query.condition_id,
                "robot_type": query.robot_type,
                "object_class": query.object_class,
                "risk_aware": query.risk_aware,
                "visual_score_count": len(visual_scores),
                "visual_weight": visual_weight,
                "semantic_score_count": len(semantic_scores),
                "semantic_weight": semantic_weight,
                "semantic_backend": semantic_backend,
                "top_k": query.top_k,
            },
            "risk_query": {
                "scenario_id": risk_query.scenario_id,
                "condition_id": risk_query.condition_id,
                "robot_type": risk_query.robot_type,
                "object_class": risk_query.object_class,
                "memory_role": risk_query.memory_role,
                "risk_aware": risk_query.risk_aware,
                "visual_score_count": len(visual_scores),
                "visual_weight": visual_weight,
                "semantic_score_count": len(semantic_scores),
                "semantic_weight": semantic_weight,
                "semantic_backend": semantic_backend,
                "top_k": risk_query.top_k,
            },
            "support_match_count": len(support_matches),
            "risk_match_count": len(risk_matches),
            "match_count": len(matches),
            "matches": match_report(matches),
        },
        "candidate_score": score,
    }


def selection_rank(candidate_report: dict[str, Any]) -> tuple[float, float, int, str]:
    score = candidate_report["candidate_score"]
    return (
        float(score["candidate_score"]),
        -float(score["risk_score"]),
        int(bool(candidate_report["executable"])),
        str(candidate_report["candidate_id"]),
    )


def select_candidate(candidate_reports: list[dict[str, Any]], execute_on: str) -> dict[str, Any] | None:
    viable = [
        report
        for report in candidate_reports
        if should_execute(str(report["candidate_score"]["decision"]), execute_on)
    ]
    if not viable:
        return None

    executable = [report for report in viable if report["executable"]]
    if executable:
        return max(executable, key=selection_rank)
    return max(viable, key=selection_rank)


def sandbox_selection_rank(candidate_report: dict[str, Any]) -> tuple[float, float, int, str]:
    fused = candidate_report.get("fused_score") or {}
    return (
        float(fused.get("combined_score", 0.0)),
        -float(fused.get("memory_risk_score", 0.0)),
        int(bool(candidate_report.get("executable", False))),
        str(candidate_report.get("candidate_id", "")),
    )


def select_sandbox_candidate(candidate_reports: list[dict[str, Any]], execute_on: str) -> dict[str, Any] | None:
    viable = [
        report
        for report in candidate_reports
        if should_execute(str((report.get("fused_score") or {}).get("decision", "reject")), execute_on)
    ]
    if not viable:
        return None
    executable = [report for report in viable if report.get("executable")]
    if executable:
        return max(executable, key=sandbox_selection_rank)
    return max(viable, key=sandbox_selection_rank)


def load_visual_scores(index_dir: Path | None, query_images: list[Path], *, top_k: int) -> dict[str, float]:
    if index_dir is None or not query_images:
        return {}
    index = VisualRetrievalIndex()
    index.load(index_dir)
    paths = [str(path.resolve()) for path in query_images if path.exists()]
    if not paths:
        return {}
    return {experience_id: score for experience_id, score in index.search(paths, top_k=top_k)}


def build_candidate_semantic_scores(
    semantic_index: TextSemanticRetrievalIndex | None,
    candidate: CandidatePlan,
    *,
    scenario: str,
    condition: str,
    object_class: str,
    top_k: int,
) -> tuple[dict[str, float], str]:
    if semantic_index is None:
        return {}, ""
    query_text = semantic_query_text(
        scenario=scenario,
        condition=condition,
        object_class=object_class,
        candidate_id=candidate.candidate_id,
        candidate_description=candidate.description,
        candidate_steps=candidate.steps,
        task_stage="task_chain",
    )
    return semantic_index.search_scores(query_text, top_k=top_k), query_text


def evaluate_candidates_batch(
    library: ExperienceLibrary,
    candidates: list[CandidatePlan],
    *,
    scenario: str,
    condition: str,
    object_class: str,
    top_k: int,
    risk_aware: bool,
    policy_calibration: dict[str, Any] | None,
    visual_scores: dict[str, float],
    visual_weight: float,
    semantic_index: TextSemanticRetrievalIndex | None,
    semantic_top_k: int,
    semantic_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_reports: list[dict[str, Any]] = []
    semantic_runtime_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        semantic_scores, semantic_query = build_candidate_semantic_scores(
            semantic_index,
            candidate,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            top_k=semantic_top_k,
        )
        semantic_runtime_reports.append({
            "candidate_id": candidate.candidate_id,
            "planner_generated": candidate.planner_generated,
            "planner_source_id": candidate.planner_source_id,
            "planner_reason": candidate.planner_reason,
            "query_text": semantic_query,
            "semantic_score_count": len(semantic_scores),
            "top_scores": dict(list(semantic_scores.items())[:5]),
        })
        report = evaluate_candidate(
            library,
            candidate,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            top_k=top_k,
            risk_aware=risk_aware,
            policy_calibration=policy_calibration,
            visual_scores=visual_scores,
            visual_weight=visual_weight,
            semantic_scores=semantic_scores,
            semantic_weight=semantic_weight,
            semantic_backend=semantic_index.backend if semantic_index is not None else "",
        )
        report["planner_generated"] = candidate.planner_generated
        report["planner_source_id"] = candidate.planner_source_id
        report["planner_reason"] = candidate.planner_reason
        candidate_reports.append(report)
    return candidate_reports, semantic_runtime_reports


def generate_recovery_plan_candidate(
    *,
    scenario: str,
    condition: str,
    base_candidate: CandidatePlan,
    candidate_pool: list[CandidatePlan],
    planner_context: dict[str, Any],
    provider: str,
    model: str,
) -> tuple[CandidatePlan | None, dict[str, Any]]:
    prompt = recovery_plan_prompt(
        scenario=scenario,
        condition=condition,
        planner_input=planner_context.get("planner_input") or {},
        candidate=base_candidate,
        candidates=candidate_pool,
    )
    raw_plan = invoke_recovery_plan_llm(
        prompt,
        provider=provider,
        model=model,
    )
    plan = normalize_recovery_plan(
        raw_plan,
        scenario=scenario,
        condition=condition,
        candidate=base_candidate,
        candidates=candidate_pool,
        planner_input=planner_context.get("planner_input") or {},
        provider=provider,
        model=model,
    )
    candidate = recovery_plan_to_candidate(
        plan,
        base_candidate=base_candidate,
        candidates=candidate_pool,
        candidate_cls=CandidatePlan,
    )
    return candidate, {
        "prompt": prompt,
        "recovery_plan": plan,
    }


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    policy_calibration = load_policy_risk_calibration(args.policy_calibration) if args.policy_calibration else None
    lessons = load_lesson_library(args.lesson_lib) if args.lesson_lib else []
    visual_scores = load_visual_scores(args.visual_index_dir, list(args.query_image or []), top_k=args.visual_top_k)
    semantic_index = TextSemanticRetrievalIndex(library.entries, backend=args.semantic_backend) if args.use_text_semantic_retrieval else None
    object_class = object_class_for_scenario(args.scenario)
    if args.use_stage_planner_guidance:
        args.use_stage_retrieval = True
        args.render_stage_context = True
    if args.use_stage_planner_candidate_rewrite:
        args.use_stage_retrieval = True
        args.render_stage_context = True

    base_candidates = candidates_for_scenario(args.scenario, include_risky=args.include_risky_candidates)
    planner_generated_candidates: list[CandidatePlan] = []
    planner_rewrite_summary: list[dict[str, Any]] = []
    candidate_reports, semantic_runtime_reports = evaluate_candidates_batch(
        library,
        base_candidates,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
        top_k=args.top_k,
        risk_aware=args.risk_aware,
        policy_calibration=policy_calibration,
        visual_scores=visual_scores,
        visual_weight=args.visual_weight,
        semantic_index=semantic_index,
        semantic_top_k=args.semantic_top_k,
        semantic_weight=args.semantic_weight,
    )
    if args.use_stage_planner_candidate_rewrite:
        seed_reports = []
        for report in candidate_reports:
            stage_report = run_stage_retrieval(
                library,
                scenario=args.scenario,
                condition=args.condition,
                object_class=object_class,
                candidate_id=str(report["candidate_id"]),
                candidate_steps=list(report["candidate_steps"]),
                top_k=args.stage_top_k,
            )
            context = build_stage_planner_context(
                stage_report,
                scenario=args.scenario,
                condition=args.condition,
                candidate_id=str(report["candidate_id"]),
                candidate_steps=list(report["candidate_steps"]),
                candidate_description=str(report.get("description") or ""),
            )
            report["stage_retrieval"] = stage_report
            report["stage_planner_context"] = context
            seed_reports.append(report)
        known_ids = {candidate.candidate_id for candidate in base_candidates}
        known_keys = {_steps_key(candidate.steps) for candidate in base_candidates}
        for report in seed_reports:
            base_candidate = next((candidate for candidate in base_candidates if candidate.candidate_id == report["candidate_id"]), None)
            if base_candidate is None:
                continue
            derived = derive_candidates_from_planner_input(
                args.scenario,
                base_candidate,
                report.get("stage_planner_context") or {},
                include_risky=args.include_risky_candidates,
            )
            accepted: list[CandidatePlan] = []
            for item in derived:
                if item.candidate_id in known_ids or _steps_key(item.steps) in known_keys:
                    continue
                known_ids.add(item.candidate_id)
                known_keys.add(_steps_key(item.steps))
                accepted.append(item)
            if accepted:
                planner_generated_candidates.extend(accepted)
                planner_rewrite_summary.append({
                    "source_candidate_id": base_candidate.candidate_id,
                    "derived_candidate_ids": [item.candidate_id for item in accepted],
                    "derived_reasons": [item.planner_reason for item in accepted],
                    "planner_input": (report.get("stage_planner_context") or {}).get("planner_input", {}),
                })
        if planner_generated_candidates:
            candidate_reports, semantic_runtime_reports = evaluate_candidates_batch(
                library,
                base_candidates + planner_generated_candidates,
                scenario=args.scenario,
                condition=args.condition,
                object_class=object_class,
                top_k=args.top_k,
                risk_aware=args.risk_aware,
                policy_calibration=policy_calibration,
                visual_scores=visual_scores,
                visual_weight=args.visual_weight,
                semantic_index=semantic_index,
                semantic_top_k=args.semantic_top_k,
                semantic_weight=args.semantic_weight,
            )

    recovery_plan_report: dict[str, Any] = {}
    if args.generate_recovery_plan and candidate_reports:
        selected_seed = max(candidate_reports, key=selection_rank)
        seed_candidate = next((candidate for candidate in base_candidates + planner_generated_candidates if candidate.candidate_id == selected_seed["candidate_id"]), None)
        if seed_candidate is not None:
            stage_report = run_stage_retrieval(
                library,
                scenario=args.scenario,
                condition=args.condition,
                object_class=object_class,
                candidate_id=str(seed_candidate.candidate_id),
                candidate_steps=list(seed_candidate.steps),
                top_k=args.stage_top_k,
            )
            planner_context = build_stage_planner_context(
                stage_report,
                scenario=args.scenario,
                condition=args.condition,
                candidate_id=str(seed_candidate.candidate_id),
                candidate_steps=list(seed_candidate.steps),
                candidate_description=str(seed_candidate.description),
            )
            candidate_pool = base_candidates + planner_generated_candidates
            recovered_candidate, recovery_plan_report = generate_recovery_plan_candidate(
                scenario=args.scenario,
                condition=args.condition,
                base_candidate=seed_candidate,
                candidate_pool=candidate_pool,
                planner_context=planner_context,
                provider=args.recovery_plan_provider,
                model=args.recovery_plan_model,
            )
            if recovered_candidate is not None:
                candidate_pool = candidate_pool + [recovered_candidate]
                candidate_report, semantic_runtime = evaluate_candidates_batch(
                    library,
                    [recovered_candidate],
                    scenario=args.scenario,
                    condition=args.condition,
                    object_class=object_class,
                    top_k=args.top_k,
                    risk_aware=args.risk_aware,
                    policy_calibration=policy_calibration,
                    visual_scores=visual_scores,
                    visual_weight=args.visual_weight,
                    semantic_index=semantic_index,
                    semantic_top_k=args.semantic_top_k,
                    semantic_weight=args.semantic_weight,
                )
                candidate_reports.extend(candidate_report)
                semantic_runtime_reports.extend(semantic_runtime)
                if candidate_report:
                    recovery_plan_report["candidate_id"] = candidate_report[0]["candidate_id"]
                    recovery_plan_report["candidate_score"] = candidate_report[0]["candidate_score"]

    if args.use_stage_retrieval:
        for report in candidate_reports:
            stage_report = run_stage_retrieval(
                library,
                scenario=args.scenario,
                condition=args.condition,
                object_class=object_class,
                candidate_id=str(report["candidate_id"]),
                candidate_steps=list(report["candidate_steps"]),
                top_k=args.stage_top_k,
            )
            apply_stage_score_adjustment(
                report,
                stage_report,
                support_weight=args.stage_support_weight,
                risk_weight=args.stage_risk_weight,
            )
            if args.render_stage_context:
                context = build_stage_planner_context(
                    stage_report,
                    scenario=args.scenario,
                    condition=args.condition,
                    candidate_id=str(report["candidate_id"]),
                    candidate_steps=list(report["candidate_steps"]),
                    candidate_description=str(report.get("description") or ""),
                )
                report["stage_planner_context"] = context
                if args.use_stage_planner_guidance:
                    apply_stage_planner_guidance(
                        report,
                        context,
                        guidance_weight=args.stage_planner_guidance_weight,
                    )
    stage_retrieval_summary = summarize_stage_retrieval(candidate_reports) if args.use_stage_retrieval else {}
    stage_planner_contexts = [
        report["stage_planner_context"]
        for report in candidate_reports
        if report.get("stage_planner_context")
    ]
    stage_planner_context_summary = summarize_stage_planner_contexts(stage_planner_contexts) if stage_planner_contexts else {}
    stage_planner_guidance_reports = [
        report["stage_planner_guidance"]
        for report in candidate_reports
        if report.get("stage_planner_guidance")
    ]
    lesson_reports = [
        {
            "candidate_id": report["candidate_id"],
            **adjust_candidate_with_lessons(
                report,
                lessons,
                scenario=args.scenario,
                condition=args.condition,
                lesson_weight=args.lesson_weight,
            ),
        }
        for report in candidate_reports
    ]
    ranked_candidates = sorted(candidate_reports, key=selection_rank, reverse=True)
    best = ranked_candidates[0] if ranked_candidates else None
    selected = select_candidate(candidate_reports, args.execute_on)
    selected_before_sandbox = selected
    best_before_sandbox = best
    sandbox_enabled = bool(args.use_sandbox_rollout)
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
    ) if args.use_sandbox_calibration else None
    sandbox_reports: list[dict[str, Any]] = []
    sandbox_summary: dict[str, Any] = {}
    if sandbox_enabled:
        for report in candidate_reports:
            candidate_id = str(report["candidate_id"])
            keyframe_dir = args.sandbox_keyframe_dir / candidate_id if args.sandbox_keyframe_dir is not None else None
            sandbox = evaluate_candidate_in_sandbox(
                scenario=args.scenario,
                condition=args.condition,
                candidate_id=candidate_id,
                control_mode=args.control_mode,
                keyframe_dir=keyframe_dir,
                sandbox_calibration=sandbox_calibration,
            )
            fused = fuse_memory_and_sandbox(report, sandbox, sandbox_weight=args.sandbox_weight)
            report["sandbox"] = sandbox
            report["fused_score"] = fused
            sandbox_reports.append({
                "candidate_id": candidate_id,
                "sandbox_score": sandbox["sandbox_score"],
                "sandbox_decision": sandbox["decision"],
                "critic_status": sandbox["critic_status"],
                "critic_risk_score": sandbox["critic_risk_score"],
                "calibration_applied": sandbox.get("calibration_applied", False),
                "calibration_risk_penalty": sandbox.get("calibration_risk_penalty", 0.0),
                "combined_score": fused["combined_score"],
                "fused_decision": fused["decision"],
            })
        ranked_candidates = sorted(candidate_reports, key=sandbox_selection_rank, reverse=True)
        best = ranked_candidates[0] if ranked_candidates else None
        selected = select_sandbox_candidate(candidate_reports, args.execute_on)
        sandbox_summary = summarize_sandbox_fusion(ranked_candidates)
    execute = selected is not None and bool(selected["executable"])

    result_payload: dict[str, Any] | None = None
    written_experience_id = ""
    write_policy: dict[str, Any] = {}
    if execute:
        result = run_task_chain(args.scenario, args.condition, args.control_mode, candidate_id=str(selected["candidate_id"]))
        entry = R1ProMujocoAdapter().normalize_episode(result)
        entry.metadata["selected_candidate_id"] = selected["candidate_id"]
        entry.metadata["selected_candidate_score"] = selected.get("fused_score") or selected["candidate_score"]
        result_payload = result.to_dict()
        if args.write_experience:
            write_policy = library.add_with_policy(entry)
            library.save(args.universal_experience_lib)
            written_experience_id = str(write_policy.get("stored_experience_id") or entry.experience_id)

    display_candidate = selected if selected is not None else best
    display_score = (display_candidate.get("fused_score") or display_candidate["candidate_score"]) if display_candidate is not None else {
        "decision": "reject",
        "candidate_score": 0.0,
        "combined_score": 0.0,
        "risk_score": 1.0,
    }
    selected_recovery_plan = recovery_plan_report.get("recovery_plan") if isinstance(recovery_plan_report, dict) else None
    if (
        selected is not None
        and isinstance(selected_recovery_plan, dict)
        and recovery_plan_report.get("candidate_id")
        and str(recovery_plan_report.get("candidate_id")) != str(selected.get("candidate_id"))
    ):
        selected_recovery_plan = None
    validated_robot_plan = build_validated_robot_plan(
        scenario=args.scenario,
        condition=args.condition,
        selected_candidate_id=str(selected["candidate_id"]) if selected is not None else "",
        selected_steps=list(selected["candidate_steps"]) if selected is not None else [],
        sandbox_report=selected.get("sandbox", {}) if selected is not None else {},
        fused_score=selected.get("fused_score", {}) if selected is not None else {},
        recovery_plan=selected_recovery_plan,
    ) if selected is not None else {}

    report = {
        "scenario": args.scenario,
        "condition": args.condition,
        "control_mode": args.control_mode,
        "candidate_steps": display_candidate["candidate_steps"] if display_candidate is not None else [],
        "retrieval": display_candidate["retrieval"] if display_candidate is not None else {},
        "candidate_score": display_score,
        "sandbox_enabled": sandbox_enabled,
        "sandbox_weight": args.sandbox_weight,
        "sandbox_calibration_enabled": args.use_sandbox_calibration,
        "sandbox_calibration": sandbox_calibration or {},
        "include_risky_candidates": args.include_risky_candidates,
        "lesson_lib": str(args.lesson_lib) if args.lesson_lib else "",
        "lesson_weight": args.lesson_weight,
        "lesson_reports": lesson_reports,
        "stage_retrieval_enabled": args.use_stage_retrieval,
        "stage_support_weight": args.stage_support_weight,
        "stage_risk_weight": args.stage_risk_weight,
        "stage_retrieval_summary": stage_retrieval_summary,
        "stage_context_enabled": bool(stage_planner_contexts),
        "stage_context_summary": stage_planner_context_summary,
        "stage_planner_guidance_enabled": args.use_stage_planner_guidance,
        "stage_planner_guidance_weight": args.stage_planner_guidance_weight,
        "stage_planner_guidance_reports": stage_planner_guidance_reports,
        "stage_planner_candidate_rewrite_enabled": args.use_stage_planner_candidate_rewrite,
        "planner_generated_candidate_count": len(planner_generated_candidates),
        "planner_generated_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "planner_source_id": candidate.planner_source_id,
                "planner_reason": candidate.planner_reason,
                "executable": candidate.executable,
                "steps": list(candidate.steps),
            }
            for candidate in planner_generated_candidates
        ],
        "planner_rewrite_summary": planner_rewrite_summary,
        "sandbox_reports": sandbox_reports,
        "sandbox_summary": sandbox_summary,
        "selected_before_sandbox": {
            "candidate_id": selected_before_sandbox["candidate_id"],
            "decision": selected_before_sandbox["candidate_score"]["decision"],
            "candidate_score": selected_before_sandbox["candidate_score"]["candidate_score"],
        } if selected_before_sandbox is not None else None,
        "selected_after_sandbox": {
            "candidate_id": selected["candidate_id"],
            "decision": (selected.get("fused_score") or selected["candidate_score"])["decision"],
            "candidate_score": (selected.get("fused_score") or selected["candidate_score"]).get("combined_score", selected["candidate_score"]["candidate_score"]),
        } if sandbox_enabled and selected is not None else None,
        "best_before_sandbox": {
            "candidate_id": best_before_sandbox["candidate_id"],
            "decision": best_before_sandbox["candidate_score"]["decision"],
            "candidate_score": best_before_sandbox["candidate_score"]["candidate_score"],
        } if best_before_sandbox is not None else None,
        "candidate_changed_by_sandbox": bool(
            sandbox_enabled
            and selected_before_sandbox is not None
            and selected is not None
            and selected_before_sandbox["candidate_id"] != selected["candidate_id"]
        ),
        "candidates": ranked_candidates,
        "best_candidate": {
            "candidate_id": best["candidate_id"],
            "description": best["description"],
            "executable": best["executable"],
            "decision": (best.get("fused_score") or best["candidate_score"])["decision"],
            "candidate_score": (best.get("fused_score") or best["candidate_score"]).get("combined_score", best["candidate_score"]["candidate_score"]),
            "risk_score": best["candidate_score"]["risk_score"],
        } if best is not None else None,
        "selected_candidate": {
            "candidate_id": selected["candidate_id"],
            "description": selected["description"],
            "executable": selected["executable"],
        } if selected is not None else None,
        "execute_on": args.execute_on,
        "policy_calibration": str(args.policy_calibration) if args.policy_calibration else "",
        "visual_index_dir": str(args.visual_index_dir) if args.visual_index_dir else "",
        "query_images": [str(path) for path in args.query_image],
        "lesson_count": len(lessons),
        "stage_retrieval_enabled": args.use_stage_retrieval,
        "stage_retrieval_summary": stage_retrieval_summary,
        "stage_context_enabled": bool(stage_planner_contexts),
        "stage_context_summary": stage_planner_context_summary,
        "stage_planner_guidance_enabled": args.use_stage_planner_guidance,
        "stage_planner_guidance_weight": args.stage_planner_guidance_weight,
        "stage_planner_guidance_count": len(stage_planner_guidance_reports),
        "stage_planner_candidate_rewrite_enabled": args.use_stage_planner_candidate_rewrite,
        "planner_generated_candidate_count": len(planner_generated_candidates),
        "visual_scores": visual_scores,
        "visual_weight": args.visual_weight,
        "recovery_plan_enabled": args.generate_recovery_plan,
        "recovery_plan_report": recovery_plan_report,
        "validated_robot_plan": validated_robot_plan,
        "validated_robot_plan_output": str(args.recovery_plan_output) if args.recovery_plan_output else "",
        "text_semantic_retrieval_enabled": args.use_text_semantic_retrieval,
        "semantic_backend": semantic_index.backend if semantic_index is not None else "",
        "semantic_backend_fallback_reason": semantic_index.fallback_reason if semantic_index is not None else "",
        "semantic_weight": args.semantic_weight,
        "semantic_top_k": args.semantic_top_k,
        "semantic_runtime_summary": semantic_index.statistics(
            query_count=len(semantic_runtime_reports),
            semantic_hit_count=sum(1 for item in semantic_runtime_reports if item["semantic_score_count"] > 0),
        ) if semantic_index is not None else {},
        "semantic_runtime_reports": semantic_runtime_reports,
        "executed": execute,
        "execution_skip_reason": "" if execute else (
            "selected candidate is not executable in the current runner"
            if selected is not None
            else "no candidate passed the execute_on policy"
        ),
        "execution_result": result_payload,
        "written_experience_id": written_experience_id,
        "write_policy": write_policy,
        "experience_library": str(args.universal_experience_lib),
        "updated_library_saved": bool(args.save_updated_library or (execute and args.write_experience)),
    }

    if args.save_updated_library and not (execute and args.write_experience):
        library.save(args.universal_experience_lib)

    if args.recovery_plan_output is not None and validated_robot_plan:
        args.recovery_plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.recovery_plan_output.write_text(json.dumps(validated_robot_plan, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "best_candidate_id": best["candidate_id"] if best is not None else "",
        "selected_candidate_id": selected["candidate_id"] if selected is not None else "",
        "decision": display_score["decision"],
        "candidate_score": display_score.get("combined_score", display_score.get("candidate_score", 0.0)),
        "risk_score": display_score.get("risk_score", display_score.get("memory_risk_score", 0.0)),
        "candidate_count": len(candidate_reports),
        "match_count": display_candidate["retrieval"]["match_count"] if display_candidate is not None else 0,
        "visual_score_count": len(visual_scores),
        "text_semantic_retrieval_enabled": args.use_text_semantic_retrieval,
        "semantic_backend": semantic_index.backend if semantic_index is not None else "",
        "semantic_score_count": sum(item["semantic_score_count"] for item in semantic_runtime_reports),
        "lesson_count": len(lessons),
        "stage_planner_guidance_enabled": args.use_stage_planner_guidance,
        "stage_planner_guidance_count": len(stage_planner_guidance_reports),
        "stage_planner_candidate_rewrite_enabled": args.use_stage_planner_candidate_rewrite,
        "planner_generated_candidate_count": len(planner_generated_candidates),
        "recovery_plan_enabled": args.generate_recovery_plan,
        "recovery_plan_report": recovery_plan_report,
        "sandbox_enabled": sandbox_enabled,
        "candidate_changed_by_sandbox": report["candidate_changed_by_sandbox"],
        "sandbox_summary": sandbox_summary,
        "executed": execute,
        "execution_skip_reason": report["execution_skip_reason"],
        "written_experience_id": written_experience_id,
        "write_policy": write_policy,
    }, ensure_ascii=False))

    if execute and result_payload is not None and not bool(result_payload.get("success", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
