"""Run sandbox rollouts for all R1Pro candidate plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import (
    ExperienceLibrary,
    adjust_candidate_with_lessons,
    build_validated_robot_plan,
    build_stage_planner_context,
    apply_stage_score_adjustment,
    load_lesson_library,
    load_policy_risk_calibration,
    normalize_recovery_plan,
    recovery_plan_prompt,
    recovery_plan_to_candidate,
    invoke_recovery_plan_llm,
    run_stage_retrieval,
    summarize_stage_retrieval,
)
from source.candidate_sandbox import evaluate_candidate_in_sandbox, fuse_memory_and_sandbox, select_sandbox_calibration, summarize_sandbox_fusion
from source.run_r1pro_memory_policy_smoke import (
    candidates_for_scenario,
    derive_candidates_from_planner_input,
    evaluate_candidate,
    object_class_for_scenario,
    selection_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sandbox rollout for candidate recovery plans.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--policy-calibration", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sandbox-weight", type=float, default=0.45)
    parser.add_argument("--include-risky-candidates", action="store_true", help="include ablation-only risky candidates for sandbox critic validation")
    parser.add_argument("--lesson-lib", type=Path, default=None)
    parser.add_argument("--lesson-weight", type=float, default=0.08)
    parser.add_argument("--use-sandbox-calibration", action="store_true", help="consume gap-derived sandbox calibration during rollout scoring")
    parser.add_argument("--sandbox-initial-state", type=Path, default=None, help="optional sandbox_initial_state_v1 JSON used to initialize MuJoCo state")
    parser.add_argument("--model-path", type=Path, default=None, help="optional runtime MuJoCo XML generated from runtime_sandbox_scene_v1")
    parser.add_argument("--use-stage-retrieval", action="store_true", help="apply stage-specific retrieval evidence before sandbox fusion")
    parser.add_argument("--stage-top-k", type=int, default=None, help="override per-stage retrieval top-k")
    parser.add_argument("--stage-support-weight", type=float, default=0.08)
    parser.add_argument("--stage-risk-weight", type=float, default=0.12)
    parser.add_argument("--use-stage-planner-candidate-rewrite", action="store_true", help="derive planner-input candidate rewrites before sandbox rollout")
    parser.add_argument("--generate-recovery-plan", action="store_true", help="generate structured LLM recovery plans for the selected candidate")
    parser.add_argument("--recovery-plan-provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--recovery-plan-model", default="")
    parser.add_argument("--recovery-plan-output", type=Path, default=None, help="optional path to save validated robot plan JSON")
    parser.add_argument("--keyframe-dir", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def _candidate_by_steps(scenario: str, steps: list[str], *, include_risky: bool) -> str:
    for candidate in candidates_for_scenario(scenario, include_risky=include_risky):
        if list(candidate.steps) == list(steps):
            return candidate.candidate_id
    return ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    policy_calibration = load_policy_risk_calibration(args.policy_calibration) if args.policy_calibration else None
    lessons = load_lesson_library(args.lesson_lib) if args.lesson_lib else []
    object_class = object_class_for_scenario(args.scenario)
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
    ) if args.use_sandbox_calibration else None
    sandbox_initial_state = {}
    if args.sandbox_initial_state is not None:
        sandbox_initial_state = json.loads(args.sandbox_initial_state.read_text(encoding="utf-8"))
    base_candidates = candidates_for_scenario(args.scenario, include_risky=args.include_risky_candidates)
    memory_reports = [
        evaluate_candidate(
            library,
            candidate,
            scenario=args.scenario,
            condition=args.condition,
            object_class=object_class,
            top_k=args.top_k,
            risk_aware=True,
            policy_calibration=policy_calibration,
        )
        for candidate in base_candidates
    ]
    planner_generated_candidates = []
    planner_rewrite_summary = []
    recovery_plan_report: dict[str, Any] = {}
    if args.use_stage_planner_candidate_rewrite:
        known_ids = {candidate.candidate_id for candidate in base_candidates}
        known_keys = {tuple(candidate.steps) for candidate in base_candidates}
        seed_reports = []
        for report in memory_reports:
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
            accepted = []
            for item in derived:
                key = tuple(item.steps)
                if item.candidate_id in known_ids or key in known_keys:
                    continue
                known_ids.add(item.candidate_id)
                known_keys.add(key)
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
            memory_reports = [
                evaluate_candidate(
                    library,
                    candidate,
                    scenario=args.scenario,
                    condition=args.condition,
                    object_class=object_class,
                    top_k=args.top_k,
                    risk_aware=True,
                    policy_calibration=policy_calibration,
                )
                for candidate in base_candidates + planner_generated_candidates
            ]
            for report, candidate in zip(memory_reports, base_candidates + planner_generated_candidates):
                report["planner_generated"] = candidate.planner_generated
                report["planner_source_id"] = candidate.planner_source_id
                report["planner_reason"] = candidate.planner_reason
    if args.generate_recovery_plan and memory_reports:
        selected_seed = max(memory_reports, key=selection_rank)
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
            prompt = recovery_plan_prompt(
                scenario=args.scenario,
                condition=args.condition,
                planner_input=planner_context.get("planner_input") or {},
                candidate=seed_candidate,
                candidates=candidate_pool,
            )
            raw_plan = invoke_recovery_plan_llm(
                prompt,
                provider=args.recovery_plan_provider,
                model=args.recovery_plan_model,
            )
            plan = normalize_recovery_plan(
                raw_plan,
                scenario=args.scenario,
                condition=args.condition,
                candidate=seed_candidate,
                candidates=candidate_pool,
                planner_input=planner_context.get("planner_input") or {},
                provider=args.recovery_plan_provider,
                model=args.recovery_plan_model,
            )
            recovery_candidate = recovery_plan_to_candidate(
                plan,
                base_candidate=seed_candidate,
                candidates=candidate_pool,
                candidate_cls=type(seed_candidate),
            )
            if recovery_candidate.candidate_id not in {candidate.candidate_id for candidate in candidate_pool}:
                planner_generated_candidates.append(recovery_candidate)
                memory_reports.append(evaluate_candidate(
                    library,
                    recovery_candidate,
                    scenario=args.scenario,
                    condition=args.condition,
                    object_class=object_class,
                    top_k=args.top_k,
                    risk_aware=True,
                    policy_calibration=policy_calibration,
                ))
                memory_reports[-1]["planner_generated"] = recovery_candidate.planner_generated
                memory_reports[-1]["planner_source_id"] = recovery_candidate.planner_source_id
                memory_reports[-1]["planner_reason"] = recovery_candidate.planner_reason
            recovery_plan_report = {
                "prompt": prompt,
                "recovery_plan": plan,
                "candidate_id": recovery_candidate.candidate_id,
            }
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
        for report in memory_reports
    ]
    if args.use_stage_retrieval:
        for report in memory_reports:
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
    stage_retrieval_summary = summarize_stage_retrieval(memory_reports) if args.use_stage_retrieval else {}
    selected_before = max(memory_reports, key=selection_rank) if memory_reports else None

    candidate_reports = []
    for memory_report in memory_reports:
        candidate_id = str(memory_report["candidate_id"])
        run_candidate_id = candidate_id
        if memory_report.get("planner_generated"):
            run_candidate_id = _candidate_by_steps(
                args.scenario,
                list(memory_report.get("candidate_steps") or []),
                include_risky=args.include_risky_candidates,
            ) or candidate_id
        keyframe_dir = args.keyframe_dir / run_candidate_id if args.keyframe_dir is not None else None
        if not bool(memory_report.get("executable", False)) and run_candidate_id == candidate_id:
            sandbox_report = {
                "scenario": args.scenario,
                "condition": args.condition,
                "candidate_id": candidate_id,
                "control_mode": args.control_mode,
                "sandbox_score": 0.0,
                "raw_sandbox_score": 0.0,
                "decision": "reject",
                "success": False,
                "task_success": False,
                "critic_status": "not_executable",
                "critic_risk_score": 1.0,
                "failure_reason": "planner_generated_candidate_has_no_executable_task_chain_mapping",
                "failed_skills": [],
                "critic_flags": [],
                "skill_trace": [],
                "keyframes": [],
                "experience_id": "",
                "experience_entry": {},
            }
        else:
            sandbox_report = evaluate_candidate_in_sandbox(
                scenario=args.scenario,
                condition=args.condition,
                candidate_id=run_candidate_id,
                control_mode=args.control_mode,
                keyframe_dir=keyframe_dir,
                model_path=args.model_path,
                sandbox_calibration=sandbox_calibration,
                sandbox_initial_state=sandbox_initial_state,
            )
        fused = fuse_memory_and_sandbox(memory_report, sandbox_report, sandbox_weight=args.sandbox_weight)
        candidate_reports.append({
            "candidate_id": candidate_id,
            "run_candidate_id": run_candidate_id,
            "description": memory_report.get("description", ""),
            "executable": bool(memory_report.get("executable", False)),
            "candidate_score": memory_report.get("candidate_score", {}),
            "stage_retrieval": memory_report.get("stage_retrieval", {}),
            "memory": memory_report,
            "sandbox": sandbox_report,
            "fused_score": fused,
        })

    ranked_after = sorted(
        candidate_reports,
        key=lambda item: (
            float(item["fused_score"]["combined_score"]),
            -float(item["fused_score"]["memory_risk_score"]),
            int(bool(item["executable"])),
            str(item["candidate_id"]),
        ),
        reverse=True,
    )
    selected_after = ranked_after[0] if ranked_after else None
    sandbox_summary = summarize_sandbox_fusion(ranked_after)
    selected_recovery_plan = recovery_plan_report.get("recovery_plan") if isinstance(recovery_plan_report, dict) else None
    if (
        selected_after is not None
        and isinstance(selected_recovery_plan, dict)
        and recovery_plan_report.get("candidate_id")
        and str(recovery_plan_report.get("candidate_id")) != str(selected_after.get("candidate_id"))
    ):
        selected_recovery_plan = None
    validated_robot_plan = build_validated_robot_plan(
        scenario=args.scenario,
        condition=args.condition,
        selected_candidate_id=str(selected_after["candidate_id"]) if selected_after is not None else "",
        selected_steps=list(selected_after.get("candidate_steps") or []) if selected_after is not None else [],
        sandbox_report=selected_after.get("sandbox", {}) if selected_after is not None else {},
        fused_score=selected_after.get("fused_score", {}) if selected_after is not None else {},
        recovery_plan=selected_recovery_plan if isinstance(selected_recovery_plan, dict) else None,
    ) if selected_after is not None else {}
    report = {
        "scenario": args.scenario,
        "condition": args.condition,
        "control_mode": args.control_mode,
        "runtime_scene": {
            "enabled": args.model_path is not None,
            "model_path": str(args.model_path) if args.model_path else "",
        },
        "experience_library": str(args.universal_experience_lib),
        "policy_calibration": str(args.policy_calibration) if args.policy_calibration else "",
        "sandbox_weight": args.sandbox_weight,
        "include_risky_candidates": args.include_risky_candidates,
        "lesson_lib": str(args.lesson_lib) if args.lesson_lib else "",
        "lesson_weight": args.lesson_weight,
        "lesson_count": len(lessons),
        "lesson_reports": lesson_reports,
        "stage_retrieval_enabled": args.use_stage_retrieval,
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
        "stage_support_weight": args.stage_support_weight,
        "stage_risk_weight": args.stage_risk_weight,
        "stage_retrieval_summary": stage_retrieval_summary,
        "sandbox_calibration_enabled": args.use_sandbox_calibration,
        "sandbox_calibration": sandbox_calibration or {},
        "sandbox_initial_state": {
            "enabled": bool(sandbox_initial_state),
            "path": str(args.sandbox_initial_state) if args.sandbox_initial_state else "",
            "source_episode_id": str(sandbox_initial_state.get("source_episode_id") or ""),
            "confidence": float(sandbox_initial_state.get("confidence") or 0.0) if sandbox_initial_state else 0.0,
            "missing_fields": list(sandbox_initial_state.get("missing_fields") or []) if sandbox_initial_state else [],
        },
        "selected_before_sandbox": selected_before["candidate_id"] if selected_before else "",
        "selected_after_sandbox": selected_after["candidate_id"] if selected_after else "",
        "candidate_changed_by_sandbox": bool(selected_before and selected_after and selected_before["candidate_id"] != selected_after["candidate_id"]),
        "sandbox_summary": sandbox_summary,
        "recovery_plan_report": recovery_plan_report,
        "validated_robot_plan": validated_robot_plan,
        "validated_robot_plan_output": str(args.recovery_plan_output) if args.recovery_plan_output else "",
        "candidate_count": len(candidate_reports),
        "candidates": ranked_after,
    }
    if args.save is not None:
        _write_json(args.save, report)
    if args.recovery_plan_output is not None and validated_robot_plan:
        _write_json(args.recovery_plan_output, validated_robot_plan)
    print(json.dumps({
        "selected_before_sandbox": report["selected_before_sandbox"],
        "selected_after_sandbox": report["selected_after_sandbox"],
        "candidate_changed_by_sandbox": report["candidate_changed_by_sandbox"],
        "sandbox_summary": sandbox_summary,
        "sandbox_calibration_enabled": report["sandbox_calibration_enabled"],
        "calibration_applied": bool((sandbox_calibration or {}).get("calibration_id")),
        "candidate_count": report["candidate_count"],
        "stage_retrieval_enabled": report["stage_retrieval_enabled"],
        "stage_planner_candidate_rewrite_enabled": report["stage_planner_candidate_rewrite_enabled"],
        "planner_generated_candidate_count": report["planner_generated_candidate_count"],
        "stage_retrieval_summary": stage_retrieval_summary,
        "save": str(args.save) if args.save else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
