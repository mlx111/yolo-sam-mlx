"""Run LLM recovery-plan generation, sandbox critic feedback, and rewrite loop."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import (
    DryRunSkillExecutor,
    ExperienceLibrary,
    build_stage_planner_context,
    build_recovery_parameter_priors,
    build_validated_robot_plan,
    default_r1pro_skill_registry,
    execute_validated_robot_plan,
    invoke_recovery_plan_llm,
    normalize_recovery_plan,
    recovery_plan_prompt,
    recovery_plan_to_candidate,
    run_stage_retrieval,
    validate_recovery_plan_semantics,
    writeback_sandbox_reports,
)
from source.candidate_sandbox import evaluate_candidate_in_sandbox, evaluate_plan_in_sandbox, select_sandbox_calibration
from source.run_r1pro_memory_policy_smoke import CandidatePlan, candidates_for_scenario, object_class_for_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recovery-plan LLM -> sandbox critic -> rewrite -> recheck loop.")
    parser.add_argument("--scenario", choices=["G3"], required=True)
    parser.add_argument("--condition", choices=["clean", "place_occupied"], required=True)
    parser.add_argument("--candidate-id", default="", help="seed candidate id; defaults to first executable candidate")
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--sandbox-initial-state", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None, help="optional runtime MuJoCo XML generated from runtime_sandbox_scene_v1")
    parser.add_argument("--use-sandbox-calibration", action="store_true")
    parser.add_argument("--stage-top-k", type=int, default=None)
    parser.add_argument("--max-rewrite-rounds", type=int, default=2)
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--dry-run-llm", action="store_true", help="use deterministic mock LLM plans for offline smoke testing")
    parser.add_argument("--mock-plan-variant", choices=["candidate", "extra_safe_transport", "duplicate_verify", "place_before_grasp"], default="candidate")
    parser.add_argument("--use-general-plan-executor", action="store_true", help="execute LLM step graph directly instead of mapping to a fixed candidate")
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--keyframe-dir", type=Path, default=None)
    parser.add_argument("--save-plan", type=Path, required=True)
    parser.add_argument("--save-report", type=Path, required=True)
    parser.add_argument("--writeback-sandbox-experiences", action="store_true", help="write each sandbox attempt into an experience-library copy")
    parser.add_argument("--writeback-library-output", type=Path, default=None, help="path for the updated experience library; defaults to --universal-experience-lib when writeback is enabled")
    parser.add_argument("--writeback-merge-duplicates", action="store_true", help="allow write policy to merge duplicate low-risk successes")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_by_id(scenario: str, candidate_id: str) -> CandidatePlan:
    candidates = candidates_for_scenario(scenario, include_risky=True)
    if candidate_id:
        for candidate in candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
    for candidate in candidates:
        if candidate.executable:
            return candidate
    return candidates[0]


def _candidate_by_steps(scenario: str, steps: list[str]) -> CandidatePlan | None:
    for candidate in candidates_for_scenario(scenario, include_risky=True):
        if list(candidate.steps) == list(steps):
            return candidate
    return None


def _dedupe_consecutive(steps: list[str]) -> list[str]:
    deduped: list[str] = []
    for step in steps:
        if deduped and deduped[-1] == step:
            continue
        deduped.append(step)
    return deduped


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    index = 0
    for item in haystack:
        if index < len(needle) and needle[index] == item:
            index += 1
    return index == len(needle)


def _comes_before(steps: list[str], first: str, second: str) -> bool:
    if first not in steps or second not in steps:
        return False
    return steps.index(first) < steps.index(second)


def _canonical_candidate_mapping(scenario: str, steps: list[str]) -> tuple[CandidatePlan | None, dict[str, Any]]:
    """Map a valid LLM skill graph to the closest executable benchmark runner."""

    scenario = scenario.upper()
    original_steps = list(steps)
    normalized = _dedupe_consecutive([str(step) for step in steps if str(step)])
    exact = _candidate_by_steps(scenario, normalized)
    if exact is not None:
        return exact, {
            "mapping_status": "exact",
            "original_steps": original_steps,
            "normalized_steps": normalized,
            "mapped_candidate_id": exact.candidate_id,
            "reason": "exact normalized step sequence match",
        }

    candidates = [candidate for candidate in candidates_for_scenario(scenario, include_risky=True) if candidate.executable]
    best: tuple[float, CandidatePlan] | None = None
    normalized_set = set(normalized)
    for candidate in candidates:
        candidate_steps = list(candidate.steps)
        candidate_set = set(candidate_steps)
        overlap = len(normalized_set & candidate_set) / max(len(candidate_set), 1)
        order_bonus = 0.2 if _is_subsequence([step for step in normalized if step in candidate_set], candidate_steps) else 0.0
        if scenario == "G3":
            if "verify_grasp" in normalized and candidate.candidate_id == "g3_cautious_place":
                overlap += 0.25
            if _comes_before(normalized, "detect_place_occupancy", "move_to_pregrasp") and candidate.candidate_id == "g3_place_first":
                overlap += 0.25
        score = overlap + order_bonus
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None or best[0] < 0.72:
        return None, {
            "mapping_status": "failed",
            "original_steps": original_steps,
            "normalized_steps": normalized,
            "mapped_candidate_id": "",
            "mapping_score": round(best[0], 4) if best else 0.0,
            "reason": "no executable candidate close enough to LLM step graph",
        }
    return best[1], {
        "mapping_status": "canonicalized",
        "original_steps": original_steps,
        "normalized_steps": normalized,
        "mapped_candidate_id": best[1].candidate_id,
        "mapped_steps": list(best[1].steps),
        "mapping_score": round(best[0], 4),
        "reason": "mapped LLM step graph to closest executable scenario runner",
    }


def _critic_feedback(sandbox_report: dict[str, Any]) -> dict[str, Any]:
    contact = sandbox_report.get("contact_stability") if isinstance(sandbox_report.get("contact_stability"), dict) else {}
    trace = sandbox_report.get("trajectory_trace") if isinstance(sandbox_report.get("trajectory_trace"), dict) else {}
    failure_diagnosis = sandbox_report.get("failure_diagnosis") if isinstance(sandbox_report.get("failure_diagnosis"), dict) else {}
    trace_feedback = _trace_feedback_summary(trace)
    return {
        "critic_status": str(sandbox_report.get("critic_status") or ""),
        "decision": str(sandbox_report.get("decision") or ""),
        "sandbox_score": float(sandbox_report.get("sandbox_score") or 0.0),
        "critic_risk_score": float(sandbox_report.get("critic_risk_score") or 0.0),
        "critic_flags": list(sandbox_report.get("critic_flags") or []),
        "failed_skills": list(sandbox_report.get("failed_skills") or []),
        "failure_reason": str(sandbox_report.get("failure_reason") or ""),
        "failure_diagnosis": failure_diagnosis,
        "primary_failure_reason": str(failure_diagnosis.get("primary_reason") or ""),
        "joint_limit_violation": bool(failure_diagnosis.get("joint_limit_violation", False)),
        "final_error": float(failure_diagnosis.get("final_error") or 0.0),
        "max_joint_tracking_error": float(failure_diagnosis.get("max_joint_tracking_error") or 0.0),
        "contact_after_close": contact.get("contact_after_close"),
        "contact_lost_step": contact.get("contact_lost_step"),
        "object_slip_distance": contact.get("object_slip_distance"),
        "object_lift_slip_distance": contact.get("object_lift_slip_distance"),
        "wrist_force_proxy": contact.get("wrist_force_proxy"),
        "grasp_stability_score": contact.get("grasp_stability_score"),
        "trajectory_summary_path": trace.get("summary_path", ""),
        "trajectory_trace_path": trace.get("trace_path", ""),
        "trace_feedback": trace_feedback,
        "trace_feedback_text": trace_feedback.get("trace_feedback_text", ""),
        "rewrite_instruction": "Revise the plan to reduce failed skills, critic flags, contact loss, slip, and unsafe motion while using only allowed skills.",
    }


def _load_json_object(path: str) -> dict[str, Any]:
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _vector_distance(a: Any, b: Any) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 3 or len(b) < 3:
        return 0.0
    try:
        return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))
    except Exception:
        return 0.0


def _trace_feedback_summary(trace: dict[str, Any]) -> dict[str, Any]:
    summary = _load_json_object(str(trace.get("summary_path") or ""))
    trace_path = str(trace.get("trace_path") or summary.get("trace_path") or "")
    sample_count = int(summary.get("sample_count") or trace.get("sample_count") or 0)
    object_path_length = float(summary.get("object_path_length") or 0.0)
    max_contact_count = int(summary.get("max_contact_count") or 0)
    min_contact_count = int(summary.get("min_contact_count") or 0)
    min_joint_limit_margin = float(summary.get("min_joint_limit_margin") or 0.0)

    first_contact_loss_hint = ""
    high_motion_speed_hint = ""
    previous_object_position: list[float] | None = None
    max_sample_motion = 0.0
    if trace_path:
        try:
            with Path(trace_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    contact_count = int(sample.get("contact_count") or 0)
                    skill = str(sample.get("skill") or "")
                    if not first_contact_loss_hint and max_contact_count > 0 and contact_count == 0:
                        first_contact_loss_hint = f"contact dropped to zero during {skill or 'unknown_skill'}"
                    object_pose = sample.get("object_pose") if isinstance(sample.get("object_pose"), dict) else {}
                    position = object_pose.get("position")
                    if isinstance(position, list) and len(position) >= 3:
                        if previous_object_position is not None:
                            max_sample_motion = max(max_sample_motion, _vector_distance(previous_object_position, position))
                        previous_object_position = [float(position[0]), float(position[1]), float(position[2])]
        except OSError:
            pass
    if max_sample_motion > 0.08:
        high_motion_speed_hint = f"large object displacement between trace samples: {max_sample_motion:.3f} m"

    notes: list[str] = []
    if sample_count:
        notes.append(f"{sample_count} trace samples")
    if object_path_length > 0.8:
        notes.append(f"long object path {object_path_length:.3f} m; prefer slower/segmented motion")
    if max_contact_count > 0 and min_contact_count == 0:
        notes.append("contact reached zero in rollout; preserve grasp before transport/place")
    if min_joint_limit_margin <= 0.001:
        notes.append("joint limit margin reached near zero; avoid aggressive arm poses")
    if first_contact_loss_hint:
        notes.append(first_contact_loss_hint)
    if high_motion_speed_hint:
        notes.append(high_motion_speed_hint)
    return {
        "trace_sample_count": sample_count,
        "object_path_length": round(object_path_length, 6),
        "max_contact_count": max_contact_count,
        "min_contact_count": min_contact_count,
        "min_joint_limit_margin": round(min_joint_limit_margin, 6),
        "first_contact_loss_hint": first_contact_loss_hint,
        "high_motion_speed_hint": high_motion_speed_hint,
        "trace_feedback_text": "; ".join(notes),
    }


def _augment_planner_input(planner_input: dict[str, Any], feedback_history: list[dict[str, Any]]) -> dict[str, Any]:
    updated = json.loads(json.dumps(planner_input, ensure_ascii=False))
    rewrite = updated.setdefault("rewrite_guidance", {})
    rewrite["sandbox_critic_feedback_history"] = feedback_history
    rewrite["latest_sandbox_critic_feedback"] = feedback_history[-1] if feedback_history else {}
    rewrite["rewrite_required"] = bool(feedback_history)
    return updated


def _attach_parameter_priors(
    planner_input: dict[str, Any],
    library: ExperienceLibrary,
    *,
    scenario: str,
    condition: str,
    feedback_history: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(planner_input, ensure_ascii=False))
    latest = feedback_history[-1] if feedback_history else {}
    primary_reason = str(latest.get("primary_failure_reason") or "")
    updated["recovery_parameter_priors"] = build_recovery_parameter_priors(
        library.entries,
        scenario=scenario,
        condition=condition,
        primary_reason=primary_reason,
    )
    return updated


def _mock_plan(*, candidate: CandidatePlan, planner_input: dict[str, Any], rewrite_round: int, variant: str = "candidate") -> dict[str, Any]:
    steps = list(candidate.steps)
    latest = ((planner_input.get("rewrite_guidance") or {}).get("latest_sandbox_critic_feedback") or {})
    if rewrite_round == 0 and variant == "extra_safe_transport" and "segmented_transport" in steps and "safe_transport_pose" not in steps:
        steps.insert(steps.index("segmented_transport"), "safe_transport_pose")
    if rewrite_round == 0 and variant == "duplicate_verify" and "verify_place_zone" in steps:
        steps.insert(steps.index("verify_place_zone"), "verify_place_zone")
    if rewrite_round == 0 and variant == "place_before_grasp":
        if "place_object" in steps:
            steps.remove("place_object")
            steps.insert(0, "place_object")
    return {
        "goal": f"Execute {candidate.candidate_id} with sandbox critic safeguards",
        "steps": [
            {
                "stage": "sandbox_rewrite" if rewrite_round else "execution",
                "action": step,
                "parameters": {},
                "reason": "mock plan follows candidate steps and sandbox feedback",
            }
            for step in steps
        ],
        "constraints": ["validate in MuJoCo sandbox before robot execution"],
        "risk_notes": ["dry-run LLM mode; replace with provider call for paper evidence"],
        "evidence_ids": [],
        "confidence": 0.5 if rewrite_round == 0 else 0.6,
    }


def _generate_plan(
    *,
    scenario: str,
    condition: str,
    candidate: CandidatePlan,
    candidate_pool: list[CandidatePlan],
    planner_input: dict[str, Any],
    provider: str,
    model: str,
    dry_run_llm: bool,
    rewrite_round: int,
    mock_plan_variant: str = "candidate",
) -> tuple[dict[str, Any], str]:
    prompt = recovery_plan_prompt(
        scenario=scenario,
        condition=condition,
        planner_input=planner_input,
        candidate=candidate,
        candidates=candidate_pool,
    )
    raw_plan = _mock_plan(
        candidate=candidate,
        planner_input=planner_input,
        rewrite_round=rewrite_round,
        variant=mock_plan_variant,
    ) if dry_run_llm else invoke_recovery_plan_llm(prompt, provider=provider, model=model)
    plan = normalize_recovery_plan(
        raw_plan,
        scenario=scenario,
        condition=condition,
        candidate=candidate,
        candidates=candidate_pool,
        planner_input=planner_input,
        provider="dry_run" if dry_run_llm else provider,
        model="mock" if dry_run_llm else model,
    )
    return plan, prompt


def _sandbox_candidate_for_plan(plan: dict[str, Any], *, base_candidate: CandidatePlan, candidate_pool: list[CandidatePlan]) -> tuple[CandidatePlan, str, dict[str, Any]]:
    candidate = recovery_plan_to_candidate(
        plan,
        base_candidate=base_candidate,
        candidates=candidate_pool,
        candidate_cls=type(base_candidate),
    )
    scenario = str(plan.get("scenario") or "").upper()
    matched, mapping = _canonical_candidate_mapping(scenario, list(candidate.steps))
    if matched is not None:
        return matched, matched.candidate_id, mapping
    if candidate.executable:
        mapping = {
            "mapping_status": "direct_executable",
            "original_steps": list(candidate.steps),
            "normalized_steps": list(candidate.steps),
            "mapped_candidate_id": candidate.candidate_id,
            "reason": "planner candidate is directly executable",
        }
        return candidate, candidate.candidate_id, mapping
    return candidate, "", mapping


def _final_status(sandbox_report: dict[str, Any]) -> str:
    decision = str(sandbox_report.get("decision") or "")
    critic_status = str(sandbox_report.get("critic_status") or "")
    if decision == "accept" and critic_status == "pass":
        return "accept"
    if decision == "reject" or critic_status == "block":
        return "reject"
    return "review"


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    object_class = object_class_for_scenario(args.scenario)
    seed_candidate = _candidate_by_id(args.scenario, args.candidate_id)
    candidate_pool = candidates_for_scenario(args.scenario, include_risky=True)
    stage_report = run_stage_retrieval(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
        candidate_id=seed_candidate.candidate_id,
        candidate_steps=list(seed_candidate.steps),
        top_k=args.stage_top_k,
    )
    planner_context = build_stage_planner_context(
        stage_report,
        scenario=args.scenario,
        condition=args.condition,
        candidate_id=seed_candidate.candidate_id,
        candidate_steps=list(seed_candidate.steps),
        candidate_description=seed_candidate.description,
    )
    planner_input = planner_context.get("planner_input") or {}
    sandbox_initial_state = json.loads(args.sandbox_initial_state.read_text(encoding="utf-8")) if args.sandbox_initial_state else {}
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
    ) if args.use_sandbox_calibration else None

    attempts: list[dict[str, Any]] = []
    feedback_history: list[dict[str, Any]] = []
    current_candidate = seed_candidate
    final_plan: dict[str, Any] = {}
    final_sandbox: dict[str, Any] = {}
    max_attempts = max(0, int(args.max_rewrite_rounds)) + 1

    for attempt_index in range(max_attempts):
        round_planner_input = _augment_planner_input(planner_input, feedback_history) if feedback_history else planner_input
        round_planner_input = _attach_parameter_priors(
            round_planner_input,
            library,
            scenario=args.scenario,
            condition=args.condition,
            feedback_history=feedback_history,
        )
        plan, prompt = _generate_plan(
            scenario=args.scenario,
            condition=args.condition,
            candidate=current_candidate,
            candidate_pool=candidate_pool,
            planner_input=round_planner_input,
            provider=args.provider,
            model=args.model,
            dry_run_llm=args.dry_run_llm,
            rewrite_round=attempt_index,
            mock_plan_variant=args.mock_plan_variant,
        )
        planned_candidate, run_candidate_id, execution_mapping = _sandbox_candidate_for_plan(
            plan,
            base_candidate=current_candidate,
            candidate_pool=candidate_pool,
        )
        plan_validation = validate_recovery_plan_semantics(plan)
        plan["semantic_validation"] = plan_validation
        if plan_validation.get("status") == "fail":
            sandbox_report = {
                "candidate_id": str(plan.get("plan_id") or planned_candidate.candidate_id),
                "decision": "reject",
                "critic_status": "block",
                "sandbox_score": 0.0,
                "critic_risk_score": 1.0,
                "critic_flags": ["plan_semantic_validation_failed"],
                "failed_skills": [],
                "failure_reason": "LLM plan failed pre-sandbox semantic validation",
                "plan_semantic_validation": plan_validation,
            }
            run_candidate_id = str(sandbox_report["candidate_id"])
            execution_mapping = {
                **execution_mapping,
                "sandbox_skipped": True,
                "reason": "pre-sandbox semantic validation failed",
            }
        elif args.use_general_plan_executor:
            trace_dir = args.trace_dir / f"round_{attempt_index:02d}_general_plan" if args.trace_dir else None
            keyframe_dir = args.keyframe_dir / f"round_{attempt_index:02d}_general_plan" if args.keyframe_dir else None
            sandbox_report = evaluate_plan_in_sandbox(
                scenario=args.scenario,
                condition=args.condition,
                plan_steps=list(plan.get("steps") or []),
                candidate_id=str(plan.get("plan_id") or "llm_general_plan"),
                control_mode=args.control_mode,
                keyframe_dir=keyframe_dir,
                trace_dir=trace_dir,
                model_path=args.model_path,
                sandbox_calibration=sandbox_calibration,
                sandbox_initial_state=sandbox_initial_state,
            )
            run_candidate_id = str(sandbox_report.get("candidate_id") or "llm_general_plan")
            execution_mapping = {
                **execution_mapping,
                "general_plan_executor_used": True,
                "mapped_candidate_id": run_candidate_id,
                "reason": "executed LLM step graph directly in sandbox",
            }
        elif run_candidate_id:
            trace_dir = args.trace_dir / f"round_{attempt_index:02d}_{run_candidate_id}" if args.trace_dir else None
            keyframe_dir = args.keyframe_dir / f"round_{attempt_index:02d}_{run_candidate_id}" if args.keyframe_dir else None
            sandbox_report = evaluate_candidate_in_sandbox(
                scenario=args.scenario,
                condition=args.condition,
                candidate_id=run_candidate_id,
                control_mode=args.control_mode,
                keyframe_dir=keyframe_dir,
                trace_dir=trace_dir,
                model_path=args.model_path,
                sandbox_calibration=sandbox_calibration,
                sandbox_initial_state=sandbox_initial_state,
            )
        else:
            sandbox_report = {
                "candidate_id": planned_candidate.candidate_id,
                "decision": "reject",
                "critic_status": "block",
                "sandbox_score": 0.0,
                "critic_risk_score": 1.0,
                "critic_flags": ["plan_has_no_executable_candidate_mapping"],
                "failed_skills": [],
                "failure_reason": "LLM plan did not match an executable MuJoCo candidate",
            }
        feedback = _critic_feedback(sandbox_report)
        status = _final_status(sandbox_report)
        attempts.append({
            "round": attempt_index,
            "is_rewrite": attempt_index > 0,
            "prompt": prompt,
            "planner_input": round_planner_input,
            "recovery_plan": plan,
            "plan_semantic_validation": plan_validation,
            "planned_candidate": {
                "candidate_id": planned_candidate.candidate_id,
                "steps": list(planned_candidate.steps),
                "executable": bool(planned_candidate.executable),
            },
            "execution_mapping": execution_mapping,
            "run_candidate_id": run_candidate_id,
            "sandbox_result": sandbox_report,
            "critic_feedback": feedback,
            "round_status": status,
        })
        final_plan = plan
        final_sandbox = sandbox_report
        if status == "accept":
            break
        feedback_history.append(feedback)
        current_candidate = planned_candidate if planned_candidate.executable else current_candidate

    final_status = _final_status(final_sandbox)
    final_candidate_id = str(final_sandbox.get("candidate_id") or current_candidate.candidate_id)
    final_steps = list(final_plan.get("candidate_steps") or current_candidate.steps)
    validated_plan = build_validated_robot_plan(
        scenario=args.scenario,
        condition=args.condition,
        selected_candidate_id=final_candidate_id,
        selected_steps=final_steps,
        sandbox_report=final_sandbox,
        fused_score={
            "decision": "accept" if final_status == "accept" else "review" if final_status == "review" else "reject",
            "combined_score": float(final_sandbox.get("sandbox_score") or 0.0),
        },
        recovery_plan=final_plan,
    )
    execution_report = execute_validated_robot_plan(
        validated_plan,
        DryRunSkillExecutor(default_r1pro_skill_registry()),
        mode="dry_run",
    ).to_dict()
    writeback_report: dict[str, Any] = {
        "enabled": False,
        "reason": "not_requested",
    }
    if args.writeback_sandbox_experiences:
        writeback_items = [
            {
                "plan_index": int(item.get("round", -1)),
                "recovery_plan": item.get("recovery_plan") if isinstance(item.get("recovery_plan"), dict) else {},
                "sandbox_result": item.get("sandbox_result") if isinstance(item.get("sandbox_result"), dict) else {},
                "plan_semantic_validation": item.get("plan_semantic_validation") if isinstance(item.get("plan_semantic_validation"), dict) else {},
                "sandbox_skipped": bool((item.get("execution_mapping") or {}).get("sandbox_skipped", False)),
            }
            for item in attempts
        ]
        writeback_report = writeback_sandbox_reports(
            library,
            writeback_items,
            selected_plan_index=len(attempts) - 1,
            source_tool="run_recovery_plan_sandbox_loop",
            merge_duplicates=bool(args.writeback_merge_duplicates),
        )
        writeback_report["enabled"] = True
        output_library = args.writeback_library_output or args.universal_experience_lib
        library.save(output_library)
        writeback_report["library_output"] = str(output_library)
    report = {
        "schema_version": "recovery_plan_sandbox_loop_report_v1",
        "scenario": args.scenario,
        "condition": args.condition,
        "seed_candidate_id": seed_candidate.candidate_id,
        "dry_run_llm": bool(args.dry_run_llm),
        "mock_plan_variant": args.mock_plan_variant if args.dry_run_llm else "",
        "general_plan_executor_enabled": bool(args.use_general_plan_executor),
        "llm_provider": "dry_run" if args.dry_run_llm else args.provider,
        "llm_model": "mock" if args.dry_run_llm else args.model,
        "max_rewrite_rounds": int(args.max_rewrite_rounds),
        "runtime_scene": {
            "enabled": args.model_path is not None,
            "model_path": str(args.model_path) if args.model_path else "",
        },
        "planner_input_used": attempts[-1]["planner_input"] if attempts else planner_input,
        "recovery_parameter_priors": (attempts[-1]["planner_input"].get("recovery_parameter_priors") if attempts and isinstance(attempts[-1].get("planner_input"), dict) else {}),
        "stage_planner_context": planner_context,
        "sandbox_calibration_enabled": bool(args.use_sandbox_calibration),
        "sandbox_calibration": sandbox_calibration or {},
        "sandbox_initial_state": {
            "enabled": bool(sandbox_initial_state),
            "path": str(args.sandbox_initial_state) if args.sandbox_initial_state else "",
            "source_episode_id": str(sandbox_initial_state.get("source_episode_id") or ""),
            "confidence": float(sandbox_initial_state.get("confidence") or 0.0) if sandbox_initial_state else 0.0,
        },
        "attempt_count": len(attempts),
        "rewrite_rounds": max(0, len(attempts) - 1),
        "critic_feedback_history": feedback_history,
        "attempts": attempts,
        "sandbox_experience_writeback": writeback_report,
        "final_sandbox_status": final_status,
        "validated_robot_plan": validated_plan,
        "validated_robot_plan_output": str(args.save_plan),
        "dry_run_execution_report": execution_report,
    }
    _write_json(args.save_plan, validated_plan)
    _write_json(args.save_report, report)
    print(json.dumps({
        "final_sandbox_status": final_status,
        "attempt_count": len(attempts),
        "rewrite_rounds": max(0, len(attempts) - 1),
        "sandbox_writeback_enabled": bool(writeback_report.get("enabled")),
        "sandbox_writeback_count": int(writeback_report.get("write_count") or 0),
        "validated_robot_plan": str(args.save_plan),
        "save_report": str(args.save_report),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
