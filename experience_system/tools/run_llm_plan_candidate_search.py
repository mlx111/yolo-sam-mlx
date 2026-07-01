"""Generate multiple recovery-plan candidates and select by parallel sandbox rollout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import (
    DryRunSkillExecutor,
    ExperienceLibrary,
    JSON_ONLY_LINE,
    build_stage_planner_context,
    build_recovery_parameter_priors,
    build_validated_robot_plan,
    default_r1pro_skill_registry,
    execute_validated_robot_plan,
    invoke_llm,
    normalize_recovery_plan,
    parse_json_payload,
    recovery_plan_prompt,
    run_stage_retrieval,
    validate_recovery_plan_semantics,
    writeback_sandbox_reports,
)
from source.legacy_r1pro.candidate_sandbox import evaluate_plan_in_sandbox, select_sandbox_calibration
from source.legacy_r1pro.run_r1pro_memory_policy_smoke import CandidatePlan, candidates_for_scenario, object_class_for_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM multi-candidate recovery-plan search with parallel sandbox validation.")
    parser.add_argument("--scenario", choices=["G3"], default="")
    parser.add_argument("--condition", choices=["clean", "place_occupied"], default="")
    parser.add_argument("--candidate-id", default="", help="seed candidate id; defaults to first executable candidate")
    parser.add_argument("--num-plans", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, default=None)
    parser.add_argument("--sandbox-initial-state", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None, help="optional runtime MuJoCo XML generated from runtime_sandbox_scene_v1")
    parser.add_argument("--use-sandbox-calibration", action="store_true")
    parser.add_argument("--stage-top-k", type=int, default=None)
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--dry-run-llm", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--keyframe-dir", type=Path, default=None)
    parser.add_argument("--save-plan", type=Path, default=None)
    parser.add_argument("--save-report", type=Path, default=None)
    parser.add_argument("--writeback-sandbox-experiences", action="store_true", help="write sandboxed plan results into an experience-library copy")
    parser.add_argument("--writeback-library-output", type=Path, default=None, help="path for the updated experience library; defaults to --universal-experience-lib when writeback is enabled")
    parser.add_argument("--writeback-merge-duplicates", action="store_true", help="allow write policy to merge duplicate low-risk successes")
    parser.add_argument("--worker-rollout", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-job", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker_rollout:
        missing = []
        for field, flag in (
            ("scenario", "--scenario"),
            ("condition", "--condition"),
            ("universal_experience_lib", "--universal-experience-lib"),
            ("save_plan", "--save-plan"),
            ("save_report", "--save-report"),
        ):
            if not getattr(args, field):
                missing.append(flag)
        if missing:
            parser.error("required arguments missing outside --worker-rollout: " + ", ".join(missing))
    return args


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


def _raw_plan_from_candidate(candidate: CandidatePlan, *, variant_index: int) -> dict[str, Any]:
    return {
        "goal": f"Candidate search plan {variant_index}: {candidate.description or candidate.candidate_id}",
        "steps": [
            {
                "stage": "candidate_generation",
                "action": str(step),
                "parameters": {},
                "reason": f"dry-run candidate search uses {candidate.candidate_id}",
            }
            for step in candidate.steps
        ],
        "constraints": ["validate in MuJoCo sandbox before robot execution"],
        "risk_notes": ["dry-run candidate; sandbox critic decides final ranking"],
        "evidence_ids": [],
        "confidence": max(0.35, 0.65 - 0.03 * variant_index),
    }


def _dry_run_raw_plans(scenario: str, num_plans: int) -> list[dict[str, Any]]:
    candidates = [item for item in candidates_for_scenario(scenario, include_risky=True) if item.executable]
    return [_raw_plan_from_candidate(candidate, variant_index=index) for index, candidate in enumerate(candidates[: max(1, num_plans)])]


def _multi_plan_prompt(
    *,
    scenario: str,
    condition: str,
    planner_input: dict[str, Any],
    seed_candidate: CandidatePlan,
    candidate_pool: list[CandidatePlan],
    num_plans: int,
) -> str:
    base_prompt = recovery_plan_prompt(
        scenario=scenario,
        condition=condition,
        planner_input=planner_input,
        candidate=seed_candidate,
        candidates=candidate_pool,
    )
    return f"""
{base_prompt}

Instead of returning one plan, return a JSON object with:
{{
  "plans": [
    <recovery plan object matching the schema above>
  ]
}}

Generate {num_plans} diverse candidate plans. Prefer physically meaningful
differences such as cautious verification, alternate placement, safer transport,
or re-leveling before place. Do not invent skills.

Use recovery_parameter_priors from planner_input when proposing parameters for
recovery skills. Prefer bounded, small-step adjustments over aggressive values.

{JSON_ONLY_LINE}
"""


def _invoke_multi_plan_llm(
    *,
    scenario: str,
    condition: str,
    planner_input: dict[str, Any],
    seed_candidate: CandidatePlan,
    candidate_pool: list[CandidatePlan],
    num_plans: int,
    provider: str,
    model: str,
) -> list[dict[str, Any]]:
    raw = invoke_llm(
        _multi_plan_prompt(
            scenario=scenario,
            condition=condition,
            planner_input=planner_input,
            seed_candidate=seed_candidate,
            candidate_pool=candidate_pool,
            num_plans=num_plans,
        ),
        provider=provider,
        model=model,
        system_prompt="You generate diverse robot recovery plans and return JSON only.",
        temperature=0.35,
    )
    payload = parse_json_payload(raw, prefer_array=False)
    if isinstance(payload, dict) and isinstance(payload.get("plans"), list):
        return [item for item in payload["plans"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise RuntimeError("LLM multi-plan response must be an object with a plans array")


def _normalize_plans(
    raw_plans: list[dict[str, Any]],
    *,
    scenario: str,
    condition: str,
    seed_candidate: CandidatePlan,
    candidate_pool: list[CandidatePlan],
    planner_input: dict[str, Any],
    provider: str,
    model: str,
    limit: int,
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in raw_plans:
        try:
            plan = normalize_recovery_plan(
                raw,
                scenario=scenario,
                condition=condition,
                candidate=seed_candidate,
                candidates=candidate_pool,
                planner_input=planner_input,
                provider=provider,
                model=model,
            )
        except Exception as exc:
            plans.append({
                "schema_version": "llm_recovery_plan_v1",
                "plan_id": f"invalid_plan_{len(plans):03d}",
                "scenario": scenario,
                "condition": condition,
                "steps": [],
                "candidate_steps": [],
                "normalization_error": str(exc),
            })
            continue
        key = tuple(plan.get("candidate_steps") or [])
        if key in seen:
            continue
        seen.add(key)
        plans.append(plan)
        if len([item for item in plans if item.get("candidate_steps")]) >= limit:
            break
    return plans


def _evaluate_plan_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    plan = job["plan"]
    sandbox = evaluate_plan_in_sandbox(
        scenario=str(job["scenario"]),
        condition=str(job["condition"]),
        plan_steps=list(plan.get("steps") or []),
        candidate_id=str(plan.get("plan_id") or "llm_candidate_plan"),
        control_mode=str(job["control_mode"]),
        keyframe_dir=Path(job["keyframe_dir"]) if job.get("keyframe_dir") else None,
        trace_dir=Path(job["trace_dir"]) if job.get("trace_dir") else None,
        model_path=job.get("model_path") or None,
        sandbox_calibration=job.get("sandbox_calibration") if isinstance(job.get("sandbox_calibration"), dict) else None,
        sandbox_initial_state=job.get("sandbox_initial_state") if isinstance(job.get("sandbox_initial_state"), dict) else None,
    )
    return {
        "plan_index": int(job["plan_index"]),
        "plan_id": str(plan.get("plan_id") or ""),
        "worker": {
            "success": True,
            "elapsed_s": round(time.time() - started, 4),
            "pid": os.getpid(),
        },
        "sandbox_result": sandbox,
    }


def _worker_main(args: argparse.Namespace) -> None:
    if args.worker_job is None or args.worker_output is None:
        raise RuntimeError("--worker-job and --worker-output are required for --worker-rollout")
    job = json.loads(args.worker_job.read_text(encoding="utf-8"))
    try:
        payload = _evaluate_plan_job(job)
    except Exception as exc:
        payload = {
            "plan_index": int(job.get("plan_index") or -1),
            "plan_id": str((job.get("plan") or {}).get("plan_id") or ""),
            "worker": {
                "success": False,
                "error": str(exc),
                "pid": os.getpid(),
            },
            "sandbox_result": {
                "decision": "reject",
                "critic_status": "block",
                "sandbox_score": 0.0,
                "critic_risk_score": 1.0,
                "critic_flags": ["llm_plan_search_worker_failed"],
                "failure_reason": f"worker failed: {exc}",
            },
        }
    _write_json(args.worker_output, payload)


def _run_job_subprocess(job: dict[str, Any], *, work_dir: Path) -> dict[str, Any]:
    job_path = work_dir / f"job_{int(job['plan_index']):03d}.json"
    output_path = work_dir / f"out_{int(job['plan_index']):03d}.json"
    _write_json(job_path, job)
    started = time.time()
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-rollout",
            "--worker-job",
            str(job_path),
            "--worker-output",
            str(output_path),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=None,
    )
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "plan_index": int(job.get("plan_index") or -1),
            "plan_id": str((job.get("plan") or {}).get("plan_id") or ""),
            "worker": {"success": False},
            "sandbox_result": {
                "decision": "reject",
                "critic_status": "block",
                "sandbox_score": 0.0,
                "critic_risk_score": 1.0,
                "critic_flags": ["llm_plan_search_worker_no_output"],
                "failure_reason": "worker produced no output",
            },
        }
    payload.setdefault("worker", {})
    payload["worker"].update({
        "subprocess_returncode": completed.returncode,
        "subprocess_elapsed_s": round(time.time() - started, 4),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-2000:],
    })
    if completed.returncode != 0:
        payload["worker"]["success"] = False
        payload["sandbox_result"]["failure_reason"] = payload["sandbox_result"].get("failure_reason") or f"worker exited {completed.returncode}"
    return payload


def _run_jobs(jobs: list[dict[str, Any]], *, parallel_workers: int) -> list[dict[str, Any]]:
    if parallel_workers <= 1:
        return [_evaluate_plan_job(job) for job in jobs]
    with tempfile.TemporaryDirectory(prefix="llm_plan_search_workers_") as tmp:
        work_dir = Path(tmp)
        reports: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(parallel_workers))) as pool:
            futures = [pool.submit(_run_job_subprocess, job, work_dir=work_dir) for job in jobs]
            for future in as_completed(futures):
                reports.append(future.result())
        return reports


def _status(sandbox: dict[str, Any]) -> str:
    if sandbox.get("decision") == "accept" and sandbox.get("critic_status") == "pass":
        return "accept"
    if sandbox.get("decision") == "reject" or sandbox.get("critic_status") == "block":
        return "reject"
    return "review"


def _rank_score(item: dict[str, Any]) -> tuple[float, float, float, float]:
    sandbox = item.get("sandbox_result") or {}
    status_bonus = {"accept": 1.0, "review": 0.35, "reject": -1.0}.get(str(item.get("search_status") or ""), 0.0)
    return (
        status_bonus,
        float(sandbox.get("sandbox_score") or 0.0),
        -float(sandbox.get("critic_risk_score") or 0.0),
        float((item.get("recovery_plan") or {}).get("confidence") or 0.0),
    )


def main() -> None:
    args = parse_args()
    if args.worker_rollout:
        _worker_main(args)
        return

    started = time.time()
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
    planner_input = dict(planner_input)
    planner_input["recovery_parameter_priors"] = build_recovery_parameter_priors(
        library.entries,
        scenario=args.scenario,
        condition=args.condition,
    )
    sandbox_initial_state = json.loads(args.sandbox_initial_state.read_text(encoding="utf-8")) if args.sandbox_initial_state else {}
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=args.scenario,
        condition=args.condition,
        object_class=object_class,
    ) if args.use_sandbox_calibration else None

    raw_plans = _dry_run_raw_plans(args.scenario, args.num_plans) if args.dry_run_llm else _invoke_multi_plan_llm(
        scenario=args.scenario,
        condition=args.condition,
        planner_input=planner_input,
        seed_candidate=seed_candidate,
        candidate_pool=candidate_pool,
        num_plans=args.num_plans,
        provider=args.provider,
        model=args.model,
    )
    plans = _normalize_plans(
        raw_plans,
        scenario=args.scenario,
        condition=args.condition,
        seed_candidate=seed_candidate,
        candidate_pool=candidate_pool,
        planner_input=planner_input,
        provider="dry_run" if args.dry_run_llm else args.provider,
        model="mock" if args.dry_run_llm else args.model,
        limit=args.num_plans,
    )

    candidate_reports: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        validation = validate_recovery_plan_semantics(plan) if plan.get("candidate_steps") else {
            "status": "fail",
            "issues": [{"severity": "fatal", "code": "normalization_failed", "message": str(plan.get("normalization_error") or "normalization failed")}],
        }
        plan["semantic_validation"] = validation
        base_report = {
            "plan_index": index,
            "plan_id": str(plan.get("plan_id") or ""),
            "recovery_plan": plan,
            "plan_semantic_validation": validation,
        }
        if validation.get("status") == "fail":
            candidate_reports.append({
                **base_report,
                "search_status": "reject",
                "sandbox_skipped": True,
                "sandbox_result": {
                    "decision": "reject",
                    "critic_status": "block",
                    "sandbox_score": 0.0,
                    "critic_risk_score": 1.0,
                    "critic_flags": ["plan_semantic_validation_failed"],
                    "failure_reason": "plan failed pre-sandbox semantic validation",
                },
                "worker": {"success": True, "skipped": True},
            })
            continue
        trace_dir = args.trace_dir / f"plan_{index:03d}" if args.trace_dir else None
        keyframe_dir = args.keyframe_dir / f"plan_{index:03d}" if args.keyframe_dir else None
        jobs.append({
            "plan_index": index,
            "scenario": args.scenario,
            "condition": args.condition,
            "control_mode": args.control_mode,
            "plan": plan,
            "sandbox_initial_state": sandbox_initial_state,
            "sandbox_calibration": sandbox_calibration or {},
            "trace_dir": str(trace_dir) if trace_dir else "",
            "keyframe_dir": str(keyframe_dir) if keyframe_dir else "",
            "model_path": str(args.model_path) if args.model_path else "",
        })
        candidate_reports.append(base_report)

    worker_reports = _run_jobs(jobs, parallel_workers=max(1, int(args.parallel_workers))) if jobs else []
    by_index = {int(item.get("plan_index")): item for item in worker_reports}
    merged_reports: list[dict[str, Any]] = []
    for report in candidate_reports:
        worker = by_index.get(int(report["plan_index"]))
        if worker is not None:
            sandbox = worker.get("sandbox_result") or {}
            merged_reports.append({
                **report,
                "sandbox_skipped": False,
                "sandbox_result": sandbox,
                "worker": worker.get("worker") or {},
                "search_status": _status(sandbox),
            })
        else:
            merged_reports.append(report)

    ranked = sorted(merged_reports, key=_rank_score, reverse=True)
    best = ranked[0] if ranked else {}
    best_plan = best.get("recovery_plan") if isinstance(best.get("recovery_plan"), dict) else {}
    best_sandbox = best.get("sandbox_result") if isinstance(best.get("sandbox_result"), dict) else {}
    final_status = str(best.get("search_status") or "reject")
    validated_plan = build_validated_robot_plan(
        scenario=args.scenario,
        condition=args.condition,
        selected_candidate_id=str(best_plan.get("plan_id") or ""),
        selected_steps=list(best_plan.get("candidate_steps") or []),
        sandbox_report=best_sandbox,
        fused_score={
            "decision": "accept" if final_status == "accept" else "review" if final_status == "review" else "reject",
            "combined_score": float(best_sandbox.get("sandbox_score") or 0.0),
        },
        recovery_plan=best_plan,
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
        writeback_report = writeback_sandbox_reports(
            library,
            ranked,
            selected_plan_index=int(best.get("plan_index", -1)) if best else -1,
            source_tool="run_llm_plan_candidate_search",
            merge_duplicates=bool(args.writeback_merge_duplicates),
        )
        writeback_report["enabled"] = True
        output_library = args.writeback_library_output or args.universal_experience_lib
        library.save(output_library)
        writeback_report["library_output"] = str(output_library)
    elapsed = time.time() - started
    failed_worker_count = sum(1 for item in worker_reports if (item.get("worker") or {}).get("success") is False)
    report = {
        "schema_version": "llm_plan_candidate_search_report_v1",
        "scenario": args.scenario,
        "condition": args.condition,
        "seed_candidate_id": seed_candidate.candidate_id,
        "dry_run_llm": bool(args.dry_run_llm),
        "llm_provider": "dry_run" if args.dry_run_llm else args.provider,
        "llm_model": "mock" if args.dry_run_llm else args.model,
        "num_plans_requested": int(args.num_plans),
        "num_plans_normalized": len(plans),
        "stage_planner_context": planner_context,
        "planner_input_used": planner_input,
        "recovery_parameter_priors": planner_input.get("recovery_parameter_priors") if isinstance(planner_input, dict) else {},
        "runtime_scene": {
            "enabled": args.model_path is not None,
            "model_path": str(args.model_path) if args.model_path else "",
        },
        "sandbox_calibration_enabled": bool(args.use_sandbox_calibration),
        "sandbox_calibration": sandbox_calibration or {},
        "sandbox_initial_state": {
            "enabled": bool(sandbox_initial_state),
            "path": str(args.sandbox_initial_state) if args.sandbox_initial_state else "",
            "source_episode_id": str(sandbox_initial_state.get("source_episode_id") or ""),
            "confidence": float(sandbox_initial_state.get("confidence") or 0.0) if sandbox_initial_state else 0.0,
        },
        "parallel_worker_count": max(1, int(args.parallel_workers)),
        "failed_worker_count": failed_worker_count,
        "elapsed_s": round(elapsed, 4),
        "sandboxed_plan_count": len(worker_reports),
        "rollouts_per_minute": round((len(worker_reports) / elapsed) * 60.0, 4) if elapsed > 0 else 0.0,
        "selected_plan_id": str(best_plan.get("plan_id") or ""),
        "selected_plan_index": int(best.get("plan_index", -1)) if best else -1,
        "final_sandbox_status": final_status,
        "candidate_reports": ranked,
        "sandbox_experience_writeback": writeback_report,
        "validated_robot_plan": validated_plan,
        "validated_robot_plan_output": str(args.save_plan),
        "dry_run_execution_report": execution_report,
    }
    _write_json(args.save_plan, validated_plan)
    _write_json(args.save_report, report)
    print(json.dumps({
        "selected_plan_id": report["selected_plan_id"],
        "selected_plan_index": report["selected_plan_index"],
        "final_sandbox_status": report["final_sandbox_status"],
        "sandboxed_plan_count": report["sandboxed_plan_count"],
        "failed_worker_count": report["failed_worker_count"],
        "rollouts_per_minute": report["rollouts_per_minute"],
        "sandbox_writeback_enabled": bool(writeback_report.get("enabled")),
        "sandbox_writeback_count": int(writeback_report.get("write_count") or 0),
        "save_plan": str(args.save_plan),
        "save_report": str(args.save_report),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
