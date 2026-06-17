from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE_ROOT = ROOT.parent / "experience_system"
for path in (ROOT, EXPERIENCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experience_core import (
    DryRunSkillExecutor,
    ExperienceLibrary,
    build_field_atomic_planner_input,
    build_validated_robot_plan,
    default_r1pro_skill_registry,
    execute_validated_robot_plan,
    field_atomic_plan_prompt,
    invoke_field_atomic_plan_llm,
    normalize_field_atomic_plan,
    validate_skill_semantic_plan,
)
from skills.field_atomic import FieldAtomicSkillExecutor, build_atomic_experience_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and execute a field-atomic LLM plan, then write success/failure experiences.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--model-path", default="r1pro_g3_sorting_scene.xml")
    parser.add_argument("--universal-experience-lib", type=Path, default=None)
    parser.add_argument("--writeback-library-output", type=Path, required=True)
    parser.add_argument("--scenario-id", default="field_atomic")
    parser.add_argument("--condition-id", default="default")
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--dry-run-llm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--save-plan", type=Path, required=True)
    parser.add_argument("--save-report", type=Path, required=True)
    parser.add_argument("--save-validated-robot-plan", type=Path, default=None)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _mock_plan(goal: str) -> dict[str, Any]:
    return {
        "goal": goal,
        "steps": [
            {
                "action": "head_camera_capture",
                "parameters": {"width": 80, "height": 60, "include_depth": True},
                "reason": "observe the scene before movement",
            },
            {
                "action": "base_lidar_scan",
                "parameters": {"ray_count": 45, "horizontal_fov_deg": 180.0, "max_range": 3.0},
                "reason": "check nearby obstacles around the base",
            },
            {
                "action": "base_move_to_pose",
                "parameters": {"base_x": 0.02, "base_y": 0.0, "base_yaw": 0.0, "steps": 120, "settle_steps": 20, "max_joint_step": 0.004, "direct_qpos": False},
                "reason": "small conservative base movement",
            },
        ],
        "constraints": ["small movements only in dry-run mock plan"],
        "risk_notes": ["field atomic smoke does not prove real-robot safety"],
        "evidence_ids": [],
        "confidence": 0.5,
    }


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib) if args.universal_experience_lib else ExperienceLibrary()
    planner_input = build_field_atomic_planner_input(
        library.entries,
        scenario_id=args.scenario_id,
        condition_id=args.condition_id,
        goal=args.goal,
    )
    prompt = field_atomic_plan_prompt(goal=args.goal, planner_input=planner_input, max_steps=args.max_steps)
    raw_plan = _mock_plan(args.goal) if args.dry_run_llm else invoke_field_atomic_plan_llm(prompt, provider=args.provider, model=args.model)
    plan = normalize_field_atomic_plan(raw_plan, goal=args.goal, planner_input=planner_input, max_steps=args.max_steps)
    semantic_validation = validate_skill_semantic_plan(plan)

    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    executor = FieldAtomicSkillExecutor()

    step_reports: list[dict[str, Any]] = []
    writeback_decisions: list[dict[str, Any]] = []
    for index, step in enumerate(plan["steps"]):
        action = str(step["action"])
        parameters = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        try:
            result = executor.execute(model, data, action, parameters)
        except Exception as exc:
            from skills.field_atomic.atomic_schema import FieldAtomicResult

            result = FieldAtomicResult(action=action, success=False, status="exception", message=str(exc), parameters=dict(parameters))
        entry = build_atomic_experience_entry(
            scenario_id=args.scenario_id,
            condition_id=args.condition_id,
            robot_type="mobile_dual_arm",
            action=action,
            result=result,
        )
        entry.metadata["field_atomic_plan_id"] = plan["plan_id"]
        entry.metadata["field_atomic_plan_step_index"] = index
        decision = library.add_with_policy(entry, merge_duplicates=False)
        writeback_decisions.append(decision)
        step_reports.append({
            "index": index,
            "action": action,
            "parameters": dict(parameters),
            "success": bool(result.success),
            "status": result.status,
            "message": result.message,
            "writeback_decision": decision,
        })

    all_steps_success = bool(step_reports) and all(bool(item["success"]) for item in step_reports)
    critic_status = "pass" if all_steps_success else "block"
    decision = "accept" if all_steps_success and semantic_validation.get("fatal_count", 0) == 0 else "reject"
    sandbox_report = {
        "candidate_id": plan["plan_id"],
        "decision": decision,
        "critic_status": critic_status,
        "task_success": all_steps_success,
        "success": all_steps_success,
        "sandbox_score": 1.0 if all_steps_success else 0.0,
        "critic_risk_score": 0.0 if all_steps_success else 1.0,
        "critic_flags": [] if all_steps_success else ["field_atomic_step_failed"],
        "failed_skills": [item["action"] for item in step_reports if not item["success"]],
        "field_atomic": True,
    }
    recovery_plan_view = {
        "plan_id": plan["plan_id"],
        "goal": plan.get("goal", ""),
        "steps": [
            {
                "action": str(step.get("action") or ""),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
                "stage": "field_atomic_execution",
                "reason": str(step.get("reason") or ""),
            }
            for step in plan.get("steps") or []
        ],
        "constraints": list(plan.get("constraints") or []),
        "risk_notes": list(plan.get("risk_notes") or []),
        "confidence": float(plan.get("confidence") or 0.0),
        "semantic_validation": semantic_validation,
    }
    validated_robot_plan = build_validated_robot_plan(
        scenario=args.scenario_id,
        condition=args.condition_id,
        selected_candidate_id=plan["plan_id"],
        selected_steps=[str(step.get("action") or "") for step in plan.get("steps") or []],
        sandbox_report=sandbox_report,
        fused_score={
            "decision": decision,
            "combined_score": 1.0 if all_steps_success else 0.0,
        },
        recovery_plan=recovery_plan_view,
    )
    dry_run_execution_report = execute_validated_robot_plan(
        validated_robot_plan,
        DryRunSkillExecutor(default_r1pro_skill_registry()),
        mode="dry_run",
    ).to_dict()

    library.save(args.writeback_library_output)
    report = {
        "schema_version": "field_atomic_llm_plan_report_v1",
        "goal": args.goal,
        "dry_run_llm": bool(args.dry_run_llm),
        "planner_input": planner_input,
        "field_atomic_plan": plan,
        "semantic_validation": semantic_validation,
        "field_atomic_sandbox_report": sandbox_report,
        "validated_robot_plan": validated_robot_plan,
        "dry_run_execution_report": dry_run_execution_report,
        "model_path": args.model_path,
        "step_count": len(step_reports),
        "success_count": sum(1 for item in step_reports if item["success"]),
        "failure_count": sum(1 for item in step_reports if not item["success"]),
        "step_reports": step_reports,
        "writeback": {
            "library_output": str(args.writeback_library_output),
            "attempted_write_count": len(writeback_decisions),
            "write_count": sum(1 for item in writeback_decisions if bool(item.get("write"))),
            "decisions": writeback_decisions,
        },
    }
    _write_json(args.save_plan, plan)
    if args.save_validated_robot_plan is not None:
        _write_json(args.save_validated_robot_plan, validated_robot_plan)
    _write_json(args.save_report, report)
    print(json.dumps({
        "plan_id": plan["plan_id"],
        "step_count": report["step_count"],
        "success_count": report["success_count"],
        "failure_count": report["failure_count"],
        "write_count": report["writeback"]["write_count"],
        "save_plan": str(args.save_plan),
        "save_validated_robot_plan": str(args.save_validated_robot_plan) if args.save_validated_robot_plan else "",
        "save_report": str(args.save_report),
        "writeback_library_output": str(args.writeback_library_output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
