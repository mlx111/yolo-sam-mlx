#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

WRAPPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WRAPPER_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WRAPPER_ROOT))

from run_experiment_v4 import DEFAULT_PREGRASP_HEIGHT, ExperimentV4, _convert_numpy
from experience_system.ur5e_core.critic import build_critic_result, critique_ur5e_failure_experience
from experience_system.memory.v3 import MemoryV3Library, build_retrieval_key, make_memory_v3_entry
from skills.field_atomic import Ur5eFieldAtomicSkillExecutor
from skills.field_atomic.action_io import load_action_steps, result_to_dict
from skills.registry import allowed_actions


DEFAULT_ACTIONS = WRAPPER_ROOT / "inputs" / "default_field_atomic_grasp_lift_actions.json"
DEFAULT_SCENE = WRAPPER_ROOT / "scene" / "scene.xml"
DEFAULT_EXPERIENCE_WRITE = WRAPPER_ROOT / "results" / "memory" / "field_atomic_smoke_experience.json"


def _stable_experience_id(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return "smoke_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_convert_numpy(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _bridge_legacy_llm_env() -> None:
    env_pairs = (
        ("EXPERIENCE_LLM_API_KEY", ("ARK_API_KEY", "DOUBAO_API_KEY", "EXPERIMENT_LLM_API_KEY")),
        ("EXPERIENCE_LLM_BASE_URL", ("ARK_BASE_URL", "DOUBAO_BASE_URL", "EXPERIMENT_LLM_BASE_URL")),
        ("EXPERIENCE_LLM_MODEL", ("DOUBAO_MODEL_NAME", "EXPERIMENT_LLM_MODEL")),
    )
    for target_name, source_names in env_pairs:
        if os.getenv(target_name):
            continue
        for source_name in source_names:
            value = os.getenv(source_name)
            if value and value.strip():
                os.environ[target_name] = value.strip()
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UR5e field-atomic skills from a JSON action file.")
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS, help='JSON file containing {"steps": [{"action": ..., "parameters": {...}}]}')
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE, help="MuJoCo scene XML")
    parser.add_argument("--save-report", type=Path, default=WRAPPER_ROOT / "scene" / "field_atomic_skill_report.json")
    parser.add_argument("--scenario-id", default="ur5e_field_atomic_smoke")
    parser.add_argument("--condition-id", default="default")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    parser.add_argument("--continue-on-failure", dest="stop_on_failure", action="store_false")
    parser.add_argument("--experience-write", type=Path, default=DEFAULT_EXPERIENCE_WRITE, help="MemoryV3 library updated automatically when the skill sequence fails.")
    parser.add_argument("--no-experience-write", action="store_true", help="Disable automatic failed-experience writeback.")
    return parser.parse_args()


def _executed_steps(steps: list[dict[str, Any]], reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        raw_step = steps[index] if index < len(steps) and isinstance(steps[index], dict) else {}
        out.append({
            "action": str(raw_step.get("action") or report.get("action") or ""),
            "parameters": raw_step.get("parameters") if isinstance(raw_step.get("parameters"), dict) else {},
            "success": bool(report.get("success")),
            "status": str(report.get("status") or ""),
            "message": str(report.get("message") or ""),
        })
    return out


def _rule_critic_for_failed_smoke(executed_steps: list[dict[str, Any]], reports: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    rule_flags: list[dict[str, Any]] = []
    actions_before_lift: list[str] = []
    for step in executed_steps:
        action = str(step.get("action") or "")
        if action == "lift":
            if "close_gripper" not in actions_before_lift:
                rule_flags.append({
                    "rule": "missing_close_gripper_before_lift",
                    "stage": "lift",
                    "severity": "block",
                    "evidence": "skill sequence reached lift without any previous close_gripper action",
                    "description_cn": "技能序列在未闭合夹爪的情况下执行提升。",
                })
            break
        actions_before_lift.append(action)
    for report in reports:
        if str(report.get("action") or "") == "lift" and not bool(report.get("success")):
            rule_flags.append({
                "rule": "object_not_lifted",
                "stage": "lift",
                "severity": "block",
                "evidence": str(report.get("message") or report.get("status") or "lift failed"),
                "description_cn": "提升技能执行后物体没有实际升高。",
            })
    for record in metrics.get("skill_results", []) or []:
        if isinstance(record, dict) and record.get("skill") == "lift" and not bool(record.get("success")):
            extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
            if extra:
                rule_flags.append({
                    "rule": "insufficient_z_change",
                    "stage": "lift",
                    "severity": "block",
                    "evidence": f"object_z_change={extra.get('object_z_change')}, threshold={extra.get('success_z_change')}",
                    "description_cn": "物体提升高度低于成功阈值。",
                })
            break
    return {"enabled": True, "rule_flags": rule_flags}


def _write_failed_experience(
    *,
    path: Path,
    steps: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    report_payload: dict[str, Any],
) -> None:
    metrics = report_payload.get("metrics") if isinstance(report_payload.get("metrics"), dict) else {}
    executed_steps = _executed_steps(steps, reports)
    failed_report = next((item for item in reports if not bool(item.get("success"))), reports[-1] if reports else {})
    failure_reason = str(failed_report.get("message") or failed_report.get("status") or "field_atomic_skill_failed")
    rule_result = _rule_critic_for_failed_smoke(executed_steps, reports, metrics)
    metrics_for_critic = dict(metrics)
    metrics_for_critic.update({
        "scenario_id": str(report_payload.get("scenario_id") or ""),
        "condition_id": str(report_payload.get("condition_id") or ""),
        "task_success": False,
        "failure_reason": failure_reason,
        "virtual_execution_result": {
            "step_trace": [
                {
                    "action": step.get("action"),
                    "success": step.get("success"),
                    "status": step.get("status"),
                    "message": step.get("message"),
                }
                for step in executed_steps
            ]
        },
    })
    llm_critic_result: dict[str, Any] = {}
    try:
        _bridge_legacy_llm_env()
        provider = os.getenv("EXPERIMENT_LLM_PROVIDER", "doubao").strip() or "doubao"
        model = os.getenv("EXPERIMENT_LLM_RECOVERY_MODEL") or os.getenv("EXPERIENCE_LLM_MODEL") or os.getenv("DOUBAO_MODEL_NAME") or ""
        llm_critic_result = critique_ur5e_failure_experience(
            method="field_atomic_smoke",
            memory_policy="hierarchical",
            metrics=metrics_for_critic,
            task_history=[
                {
                    "action": step.get("action"),
                    "status": "SUCCESS" if step.get("success") else "FAILURE",
                    "reason": step.get("message") or step.get("status") or "",
                }
                for step in executed_steps
            ],
            recovery_steps=executed_steps,
            retrieved_memories=[],
            provider=provider,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001
        llm_critic_result = {"enabled": True, "error": str(exc)}
    critic_result = build_critic_result(rule_result=rule_result, llm_result=llm_critic_result, is_failure=True)
    failure_taxonomy = {
        "failure_stage": str(failed_report.get("action") or "unknown"),
        "failure_type": failure_reason,
        "failure_reason": failure_reason,
        "failed_action": str(failed_report.get("action") or ""),
        "failed_predicates": ["object_lifted"] if failure_reason == "object_not_lifted" else [],
        "rule_critic": rule_result,
    }
    if llm_critic_result.get("enabled") and not llm_critic_result.get("error"):
        failure_taxonomy["llm_critic"] = llm_critic_result
        if llm_critic_result.get("failure_stage"):
            failure_taxonomy["failure_stage"] = llm_critic_result["failure_stage"]
        if llm_critic_result.get("failure_type"):
            failure_taxonomy["failure_type"] = llm_critic_result["failure_type"]
        if llm_critic_result.get("failed_predicates"):
            failure_taxonomy["failed_predicates"] = llm_critic_result["failed_predicates"]
        if llm_critic_result.get("failure_evidence"):
            failure_taxonomy["failure_evidence"] = llm_critic_result["failure_evidence"]
        if llm_critic_result.get("corrective_direction"):
            failure_taxonomy["corrective_direction"] = llm_critic_result["corrective_direction"]
        if llm_critic_result.get("missing_phases"):
            failure_taxonomy["missing_phases"] = llm_critic_result["missing_phases"]
    elif llm_critic_result:
        failure_taxonomy["llm_critic"] = llm_critic_result
    entry = make_memory_v3_entry(
        condition_id=str(report_payload.get("condition_id") or ""),
        scenario_id=str(report_payload.get("scenario_id") or ""),
        available_actions=allowed_actions(str(report_payload.get("scenario_id") or "")),
        skill_sequence=executed_steps,
        task_success=False,
        failure_reason=failure_reason,
        source="simulation",
        summary=f"Failed UR5e field atomic smoke sequence: {failure_reason}",
        metadata={
            "source_tool": "run_field_atomic_skill_smoke",
            "actions_path": str(report_payload.get("actions_path") or ""),
            "report_path": str(report_payload.get("save_report") or ""),
        },
        validation_evidence={
            "actions": reports,
            "skill_results": metrics.get("skill_results", []),
        },
        recovery_plan={"steps": executed_steps},
        execution_feedback={
            "skill_results": metrics.get("skill_results", []),
            "field_atomic_reports": reports,
            "llm_critic": llm_critic_result,
        },
        anomaly_state={
            "scenario_id": str(report_payload.get("scenario_id") or ""),
            "condition_id": str(report_payload.get("condition_id") or ""),
        },
        failure_taxonomy=failure_taxonomy,
        validation_status="failed",
        validation_source="mujoco_field_atomic_smoke",
        critic_result=critic_result,
        memory_tags={
            "memory_type": "episodic",
            "memory_scope": "condition",
            "memory_role": "failure_case",
        },
    )
    entry.experience_id = _stable_experience_id(
        report_payload.get("scenario_id"),
        report_payload.get("condition_id"),
        json.dumps([{"action": step.get("action")} for step in executed_steps], sort_keys=True, ensure_ascii=False),
        failure_reason,
    )
    entry.retrieval_key = build_retrieval_key(entry)
    library = MemoryV3Library.load(path)
    library.upsert(entry)
    library.save(path)
    print(f"saved failed experience: {path}")
    if llm_critic_result.get("enabled") and not llm_critic_result.get("error"):
        print("saved llm critic lesson:", llm_critic_result.get("parameter_failure_summary", {}).get("overall_lesson") or llm_critic_result.get("root_cause", ""))
    elif llm_critic_result.get("error"):
        print(f"  [WARN] llm critic failed: {llm_critic_result.get('error')}")


def main() -> None:
    args = parse_args()
    steps = load_action_steps(args.actions)
    exp = ExperimentV4(
        enable_viewer=bool(args.viewer),
        condition="direct",
        noise_scale=0.0,
        scene_xml=str(args.scene_xml),
    )
    exp.metrics["scenario_id"] = args.scenario_id
    exp.metrics["condition_id"] = args.condition_id
    executor = Ur5eFieldAtomicSkillExecutor(exp, default_pregrasp_height=DEFAULT_PREGRASP_HEIGHT)
    reports: list[dict[str, Any]] = []
    stopped_on_failure = False
    t0 = time.time()
    try:
        for index, step in enumerate(steps):
            action = str(step.get("action") or "")
            parameters = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
            print(f"[{index + 1}/{len(steps)}] {action}")
            result = executor.execute(action, parameters)
            report = result_to_dict(result, index=index)
            reports.append(report)
            print(f"  success={bool(result.success)} status={result.status} message={result.message}")
            if args.stop_on_failure and not bool(result.success):
                stopped_on_failure = True
                break
        report_payload = {
            "schema_version": "ur5e_field_atomic_skill_smoke_report_v1",
            "actions_path": str(args.actions),
            "scene_xml": str(args.scene_xml),
            "scenario_id": args.scenario_id,
            "condition_id": args.condition_id,
            "action_count": len(reports),
            "success_count": sum(1 for item in reports if item.get("success")),
            "failure_count": sum(1 for item in reports if not item.get("success")),
            "stopped_on_failure": stopped_on_failure,
            "time_cost_s": round(time.time() - t0, 3),
            "actions": reports,
            "skill_results": exp.metrics.get("skill_results", []),
            "metrics": exp.metrics,
        }
        report_payload["save_report"] = str(args.save_report)
        _write_json(args.save_report, report_payload)
        print(f"saved report: {args.save_report}")
        if not args.no_experience_write and report_payload["failure_count"] > 0:
            _write_failed_experience(path=args.experience_write, steps=steps, reports=reports, report_payload=report_payload)
    finally:
        exp.close()


if __name__ == "__main__":
    main()
