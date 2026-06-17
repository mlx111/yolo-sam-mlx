"""Sandbox rollout utilities for R1Pro memory-policy candidates."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from experience_adapters import R1ProMujocoAdapter
from experience_core import ExperienceLibrary, SandboxCalibration, apply_critic, coerce_sandbox_initial_state, compute_group_calibrations
from source.run_r1pro_task_chain import run_task_chain, run_task_plan_chain


REPO_ROOT = Path(__file__).resolve().parents[2]
GALAXEA_ROOT = REPO_ROOT / "galaxea_mujoco"


DEFAULT_CRITIC_THRESHOLDS = {
    "min_object_lift": 0.05,
    "max_place_xy_error": 0.05,
    "max_place_z_error": 0.08,
    "max_dual_arm_height_mismatch": 0.02,
    "high_sim_real_gap": 0.65,
}


def select_sandbox_calibration(
    library: ExperienceLibrary,
    *,
    scenario: str,
    condition: str,
    object_class: str,
    robot_type: str = "mobile_dual_arm",
) -> dict[str, Any] | None:
    target_key = (robot_type, scenario, condition, object_class)
    for entry in library.entries:
        calibration = entry.sandbox_calibration
        details = calibration.details if isinstance(calibration.details, dict) else {}
        group = details.get("group_key") if isinstance(details.get("group_key"), dict) else {}
        if (
            calibration.calibration_id
            and str(group.get("robot_type") or entry.robot.robot_type) == target_key[0]
            and str(group.get("scenario_id") or entry.scenario_id) == target_key[1]
            and str(group.get("condition_id") or entry.condition_id) == target_key[2]
            and str(group.get("object_class") or entry.object_state.object_class) == target_key[3]
        ):
            return calibration.__dict__

    calibration = compute_group_calibrations(library.entries).get(target_key)
    if calibration is not None and calibration.calibration_id:
        return calibration.__dict__
    return None


@contextmanager
def _galaxea_workdir() -> Any:
    previous = Path.cwd()
    if GALAXEA_ROOT.exists():
        os.chdir(GALAXEA_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _calibration_dict(calibration: SandboxCalibration | dict[str, Any] | None) -> dict[str, Any]:
    if calibration is None:
        return {}
    if isinstance(calibration, SandboxCalibration):
        return calibration.__dict__
    return calibration if isinstance(calibration, dict) else {}


def _sandbox_state_dict(sandbox_initial_state: Any | None) -> dict[str, Any]:
    state = coerce_sandbox_initial_state(sandbox_initial_state)
    if state is None:
        return {}
    payload = state.to_dict()
    if isinstance(sandbox_initial_state, dict):
        for key in ("perturbation",):
            if key in sandbox_initial_state:
                payload[key] = sandbox_initial_state[key]
        evidence = sandbox_initial_state.get("evidence")
        if isinstance(evidence, dict):
            payload.setdefault("evidence", {}).update(evidence)
    return payload


def calibration_risk(calibration: SandboxCalibration | dict[str, Any] | None) -> dict[str, Any]:
    value = _calibration_dict(calibration)
    calibration_id = str(value.get("calibration_id") or "")
    if not calibration_id:
        return {
            "calibration_applied": False,
            "calibration_id": "",
            "calibration_risk_penalty": 0.0,
            "source_gap_ids": [],
            "object_pose_bias": [],
            "slip_risk_bias": 0.0,
            "contact_success_bias": 0.0,
            "calibration_confidence": 0.0,
        }

    object_pose_bias = [float(item) for item in (value.get("object_pose_bias") or [])[:3]]
    while len(object_pose_bias) < 3:
        object_pose_bias.append(0.0)
    slip_risk_bias = max(0.0, min(float(value.get("slip_risk_bias") or 0.0), 1.0))
    contact_success_bias = max(-1.0, min(float(value.get("contact_success_bias") or 0.0), 1.0))
    confidence = max(0.0, min(float(value.get("calibration_confidence") or 0.0), 1.0))
    pose_bias_norm = sum(component * component for component in object_pose_bias) ** 0.5
    pose_penalty = min(pose_bias_norm / 0.06, 1.0) * 0.08
    slip_penalty = 0.22 * slip_risk_bias
    contact_penalty = 0.18 * abs(min(contact_success_bias, 0.0))
    penalty = min((pose_penalty + slip_penalty + contact_penalty) * max(confidence, 0.25), 0.35)
    return {
        "calibration_applied": True,
        "calibration_id": calibration_id,
        "calibration_risk_penalty": round(penalty, 4),
        "source_gap_ids": [str(item) for item in value.get("source_gap_ids") or []],
        "object_pose_bias": [round(item, 6) for item in object_pose_bias],
        "slip_risk_bias": round(slip_risk_bias, 4),
        "contact_success_bias": round(contact_success_bias, 4),
        "calibration_confidence": round(confidence, 4),
    }


def sandbox_score(entry: Any, *, calibration: SandboxCalibration | dict[str, Any] | None = None) -> dict[str, Any]:
    success = bool(entry.result.get("success", entry.result.get("recovery_success", False)))
    task_success = bool(entry.result.get("task_success", False))
    critic_status = str(entry.critic_result.overall_status or "unknown")
    critic_risk = float(entry.critic_result.critic_risk_score or 0.0)
    calibration_report = calibration_risk(calibration)
    calibration_penalty = float(calibration_report["calibration_risk_penalty"])

    score = 0.0
    if task_success:
        score += 1.0
    elif success:
        score += 0.65
    if success and task_success:
        score += 0.15
    score -= critic_risk
    score -= calibration_penalty
    if critic_status == "warn":
        score -= 0.20
    elif critic_status == "block":
        score -= 0.60
    elif critic_status == "unknown":
        score -= 0.10

    normalized = max(0.0, min(score, 1.0))
    if critic_status == "block" or normalized < 0.25:
        decision = "reject"
    elif critic_status == "warn" or normalized < 0.55:
        decision = "review"
    else:
        decision = "accept"
    return {
        "sandbox_score": round(normalized, 4),
        "raw_sandbox_score": round(score, 4),
        "decision": decision,
        "success": success,
        "task_success": task_success,
        "critic_status": critic_status,
        "critic_risk_score": round(critic_risk, 4),
        **calibration_report,
    }


def evaluate_candidate_in_sandbox(
    *,
    scenario: str,
    condition: str,
    candidate_id: str,
    control_mode: str = "physical",
    keyframe_dir: Path | None = None,
    trace_dir: Path | None = None,
    model_path: str | Path | None = None,
    critic_thresholds: dict[str, float] | None = None,
    sandbox_calibration: SandboxCalibration | dict[str, Any] | None = None,
    sandbox_initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _galaxea_workdir():
        result = run_task_chain(
            scenario,
            condition,
            control_mode,
            candidate_id=candidate_id,
            keyframe_dir=keyframe_dir,
            trace_dir=trace_dir,
            model_path_override=str(model_path) if model_path else None,
            sandbox_calibration=_calibration_dict(sandbox_calibration),
            sandbox_initial_state=_sandbox_state_dict(sandbox_initial_state),
        )
    entry = R1ProMujocoAdapter().normalize_episode(result)
    apply_critic(entry, thresholds=critic_thresholds or DEFAULT_CRITIC_THRESHOLDS)
    score = sandbox_score(entry, calibration=sandbox_calibration)
    metrics = dict(result.metrics or {})
    failed_skills = [item.get("skill") for item in result.skill_trace if not item.get("success")]
    return {
        "scenario": scenario,
        "condition": condition,
        "candidate_id": candidate_id,
        "control_mode": control_mode,
        "runtime_scene_enabled": bool(model_path),
        "runtime_scene_model_path": str(model_path) if model_path else "",
        **score,
        "object_lift": metrics.get("object_lift"),
        "motion_critic": metrics.get("motion_critic", {}),
        "contact_stability": metrics.get("contact_stability", {}),
        "failure_diagnosis": metrics.get("failure_diagnosis", {}),
        "sandbox_calibration_effect": metrics.get("sandbox_calibration", {}),
        "sandbox_initial_state_effect": metrics.get("sandbox_initial_state", {}),
        "sandbox_dynamics_profile": metrics.get("sandbox_dynamics_profile", {}),
        "trajectory_trace": metrics.get("trajectory_trace", {}),
        "selected_place_site": result.selected_place_site,
        "failure_reason": result.failure_reason,
        "failed_skills": failed_skills,
        "critic_flags": list(entry.critic_result.rule_flags or []),
        "skill_trace": list(result.skill_trace or []),
        "keyframes": list(result.keyframes or []),
        "experience_id": entry.experience_id,
        "experience_entry": entry.to_dict(),
    }


def evaluate_plan_in_sandbox(
    *,
    scenario: str,
    condition: str,
    plan_steps: list[str],
    candidate_id: str = "llm_general_plan",
    control_mode: str = "physical",
    keyframe_dir: Path | None = None,
    trace_dir: Path | None = None,
    model_path: str | Path | None = None,
    critic_thresholds: dict[str, float] | None = None,
    sandbox_calibration: SandboxCalibration | dict[str, Any] | None = None,
    sandbox_initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _galaxea_workdir():
        result = run_task_plan_chain(
            scenario,
            condition,
            control_mode,
            plan_steps=plan_steps,
            keyframe_dir=keyframe_dir,
            trace_dir=trace_dir,
            model_path_override=str(model_path) if model_path else None,
            sandbox_calibration=_calibration_dict(sandbox_calibration),
            sandbox_initial_state=_sandbox_state_dict(sandbox_initial_state),
        )
    entry = R1ProMujocoAdapter().normalize_episode(result)
    apply_critic(entry, thresholds=critic_thresholds or DEFAULT_CRITIC_THRESHOLDS)
    score = sandbox_score(entry, calibration=sandbox_calibration)
    metrics = dict(result.metrics or {})
    failed_skills = [item.get("skill") for item in result.skill_trace if not item.get("success")]
    return {
        "scenario": scenario,
        "condition": condition,
        "candidate_id": candidate_id,
        "control_mode": control_mode,
        "runtime_scene_enabled": bool(model_path),
        "runtime_scene_model_path": str(model_path) if model_path else "",
        "general_plan_executor": True,
        "plan_steps": list(plan_steps),
        **score,
        "object_lift": metrics.get("object_lift"),
        "motion_critic": metrics.get("motion_critic", {}),
        "contact_stability": metrics.get("contact_stability", {}),
        "failure_diagnosis": metrics.get("failure_diagnosis", {}),
        "sandbox_calibration_effect": metrics.get("sandbox_calibration", {}),
        "sandbox_initial_state_effect": metrics.get("sandbox_initial_state", {}),
        "sandbox_dynamics_profile": metrics.get("sandbox_dynamics_profile", {}),
        "trajectory_trace": metrics.get("trajectory_trace", {}),
        "selected_place_site": result.selected_place_site,
        "failure_reason": result.failure_reason,
        "failed_skills": failed_skills,
        "critic_flags": list(entry.critic_result.rule_flags or []),
        "skill_trace": list(result.skill_trace or []),
        "keyframes": list(result.keyframes or []),
        "experience_id": entry.experience_id,
        "experience_entry": entry.to_dict(),
    }


def fuse_memory_and_sandbox(candidate_report: dict[str, Any], sandbox_report: dict[str, Any], *, sandbox_weight: float) -> dict[str, Any]:
    memory_score = float((candidate_report.get("candidate_score") or {}).get("candidate_score", 0.0))
    memory_risk = float((candidate_report.get("candidate_score") or {}).get("risk_score", 0.0))
    sandbox_value = float(sandbox_report.get("sandbox_score", 0.0))
    weight = max(0.0, min(float(sandbox_weight), 1.0))
    combined = (1.0 - weight) * memory_score + weight * sandbox_value
    sandbox_decision = str(sandbox_report.get("decision") or "review")
    memory_decision = str((candidate_report.get("candidate_score") or {}).get("decision") or "review")
    if sandbox_decision == "reject":
        decision = "reject"
    elif combined >= 0.65 and sandbox_decision == "accept":
        decision = "accept"
    elif combined >= 0.45 and memory_decision != "reject":
        decision = "review"
    else:
        decision = "reject"
    return {
        "memory_score": round(memory_score, 4),
        "memory_risk_score": round(memory_risk, 4),
        "sandbox_score": round(sandbox_value, 4),
        "sandbox_weight": round(weight, 4),
        "combined_score": round(combined, 4),
        "memory_decision": memory_decision,
        "sandbox_decision": sandbox_decision,
        "decision": decision,
    }


def summarize_sandbox_fusion(candidate_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sandbox rollout evidence for reporting and ablation tables."""

    if not candidate_reports:
        return {
            "candidate_count": 0,
            "sandbox_score_delta_avg": 0.0,
            "sandbox_score_delta_max": 0.0,
            "critic_status_counts": {},
            "critic_block_rate": 0.0,
            "critic_warn_rate": 0.0,
            "accepted_count": 0,
            "review_count": 0,
            "rejected_count": 0,
        }

    deltas: list[float] = []
    critic_status_counts: dict[str, int] = {}
    decision_counts = {"accept": 0, "review": 0, "reject": 0}
    for report in candidate_reports:
        fused = report.get("fused_score") or {}
        sandbox = report.get("sandbox") or {}
        memory_score = float(fused.get("memory_score", 0.0))
        sandbox_score_value = float(fused.get("sandbox_score", sandbox.get("sandbox_score", 0.0)))
        deltas.append(sandbox_score_value - memory_score)

        critic_status = str(sandbox.get("critic_status") or fused.get("sandbox_decision") or "unknown")
        critic_status_counts[critic_status] = critic_status_counts.get(critic_status, 0) + 1

        decision = str(fused.get("decision") or "reject")
        if decision not in decision_counts:
            decision_counts[decision] = 0
        decision_counts[decision] += 1

    total = len(candidate_reports)
    block_count = critic_status_counts.get("block", 0)
    warn_count = critic_status_counts.get("warn", 0)
    return {
        "candidate_count": total,
        "sandbox_score_delta_avg": round(sum(deltas) / total, 4),
        "sandbox_score_delta_max": round(max(deltas, key=abs), 4),
        "critic_status_counts": dict(sorted(critic_status_counts.items())),
        "critic_block_rate": round(block_count / total, 4),
        "critic_warn_rate": round(warn_count / total, 4),
        "accepted_count": int(decision_counts.get("accept", 0)),
        "review_count": int(decision_counts.get("review", 0)),
        "rejected_count": int(decision_counts.get("reject", 0)),
    }
