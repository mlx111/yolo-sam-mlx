"""Field atomic experience retrieval and parameter-prior summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any

from .schema import ExperienceEntry, GALAXEA_R1PRO_TORSO_NAMESPACE, MemoryGate, canonical_skill_action, utc_now
from .scoring import (
    action_lcs_ratio,
    action_set_overlap,
    critic_risk_score,
    entry_risk_adjustment,
    estimate_gap_uncertainty,
    estimate_real_success_prior,
    mitigation_profile,
    real_validation_bonus,
    risk_transfer_weight,
    top_failure_risk_memory,
)


def is_field_atomic_entry(entry: ExperienceEntry) -> bool:
    if _is_legacy_g3_g4_entry(entry):
        return False
    return (
        entry.memory_tags.get("memory_type") == "field_atomic_experience"
        or bool(entry.metadata.get("field_atomic"))
        or str(entry.memory_tags.get("memory_role") or "") == "semantic_plan_failure"
        or str(entry.memory_tags.get("memory_role") or "").startswith("field_atomic_")
    )


def _is_legacy_g3_g4_entry(entry: ExperienceEntry) -> bool:
    text = " ".join(
        str(item)
        for item in (
            entry.experience_id,
            entry.scenario_id,
            entry.condition_id,
            entry.task.get("name") if isinstance(entry.task, dict) else "",
            entry.raw_refs.get("model_path") if isinstance(entry.raw_refs, dict) else "",
        )
    ).lower()
    return any(marker in text for marker in ("g3", "g4", "pseudo_real_g3", "pseudo_real_g4", "r1pro_g3", "r1pro_g4"))


def field_atomic_action(entry: ExperienceEntry) -> str:
    if str(entry.memory_tags.get("memory_role") or "") == "semantic_plan_failure":
        failed_action = str(entry.failure_taxonomy.get("failed_action") or entry.failure_taxonomy.get("failure_action") or "")
        if failed_action:
            return canonical_skill_action(failed_action)
        semantic_failure = entry.execution_feedback.get("semantic_failure") if isinstance(entry.execution_feedback, dict) else {}
        if isinstance(semantic_failure, dict) and semantic_failure.get("failed_action"):
            return canonical_skill_action(str(semantic_failure.get("failed_action")))
    if str(entry.memory_tags.get("memory_type") or "") == "field_atomic_episode":
        failed_action = str(entry.failure_taxonomy.get("failure_action") or "")
        if failed_action:
            return canonical_skill_action(failed_action)
        failed = entry.execution_feedback.get("failed_action") if isinstance(entry.execution_feedback, dict) else {}
        if isinstance(failed, dict) and failed.get("action"):
            return canonical_skill_action(str(failed.get("action")))
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    return canonical_skill_action(str(feedback.get("field_atomic_action") or entry.metadata.get("field_atomic_action") or (entry.skill_sequence[0].name if entry.skill_sequence else "")))


def field_atomic_parameters(entry: ExperienceEntry) -> dict[str, Any]:
    if str(entry.memory_tags.get("memory_role") or "") == "semantic_plan_failure":
        semantic_failure = entry.execution_feedback.get("semantic_failure") if isinstance(entry.execution_feedback, dict) else {}
        if isinstance(semantic_failure, dict):
            params = semantic_failure.get("failed_action_parameters")
            if isinstance(params, dict):
                return _public_field_atomic_parameters(params)
    if str(entry.memory_tags.get("memory_type") or "") == "field_atomic_episode":
        failed = entry.execution_feedback.get("failed_action") if isinstance(entry.execution_feedback, dict) else {}
        if isinstance(failed, dict):
            params = failed.get("parameters")
            if isinstance(params, dict):
                return _public_field_atomic_parameters(params)
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    params = feedback.get("field_atomic_parameters")
    if isinstance(params, dict):
        return _public_field_atomic_parameters(params)
    for item in entry.skill_sequence:
        raw = item.raw if isinstance(item.raw, dict) else {}
        params = raw.get("parameters")
        if isinstance(params, dict):
            return _public_field_atomic_parameters(params)
    return {}


def _public_field_atomic_parameters(params: dict[str, Any]) -> dict[str, Any]:
    internal_keys = {
        "steps",
        "settle_steps",
        "max_joint_step",
        "fail_threshold",
        "success_threshold",
        "pregrasp_success_threshold",
        "direct_qpos",
        "stabilize",
        "lock_posture",
        "orientation_threshold",
    }
    return {
        str(key): value
        for key, value in params.items()
        if not str(key).startswith("_") and str(key) not in internal_keys
    }


def field_atomic_success(entry: ExperienceEntry) -> bool:
    role = str(entry.memory_tags.get("memory_role") or "")
    if role in {"field_atomic_success", "field_atomic_success_episode"}:
        return True
    if role in {"field_atomic_failure", "field_atomic_failure_episode", "field_atomic_recovery_failure_episode"}:
        return False
    return bool(entry.result.get("success", False))


def is_field_atomic_failure_rule(entry: ExperienceEntry) -> bool:
    return str(entry.memory_tags.get("memory_role") or "") == "field_atomic_failure_rule"


def field_atomic_experience_brief(entry: ExperienceEntry) -> dict[str, Any]:
    role = str(entry.memory_tags.get("memory_role") or "")
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    failed = feedback.get("failed_action") if isinstance(feedback.get("failed_action"), dict) else {}
    single_evidence = feedback.get("failure_evidence") if isinstance(feedback.get("failure_evidence"), dict) else {}
    episode_summary = feedback.get("episode_parameter_summary") if isinstance(feedback.get("episode_parameter_summary"), dict) else {}
    if not episode_summary:
        episode_summary = _episode_summary_from_action_trace(entry.action_trace)
    action_sequence = [item.name for item in entry.skill_sequence if item.name]
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    return {
        "experience_id": entry.experience_id,
        "memory_role": role,
        "memory_type": str(entry.memory_tags.get("memory_type") or ""),
        "scenario_id": entry.scenario_id,
        "episode_role": str(entry.memory_tags.get("episode_role") or key.get("episode_role") or ""),
        "action": field_atomic_action(entry),
        "parameters": field_atomic_parameters(entry),
        "success": field_atomic_success(entry),
        "status": entry.result.get("field_atomic_status", ""),
        "failure_type": str(entry.failure_taxonomy.get("failure_type") or ""),
        "failure_stage": str(entry.failure_taxonomy.get("failure_stage") or ""),
        "failure_reason": str(entry.failure_taxonomy.get("failure_reason") or feedback.get("llm_failure_summary") or ""),
        "plan_signature": str(key.get("plan_signature") or "->".join(action_sequence)),
        "action_sequence": action_sequence,
        "target_class": str(key.get("target_class") or entry.object_state.object_class or ""),
        "trajectory_mode": str(key.get("trajectory_mode") or ""),
        "source_failure_experience_id": str(key.get("source_failure_experience_id") or ""),
        "final_error": single_evidence.get("final_error") or entry.failure_taxonomy.get("final_error"),
        "stage_errors": single_evidence.get("stage_errors") or entry.failure_taxonomy.get("stage_errors"),
        "target_torso": single_evidence.get("target_torso"),
        "pregrasp_torso": single_evidence.get("pregrasp_torso"),
        "grasp_torso": single_evidence.get("grasp_torso"),
        "target_world": single_evidence.get("target_world"),
        "pregrasp_world": single_evidence.get("pregrasp_world"),
        "final_tcp_torso": single_evidence.get("final_tcp_torso"),
        "final_tcp_world": single_evidence.get("final_tcp_world"),
        "final_tcp_minus_pregrasp_torso": single_evidence.get("final_tcp_minus_pregrasp_torso"),
        "final_tcp_minus_pregrasp_world": single_evidence.get("final_tcp_minus_pregrasp_world"),
        "final_tcp_pregrasp_error_norm": single_evidence.get("final_tcp_pregrasp_error_norm"),
        "debug_tcp_minus_target_world": single_evidence.get("debug_tcp_minus_target_world"),
        "debug_tcp_target_error_norm": single_evidence.get("debug_tcp_target_error_norm"),
        "last_base_move_relative": episode_summary.get("last_base_move_relative"),
        "last_torso_posture": episode_summary.get("last_torso_posture"),
        "last_trajectory_plan": episode_summary.get("last_trajectory_plan"),
        "last_pregrasp": episode_summary.get("last_pregrasp"),
        "object_lift_world": entry.result.get("object_lift_world") or single_evidence.get("object_lift_world"),
        "skill_description": str(feedback.get("skill_description") or entry.metadata.get("skill_description") or ""),
        "parameter_schema": feedback.get("parameter_schema") if isinstance(feedback.get("parameter_schema"), dict) else {},
        "llm_failure_summary": str(feedback.get("llm_failure_summary") or entry.metadata.get("llm_failure_summary") or ""),
        "text_summary": str(entry.metadata.get("text_summary") or feedback.get("llm_failure_summary") or ""),
    }


def field_atomic_llm_failure_brief(entry: ExperienceEntry) -> dict[str, Any]:
    """Compact field-atomic memory view for LLM planning prompts."""
    full = field_atomic_experience_brief(entry)
    action = str(full.get("action") or "")
    params = full.get("parameters") if isinstance(full.get("parameters"), dict) else {}
    result: dict[str, Any] = {
        "experience_id": full.get("experience_id"),
        "memory_role": full.get("memory_role"),
        "source": entry.source,
        "validation_status": entry.validation_status,
        "dual_source_hint": _format_dual_source_memory_hint(entry),
        "action": action,
        "parameters": _llm_relevant_parameters(action, params),
        "llm_critic": _entry_llm_critic(entry),
    }
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def field_atomic_llm_planning_brief(entry: ExperienceEntry) -> dict[str, Any]:
    """Wrapper1-style memory brief for prompt generation.

    Success entries emphasize reusable execution patterns. Failure entries emphasize
    root cause, corrective direction, missing phases, and failed predicates.
    """

    full = field_atomic_experience_brief(entry)
    action = str(full.get("action") or "")
    params = full.get("parameters") if isinstance(full.get("parameters"), dict) else {}
    success = field_atomic_success(entry)
    result: dict[str, Any] = {
        "experience_id": full.get("experience_id"),
        "memory_role": full.get("memory_role"),
        "source": entry.source,
        "validation_status": entry.validation_status,
        "dual_source_hint": _format_dual_source_memory_hint(entry),
        "action": action,
        "parameters": _llm_relevant_parameters(action, params),
        "plan_signature": full.get("plan_signature"),
        "action_sequence": full.get("action_sequence"),
        "target_class": full.get("target_class"),
        "trajectory_mode": full.get("trajectory_mode"),
        "success": success,
        "status": full.get("status"),
        "llm_critic": _entry_llm_critic(entry),
    }
    if success:
        result.update({
            "success_reason": str(full.get("llm_failure_summary") or entry.metadata.get("text_summary") or ""),
            "text_summary": str(full.get("text_summary") or ""),
            "object_lift_world": full.get("object_lift_world"),
            "skill_description": full.get("skill_description"),
            "parameter_schema": full.get("parameter_schema"),
        })
        feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
        if feedback:
            success_feedback = {
                "task_success": bool(entry.result.get("task_success", False)),
                "object_lift_success": entry.result.get("object_lift_success"),
                "episode_parameter_summary": feedback.get("episode_parameter_summary"),
                "task_success_criteria": feedback.get("task_success_criteria"),
            }
            result["success_evidence"] = {key: value for key, value in success_feedback.items() if value not in (None, "", [], {})}
    else:
        result.update({
            "llm_critic": _entry_llm_critic(entry),
        })
        if action == "lift":
            result.update({
                "object_lift_world": full.get("object_lift_world"),
            })
        elif action == "move_to_pregrasp":
            result.update({
                "target_torso": full.get("target_torso"),
                "pregrasp_torso": full.get("pregrasp_torso"),
                "final_tcp_torso": full.get("final_tcp_torso"),
                "pregrasp_error_vector_torso": full.get("final_tcp_minus_pregrasp_torso"),
                "pregrasp_error_vector_world": full.get("final_tcp_minus_pregrasp_world"),
                "pregrasp_error_norm": full.get("final_tcp_pregrasp_error_norm") or full.get("final_error"),
                "pregrasp_error_hint": _pregrasp_error_hint(full.get("final_tcp_minus_pregrasp_world")),
            })
        elif action == "transport_to_detected_target":
            result.update({
                "target_source_torso": _single_evidence_value(entry, "target_source_torso"),
            })
        elif action == "head_camera_grounded_sam2_pose":
            result.update({
                "target_class": params.get("target_class") or full.get("target_class"),
                "median_world": _single_evidence_value(entry, "median_world"),
            })
        anomaly_state = _entry_anomaly_state(entry)
        if anomaly_state:
            result["anomaly_state"] = {
                key: anomaly_state.get(key)
                for key in (
                    "failure_stage",
                    "failure_type",
                    "critic_failure_stage",
                    "critic_failure_type",
                    "target_class",
                    "side",
                    "target_torso_y_sign",
                    "final_error_bucket",
                    "object_lift_bucket",
                )
                if anomaly_state.get(key) not in (None, "", [], {})
            }
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _format_dual_source_memory_hint(entry: ExperienceEntry) -> str:
    gap = entry.sim_real_gap
    outcome = gap.outcome_gap if isinstance(gap.outcome_gap, dict) else {}
    outcome_type = str(outcome.get("type") or "")
    tags = entry.memory_tags if isinstance(entry.memory_tags, dict) else {}
    parts: list[str] = []
    if entry.source or entry.validation_status:
        parts.append(f"来源={entry.source or 'unknown'}, 验证={entry.validation_status or 'unknown'}")
    if tags.get("memory_role"):
        parts.append(f"角色={tags.get('memory_role')}")
    if outcome_type:
        bits = [f"outcome={outcome_type}"]
        if gap.gap_score is not None:
            bits.append(f"gap_score={float(gap.gap_score or 0.0):.2f}")
        if gap.uncertainty is not None:
            bits.append(f"uncertainty={float(gap.uncertainty or 0.0):.2f}")
        parts.append("Sim-Real Gap: " + ", ".join(bits))
        if outcome_type == "sim_success_real_fail":
            parts.append("风险提示: 该方案仿真成功但真实/伪真机失败，不能直接照搬，应调整接触、夹爪或感知步骤")
        elif outcome_type == "matched_success" and entry.source in {"real", "pseudo_real"}:
            parts.append("可信提示: 该经验与真实/伪真机成功结果一致，可优先参考")
    elif entry.validation_status in {"real_executed", "real_validated"} and field_atomic_success(entry):
        parts.append("可信提示: 真机/伪真机执行成功经验，可优先参考")
    return "; ".join(parts[:4])


def query_field_atomic_experiences(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    anomaly_state: dict[str, Any] | None = None,
    task_stage: str = "",
    text_summary: str = "",
    limit: int = 8,
) -> list[ExperienceEntry]:
    """Wrapper1-style field atomic retrieval for Galaxea recovery planning."""
    query_state = anomaly_state if isinstance(anomaly_state, dict) else {}
    scored: list[tuple[float, int, ExperienceEntry]] = []
    for order, entry in enumerate(entries):
        if not is_field_atomic_entry(entry):
            continue
        if skill_namespace and entry.skill_namespace != skill_namespace:
            continue
        score = 0.05
        key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
        memory_tags = entry.memory_tags if isinstance(entry.memory_tags, dict) else {}
        role = str(memory_tags.get("memory_role") or "")
        memory_type = str(memory_tags.get("memory_type") or "")
        entry_stage = str(key.get("task_stage") or key.get("failure_stage") or entry.task.get("stage") or "")

        if scenario_id and entry.scenario_id == scenario_id:
            score += 1.8
        elif scenario_id and entry.scenario_id:
            score += 0.15
        elif scenario_id:
            score += 0.08
        if condition_id and entry.condition_id == condition_id:
            score += 0.8
        elif condition_id:
            score += 0.04
        if memory_type == "field_atomic_episode":
            score += 1.0
        elif memory_type == "field_atomic_experience":
            score += 0.25
        if "failure" in role:
            score += 0.7
        if "success_episode" in role:
            score += 0.45
        if role.startswith("field_atomic_"):
            score += 0.08
        if task_stage and entry_stage and task_stage == entry_stage:
            score += 0.8

        score += 2.0 * _dict_similarity(query_state, _entry_anomaly_state(entry))
        score += 0.8 * _dict_similarity(query_state, key)
        score += 0.5 * _token_jaccard(text_summary, str(entry.metadata.get("text_summary") or ""))
        score += 0.35 * _token_jaccard(text_summary, _entry_memory_lesson(entry))

        critic = _entry_llm_critic(entry)
        if critic:
            score += 0.35
            missing = critic.get("missing_phases")
            if isinstance(missing, list) and missing:
                score += 0.15
        if _entry_failed_predicates(entry):
            score += 0.2
        score += _memory_tier_boost(entry)
        scored.append((score, order, entry))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [entry for _, _, entry in scored[: max(0, int(limit))]]


def query_field_atomic_experience_matches(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    available_actions: set[str] | list[str] | None = None,
    retrieval_key: dict[str, Any] | None = None,
    anomaly_state: dict[str, Any] | None = None,
    task_stage: str = "",
    text_summary: str = "",
    include_failed: bool = True,
    visual_scores: dict[str, float] | None = None,
    gap_aware: bool = False,
    risk_aware: bool = False,
    diversity_lambda: float = 0.0,
    limit: int = 8,
) -> list[tuple[ExperienceEntry, float, dict[str, Any]]]:
    """Return field-atomic retrieval matches with wrapper1-style scores."""
    allowed = {canonical_skill_action(str(item)) for item in (available_actions or []) if str(item)}
    query_state = anomaly_state if isinstance(anomaly_state, dict) else {}
    query_key = retrieval_key if isinstance(retrieval_key, dict) else {}
    visual_scores = visual_scores if isinstance(visual_scores, dict) else {}
    matches: list[tuple[ExperienceEntry, float, dict[str, Any]]] = []
    for entry in entries:
        if not is_field_atomic_entry(entry):
            continue
        if skill_namespace and entry.skill_namespace != skill_namespace:
            continue
        if not include_failed and not field_atomic_success(entry):
            continue
        actions = _entry_actions(entry)
        score, explanation = _score_field_atomic_entry(
            entry,
            actions=actions,
            allowed=allowed,
            scenario_id=scenario_id,
            condition_id=condition_id,
            retrieval_key=query_key,
            anomaly_state=query_state,
            task_stage=task_stage,
            text_summary=text_summary,
            visual_score=float(visual_scores.get(entry.experience_id, 0.0) or 0.0),
            gap_aware=gap_aware,
            risk_aware=risk_aware,
        )
        if scenario_id and entry.scenario_id == scenario_id:
            score += 0.20
        elif scenario_id and entry.scenario_id:
            score += 0.05
        if condition_id and entry.condition_id == condition_id:
            score += 0.10
        elif condition_id and entry.condition_id:
            score += 0.03
        if allowed and actions and not set(actions).issubset(allowed):
            score *= 0.85
        matches.append((entry, score, explanation))
    matches.sort(key=lambda item: (-item[1], item[0].created_at))
    if not matches:
        return []
    if diversity_lambda > 0.0 and len(matches) > limit:
        matches = _mmr_field_atomic_matches(matches, limit=limit, diversity_lambda=diversity_lambda)
    return matches[: max(0, int(limit))]


def score_field_atomic_candidate_plan(
    candidate_steps: list[dict[str, Any]],
    retrieved_experiences: list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]],
    query_context: dict[str, Any] | None = None,
    policy_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a Galaxea field-atomic candidate against retrieved memories."""
    candidate_actions = [
        canonical_skill_action(str(step.get("action") or step.get("name") or step.get("skill") or ""))
        for step in candidate_steps
        if isinstance(step, dict) and str(step.get("action") or step.get("name") or step.get("skill") or "")
    ]
    candidate_step_params = {
        canonical_skill_action(str(step.get("action") or step.get("name") or step.get("skill") or "")): _public_field_atomic_parameters(
            step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        )
        for step in candidate_steps
        if isinstance(step, dict) and str(step.get("action") or step.get("name") or step.get("skill") or "")
    }
    positive_support = 0.0
    failure_overlap_risk = 0.0
    parameter_failure_risk = 0.0
    max_gap_uncertainty = 0.0
    max_critic_risk = 0.0
    mitigation_support = 0.0
    evidence: list[dict[str, Any]] = []
    normalized_matches: list[tuple[ExperienceEntry, float]] = []
    for item in retrieved_experiences or []:
        if len(item) >= 2:
            normalized_matches.append((item[0], float(item[1] or 0.0)))

    for entry, score in normalized_matches:
        entry_actions = _entry_actions(entry)
        overlap = action_lcs_ratio(candidate_actions, entry_actions)
        if overlap <= 0.0:
            overlap = 0.05
        jaccard = action_set_overlap(candidate_actions, entry_actions)
        parameter_overlap = _parameter_failure_overlap(candidate_step_params, _parameter_failure_summary(entry))
        transfer = risk_transfer_weight(
            entry,
            action_overlap=overlap,
            action_jaccard=jaccard,
            query_context=query_context,
            policy_calibration=policy_calibration,
        )
        risk_weight = max(float(transfer.get("risk_transfer_weight", 0.0)), 0.05 if overlap > 0.0 else 0.0)
        if parameter_overlap > 0.0:
            risk_weight = min(1.0, risk_weight + 0.35 * parameter_overlap)
            parameter_failure_risk = max(parameter_failure_risk, parameter_overlap)
        adjustment = entry_risk_adjustment(entry)
        mitigation = mitigation_profile(candidate_actions, entry, query_context)
        entry_success = field_atomic_success(entry)
        weight = max(float(score), 0.0) * max(overlap, 0.05)
        entry_gap = estimate_gap_uncertainty(entry)
        entry_critic = critic_risk_score(entry)
        is_risky_memory = (not entry_success) or str(entry.sim_real_gap.outcome_gap.get("type") or "") == "sim_success_real_fail"
        if is_risky_memory and mitigation.get("mitigations"):
            risk_weight *= float(mitigation.get("risk_scale", 1.0))
            mitigation_support += max(float(score), 0.0) * float(mitigation.get("support_bonus", 0.0))
        if float(adjustment.get("score_adjustment", 0.0)) < 0.0:
            adjusted_weight = max(0.0, weight + float(adjustment["score_adjustment"]) * risk_weight)
        else:
            adjusted_weight = max(0.0, weight + float(adjustment.get("score_adjustment", 0.0)))
        if entry_success:
            positive_support += adjusted_weight
        else:
            failure_overlap_risk = max(failure_overlap_risk, risk_weight)
        max_gap_uncertainty = max(max_gap_uncertainty, entry_gap * risk_weight)
        max_critic_risk = max(max_critic_risk, entry_critic * max(risk_weight, 0.05))
        evidence.append({
            "experience_id": entry.experience_id,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "source": entry.source,
            "success": entry_success,
            "retrieval_score": round(float(score), 4),
            "action_overlap": round(overlap, 4),
            "action_jaccard": round(jaccard, 4),
            "parameter_overlap": round(parameter_overlap, 4),
            "risk_transfer": transfer,
            "mitigation": mitigation,
            "adjustment": adjustment,
            "gap_uncertainty": round(entry_gap, 4),
            "critic_risk": round(entry_critic, 4),
        })

    query_context = query_context if isinstance(query_context, dict) else {}
    real_prior = estimate_real_success_prior(normalized_matches)
    failure_memory = top_failure_risk_memory(
        candidate_actions,
        normalized_matches,
        query_context=query_context,
        policy_calibration=policy_calibration,
        top_o=3,
    )
    failure_risk_penalty = float(failure_memory.get("failure_risk_penalty", 0.0))
    support_score = _clamp01(positive_support + mitigation_support)
    risk_score = _clamp01(
        0.28 * failure_overlap_risk
        + 0.18 * parameter_failure_risk
        + 0.27 * max_gap_uncertainty
        + 0.25 * max_critic_risk
        + 0.20 * failure_risk_penalty
    )
    candidate_score = _clamp01(
        0.55 * support_score
        + 0.25 * float(real_prior.get("real_success_prior", 0.0))
        + 0.20 * (1.0 - risk_score)
    )
    if risk_score >= 0.65:
        decision = "rewrite"
    elif candidate_score >= 0.60:
        decision = "accept"
    elif candidate_score >= 0.40:
        decision = "review"
    else:
        decision = "reject"
    return {
        "candidate_score": round(candidate_score, 4),
        "decision": decision,
        "support_score": round(support_score, 4),
        "mitigation_support": round(mitigation_support, 4),
        "risk_score": round(risk_score, 4),
        "failure_overlap_risk": round(failure_overlap_risk, 4),
        "parameter_failure_risk": round(parameter_failure_risk, 4),
        "terminal_risk_score": failure_memory.get("terminal_risk_score", 0.0),
        "failure_risk_penalty": failure_memory.get("failure_risk_penalty", 0.0),
        "top_failure_risks": failure_memory.get("top_failure_risks", []),
        "gap_uncertainty": round(max_gap_uncertainty, 4),
        "critic_risk": round(max_critic_risk, 4),
        **real_prior,
        "candidate_actions": candidate_actions,
        "evidence": evidence,
    }


def _compact_failure_reason(text: str, *, max_chars: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _entry_actions(entry: ExperienceEntry) -> list[str]:
    actions = [canonical_skill_action(item.name) for item in entry.skill_sequence if item.name]
    if actions:
        return actions
    if isinstance(entry.action_trace, list):
        return [
            canonical_skill_action(str(item.get("action") or ""))
            for item in entry.action_trace
            if isinstance(item, dict) and str(item.get("action") or "")
        ]
    action = field_atomic_action(entry)
    return [action] if action else []


def _score_field_atomic_entry(
    entry: ExperienceEntry,
    *,
    actions: list[str],
    allowed: set[str],
    scenario_id: str,
    condition_id: str,
    retrieval_key: dict[str, Any],
    anomaly_state: dict[str, Any],
    task_stage: str,
    text_summary: str,
    visual_score: float,
    gap_aware: bool,
    risk_aware: bool,
) -> tuple[float, dict[str, Any]]:
    action_coverage = len(set(actions) & allowed) / max(len(allowed), 1) if allowed else 1.0
    validation_score = _validation_score(entry.validation_status)
    result_success = 1.0 if field_atomic_success(entry) else 0.0
    entry_key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    retrieval_key_similarity = _dict_similarity(retrieval_key, entry_key)
    anomaly_state_similarity = _dict_similarity(anomaly_state, _entry_anomaly_state(entry))
    entry_stage = str(entry_key.get("task_stage") or entry_key.get("failure_stage") or entry.task.get("stage") or "")
    task_stage_match = 1.0 if task_stage and entry_stage == task_stage else 0.0
    condition_id_match = 1.0 if condition_id and entry.condition_id == condition_id else 0.0
    text_summary_similarity = _token_jaccard(text_summary, str(entry.metadata.get("text_summary") or ""))
    structured_similarity = (
        0.20 * validation_score
        + 0.15 * result_success
        + 0.15 * retrieval_key_similarity
        + 0.15 * condition_id_match
        + 0.15 * anomaly_state_similarity
        + 0.10 * task_stage_match
        + 0.05 * min(action_coverage, 1.0)
    )
    text_score = 0.05 * text_summary_similarity
    tier_boost = _memory_tier_boost(entry)
    visual_boost = 0.12 * _clamp01(visual_score)
    gap_uncertainty = estimate_gap_uncertainty(entry) if gap_aware else 0.0
    critic_risk = critic_risk_score(entry) if risk_aware else 0.0
    real_bonus = real_validation_bonus(entry) if gap_aware or risk_aware else 0.0
    risk_adjustment = 0.10 * real_bonus - 0.15 * gap_uncertainty - 0.15 * critic_risk
    score = 1.0 + structured_similarity + text_score + tier_boost + visual_boost + risk_adjustment
    return score, {
        "final_score": round(score, 4),
        "structured_similarity": round(structured_similarity, 4),
        "validation_score": round(validation_score, 4),
        "result_success": result_success,
        "retrieval_key_similarity": round(retrieval_key_similarity, 4),
        "anomaly_state_similarity": round(anomaly_state_similarity, 4),
        "task_stage_match": task_stage_match,
        "condition_id_match": condition_id_match,
        "action_coverage": round(min(action_coverage, 1.0), 4),
        "text_summary_similarity": round(text_summary_similarity, 4),
        "text_score": round(text_score, 4),
        "memory_tier_boost": round(tier_boost, 4),
        "visual_similarity": round(_clamp01(visual_score), 4),
        "visual_boost": round(visual_boost, 4),
        "gap_aware": bool(gap_aware),
        "risk_aware": bool(risk_aware),
        "gap_uncertainty": round(gap_uncertainty, 4),
        "critic_risk": round(critic_risk, 4),
        "real_validation_bonus": round(real_bonus, 4),
        "risk_adjustment": round(risk_adjustment, 4),
    }


def _validation_score(status: str) -> float:
    status = str(status or "")
    if status in {"real_validated"}:
        return 1.0
    if status in {"real_executed", "pseudo_real_executed", "simulation_validated", "sandbox_validated"}:
        return 0.65
    if status in {"simulation_failed", "failed", "failure"}:
        return 0.35
    return 0.0


def _field_atomic_real_success_prior(matches: list[tuple[ExperienceEntry, float]]) -> dict[str, Any]:
    real_success = 0
    real_total = 0
    for entry, _ in matches:
        if entry.source not in {"real", "pseudo_real"} and entry.validation_status not in {"real_validated", "real_executed"}:
            continue
        real_total += 1
        if field_atomic_success(entry):
            real_success += 1
    prior = real_success / real_total if real_total else 0.0
    return {
        "real_success_prior": round(prior, 4),
        "real_evidence_count": real_total,
    }


def _mmr_field_atomic_matches(
    matches: list[tuple[ExperienceEntry, float, dict[str, Any]]],
    *,
    limit: int,
    diversity_lambda: float,
) -> list[tuple[ExperienceEntry, float, dict[str, Any]]]:
    selected: list[tuple[ExperienceEntry, float, dict[str, Any]]] = []
    remaining = list(matches)
    while remaining and len(selected) < max(0, int(limit)):
        best_index = 0
        best_score = float("-inf")
        for index, item in enumerate(remaining):
            entry, score, _ = item
            max_similarity = 0.0
            for selected_entry, _, _ in selected:
                max_similarity = max(max_similarity, action_lcs_ratio(_entry_actions(entry), _entry_actions(selected_entry)))
            mmr_score = (1.0 - diversity_lambda) * float(score) - diversity_lambda * max_similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _llm_relevant_parameters(action: str, params: dict[str, Any]) -> dict[str, Any]:
    common = {
        "side",
        "target_class",
        "pregrasp_offset_x",
        "pregrasp_offset_y",
        "pregrasp_offset_z",
        "grasp_offset_x",
        "grasp_offset_y",
        "grasp_offset_z",
        "visual_grasp_offset_z",
        "topdown_mode",
        "mode",
        "side_offset_x",
        "side_offset_y",
        "side_offset_z",
        "place_offset_x",
        "place_offset_y",
        "x",
        "y",
        "yaw",
        "level",
        "distance",
    }
    if action == "head_camera_grounded_sam2_pose":
        common = {"target_class"}
    elif action in {"close_gripper", "open_gripper"}:
        common = {"side"}
    return {key: params[key] for key in sorted(common) if key in params}


def _single_evidence_value(entry: ExperienceEntry, key: str) -> Any:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    evidence = feedback.get("failure_evidence") if isinstance(feedback.get("failure_evidence"), dict) else {}
    if key in evidence:
        return evidence.get(key)
    failed = feedback.get("failed_action") if isinstance(feedback.get("failed_action"), dict) else {}
    raw = failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {}
    return raw.get(key)


def _pregrasp_error_hint(vector: Any) -> str:
    if not isinstance(vector, list) or len(vector) < 3:
        return ""
    try:
        values = [float(vector[index]) for index in range(3)]
    except (TypeError, ValueError):
        return ""
    axes = ("x", "y", "z")
    parts = [
        f"{axis}{'+' if value >= 0 else '-'}{abs(value):.3f}m"
        for axis, value in zip(axes, values)
        if abs(value) >= 0.01
    ]
    if not parts:
        return "actual TCP reached the pregrasp target within 1cm on each axis."
    norm = sum(value * value for value in values) ** 0.5
    return (
        "move_to_pregrasp actual TCP offset from requested pregrasp: "
        + ", ".join(parts)
        + f"; norm={norm:.3f}m. Treat this as reachability/base-torso evidence, not as a reason to blindly perturb grasp offsets."
    )


def _entry_llm_critic(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    critic = feedback.get("llm_critic")
    if isinstance(critic, dict) and critic and (not critic.get("error") or critic.get("rule_fallback")):
        return critic
    return {}


def _entry_memory_lesson(entry: ExperienceEntry) -> str:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    if feedback.get("memory_lesson"):
        return str(feedback.get("memory_lesson"))
    critic = _entry_llm_critic(entry)
    parts = []
    if critic.get("root_cause"):
        parts.append(str(critic.get("root_cause")))
    return "；".join(parts)[:700]


def _parameter_failure_summary(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    critic = feedback.get("llm_critic") if isinstance(feedback.get("llm_critic"), dict) else {}
    summary = critic.get("parameter_failure_summary") if isinstance(critic, dict) else {}
    return summary if isinstance(summary, dict) else {}


def _parameter_failure_items(entry: ExperienceEntry) -> list[dict[str, Any]]:
    summary = _parameter_failure_summary(entry)
    items = summary.get("items") if isinstance(summary.get("items"), list) else []
    if items:
        return [item for item in items if isinstance(item, dict)]
    return [summary] if summary else []


def _parameter_failure_overlap(
    candidate_step_params: dict[str, dict[str, Any]],
    parameter_failure_summary: dict[str, Any],
) -> float:
    items = parameter_failure_summary.get("items") if isinstance(parameter_failure_summary.get("items"), list) else []
    if items:
        return max(
            (_parameter_failure_overlap(candidate_step_params, item) for item in items if isinstance(item, dict)),
            default=0.0,
        )
    action = canonical_skill_action(str(parameter_failure_summary.get("action") or ""))
    if not action:
        return 0.0
    candidate_params = candidate_step_params.get(action)
    if not isinstance(candidate_params, dict) or not candidate_params:
        return 0.0
    bad_keys = parameter_failure_summary.get("bad_keys")
    bad_values = parameter_failure_summary.get("bad_values")
    expected_range = parameter_failure_summary.get("expected_range")
    score = 0.0
    total = 0.0
    if isinstance(bad_keys, list) and bad_keys:
        total += 1.0
        matched_bad = sum(1 for key in bad_keys if key in candidate_params)
        score += matched_bad / max(len(bad_keys), 1)
    if isinstance(bad_values, dict) and bad_values:
        total += 1.0
        matched = 0
        for key, value in bad_values.items():
            if key in candidate_params and str(candidate_params.get(key)) == str(value):
                matched += 1
        score += matched / max(len(bad_values), 1)
    if isinstance(expected_range, dict) and expected_range:
        total += 1.0
        matched = 0
        checked = 0
        for key, bounds in expected_range.items():
            if key not in candidate_params:
                continue
            checked += 1
            if isinstance(bounds, list) and len(bounds) >= 2:
                try:
                    value = float(candidate_params.get(key))
                    low = float(bounds[0])
                    high = float(bounds[1])
                    if low <= value <= high:
                        matched += 1
                except (TypeError, ValueError):
                    continue
        if checked:
            score += matched / checked
        else:
            total -= 1.0
    return score / total if total > 0.0 else 0.0


def _entry_failed_predicates(entry: ExperienceEntry) -> list[str]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    critic = feedback.get("llm_critic") if isinstance(feedback.get("llm_critic"), dict) else {}
    predicates = critic.get("failed_predicates") if isinstance(critic, dict) else None
    if isinstance(predicates, list):
        return [str(item) for item in predicates if str(item)]
    return []


def _entry_anomaly_state(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    failed = feedback.get("failed_action") if isinstance(feedback.get("failed_action"), dict) else {}
    raw = failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {}
    params = failed.get("parameters") if isinstance(failed.get("parameters"), dict) else {}
    target_torso = raw.get("target_torso") if isinstance(raw.get("target_torso"), list) else None
    return {
        "failure_stage": entry.failure_taxonomy.get("failure_stage", ""),
        "failure_type": entry.failure_taxonomy.get("failure_type", ""),
        "target_class": params.get("target_class") or key.get("target_class") or entry.object_state.object_class or "",
        "side": params.get("side") or key.get("side") or "",
        "target_torso_y_sign": _sign_bucket(target_torso[1]) if target_torso and len(target_torso) > 1 else key.get("target_torso_y_sign", ""),
        "final_error_bucket": _error_bucket(raw.get("final_error") or entry.failure_taxonomy.get("final_error")),
        "failed_predicates": _entry_failed_predicates(entry),
    }


def _dict_similarity(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    if not query or not candidate:
        return 0.0
    comparable = [
        key for key, value in query.items()
        if value not in (None, "", [], {}) and candidate.get(key) not in (None, "", [], {})
    ]
    if not comparable:
        return 0.0
    score = 0.0
    total = 0.0
    for key in comparable:
        total += 1.0
        left = query.get(key)
        right = candidate.get(key)
        if isinstance(left, list) or isinstance(right, list):
            score += _list_similarity(left, right)
        elif str(left) == str(right):
            score += 1.0
    return score / total if total else 0.0


def _list_similarity(left: Any, right: Any) -> float:
    left_set = {str(item) for item in left} if isinstance(left, list) else ({str(left)} if left not in (None, "") else set())
    right_set = {str(item) for item in right} if isinstance(right, list) else ({str(right)} if right not in (None, "") else set())
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() or ch in "_-" else " " for ch in str(text))
    return {item for item in normalized.split() if len(item) >= 2}


def _memory_tier_boost(entry: ExperienceEntry) -> float:
    lifecycle = entry.lifecycle if isinstance(getattr(entry, "lifecycle", None), dict) else {}
    tier = str(lifecycle.get("memory_tier") or entry.memory_tags.get("memory_tier") or "")
    if tier == "ltm":
        return 0.35
    if tier == "stm":
        return 0.1
    return 0.0


def _sign_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric > 0.03:
        return "positive"
    if numeric < -0.03:
        return "negative"
    return "center"


def _error_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric < 0.02:
        return "low"
    if numeric < 0.10:
        return "medium"
    return "high"


def _episode_summary_from_action_trace(action_trace: Any) -> dict[str, Any]:
    if not isinstance(action_trace, list):
        return {}
    summary: dict[str, Any] = {"actions": []}
    for item in action_trace:
        if not isinstance(item, dict):
            continue
        action = canonical_skill_action(str(item.get("action") or ""))
        params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        public_params = _public_field_atomic_parameters(params)
        row = {
            "index": item.get("index"),
            "action": action,
            "success": bool(item.get("success", False)),
            "parameters": public_params,
            "final_error": raw.get("final_error"),
            "debug_tcp_minus_target_world": raw.get("debug_tcp_minus_target_world"),
            "debug_tcp_target_error_norm": raw.get("debug_tcp_target_error_norm"),
            "object_lift_world": raw.get("object_lift_world"),
        }
        summary["actions"].append({key: value for key, value in row.items() if value not in (None, "", [], {})})
        if action == "move_base_relative":
            summary["last_base_move_relative"] = public_params
        elif action == "set_torso_posture":
            summary["last_torso_posture"] = public_params
        elif action == "plan_cartesian_trajectory":
            summary["last_trajectory_plan"] = public_params
        elif action == "move_to_pregrasp":
            summary["last_pregrasp"] = row
        elif action == "approach_object":
            summary["last_approach"] = row
    return summary


def build_field_atomic_parameter_priors(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    action: str = "",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    limit: int = 12,
) -> dict[str, Any]:
    matched: list[ExperienceEntry] = []
    for entry in entries:
        if not is_field_atomic_entry(entry):
            continue
        if skill_namespace and entry.skill_namespace != skill_namespace:
            continue
        if scenario_id and entry.scenario_id and entry.scenario_id != scenario_id:
            continue
        if condition_id and entry.condition_id and entry.condition_id != condition_id:
            continue
        if action and field_atomic_action(entry) != canonical_skill_action(action):
            continue
        matched.append(entry)

    by_action: dict[str, dict[str, Any]] = {}
    action_counts = Counter(field_atomic_action(entry) for entry in matched)
    role_counts = Counter(str(entry.memory_tags.get("memory_role") or "") for entry in matched)
    for action_name in sorted(action_counts):
        action_entries = [entry for entry in matched if field_atomic_action(entry) == action_name]
        success_entries = [entry for entry in action_entries if field_atomic_success(entry)]
        failure_entries = [entry for entry in action_entries if not field_atomic_success(entry)]
        by_action[action_name] = {
            "entry_count": len(action_entries),
            "success_count": len(success_entries),
            "failure_count": len(failure_entries),
            "success_rate": round(len(success_entries) / len(action_entries), 4) if action_entries else 0.0,
            "recommended_from_success": _summarize_parameter_entries(success_entries),
            "avoid_from_failure": _summarize_parameter_entries(failure_entries),
            "parameter_failure_summary": _summarize_parameter_failures(failure_entries),
            "success_ids": [entry.experience_id for entry in success_entries[:limit]],
            "failure_ids": [entry.experience_id for entry in failure_entries[:limit]],
        }

    semantic_failures = [
        entry for entry in matched
        if str(entry.memory_tags.get("memory_role") or "") == "semantic_plan_failure"
    ]
    if semantic_failures:
        semantic_failure_summary = {
            "entry_count": len(semantic_failures),
            "failure_ids": [entry.experience_id for entry in semantic_failures[:limit]],
            "failure_types": Counter(str(entry.failure_taxonomy.get("failure_type") or "") for entry in semantic_failures),
            "failure_reasons": [
                _compact_failure_reason(str(entry.failure_taxonomy.get("failure_reason") or _entry_memory_lesson(entry)))
                for entry in semantic_failures[:limit]
            ],
            "memory_lessons": [_entry_memory_lesson(entry) for entry in semantic_failures[:limit]],
            "critic_root_causes": [
                _entry_llm_critic(entry).get("root_cause")
                for entry in semantic_failures[:limit]
                if _entry_llm_critic(entry)
            ],
        }
    else:
        semantic_failure_summary = {}

    return {
        "schema_version": "field_atomic_parameter_priors_v1",
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "skill_namespace": skill_namespace,
        "action_filter": canonical_skill_action(action),
        "field_atomic_entry_count": len(matched),
        "field_atomic_success_count": sum(1 for entry in matched if field_atomic_success(entry)),
        "field_atomic_failure_count": sum(1 for entry in matched if not field_atomic_success(entry)),
        "memory_role_distribution": dict(role_counts),
        "action_distribution": dict(action_counts),
        "by_action": by_action,
        "semantic_plan_failure_summary": semantic_failure_summary,
        "evidence_ids": [entry.experience_id for entry in matched[:limit]],
        "usage": "Use recommended_from_success as parameter examples, avoid_from_failure as failure priors, and semantic_plan_failure_summary as pre-sandbox failure memory for future field_atomic plans.",
    }


def build_field_atomic_planner_input(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    goal: str = "",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    limit: int = 12,
) -> dict[str, Any]:
    priors = build_field_atomic_parameter_priors(
        entries,
        scenario_id=scenario_id,
        condition_id=condition_id,
        skill_namespace=skill_namespace,
        limit=limit,
    )
    recovery_rules = build_galaxea_recovery_rules(
        entries,
        scenario_id=scenario_id,
        condition_id=condition_id,
        skill_namespace=skill_namespace,
        limit=limit,
    )
    recent = []
    recent_success = []
    recent_failure = []
    for entry in reversed(entries):
        if not is_field_atomic_entry(entry):
            continue
        if skill_namespace and entry.skill_namespace != skill_namespace:
            continue
        if scenario_id and entry.scenario_id and entry.scenario_id != scenario_id:
            continue
        if condition_id and entry.condition_id and entry.condition_id != condition_id:
            continue
        brief = field_atomic_llm_planning_brief(entry)
        recent.append(brief)
        if field_atomic_success(entry):
            recent_success.append(brief)
        else:
            recent_failure.append(brief)
        if len(recent) >= limit:
            break
    return {
        "schema_version": "field_atomic_planner_input_v2",
        "goal": goal,
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "skill_namespace": skill_namespace,
        "field_atomic_memory_count": priors["field_atomic_entry_count"],
        "field_atomic_parameter_priors": priors,
        "galaxea_recovery_rules": recovery_rules,
        "recent_field_atomic_experiences": recent,
        "recent_field_atomic_successes": recent_success,
        "recent_field_atomic_failures": recent_failure,
        "usage": "Prefer successful parameter ranges and avoid repeated failed parameter patterns.",
    }


def build_galaxea_recovery_rules(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    limit: int = 12,
) -> dict[str, Any]:
    matched = [
        entry for entry in entries
        if is_field_atomic_entry(entry)
        and not field_atomic_success(entry)
        and (not skill_namespace or entry.skill_namespace == skill_namespace)
        and (not scenario_id or not entry.scenario_id or entry.scenario_id == scenario_id)
        and (not condition_id or not entry.condition_id or entry.condition_id == condition_id)
    ]
    rule_entries = [entry for entry in matched if is_field_atomic_failure_rule(entry)]
    generated = [rule for rule in (_rule_from_failure_entry(entry) for entry in matched if not is_field_atomic_failure_rule(entry)) if rule]
    rules = [_rule_payload_from_entry(entry) for entry in rule_entries]
    rules = [rule for rule in rules if rule]
    rules.extend(generated)
    merged = _merge_rule_payloads(rules)
    top_rules = sorted(
        merged,
        key=lambda item: (-int(item.get("support_count") or 0), str(item.get("rule_id") or "")),
    )[:limit]
    return {
        "schema_version": "galaxea_recovery_rules_v1",
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "rule_count": len(top_rules),
        "required_actions": _unique_flatten(rule.get("required_actions") for rule in top_rules),
        "must_relocalize_after": _unique_flatten(rule.get("must_relocalize_after") for rule in top_rules),
        "forbidden_patterns": _unique_flatten(rule.get("forbidden_patterns") for rule in top_rules),
        "suggested_parameter_region": _merge_suggested_regions(top_rules),
        "rules": top_rules,
        "usage": "Use as compact Galaxea recovery-rule context. These rules are prompt guidance and report evidence; candidate hard filtering is handled separately when enabled.",
    }


def update_galaxea_failure_rule_entries(
    entries: list[ExperienceEntry],
    *,
    scenario_id: str = "",
    condition_id: str = "",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
) -> dict[str, Any]:
    existing_by_id = {
        str(entry.retrieval_key.get("failure_rule_id") or entry.experience_id): entry
        for entry in entries
        if is_field_atomic_failure_rule(entry)
    }
    generated = [
        rule for rule in (
            _rule_from_failure_entry(entry)
            for entry in entries
            if is_field_atomic_entry(entry)
            and not is_field_atomic_failure_rule(entry)
            and not field_atomic_success(entry)
            and (not skill_namespace or entry.skill_namespace == skill_namespace)
            and (not scenario_id or not entry.scenario_id or entry.scenario_id == scenario_id)
            and (not condition_id or not entry.condition_id or entry.condition_id == condition_id)
        )
        if rule
    ]
    merged = _merge_rule_payloads([*_rule_payload_from_entries(existing_by_id.values()), *generated])
    created = 0
    updated = 0
    for rule in merged:
        rule_id = str(rule.get("rule_id") or "")
        if not rule_id:
            continue
        if rule_id in existing_by_id:
            _update_failure_rule_entry(existing_by_id[rule_id], rule)
            updated += 1
        else:
            entries.append(_make_failure_rule_entry(rule, skill_namespace=skill_namespace))
            created += 1
    return {
        "schema_version": "galaxea_failure_rule_update_report_v1",
        "generated_rule_count": len(generated),
        "merged_rule_count": len(merged),
        "created_count": created,
        "updated_count": updated,
        "rule_ids": [rule.get("rule_id") for rule in merged if rule.get("rule_id")],
    }


def _rule_payload_from_entries(entries: Any) -> list[dict[str, Any]]:
    return [rule for rule in (_rule_payload_from_entry(entry) for entry in entries) if rule]


def _rule_payload_from_entry(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    rule = feedback.get("galaxea_failure_rule")
    if isinstance(rule, dict):
        return dict(rule)
    rule = entry.metadata.get("galaxea_failure_rule") if isinstance(entry.metadata, dict) else {}
    return dict(rule) if isinstance(rule, dict) else {}


def _rule_from_failure_entry(entry: ExperienceEntry) -> dict[str, Any]:
    critic = _entry_llm_critic(entry)
    summary = critic.get("parameter_failure_summary") if isinstance(critic.get("parameter_failure_summary"), dict) else {}
    if not summary:
        return {}
    items = summary.get("items") if isinstance(summary.get("items"), list) else []
    required_actions = _sanitize_actions(summary.get("required_actions"))
    must_relocalize_after = _sanitize_actions(summary.get("must_relocalize_after"))
    forbidden_parameters = summary.get("forbidden_parameters") if isinstance(summary.get("forbidden_parameters"), dict) else {}
    suggested = summary.get("suggested_parameter_region") if isinstance(summary.get("suggested_parameter_region"), dict) else {}
    derived = _derive_rule_guidance_from_items(items)
    if not required_actions:
        required_actions = derived.get("required_actions", [])
    if not must_relocalize_after:
        must_relocalize_after = derived.get("must_relocalize_after", [])
    if not forbidden_parameters:
        forbidden_parameters = derived.get("forbidden_parameters", {})
    if not suggested:
        suggested = derived.get("suggested_parameter_region", {})
    forbidden_patterns = _forbidden_patterns_from_summary(summary)
    if not required_actions and not forbidden_patterns and not suggested:
        return {}
    failure_stage = str(critic.get("failure_stage") or entry.failure_taxonomy.get("failure_stage") or "")
    failure_type = str(critic.get("failure_type") or entry.failure_taxonomy.get("failure_type") or "")
    rule_id = _failure_rule_id(entry.scenario_id, failure_stage, failure_type, forbidden_patterns, required_actions)
    pregrasp_error_vector_world = _single_evidence_value(entry, "final_tcp_minus_pregrasp_world")
    return {
        "schema_version": "galaxea_failure_rule_v1",
        "rule_id": rule_id,
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "trigger": {
            "failure_stage": failure_stage,
            "failure_type": failure_type,
            "action": field_atomic_action(entry),
            "final_error_bucket": _error_bucket(_single_evidence_value(entry, "final_error") or entry.failure_taxonomy.get("final_error")),
            "pregrasp_error_vector_world": pregrasp_error_vector_world,
            "pregrasp_error_hint": _pregrasp_error_hint(pregrasp_error_vector_world),
        },
        "required_actions": required_actions,
        "must_relocalize_after": must_relocalize_after,
        "forbidden_patterns": forbidden_patterns,
        "forbidden_parameters": forbidden_parameters,
        "suggested_parameter_region": suggested,
        "support_count": 1,
        "evidence_ids": [entry.experience_id],
        "memory_lesson": str(critic.get("memory_lesson") or summary.get("overall_lesson") or "")[:260],
    }


def _derive_rule_guidance_from_items(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    required_actions: list[str] = []
    must_relocalize_after: list[str] = []
    forbidden: dict[str, dict[str, Any]] = {}
    suggested: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        action = canonical_skill_action(str(item.get("action") or ""))
        expected = item.get("expected_direction") if isinstance(item.get("expected_direction"), dict) else {}
        bad_values = item.get("bad_values") if isinstance(item.get("bad_values"), dict) else {}
        if action == "move_base_relative" and ("increase" in {str(v) for v in expected.values()}):
            for name in ("move_base_relative", "set_torso_posture", "head_camera_rgbd_save", "head_camera_grounded_sam2_pose"):
                if name not in required_actions:
                    required_actions.append(name)
            must_relocalize_after.append("move_base_relative")
            limits: dict[str, Any] = {}
            region: dict[str, Any] = {}
            for axis, default_low in (("x", 0.3), ("y", 0.2)):
                raw = bad_values.get(axis)
                try:
                    numeric = float(raw)
                except (TypeError, ValueError):
                    numeric = None
                if numeric is not None:
                    limits[f"{axis}_max"] = round(numeric, 6)
                    region[axis] = [round(min(max(numeric + 0.1, default_low), 0.4), 6), 0.4]
                else:
                    region[axis] = [default_low, 0.4]
            forbidden["move_base_relative"] = limits
            suggested["move_base_relative"] = region
            suggested.setdefault("set_torso_posture", {"level": "high"})
        elif action == "set_torso_posture":
            if "set_torso_posture" not in required_actions:
                required_actions.append("set_torso_posture")
            must_relocalize_after.append("set_torso_posture")
            suggested.setdefault("set_torso_posture", {"level": "high"})
    return {
        "required_actions": list(dict.fromkeys(required_actions)),
        "must_relocalize_after": list(dict.fromkeys(must_relocalize_after)),
        "forbidden_parameters": forbidden,
        "suggested_parameter_region": suggested,
    }


def _failure_rule_id(
    scenario_id: str,
    failure_stage: str,
    failure_type: str,
    forbidden_patterns: list[str],
    required_actions: list[str],
) -> str:
    key = "|".join([
        str(scenario_id or "unknown"),
        str(failure_stage or "unknown"),
        str(failure_type or "unknown"),
        ",".join(sorted(forbidden_patterns)),
        ",".join(sorted(required_actions)),
    ])
    import hashlib
    return "galaxea_rule_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _forbidden_patterns_from_summary(summary: dict[str, Any]) -> list[str]:
    patterns: list[str] = []
    forbidden = summary.get("forbidden_parameters") if isinstance(summary.get("forbidden_parameters"), dict) else {}
    base = forbidden.get("move_base_relative") if isinstance(forbidden.get("move_base_relative"), dict) else {}
    if base:
        x_max = base.get("x_max")
        y_max = base.get("y_max")
        parts = ["move_base_relative"]
        if x_max is not None:
            parts.append(f"x<={x_max}")
        if y_max is not None:
            parts.append(f"y<={y_max}")
        patterns.append(" ".join(parts))
    items = summary.get("items") if isinstance(summary.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict) or item.get("action") != "move_base_relative":
            continue
        bad_values = item.get("bad_values") if isinstance(item.get("bad_values"), dict) else {}
        expected = item.get("expected_direction") if isinstance(item.get("expected_direction"), dict) else {}
        if expected.get("x") == "increase" or expected.get("y") == "increase":
            parts = ["move_base_relative"]
            if bad_values.get("x") is not None:
                parts.append(f"x<={bad_values.get('x')}")
            if bad_values.get("y") is not None:
                parts.append(f"y<={bad_values.get('y')}")
            patterns.append(" ".join(parts))
    return list(dict.fromkeys(patterns))[:8]


def _merge_rule_payloads(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rule in rules:
        rule_id = str(rule.get("rule_id") or "")
        if not rule_id:
            continue
        if rule_id not in by_id:
            by_id[rule_id] = dict(rule)
            continue
        target = by_id[rule_id]
        target["support_count"] = int(target.get("support_count") or 0) + int(rule.get("support_count") or 1)
        target["evidence_ids"] = list(dict.fromkeys([*(target.get("evidence_ids") or []), *(rule.get("evidence_ids") or [])]))[:24]
        for key in ("required_actions", "must_relocalize_after", "forbidden_patterns"):
            target[key] = list(dict.fromkeys([*(target.get(key) or []), *(rule.get(key) or [])]))
        target["suggested_parameter_region"] = _merge_suggested_region_dict(
            target.get("suggested_parameter_region"),
            rule.get("suggested_parameter_region"),
        )
        target["forbidden_parameters"] = _merge_dict_dict(target.get("forbidden_parameters"), rule.get("forbidden_parameters"))
    return list(by_id.values())


def _merge_dict_dict(left: Any, right: Any) -> dict[str, Any]:
    result = dict(left) if isinstance(left, dict) else {}
    if not isinstance(right, dict):
        return result
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def _merge_suggested_regions(rules: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for rule in rules:
        merged = _merge_suggested_region_dict(merged, rule.get("suggested_parameter_region"))
    return merged


def _merge_suggested_region_dict(left: Any, right: Any) -> dict[str, Any]:
    result = dict(left) if isinstance(left, dict) else {}
    if not isinstance(right, dict):
        return result
    for action, value in right.items():
        if not isinstance(value, dict):
            result[action] = value
            continue
        current = result.get(action)
        if not isinstance(current, dict):
            result[action] = dict(value)
            continue
        if action == "move_base_relative":
            result[action] = _merge_base_suggested_region(current, value)
        elif action == "set_torso_posture":
            result[action] = _merge_torso_suggested_region(current, value)
        else:
            result[action] = {**current, **value}
    return result


def _merge_base_suggested_region(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for axis in ("x", "y"):
        bounds = _merge_numeric_bounds(left.get(axis), right.get(axis), prefer_higher=True)
        if bounds:
            merged[axis] = bounds
    for key, value in right.items():
        if key not in {"x", "y"}:
            merged[key] = value
    return merged


def _merge_torso_suggested_region(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    levels = _as_list(left.get("level")) + _as_list(right.get("level"))
    if any(str(level).lower() == "high" for level in levels):
        return {**left, **right, "level": "high"}
    if levels:
        return {**left, **right, "level": list(dict.fromkeys(levels))}
    return {**left, **right}


def _merge_numeric_bounds(left: Any, right: Any, *, prefer_higher: bool) -> list[float] | None:
    left_bounds = _numeric_bounds(left)
    right_bounds = _numeric_bounds(right)
    if not left_bounds:
        return right_bounds
    if not right_bounds:
        return left_bounds
    if prefer_higher:
        return [max(left_bounds[0], right_bounds[0]), max(left_bounds[1], right_bounds[1])]
    return [min(left_bounds[0], right_bounds[0]), min(left_bounds[1], right_bounds[1])]


def _numeric_bounds(value: Any) -> list[float] | None:
    if isinstance(value, (int, float, str)):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return [numeric, numeric]
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError):
        return None
    return [min(low, high), max(low, high)]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _unique_flatten(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or []:
        if isinstance(value, list):
            for item in value:
                text = str(item)
                if text and text not in result:
                    result.append(text)
        elif value not in (None, ""):
            text = str(value)
            if text not in result:
                result.append(text)
    return result[:24]


def _sanitize_actions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [canonical_skill_action(str(item)) for item in value if canonical_skill_action(str(item))]


def _make_failure_rule_entry(rule: dict[str, Any], *, skill_namespace: str) -> ExperienceEntry:
    entry = ExperienceEntry(
        source="simulation",
        backend="mujoco",
        validation_status="simulation_failed",
        skill_namespace=skill_namespace,
        scenario={"scenario_id": rule.get("scenario_id", "")},
        condition={"condition_id": rule.get("condition_id", "")},
        task={"name": "galaxea_failure_rule"},
        result={"success": False, "rule_support_count": rule.get("support_count", 0)},
        execution_feedback={"galaxea_failure_rule": rule},
        retrieval_key={
            "failure_rule_id": rule.get("rule_id", ""),
            "failure_stage": rule.get("failure_stage", ""),
            "failure_type": rule.get("failure_type", ""),
            "scenario_id": rule.get("scenario_id", ""),
        },
        memory_tags={
            "memory_type": "field_atomic_experience",
            "memory_role": "field_atomic_failure_rule",
            "memory_gate_decision": "merge_repeated_failures",
        },
        failure_taxonomy={
            "failure_stage": rule.get("failure_stage", ""),
            "failure_type": rule.get("failure_type", ""),
            "failure_rule": True,
        },
        metadata={"field_atomic": True, "galaxea_failure_rule": rule},
    )
    entry.memory_gate = MemoryGate(
        anomaly_score=1.0,
        failure_score=1.0,
        recovery_utility_score=0.8,
        surprise_score=0.2,
        write_score=0.75,
        write_decision="write_failure_rule",
        trigger_events=["repeated_failure_pattern"],
        explanation={"rule_id": rule.get("rule_id", ""), "support_count": rule.get("support_count", 0)},
    )
    return entry


def _update_failure_rule_entry(entry: ExperienceEntry, rule: dict[str, Any]) -> None:
    entry.execution_feedback["galaxea_failure_rule"] = rule
    entry.metadata["galaxea_failure_rule"] = rule
    entry.result["rule_support_count"] = rule.get("support_count", 0)
    entry.updated_at = utc_now()
    entry.memory_gate.write_decision = "merge_into_failure_rule"
    entry.memory_gate.recovery_utility_score = max(float(entry.memory_gate.recovery_utility_score or 0.0), 0.8)
    entry.memory_gate.explanation = {"rule_id": rule.get("rule_id", ""), "support_count": rule.get("support_count", 0)}


def _summarize_parameter_entries(entries: list[ExperienceEntry]) -> dict[str, Any]:
    values: dict[str, list[Any]] = defaultdict(list)
    evidence: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        params = field_atomic_parameters(entry)
        for key, value in params.items():
            values[str(key)].append(value)
            evidence[str(key)].append(entry.experience_id)
    return {
        key: {
            **_summarize_values(items),
            "evidence_ids": evidence[key][:6],
        }
        for key, items in sorted(values.items())
    }


def _summarize_parameter_failures(entries: list[ExperienceEntry]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[ExperienceEntry, dict[str, Any]]]] = defaultdict(list)
    for entry in entries:
        for summary in _parameter_failure_items(entry):
            action = str(summary.get("action") or field_atomic_action(entry) or "")
            if not action:
                continue
            grouped[action].append((entry, summary))
    result: dict[str, Any] = {}
    for action, items in grouped.items():
        summaries = [summary for _, summary in items]
        result[action] = {
            "entry_count": len(items),
            "bad_keys": _count_values([item.get("bad_keys") for item in summaries]),
            "parameter_lesson": [item.get("parameter_lesson") for item in summaries if item.get("parameter_lesson")][:5],
            "evidence_ids": list(dict.fromkeys(entry.experience_id for entry, _ in items))[:6],
        }
    return result


def _count_values(items: list[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        if isinstance(item, list):
            for value in item:
                if value not in (None, ""):
                    counter[str(value)] += 1
        elif item not in (None, ""):
            counter[str(item)] += 1
    return dict(counter)


def _summarize_values(items: list[Any]) -> dict[str, Any]:
    nums = [float(item) for item in items if isinstance(item, (int, float)) and not isinstance(item, bool)]
    if nums:
        return {
            "count": len(nums),
            "median": round(float(median(nums)), 6),
            "min": round(float(min(nums)), 6),
            "max": round(float(max(nums)), 6),
        }
    lists = [item for item in items if isinstance(item, list) and item]
    if lists:
        dims = min(len(item) for item in lists)
        return {
            "count": len(lists),
            "median": [round(float(median([float(item[index]) for item in lists if isinstance(item[index], (int, float))])), 6) for index in range(dims)],
        }
    texts = [str(item) for item in items]
    counts = Counter(texts)
    return {
        "count": len(texts),
        "top_values": [{"value": value, "count": count} for value, count in counts.most_common(5)],
    }
