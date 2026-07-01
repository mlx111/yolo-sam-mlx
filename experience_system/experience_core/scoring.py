"""Risk-aware scoring for universal experience retrieval and candidate plans."""

from __future__ import annotations

from typing import Any

from .failure_taxonomy import is_actionable_failure_type
from .policy_calibration import find_policy_group, policy_weight
from .schema import ExperienceEntry


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _success(entry: ExperienceEntry) -> bool:
    return bool(entry.result.get("success", entry.result.get("task_success", False)))


def _actions_from_candidate(candidate_steps: list[dict[str, Any]] | list[str]) -> list[str]:
    actions: list[str] = []
    for step in candidate_steps:
        if isinstance(step, str):
            actions.append(step)
        elif isinstance(step, dict):
            name = str(step.get("skill") or step.get("name") or step.get("action") or "")
            if name:
                actions.append(name)
    return actions


def _actions_from_entry(entry: ExperienceEntry) -> list[str]:
    return [item.name for item in entry.skill_sequence if item.name]


def action_lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a == b else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(left), len(right))


def action_set_overlap(left: list[str], right: list[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _index(actions: list[str], name: str) -> int:
    try:
        return actions.index(name)
    except ValueError:
        return -1


def _before(actions: list[str], first: str, second: str) -> bool:
    first_index = _index(actions, first)
    second_index = _index(actions, second)
    return first_index >= 0 and second_index >= 0 and first_index < second_index


def _count(actions: list[str], name: str) -> int:
    return sum(1 for action in actions if action == name)


def _between(actions: list[str], name: str, after: str, before: str) -> bool:
    after_index = _index(actions, after)
    before_index = _index(actions, before)
    if after_index < 0 or before_index < 0 or after_index >= before_index:
        return False
    return any(action == name for action in actions[after_index + 1 : before_index])


def mitigation_profile(candidate_actions: list[str], entry: ExperienceEntry, query_context: dict[str, Any] | None) -> dict[str, Any]:
    """Estimate whether candidate actions mitigate a retrieved failure memory."""

    failure_type = str(entry.failure_taxonomy.get("failure_type") or entry.failure_taxonomy.get("standard_failure_type") or "")
    outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
    condition_id = entry.condition_id
    query_condition = _query_context_value(query_context or {}, "condition_id")
    condition_match = not query_condition or condition_id == query_condition
    if not failure_type and outcome_type == "sim_success_real_fail":
        failure_type = condition_id

    mitigations: list[str] = []
    risk_scale = 1.0
    support_bonus = 0.0
    if not condition_match:
        return {
            "failure_type": failure_type,
            "mitigations": mitigations,
            "risk_scale": round(risk_scale, 4),
            "support_bonus": round(support_bonus, 4),
            "condition_match": condition_match,
        }

    if failure_type in {"grasp_miss", "grasp_slip", "object_not_lifted"}:
        if _before(candidate_actions, "verify_grasp", "left_vertical_lift") or _before(candidate_actions, "verify_grasp", "dual_arm_synchronized_lift"):
            mitigations.append("verify_grasp_before_lift")
            risk_scale = min(risk_scale, 0.55)
            support_bonus = max(support_bonus, 0.12)

    if failure_type == "place_occupied" or condition_id == "place_occupied":
        detects_early_single = _before(candidate_actions, "detect_place_occupancy", "move_to_pregrasp")
        detects_early_dual = _before(candidate_actions, "detect_place_occupancy", "dual_arm_pregrasp")
        chooses_early_single = _before(candidate_actions, "choose_alternate_place", "move_to_pregrasp")
        chooses_early_dual = _before(candidate_actions, "choose_alternate_place", "dual_arm_pregrasp")
        if (detects_early_single and chooses_early_single) or (detects_early_dual and chooses_early_dual):
            mitigations.append("choose_place_before_carry")
            risk_scale = min(risk_scale, 0.45)
            support_bonus = max(support_bonus, 0.15)
        elif "detect_place_occupancy" in candidate_actions and "choose_alternate_place" in candidate_actions:
            mitigations.append("choose_place_late")
            risk_scale = min(risk_scale, 0.85)
            support_bonus = max(support_bonus, 0.04)

    if failure_type == "transport_collision" or condition_id == "transport_collision":
        if _before(candidate_actions, "safe_transport_pose", "segmented_transport"):
            mitigations.append("safe_transport_pose_before_transport")
            risk_scale = min(risk_scale, 0.50)
            support_bonus = max(support_bonus, 0.14)

    if failure_type == "dual_arm_mismatch" or condition_id == "dual_arm_mismatch":
        if _count(candidate_actions, "dual_arm_level_object") >= 2 and _between(candidate_actions, "dual_arm_level_object", "base_move_to_place", "dual_arm_place"):
            mitigations.append("relevel_before_place")
            risk_scale = min(risk_scale, 0.45)
            support_bonus = max(support_bonus, 0.14)
        elif "dual_arm_level_object" in candidate_actions:
            mitigations.append("dual_arm_level_object")
            risk_scale = min(risk_scale, 0.90)
            support_bonus = max(support_bonus, 0.02)

    return {
        "failure_type": failure_type,
        "mitigations": mitigations,
        "risk_scale": round(risk_scale, 4),
        "support_bonus": round(support_bonus, 4),
        "condition_match": condition_match,
    }


def _query_context_value(query_context: dict[str, Any], key: str) -> str:
    return str(query_context.get(key) or "")


def context_similarity(entry: ExperienceEntry, query_context: dict[str, Any] | None) -> dict[str, Any]:
    query_context = query_context or {}
    scenario_match = not _query_context_value(query_context, "scenario_id") or entry.scenario_id == _query_context_value(query_context, "scenario_id")
    condition_query = _query_context_value(query_context, "condition_id")
    condition_match = not condition_query or entry.condition_id == condition_query
    task_stage_query = _query_context_value(query_context, "task_stage")
    task_stage = str(entry.task.get("stage") or "")
    task_stage_match = not task_stage_query or task_stage == task_stage_query
    object_class_query = _query_context_value(query_context, "object_class")
    object_class_match = not object_class_query or entry.object_state.object_class == object_class_query
    score = 0.0
    score += 0.35 if scenario_match else 0.0
    score += 0.35 if condition_match else 0.0
    score += 0.15 if task_stage_match else 0.0
    score += 0.15 if object_class_match else 0.0
    return {
        "context_score": round(score, 4),
        "scenario_match": scenario_match,
        "condition_match": condition_match,
        "task_stage_match": task_stage_match,
        "object_class_match": object_class_match,
    }


def risk_transfer_weight(
    entry: ExperienceEntry,
    *,
    action_overlap: float,
    action_jaccard: float,
    query_context: dict[str, Any] | None,
    policy_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context_similarity(entry, query_context)
    group = find_policy_group(policy_calibration, entry)
    outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
    failure_type = str(entry.failure_taxonomy.get("failure_type") or "")
    has_gap_risk = bool(outcome_type)
    has_failure_risk = is_actionable_failure_type(failure_type) or not _success(entry)
    local_action_overlap = max(0.65 * action_overlap + 0.35 * action_jaccard, 0.0)
    context_score = float(context["context_score"])
    base = local_action_overlap * context_score
    if has_gap_risk and not context["condition_match"]:
        base *= policy_weight(policy_calibration, group, "gap_condition_mismatch_scale", 0.30)
    elif has_failure_risk and not context["condition_match"]:
        base *= policy_weight(policy_calibration, group, "failure_condition_mismatch_scale", 0.50)
    if has_gap_risk and not context["task_stage_match"]:
        base *= policy_weight(policy_calibration, group, "gap_task_stage_mismatch_scale", 0.60)
    if has_gap_risk and not context["object_class_match"]:
        base *= policy_weight(policy_calibration, group, "gap_object_class_mismatch_scale", 0.50)
    return {
        **context,
        "action_jaccard": round(action_jaccard, 4),
        "policy_calibration_group": {
            "scenario_id": group.get("scenario_id", ""),
            "condition_id": group.get("condition_id", ""),
            "entry_count": group.get("entry_count", 0),
            "evidence_confidence": group.get("evidence_confidence", 0.0),
        } if group else {},
        "risk_transfer_weight": round(_clamp01(base), 4),
    }


def estimate_gap_uncertainty(entry: ExperienceEntry) -> float:
    outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
    uncertainty = _clamp01(entry.sim_real_gap.uncertainty)
    gap_score = _clamp01(entry.sim_real_gap.gap_score)
    if outcome_type == "sim_success_real_fail":
        uncertainty = max(uncertainty, 0.85)
    elif outcome_type == "matched_success" and gap_score <= 0.2:
        uncertainty = min(max(uncertainty, 0.1), 0.25)
    return _clamp01(max(uncertainty, 0.5 * gap_score))


def critic_risk_score(entry: ExperienceEntry) -> float:
    risk = _clamp01(entry.critic_result.critic_risk_score)
    status = str(entry.critic_result.overall_status or "").lower()
    if status == "block":
        risk = max(risk, 0.9)
    elif status in {"warn", "warning"}:
        risk = max(risk, 0.55)
    elif status == "pass":
        risk = min(risk, 0.2)
    if entry.critic_result.rule_flags:
        risk = max(risk, min(len(entry.critic_result.rule_flags) * 0.15, 0.75))
    return _clamp01(risk)


def real_validation_bonus(entry: ExperienceEntry) -> float:
    if entry.source == "real":
        return 1.0
    if entry.source == "pseudo_real":
        return 0.65
    if entry.validation_status == "real_validated":
        return 1.0
    if entry.validation_status == "real_executed":
        return 0.8
    return 0.0


def sim_real_failure_penalty(entry: ExperienceEntry) -> float:
    outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
    if outcome_type == "sim_success_real_fail":
        return 0.45
    return 0.0


def entry_risk_adjustment(entry: ExperienceEntry) -> dict[str, Any]:
    success = _success(entry)
    gap_uncertainty = estimate_gap_uncertainty(entry)
    critic_risk = critic_risk_score(entry)
    real_bonus = real_validation_bonus(entry)
    sensor_bonus = sensor_evidence_bonus(entry)
    failure_penalty = 0.25 if not success else 0.0
    sim_real_penalty = sim_real_failure_penalty(entry)
    gap_score_penalty = 0.25 * _clamp01(entry.sim_real_gap.gap_score)
    gap_uncertainty_penalty = 0.18 * gap_uncertainty
    critic_penalty = 0.18 * critic_risk
    real_success_bonus = 0.20 * real_bonus if success else 0.0
    paired_bonus = 0.05 if entry.sim_real_pair.get("validation_status") == "paired" else 0.0
    risk_penalty = failure_penalty + sim_real_penalty + gap_score_penalty + gap_uncertainty_penalty + critic_penalty
    trust_bonus = real_success_bonus + paired_bonus + sensor_bonus
    return {
        "gap_uncertainty": round(gap_uncertainty, 4),
        "critic_risk": round(critic_risk, 4),
        "real_validation_bonus": round(real_bonus, 4),
        "sensor_evidence_bonus": round(sensor_bonus, 4),
        "real_success_bonus": round(real_success_bonus, 4),
        "paired_bonus": round(paired_bonus, 4),
        "failure_penalty": round(failure_penalty, 4),
        "sim_real_failure_penalty": round(sim_real_penalty, 4),
        "gap_score_penalty": round(gap_score_penalty, 4),
        "gap_uncertainty_penalty": round(gap_uncertainty_penalty, 4),
        "critic_penalty": round(critic_penalty, 4),
        "risk_penalty": round(risk_penalty, 4),
        "trust_bonus": round(trust_bonus, 4),
        "score_adjustment": round(trust_bonus - risk_penalty, 4),
    }


def sensor_evidence_bonus(entry: ExperienceEntry) -> float:
    modalities = {str(item) for item in entry.sensor_evidence.modalities or [] if item}
    if not modalities:
        return 0.0
    bonus = 0.0
    if "rgbd" in modalities or "rgb" in modalities:
        bonus += 0.03
    if "lidar" in modalities:
        bonus += 0.02
    if "wrist_force" in modalities:
        bonus += 0.03
    if entry.source in {"real", "pseudo_real"}:
        bonus += 0.02
    if entry.sensor_evidence.evidence_refs:
        bonus += 0.01
    return min(bonus, 0.10)


def estimate_real_success_prior(entries: list[tuple[ExperienceEntry, float]]) -> dict[str, Any]:
    total = 0.0
    success = 0.0
    evidence: list[dict[str, Any]] = []
    for entry, retrieval_score in entries:
        real_weight = real_validation_bonus(entry)
        if real_weight <= 0.0:
            continue
        weight = real_weight * max(float(retrieval_score), 0.0)
        total += weight
        ok = _success(entry)
        success += weight * float(ok)
        evidence.append({
            "experience_id": entry.experience_id,
            "source": entry.source,
            "success": ok,
            "weight": round(weight, 4),
        })
    prior = success / total if total > 0.0 else 0.5
    return {
        "real_success_prior": round(prior, 4),
        "real_evidence_count": len(evidence),
        "evidence": evidence,
    }


def top_failure_risk_memory(
    candidate_actions: list[str],
    retrieved_experiences: list[tuple[ExperienceEntry, float]],
    *,
    query_context: dict[str, Any] | None = None,
    policy_calibration: dict[str, Any] | None = None,
    top_o: int = 3,
) -> dict[str, Any]:
    """Rank critical failure memories as risk priors for a candidate plan."""

    ranked: list[dict[str, Any]] = []
    for entry, retrieval_score in retrieved_experiences:
        entry_actions = _actions_from_entry(entry)
        lcs_overlap = action_lcs_ratio(candidate_actions, entry_actions)
        failed_action_overlap = action_set_overlap(candidate_actions, entry_actions)
        transfer = risk_transfer_weight(
            entry,
            action_overlap=lcs_overlap if lcs_overlap > 0.0 else 0.05,
            action_jaccard=failed_action_overlap,
            query_context=query_context,
            policy_calibration=policy_calibration,
        )
        context_score = float(transfer.get("context_score", 0.0))
        retrieval = _clamp01(retrieval_score)
        critic_risk = critic_risk_score(entry)
        gap_uncertainty = estimate_gap_uncertainty(entry)
        failure_type = str(entry.failure_taxonomy.get("failure_type") or entry.failure_taxonomy.get("standard_failure_type") or "")
        outcome_type = str(entry.sim_real_gap.outcome_gap.get("type") or "")
        is_failure = not _success(entry)
        is_critical = is_failure or outcome_type == "sim_success_real_fail" or critic_risk >= 0.55
        if not is_critical:
            continue

        mitigation = mitigation_profile(candidate_actions, entry, query_context)
        base_failure_risk = 0.35 * float(is_failure) + 0.30 * critic_risk + 0.25 * gap_uncertainty + 0.10 * sim_real_failure_penalty(entry)
        failure_similarity = _clamp01(0.40 * failed_action_overlap + 0.25 * lcs_overlap + 0.25 * context_score + 0.10 * retrieval)
        terminal_risk = _clamp01(base_failure_risk * failure_similarity * float(mitigation["risk_scale"]))
        ranked.append({
            "experience_id": entry.experience_id,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "source": entry.source,
            "success": _success(entry),
            "failure_type": failure_type or ("sim_success_real_fail" if outcome_type == "sim_success_real_fail" else "unknown_failure"),
            "critic_status": entry.critic_result.overall_status,
            "critic_risk": round(critic_risk, 4),
            "sim_real_gap_type": outcome_type,
            "retrieval_score": round(float(retrieval_score), 4),
            "failed_action_overlap": round(failed_action_overlap, 4),
            "action_lcs_overlap": round(lcs_overlap, 4),
            "failure_similarity": round(failure_similarity, 4),
            "terminal_risk_score": round(terminal_risk, 4),
            "mitigation": mitigation,
        })

    top = sorted(ranked, key=lambda item: float(item["terminal_risk_score"]), reverse=True)[: max(0, top_o)]
    if not top:
        return {
            "top_failure_risks": [],
            "terminal_risk_score": 0.0,
            "failure_risk_penalty": 0.0,
        }
    terminal_risk_score = max(float(item["terminal_risk_score"]) for item in top)
    mean_top_risk = sum(float(item["terminal_risk_score"]) for item in top) / len(top)
    failure_risk_penalty = _clamp01(0.65 * terminal_risk_score + 0.35 * mean_top_risk)
    return {
        "top_failure_risks": top,
        "terminal_risk_score": round(terminal_risk_score, 4),
        "failure_risk_penalty": round(failure_risk_penalty, 4),
    }


def score_candidate_plan(
    candidate_steps: list[dict[str, Any]] | list[str],
    retrieved_experiences: list[tuple[ExperienceEntry, float]],
    query_context: dict[str, Any] | None = None,
    policy_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_actions = _actions_from_candidate(candidate_steps)
    positive_support = 0.0
    failure_overlap_risk = 0.0
    max_gap_uncertainty = 0.0
    max_critic_risk = 0.0
    mitigation_support = 0.0
    evidence: list[dict[str, Any]] = []

    for entry, retrieval_score in retrieved_experiences:
        entry_actions = _actions_from_entry(entry)
        overlap = action_lcs_ratio(candidate_actions, entry_actions)
        if overlap <= 0.0:
            overlap = 0.05
        jaccard = action_set_overlap(candidate_actions, entry_actions)
        transfer = risk_transfer_weight(
            entry,
            action_overlap=overlap,
            action_jaccard=jaccard,
            query_context=query_context,
            policy_calibration=policy_calibration,
        )
        min_nonzero = policy_weight(policy_calibration, {}, "min_nonzero_risk_transfer", 0.05)
        risk_weight = max(float(transfer["risk_transfer_weight"]), min_nonzero if overlap > 0.0 else 0.0)
        weight = max(float(retrieval_score), 0.0) * overlap
        adjustment = entry_risk_adjustment(entry)
        mitigation = mitigation_profile(candidate_actions, entry, query_context)
        is_risky_memory = (not _success(entry)) or str(entry.sim_real_gap.outcome_gap.get("type") or "") == "sim_success_real_fail"
        if is_risky_memory and mitigation["mitigations"]:
            risk_weight *= float(mitigation["risk_scale"])
            mitigation_support += max(float(retrieval_score), 0.0) * float(mitigation["support_bonus"])
        if float(adjustment["score_adjustment"]) < 0.0:
            adjusted_weight = max(0.0, weight + float(adjustment["score_adjustment"]) * risk_weight)
        else:
            adjusted_weight = max(0.0, weight + float(adjustment["score_adjustment"]))
        if _success(entry):
            positive_support += adjusted_weight
        else:
            failure_overlap_risk = max(failure_overlap_risk, risk_weight)
        max_gap_uncertainty = max(max_gap_uncertainty, float(adjustment["gap_uncertainty"]) * risk_weight)
        max_critic_risk = max(max_critic_risk, float(adjustment["critic_risk"]) * max(risk_weight, 0.05))
        evidence.append({
            "experience_id": entry.experience_id,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "source": entry.source,
            "success": _success(entry),
            "retrieval_score": round(float(retrieval_score), 4),
            "action_overlap": round(overlap, 4),
            "risk_transfer": transfer,
            "mitigation": mitigation,
            "adjustment": adjustment,
        })

    real_prior = estimate_real_success_prior(retrieved_experiences)
    failure_memory = top_failure_risk_memory(
        candidate_actions,
        retrieved_experiences,
        query_context=query_context,
        policy_calibration=policy_calibration,
        top_o=3,
    )
    failure_risk_penalty = float(failure_memory["failure_risk_penalty"])
    support_score = _clamp01(positive_support + mitigation_support)
    risk_score = _clamp01(
        0.28 * failure_overlap_risk
        + 0.27 * max_gap_uncertainty
        + 0.25 * max_critic_risk
        + 0.20 * failure_risk_penalty
    )
    candidate_score = _clamp01(0.55 * support_score + 0.25 * real_prior["real_success_prior"] + 0.20 * (1.0 - risk_score))
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
        "terminal_risk_score": failure_memory["terminal_risk_score"],
        "failure_risk_penalty": failure_memory["failure_risk_penalty"],
        "top_failure_risks": failure_memory["top_failure_risks"],
        "gap_uncertainty": round(max_gap_uncertainty, 4),
        "critic_risk": round(max_critic_risk, 4),
        **real_prior,
        "evidence": evidence,
    }
