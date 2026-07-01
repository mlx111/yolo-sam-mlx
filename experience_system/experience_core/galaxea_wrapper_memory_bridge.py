"""Wrapper1-style memory utilities adapted to Galaxea field atomic skills.

This module intentionally mirrors the reusable parts of
``ur5e_mujoco/experiment_method_runner.py`` and
``experience_system.memory.scoring``:

- canonical action signatures
- LCS failed-plan blocker
- candidate plan score against success/failure/critic memories
- memory usefulness accounting

It does not import wrapper1 directly because wrapper1 depends on UR5e action
registries. The logic here keeps the same data flow but reads Galaxea
``ExperienceEntry`` objects and field-atomic action names.
"""

from __future__ import annotations

import json
from typing import Any

from .field_atomic_memory import field_atomic_parameters, field_atomic_success
from .schema import ExperienceEntry, canonical_skill_action
from .scoring import critic_risk_score, estimate_gap_uncertainty, estimate_real_success_prior, real_validation_bonus


def signature_actions_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = canonical_skill_action(str(step.get("action") or step.get("name") or step.get("skill") or ""))
        if action:
            actions.append(action)
    return actions


def canonical_action_signature_from_steps(steps: list[dict[str, Any]]) -> str:
    payload = [{"action": action} for action in signature_actions_from_steps(steps)]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def signature_actions_from_entry(entry: ExperienceEntry) -> list[str]:
    actions = [canonical_skill_action(item.name) for item in entry.skill_sequence if item.name]
    if actions:
        return actions
    if isinstance(entry.action_trace, list):
        return signature_actions_from_steps(entry.action_trace)
    key = entry.retrieval_key if isinstance(entry.retrieval_key, dict) else {}
    plan_signature = str(key.get("plan_signature") or "")
    if plan_signature:
        return [canonical_skill_action(item) for item in plan_signature.split("->") if item]
    return []


def canonical_action_signature_from_entry(entry: ExperienceEntry) -> str:
    return json.dumps([{"action": action} for action in signature_actions_from_entry(entry)], ensure_ascii=False, sort_keys=True)


def action_lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a == b else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(left), len(right))


def action_sequence_similarity_wrapper1_style(left: ExperienceEntry, right: ExperienceEntry) -> float:
    """Wrapper1 Dice similarity on ordered action bigrams."""
    seq1 = signature_actions_from_entry(left)
    seq2 = signature_actions_from_entry(right)
    if not seq1 or not seq2:
        return 0.0
    bigrams1 = set(zip(seq1[:-1], seq1[1:])) if len(seq1) > 1 else {(seq1[0],)}
    bigrams2 = set(zip(seq2[:-1], seq2[1:])) if len(seq2) > 1 else {(seq2[0],)}
    if not bigrams1 or not bigrams2:
        return 0.0
    return 2.0 * len(bigrams1 & bigrams2) / (len(bigrams1) + len(bigrams2))


def critic_prefilter_wrapper1_style(
    matches: list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]],
) -> list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]]:
    """Wrapper1 retrieval prefilter: keep failure diversity before MMR."""
    if not matches:
        return []
    from collections import defaultdict

    def _dedup_key(entry: ExperienceEntry) -> str:
        ft = entry.failure_taxonomy if isinstance(entry.failure_taxonomy, dict) else {}
        result = entry.result if isinstance(entry.result, dict) else {}
        return str(ft.get("cluster_id") or ft.get("failure_type") or result.get("failure_reason") or "unknown")

    by_type: dict[str, list[Any]] = defaultdict(list)
    for item in matches:
        by_type[_dedup_key(item[0])].append(item)

    filtered: list[Any] = []
    removed_condition_ids: set[str] = set()
    for group in by_type.values():
        group.sort(key=lambda item: str(getattr(item[0], "created_at", "")), reverse=True)
        if len(group) > 2:
            for item in group[2:]:
                if getattr(item[0], "condition_id", ""):
                    removed_condition_ids.add(str(item[0].condition_id))
        filtered.extend(group[:2])

    surviving_condition_ids = {str(getattr(item[0], "condition_id", "")) for item in filtered if getattr(item[0], "condition_id", "")}
    if len(surviving_condition_ids) <= 1 and (removed_condition_ids - surviving_condition_ids):
        for item in matches:
            condition_id = str(getattr(item[0], "condition_id", ""))
            if condition_id and condition_id not in surviving_condition_ids:
                filtered.append(item)
                break

    filtered.sort(key=lambda item: (-float(item[1] or 0.0), str(getattr(item[0], "created_at", ""))))
    return filtered


def mmr_select_wrapper1_style(
    matches: list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]],
    *,
    top_k: int,
    diversity_lambda: float,
) -> list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]]:
    """Wrapper1 greedy MMR selection adapted to Galaxea entries."""
    if not matches:
        return []
    selected: list[Any] = [matches[0]]
    pool = list(matches[1:])
    while len(selected) < max(0, int(top_k)) and pool:
        best_idx = -1
        best_score = -float("inf")
        for index, item in enumerate(pool):
            entry, score = item[0], float(item[1] or 0.0)
            max_sim = max(action_sequence_similarity_wrapper1_style(entry, selected_item[0]) for selected_item in selected)
            mmr = score - float(diversity_lambda) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = index
        if best_idx < 0:
            break
        selected.append(pool.pop(best_idx))
    return selected


def count_plan_quality_issues_wrapper1_style(
    steps: list[dict[str, Any]],
    *,
    allowed_actions: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Wrapper1 plan-quality counters using Galaxea atomic skill names."""
    allowed = {canonical_skill_action(str(item)) for item in (allowed_actions or []) if str(item)}
    invalid = 0
    unsafe = 0
    seen_pregrasp = False
    seen_approach = False
    seen_close = False
    seen_lift = False
    actions: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = canonical_skill_action(str(step.get("action") or ""))
        if not action:
            continue
        actions.append(action)
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if allowed and action not in allowed:
            invalid += 1
        if action == "approach_object" and not seen_pregrasp:
            unsafe += 1
        if action == "close_gripper" and not seen_approach:
            unsafe += 1
        if action == "lift":
            if not seen_close:
                unsafe += 1
            seen_lift = True
        if action in {"transport_object", "transport_to_detected_target"} and not seen_lift:
            unsafe += 1
        if action == "open_gripper" and not (seen_lift or "transport_object" in actions or "transport_to_detected_target" in actions):
            unsafe += 1
        if action == "set_gripper":
            state = params.get("state")
            if state not in {0, 1, "open", "close"}:
                invalid += 1
        if action == "move_to_pregrasp":
            seen_pregrasp = True
        if action == "approach_object":
            seen_approach = True
        if action == "close_gripper":
            seen_close = True
    return {
        "invalid_plan_count": invalid,
        "unsafe_gripper_action_count": unsafe,
        "candidate_actions": actions,
        "quality_status": "fail" if invalid else ("warn" if unsafe else "pass"),
    }


def repeated_failure_wrapper1_style(
    executed_steps: list[dict[str, Any]],
    retrieved_experiences: list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]],
) -> dict[str, Any]:
    """Wrapper1 repeated-failure marker."""
    signature = canonical_action_signature_from_steps(executed_steps)
    matches: list[dict[str, Any]] = []
    if not signature or signature == "[]":
        return {
            "executed_plan_signature": signature,
            "repeated_failure_detected": False,
            "repeated_failure_matches": matches,
        }
    for item in retrieved_experiences or []:
        if len(item) < 2:
            continue
        entry = item[0]
        if field_atomic_success(entry):
            continue
        if canonical_action_signature_from_entry(entry) != signature:
            continue
        matches.append({
            "experience_id": entry.experience_id,
            "partition": str(entry.memory_tags.get("memory_role") or ""),
            "score": float(item[1] or 0.0),
        })
    return {
        "executed_plan_signature": signature,
        "repeated_failure_detected": bool(matches),
        "repeated_failure_matches": matches,
    }


def score_candidate_plan_wrapper1_style(
    candidate_steps: list[dict[str, Any]],
    retrieved_experiences: list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]],
) -> dict[str, Any]:
    """Direct Galaxea equivalent of wrapper1 ``score_candidate_plan``."""
    candidate_actions = signature_actions_from_steps(candidate_steps)
    candidate_step_params = {
        canonical_skill_action(str(step.get("action") or step.get("name") or step.get("skill") or "")): field_atomic_parameters_from_step(step)
        for step in candidate_steps
        if isinstance(step, dict) and str(step.get("action") or step.get("name") or step.get("skill") or "")
    }
    positive_support = 0.0
    failure_overlap_risk = 0.0
    parameter_failure_risk = 0.0
    gap_uncertainty = 0.0
    critic_risk = 0.0
    real_success_support = 0.0
    evidence: list[dict[str, Any]] = []

    normalized: list[tuple[ExperienceEntry, float]] = []
    for item in retrieved_experiences or []:
        if len(item) >= 2:
            normalized.append((item[0], float(item[1] or 0.0)))

    for entry, score in normalized:
        entry_actions = signature_actions_from_entry(entry)
        overlap = action_lcs_ratio(candidate_actions, entry_actions)
        entry_success = field_atomic_success(entry)
        weight = max(float(score), 0.0) * max(overlap, 0.05)
        entry_gap = estimate_gap_uncertainty(entry)
        entry_critic = critic_risk_score(entry)
        entry_real = real_validation_bonus(entry)
        parameter_overlap = _parameter_failure_overlap(candidate_step_params, field_atomic_parameter_failure_summary(entry))

        if parameter_overlap > 0.0:
            failure_overlap_risk = max(failure_overlap_risk, max(overlap, parameter_overlap))
            parameter_failure_risk = max(parameter_failure_risk, parameter_overlap)
        if entry_success:
            positive_support += weight
            real_success_support += weight * entry_real
        else:
            failure_overlap_risk = max(failure_overlap_risk, overlap)
        gap_uncertainty = max(gap_uncertainty, entry_gap * overlap)
        critic_risk = max(critic_risk, entry_critic * max(overlap, 0.25))
        evidence.append({
            "experience_id": entry.experience_id,
            "success": entry_success,
            "score": float(score),
            "action_overlap": round(overlap, 4),
            "parameter_overlap": round(parameter_overlap, 4),
            "gap_uncertainty": round(entry_gap, 4),
            "critic_risk": round(entry_critic, 4),
            "real_validation_bonus": round(entry_real, 4),
        })

    support_score = _clamp01(positive_support / max(len(normalized), 1))
    real_success_prior = estimate_real_success_prior(normalized)
    final_score = _clamp01(
        0.50
        + 0.25 * support_score
        + 0.15 * float(real_success_prior.get("real_success_prior", 0.0))
        + 0.10 * _clamp01(real_success_support)
        - 0.25 * failure_overlap_risk
        - 0.18 * parameter_failure_risk
        - 0.20 * gap_uncertainty
        - 0.20 * critic_risk
    )
    if final_score >= 0.70:
        decision = "prefer"
    elif final_score >= 0.45:
        decision = "allow"
    elif final_score >= 0.25:
        decision = "rewrite_recommended"
    else:
        decision = "reject_recommended"
    return {
        "candidate_score": round(final_score, 4),
        "decision": decision,
        "support_score": round(support_score, 4),
        "real_success_prior": real_success_prior.get("real_success_prior", 0.0),
        "real_evidence_count": real_success_prior.get("real_evidence_count", 0),
        "failure_overlap_risk": round(failure_overlap_risk, 4),
        "parameter_failure_risk": round(parameter_failure_risk, 4),
        "gap_uncertainty": round(gap_uncertainty, 4),
        "critic_risk": round(critic_risk, 4),
        "candidate_actions": candidate_actions,
        "evidence": evidence,
    }


def field_atomic_parameter_failure_summary(entry: ExperienceEntry) -> dict[str, Any]:
    feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
    summary = feedback.get("parameter_failure_summary")
    if not isinstance(summary, dict):
        critic = feedback.get("llm_critic") if isinstance(feedback.get("llm_critic"), dict) else {}
        summary = critic.get("parameter_failure_summary") if isinstance(critic.get("parameter_failure_summary"), dict) else {}
    return summary if isinstance(summary, dict) else {}


def field_atomic_parameters_from_step(step: dict[str, Any]) -> dict[str, Any]:
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
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
    score = 0.0
    total = 0.0
    bad_keys = parameter_failure_summary.get("bad_keys")
    bad_values = parameter_failure_summary.get("bad_values")
    expected_range = parameter_failure_summary.get("expected_range")
    if isinstance(bad_keys, list) and bad_keys:
        total += 1.0
        score += sum(1 for key in bad_keys if key in candidate_params) / max(len(bad_keys), 1)
    if isinstance(bad_values, dict) and bad_values:
        total += 1.0
        score += sum(1 for key, value in bad_values.items() if str(candidate_params.get(key)) == str(value)) / max(len(bad_values), 1)
    if isinstance(expected_range, dict) and expected_range:
        total += 1.0
        checked = 0
        matched = 0
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


def failed_plan_blocker_matches_wrapper1_style(
    candidate_steps: list[dict[str, Any]],
    retrieved_experiences: list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]],
    *,
    stage: str,
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Direct Galaxea equivalent of wrapper1 ``_mark_failed_plan_blocker``."""
    signature = canonical_action_signature_from_steps(candidate_steps)
    if not signature or signature == "[]":
        return []
    candidate_actions = signature_actions_from_steps(candidate_steps)
    matches: list[dict[str, Any]] = []
    for item in retrieved_experiences or []:
        if len(item) < 2:
            continue
        entry = item[0]
        score = float(item[1] or 0.0)
        if field_atomic_success(entry):
            continue
        failed_signature = canonical_action_signature_from_entry(entry)
        failed_actions = signature_actions_from_entry(entry)
        overlap = action_lcs_ratio(candidate_actions, failed_actions)
        exact_match = bool(failed_signature and failed_signature == signature)
        short_failure_block = _short_failure_block(candidate_steps, candidate_actions, entry, failed_actions)
        if not exact_match and overlap < threshold and not short_failure_block:
            continue
        failure_taxonomy = _taxonomy_with_galaxea_hint(entry)
        feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
        critic = feedback.get("llm_critic") if isinstance(feedback.get("llm_critic"), dict) else {}
        if not critic:
            critic = failure_taxonomy.get("llm_critic") if isinstance(failure_taxonomy.get("llm_critic"), dict) else {}
        matches.append({
            "stage": stage,
            "experience_id": entry.experience_id,
            "partition": str(entry.memory_tags.get("memory_role") or ""),
            "score": score,
            "overlap": overlap,
            "exact_signature_match": exact_match,
            "short_failure_block": short_failure_block,
            "candidate_signature": signature,
            "failed_signature": failed_signature,
            "failure_stage": failure_taxonomy.get("failure_stage", ""),
            "failure_type": failure_taxonomy.get("failure_type", ""),
            "cluster_id": failure_taxonomy.get("cluster_id", ""),
            "critic_root_cause": critic.get("root_cause", ""),
            "corrective_direction": critic.get("corrective_direction", ""),
            "missing_phases": critic.get("missing_phases", []),
            "blocked": True,
            "rewrite_triggered": False,
        })
    return matches


def _short_failure_block(
    candidate_steps: list[dict[str, Any]],
    candidate_actions: list[str],
    entry: ExperienceEntry,
    failed_actions: list[str],
) -> bool:
    """Galaxea extension: block only when the same failure reappears with no real parameter shift.

    The old action-level rule was too strict: a plan could reuse the same skill
    with different reachability parameters and still be valid. Here we block only
    when the failed atomic skill is repeated *and* the candidate does not change
    the relevant parameters or reachability conditions.
    """
    if len(failed_actions) != 1:
        return False
    failed_action = failed_actions[0]
    if failed_action not in set(candidate_actions):
        return False
    taxonomy = _taxonomy_with_galaxea_hint(entry)
    failure_type = str(taxonomy.get("failure_type") or "")
    failure_stage = str(taxonomy.get("failure_stage") or taxonomy.get("failure_action") or "")
    candidate_set = set(candidate_actions)
    failed_params = _action_parameters_for_entry(entry, failed_action)
    candidate_params = _candidate_parameters_for_action(candidate_steps, failed_action)
    if failed_params or candidate_params:
        if _parameters_changed_significantly(failed_action, failed_params, candidate_params):
            return False
    if failure_type == "actuation_limit" or failure_stage in {"move_to_pregrasp", "approach_object"}:
        if "move_base_relative" in candidate_set or "set_torso_posture" in candidate_set:
            return False
        if {"head_camera_rgbd_save", "head_camera_grounded_sam2_pose"}.issubset(candidate_set):
            return False
        return True
    if failure_type in {"object_not_lifted", "grasp_miss"}:
        return not {"head_camera_rgbd_save", "head_camera_grounded_sam2_pose", "approach_object", "close_gripper", "lift"}.issubset(candidate_set)
    return True


def _action_parameters_for_entry(entry: ExperienceEntry, action: str) -> dict[str, Any]:
    params = field_atomic_parameters(entry)
    if params:
        return params
    if not action:
        return {}
    for item in entry.skill_sequence:
        if canonical_skill_action(item.name) != action:
            continue
        raw = item.raw if isinstance(item.raw, dict) else {}
        raw_params = raw.get("parameters")
        if isinstance(raw_params, dict):
            return {str(key): value for key, value in raw_params.items() if not str(key).startswith("_")}
    return {}


def _candidate_parameters_for_action(candidate_steps: list[dict[str, Any]], action: str) -> dict[str, Any]:
    for step in candidate_steps or []:
        if not isinstance(step, dict):
            continue
        step_action = canonical_skill_action(str(step.get("action") or step.get("name") or step.get("skill") or ""))
        if step_action != action:
            continue
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        return {str(key): value for key, value in params.items() if not str(key).startswith("_")}
    return {}


def _parameters_changed_significantly(action: str, failed_params: dict[str, Any], candidate_params: dict[str, Any]) -> bool:
    if not failed_params or not candidate_params:
        return False
    relevant_keys = _relevant_parameter_keys(action)
    if relevant_keys:
        keys = [key for key in relevant_keys if key in failed_params or key in candidate_params]
    else:
        keys = [key for key in set(failed_params) | set(candidate_params) if not str(key).startswith("_")]
    if not keys:
        return False
    for key in keys:
        if _normalize_param_value(failed_params.get(key)) != _normalize_param_value(candidate_params.get(key)):
            return True
    return False


def _relevant_parameter_keys(action: str) -> set[str]:
    if action == "move_base_relative":
        return {"x", "y", "yaw", "distance"}
    if action == "set_torso_posture":
        return {"level", "height", "z", "y"}
    if action == "move_to_pregrasp":
        return {"target_torso", "pregrasp_torso", "offset", "offset_x", "offset_y", "offset_z", "mode", "trajectory_mode", "side"}
    if action == "approach_object":
        return {"target_torso", "approach_torso", "pregrasp_torso", "offset", "offset_x", "offset_y", "offset_z", "mode", "trajectory_mode", "side"}
    return set()


def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_normalize_param_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_param_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])) if not str(key).startswith("_")}
    return value


def memory_usefulness_wrapper1_style(
    executed_steps: list[dict[str, Any]],
    retrieved_experiences: list[tuple[ExperienceEntry, float]] | list[tuple[ExperienceEntry, float, dict[str, Any]]],
) -> dict[str, Any]:
    executed_signature = canonical_action_signature_from_steps(executed_steps)
    executed_actions = signature_actions_from_steps(executed_steps)
    records: list[dict[str, Any]] = []
    positive_count = 0
    failed_count = 0
    avoided_failed = False
    for item in retrieved_experiences or []:
        if len(item) < 2:
            continue
        entry = item[0]
        score = float(item[1] or 0.0)
        entry_signature = canonical_action_signature_from_entry(entry)
        entry_actions = signature_actions_from_entry(entry)
        is_failed = not field_atomic_success(entry)
        positive_count += int(not is_failed)
        failed_count += int(is_failed)
        overlap = action_lcs_ratio(executed_actions, entry_actions)
        exact_match = bool(executed_signature and entry_signature == executed_signature)
        if is_failed and not exact_match:
            avoided_failed = True
        useful = (not is_failed and overlap >= 0.6) or (is_failed and not exact_match)
        records.append({
            "experience_id": entry.experience_id,
            "partition": str(entry.memory_tags.get("memory_role") or ""),
            "score": score,
            "used_as": "negative" if is_failed else "positive",
            "plan_signature": entry_signature,
            "action_overlap": overlap,
            "exact_signature_match": exact_match,
            "useful_memory": useful,
        })
    overlaps = [row["action_overlap"] for row in records if row["used_as"] == "positive"]
    return {
        "executed_plan_signature": executed_signature,
        "retrieved_positive_count": positive_count,
        "retrieved_failed_count": failed_count,
        "memory_action_overlap_mean": sum(overlaps) / len(overlaps) if overlaps else None,
        "memory_action_overlap_max": max(overlaps) if overlaps else None,
        "useful_memory_ratio": sum(1 for row in records if row["useful_memory"]) / len(records) if records else None,
        "avoided_failed_plan": bool(failed_count and avoided_failed),
        "memory_usefulness": records,
    }


def deterministic_rule_critic_wrapper1_style(metrics: dict[str, Any]) -> dict[str, Any]:
    """Wrapper1-style deterministic critic for Galaxea execution metrics."""
    flags: list[dict[str, Any]] = []
    criteria = metrics.get("task_success_criteria") or {}
    failed_action = metrics.get("failed_action") if isinstance(metrics.get("failed_action"), dict) else {}
    rule_metrics = metrics.get("rule_metrics") if isinstance(metrics.get("rule_metrics"), dict) else {}

    if failed_action:
        action = str(failed_action.get("action") or "")
        raw = failed_action.get("raw_result") if isinstance(failed_action.get("raw_result"), dict) else {}
        final_error = raw.get("final_error")
        if action in {"move_to_pregrasp", "approach_object"}:
            flags.append({
                "rule": "actuation_limit",
                "stage": action,
                "evidence": f"{action} failed; final_error={final_error}; target_torso={raw.get('target_torso')}",
            })
        elif action == "head_camera_grounded_sam2_pose":
            flags.append({"rule": "perception_miss", "stage": "detection", "evidence": str(failed_action.get("message") or "")})

    object_lift_success = metrics.get("object_lift_success", criteria.get("object_lift_success"))
    object_lift_world = metrics.get("object_lift_world", criteria.get("object_lift_world"))
    min_lift = metrics.get("min_object_lift", criteria.get("min_object_lift"))
    if object_lift_success is False:
        flags.append({
            "rule": "object_not_lifted",
            "stage": "lift",
            "evidence": f"object_lift_world={object_lift_world}; min_object_lift={min_lift}",
        })

    failed_predicates = criteria.get("failed_predicates") if isinstance(criteria, dict) else []
    for predicate in failed_predicates or []:
        flags.append({"rule": str(predicate), "stage": "task_completion", "evidence": "failed predicate"})

    if rule_metrics.get("final_error") is not None:
        try:
            final_error = float(rule_metrics.get("final_error"))
        except (TypeError, ValueError):
            final_error = 0.0
        if final_error > 0.08:
            flags.append({"rule": "end_effector_pose_error_high", "stage": "motion", "evidence": f"final_error={final_error:.4f}"})

    return {
        "enabled": True,
        "rule_flags": flags,
        "critic_warning": bool(flags),
    }


def _taxonomy_with_galaxea_hint(entry: ExperienceEntry) -> dict[str, Any]:
    taxonomy = dict(entry.failure_taxonomy if isinstance(entry.failure_taxonomy, dict) else {})
    critic = taxonomy.get("llm_critic") if isinstance(taxonomy.get("llm_critic"), dict) else {}
    failure_type = str(taxonomy.get("failure_type") or "")
    failure_stage = str(taxonomy.get("failure_stage") or taxonomy.get("failure_action") or "")
    if failure_type == "actuation_limit" or failure_stage in {"move_to_pregrasp", "approach_object"}:
        taxonomy.setdefault("llm_critic", critic or {})
    return taxonomy


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))
