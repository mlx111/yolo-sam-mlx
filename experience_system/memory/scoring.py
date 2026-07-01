"""Dual-source retrieval and candidate-plan scoring helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any


def _asdict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {}


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _result_success(entry: Any) -> bool:
    result = getattr(entry, "result", None)
    return bool(getattr(result, "success", False) or getattr(result, "task_success", False))


def _gap_fields(entry: Any) -> tuple[dict[str, Any], dict[str, Any], str, float, float]:
    gap = _asdict(getattr(entry, "sim_real_gap", {}) or {})
    outcome = gap.get("outcome_gap") if isinstance(gap.get("outcome_gap"), dict) else {}
    outcome_type = str(outcome.get("type") or "")
    gap_score = _clamp01(gap.get("gap_score"))
    uncertainty = _clamp01(gap.get("uncertainty"))
    return gap, outcome, outcome_type, gap_score, uncertainty


def _signature_actions_from_steps(steps: Any) -> list[str]:
    actions: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if action:
            actions.append(action)
    return actions


def _signature_actions(signature: str) -> list[str]:
    try:
        payload = json.loads(signature)
    except (TypeError, json.JSONDecodeError):
        return []
    actions: list[str] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "")
        if action:
            actions.append(action)
    return actions


def _entry_signature_actions(entry: Any) -> list[str]:
    retrieval_key = getattr(entry, "retrieval_key", {}) if isinstance(getattr(entry, "retrieval_key", {}), dict) else {}
    signature = str(retrieval_key.get("plan_signature") or getattr(entry, "plan_signature", "") or "")
    if signature:
        return _signature_actions(signature)
    return _signature_actions_from_steps(getattr(entry, "skill_sequence", []) or [])


def action_lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a == b else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(left), len(right))


def estimate_gap_uncertainty(entry: Any) -> float:
    _gap, _outcome, outcome_type, gap_score, uncertainty = _gap_fields(entry)
    if outcome_type == "sim_success_real_fail":
        uncertainty = max(uncertainty, 0.85)
    elif outcome_type in {"matched_success", "matched_failure"} and gap_score <= 0.2:
        uncertainty = max(uncertainty, 0.1)
    return _clamp01(max(uncertainty, gap_score * 0.5))


def critic_risk_score(entry: Any) -> float:
    critic = _asdict(getattr(entry, "critic_result", {}) or {})
    risk = _clamp01(critic.get("critic_risk_score"))
    status = str(critic.get("overall_status") or "").lower()
    if status == "block":
        risk = max(risk, 0.9)
    elif status in {"warn", "warning"}:
        risk = max(risk, 0.55)
    elif status == "pass":
        risk = min(risk, 0.2)
    if critic.get("rule_flags"):
        risk = max(risk, min(len(critic.get("rule_flags") or []) * 0.15, 0.75))
    return _clamp01(risk)


def real_validation_bonus(entry: Any) -> float:
    status = str(getattr(entry, "validation_status", "") or "")
    if status == "real_validated":
        return 1.0
    if status == "real_executed":
        return 0.8
    return 0.0


def sim_real_failure_penalty(entry: Any) -> float:
    _gap, _outcome, outcome_type, _gap_score, _uncertainty = _gap_fields(entry)
    if outcome_type == "sim_success_real_fail":
        return 0.45
    if outcome_type == "sim_fail_real_success":
        return 0.0
    return 0.0


def gap_score_penalty(entry: Any) -> float:
    _gap, _outcome, outcome_type, gap_score, _uncertainty = _gap_fields(entry)
    if gap_score <= 0.0:
        return 0.0
    if outcome_type == "matched_success" and gap_score <= 0.2:
        return 0.0
    return 0.25 * gap_score


def gap_uncertainty_penalty(entry: Any) -> float:
    return 0.18 * estimate_gap_uncertainty(entry)


def estimate_real_success_prior(entries: list[tuple[Any, float]]) -> dict[str, Any]:
    total = 0.0
    success = 0.0
    evidence: list[dict[str, Any]] = []
    for entry, score in entries or []:
        real_weight = real_validation_bonus(entry)
        if real_weight <= 0.0:
            continue
        weight = real_weight * max(float(score), 0.0)
        total += weight
        result = getattr(entry, "result", None)
        ok = bool(getattr(result, "success", False) or getattr(result, "task_success", False))
        success += weight * float(ok)
        evidence.append({
            "experience_id": getattr(entry, "experience_id", ""),
            "validation_status": getattr(entry, "validation_status", ""),
            "success": ok,
            "weight": round(weight, 4),
        })
    prior = success / total if total > 0.0 else 0.5
    return {
        "real_success_prior": round(prior, 4),
        "real_evidence_count": len(evidence),
        "evidence": evidence,
    }


def entry_risk_adjustment(entry: Any) -> dict[str, Any]:
    gap_uncertainty = estimate_gap_uncertainty(entry)
    critic_risk = critic_risk_score(entry)
    real_bonus = real_validation_bonus(entry)
    success = _result_success(entry)
    failure_penalty = 0.25 if not success else 0.0
    sim_real_penalty = sim_real_failure_penalty(entry)
    gap_penalty = gap_score_penalty(entry)
    uncertainty_penalty = gap_uncertainty_penalty(entry)
    critic_penalty = 0.18 * critic_risk
    real_success_bonus = 0.20 * real_bonus if success else 0.0
    paired_bonus = 0.05 if _asdict(getattr(entry, "sim_real_pair", {}) or {}).get("validation_status") == "paired" else 0.0
    risk_penalty = uncertainty_penalty + critic_penalty + failure_penalty + sim_real_penalty + gap_penalty
    trust_bonus = real_success_bonus + paired_bonus
    adjustment = trust_bonus - risk_penalty
    return {
        "gap_uncertainty": round(gap_uncertainty, 4),
        "critic_risk": round(critic_risk, 4),
        "real_validation_bonus": round(real_bonus, 4),
        "real_success_bonus": round(real_success_bonus, 4),
        "paired_bonus": round(paired_bonus, 4),
        "gap_score_penalty": round(gap_penalty, 4),
        "gap_uncertainty_penalty": round(uncertainty_penalty, 4),
        "sim_real_failure_penalty": round(sim_real_penalty, 4),
        "critic_penalty": round(critic_penalty, 4),
        "failure_penalty": round(failure_penalty, 4),
        "risk_penalty": round(risk_penalty, 4),
        "trust_bonus": round(trust_bonus, 4),
        "score_adjustment": round(adjustment, 4),
    }


def score_candidate_plan(
    candidate_steps: list[dict[str, Any]],
    retrieved_experiences: list[tuple[Any, float]],
) -> dict[str, Any]:
    """Score a candidate plan against retrieved success/failure/gap/critic memories."""

    candidate_actions = _signature_actions_from_steps(candidate_steps)
    positive_support = 0.0
    failure_overlap_risk = 0.0
    gap_uncertainty = 0.0
    critic_risk = 0.0
    real_success_support = 0.0
    evidence: list[dict[str, Any]] = []

    for entry, score in retrieved_experiences or []:
        entry_actions = _entry_signature_actions(entry)
        overlap = action_lcs_ratio(candidate_actions, entry_actions)
        result = getattr(entry, "result", None)
        entry_success = bool(getattr(result, "success", False) or getattr(result, "task_success", False))
        weight = max(float(score), 0.0) * max(overlap, 0.05)
        entry_gap = estimate_gap_uncertainty(entry)
        entry_critic = critic_risk_score(entry)
        entry_real = real_validation_bonus(entry)

        if entry_success:
            positive_support += weight
            real_success_support += weight * entry_real
        else:
            failure_overlap_risk = max(failure_overlap_risk, overlap)
        gap_uncertainty = max(gap_uncertainty, entry_gap * overlap)
        critic_risk = max(critic_risk, entry_critic * max(overlap, 0.25))
        evidence.append({
            "experience_id": getattr(entry, "experience_id", ""),
            "success": entry_success,
            "score": float(score),
            "action_overlap": round(overlap, 4),
            "gap_uncertainty": round(entry_gap, 4),
            "critic_risk": round(entry_critic, 4),
            "real_validation_bonus": round(entry_real, 4),
        })

    support_score = _clamp01(positive_support / max(len(retrieved_experiences or []), 1))
    real_success_prior = estimate_real_success_prior(retrieved_experiences or [])
    final_score = _clamp01(
        0.50
        + 0.25 * support_score
        + 0.15 * real_success_prior["real_success_prior"]
        + 0.10 * _clamp01(real_success_support)
        - 0.25 * failure_overlap_risk
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
        "real_success_prior": real_success_prior["real_success_prior"],
        "real_evidence_count": real_success_prior["real_evidence_count"],
        "failure_overlap_risk": round(failure_overlap_risk, 4),
        "gap_uncertainty": round(gap_uncertainty, 4),
        "critic_risk": round(critic_risk, 4),
        "candidate_actions": candidate_actions,
        "evidence": evidence,
    }
