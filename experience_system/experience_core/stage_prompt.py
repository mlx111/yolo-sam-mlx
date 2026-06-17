"""Render compact stage-aware planner context from retrieval evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any


STAGE_ORDER = [
    "candidate_generation",
    "candidate_ranking",
    "sandbox_rewrite",
    "execution_writeback",
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_value(item: dict[str, Any]) -> float:
    values = [
        _float(item.get("critic_risk")),
        _float(item.get("gap_uncertainty")),
        _float(item.get("gap_score")),
    ]
    if not item.get("success"):
        values.append(0.45)
    if str(item.get("critic_status") or "") == "warn":
        values.append(0.55)
    if str(item.get("critic_status") or "") == "block":
        values.append(0.85)
    if str(item.get("gap_type") or "") == "sim_success_real_fail":
        values.append(0.75)
    return max(values) if values else 0.0


def _skill_sequence(item: dict[str, Any]) -> list[str]:
    skills = item.get("skill_sequence") or []
    if not isinstance(skills, list):
        return []
    return [str(skill) for skill in skills if str(skill)]


def _matching_skills(candidate_steps: list[str], evidence_steps: list[str]) -> list[str]:
    candidate_set = set(candidate_steps)
    return [skill for skill in evidence_steps if skill in candidate_set]


def _compact_match(item: dict[str, Any], candidate_steps: list[str]) -> dict[str, Any]:
    evidence_steps = _skill_sequence(item)
    return {
        "experience_id": item.get("experience_id", ""),
        "score": item.get("score", 0.0),
        "source": item.get("source", ""),
        "scenario_id": item.get("scenario_id", ""),
        "condition_id": item.get("condition_id", ""),
        "success": bool(item.get("success", False)),
        "memory_partition": item.get("memory_partition", ""),
        "memory_role": item.get("memory_role", ""),
        "critic_status": item.get("critic_status", ""),
        "critic_risk": item.get("critic_risk", 0.0),
        "gap_type": item.get("gap_type", ""),
        "gap_uncertainty": item.get("gap_uncertainty", 0.0),
        "failure_type": item.get("failure_type", ""),
        "action_lcs_overlap": item.get("action_lcs_overlap", 0.0),
        "action_set_overlap": item.get("action_set_overlap", 0.0),
        "matching_skills": _matching_skills(candidate_steps, evidence_steps),
    }


def _top_matches(matches: list[dict[str, Any]], *, limit: int, risk_first: bool = False) -> list[dict[str, Any]]:
    if risk_first:
        return sorted(matches, key=lambda item: (_risk_value(item), _float(item.get("score"))), reverse=True)[:limit]
    return sorted(matches, key=lambda item: _float(item.get("score")), reverse=True)[:limit]


def _recommended_skills(matches: list[dict[str, Any]], candidate_steps: list[str], *, limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    support: Counter[str] = Counter()
    candidate_set = set(candidate_steps)
    for item in matches:
        if not item.get("success"):
            continue
        for skill in _skill_sequence(item):
            if skill not in candidate_set:
                continue
            counts[skill] += 1
            support[skill] += _float(item.get("score"))
    ranked = sorted(counts, key=lambda skill: (counts[skill], support[skill], -candidate_steps.index(skill)), reverse=True)
    return [
        {
            "skill": skill,
            "support_count": counts[skill],
            "support_score_sum": round(float(support[skill]), 4),
        }
        for skill in ranked[:limit]
    ]


def _positive_examples(stage: dict[str, Any], candidate_steps: list[str], *, limit: int) -> list[dict[str, Any]]:
    matches = [item for item in stage.get("matches", []) if isinstance(item, dict) and item.get("success")]
    return [_compact_match(item, candidate_steps) for item in _top_matches(matches, limit=limit)]


def _failure_risks(stage: dict[str, Any], candidate_steps: list[str], *, limit: int) -> list[dict[str, Any]]:
    matches = [
        item
        for item in stage.get("matches", [])
        if isinstance(item, dict)
        and (
            not item.get("success")
            or str(item.get("gap_type") or "")
            or _float(item.get("critic_risk")) > 0.0
            or str(item.get("critic_status") or "") in {"warn", "block"}
        )
    ]
    risks = []
    for item in _top_matches(matches, limit=limit, risk_first=True):
        compact = _compact_match(item, candidate_steps)
        compact["risk_value"] = round(_risk_value(item), 4)
        risks.append(compact)
    return risks


def _gap_uncertainty(stage: dict[str, Any], candidate_steps: list[str], *, limit: int) -> list[dict[str, Any]]:
    matches = [
        item
        for item in stage.get("matches", [])
        if isinstance(item, dict) and (_float(item.get("gap_uncertainty")) > 0.0 or str(item.get("gap_type") or ""))
    ]
    return [_compact_match(item, candidate_steps) for item in _top_matches(matches, limit=limit, risk_first=True)]


def _critic_warnings(stage: dict[str, Any], candidate_steps: list[str], *, limit: int) -> list[dict[str, Any]]:
    matches = [
        item
        for item in stage.get("matches", [])
        if isinstance(item, dict)
        and (str(item.get("critic_status") or "") in {"warn", "block"} or _float(item.get("critic_risk")) >= 0.35)
    ]
    return [_compact_match(item, candidate_steps) for item in _top_matches(matches, limit=limit, risk_first=True)]


def _avoidance_hints(warnings: list[dict[str, Any]], candidate_steps: list[str], *, limit: int) -> list[dict[str, Any]]:
    hints = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in warnings:
        overlapping = [skill for skill in item.get("matching_skills", []) if skill in candidate_steps]
        if not overlapping:
            continue
        key = (str(item.get("failure_type") or item.get("critic_status") or item.get("gap_type") or "risk"), tuple(overlapping))
        if key in seen:
            continue
        seen.add(key)
        hints.append({
            "risk_source": item.get("experience_id", ""),
            "risk_type": item.get("failure_type") or item.get("critic_status") or item.get("gap_type") or "retrieved_risk",
            "watch_skills": overlapping[:4],
            "evidence_risk": item.get("risk_value", item.get("critic_risk", 0.0)),
        })
        if len(hints) >= limit:
            break
    return hints


def _recent_executions(stage: dict[str, Any], candidate_steps: list[str], *, limit: int) -> list[dict[str, Any]]:
    matches = [
        item
        for item in stage.get("matches", [])
        if isinstance(item, dict)
        and (
            str(item.get("memory_role") or "") == "writeback_demo_executed"
            or str(item.get("source") or "") in {"real", "pseudo_real"}
            or _float(item.get("score")) > 0.0
        )
    ]
    return [_compact_match(item, candidate_steps) for item in _top_matches(matches, limit=limit)]


def _approx_token_count(text: str) -> int:
    # Conservative enough for reporting without depending on a tokenizer.
    return max(1, int(len(text) / 4)) if text else 0


def render_stage_prompt_text(context: dict[str, Any]) -> str:
    planner_input = context.get("planner_input") or {}
    lines = [
        f"Stage-aware planner context for {context.get('scenario')}/{context.get('condition')} candidate={context.get('candidate_id')}",
        f"Candidate steps: {', '.join(context.get('candidate_steps') or [])}",
    ]
    generation = context.get("candidate_generation") or {}
    ranking = context.get("candidate_ranking") or {}
    rewrite = context.get("sandbox_rewrite") or {}
    writeback = context.get("execution_writeback") or {}
    recommended = [item.get("skill", "") for item in generation.get("recommended_skills", []) if item.get("skill")]
    lines.append(f"[candidate_generation] use {len(generation.get('positive_examples') or [])} positive memories; recommended skills: {', '.join(recommended) or 'none'}")
    risks = ranking.get("top_failure_risks") or []
    lines.append(f"[candidate_ranking] compare against {len(risks)} risk memories; max risk={max([_float(item.get('risk_value')) for item in risks], default=0.0):.4f}")
    warnings = rewrite.get("critic_warnings") or []
    watch = []
    for hint in rewrite.get("avoidance_hints") or []:
        watch.extend(hint.get("watch_skills") or [])
    lines.append(f"[sandbox_rewrite] critic warnings={len(warnings)}; watch skills: {', '.join(dict.fromkeys(watch)) or 'none'}")
    lines.append(f"[execution_writeback] recent similar executions={len(writeback.get('recent_similar_executions') or [])}")
    if planner_input:
        constraints = planner_input.get("planner_constraints") or []
        objectives = planner_input.get("planner_objectives") or []
        lines.append(f"[planner_input] objectives: {'; '.join(objectives) or 'none'}")
        lines.append(f"[planner_input] constraints: {'; '.join(constraints) or 'none'}")
    lines.append("Use generation evidence for candidate construction, ranking evidence for risk penalties, rewrite evidence for sandbox repair, and writeback evidence before committing execution.")
    return "\n".join(lines)


def build_structured_planner_input(context: dict[str, Any]) -> dict[str, Any]:
    """Return a planner/LLM-friendly structured view of stage context."""

    generation = context.get("candidate_generation") or {}
    ranking = context.get("candidate_ranking") or {}
    rewrite = context.get("sandbox_rewrite") or {}
    writeback = context.get("execution_writeback") or {}
    metrics = context.get("metrics") or {}
    recommended_skills = [
        str(item.get("skill"))
        for item in generation.get("recommended_skills") or []
        if item.get("skill")
    ]
    risk_memories = ranking.get("top_failure_risks") or []
    gap_memories = ranking.get("gap_uncertainty") or []
    critic_warnings = rewrite.get("critic_warnings") or []
    avoidance_hints = rewrite.get("avoidance_hints") or []
    watched_skills = list(dict.fromkeys(
        str(skill)
        for hint in avoidance_hints
        for skill in (hint.get("watch_skills") or [])
        if str(skill)
    ))
    return {
        "schema_version": "stage_structured_planner_input_v1",
        "planner_objectives": [
            "construct_or_rank_candidate_using_stage_specific_memory",
            "prefer_supported_steps_from_candidate_generation",
            "penalize_retrieved_failure_and_gap_risks",
            "prepare_rewrite_hints_before_sandbox_rollout",
            "check_writeback_evidence_before_execution_commit",
        ],
        "candidate": {
            "scenario": context.get("scenario", ""),
            "condition": context.get("condition", ""),
            "candidate_id": context.get("candidate_id", ""),
            "description": context.get("candidate_description", ""),
            "steps": list(context.get("candidate_steps") or []),
        },
        "generation_guidance": {
            "support_score": generation.get("stage_support_score", 0.0),
            "recommended_skills": recommended_skills,
            "positive_memory_ids": [
                item.get("experience_id", "")
                for item in generation.get("positive_examples") or []
                if item.get("experience_id")
            ],
        },
        "ranking_guidance": {
            "risk_score": ranking.get("stage_risk_score", 0.0),
            "risk_memory_ids": [
                item.get("experience_id", "")
                for item in risk_memories
                if item.get("experience_id")
            ],
            "gap_memory_ids": [
                item.get("experience_id", "")
                for item in gap_memories
                if item.get("experience_id")
            ],
            "max_retrieved_risk": round(max([_float(item.get("risk_value")) for item in risk_memories], default=0.0), 4),
        },
        "rewrite_guidance": {
            "rewrite_risk_score": rewrite.get("stage_risk_score", 0.0),
            "critic_warning_ids": [
                item.get("experience_id", "")
                for item in critic_warnings
                if item.get("experience_id")
            ],
            "avoidance_hints": avoidance_hints,
            "watched_skills": watched_skills,
        },
        "writeback_guidance": {
            "support_score": writeback.get("stage_support_score", 0.0),
            "recent_execution_ids": [
                item.get("experience_id", "")
                for item in writeback.get("recent_similar_executions") or []
                if item.get("experience_id")
            ],
        },
        "planner_constraints": [
            "do_not_claim_real_robot_validation_without_real_episode_evidence",
            "do_not_override_sandbox_or_motion_critic_blocks",
            "use_stage_context_as_auxiliary_guidance_not_as_final_safety_decision",
        ],
        "metrics": {
            "stage_specificity_score": metrics.get("stage_specificity_score", 0.0),
            "distinct_memory_count": metrics.get("stage_context_distinct_memory_count", 0),
            "positive_example_count": metrics.get("positive_example_count", 0),
            "risk_evidence_count": metrics.get("risk_evidence_count", 0),
            "critic_warning_count": metrics.get("critic_warning_count", 0),
            "writeback_evidence_count": metrics.get("writeback_evidence_count", 0),
        },
    }


def build_stage_planner_context(
    stage_report: dict[str, Any],
    *,
    scenario: str,
    condition: str,
    candidate_id: str,
    candidate_steps: list[str],
    candidate_description: str = "",
    max_examples: int = 3,
    max_risks: int = 4,
    max_warnings: int = 4,
    max_writeback: int = 3,
) -> dict[str, Any]:
    stages = stage_report.get("stages") or {}
    generation_stage = stages.get("candidate_generation") or {}
    ranking_stage = stages.get("candidate_ranking") or {}
    rewrite_stage = stages.get("sandbox_rewrite") or {}
    writeback_stage = stages.get("execution_writeback") or {}

    positive_examples = _positive_examples(generation_stage, candidate_steps, limit=max_examples)
    failure_risks = _failure_risks(ranking_stage, candidate_steps, limit=max_risks)
    gap_items = _gap_uncertainty(ranking_stage, candidate_steps, limit=max_risks)
    critic_warnings = _critic_warnings(rewrite_stage, candidate_steps, limit=max_warnings)
    avoidance_hints = _avoidance_hints(critic_warnings, candidate_steps, limit=max_warnings)
    recent_executions = _recent_executions(writeback_stage, candidate_steps, limit=max_writeback)

    distinct_ids = {
        str(item.get("experience_id"))
        for group in (positive_examples, failure_risks, gap_items, critic_warnings, recent_executions)
        for item in group
        if item.get("experience_id")
    }
    specificity = stage_report.get("specificity") or {}
    context = {
        "schema_version": "stage_planner_context_v1",
        "scenario": scenario,
        "condition": condition,
        "candidate_id": candidate_id,
        "candidate_description": candidate_description,
        "candidate_steps": candidate_steps,
        "candidate_generation": {
            "stage_support_score": generation_stage.get("stage_support_score", 0.0),
            "positive_examples": positive_examples,
            "recommended_skills": _recommended_skills(generation_stage.get("matches") or [], candidate_steps, limit=max_examples * 2),
        },
        "candidate_ranking": {
            "stage_risk_score": ranking_stage.get("stage_risk_score", 0.0),
            "top_failure_risks": failure_risks,
            "gap_uncertainty": gap_items,
        },
        "sandbox_rewrite": {
            "stage_risk_score": rewrite_stage.get("stage_risk_score", 0.0),
            "critic_warnings": critic_warnings,
            "avoidance_hints": avoidance_hints,
        },
        "execution_writeback": {
            "stage_support_score": writeback_stage.get("stage_support_score", 0.0),
            "recent_similar_executions": recent_executions,
        },
        "metrics": {
            "stage_context_distinct_memory_count": len(distinct_ids),
            "stage_overlap": specificity.get("stage_overlap", {}),
            "mean_stage_overlap": specificity.get("mean_stage_overlap", 0.0),
            "stage_specificity_score": specificity.get("stage_specificity_score", 0.0),
            "positive_example_count": len(positive_examples),
            "risk_evidence_count": len(failure_risks),
            "gap_evidence_count": len(gap_items),
            "critic_warning_count": len(critic_warnings),
            "writeback_evidence_count": len(recent_executions),
        },
        "source_stage_retrieval_schema": stage_report.get("schema_version", ""),
    }
    context["planner_input"] = build_structured_planner_input(context)
    prompt_text = render_stage_prompt_text(context)
    context["prompt_text"] = prompt_text
    context["metrics"]["stage_context_token_count"] = _approx_token_count(prompt_text)
    return context


def apply_stage_planner_guidance(
    candidate_report: dict[str, Any],
    context: dict[str, Any],
    *,
    guidance_weight: float = 0.10,
) -> dict[str, Any]:
    """Use rendered stage context as a planner-facing candidate adjustment.

    Stage retrieval scores are raw evidence. This function consumes the compact
    planner context that would be given to a planner/LLM and turns it into an
    explicit, bounded adjustment so the context participates in selection.
    """

    score = dict(candidate_report.get("candidate_score") or {})
    if not score or not context:
        return {
            "applied": False,
            "reason": "missing_candidate_score_or_context",
            "planner_bonus": 0.0,
            "planner_penalty": 0.0,
            "net_adjustment": 0.0,
        }

    weight = max(0.0, min(float(guidance_weight), 0.5))
    generation = context.get("candidate_generation") or {}
    ranking = context.get("candidate_ranking") or {}
    rewrite = context.get("sandbox_rewrite") or {}
    writeback = context.get("execution_writeback") or {}
    metrics = context.get("metrics") or {}

    generation_support = _float(generation.get("stage_support_score"))
    writeback_support = _float(writeback.get("stage_support_score"))
    specificity = _float(metrics.get("stage_specificity_score"))
    ranking_risk = _float(ranking.get("stage_risk_score"))
    rewrite_risk = _float(rewrite.get("stage_risk_score"))
    avoidance_hints = rewrite.get("avoidance_hints") or []

    positive_count = _float(metrics.get("positive_example_count"))
    risk_count = _float(metrics.get("risk_evidence_count"))
    critic_count = _float(metrics.get("critic_warning_count"))
    writeback_count = _float(metrics.get("writeback_evidence_count"))

    support_signal = min(
        0.55 * generation_support
        + 0.25 * writeback_support
        + 0.20 * specificity
        + 0.02 * min(positive_count + writeback_count, 4.0),
        1.0,
    )
    risk_signal = min(
        0.55 * ranking_risk
        + 0.35 * rewrite_risk
        + 0.03 * min(risk_count + critic_count, 6.0)
        + 0.02 * min(len(avoidance_hints), 4),
        1.0,
    )
    bonus = round(weight * support_signal, 4)
    penalty = round(weight * risk_signal, 4)
    net = round(bonus - penalty, 4)

    original_score = _float(score.get("candidate_score"))
    original_risk = _float(score.get("risk_score"))
    adjusted_score = max(0.0, min(original_score + net, 1.0))
    adjusted_risk = max(0.0, min(original_risk + penalty - 0.5 * bonus, 1.0))
    original_decision = str(score.get("decision") or "review")
    if adjusted_risk >= 0.70:
        decision = "rewrite"
    elif adjusted_score >= 0.60 and original_decision != "rewrite":
        decision = "accept"
    elif adjusted_score >= 0.40 and original_decision != "reject":
        decision = "review"
    else:
        decision = "reject"

    guidance = {
        "applied": True,
        "schema_version": "stage_planner_guidance_adjustment_v1",
        "guidance_weight": round(weight, 4),
        "planner_bonus": bonus,
        "planner_penalty": penalty,
        "net_adjustment": net,
        "support_signal": round(support_signal, 4),
        "risk_signal": round(risk_signal, 4),
        "generation_support": round(generation_support, 4),
        "writeback_support": round(writeback_support, 4),
        "specificity": round(specificity, 4),
        "ranking_risk": round(ranking_risk, 4),
        "rewrite_risk": round(rewrite_risk, 4),
        "avoidance_hint_count": len(avoidance_hints),
        "watched_skills": list(dict.fromkeys(
            str(skill)
            for hint in avoidance_hints
            for skill in (hint.get("watch_skills") or [])
            if str(skill)
        )),
        "score_before_guidance": round(original_score, 4),
        "risk_before_guidance": round(original_risk, 4),
        "decision_before_guidance": original_decision,
        "score_after_guidance": round(adjusted_score, 4),
        "risk_after_guidance": round(adjusted_risk, 4),
        "decision_after_guidance": decision,
        "prompt_text": context.get("prompt_text", ""),
    }
    score.update({
        "candidate_score_before_stage_planner_guidance": round(original_score, 4),
        "risk_score_before_stage_planner_guidance": round(original_risk, 4),
        "decision_before_stage_planner_guidance": original_decision,
        "candidate_score": round(adjusted_score, 4),
        "risk_score": round(adjusted_risk, 4),
        "decision": decision,
        "stage_planner_guidance": guidance,
    })
    candidate_report["candidate_score"] = score
    candidate_report["stage_planner_guidance"] = guidance
    return guidance


def summarize_stage_planner_contexts(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contexts:
        return {
            "context_count": 0,
            "stage_context_token_count_avg": 0.0,
            "stage_context_distinct_memory_count_avg": 0.0,
            "stage_specificity_score_avg": 0.0,
            "risk_evidence_count_total": 0,
            "critic_warning_count_total": 0,
        }
    metrics = [context.get("metrics") or {} for context in contexts]
    return {
        "context_count": len(contexts),
        "stage_context_token_count_avg": round(sum(_float(item.get("stage_context_token_count")) for item in metrics) / len(metrics), 4),
        "stage_context_distinct_memory_count_avg": round(sum(_float(item.get("stage_context_distinct_memory_count")) for item in metrics) / len(metrics), 4),
        "stage_specificity_score_avg": round(sum(_float(item.get("stage_specificity_score")) for item in metrics) / len(metrics), 4),
        "risk_evidence_count_total": int(sum(_float(item.get("risk_evidence_count")) for item in metrics)),
        "critic_warning_count_total": int(sum(_float(item.get("critic_warning_count")) for item in metrics)),
    }
