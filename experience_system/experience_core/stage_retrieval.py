"""Stage-aware retrieval evidence for candidate planning and sandbox reports."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .library import ExperienceLibrary
from .retrieval import RetrievalMatch, RetrievalQuery
from .schema import ExperienceEntry
from .scoring import action_lcs_ratio, action_set_overlap, critic_risk_score, estimate_gap_uncertainty


STAGE_POLICIES: dict[str, dict[str, Any]] = {
    "candidate_generation": {
        "description": "successful or validated episodes used as positive candidate evidence",
        "top_k": 4,
        "include_failed": False,
        "risk_aware": False,
        "memory_partition": "validated_memory",
        "memory_role": "",
        "use_candidate_steps": False,
    },
    "candidate_ranking": {
        "description": "failure and sim-real gap memories used as ranking risk evidence",
        "top_k": 6,
        "include_failed": True,
        "risk_aware": True,
        "memory_partition": "",
        "memory_role": "sim_real_gap_memory",
        "use_candidate_steps": True,
    },
    "sandbox_rewrite": {
        "description": "critic warnings, blocks, and risky histories used as rewrite evidence",
        "top_k": 4,
        "include_failed": True,
        "risk_aware": True,
        "memory_partition": "",
        "memory_role": "",
        "use_candidate_steps": True,
    },
    "execution_writeback": {
        "description": "recent executed/writeback and real-format evidence used before committing execution",
        "top_k": 4,
        "include_failed": True,
        "risk_aware": False,
        "memory_partition": "",
        "memory_role": "",
        "use_candidate_steps": True,
    },
}


def _success(entry: ExperienceEntry) -> bool:
    return bool(entry.result.get("success", entry.result.get("task_success", False)))


def _entry_actions(entry: ExperienceEntry) -> list[str]:
    return [item.name for item in entry.skill_sequence if item.name]


def _memory_role(entry: ExperienceEntry) -> str:
    return str(entry.memory_tags.get("memory_role") or "")


def _gap_type(entry: ExperienceEntry) -> str:
    return str(entry.sim_real_gap.outcome_gap.get("type") or "")


def _failure_type(entry: ExperienceEntry) -> str:
    return str(entry.failure_taxonomy.get("failure_type") or entry.failure_taxonomy.get("standard_failure_type") or "")


def _match_summary(match: RetrievalMatch, candidate_steps: list[str]) -> dict[str, Any]:
    entry = match.entry
    entry_actions = _entry_actions(entry)
    return {
        "experience_id": entry.experience_id,
        "score": round(float(match.score), 4),
        "source": entry.source,
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "task_stage": str(entry.task.get("stage") or ""),
        "success": _success(entry),
        "memory_partition": entry.memory_partition,
        "memory_role": _memory_role(entry),
        "memory_type": str(entry.memory_tags.get("memory_type") or ""),
        "critic_status": entry.critic_result.overall_status,
        "critic_risk": round(critic_risk_score(entry), 4),
        "gap_type": _gap_type(entry),
        "gap_score": round(float(entry.sim_real_gap.gap_score or 0.0), 4),
        "gap_uncertainty": round(estimate_gap_uncertainty(entry), 4),
        "failure_type": _failure_type(entry),
        "skill_sequence": entry_actions,
        "action_lcs_overlap": round(action_lcs_ratio(candidate_steps, entry_actions), 4),
        "action_set_overlap": round(action_set_overlap(candidate_steps, entry_actions), 4),
        "explanation": match.explanation,
    }


def _counter(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key) or "") for item in items).items()))


def _stage_query(
    *,
    stage: str,
    policy: dict[str, Any],
    scenario: str,
    condition: str,
    object_class: str,
    candidate_steps: list[str],
    top_k: int | None,
) -> RetrievalQuery:
    limit = int(top_k if top_k is not None else policy.get("top_k", 5))
    return RetrievalQuery(
        scenario_id=scenario,
        condition_id=condition,
        robot_type="mobile_dual_arm",
        object_class=object_class,
        task_stage="task_chain",
        memory_partition=str(policy.get("memory_partition") or ""),
        memory_role=str(policy.get("memory_role") or ""),
        skill_sequence=candidate_steps if policy.get("use_candidate_steps") else [],
        include_failed=bool(policy.get("include_failed", True)),
        risk_aware=bool(policy.get("risk_aware", False)),
        update_retrieval_stats=False,
        top_k=limit,
    )


def _support_component(stage: str, item: dict[str, Any]) -> float:
    if not item.get("success"):
        return 0.0
    score = float(item.get("score") or 0.0)
    source = str(item.get("source") or "")
    role = str(item.get("memory_role") or "")
    partition = str(item.get("memory_partition") or "")
    bonus = 1.0
    if source == "real":
        bonus += 0.25
    elif source == "pseudo_real":
        bonus += 0.12
    if partition == "validated_memory":
        bonus += 0.10
    if role == "writeback_demo_executed":
        bonus += 0.12
    if stage == "candidate_generation":
        bonus += 0.10
    return min(score * bonus, 1.0)


def _risk_component(stage: str, item: dict[str, Any]) -> float:
    score = float(item.get("score") or 0.0)
    risk = 0.0
    if not item.get("success"):
        risk = max(risk, 0.45)
    if str(item.get("gap_type") or "") == "sim_success_real_fail":
        risk = max(risk, 0.70)
    risk = max(risk, 0.55 * float(item.get("gap_uncertainty") or 0.0))
    risk = max(risk, float(item.get("critic_risk") or 0.0))
    overlap = max(float(item.get("action_lcs_overlap") or 0.0), float(item.get("action_set_overlap") or 0.0), 0.25)
    if stage == "sandbox_rewrite" and str(item.get("critic_status") or "") in {"warn", "block"}:
        risk = max(risk, 0.75)
    if stage == "candidate_ranking":
        overlap = max(overlap, 0.45)
    return min(score * risk * overlap, 1.0)


def _candidate_intrinsic_risk(candidate_steps: list[str]) -> float:
    actions = set(candidate_steps)
    risk = 0.0
    if "segmented_transport_fast" in actions:
        risk = max(risk, 0.55)
    if "detect_place_occupancy" not in actions:
        risk = max(risk, 0.12)
    if "verify_place_zone" not in actions:
        risk = max(risk, 0.20)
    return risk


def _score_stage(stage: str, matches: list[dict[str, Any]], candidate_steps: list[str]) -> dict[str, Any]:
    support_values = [_support_component(stage, item) for item in matches]
    risk_values = [_risk_component(stage, item) for item in matches]
    support = sum(support_values) / len(support_values) if support_values else 0.0
    evidence_risk = max(risk_values) if risk_values else 0.0
    intrinsic_risk = 0.0
    risk = evidence_risk
    if stage in {"candidate_ranking", "sandbox_rewrite"}:
        intrinsic_risk = _candidate_intrinsic_risk(candidate_steps)
        risk = max(risk, intrinsic_risk)
    return {
        "stage_support_score": round(max(0.0, min(support, 1.0)), 4),
        "stage_evidence_risk_score": round(max(0.0, min(evidence_risk, 1.0)), 4),
        "stage_intrinsic_risk_score": round(max(0.0, min(intrinsic_risk, 1.0)), 4),
        "stage_risk_score": round(max(0.0, min(risk, 1.0)), 4),
    }


def _specificity(matches_by_stage: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids_by_stage = {stage: {item["experience_id"] for item in matches} for stage, matches in matches_by_stage.items()}
    all_ids = set().union(*ids_by_stage.values()) if ids_by_stage else set()
    overlap_pairs: dict[str, float] = {}
    stages = list(ids_by_stage)
    for index, left in enumerate(stages):
        for right in stages[index + 1 :]:
            union = ids_by_stage[left] | ids_by_stage[right]
            inter = ids_by_stage[left] & ids_by_stage[right]
            overlap_pairs[f"{left}__{right}"] = round(len(inter) / len(union), 4) if union else 0.0
    mean_overlap = round(sum(overlap_pairs.values()) / len(overlap_pairs), 4) if overlap_pairs else 0.0
    return {
        "unique_retrieved_count": len(all_ids),
        "stage_overlap": overlap_pairs,
        "mean_stage_overlap": mean_overlap,
        "stage_specificity_score": round(1.0 - mean_overlap, 4),
    }


def run_stage_retrieval(
    library: ExperienceLibrary,
    *,
    scenario: str,
    condition: str,
    object_class: str,
    candidate_id: str,
    candidate_steps: list[str],
    top_k: int | None = None,
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    matches_by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage, policy in STAGE_POLICIES.items():
        query = _stage_query(
            stage=stage,
            policy=policy,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            candidate_steps=candidate_steps,
            top_k=top_k,
        )
        matches = [_match_summary(match, candidate_steps) for match in library.query_structured(query)]
        matches_by_stage[stage] = matches
        stage_scores = _score_stage(stage, matches, candidate_steps)
        stages[stage] = {
            "stage": stage,
            "description": policy["description"],
            "query": {
                "scenario_id": query.scenario_id,
                "condition_id": query.condition_id,
                "robot_type": query.robot_type,
                "object_class": query.object_class,
                "task_stage": query.task_stage,
                "memory_partition": query.memory_partition,
                "memory_role": query.memory_role,
                "include_failed": query.include_failed,
                "risk_aware": query.risk_aware,
                "skill_count": len(query.skill_sequence),
                "top_k": query.top_k,
            },
            "match_count": len(matches),
            "memory_role_distribution": _counter(matches, "memory_role"),
            "memory_partition_distribution": _counter(matches, "memory_partition"),
            "critic_status_distribution": _counter(matches, "critic_status"),
            "gap_type_distribution": _counter(matches, "gap_type"),
            "failure_type_distribution": _counter(matches, "failure_type"),
            **stage_scores,
            "matches": matches,
        }

    support_scores = [float(item["stage_support_score"]) for item in stages.values()]
    risk_scores = [float(item["stage_risk_score"]) for item in stages.values()]
    support = sum(support_scores) / len(support_scores) if support_scores else 0.0
    risk = max(risk_scores) if risk_scores else 0.0
    specificity = _specificity(matches_by_stage)
    return {
        "schema_version": "stage_aware_candidate_retrieval_v1",
        "candidate_id": candidate_id,
        "stage_count": len(stages),
        "stage_support_score": round(support, 4),
        "stage_risk_score": round(risk, 4),
        "specificity": specificity,
        "stages": stages,
    }


def apply_stage_score_adjustment(
    candidate_report: dict[str, Any],
    stage_report: dict[str, Any],
    *,
    support_weight: float = 0.08,
    risk_weight: float = 0.12,
) -> dict[str, Any]:
    score = dict(candidate_report.get("candidate_score") or {})
    base = float(score.get("candidate_score", 0.0))
    support = float(stage_report.get("stage_support_score", 0.0))
    risk = float(stage_report.get("stage_risk_score", 0.0))
    support_delta = max(float(support_weight), 0.0) * support
    risk_delta = max(float(risk_weight), 0.0) * risk
    adjusted = max(0.0, min(base + support_delta - risk_delta, 1.0))
    original_decision = str(score.get("decision") or "review")
    if adjusted >= 0.60 and original_decision != "rewrite":
        decision = "accept"
    elif adjusted >= 0.40 and original_decision != "reject":
        decision = "review"
    else:
        decision = "reject"
    score.update({
        "candidate_score_before_stage": round(base, 4),
        "candidate_score": round(adjusted, 4),
        "decision_before_stage": original_decision,
        "decision": decision,
        "stage_support_score": round(support, 4),
        "stage_risk_score": round(risk, 4),
        "stage_support_delta": round(support_delta, 4),
        "stage_risk_delta": round(risk_delta, 4),
    })
    candidate_report["candidate_score"] = score
    candidate_report["stage_retrieval"] = stage_report
    return score


def summarize_stage_retrieval(candidate_reports: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [item.get("stage_retrieval") or {} for item in candidate_reports if item.get("stage_retrieval")]
    if not reports:
        return {
            "enabled_candidate_count": 0,
            "stage_count": 0,
            "mean_stage_overlap": 0.0,
            "stage_specificity_score": 0.0,
            "support_score_avg": 0.0,
            "risk_score_avg": 0.0,
            "risk_score_max": 0.0,
        }
    support = [float(item.get("stage_support_score") or 0.0) for item in reports]
    risk = [float(item.get("stage_risk_score") or 0.0) for item in reports]
    specificity = [float((item.get("specificity") or {}).get("stage_specificity_score") or 0.0) for item in reports]
    overlap = [float((item.get("specificity") or {}).get("mean_stage_overlap") or 0.0) for item in reports]
    return {
        "enabled_candidate_count": len(reports),
        "stage_count": max(int(item.get("stage_count") or 0) for item in reports),
        "mean_stage_overlap": round(sum(overlap) / len(overlap), 4),
        "stage_specificity_score": round(sum(specificity) / len(specificity), 4),
        "support_score_avg": round(sum(support) / len(support), 4),
        "risk_score_avg": round(sum(risk) / len(risk), 4),
        "risk_score_max": round(max(risk), 4),
    }
