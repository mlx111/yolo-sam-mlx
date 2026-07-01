"""Structured retrieval for universal robot experience memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .lifecycle import increment_retrieval_count, memory_tier
from .schema import ExperienceEntry
from .scoring import action_lcs_ratio, entry_risk_adjustment


@dataclass
class RetrievalQuery:
    scenario_id: str = ""
    condition_id: str = ""
    robot_type: str = ""
    backend: str = ""
    skill_namespace: str = ""
    task_stage: str = ""
    source: str = ""
    memory_role: str = ""
    memory_type: str = ""
    memory_partition: str = ""
    failure_type: str = ""
    critic_status: str = ""
    gap_type: str = ""
    object_class: str = ""
    target_object: str = ""
    plan_signature: str = ""
    skill_sequence: list[str] = field(default_factory=list)
    include_failed: bool = True
    require_scenario: bool = True
    risk_aware: bool = False
    visual_scores: dict[str, float] = field(default_factory=dict)
    visual_weight: float = 0.12
    semantic_scores: dict[str, float] = field(default_factory=dict)
    semantic_weight: float = 0.10
    required_sensor_modalities: list[str] = field(default_factory=list)
    preferred_sensor_modalities: list[str] = field(default_factory=list)
    prefer_real_sensor_evidence: bool = False
    sensor_evidence_weight: float = 0.10
    wrist_force_norm_range: tuple[float, float] | None = None
    nearest_obstacle_distance_range: tuple[float, float] | None = None
    ltm_weight: float = 0.05
    update_retrieval_stats: bool = True
    top_k: int = 5


@dataclass
class RetrievalMatch:
    entry: ExperienceEntry
    score: float
    explanation: dict[str, Any] = field(default_factory=dict)


def retrieve_experiences(entries: list[ExperienceEntry], query: RetrievalQuery | dict[str, Any]) -> list[RetrievalMatch]:
    """Return scored matches for a structured retrieval query."""

    query = query if isinstance(query, RetrievalQuery) else RetrievalQuery(**query)
    matches: list[RetrievalMatch] = []
    for entry in entries:
        match = score_entry(entry, query)
        if match is not None:
            matches.append(match)
    matches.sort(key=lambda item: (-item.score, item.entry.experience_id))
    selected = matches[: max(int(query.top_k), 0)]
    if query.update_retrieval_stats:
        for match in selected:
            increment_retrieval_count(match.entry)
    return selected


def score_entry(entry: ExperienceEntry, query: RetrievalQuery) -> RetrievalMatch | None:
    score = 0.0
    explanation: dict[str, Any] = {}

    if not query.include_failed and entry.memory_partition == "failed_memory":
        return None
    if not _passes_hard_filters(entry, query):
        return None

    scenario_score = _exact_score(entry.scenario_id, query.scenario_id, 0.30)
    if query.scenario_id and scenario_score <= 0.0 and query.require_scenario:
        return None
    score += scenario_score
    if query.scenario_id:
        explanation["scenario_id"] = scenario_score

    score += _record_exact(explanation, "condition_id", entry.condition_id, query.condition_id, 0.18)
    score += _record_exact(explanation, "robot_type", entry.robot.robot_type, query.robot_type, 0.10, mismatch_penalty=-0.05)
    score += _record_exact(explanation, "backend", entry.backend, query.backend, 0.06)
    score += _record_exact(explanation, "skill_namespace", entry.skill_namespace, query.skill_namespace, 0.12)
    score += _record_exact(explanation, "task_stage", str(entry.task.get("stage") or ""), query.task_stage, 0.06)
    score += _record_exact(explanation, "source", entry.source, query.source, 0.08)
    score += _record_exact(explanation, "memory_role", str(entry.memory_tags.get("memory_role") or ""), query.memory_role, 0.08)
    score += _record_exact(explanation, "memory_type", str(entry.memory_tags.get("memory_type") or ""), query.memory_type, 0.04)
    score += _record_exact(explanation, "memory_partition", entry.memory_partition, query.memory_partition, 0.06)
    score += _record_exact(explanation, "failure_type", str(entry.failure_taxonomy.get("failure_type") or ""), query.failure_type, 0.08)
    score += _record_exact(explanation, "critic_status", entry.critic_result.overall_status, query.critic_status, 0.08)
    score += _record_exact(explanation, "gap_type", str(entry.sim_real_gap.outcome_gap.get("type") or ""), query.gap_type, 0.09)
    score += _record_exact(explanation, "object_class", entry.object_state.object_class, query.object_class, 0.06)
    score += _record_exact(explanation, "target_object", entry.object_state.target_object, query.target_object, 0.04)

    plan_score = _plan_score(entry, query)
    if plan_score > 0.0:
        score += plan_score
        explanation["plan_overlap"] = round(plan_score, 4)

    visual_score = _visual_score(entry, query)
    if visual_score > 0.0:
        score += visual_score
        explanation["visual_similarity"] = round(query.visual_scores.get(entry.experience_id, 0.0), 4)
        explanation["visual_score"] = round(visual_score, 4)

    semantic_score = _semantic_score(entry, query)
    if semantic_score > 0.0:
        score += semantic_score
        explanation["semantic_similarity"] = round(query.semantic_scores.get(entry.experience_id, 0.0), 4)
        explanation["semantic_score"] = round(semantic_score, 4)

    sensor_score = _sensor_score(entry, query)
    if sensor_score > 0.0:
        score += sensor_score
        explanation["sensor_evidence_score"] = round(sensor_score, 4)
        explanation["sensor_modalities"] = list(entry.sensor_evidence.modalities or [])
        explanation["sensor_evidence_summary"] = dict(entry.sensor_evidence.summary or {})

    tier_score = _tier_score(entry, query)
    if tier_score > 0.0:
        score += tier_score
        explanation["memory_tier"] = memory_tier(entry)
        explanation["ltm_score"] = round(tier_score, 4)

    if entry.source in {"real", "pseudo_real"} and not query.source:
        score += 0.04
        explanation["real_source_bonus"] = 0.04

    if query.risk_aware:
        adjustment = entry_risk_adjustment(entry)
        score += float(adjustment["score_adjustment"])
        explanation["risk_adjustment"] = adjustment

    score = round(max(score, 0.0), 4)
    if score <= 0.0:
        return None
    explanation["total"] = score
    return RetrievalMatch(entry=entry, score=score, explanation=explanation)


def _passes_hard_filters(entry: ExperienceEntry, query: RetrievalQuery) -> bool:
    filters = {
        "source": entry.source,
        "skill_namespace": entry.skill_namespace,
        "memory_role": str(entry.memory_tags.get("memory_role") or ""),
        "memory_type": str(entry.memory_tags.get("memory_type") or ""),
        "memory_partition": entry.memory_partition,
        "failure_type": str(entry.failure_taxonomy.get("failure_type") or ""),
        "critic_status": entry.critic_result.overall_status,
        "gap_type": str(entry.sim_real_gap.outcome_gap.get("type") or ""),
        "object_class": entry.object_state.object_class,
        "target_object": entry.object_state.target_object,
    }
    for name, value in filters.items():
        wanted = getattr(query, name)
        if wanted and str(value) != str(wanted):
            return False
    return True


def _exact_score(value: str, query_value: str, weight: float) -> float:
    if not query_value:
        return 0.0
    return weight if str(value) == str(query_value) else 0.0


def _record_exact(
    explanation: dict[str, Any],
    name: str,
    value: str,
    query_value: str,
    weight: float,
    *,
    mismatch_penalty: float = 0.0,
) -> float:
    if not query_value:
        return 0.0
    if str(value) == str(query_value):
        explanation[name] = weight
        return weight
    if mismatch_penalty:
        explanation[f"{name}_mismatch"] = mismatch_penalty
        return mismatch_penalty
    return 0.0


def _plan_score(entry: ExperienceEntry, query: RetrievalQuery) -> float:
    query_actions = list(query.skill_sequence)
    if not query_actions and query.plan_signature:
        query_actions = [item for item in str(query.plan_signature).split("->") if item]
    if not query_actions:
        return 0.0
    entry_actions = [item.name for item in entry.skill_sequence if item.name]
    overlap = action_lcs_ratio(query_actions, entry_actions)
    return round(0.13 * overlap, 4)


def _visual_score(entry: ExperienceEntry, query: RetrievalQuery) -> float:
    if not query.visual_scores:
        return 0.0
    similarity = max(0.0, min(float(query.visual_scores.get(entry.experience_id, 0.0)), 1.0))
    if similarity <= 0.0:
        return 0.0
    return round(max(float(query.visual_weight), 0.0) * similarity, 4)


def _semantic_score(entry: ExperienceEntry, query: RetrievalQuery) -> float:
    if not query.semantic_scores:
        return 0.0
    similarity = max(0.0, min(float(query.semantic_scores.get(entry.experience_id, 0.0)), 1.0))
    if similarity <= 0.0:
        return 0.0
    return round(max(float(query.semantic_weight), 0.0) * similarity, 4)


def _sensor_score(entry: ExperienceEntry, query: RetrievalQuery) -> float:
    modalities = {str(item) for item in entry.sensor_evidence.modalities or [] if item}
    if not modalities and not query.required_sensor_modalities and not query.preferred_sensor_modalities:
        return 0.0

    required = {str(item) for item in query.required_sensor_modalities if item}
    preferred = {str(item) for item in query.preferred_sensor_modalities if item}
    if required and not required.issubset(modalities):
        return 0.0

    score = 0.0
    if required:
        score += 0.05 * len(required)
    if preferred:
        overlap = len(preferred & modalities) / max(len(preferred), 1)
        score += 0.06 * overlap
    else:
        score += 0.03 * len(modalities)

    summary = entry.sensor_evidence.summary or {}
    if query.prefer_real_sensor_evidence and entry.source in {"real", "pseudo_real"}:
        score += 0.05 if entry.source == "real" else 0.03
    if query.wrist_force_norm_range is not None:
        low, high = query.wrist_force_norm_range
        force_norm = summary.get("max_wrist_force_norm")
        if force_norm is not None:
            force_norm = float(force_norm)
            if low <= force_norm <= high:
                score += 0.06
            elif force_norm > 0.0:
                score += 0.02
    if query.nearest_obstacle_distance_range is not None:
        low, high = query.nearest_obstacle_distance_range
        obstacle = summary.get("nearest_obstacle_distance")
        if obstacle is None:
            obstacle = entry.sensor_evidence.lidar_observation.get("nearest_obstacle_distance")
        if obstacle is not None:
            obstacle = float(obstacle)
            if low <= obstacle <= high:
                score += 0.06
            elif obstacle > 0.0:
                score += 0.02
    if modalities and entry.sensor_evidence.evidence_refs:
        score += 0.02
    return round(score * max(float(query.sensor_evidence_weight), 0.0), 4)


def _tier_score(entry: ExperienceEntry, query: RetrievalQuery) -> float:
    if memory_tier(entry) != "ltm":
        return 0.0
    return round(max(float(query.ltm_weight), 0.0), 4)


def matches_to_tuples(matches: list[RetrievalMatch]) -> list[tuple[ExperienceEntry, float]]:
    return [(match.entry, match.score) for match in matches]
