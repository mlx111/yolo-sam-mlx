"""Condition-isolated memory v3+ for UR5e anomaly recovery.

This module keeps the v3 condition_id hard boundary, while restoring the most
useful episode-level fields from the earlier experience library.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.scoring import entry_risk_adjustment


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class MemoryV3Result:
    recovery_success: bool = False
    task_success: bool = False
    failure_reason: str = ""
    success: bool | None = None
    z_change: float = 0.0
    time_cost: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return bool(self.recovery_success)


@dataclass
class AnomalyInfo:
    type: str = ""
    injection_step: str = ""
    description: str = ""
    condition_id: str = ""
    scenario_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneInfo:
    objects: list[str] = field(default_factory=list)
    camera_view: str = ""
    scene_name: str = ""
    xml_path: str = ""


@dataclass
class TaskInfo:
    name: str = ""
    stage: str = ""
    object_class: str = ""
    scene_name: str = ""
    condition: str = ""
    trial_index: int = 0
    condition_id: str = ""
    scenario_id: str = ""
    condition_name: str = ""
    strategy_family: str = ""


@dataclass
class PerceptionSnapshot:
    object_pos: list[float] | None = None
    confidence: float = 0.0
    detection_ok: bool = False
    mask_nonzero: int = 0


@dataclass
class PerceptionInfo:
    before_anomaly: PerceptionSnapshot = field(default_factory=PerceptionSnapshot)
    after_anomaly: PerceptionSnapshot | None = None
    detection_method: str = ""


@dataclass
class ReconstructionArtifacts:
    reconstruction_type: str = ""
    object_positions: dict[str, list[float]] = field(default_factory=dict)
    object_quats: dict[str, list[float]] = field(default_factory=dict)
    camera_poses: dict[str, dict[str, Any]] = field(default_factory=dict)
    scene_out: str = ""
    scene_out_refined: str = ""
    runtime_pose_calibration_path: str = ""
    sim_intrinsics_json: str = ""
    mesh_quats: dict[str, list[float]] = field(default_factory=dict)
    support_height_adjustments: dict[str, Any] = field(default_factory=dict)
    reconstruction_signature: str = ""
    virtual_scene_built: bool = False
    recovery_pos: list[float] = field(default_factory=list)
    condition: str = ""


@dataclass
class RecoveryStep:
    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    condition: str = ""
    steps: list[RecoveryStep] = field(default_factory=list)


@dataclass
class ExecutionFeedback:
    recovery_success: bool = False
    task_success: bool = False
    apple_z_after_recovery: float | None = None
    execution_deviation: dict[str, Any] = field(default_factory=dict)
    contact_after_close: dict[str, Any] = field(default_factory=dict)
    contact_after_lift: dict[str, Any] = field(default_factory=dict)
    time_costs: dict[str, float] = field(default_factory=dict)
    failure_reason: str = ""
    observed_pos: list[float] | None = None
    recovery_success_criteria: dict[str, Any] = field(default_factory=dict)
    task_success_criteria: dict[str, Any] = field(default_factory=dict)
    virtual_validation_success: bool | None = None


@dataclass
class SensorSummaryInfo:
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    end_effector_pose: dict[str, Any] = field(default_factory=dict)
    gripper_state: dict[str, Any] = field(default_factory=dict)
    contact_state: dict[str, Any] = field(default_factory=dict)
    force_torque: dict[str, Any] = field(default_factory=dict)
    timestamps: dict[str, Any] = field(default_factory=dict)
    sensor_modalities: list[str] = field(default_factory=list)
    raw_refs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeSlice:
    slice_id: str = ""
    stage: str = ""
    description: str = ""
    timestamp: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyframeInfo:
    stage: str = ""
    image_path: str = ""
    description: str = ""
    used_for_retrieval: bool = False


@dataclass
class MemoryGateInfo:
    anomaly_score: float = 0.0
    failure_score: float = 0.0
    sim_real_gap_score: float = 0.0
    recovery_utility_score: float = 0.0
    surprise_score: float = 0.0
    write_score: float = 0.0
    write_decision: str = ""
    trigger_events: list[str] = field(default_factory=list)
    explanation: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimRealGapInfo:
    gap_id: str = ""
    gap_score: float = 0.0
    pose_gap: dict[str, Any] = field(default_factory=dict)
    contact_gap: dict[str, Any] = field(default_factory=dict)
    outcome_gap: dict[str, Any] = field(default_factory=dict)
    perception_gap: dict[str, Any] = field(default_factory=dict)
    actuation_gap: dict[str, Any] = field(default_factory=dict)
    robot_state_gap: dict[str, Any] = field(default_factory=dict)
    scene_reconstruction_gap: dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimRealPairInfo:
    pair_id: str = ""
    sim_experience_id: str = ""
    real_experience_id: str = ""
    paired_by: str = ""
    pair_score: float = 0.0
    gap_score: float = 0.0
    validation_status: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SandboxCalibrationInfo:
    calibration_id: str = ""
    source_gap_ids: list[str] = field(default_factory=list)
    object_pose_bias: list[float] = field(default_factory=list)
    gripper_delay_bias: float = 0.0
    slip_risk_bias: float = 0.0
    contact_success_bias: float = 0.0
    perception_noise_bias: list[float] = field(default_factory=list)
    applied_to_candidate: bool = False
    calibration_confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticResultInfo:
    overall_status: str = ""
    critic_risk_score: float = 0.0
    collision: dict[str, Any] = field(default_factory=dict)
    joint: dict[str, Any] = field(default_factory=dict)
    gripper_contact: dict[str, Any] = field(default_factory=dict)
    end_effector_pose: dict[str, Any] = field(default_factory=dict)
    rule_flags: list[dict[str, Any]] = field(default_factory=list)
    feedback_for_rewrite: str = ""


@dataclass
class RealEpisodeRef:
    raw_episode_id: str = ""
    hdf5_path: str = ""
    video_dir: str = ""
    keyframe_dir: str = ""
    robot_log_path: str = ""
    time_range: list[float] = field(default_factory=list)
    sensor_modalities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryV3Entry:
    schema_version: str = "memory_v3_plus"
    experience_id: str = ""
    episode_type: str = "anomaly_recovery"
    condition_id: str = ""
    scenario_id: str = ""
    available_actions: list[str] = field(default_factory=list)
    skill_sequence: list[dict[str, Any]] = field(default_factory=list)
    result: MemoryV3Result = field(default_factory=MemoryV3Result)
    status: str = "success"
    source: str = "simulation"
    domain: str = ""
    source_run_id: str = ""
    source_trial_id: str = ""
    confidence_score: float | None = None
    memory_partition: str = ""
    created_at: str = ""
    updated_at: str = ""
    summary: str = ""
    text_summary: str = ""
    text_embedding_id: str = ""
    embeddings: dict[str, Any] = field(default_factory=dict)
    anomaly: AnomalyInfo = field(default_factory=AnomalyInfo)
    scene: SceneInfo = field(default_factory=SceneInfo)
    task: TaskInfo = field(default_factory=TaskInfo)
    perception: PerceptionInfo = field(default_factory=PerceptionInfo)
    reconstruction_artifacts: ReconstructionArtifacts = field(default_factory=ReconstructionArtifacts)
    recovery_plan: RecoveryPlan = field(default_factory=RecoveryPlan)
    execution_feedback: ExecutionFeedback = field(default_factory=ExecutionFeedback)
    sensor_summary: SensorSummaryInfo = field(default_factory=SensorSummaryInfo)
    key_slices: list[EpisodeSlice] = field(default_factory=list)
    keyframes: list[KeyframeInfo] = field(default_factory=list)
    anomaly_state: dict[str, Any] = field(default_factory=dict)
    retrieval_key: dict[str, Any] = field(default_factory=dict)
    failure_taxonomy: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "simulation_only"
    validation_source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_evidence: dict[str, Any] = field(default_factory=dict)
    promotion_history: list[dict[str, Any]] = field(default_factory=list)
    memory_tier: str = "stm"
    memory_gate: MemoryGateInfo = field(default_factory=MemoryGateInfo)
    sim_real_gap: SimRealGapInfo = field(default_factory=SimRealGapInfo)
    sim_real_pair: SimRealPairInfo = field(default_factory=SimRealPairInfo)
    sandbox_calibration: SandboxCalibrationInfo = field(default_factory=SandboxCalibrationInfo)
    critic_result: CriticResultInfo = field(default_factory=CriticResultInfo)
    real_episode_ref: RealEpisodeRef = field(default_factory=RealEpisodeRef)
    memory_tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experience_id:
            self.experience_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = _utc_now()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.scenario_id and self.condition_id:
            self.scenario_id = self.condition_id.split("-", 1)[0]
        if isinstance(self.result, dict):
            self.result = _coerce_dataclass(MemoryV3Result, self.result)
        if self.result.success is None:
            self.result.success = bool(self.result.recovery_success)
        self._coerce_episode_fields()
        self.status = "success" if self.result.recovery_success else "failure"
        self.domain = self.domain or self.source or "simulation"
        self._promote_metadata_defaults()
        if not self.validation_status:
            self.validation_status = "simulation_only" if self.result.success else "failed"
        if not self.memory_partition:
            self.memory_partition = infer_memory_partition(self)
        if not self.source_trial_id and isinstance(self.validation_evidence, dict):
            self.source_trial_id = str(self.validation_evidence.get("trial_id") or "")
        if not self.anomaly_state:
            self.anomaly_state = {
                "condition_id": self.condition_id,
                "scenario_id": self.scenario_id,
                "task_stage": self.task.stage,
            }
        if not self.retrieval_key:
            self.retrieval_key = build_retrieval_key(self)
        else:
            self.retrieval_key.setdefault("condition_id", self.condition_id)
            self.retrieval_key.setdefault("scenario_id", self.scenario_id)
            self.retrieval_key.setdefault("plan_signature", self.plan_signature)
        if not self.failure_taxonomy and self.result.failure_reason:
            self.failure_taxonomy = {"failure_type": self.result.failure_reason}
        if not self.text_summary:
            self.text_summary = build_text_summary(self)
        if self.confidence_score is None:
            self.confidence_score = default_confidence_score(self)

    def _coerce_episode_fields(self) -> None:
        self.anomaly = _coerce_dataclass(AnomalyInfo, self.anomaly)
        self.scene = _coerce_dataclass(SceneInfo, self.scene)
        self.task = _coerce_dataclass(TaskInfo, self.task)
        self.perception = _coerce_perception(self.perception)
        self.reconstruction_artifacts = _coerce_dataclass(ReconstructionArtifacts, self.reconstruction_artifacts)
        self.recovery_plan = _coerce_recovery_plan(self.recovery_plan)
        self.execution_feedback = _coerce_dataclass(ExecutionFeedback, self.execution_feedback)
        self.sensor_summary = _coerce_dataclass(SensorSummaryInfo, self.sensor_summary)
        self.key_slices = [_coerce_dataclass(EpisodeSlice, item) for item in self.key_slices or []]
        self.keyframes = [_coerce_dataclass(KeyframeInfo, item) for item in self.keyframes or []]
        self.memory_gate = _coerce_dataclass(MemoryGateInfo, self.memory_gate)
        self.sim_real_gap = _coerce_dataclass(SimRealGapInfo, self.sim_real_gap)
        self.sim_real_pair = _coerce_dataclass(SimRealPairInfo, self.sim_real_pair)
        self.sandbox_calibration = _coerce_dataclass(SandboxCalibrationInfo, self.sandbox_calibration)
        self.critic_result = _coerce_dataclass(CriticResultInfo, self.critic_result)
        self.real_episode_ref = _coerce_dataclass(RealEpisodeRef, self.real_episode_ref)
        self.memory_tags = self.memory_tags if isinstance(self.memory_tags, dict) else {}

    def _promote_metadata_defaults(self) -> None:
        metadata = self.metadata if isinstance(self.metadata, dict) else {}
        if not self.key_slices:
            self.key_slices = [_coerce_dataclass(EpisodeSlice, item) for item in metadata.get("key_slices") or []]
        if not self.keyframes:
            self.keyframes = [_coerce_dataclass(KeyframeInfo, item) for item in metadata.get("keyframes") or []]
        if not _dataclass_has_values(self.reconstruction_artifacts):
            self.reconstruction_artifacts = _coerce_dataclass(
                ReconstructionArtifacts,
                metadata.get("reconstruction_artifacts") or {},
            )
        if not self.recovery_plan.steps and metadata.get("recovery_plan"):
            self.recovery_plan = _coerce_recovery_plan(metadata.get("recovery_plan") or {})
        if not self.failure_taxonomy:
            self.failure_taxonomy = dict(metadata.get("failure_taxonomy") or {})
        if not _dataclass_has_values(self.anomaly):
            self.anomaly = AnomalyInfo(
                type=str(
                    metadata.get("condition_name")
                    or metadata.get("failure_type")
                    or self.result.failure_reason
                    or self.condition_id
                    or ""
                ),
                injection_step=str(metadata.get("injection_stage") or ""),
                description=str(metadata.get("condition_name") or self.summary or ""),
                condition_id=self.condition_id,
                scenario_id=self.scenario_id,
                params=metadata.get("injection_params") or {},
            )
        if not _dataclass_has_values(self.scene):
            self.scene = SceneInfo(
                objects=["apple", "pear", "plate"],
                camera_view="sim_camera",
                scene_name="apple_pear_runtime_refined",
                xml_path=str(metadata.get("scene_xml") or ""),
            )
        if not _dataclass_has_values(self.task):
            self.task = TaskInfo(
                name="ur5e_anomaly_recovery",
                stage=str(metadata.get("task_stage", "") or ""),
                object_class="apple",
                scene_name="apple_pear_runtime_refined",
                condition=str(metadata.get("condition", "") or ""),
                condition_name=str(metadata.get("condition_name", "") or ""),
                strategy_family=str(metadata.get("strategy_family", "") or ""),
                condition_id=self.condition_id,
                scenario_id=self.scenario_id,
            )
        if not _dataclass_has_values(self.perception):
            self.perception = PerceptionInfo(
                before_anomaly=_coerce_dataclass(PerceptionSnapshot, metadata.get("perception_before") or {}),
                after_anomaly=_coerce_dataclass(PerceptionSnapshot, metadata.get("perception_after") or {})
                if metadata.get("perception_after")
                else None,
            )
        if not _dataclass_has_values(self.execution_feedback):
            self.execution_feedback = ExecutionFeedback(
                recovery_success=bool(self.result.recovery_success),
                task_success=bool(self.result.task_success),
                failure_reason=self.result.failure_reason,
                apple_z_after_recovery=metadata.get("apple_z_after_recovery"),
                contact_after_close=metadata.get("contact_after_close", {}) or {},
                contact_after_lift=metadata.get("contact_after_lift", {}) or {},
                observed_pos=metadata.get("observed_pos"),
                recovery_success_criteria=metadata.get("recovery_success_criteria", {}) or {},
                task_success_criteria=metadata.get("task_success_criteria", {}) or {},
                virtual_validation_success=metadata.get("virtual_validation_success"),
            )
        if not _dataclass_has_values(self.sensor_summary):
            self.sensor_summary = _coerce_dataclass(SensorSummaryInfo, metadata.get("sensor_summary") or {})
        if not _dataclass_has_values(self.memory_gate):
            self.memory_gate = _coerce_dataclass(MemoryGateInfo, metadata.get("memory_gate") or {})
        if not _dataclass_has_values(self.sim_real_gap):
            self.sim_real_gap = _coerce_dataclass(SimRealGapInfo, metadata.get("sim_real_gap") or {})
        if not _dataclass_has_values(self.sim_real_pair):
            self.sim_real_pair = _coerce_dataclass(SimRealPairInfo, metadata.get("sim_real_pair") or {})
        if not _dataclass_has_values(self.sandbox_calibration):
            self.sandbox_calibration = _coerce_dataclass(
                SandboxCalibrationInfo,
                metadata.get("sandbox_calibration") or {},
            )
        if not _dataclass_has_values(self.critic_result):
            self.critic_result = _coerce_dataclass(CriticResultInfo, metadata.get("critic_result") or {})
        if not _dataclass_has_values(self.real_episode_ref):
            self.real_episode_ref = _coerce_dataclass(RealEpisodeRef, metadata.get("real_episode_ref") or {})
        if not self.memory_tags:
            self.memory_tags = dict(metadata.get("memory_tags") or {})

    @property
    def plan_signature(self) -> str:
        return canonical_action_signature_from_steps(self.skill_sequence)

    def get_partition(self) -> str:
        return self.memory_partition


class MemoryV3Library:
    def __init__(self, stm_capacity: int = 30) -> None:
        self._entries: list[MemoryV3Entry] = []
        self._last_score_explanations: dict[str, dict[str, Any]] = {}
        self.stm_capacity = stm_capacity
        self._consolidation_counters: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def stm_entries(self) -> list[MemoryV3Entry]:
        return [e for e in self._entries if e.memory_tier == "stm"]

    @property
    def ltm_entries(self) -> list[MemoryV3Entry]:
        return [e for e in self._entries if e.memory_tier == "ltm"]

    def upsert(self, entry: MemoryV3Entry) -> None:
        for idx, existing in enumerate(self._entries):
            if existing.experience_id == entry.experience_id:
                entry.updated_at = _utc_now()
                # Preserve tier on update; tier is only changed via promote/consolidate
                entry.memory_tier = existing.memory_tier
                self._entries[idx] = entry
                return
        if not entry.memory_tier:
            entry.memory_tier = "stm"
        self._entries.append(entry)

    def add(self, entry: MemoryV3Entry) -> None:
        self.upsert(entry)

    def query(
        self,
        *,
        scenario_id: str,
        condition_id: str = "",
        available_actions: set[str] | list[str],
        retrieval_key: dict[str, Any] | None = None,
        anomaly_state: dict[str, Any] | None = None,
        task_stage: str = "",
        text_summary: str = "",
        top_k: int = 5,
        include_failed: bool = True,
        diversity_lambda: float = 0.0,
        critic_prefilter: bool = False,
        visual_index: Any = None,
        visual_context: list[str] | None = None,
        visual_base_dir: str | Path | None = None,
        gap_aware: bool = False,
        risk_aware: bool = False,
        **_: Any,
    ) -> list[tuple[MemoryV3Entry, float]]:
        if not scenario_id:
            return []
        allowed = set(available_actions)

        # Optionally search visual index once for the whole query
        visual_scores: dict[str, float] = {}
        if visual_index is not None and visual_context:
            try:
                results = visual_index.search(visual_context, top_k=top_k * 2)
                visual_scores = dict(results)
            except Exception as exc:
                print(f"  [WARN] 视觉检索失败: {exc}")

        matches: list[tuple[MemoryV3Entry, float]] = []
        for entry in self._entries:
            if entry.scenario_id != scenario_id:
                continue
            actions = {str(step.get("action", "")) for step in entry.skill_sequence if isinstance(step, dict)}
            if not actions.issubset(allowed):
                continue
            if not include_failed and not entry.result.recovery_success:
                continue
            score, explanation = self._score(
                entry,
                actions,
                allowed,
                query_condition_id=condition_id,
                query_retrieval_key=retrieval_key or {},
                query_anomaly_state=anomaly_state or {},
                query_task_stage=task_stage,
                query_text_summary=text_summary,
            )
            # Visual similarity bonus (0.10 weight in the same scale as structured terms)
            visual_sim = visual_scores.get(entry.experience_id, 0.0)
            visual_boost = 0.10 * visual_sim if visual_sim > 0 else 0.0
            # Tier-aware boost: LTM entries get +0.2 stability bonus
            tier_boost = 0.2 if entry.memory_tier == "ltm" else 0.0
            dual_source_adjustment = entry_risk_adjustment(entry) if (gap_aware or risk_aware) else {
                "gap_uncertainty": 0.0,
                "critic_risk": 0.0,
                "real_validation_bonus": 0.0,
                "real_success_bonus": 0.0,
                "paired_bonus": 0.0,
                "gap_score_penalty": 0.0,
                "gap_uncertainty_penalty": 0.0,
                "sim_real_failure_penalty": 0.0,
                "critic_penalty": 0.0,
                "failure_penalty": 0.0,
                "risk_penalty": 0.0,
                "trust_bonus": 0.0,
                "score_adjustment": 0.0,
            }
            score_adjustment = float(dual_source_adjustment["score_adjustment"]) if (gap_aware or risk_aware) else 0.0
            boosted_score = score + tier_boost + visual_boost + score_adjustment
            self._last_score_explanations[entry.experience_id] = {
                **explanation,
                "scenario_id_match": 1.0,
                "available_action_compatible": 1.0,
                "memory_tier": entry.memory_tier,
                "tier_boost": tier_boost,
                "visual_similarity": visual_sim,
                "visual_boost": visual_boost,
                "gap_aware": bool(gap_aware),
                "risk_aware": bool(risk_aware),
                "dual_source_adjustment": dual_source_adjustment,
                "gap_uncertainty": dual_source_adjustment["gap_uncertainty"],
                "critic_risk": dual_source_adjustment["critic_risk"],
                "real_validation_bonus": dual_source_adjustment["real_validation_bonus"],
                "real_success_bonus": dual_source_adjustment.get("real_success_bonus", 0.0),
                "paired_bonus": dual_source_adjustment.get("paired_bonus", 0.0),
                "gap_score_penalty": dual_source_adjustment.get("gap_score_penalty", 0.0),
                "gap_uncertainty_penalty": dual_source_adjustment.get("gap_uncertainty_penalty", 0.0),
                "sim_real_failure_penalty": dual_source_adjustment.get("sim_real_failure_penalty", 0.0),
                "critic_penalty": dual_source_adjustment.get("critic_penalty", 0.0),
                "risk_penalty": dual_source_adjustment["risk_penalty"],
                "trust_bonus": dual_source_adjustment["trust_bonus"],
                "score_adjustment": score_adjustment,
                "raw_score": score,
            }
            matches.append((entry, boosted_score))
        matches.sort(key=lambda item: (-item[1], item[0].created_at))
        if critic_prefilter:
            matches = _critic_prefilter(matches)
        if diversity_lambda > 0.0 and len(matches) > top_k:
            matches = _mmr_select(matches, top_k, diversity_lambda)
        # Track retrieval count for consolidation decisions
        for entry, _ in matches:
            eid = entry.experience_id
            self._consolidation_counters[eid] = self._consolidation_counters.get(eid, 0) + 1
        return matches[:top_k]

    def _score(
        self,
        entry: MemoryV3Entry,
        actions: set[str],
        allowed: set[str],
        *,
        query_condition_id: str = "",
        query_retrieval_key: dict[str, Any],
        query_anomaly_state: dict[str, Any],
        query_task_stage: str,
        query_text_summary: str,
    ) -> tuple[float, dict[str, Any]]:
        action_coverage = len(actions) / max(len(allowed), 1)
        validation_score = validation_status_score(entry.validation_status)
        result_success = float(entry.result.recovery_success)
        retrieval_key_similarity = dict_similarity(query_retrieval_key, entry.retrieval_key)
        anomaly_state_similarity = dict_similarity(query_anomaly_state, entry.anomaly_state)
        entry_task_stage = str(entry.task.stage or "")
        task_stage_match = 1.0 if query_task_stage and entry_task_stage == query_task_stage else 0.0
        condition_id_match = 1.0 if query_condition_id and entry.condition_id == query_condition_id else 0.0
        text_summary_similarity = token_jaccard(query_text_summary, entry.text_summary) if query_text_summary else 0.0
        action_coverage_score = min(action_coverage, 1.0)
        structured_similarity = (
            0.20 * validation_score
            + 0.15 * result_success
            + 0.15 * retrieval_key_similarity
            + 0.15 * condition_id_match
            + 0.15 * anomaly_state_similarity
            + 0.10 * task_stage_match
            + 0.05 * action_coverage_score
        )
        text_score = 0.05 * text_summary_similarity
        final_score = 1.0 + structured_similarity + text_score
        return final_score, {
            "final_score": final_score,
            "structured_similarity": structured_similarity,
            "validation_score": validation_score,
            "result_success": result_success,
            "retrieval_key_similarity": retrieval_key_similarity,
            "anomaly_state_similarity": anomaly_state_similarity,
            "task_stage_match": task_stage_match,
            "action_coverage": action_coverage_score,
            "text_summary_similarity": text_summary_similarity,
            "text_score": text_score,
        }

    def get_last_score_explanation(self, experience_id: str) -> dict[str, Any]:
        return self._last_score_explanations.get(experience_id, {})

    # ── STM/LTM consolidation ────────────────────────────────────────

    MIN_RETRIEVAL_FOR_PROMOTION = 3
    MIN_ENTRIES_FOR_PROMOTION = 5
    EVICT_BATCH_SIZE = 5

    def promote_to_ltm(self, experience_id: str) -> bool:
        """Manually promote one entry from STM to LTM."""
        for entry in self._entries:
            if entry.experience_id == experience_id and entry.memory_tier == "stm":
                entry.memory_tier = "ltm"
                entry.promotion_history.append({
                    "action": "promote_to_ltm",
                    "reason": "manual",
                    "timestamp": _utc_now(),
                })
                return True
        return False

    def _should_promote(self, entry: MemoryV3Entry) -> bool:
        """Check if a STM entry qualifies for promotion to LTM."""
        if entry.memory_tier != "stm":
            return False
        if len(self._entries) < self.MIN_ENTRIES_FOR_PROMOTION:
            return False
        # Criteria 1: validated success
        if entry.validation_status in ("simulation_validated", "real_validated", "real_executed"):
            if entry.result.recovery_success:
                return True
        # Criteria 2: frequently retrieved
        retrieval_count = self._consolidation_counters.get(entry.experience_id, 0)
        if entry.result.recovery_success and retrieval_count >= self.MIN_RETRIEVAL_FOR_PROMOTION:
            return True
        return False

    def consolidate(self) -> dict[str, Any]:
        """Consolidate STM: promote qualified entries, evict low-value ones.

        Call after each upsert().  Returns a summary dict for logging/metrics.
        """
        stm = self.stm_entries
        if len(stm) <= self.stm_capacity:
            return {"action": "none", "reason": "within_capacity", "stm_count": len(stm)}

        report: dict[str, Any] = {
            "action": "consolidate",
            "stm_count_before": len(stm),
            "stm_capacity": self.stm_capacity,
            "promoted": [],
            "evicted": [],
        }

        # Phase 1: promote qualified entries
        for entry in stm:
            if self._should_promote(entry):
                entry.memory_tier = "ltm"
                entry.promotion_history.append({
                    "action": "promote_to_ltm",
                    "reason": "auto_consolidate",
                    "retrieval_count": self._consolidation_counters.get(entry.experience_id, 0),
                    "timestamp": _utc_now(),
                })
                report["promoted"].append(entry.experience_id)

        # Phase 2: evict lowest-value STM entries if still over capacity
        remaining_stm = self.stm_entries
        over = len(remaining_stm) - self.stm_capacity
        if over > 0:
            def _eviction_key(e: MemoryV3Entry) -> tuple:
                protection = 0 if e.result.recovery_success else 1
                retrieval = -(self._consolidation_counters.get(e.experience_id, 0))
                created = e.created_at or ""
                return (protection, retrieval, created)

            candidates = sorted(remaining_stm, key=_eviction_key)
            to_evict = candidates[:min(over, self.EVICT_BATCH_SIZE)]
            evict_ids = {e.experience_id for e in to_evict}
            self._entries = [e for e in self._entries if e.experience_id not in evict_ids]
            for eid in evict_ids:
                self._consolidation_counters.pop(eid, None)
            report["evicted"] = list(evict_ids)

        report["stm_count_after"] = len(self.stm_entries)
        report["ltm_count"] = len(self.ltm_entries)
        return report

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "memory_v3_plus",
            "stm_capacity": self.stm_capacity,
            "consolidation_counters": dict(self._consolidation_counters),
            "entries": [entry_to_dict(entry) for entry in self._entries],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "MemoryV3Library":
        lib = cls()
        path = Path(path)
        if not path.exists():
            return lib
        payload = json.loads(path.read_text())
        if payload.get("schema_version") not in {"memory_v3", "memory_v3_plus"}:
            return lib
        lib.stm_capacity = int(payload.get("stm_capacity") or 30)
        lib._consolidation_counters = {
            str(k): int(v) for k, v in (payload.get("consolidation_counters") or {}).items()
        }
        for raw in payload.get("entries", []):
            lib.upsert(entry_from_dict(raw))
        return lib


def entry_to_dict(entry: MemoryV3Entry) -> dict[str, Any]:
    data = asdict(entry)
    data["result"] = asdict(entry.result)
    return data


def entry_from_dict(raw: dict[str, Any]) -> MemoryV3Entry:
    allowed = set(MemoryV3Entry.__dataclass_fields__.keys())
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return MemoryV3Entry(**filtered)


def _coerce_dataclass(cls: type, value: Any):
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        return cls()
    allowed = set(cls.__dataclass_fields__.keys())
    return cls(**{key: value for key, value in value.items() if key in allowed})


def _coerce_perception(value: Any) -> PerceptionInfo:
    if isinstance(value, PerceptionInfo):
        return value
    if not isinstance(value, dict):
        return PerceptionInfo()
    before = _coerce_dataclass(PerceptionSnapshot, value.get("before_anomaly") or {})
    raw_after = value.get("after_anomaly")
    after = _coerce_dataclass(PerceptionSnapshot, raw_after) if isinstance(raw_after, dict) else None
    return PerceptionInfo(
        before_anomaly=before,
        after_anomaly=after,
        detection_method=str(value.get("detection_method") or ""),
    )


def _coerce_recovery_plan(value: Any) -> RecoveryPlan:
    if isinstance(value, RecoveryPlan):
        return value
    if not isinstance(value, dict):
        return RecoveryPlan()
    steps: list[RecoveryStep] = []
    for step in value.get("steps", []) or []:
        if isinstance(step, RecoveryStep):
            steps.append(step)
        elif isinstance(step, dict):
            if "type" in step:
                steps.append(_coerce_dataclass(RecoveryStep, step))
            elif "action" in step:
                steps.append(RecoveryStep(type=str(step.get("action") or ""), params=step.get("parameters") or {}))
    return RecoveryPlan(condition=str(value.get("condition") or ""), steps=steps)


def _asdict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {}


def _dataclass_has_values(value: Any) -> bool:
    return _has_meaningful_value(_asdict(value))


def _has_meaningful_value(value: Any) -> bool:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, dict):
        return any(_has_meaningful_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_meaningful_value(item) for item in value)
    return value not in (None, "", False, 0, 0.0)


def infer_memory_partition(entry: MemoryV3Entry) -> str:
    validation_status = (entry.validation_status or "").lower()
    if validation_status in {"failed", "failure"} or not entry.result.recovery_success:
        return "failed_memory"
    if validation_status in {"real_executed", "real_validated"}:
        return "real_memory"
    if validation_status == "simulation_validated":
        return "validated_memory"
    return "simulation_memory"


def validation_status_score(status: str) -> float:
    status = (status or "").lower()
    if status == "real_validated":
        return 1.0
    if status == "real_executed":
        return 0.9
    if status == "simulation_validated":
        return 0.75
    if status == "simulation_only":
        return 0.5
    if status in {"failed", "failure"}:
        return 0.1
    return 0.35


def dict_similarity(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    if not query:
        return 0.0
    comparable = [key for key, value in query.items() if value not in (None, "", [], {})]
    if not comparable:
        return 0.0
    matches = 0.0
    for key in comparable:
        if candidate.get(key) == query.get(key):
            matches += 1.0
    return matches / max(len(comparable), 1)


def uncertainty_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric >= 0.67:
        return "high"
    if numeric >= 0.34:
        return "medium"
    if numeric > 0.0:
        return "low"
    return ""


def critic_flag_names(critic_result: Any) -> list[str]:
    critic = _asdict(critic_result)
    names: list[str] = []
    for flag in critic.get("rule_flags") or []:
        if isinstance(flag, dict) and flag.get("rule"):
            names.append(str(flag["rule"]))
    return names


def token_jaccard(a: str, b: str) -> float:
    left = {token for token in str(a).replace(";", " ").replace(",", " ").split() if token}
    right = {token for token in str(b).replace(";", " ").replace(",", " ").split() if token}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_retrieval_key(entry: MemoryV3Entry) -> dict[str, Any]:
    feedback = _asdict(entry.execution_feedback)
    criteria = feedback.get("recovery_success_criteria") or {}
    contact_after_close = feedback.get("contact_after_close") or {}
    contact_after_lift = feedback.get("contact_after_lift") or {}
    gap = _asdict(entry.sim_real_gap)
    outcome_gap = gap.get("outcome_gap") if isinstance(gap.get("outcome_gap"), dict) else {}
    pair = _asdict(entry.sim_real_pair)
    critic = _asdict(entry.critic_result)
    memory_tags = entry.memory_tags if isinstance(entry.memory_tags, dict) else {}
    return {
        "condition_id": entry.condition_id,
        "scenario_id": entry.scenario_id,
        "task_stage": entry.task.stage,
        "plan_signature": entry.plan_signature,
        "contact_pattern": contact_pattern(contact_after_close, contact_after_lift),
        "lift_success": bool(criteria.get("success", entry.result.recovery_success)),
        "failure_type": (entry.failure_taxonomy or {}).get("failure_type", entry.result.failure_reason),
        "gap_type": str(outcome_gap.get("type") or ""),
        "gap_uncertainty_bucket": uncertainty_bucket(gap.get("uncertainty")),
        "critic_status": str(critic.get("overall_status") or ""),
        "critic_flags": critic_flag_names(entry.critic_result),
        "memory_type": str(memory_tags.get("memory_type") or ""),
        "memory_role": str(memory_tags.get("memory_role") or ""),
        "real_validated": entry.validation_status in {"real_executed", "real_validated"},
        "pair_status": str(pair.get("validation_status") or ""),
    }


def contact_pattern(contact_after_close: Any, contact_after_lift: Any) -> str:
    def has_contact(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return bool(value.get("left_contact") or value.get("right_contact") or value.get("contact"))

    close = has_contact(contact_after_close)
    lift = has_contact(contact_after_lift)
    if close and lift:
        return "contact_close_and_lift"
    if close:
        return "contact_after_close_only"
    if lift:
        return "contact_after_lift_only"
    return "no_contact"


def build_text_summary(entry: MemoryV3Entry) -> str:
    bits = [
        entry.summary,
        f"condition_id={entry.condition_id}",
        f"scenario_id={entry.scenario_id}",
        f"status={entry.status}",
        f"validation_status={entry.validation_status}",
        f"skills={entry.plan_signature}",
    ]
    if entry.result.failure_reason:
        bits.append(f"failure_reason={entry.result.failure_reason}")
    # Append success criteria evidence for text retrieval distinguishability
    ef = entry.execution_feedback
    if ef and hasattr(ef, "__dataclass_fields__"):
        criteria = getattr(ef, "recovery_success_criteria", None) or {}
    else:
        criteria = {}
    if isinstance(criteria, dict) and criteria.get("type"):
        bits.append(
            f"criteria={criteria.get('type')}, "
            f"lift={criteria.get('lift_from_table')}, "
            f"pinch={criteria.get('pinch_distance')}"
        )
    return "; ".join(str(bit) for bit in bits if bit)


def default_confidence_score(entry: MemoryV3Entry) -> float:
    if entry.validation_status in {"real_validated", "simulation_validated"} and entry.result.recovery_success:
        return 0.85
    if entry.result.recovery_success:
        return 0.7
    return 0.35


def _parse_action_sequence(entry: MemoryV3Entry | Any) -> list[str]:
    """Extract ordered action+key-param sequence from plan_signature.

    Each element is like "gripper-action:state=1" or "approach-grasp",
    capturing both action type and key parameters so similarity is
    sensitive to both order and parameter choices.
    """
    try:
        payload = json.loads(entry.plan_signature)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    seq: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", ""))
        if not action:
            continue
        # Append key params to distinguish e.g. gripper-action:state=1 vs state=0
        parts = [action]
        if action == "gripper-action" and "state" in item:
            parts.append(f"state={item['state']}")
        seq.append(":".join(parts))
    return seq


def _action_sequence_similarity(e1: MemoryV3Entry, e2: MemoryV3Entry) -> float:
    """Dice coefficient on action bigrams, sensitive to order and key params.

    Returns a value in [0, 1] where 1 means both entries have the exact
    same ordered bigram set (actions + key parameters).
    """
    seq1 = _parse_action_sequence(e1)
    seq2 = _parse_action_sequence(e2)
    if not seq1 or not seq2:
        return 0.0
    bigrams1 = set(zip(seq1[:-1], seq1[1:])) if len(seq1) > 1 else {(seq1[0],)}
    bigrams2 = set(zip(seq2[:-1], seq2[1:])) if len(seq2) > 1 else {(seq2[0],)}
    if not bigrams1 or not bigrams2:
        return 0.0
    return 2.0 * len(bigrams1 & bigrams2) / (len(bigrams1) + len(bigrams2))


def _mmr_select(
    entries_with_scores: list[tuple[MemoryV3Entry, float]],
    top_k: int,
    diversity_lambda: float,
) -> list[tuple[MemoryV3Entry, float]]:
    """Greedy MMR (Maximum Marginal Relevance) selection.

    Picks the highest-scored entry first, then iteratively selects
    the next entry that maximises:

        score(i) - diversity_lambda * max_{j in selected} similarity(i, j)

    The first entry is always the raw top scorer so the result is
    anchored on the best match.
    """
    if not entries_with_scores:
        return []

    selected: list[tuple[MemoryV3Entry, float]] = [entries_with_scores[0]]
    pool = list(entries_with_scores[1:])

    while len(selected) < top_k and pool:
        best_idx = -1
        best_score = -float("inf")
        for i, (entry, score) in enumerate(pool):
            max_sim = max(
                _action_sequence_similarity(entry, sel_entry)
                for sel_entry, _ in selected
            )
            mmr = score - diversity_lambda * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        if best_idx < 0:
            break
        selected.append(pool.pop(best_idx))

    return selected


def _critic_prefilter(
    matches: list[tuple[MemoryV3Entry, float]],
) -> list[tuple[MemoryV3Entry, float]]:
    """Prefilter matched entries to improve retrieval diversity.

    1. **Failure-type dedup**: same ``failure_type`` keeps only the 2 most
       recent entries (by ``created_at``).  This prevents a single common
       failure mode from overwhelming the candidate pool.
    2. **Cross-condition balance**: if after dedup all survivors share the
       same ``condition_id``, attempt to restore one entry from a different
       condition if any were removed.
    """
    if not matches:
        return []

    # --- 1. Dedup by cluster_id (fallback to failure_type) -------------------------
    from collections import defaultdict

    def _dedup_key(entry: MemoryV3Entry) -> str:
        ft = entry.failure_taxonomy or {}
        return str(ft.get("cluster_id") or ft.get("failure_type") or entry.result.failure_reason or "unknown")

    by_type: dict[str, list[tuple[MemoryV3Entry, float]]] = defaultdict(list)
    for entry, score in matches:
        by_type[_dedup_key(entry)].append((entry, score))

    filtered: list[tuple[MemoryV3Entry, float]] = []
    removed_condition_ids: set[str] = set()
    for ft, group in by_type.items():
        group.sort(key=lambda x: x[0].created_at, reverse=True)
        if len(group) > 2:
            for entry, _score in group[2:]:
                removed_condition_ids.add(entry.condition_id)
        filtered.extend(group[:2])

    # --- 2. Cross-condition balance -------------------------------------------------
    surviving_condition_ids = {e.condition_id for e, _ in filtered}
    if (
        len(surviving_condition_ids) <= 1
        and removed_condition_ids - surviving_condition_ids
    ):
        # Add back one entry from a removed condition (highest-score among them)
        for entry, score in matches:
            if entry.condition_id not in surviving_condition_ids:
                filtered.append((entry, score))
                break

    return filtered


def canonical_action_signature_from_steps(steps: Any) -> str:
    actions: list[dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        if not action:
            continue
        item: dict[str, Any] = {"action": action}
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if action == "gripper-action" and "state" in params:
            item["state"] = params["state"]
        actions.append(item)
    return json.dumps(actions, ensure_ascii=False, sort_keys=True)


def canonical_action_signature_from_entry(entry: Any) -> str:
    return canonical_action_signature_from_steps(getattr(entry, "skill_sequence", []) or [])


def make_memory_v3_entry(
    *,
    condition_id: str,
    scenario_id: str,
    available_actions: set[str] | list[str],
    skill_sequence: list[dict[str, Any]],
    recovery_success: bool,
    task_success: bool,
    failure_reason: str = "",
    source: str = "simulation",
    summary: str = "",
    metadata: dict[str, Any] | None = None,
    validation_evidence: dict[str, Any] | None = None,
    episode_type: str = "anomaly_recovery",
    anomaly: dict[str, Any] | AnomalyInfo | None = None,
    scene: dict[str, Any] | SceneInfo | None = None,
    task: dict[str, Any] | TaskInfo | None = None,
    perception: dict[str, Any] | PerceptionInfo | None = None,
    reconstruction_artifacts: dict[str, Any] | ReconstructionArtifacts | None = None,
    recovery_plan: dict[str, Any] | RecoveryPlan | None = None,
    execution_feedback: dict[str, Any] | ExecutionFeedback | None = None,
    sensor_summary: dict[str, Any] | SensorSummaryInfo | None = None,
    key_slices: list[dict[str, Any]] | None = None,
    keyframes: list[dict[str, Any]] | None = None,
    anomaly_state: dict[str, Any] | None = None,
    retrieval_key: dict[str, Any] | None = None,
    failure_taxonomy: dict[str, Any] | None = None,
    validation_status: str = "",
    validation_source: str = "",
    source_run_id: str = "",
    source_trial_id: str = "",
    z_change: float = 0.0,
    time_cost: float = 0.0,
    attempts: int = 1,
    memory_gate: dict[str, Any] | MemoryGateInfo | None = None,
    sim_real_gap: dict[str, Any] | SimRealGapInfo | None = None,
    sim_real_pair: dict[str, Any] | SimRealPairInfo | None = None,
    sandbox_calibration: dict[str, Any] | SandboxCalibrationInfo | None = None,
    critic_result: dict[str, Any] | CriticResultInfo | None = None,
    real_episode_ref: dict[str, Any] | RealEpisodeRef | None = None,
    memory_tags: dict[str, Any] | None = None,
) -> MemoryV3Entry:
    return MemoryV3Entry(
        episode_type=episode_type,
        condition_id=condition_id,
        scenario_id=scenario_id,
        available_actions=sorted(set(available_actions)),
        skill_sequence=skill_sequence,
        result=MemoryV3Result(
            recovery_success=bool(recovery_success),
            task_success=bool(task_success),
            failure_reason=failure_reason,
            success=bool(recovery_success),
            z_change=float(z_change or 0.0),
            time_cost=float(time_cost or 0.0),
            attempts=int(attempts or 1),
        ),
        source=source,
        source_run_id=source_run_id,
        source_trial_id=source_trial_id,
        summary=summary,
        anomaly=anomaly or {},
        scene=scene or {},
        task=task or {},
        perception=perception or {},
        reconstruction_artifacts=reconstruction_artifacts or {},
        recovery_plan=recovery_plan or {},
        execution_feedback=execution_feedback or {},
        sensor_summary=sensor_summary or {},
        key_slices=key_slices or [],
        keyframes=keyframes or [],
        anomaly_state=anomaly_state or {},
        retrieval_key=retrieval_key or {},
        failure_taxonomy=failure_taxonomy or {},
        validation_status=validation_status or ("simulation_only" if recovery_success else "failed"),
        validation_source=validation_source,
        metadata=metadata or {},
        validation_evidence=validation_evidence or {},
        memory_gate=memory_gate or {},
        sim_real_gap=sim_real_gap or {},
        sim_real_pair=sim_real_pair or {},
        sandbox_calibration=sandbox_calibration or {},
        critic_result=critic_result or {},
        real_episode_ref=real_episode_ref or {},
        memory_tags=memory_tags or {},
    )
