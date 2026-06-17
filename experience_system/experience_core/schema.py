"""Universal experience schema for robot simulation and real episodes."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def as_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return value if isinstance(value, dict) else {}


@dataclass
class RobotState:
    robot_id: str = ""
    robot_type: str = ""
    embodiment_tags: list[str] = field(default_factory=list)
    backend: str = ""
    kinematic_groups: dict[str, Any] = field(default_factory=dict)
    end_effectors: dict[str, Any] = field(default_factory=dict)
    mobile_base: dict[str, Any] = field(default_factory=dict)
    torso: dict[str, Any] = field(default_factory=dict)
    grippers: dict[str, Any] = field(default_factory=dict)
    joints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectState:
    objects: dict[str, Any] = field(default_factory=dict)
    target_object: str = ""
    object_class: str = ""
    spatial_relations: list[dict[str, Any]] = field(default_factory=list)
    support_relations: list[dict[str, Any]] = field(default_factory=list)
    occupancy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillTraceItem:
    name: str = ""
    primitive_type: str = ""
    phase: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: float | None = None
    duration: float | None = None
    safety_flags: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorSummary:
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
class SensorEvidence:
    visual_observation: dict[str, Any] = field(default_factory=dict)
    lidar_observation: dict[str, Any] = field(default_factory=dict)
    wrist_force_observation: dict[str, Any] = field(default_factory=dict)
    evidence_refs: dict[str, Any] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryGate:
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
class CriticResult:
    overall_status: str = ""
    critic_risk_score: float = 0.0
    rule_flags: list[dict[str, Any]] = field(default_factory=list)
    feedback_for_rewrite: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimRealGap:
    gap_id: str = ""
    gap_score: float = 0.0
    uncertainty: float = 0.0
    outcome_gap: dict[str, Any] = field(default_factory=dict)
    pose_gap: dict[str, Any] = field(default_factory=dict)
    contact_gap: dict[str, Any] = field(default_factory=dict)
    perception_gap: dict[str, Any] = field(default_factory=dict)
    actuation_gap: dict[str, Any] = field(default_factory=dict)
    robot_state_gap: dict[str, Any] = field(default_factory=dict)
    timing_gap: dict[str, Any] = field(default_factory=dict)
    scene_reconstruction_gap: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxCalibration:
    calibration_id: str = ""
    source_gap_ids: list[str] = field(default_factory=list)
    object_pose_bias: list[float] = field(default_factory=list)
    perception_noise_bias: list[float] = field(default_factory=list)
    actuation_delay_bias: float = 0.0
    contact_success_bias: float = 0.0
    slip_risk_bias: float = 0.0
    calibration_confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceEntry:
    schema_version: str = "universal_experience_v1"
    experience_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    source: str = "simulation"
    domain: str = ""
    backend: str = ""
    validation_status: str = ""
    memory_partition: str = ""
    robot: RobotState = field(default_factory=RobotState)
    embodiment: dict[str, Any] = field(default_factory=dict)
    scenario: dict[str, Any] = field(default_factory=dict)
    condition: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    anomaly: dict[str, Any] = field(default_factory=dict)
    skill_sequence: list[SkillTraceItem] = field(default_factory=list)
    action_trace: list[dict[str, Any]] = field(default_factory=list)
    observation_trace: list[dict[str, Any]] = field(default_factory=list)
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    sensor_summary: SensorSummary = field(default_factory=SensorSummary)
    sensor_evidence: SensorEvidence = field(default_factory=SensorEvidence)
    spatial_state: dict[str, Any] = field(default_factory=dict)
    object_state: ObjectState = field(default_factory=ObjectState)
    result: dict[str, Any] = field(default_factory=dict)
    execution_feedback: dict[str, Any] = field(default_factory=dict)
    key_slices: list[dict[str, Any]] = field(default_factory=list)
    keyframes: list[dict[str, Any]] = field(default_factory=list)
    retrieval_key: dict[str, Any] = field(default_factory=dict)
    memory_tags: dict[str, Any] = field(default_factory=dict)
    memory_gate: MemoryGate = field(default_factory=MemoryGate)
    critic_result: CriticResult = field(default_factory=CriticResult)
    failure_taxonomy: dict[str, Any] = field(default_factory=dict)
    sim_real_pair: dict[str, Any] = field(default_factory=dict)
    sim_real_gap: SimRealGap = field(default_factory=SimRealGap)
    sandbox_calibration: SandboxCalibration = field(default_factory=SandboxCalibration)
    real_episode_ref: dict[str, Any] = field(default_factory=dict)
    raw_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experience_id:
            self.experience_id = f"exp_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = utc_now()
        if not self.updated_at:
            self.updated_at = self.created_at
        self.robot = coerce_dataclass(RobotState, self.robot)
        self.object_state = coerce_dataclass(ObjectState, self.object_state)
        self.sensor_summary = coerce_dataclass(SensorSummary, self.sensor_summary)
        self.sensor_evidence = coerce_dataclass(SensorEvidence, self.sensor_evidence)
        self.memory_gate = coerce_dataclass(MemoryGate, self.memory_gate)
        self.critic_result = coerce_dataclass(CriticResult, self.critic_result)
        self.sim_real_gap = coerce_dataclass(SimRealGap, self.sim_real_gap)
        self.sandbox_calibration = coerce_dataclass(SandboxCalibration, self.sandbox_calibration)
        self.skill_sequence = [coerce_dataclass(SkillTraceItem, item) for item in self.skill_sequence]
        if not self.memory_partition:
            self.memory_partition = infer_memory_partition(self)
        if not self.retrieval_key:
            self.retrieval_key = build_retrieval_key(self)

    @property
    def scenario_id(self) -> str:
        return str(self.scenario.get("scenario_id") or self.scenario.get("id") or "")

    @property
    def condition_id(self) -> str:
        return str(self.condition.get("condition_id") or self.condition.get("id") or "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_dataclass(cls: type, value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        fields = getattr(cls, "__dataclass_fields__", {})
        return cls(**{key: value for key, value in value.items() if key in fields})
    return cls()


def infer_memory_partition(entry: ExperienceEntry) -> str:
    if entry.source == "real" or entry.validation_status in {"real_executed", "real_validated"}:
        return "real_memory"
    if not bool(entry.result.get("success", False)):
        return "failed_memory"
    if entry.validation_status in {"simulation_validated", "sandbox_validated"}:
        return "validated_memory"
    return "simulation_memory"


def build_retrieval_key(entry: ExperienceEntry) -> dict[str, Any]:
    action_names = [item.name for item in entry.skill_sequence if item.name]
    return {
        "source": entry.source,
        "backend": entry.backend,
        "robot_id": entry.robot.robot_id,
        "robot_type": entry.robot.robot_type,
        "embodiment_tags": list(entry.robot.embodiment_tags),
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "task_stage": str(entry.task.get("stage") or ""),
        "task_name": str(entry.task.get("name") or ""),
        "object_class": entry.object_state.object_class,
        "target_object": entry.object_state.target_object,
        "plan_signature": "->".join(action_names),
        "memory_type": entry.memory_tags.get("memory_type", ""),
        "memory_role": entry.memory_tags.get("memory_role", ""),
        "failure_type": entry.failure_taxonomy.get("failure_type", ""),
        "critic_status": entry.critic_result.overall_status,
        "pair_status": entry.sim_real_pair.get("validation_status", ""),
        "gap_type": entry.sim_real_gap.outcome_gap.get("type", ""),
        "real_episode_ref": str(entry.real_episode_ref.get("raw_episode_id") or ""),
    }
