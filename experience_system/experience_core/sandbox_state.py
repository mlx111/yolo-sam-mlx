"""Sandbox initial-state extraction from universal experience entries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .schema import ExperienceEntry, utc_now


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _numeric_vector(value: Any, *, limit: int | None = None) -> list[float]:
    out: list[float] = []
    for item in _as_list(value):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out[:limit] if limit is not None else out


def _object_pose(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    pos = raw.get("position") or raw.get("start_position") or raw.get("pose") or raw.get("xyz") or raw.get("translation")
    position = _numeric_vector(pos, limit=3)
    quat = _numeric_vector(raw.get("quaternion") or raw.get("quat_wxyz"), limit=4)
    pose: dict[str, Any] = {}
    if position:
        while len(position) < 3:
            position.append(0.0)
        pose["position"] = position
    if quat:
        while len(quat) < 4:
            quat.append(0.0)
        pose["quaternion_wxyz"] = quat
    return pose


def _entry_objects(entry: ExperienceEntry) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    object_state = entry.object_state.objects if isinstance(entry.object_state.objects, dict) else {}
    before_objects = entry.state_before.get("objects") if isinstance(entry.state_before.get("objects"), dict) else {}
    for source in (object_state, before_objects):
        for name, raw in source.items():
            pose = _object_pose(raw)
            if pose:
                objects[str(name)] = pose
    return objects


def _confidence(report: dict[str, Any]) -> float:
    score = 0.0
    if report.get("object_pose_count", 0) > 0:
        score += 0.45
    if report.get("joint_position_count", 0) > 0:
        score += 0.25
    if report.get("gripper_state_count", 0) > 0:
        score += 0.15
    if report.get("contact_state_count", 0) > 0:
        score += 0.10
    if report.get("source_episode_id"):
        score += 0.05
    return round(min(score, 1.0), 4)


@dataclass
class SandboxInitialState:
    schema_version: str = "sandbox_initial_state_v1"
    created_at: str = ""
    scenario_id: str = ""
    condition_id: str = ""
    source: str = ""
    source_episode_id: str = ""
    robot_qpos: list[float] = field(default_factory=list)
    robot_qvel: list[float] = field(default_factory=list)
    end_effector_pose: dict[str, Any] = field(default_factory=dict)
    gripper_state: dict[str, Any] = field(default_factory=dict)
    object_poses: dict[str, dict[str, Any]] = field(default_factory=dict)
    obstacle_poses: dict[str, dict[str, Any]] = field(default_factory=dict)
    contact_state: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    confidence: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SandboxInitialState":
        fields = getattr(cls, "__dataclass_fields__", {})
        return cls(**{key: value for key, value in payload.items() if key in fields})


def build_sandbox_initial_state(entry: ExperienceEntry) -> SandboxInitialState:
    objects = _entry_objects(entry)
    target = entry.object_state.target_object
    object_poses: dict[str, dict[str, Any]] = {}
    obstacle_poses: dict[str, dict[str, Any]] = {}
    for name, pose in objects.items():
        if name == target or not object_poses:
            object_poses[name] = pose
        else:
            obstacle_poses[name] = pose

    sensor = entry.sensor_summary
    report = {
        "source_episode_id": entry.experience_id,
        "object_pose_count": len(object_poses) + len(obstacle_poses),
        "joint_position_count": len(sensor.joint_positions or []),
        "joint_velocity_count": len(sensor.joint_velocities or []),
        "gripper_state_count": len(sensor.gripper_state or {}),
        "contact_state_count": len(sensor.contact_state or {}),
    }
    missing: list[str] = []
    if not object_poses:
        missing.append("object_poses")
    if not sensor.joint_positions:
        missing.append("robot_qpos")
    if not sensor.joint_velocities:
        missing.append("robot_qvel")
    if not sensor.gripper_state:
        missing.append("gripper_state")
    if not sensor.contact_state:
        missing.append("contact_state")

    return SandboxInitialState(
        scenario_id=entry.scenario_id,
        condition_id=entry.condition_id,
        source=entry.source,
        source_episode_id=entry.experience_id,
        robot_qpos=[float(item) for item in sensor.joint_positions or []],
        robot_qvel=[float(item) for item in sensor.joint_velocities or []],
        end_effector_pose=dict(sensor.end_effector_pose or {}),
        gripper_state=dict(sensor.gripper_state or {}),
        object_poses=object_poses,
        obstacle_poses=obstacle_poses,
        contact_state=dict(sensor.contact_state or {}),
        timestamp=str((sensor.timestamps or {}).get("episode_time") or entry.created_at),
        confidence=_confidence(report),
        missing_fields=missing,
        evidence={
            **report,
            "target_object": target,
            "selected_place_site": entry.spatial_state.get("selected_place_site", ""),
            "validation_status": entry.validation_status,
        },
    )


def coerce_sandbox_initial_state(value: Any) -> SandboxInitialState | None:
    if value is None:
        return None
    if isinstance(value, SandboxInitialState):
        return value
    if is_dataclass(value):
        return SandboxInitialState.from_dict(asdict(value))
    if isinstance(value, dict):
        return SandboxInitialState.from_dict(value)
    return None


def choose_sandbox_state_entry(
    entries: list[ExperienceEntry],
    *,
    scenario: str,
    condition: str,
    prefer_success: bool = True,
) -> ExperienceEntry | None:
    candidates = [entry for entry in entries if entry.scenario_id == scenario and entry.condition_id == condition]
    if not candidates:
        return None
    states = [(entry, build_sandbox_initial_state(entry)) for entry in candidates]
    states.sort(
        key=lambda item: (
            int(bool(item[0].result.get("success"))) if prefer_success else 0,
            item[1].confidence,
            len(item[1].object_poses) + len(item[1].obstacle_poses),
            item[0].updated_at,
        ),
        reverse=True,
    )
    return states[0][0]
