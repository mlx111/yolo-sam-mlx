"""Adapter for experiment-sim-wrapper1 MemoryV3+ UR5e entries."""

from __future__ import annotations

from typing import Any

from experience_core import (
    CriticResult,
    ExperienceEntry,
    MemoryGate,
    ObjectState,
    RobotState,
    SandboxCalibration,
    SensorSummary,
    SimRealGap,
    SkillTraceItem,
    standardize_failure_taxonomy,
)


class Wrapper1UR5eAdapter:
    robot_type = "fixed_single_arm"
    backend = "mujoco"

    def __init__(self, robot_id: str = "ur5e_wrapper1_mujoco") -> None:
        self.robot_id = robot_id

    def normalize_entry(self, raw_entry: dict[str, Any]) -> ExperienceEntry:
        condition_id = str(raw_entry.get("condition_id") or "")
        scenario_id = str(raw_entry.get("scenario_id") or "")
        result = raw_entry.get("result") if isinstance(raw_entry.get("result"), dict) else {}
        task = raw_entry.get("task") if isinstance(raw_entry.get("task"), dict) else {}
        scene = raw_entry.get("scene") if isinstance(raw_entry.get("scene"), dict) else {}
        feedback = raw_entry.get("execution_feedback") if isinstance(raw_entry.get("execution_feedback"), dict) else {}
        perception = raw_entry.get("perception") if isinstance(raw_entry.get("perception"), dict) else {}
        sensor_summary = raw_entry.get("sensor_summary") if isinstance(raw_entry.get("sensor_summary"), dict) else {}

        target_object = str(task.get("object_class") or "apple")
        observed_pos = feedback.get("observed_pos") or self._perception_pos(perception)
        object_state = ObjectState(
            objects={target_object: {"observed_position": observed_pos}},
            target_object=target_object,
            object_class=target_object,
            metadata={
                "scene_objects": scene.get("objects", []),
                "apple_z_after_recovery": feedback.get("apple_z_after_recovery"),
            },
        )

        entry = ExperienceEntry(
            experience_id=f"wrapper1_{raw_entry.get('experience_id', '')}" if raw_entry.get("experience_id") else "",
            created_at=str(raw_entry.get("created_at") or ""),
            updated_at=str(raw_entry.get("updated_at") or ""),
            source=self._source(raw_entry),
            domain=str(raw_entry.get("domain") or raw_entry.get("source") or "simulation"),
            backend=self.backend,
            validation_status=str(raw_entry.get("validation_status") or ""),
            memory_partition=str(raw_entry.get("memory_partition") or ""),
            robot=self._robot_state(raw_entry),
            embodiment={"control_mode": "actuator_or_wrapper", "attach_mode": self._contact_pattern(feedback)},
            scenario={
                "scenario_id": scenario_id,
                "name": scenario_id,
                "scene_name": scene.get("scene_name", ""),
                "xml_path": scene.get("xml_path", ""),
            },
            condition={
                "condition_id": condition_id,
                "name": task.get("condition_name") or condition_id,
            },
            task={
                "name": task.get("name", "ur5e_anomaly_recovery"),
                "stage": task.get("stage", ""),
                "object_class": task.get("object_class", ""),
                "condition": task.get("condition", ""),
            },
            anomaly=raw_entry.get("anomaly") if isinstance(raw_entry.get("anomaly"), dict) else {},
            skill_sequence=[self._skill_trace_item(item) for item in raw_entry.get("skill_sequence") or []],
            state_before={"perception": perception.get("before_anomaly", {})},
            state_after={"observed_pos": observed_pos},
            sensor_summary=SensorSummary(**{key: value for key, value in sensor_summary.items() if key in SensorSummary.__dataclass_fields__}),
            spatial_state={
                "scene_name": scene.get("scene_name", ""),
                "camera_view": scene.get("camera_view", ""),
                "contact_pattern": self._contact_pattern(feedback),
            },
            object_state=object_state,
            result={
                "success": bool(result.get("success", result.get("recovery_success", False))),
                "recovery_success": bool(result.get("recovery_success", result.get("success", False))),
                "task_success": bool(result.get("task_success", False)),
                "failure_reason": result.get("failure_reason", ""),
                "z_change": result.get("z_change", 0.0),
                "time_cost": result.get("time_cost", 0.0),
                "attempts": result.get("attempts", 1),
            },
            execution_feedback=feedback,
            key_slices=list(raw_entry.get("key_slices") or []),
            keyframes=list(raw_entry.get("keyframes") or []),
            memory_tags=dict(raw_entry.get("memory_tags") or self._default_memory_tags(raw_entry)),
            memory_gate=self._memory_gate(raw_entry),
            critic_result=self._critic_result(raw_entry),
            failure_taxonomy=dict(raw_entry.get("failure_taxonomy") or {}),
            sim_real_pair=dict(raw_entry.get("sim_real_pair") or {}),
            sim_real_gap=self._sim_real_gap(raw_entry),
            sandbox_calibration=self._sandbox_calibration(raw_entry),
            raw_refs={
                "source_format": "experiment-sim-wrapper1.memory_v3_plus",
                "source_experience_id": raw_entry.get("experience_id", ""),
                "scene_xml": scene.get("xml_path", ""),
            },
            metadata={
                "wrapper1_summary": raw_entry.get("summary", ""),
                "wrapper1_text_summary": raw_entry.get("text_summary", ""),
                "retrieval_key": raw_entry.get("retrieval_key", {}),
            },
        )
        return standardize_failure_taxonomy(entry)

    def _robot_state(self, raw_entry: dict[str, Any]) -> RobotState:
        return RobotState(
            robot_id=self.robot_id,
            robot_type=self.robot_type,
            embodiment_tags=["fixed_base", "single_arm", "parallel_gripper"],
            backend=self.backend,
            kinematic_groups={"arm": {"role": "manipulator"}},
            grippers={"main": {"type": "parallel", "available": True}},
            metadata={"source": raw_entry.get("source", "")},
        )

    def _skill_trace_item(self, raw: dict[str, Any]) -> SkillTraceItem:
        action = str(raw.get("action") or raw.get("type") or raw.get("skill") or "")
        params = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else raw.get("params", {})
        return SkillTraceItem(
            name=action,
            primitive_type=self._primitive_type(action),
            phase=self._phase(action),
            inputs=params if isinstance(params, dict) else {},
            success=True,
            raw=dict(raw),
        )

    @staticmethod
    def _primitive_type(action: str) -> str:
        if "detect" in action or "verify" in action or "check" in action:
            return "perception_or_verification"
        if "gripper" in action:
            return "gripper"
        if "place" in action:
            return "place"
        if "grasp" in action:
            return "grasp"
        if "recover" in action:
            return "recovery"
        return "motion"

    @staticmethod
    def _phase(action: str) -> str:
        for phase in ("detect", "pregrasp", "grasp", "place", "release", "verify", "recover"):
            if phase in action:
                return phase
        if "gripper" in action:
            return "gripper"
        return "task_chain"

    @staticmethod
    def _source(raw_entry: dict[str, Any]) -> str:
        source = str(raw_entry.get("source") or "simulation")
        if source == "real":
            return "real"
        if source == "pseudo_real":
            return "pseudo_real"
        return "simulation"

    @staticmethod
    def _perception_pos(perception: dict[str, Any]) -> Any:
        after = perception.get("after_anomaly") if isinstance(perception.get("after_anomaly"), dict) else {}
        before = perception.get("before_anomaly") if isinstance(perception.get("before_anomaly"), dict) else {}
        return after.get("object_pos") or before.get("object_pos")

    @staticmethod
    def _contact_pattern(feedback: dict[str, Any]) -> str:
        close = feedback.get("contact_after_close") if isinstance(feedback.get("contact_after_close"), dict) else {}
        lift = feedback.get("contact_after_lift") if isinstance(feedback.get("contact_after_lift"), dict) else {}

        def has_contact(value: dict[str, Any]) -> bool:
            return bool(value.get("left_contact") or value.get("right_contact") or value.get("contact"))

        close_ok = has_contact(close)
        lift_ok = has_contact(lift)
        if close_ok and lift_ok:
            return "contact_close_and_lift"
        if close_ok:
            return "contact_after_close_only"
        if lift_ok:
            return "contact_after_lift_only"
        return "no_contact"

    @staticmethod
    def _default_memory_tags(raw_entry: dict[str, Any]) -> dict[str, Any]:
        result = raw_entry.get("result") if isinstance(raw_entry.get("result"), dict) else {}
        return {
            "memory_type": "episodic",
            "memory_scope": "condition",
            "memory_role": "success_prior" if result.get("success") else "failure_case",
        }

    @staticmethod
    def _memory_gate(raw_entry: dict[str, Any]) -> MemoryGate:
        value = raw_entry.get("memory_gate") if isinstance(raw_entry.get("memory_gate"), dict) else {}
        return MemoryGate(**{key: item for key, item in value.items() if key in MemoryGate.__dataclass_fields__})

    @staticmethod
    def _critic_result(raw_entry: dict[str, Any]) -> CriticResult:
        value = raw_entry.get("critic_result") if isinstance(raw_entry.get("critic_result"), dict) else {}
        return CriticResult(**{key: item for key, item in value.items() if key in CriticResult.__dataclass_fields__})

    @staticmethod
    def _sim_real_gap(raw_entry: dict[str, Any]) -> SimRealGap:
        value = raw_entry.get("sim_real_gap") if isinstance(raw_entry.get("sim_real_gap"), dict) else {}
        return SimRealGap(**{key: item for key, item in value.items() if key in SimRealGap.__dataclass_fields__})

    @staticmethod
    def _sandbox_calibration(raw_entry: dict[str, Any]) -> SandboxCalibration:
        value = raw_entry.get("sandbox_calibration") if isinstance(raw_entry.get("sandbox_calibration"), dict) else {}
        return SandboxCalibration(**{key: item for key, item in value.items() if key in SandboxCalibration.__dataclass_fields__})
