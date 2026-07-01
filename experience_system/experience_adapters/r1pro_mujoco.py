"""R1Pro MuJoCo adapter for universal experience entries."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from experience_core import (
    ExperienceEntry,
    MemoryGate,
    ObjectState,
    RobotState,
    SensorSummary,
    SkillTraceItem,
    compute_memory_gate,
    standardize_failure_taxonomy,
)
from experience_core.schema import UNKNOWN_SKILL_NAMESPACE


class R1ProMujocoAdapter:
    robot_type = "mobile_dual_arm"
    backend = "mujoco"

    def __init__(self, robot_id: str = "r1pro_mujoco_001") -> None:
        self.robot_id = robot_id

    def normalize_episode(self, task_result: Any) -> ExperienceEntry:
        result = asdict(task_result) if is_dataclass(task_result) else dict(task_result)
        scenario_id = str(result.get("scenario_id") or "")
        condition_id = str(result.get("condition_id") or "")
        control_mode = str(result.get("control_mode") or "ideal")
        metrics = dict(result.get("metrics") or {})
        contact_stability = metrics.get("contact_stability") if isinstance(metrics.get("contact_stability"), dict) else {}
        skill_trace = [self._skill_trace_item(item) for item in result.get("skill_trace") or []]

        metrics.setdefault("condition_id", condition_id)
        metrics.setdefault("skill_trace", result.get("skill_trace") or [])
        metrics["selected_alternate_place"] = result.get("selected_place_site") == "alternate_place_zone_site"
        memory_gate = compute_memory_gate(
            metrics,
            task_success=bool(result.get("task_success")),
            validation_status="simulation_validated",
        )

        failure_reason = str(result.get("failure_reason") or "")
        failure_taxonomy = {"failure_type": failure_reason} if failure_reason else {}
        target_object = str(result.get("target_object") or "")
        object_start = result.get("object_start") or []
        object_final = result.get("object_final") or []
        selected_place_site = str(result.get("selected_place_site") or "")
        keyframes = list(result.get("keyframes") or [])
        keyframe_dir = str(metrics.get("keyframe_dir") or "")

        entry = ExperienceEntry(
            source="simulation",
            domain="r1pro_anomaly_recovery",
            backend=self.backend,
            skill_namespace=UNKNOWN_SKILL_NAMESPACE,
            validation_status="simulation_validated",
            robot=self._robot_state(result),
            embodiment={
                "control_mode": control_mode,
                "attach_mode": metrics.get("attach_mode", ""),
            },
            scenario={
                "scenario_id": scenario_id,
                "name": f"R1Pro {scenario_id} task chain",
                "model_path": result.get("model_path", ""),
            },
            condition={
                "condition_id": condition_id,
                "name": condition_id,
                "place_occupied": bool(metrics.get("place_occupied", False)),
            },
            task={
                "name": f"r1pro_{scenario_id.lower()}_task_chain",
                "stage": "task_chain",
                "control_mode": control_mode,
            },
            anomaly={
                "type": "place_occupied" if metrics.get("place_occupied") else "",
                "description": "primary place zone occupied" if metrics.get("place_occupied") else "",
            },
            skill_sequence=skill_trace,
            state_before={"objects": {target_object: {"position": object_start}}},
            state_after={"objects": {target_object: {"position": object_final}}},
            sensor_summary=SensorSummary(
                sensor_modalities=["mujoco_state"],
                contact_state={
                    "contact_after_close": bool(contact_stability.get("contact_after_close", False)),
                    "contact_count_after_close": contact_stability.get("contact_count_after_close", 0),
                    "contact_count_after_lift": contact_stability.get("contact_count_after_lift", 0),
                    "source": contact_stability.get("source", ""),
                } if contact_stability else {},
                force_torque={
                    "wrist_force_proxy": contact_stability.get("wrist_force_proxy", 0.0),
                    "source": "mujoco_proxy",
                } if contact_stability else {},
                raw_refs={"model_path": result.get("model_path", "")},
            ),
            spatial_state={
                "selected_place_site": selected_place_site,
                "place_occupied": bool(metrics.get("place_occupied", False)),
                "attach_mode": metrics.get("attach_mode", ""),
            },
            object_state=ObjectState(
                objects={
                    target_object: {
                        "start_position": object_start,
                        "final_position": object_final,
                    }
                },
                target_object=target_object,
                object_class=self._object_class_for_scenario(scenario_id),
                occupancy={
                    "place_occupied": bool(metrics.get("place_occupied", False)),
                    "selected_place_site": selected_place_site,
                },
            ),
            result={
                "success": bool(result.get("success")),
                "task_success": bool(result.get("task_success")),
                "failure_reason": failure_reason,
            },
            execution_feedback={
                "object_start": object_start,
                "object_final": object_final,
                "selected_place_site": selected_place_site,
                "object_lift": metrics.get("object_lift"),
                "metrics": metrics,
                "contact_stability": contact_stability,
                "contact_after_close": {"active": bool(contact_stability.get("contact_after_close", False))} if contact_stability else {},
                "contact_after_lift": {"active": float(contact_stability.get("contact_during_lift_ratio", 0.0)) > 0.0} if contact_stability else {},
            },
            keyframes=keyframes,
            memory_gate=MemoryGate(**memory_gate),
            failure_taxonomy=failure_taxonomy,
            memory_tags={
                "memory_type": "episodic",
                "memory_scope": "condition",
                "memory_role": "success_prior" if result.get("success") else "failure_case",
            },
            raw_refs={
                "source_result_type": "TaskChainResult",
                "model_path": result.get("model_path", ""),
                "keyframe_dir": keyframe_dir,
            },
            metadata={
                "legacy_task_chain_result": result,
            },
        )
        return standardize_failure_taxonomy(entry)

    def _robot_state(self, result: dict[str, Any]) -> RobotState:
        scenario_id = str(result.get("scenario_id") or "")
        tags = ["mobile_base", "torso", "gripper"]
        if scenario_id == "G4":
            tags.append("dual_arm")
        else:
            tags.append("single_arm")
        return RobotState(
            robot_id=self.robot_id,
            robot_type=self.robot_type,
            embodiment_tags=tags,
            backend=self.backend,
            kinematic_groups={
                "left_arm": {"role": "manipulator"},
                "right_arm": {"role": "manipulator"},
                "torso": {"role": "workspace_positioning"},
                "base": {"role": "mobile_positioning"},
            },
            mobile_base={"available": True},
            torso={"available": True},
            grippers={
                "left": {"type": "parallel", "available": True},
                "right": {"type": "parallel", "available": True},
            },
        )

    def _skill_trace_item(self, raw: dict[str, Any]) -> SkillTraceItem:
        name = str(raw.get("skill") or raw.get("name") or "")
        outputs = {key: value for key, value in raw.items() if key not in {"skill", "name", "success", "message"}}
        return SkillTraceItem(
            name=name,
            primitive_type=self._primitive_type(name),
            phase=self._phase(name),
            outputs=outputs,
            success=bool(raw.get("success", False)),
            error=raw.get("error"),
            message=str(raw.get("message") or ""),
            raw=dict(raw),
        )

    @staticmethod
    def _primitive_type(name: str) -> str:
        if "detect" in name or "select" in name or "verify" in name:
            return "perception_or_verification"
        if "place" in name:
            return "place"
        if "gripper" in name:
            return "gripper"
        if "lift" in name or "transport" in name:
            return "transport"
        if "base" in name or "torso" in name:
            return "workspace_positioning"
        return "reach"

    @staticmethod
    def _phase(name: str) -> str:
        for phase in ("pregrasp", "approach", "close", "lift", "transport", "place", "release", "verify"):
            if phase in name:
                return phase
        if "detect" in name or "select" in name:
            return "perception"
        return "task_chain"

    @staticmethod
    def _object_class_for_scenario(scenario_id: str) -> str:
        if scenario_id == "G4":
            return "large_object"
        if scenario_id == "G3":
            return "sortable_object"
        return "object"
