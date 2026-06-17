from __future__ import annotations

from dataclasses import asdict
from typing import Any

import mujoco

from skills.base.base_lidar_skill import load_skill as load_base_lidar
from skills.base.base_motion_skill import load_skill as load_base_motion
from skills.base.head_camera_skill import load_skill as load_head_camera
from skills.base.torso_move_skill import load_skill as load_torso_move
from skills.base.left_arm_move_skill import load_skill as load_left_arm
from skills.base.right_arm_move_skill import load_skill as load_right_arm
from experience_core import ExperienceEntry, SkillTraceItem

from .atomic_registry import field_atomic_skill_registry
from .atomic_schema import FieldAtomicResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


class FieldAtomicSkillExecutor:
    """Thin field-style wrapper around existing low-level simulation skills."""

    def __init__(self) -> None:
        self._registry = field_atomic_skill_registry()

    def can_execute(self, action: str) -> bool:
        return action in self._registry

    def execute(self, model: mujoco.MjModel, data: mujoco.MjData, action: str, parameters: dict[str, Any]) -> FieldAtomicResult:
        if action not in self._registry:
            return FieldAtomicResult(action=action, success=False, status="unsupported_action", message=f"unsupported field atomic action: {action}", parameters=dict(parameters))
        result = self._dispatch(model, data, action, parameters)
        return result

    def _dispatch(self, model: mujoco.MjModel, data: mujoco.MjData, action: str, parameters: dict[str, Any]) -> FieldAtomicResult:
        if action == "left_arm_move_to_position":
            raw = load_left_arm().move_to_pose(model, data, [parameters["target_x"], parameters["target_y"], parameters["target_z"]], **_arm_kwargs(parameters))
            return _from_result(action, raw, parameters)
        if action == "right_arm_move_to_position":
            raw = load_right_arm().move_to_pose(model, data, [parameters["target_x"], parameters["target_y"], parameters["target_z"]], **_arm_kwargs(parameters))
            return _from_result(action, raw, parameters)
        if action == "left_gripper_set":
            raw = _set_side_gripper(model, data, "left", _gripper_command(parameters), direct_qpos=bool(parameters.get("direct_qpos", False)))
            return FieldAtomicResult(action=action, success=True, status="ok", message="left gripper command applied", parameters=dict(parameters), raw_result={"value": raw})
        if action == "right_gripper_set":
            raw = _set_side_gripper(model, data, "right", _gripper_command(parameters), direct_qpos=bool(parameters.get("direct_qpos", False)))
            return FieldAtomicResult(action=action, success=True, status="ok", message="right gripper command applied", parameters=dict(parameters), raw_result={"value": raw})
        if action == "torso_move_to_posture":
            raw = load_torso_move().execute_recovery_action(model, data, parameters)
            return _from_result(action, raw, parameters)
        if action == "base_move_to_pose":
            raw = load_base_motion().move_to_pose(model, data, _base_target(parameters), **_base_kwargs(parameters))
            return _from_result(action, raw, parameters)
        if action == "head_camera_capture":
            raw = load_head_camera().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=True, status="ok", message="head camera captured", parameters=dict(parameters), raw_result={"camera_name": getattr(raw, "camera_name", "")})
        if action == "base_lidar_scan":
            raw = load_base_lidar().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=True, status="ok", message="base lidar scan captured", parameters=dict(parameters), raw_result={"site_name": getattr(raw, "site_name", "")})
        return FieldAtomicResult(action=action, success=False, status="unsupported_action", message=f"unsupported field atomic action: {action}", parameters=dict(parameters))


def _arm_kwargs(parameters: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "steps": int(parameters.get("steps", 1500)),
        "settle_steps": int(parameters.get("settle_steps", 3000)),
        "max_joint_step": float(parameters.get("max_joint_step", 0.006)),
        "fail_threshold": float(parameters.get("fail_threshold", 0.02)),
        "direct_qpos": bool(parameters.get("direct_qpos", False)),
        "stabilize": bool(parameters.get("stabilize", True)),
        "lock_posture": bool(parameters.get("lock_posture", True)),
        "control_frame": str(parameters.get("control_frame", "grasp_tool")),
    }
    if "target_quat_wxyz" in parameters:
        kwargs["target_quat_wxyz"] = parameters["target_quat_wxyz"]
    return kwargs


def _base_target(parameters: dict[str, Any]) -> list[float]:
    if "target_qpos" in parameters:
        target = parameters["target_qpos"]
        if isinstance(target, list) and len(target) == 3:
            return [float(target[0]), float(target[1]), float(target[2])]
    return [
        float(parameters.get("base_x", 0.0)),
        float(parameters.get("base_y", 0.0)),
        float(parameters.get("base_yaw", 0.0)),
    ]


def _base_kwargs(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": int(parameters.get("steps", 900)),
        "settle_steps": int(parameters.get("settle_steps", 120)),
        "max_joint_step": float(parameters.get("max_joint_step", 0.01)),
        "fail_threshold": float(parameters.get("fail_threshold", 0.02)),
        "direct_qpos": bool(parameters.get("direct_qpos", False)),
    }


def _gripper_command(parameters: dict[str, Any]) -> str | float:
    if "gripper_value" in parameters:
        return float(parameters["gripper_value"])
    return "close" if int(parameters.get("state", 0)) == 1 else "open"


def _set_side_gripper(model: mujoco.MjModel, data: mujoco.MjData, side: str, command: str | float, *, direct_qpos: bool) -> float:
    if isinstance(command, str):
        if command == "open":
            value = 0.0
        elif command == "close":
            value = 0.025
        else:
            raise ValueError(f"unsupported gripper command: {command}")
    else:
        value = max(0.0, min(float(command), 0.025))
    for suffix in ("finger_joint1", "finger_joint2"):
        joint_name = f"{side}_gripper_{suffix}"
        actuator_name = f"{joint_name}_pos"
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value
        if direct_qpos:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)
    return value


def _from_result(action: str, raw: Any, parameters: dict[str, Any]) -> FieldAtomicResult:
    payload = asdict(raw) if hasattr(raw, "__dataclass_fields__") else dict(getattr(raw, "__dict__", {}))
    success = bool(payload.get("success", payload.get("task_success", False)))
    status = "ok" if success else "failed"
    message = f"{action} completed" if success else f"{action} failed"
    return FieldAtomicResult(action=action, success=success, status=status, message=message, parameters=dict(parameters), raw_result=payload)


def build_atomic_experience_entry(
    *,
    scenario_id: str,
    condition_id: str,
    robot_type: str,
    action: str,
    result: FieldAtomicResult,
    experience_source: str = "simulation",
    backend: str = "mujoco",
) -> ExperienceEntry:
    entry = ExperienceEntry(
        source=experience_source,
        backend=backend,
        scenario={"scenario_id": scenario_id},
        condition={"condition_id": condition_id},
        robot={"robot_type": robot_type},
        object_state={"object_class": result.parameters.get("object_class", "unknown")},
        skill_sequence=[
            SkillTraceItem(
                name=action,
                primitive_type="field_atomic",
                phase="execution",
                inputs={"parameters": _jsonable(dict(result.parameters))},
                outputs={"result": _jsonable(dict(result.raw_result))},
                success=bool(result.success),
                message=result.message,
                raw={"parameters": _jsonable(dict(result.parameters))},
            )
        ],
        result={
            "success": bool(result.success),
            "task_success": bool(result.success),
            "field_atomic_action": action,
            "field_atomic_status": result.status,
        },
        execution_feedback={
            "field_atomic_action": action,
            "field_atomic_parameters": _jsonable(dict(result.parameters)),
            "field_atomic_result": _jsonable(dict(result.raw_result)),
        },
        memory_tags={
            "memory_type": "field_atomic_experience",
            "memory_role": "field_atomic_success" if result.success else "field_atomic_failure",
        },
        metadata={
            "field_atomic": True,
            "field_atomic_action": action,
        },
    )
    return entry
