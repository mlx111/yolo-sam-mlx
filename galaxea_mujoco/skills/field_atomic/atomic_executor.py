from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Any

import mujoco
import numpy as np

from skills.base.base_lidar_skill import load_skill as load_base_lidar
from skills.base.base_motion_skill import load_skill as load_base_motion
from skills.base.head_camera_skill import load_skill as load_head_camera
from skills.base.head_camera_rgbd_save_skill import load_skill as load_head_camera_rgbd_save
from skills.base.head_camera_grounded_sam2_pose_skill import load_skill as load_head_camera_grounded_sam2_pose
from skills.base.torso_move_skill import load_skill as load_torso_move
from skills.base.arm_ik_skill import R1ProArmIKSkill
from skills.base.left_arm_move_skill import load_skill as load_left_arm
from skills.base.right_arm_move_skill import load_skill as load_right_arm
from skills.torso_frame.torso_frame_move_to_pregrasp_skill import load_skill as load_torso_frame_move_to_pregrasp
from skills.torso_frame.torso_frame_plan_cartesian_trajectory_skill import load_skill as load_torso_frame_plan_cartesian_trajectory
from skills.torso_frame.torso_frame_approach_object_skill import load_skill as load_torso_frame_approach_object
from skills.torso_frame.torso_frame_close_gripper_skill import load_skill as load_torso_frame_close_gripper
from skills.torso_frame.torso_frame_open_gripper_skill import load_skill as load_torso_frame_open_gripper
from skills.torso_frame.torso_frame_lift_skill import load_skill as load_torso_frame_lift
from skills.torso_frame.torso_frame_lower_held_object_skill import load_skill as load_torso_frame_lower_held_object
from skills.torso_frame.torso_frame_transport_object_skill import load_skill as load_torso_frame_transport_object
from skills.torso_frame._common import target_position_path
from experience_core import ExperienceEntry, SkillTraceItem
from experience_core.schema import GALAXEA_R1PRO_TORSO_NAMESPACE

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
        return _canonical_action(action) in self._registry

    def execute(self, model: mujoco.MjModel, data: mujoco.MjData, action: str, parameters: dict[str, Any]) -> FieldAtomicResult:
        action = _canonical_action(action)
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
        if action == "move_base_relative":
            raw = load_base_motion().move_to_pose(model, data, _base_relative_target(model, data, parameters), **_base_kwargs(parameters))
            return _from_result(action, raw, parameters)
        if action == "set_torso_posture":
            raw = _execute_torso_height_level(model, data, parameters)
            if bool(getattr(raw, "success", False)):
                _clear_runtime_motion_cache(parameters)
            return _from_result(action, raw, parameters)
        if action == "head_camera_capture":
            raw = load_head_camera().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=True, status="ok", message="head camera captured", parameters=dict(parameters), raw_result={"camera_name": getattr(raw, "camera_name", "")})
        if action == "head_camera_rgbd_save":
            _apply_posture_dependent_head_camera_pose(model, data, parameters)
            raw = load_head_camera_rgbd_save().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=True, status="ok", message="head camera RGB-D saved", parameters=dict(parameters), raw_result=_compact_head_camera_rgbd(raw.to_dict()))
        if action == "head_camera_grounded_sam2_pose":
            raw = load_head_camera_grounded_sam2_pose().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(
                action=action,
                success=bool(raw.success),
                status="ok" if raw.success else "failed",
                message=raw.message,
                parameters=dict(parameters),
                raw_result=_compact_head_camera_pose(raw.to_dict()),
            )
        if action == "head_camera_rgbd_save":
            _apply_posture_dependent_head_camera_pose(model, data, parameters)
            raw = load_head_camera_rgbd_save().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_compact_head_camera_rgbd(raw.to_dict()))
        if action == "head_camera_grounded_sam2_pose":
            raw = load_head_camera_grounded_sam2_pose().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(
                action=action,
                success=bool(raw.success),
                status="ok" if raw.success else "failed",
                message=raw.message,
                parameters=dict(parameters),
                raw_result=_compact_head_camera_pose(raw.to_dict()),
            )
        if action == "move_to_pregrasp":
            raw = load_torso_frame_move_to_pregrasp().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_jsonable(raw.to_dict()))
        if action == "plan_cartesian_trajectory":
            raw = load_torso_frame_plan_cartesian_trajectory().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_jsonable(raw.to_dict()))
        if action == "approach_object":
            raw = load_torso_frame_approach_object().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_jsonable(raw.to_dict()))
        if action == "close_gripper":
            raw = load_torso_frame_close_gripper().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_jsonable(raw.to_dict()))
        if action == "open_gripper":
            raw = load_torso_frame_open_gripper().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_jsonable(raw.to_dict()))
        if action == "lift":
            raw = load_torso_frame_lift().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_jsonable(raw.to_dict()))
        if action == "lower_held_object":
            try:
                raw = load_torso_frame_lower_held_object().execute_recovery_action(model, data, parameters)
            except ValueError as exc:
                return FieldAtomicResult(action=action, success=False, status="invalid_lower_distance", message=str(exc), parameters=dict(parameters), raw_result={})
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(parameters), raw_result=_jsonable(raw.to_dict()))
        if action == "transport_to_detected_target":
            checked = _transport_parameters_from_detected_target(action, parameters, allow_offset=True)
            if isinstance(checked, FieldAtomicResult):
                return checked
            raw = load_torso_frame_transport_object().execute_recovery_action(model, data, checked)
            return FieldAtomicResult(action=action, success=bool(raw.success), status="ok" if raw.success else "failed", message=raw.message, parameters=dict(checked), raw_result=_jsonable(raw.to_dict()))
        if action == "frame_alignment_debug":
            raw = _frame_alignment_debug(model, data, parameters)
            return FieldAtomicResult(action=action, success=True, status="ok", message="frame alignment debug captured", parameters=dict(parameters), raw_result=_jsonable(raw))
        if action == "base_lidar_scan":
            raw = load_base_lidar().execute_recovery_action(model, data, parameters)
            return FieldAtomicResult(action=action, success=True, status="ok", message="base lidar scan captured", parameters=dict(parameters), raw_result={"site_name": getattr(raw, "site_name", "")})
        return FieldAtomicResult(action=action, success=False, status="unsupported_action", message=f"unsupported field atomic action: {action}", parameters=dict(parameters))


def _canonical_action(action: str) -> str:
    text = str(action)
    return text[len("torso_frame_"):] if text.startswith("torso_frame_") else text


INTERNAL_CONTROL_PARAMETER_KEYS = {
    "steps",
    "settle_steps",
    "max_joint_step",
    "fail_threshold",
    "success_threshold",
    "pregrasp_success_threshold",
    "direct_qpos",
    "stabilize",
    "lock_posture",
    "orientation_threshold",
}


def public_field_atomic_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in parameters.items()
        if not str(key).startswith("_") and str(key) not in INTERNAL_CONTROL_PARAMETER_KEYS
    }


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


def _transport_parameters_from_detected_target(action: str, parameters: dict[str, Any], *, allow_offset: bool) -> dict[str, Any] | FieldAtomicResult:
    target_class = str(parameters.get("target_class") or "").strip()
    if not target_class:
        return FieldAtomicResult(
            action=action,
            success=False,
            status="missing_target_class",
            message="transport requires target_class",
            parameters=dict(parameters),
            raw_result={},
        )
    path = target_position_path(target_class, parameters.get("_runtime_tmp_dir"))
    if not Path(path).exists():
        return FieldAtomicResult(
            action=action,
            success=False,
            status="missing_position_input",
            message=f"missing detected target position JSON for target_class={target_class}: {path}",
            parameters=dict(parameters),
            raw_result={"target_class": target_class, "position_input_path": str(path)},
        )
    checked = dict(parameters)
    checked["target_class"] = target_class
    checked["position_input_path"] = str(path)
    checked.setdefault("target_z_mode", "keep_current")
    checked.setdefault("use_position_xy_only", True)
    if allow_offset:
        try:
            checked["place_offset_x"] = _small_place_offset(parameters.get("place_offset_x", 0.0), "place_offset_x")
            checked["place_offset_y"] = _small_place_offset(parameters.get("place_offset_y", 0.0), "place_offset_y")
        except ValueError as exc:
            return FieldAtomicResult(
                action=action,
                success=False,
                status="invalid_place_offset",
                message=str(exc),
                parameters=dict(parameters),
                raw_result={"target_class": target_class, "position_input_path": str(path)},
            )
        checked.pop("place_offset_z", None)
    else:
        checked.pop("place_offset_x", None)
        checked.pop("place_offset_y", None)
        checked.pop("place_offset_z", None)
    return checked


def _small_place_offset(value: Any, name: str) -> float:
    offset = float(value)
    limit = 0.02
    if abs(offset) > limit:
        raise ValueError(f"{name} must be within +/-{limit:.2f} m for transport_to_detected_target: {offset}")
    return offset


def _base_kwargs(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": int(parameters.get("steps", 900)),
        "settle_steps": int(parameters.get("settle_steps", 120)),
        "max_joint_step": float(parameters.get("max_joint_step", 0.01)),
        "fail_threshold": float(parameters.get("fail_threshold", 0.02)),
        "direct_qpos": bool(parameters.get("direct_qpos", False)),
    }


def _base_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = []
    for name in ("base_x", "base_y", "base_yaw"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return np.asarray(values, dtype=np.float64)


def _base_relative_target(model: mujoco.MjModel, data: mujoco.MjData, parameters: dict[str, Any]) -> list[float]:
    dx = float(parameters.get("x", 0.0))
    dy = float(parameters.get("y", 0.0))
    dyaw = float(parameters.get("yaw", 0.0))
    max_move = abs(float(parameters.get("max_relative_move", 0.50)))
    if abs(dx) > max_move or abs(dy) > max_move:
        raise ValueError(f"relative base x/y move exceeds max_relative_move={max_move}: x={dx}, y={dy}")
    current = _base_qpos(model, data)
    target = current + np.asarray([dx, dy, dyaw], dtype=np.float64)
    return [float(target[0]), float(target[1]), float(target[2])]


TORSO_FRAME_POSTURE_LEVELS: dict[str, list[float]] = {
    # The real robot torso skill uses mid=[0.87, -1.35, -0.48, 0.0].
    # This MuJoCo model has the first three torso joint directions inverted
    # relative to that command convention, so keep the public level semantics
    # aligned with the real skill while sending the model-stable mapped qpos.
    "mid": [-0.87, 1.35, 0.48, 0.0],
    "high": [0.0, 0.0, 0.0, 0.0],
}

TORSO_FRAME_JOINT_NAMES = ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4")

HEAD_CAMERA_POSES_BY_TORSO_LEVEL: dict[str, dict[str, list[float]]] = {
    "high": {
        "pos": [0.15, -0.05, 0.57],
        "quat": [0.698886079, 0.107509296, -0.107509296, -0.698886079],
    },
    "mid": {
        "pos": [0.15, -0.05, 0.57],
        "quat": [0.640856469, 0.298836052, -0.298836052, -0.640856469],
    },
}


def _torso_posture_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    params = dict(parameters)
    if "target_qpos" in params or "posture" in params:
        raise ValueError("set_torso_posture only accepts vertical height levels; use level=mid/high")
    level = str(params.get("level", "mid"))
    if level not in TORSO_FRAME_POSTURE_LEVELS:
        raise ValueError(f"unsupported torso level: {level}. available={sorted(TORSO_FRAME_POSTURE_LEVELS)}")
    params["target_qpos"] = list(TORSO_FRAME_POSTURE_LEVELS[level])
    params.setdefault("steps", 900)
    params.setdefault("settle_steps", 120)
    params.setdefault("max_joint_step", 0.004)
    params.setdefault("fail_threshold", 0.02)
    params.setdefault("closed_loop_gain", 1.0)
    params.setdefault("direct_qpos", False)
    params.setdefault("lock_posture", True)
    return params


def _execute_torso_height_level(model: mujoco.MjModel, data: mujoco.MjData, parameters: dict[str, Any]) -> Any:
    params = _torso_posture_parameters(parameters)
    torso = load_torso_move()
    return torso.execute_recovery_action(model, data, params)


def _apply_posture_dependent_head_camera_pose(model: mujoco.MjModel, data: mujoco.MjData, parameters: dict[str, Any]) -> None:
    camera_name = str(parameters.get("camera_name", "head_top_work_camera"))
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        return
    level = _nearest_torso_posture_level(model, data)
    pose = HEAD_CAMERA_POSES_BY_TORSO_LEVEL.get(level, HEAD_CAMERA_POSES_BY_TORSO_LEVEL["high"])
    model.cam_pos[camera_id] = np.asarray(pose["pos"], dtype=np.float64)
    quat = np.asarray(pose["quat"], dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    model.cam_quat[camera_id] = quat
    mujoco.mj_forward(model, data)


def _nearest_torso_posture_level(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    qpos_indices: list[int] = []
    for joint_name in TORSO_FRAME_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return "high"
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    current = np.asarray(data.qpos[qpos_indices], dtype=np.float64)
    distances = {
        level: float(np.linalg.norm(current - np.asarray(target, dtype=np.float64)))
        for level, target in TORSO_FRAME_POSTURE_LEVELS.items()
    }
    return min(distances, key=distances.get)


def _clear_runtime_motion_cache(parameters: dict[str, Any]) -> None:
    runtime_tmp_dir = parameters.get("_runtime_tmp_dir")
    if not runtime_tmp_dir:
        return
    root = Path(str(runtime_tmp_dir))
    for name in ("object_positions", "trajectories"):
        path = root / name
        if path.exists():
            shutil.rmtree(path)


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


def _frame_alignment_debug(model: mujoco.MjModel, data: mujoco.MjData, parameters: dict[str, Any]) -> dict[str, Any]:
    side = str(parameters.get("side", "right"))
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported side: {side}")
    ik = R1ProArmIKSkill(parameters.get("urdf_path", "urdf/r1_pro_with_gripper.urdf"))
    mujoco.mj_forward(model, data)
    q_pin = ik.sync_q_from_mujoco(model, data)
    pin_tcp = ik.current_tcp_position(q_pin, side)
    base_xy, base_yaw = ik.mobile_base_pose(model, data)
    c = float(np.cos(base_yaw))
    s = float(np.sin(base_yaw))
    world_from_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pin_tcp_world = world_from_base @ np.asarray(pin_tcp, dtype=np.float64).reshape(3)
    pin_tcp_world[:2] += base_xy

    result: dict[str, Any] = {
        "side": side,
        "urdf_path": str(ik.urdf_path),
        "base_xy": np.round(base_xy, 6).tolist(),
        "base_yaw": float(base_yaw),
        "pinocchio_hand_tcp_world": np.round(pin_tcp_world, 6).tolist(),
    }
    for site_name in (f"{side}_hand_tcp", f"{side}_grasp_tool"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            result[site_name] = {"error": "site_not_found"}
            continue
        pos = data.site_xpos[site_id].copy()
        xmat = data.site_xmat[site_id].reshape(3, 3).copy()
        delta = pos - pin_tcp_world
        result[site_name] = {
            "world": np.round(pos, 6).tolist(),
            "x_axis_world": np.round(xmat[:, 0], 6).tolist(),
            "y_axis_world": np.round(xmat[:, 1], 6).tolist(),
            "z_axis_world": np.round(xmat[:, 2], 6).tolist(),
            "minus_pinocchio_hand_tcp_world": np.round(delta, 6).tolist(),
            "minus_pinocchio_hand_tcp_norm": float(np.linalg.norm(delta)),
        }
    hand = result.get(f"{side}_hand_tcp", {})
    grasp = result.get(f"{side}_grasp_tool", {})
    if isinstance(hand, dict) and isinstance(grasp, dict) and "world" in hand and "world" in grasp:
        hand_pos = np.asarray(hand["world"], dtype=np.float64)
        grasp_pos = np.asarray(grasp["world"], dtype=np.float64)
        result["grasp_tool_minus_hand_tcp_world"] = np.round(grasp_pos - hand_pos, 6).tolist()
        result["grasp_tool_minus_hand_tcp_norm"] = float(np.linalg.norm(grasp_pos - hand_pos))
    return result


def _from_result(action: str, raw: Any, parameters: dict[str, Any]) -> FieldAtomicResult:
    payload = asdict(raw) if hasattr(raw, "__dataclass_fields__") else dict(getattr(raw, "__dict__", {}))
    success = bool(payload.get("success", payload.get("task_success", False)))
    status = "ok" if success else "failed"
    message = f"{action} completed" if success else f"{action} failed"
    return FieldAtomicResult(action=action, success=success, status=status, message=message, parameters=dict(parameters), raw_result=payload)


def _compact_head_camera_rgbd(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(payload.get("success", False)),
        "camera_name": payload.get("camera_name"),
        "rgb_path": payload.get("rgb_path"),
        "depth_path": payload.get("depth_path"),
        "metadata_path": payload.get("metadata_path"),
    }


def _compact_head_camera_pose(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(payload.get("success", False)),
        "target_class": payload.get("target_class"),
        "reference_frame": payload.get("reference_frame"),
        "center_reference": payload.get("center_reference"),
        "median_reference": payload.get("median_reference"),
        "center_world": payload.get("center_world"),
        "median_world": payload.get("median_world"),
        "center_camera_cv": payload.get("center_camera_cv"),
        "bbox_xyxy": payload.get("bbox_xyxy"),
        "valid_depth_count": payload.get("valid_depth_count"),
        "mask_pixel_count": payload.get("mask_pixel_count"),
        "position_output_path": payload.get("position_output_path"),
    }


def build_atomic_experience_entry(
    *,
    scenario_id: str,
    condition_id: str,
    robot_type: str,
    action: str,
    result: FieldAtomicResult,
    experience_source: str = "simulation",
    backend: str = "mujoco",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    skill_catalog_version: str = "v1",
) -> ExperienceEntry:
    action = _canonical_action(action)
    skill_spec = field_atomic_skill_registry().get(action)
    failure_taxonomy = _field_atomic_failure_taxonomy(action, result)
    evidence = _field_atomic_failure_evidence(result)
    llm_failure_summary = _field_atomic_failure_summary(action, result, failure_taxonomy, evidence)
    public_parameters = public_field_atomic_parameters(dict(result.parameters))
    entry = ExperienceEntry(
        source=experience_source,
        backend=backend,
        skill_namespace=skill_namespace,
        skill_catalog_version=skill_catalog_version,
        scenario={"scenario_id": scenario_id},
        condition={"condition_id": condition_id},
        robot={"robot_type": robot_type},
        object_state={"object_class": result.parameters.get("object_class", "unknown")},
        skill_sequence=[
            SkillTraceItem(
                name=action,
                primitive_type="field_atomic",
                phase="execution",
                inputs={"parameters": _jsonable(public_parameters)},
                outputs={"result": _jsonable(dict(result.raw_result))},
                success=bool(result.success),
                message=result.message,
                raw={"parameters": _jsonable(public_parameters)},
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
            "field_atomic_parameters": _jsonable(public_parameters),
            "field_atomic_result": _jsonable(dict(result.raw_result)),
            "skill_description": skill_spec.description if skill_spec is not None else "",
            "parameter_schema": dict(skill_spec.parameter_schema) if skill_spec is not None else {},
            "failure_taxonomy": failure_taxonomy,
            "failure_evidence": evidence,
            "llm_failure_summary": llm_failure_summary,
        },
        failure_taxonomy=failure_taxonomy if not result.success else {},
        memory_tags={
            "memory_type": "field_atomic_experience",
            "memory_role": "field_atomic_success" if result.success else "field_atomic_failure",
        },
        metadata={
            "field_atomic": True,
            "field_atomic_action": action,
            "skill_namespace": skill_namespace,
            "skill_catalog_version": skill_catalog_version,
            "skill_description": skill_spec.description if skill_spec is not None else "",
            "parameter_schema": dict(skill_spec.parameter_schema) if skill_spec is not None else {},
            "llm_failure_summary": llm_failure_summary,
        },
    )
    return entry


def _field_atomic_failure_taxonomy(action: str, result: FieldAtomicResult) -> dict[str, Any]:
    if bool(result.success):
        return {}
    raw = result.raw_result if isinstance(result.raw_result, dict) else {}
    status = str(result.status or "")
    failure_type = "unknown_failure"
    if action == "head_camera_grounded_sam2_pose":
        failure_type = "perception_miss"
        if raw.get("valid_depth_count") == 0:
            failure_type = "depth_invalid"
    elif action in {"move_to_pregrasp", "approach_object", "plan_cartesian_trajectory", "move_base_relative", "set_torso_posture"}:
        failure_type = "actuation_limit"
    elif action == "lift":
        failure_type = "object_not_lifted" if raw.get("object_lift_success") is False or raw.get("object_lift_world") is not None else "actuation_limit"
    elif action in {"close_gripper"}:
        failure_type = "grasp_miss"
    elif action in {"transport_to_detected_target"}:
        if status in {"missing_position_input", "missing_target_class"}:
            failure_type = "perception_miss"
        elif raw.get("object_follow_error") is not None:
            failure_type = "transport_collision"
        else:
            failure_type = "place_error"
    elif action == "lower_held_object":
        failure_type = "place_error"

    return {
        "failure_type": failure_type,
        "standard_failure_type": failure_type,
        "failure_stage": action,
        "failure_action": action,
        "failure_status": status,
        "failure_reason": _field_atomic_failure_reason(action, result, failure_type),
        "final_error": raw.get("final_error"),
        "stage_errors": raw.get("stage_errors"),
        "stage_orientation_errors": raw.get("stage_orientation_errors"),
    }


def _field_atomic_failure_reason(action: str, result: FieldAtomicResult, failure_type: str) -> str:
    raw = result.raw_result if isinstance(result.raw_result, dict) else {}
    if action in {"move_to_pregrasp", "approach_object"}:
        return f"{action} failed with final_error={raw.get('final_error')}, target_torso={raw.get('target_torso')}, pregrasp_torso={raw.get('pregrasp_torso') or raw.get('grasp_torso')}"
    if action == "lift":
        return f"lift failed: object_lift_world={raw.get('object_lift_world')}, min_object_lift={raw.get('min_object_lift')}, object_body={raw.get('object_body')}"
    if action == "head_camera_grounded_sam2_pose":
        return f"perception failed for target_class={result.parameters.get('target_class')}: {result.message}"
    if action == "transport_to_detected_target":
        return f"transport failed: status={result.status}, target_class={result.parameters.get('target_class')}, object_follow_error={raw.get('object_follow_error')}"
    return result.message or f"{action} failed: {failure_type}"


def _field_atomic_failure_evidence(result: FieldAtomicResult) -> dict[str, Any]:
    raw = result.raw_result if isinstance(result.raw_result, dict) else {}
    keys = (
        "target_class",
        "side",
        "target_torso",
        "pregrasp_torso",
        "grasp_torso",
        "target_world",
        "pregrasp_world",
        "grasp_world",
        "final_tcp_torso",
        "final_tcp_world",
        "final_tcp_minus_pregrasp_torso",
        "final_tcp_minus_pregrasp_world",
        "final_tcp_pregrasp_error_norm",
        "start_torso",
        "target_source_torso",
        "place_offset_torso",
        "final_error",
        "stage_errors",
        "stage_orientation_errors",
        "object_body",
        "object_start_world",
        "object_final_world",
        "object_lift_world",
        "object_lift_success",
        "min_object_lift",
        "object_follow_error",
        "bbox_xyxy",
        "valid_depth_count",
        "position_output_path",
    )
    evidence = {key: raw.get(key) for key in keys if raw.get(key) not in (None, "", [], {})}
    public_params = public_field_atomic_parameters(dict(result.parameters))
    if public_params:
        evidence["parameters"] = public_params
    return _jsonable(evidence)


def _field_atomic_failure_summary(action: str, result: FieldAtomicResult, taxonomy: dict[str, Any], evidence: dict[str, Any]) -> str:
    if bool(result.success):
        return ""
    parts = [
        f"action={action}",
        f"failure_type={taxonomy.get('failure_type', '')}",
        f"status={result.status}",
    ]
    params = evidence.get("parameters") if isinstance(evidence.get("parameters"), dict) else {}
    side = params.get("side") or evidence.get("side")
    target_class = params.get("target_class") or evidence.get("target_class")
    if target_class:
        parts.append(f"target_class={target_class}")
    if side:
        parts.append(f"side={side}")
    if evidence.get("target_torso") is not None:
        parts.append(f"target_torso={evidence.get('target_torso')}")
    if evidence.get("pregrasp_torso") is not None:
        parts.append(f"pregrasp_torso={evidence.get('pregrasp_torso')}")
    if evidence.get("final_error") is not None:
        parts.append(f"final_error={evidence.get('final_error')}")
    if evidence.get("object_lift_world") is not None:
        parts.append(f"object_lift_world={evidence.get('object_lift_world')}")
    reason = taxonomy.get("failure_reason")
    if reason:
        parts.append(f"reason={reason}")
    return "; ".join(str(item) for item in parts if item)
