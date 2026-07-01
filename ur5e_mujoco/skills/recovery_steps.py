"""Executable UR5e recovery skills backed by ExperimentV4 motion helpers."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import spatialmath as sm

from experiment_runtime.runtime_backend import (
    DOWN_GRASP_ROTATION,
)
from experiment_runtime import runtime_backend
from experiment_runtime.experiment_config import TASK_LIFT_Z_CHANGE


Q_SAFE = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=np.float64)
PLACE_TARGET_Z = 0.04
PLACE_TCP_RELEASE_HEIGHT = 0.09
PLACE_TCP_RETRACT_HEIGHT = 0.19


def _params(params: dict | None) -> dict:
    return params if isinstance(params, dict) else {}


def _require_float(params: dict[str, Any], key: str, skill: str) -> float:
    if key not in params:
        raise ValueError(f"{skill} requires parameter {key}")
    try:
        return float(params[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{skill} parameter {key} must be a float") from exc


def _require_offset(params: dict[str, Any], skill: str) -> np.ndarray:
    return np.array([
        _require_float(params, "dx", skill),
        _require_float(params, "dy", skill),
        _require_float(params, "dz", skill),
    ], dtype=np.float64)


def _record(experiment: Any, skill: str, success: bool = True, reason: str = "ok", **extra: Any) -> None:
    if hasattr(experiment, "_record_basic_skill"):
        experiment._record_basic_skill(skill, success, reason, **extra)


def _target_position(experiment: Any) -> np.ndarray:
    if hasattr(experiment, "_resolve_skill_target_position"):
        return np.asarray(experiment._resolve_skill_target_position("detect_object_pose"), dtype=np.float64).reshape(3)
    metrics = getattr(experiment, "metrics", {}) or {}
    for key in ("observed_pos", "perceived_position"):
        value = metrics.get(key)
        if value is not None:
            return np.asarray(value, dtype=np.float64).reshape(3)
    return np.asarray(experiment.data.body(experiment.apple_body_id).xpos.copy(), dtype=np.float64)


def _target_object_z(experiment: Any) -> float:
    return float(experiment.data.body(experiment.apple_body_id).xpos[2])


def _rotation(experiment: Any) -> np.ndarray:
    if hasattr(experiment, "T_wo") and experiment.T_wo is not None:
        return np.asarray(experiment.T_wo.R, dtype=np.float64)
    return np.asarray(DOWN_GRASP_ROTATION, dtype=np.float64)


def _current_grasp_context(experiment: Any) -> Any:
    return getattr(experiment, "_fixed_vertical_grasp_context", None)


def _semantic_grasp_position(experiment: Any) -> np.ndarray:
    context = _current_grasp_context(experiment)
    if context is not None and hasattr(context, "target_pos"):
        return np.asarray(context.target_pos, dtype=np.float64).reshape(3)
    metrics = getattr(experiment, "metrics", {}) or {}
    value = metrics.get("semantic_grasp_pos") or metrics.get("grasp_target_pos")
    if value is not None:
        return np.asarray(value, dtype=np.float64).reshape(3)
    return _target_position(experiment)


def _install_adjusted_grasp_pose(experiment: Any, semantic_grasp_pos: np.ndarray, pregrasp_height: float | None = None) -> dict[str, Any]:
    context = _current_grasp_context(experiment)
    top_points = getattr(context, "top_points_world", None) if context is not None else None
    if pregrasp_height is None and hasattr(experiment, "T_pregrasp") and experiment.T_pregrasp is not None and hasattr(experiment, "T_wo") and experiment.T_wo is not None:
        pregrasp_height = float(np.linalg.norm(np.asarray(experiment.T_pregrasp.t, dtype=np.float64) - np.asarray(experiment.T_wo.t, dtype=np.float64)))
    if pregrasp_height is None or pregrasp_height <= 1e-6:
        pregrasp_height = 0.127
    mapping = runtime_backend.install_fixed_vertical_tcp_pose(
        experiment,
        np.asarray(semantic_grasp_pos, dtype=np.float64).reshape(3),
        top_points_world=top_points,
        pregrasp_height=float(pregrasp_height),
        require_ik=False,
    )
    if context is not None and hasattr(context, "target_pos"):
        context.target_pos = np.asarray(semantic_grasp_pos, dtype=np.float64).reshape(3)
        context.pregrasp_pos = np.asarray(mapping["T_pregrasp"].t, dtype=np.float64)
        context.report.update(runtime_backend.tcp_mapping_report(mapping))
        context.report["semantic_grasp_pos"] = context.target_pos.tolist()
        experiment._fixed_vertical_grasp_context = context
    metrics = getattr(experiment, "metrics", {}) or {}
    metrics["semantic_grasp_pos"] = np.asarray(semantic_grasp_pos, dtype=np.float64).reshape(3).tolist()
    metrics["grasp_target_pos"] = metrics["semantic_grasp_pos"]
    metrics["tcp_grasp_pos"] = np.asarray(mapping["T_grasp"].t, dtype=np.float64).tolist()
    metrics["tcp_pregrasp_pos"] = np.asarray(mapping["T_pregrasp"].t, dtype=np.float64).tolist()
    return runtime_backend.tcp_mapping_report(mapping)


def camera_image(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    stage = str(params.get("stage") or "recovery_camera_image")
    target_class = str(params.get("target_class") or "apple").lower()
    work_dir = params.get("work_dir") or f"/tmp/wrapper1_detect_{target_class}"
    camera_context = runtime_backend.save_camera_rgbd(experiment, work_dir=work_dir)
    if hasattr(experiment, "_save_keyframe"):
        experiment._save_keyframe(stage, "恢复技能相机图像")
    _record(experiment, "camera_rgbd_save", True, "rgbd_saved", stage=stage, **camera_context)


def detect_object(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    if not str(params.get("target_class") or "").strip():
        raise ValueError("detect_object_pose requires parameter target_class")
    target_class = str(params["target_class"]).strip().lower()
    if hasattr(experiment, "_clear_perception_override_for_recovery") and target_class == "apple":
        experiment._clear_perception_override_for_recovery()
    try:
        scene = runtime_backend.detect_object(experiment, target_class=target_class, work_dir=params.get("work_dir"))
        if not scene.detection_ok or scene.apple_pos is None:
            raise RuntimeError("object_detection_failed")
        observed_pos = np.asarray(scene.apple_pos, dtype=np.float64).reshape(3)
        metrics = getattr(experiment, "metrics", {}) or {}
        detected_objects = metrics.setdefault("detected_objects", {})
        if isinstance(detected_objects, dict):
            detected_objects[target_class] = observed_pos.tolist()
        if target_class in {"plate", "bowl", "container", "target"}:
            place_pos = np.array([observed_pos[0], observed_pos[1], PLACE_TARGET_Z], dtype=np.float64)
            metrics["transport_target_class"] = target_class
            metrics["transport_target_pos"] = place_pos.tolist()
            metrics["place_target_class"] = target_class
            metrics["place_target_pos"] = place_pos.tolist()
    except Exception as exc:
        _record(experiment, "detect_object_pose", False, f"perception_failed: {exc}", target_class=target_class)
        return
    report = {
        "target_class": target_class,
        "detection_ok": bool(scene.detection_ok),
        "confidence": float(getattr(scene, "confidence", 0.0)),
        "mask_nonzero": int(getattr(scene, "mask_nonzero", 0)),
        "observed_pos": observed_pos.tolist(),
        "raw_points": int(len(scene.raw_points_world)) if scene.raw_points_world is not None else 0,
        "rgb_path": getattr(scene, "rgb_path", None),
        "depth_path": getattr(scene, "depth_path", None),
        "mask_path": getattr(scene, "mask_path", None),
        "role": "place_target" if target_class in {"plate", "bowl", "container", "target"} else "grasp_target",
    }
    _record(experiment, "detect_object_pose", True, "object_pose_detected", **report)


def create_grasp(experiment: Any, params: dict | None = None, default_pregrasp_height: float = 0.127) -> None:
    params = _params(params)
    target_class = str(params.get("target_class") or getattr(experiment, "_last_perceived_scene_target", None) or "apple").lower()
    grasp_params = {**params, "target_class": target_class}
    runtime_backend.build_fixed_vertical_grasp_source(experiment, target_class=target_class, params=grasp_params)
    context = runtime_backend.create_fixed_vertical_grasp(experiment, grasp_params, default_pregrasp_height)
    report = dict(context.report)
    report.setdefault("target_class", target_class)
    _record(
        experiment,
        "create_fixed_vertical_grasp",
        True,
        "fixed_vertical_grasp_created",
        **report,
    )


def move_pregrasp(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    offset = _require_offset(params, "move_to_pregrasp")
    duration = float(params.get("duration", 1.0))
    if not hasattr(experiment, "T_wo") or experiment.T_wo is None:
        raise RuntimeError("move_to_pregrasp requires detect_object_pose to create T_wo first")
    experiment.T_pregrasp = sm.SE3.Trans(np.asarray(experiment.T_wo.t, dtype=np.float64) + offset) * sm.SE3(sm.SO3(experiment.T_wo.R, check=False))
    result = experiment._move_cartesian(experiment.T_pregrasp, duration)
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)


def move_grasp(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    offset = _require_offset(params, "approach_object")
    duration = float(params.get("duration", 1.0))
    if not hasattr(experiment, "T_wo") or experiment.T_wo is None:
        raise RuntimeError("approach_object requires detect_object_pose to create T_wo first")
    semantic_target = _semantic_grasp_position(experiment) + offset
    mapping_report = _install_adjusted_grasp_pose(experiment, semantic_target)
    result = experiment._move_cartesian(experiment.T_wo, duration)
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    experiment._step_n(int(params.get("settle_steps", 500)))
    _record(
        experiment,
        "approach_object",
        bool(getattr(result, "success", True)),
        "approach_executed",
        semantic_grasp_pos=semantic_target.tolist(),
        **mapping_report,
    )


def gripper_action(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    state = int(params.get("state", 0))
    if state not in {0, 1}:
        raise ValueError(f"gripper state must be 0 or 1, got {state!r}")
    if state == 0:
        result = experiment._gripper_open()
        if hasattr(experiment, "_record_skill_result"):
            experiment._record_skill_result(result)
        _record(experiment, "open_gripper", True, "gripper_opened")
        return
    result = experiment._gripper_close()
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    contact = experiment._contact_summary() if hasattr(experiment, "_contact_summary") else {}
    has_contact = bool(contact.get("left_contact") or contact.get("right_contact"))
    _record(experiment, "close_gripper", has_contact, "physical_contact_detected" if has_contact else "no_contact_after_close", contact=contact)


def vertical_grasp(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    duration = float(params.get("duration", 1.0))
    lift_height = _require_float(params, "lift_height", "lift")
    object_z_before = _target_object_z(experiment)
    if _current_grasp_context(experiment) is not None and hasattr(experiment, "T_wo") and experiment.T_wo is not None:
        T_start = experiment.T_wo
    else:
        T_start = experiment.robot.get_cartesian()
    T_lift = sm.SE3.Trans(0.0, 0.0, lift_height) * T_start
    result = experiment._move_cartesian(T_lift, duration)
    experiment.T_wo = T_lift
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    if getattr(experiment, "anomaly_type", "") == "collision" and hasattr(experiment, "_stabilize_collision_lift"):
        experiment._stabilize_collision_lift()
    object_z_after = _target_object_z(experiment)
    object_z_change = object_z_after - object_z_before
    success = bool(object_z_change > TASK_LIFT_Z_CHANGE)
    _record(
        experiment,
        "lift",
        success,
        "object_lifted" if success else "object_not_lifted",
        object_z_before=object_z_before,
        object_z_after=object_z_after,
        object_z_change=object_z_change,
        success_z_change=TASK_LIFT_Z_CHANGE,
        lift_height=lift_height,
    )


def execute_init(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    if hasattr(experiment, "robot") and hasattr(experiment, "_move_cartesian"):
        current = experiment.robot.get_cartesian()
        current_z = float(np.asarray(current.t, dtype=np.float64)[2])
        if current_z < PLACE_TCP_RETRACT_HEIGHT - 1e-6:
            retract_target = np.asarray(current.t, dtype=np.float64).copy()
            retract_target[2] = PLACE_TCP_RETRACT_HEIGHT
            T_retract = sm.SE3.Trans(retract_target) * sm.SE3(sm.SO3(np.asarray(current.R, dtype=np.float64), check=False))
            retract_result = experiment._move_cartesian(T_retract, float(params.get("retract_duration", 0.8)))
            if hasattr(experiment, "_record_skill_result"):
                experiment._record_skill_result(retract_result)
            _record(
                experiment,
                "go_home_retract",
                bool(getattr(retract_result, "success", True)),
                "tcp_retracted_before_go_home",
                tcp_target_pos=retract_target.tolist(),
            )
    result = experiment._move_joints(np.asarray(params.get("q", Q_SAFE), dtype=np.float64), float(params.get("duration", 1.0)))
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)


def move_lifted_object_to(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    target_name = str(params.get("target") or "").strip().lower()
    target_pos = _resolve_place_position(experiment, params)
    height = PLACE_TCP_RELEASE_HEIGHT
    target = np.array([target_pos[0], target_pos[1], height], dtype=np.float64)
    R = _rotation(experiment)
    T_target = sm.SE3.Trans(target) * sm.SE3(sm.SO3(R, check=False))
    result = experiment._move_cartesian(T_target, float(params.get("duration", 1.0)))
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    if hasattr(result, "success"):
        motion_success = bool(result.success)
        motion_reason = str(getattr(result, "reason", "ok"))
        motion_result = result.to_dict() if hasattr(result, "to_dict") else {}
    elif isinstance(result, dict):
        motion_success = bool(result.get("success", True))
        motion_reason = str(result.get("reason", "ok"))
        motion_result = dict(result)
    else:
        motion_success = True
        motion_reason = "ok"
        motion_result = {}
    _record(
        experiment,
        "move_lifted_object_to",
        motion_success,
        "lifted_object_move_executed" if motion_success else motion_reason,
        target=target_name,
        target_pos=target_pos.tolist(),
        tcp_target_pos=target.tolist(),
        fixed_place_target_z=PLACE_TARGET_Z,
        fixed_tcp_release_height=PLACE_TCP_RELEASE_HEIGHT,
        motion_result=motion_result,
    )


def _resolve_place_position(experiment: Any, params: dict[str, Any]) -> np.ndarray:
    metrics = getattr(experiment, "metrics", {}) or {}
    detected_objects = metrics.get("detected_objects") if isinstance(metrics.get("detected_objects"), dict) else {}
    target_name = str(params.get("target") or "").strip().lower()
    if not target_name:
        raise ValueError("move_lifted_object_to requires parameter target")
    if not isinstance(detected_objects, dict) or target_name not in detected_objects:
        detected_names = sorted(str(name) for name in detected_objects.keys()) if isinstance(detected_objects, dict) else []
        raise ValueError(
            "move_lifted_object_to target must be detected by a previous detect_object_pose: "
            f"target={target_name!r}, detected={detected_names}"
        )
    base = np.asarray(detected_objects[target_name], dtype=np.float64).reshape(3).copy()
    base[2] = PLACE_TARGET_Z
    return base


def _plate_position(experiment: Any) -> np.ndarray:
    plate_id = mujoco.mj_name2id(experiment.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    if plate_id < 0:
        raise ValueError("plate body not found")
    return experiment.data.body(plate_id).xpos.copy()


def check_gripper_state(experiment: Any, params: dict | None = None) -> None:
    obs = experiment._current_skill_observation() if hasattr(experiment, "_current_skill_observation") else {}
    _record(experiment, "check-gripper-state", True, "state_checked", observation=obs)


def set_gripper_force(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    experiment.metrics["gripper_force_policy"] = dict(params)
    _record(experiment, "set-gripper-force", True, "force_policy_recorded", parameters=params)


def verify_object_secured(experiment: Any, params: dict | None = None) -> None:
    obs = experiment._current_skill_observation() if hasattr(experiment, "_current_skill_observation") else {}
    contact = obs.get("contact", {}) if isinstance(obs, dict) else {}
    secured = bool(contact.get("left_contact") or contact.get("right_contact"))
    _record(experiment, "verify-object-secured", secured, "physical_contact_or_close_distance" if secured else "not_physically_secured")


def recover_slip(experiment: Any, params: dict | None = None) -> None:
    gripper_action(experiment, {"state": 0})
    create_grasp(experiment, params or {}, 0.127)
    move_pregrasp(experiment, {})
    move_grasp(experiment, {})
    gripper_action(experiment, {"state": 1})
    vertical_grasp(experiment, params or {})
    _record(experiment, "recover-slip", True, "regrasp_sequence_executed")


def replan_path(experiment: Any, params: dict | None = None) -> None:
    experiment.metrics["path_replan"] = {"triggered": True, "params": dict(_params(params))}
    move_pregrasp(experiment, {"duration": float(_params(params).get("duration", 1.0))})
    _record(experiment, "replan-path", True, "safe_pregrasp_replanned")


def avoid_obstacle(experiment: Any, params: dict | None = None) -> None:
    experiment.metrics["obstacle_avoidance"] = {"triggered": True, "params": dict(_params(params))}
    _record(experiment, "avoid-obstacle", True, "obstacle_avoidance_recorded")


def strategy_switch(experiment: Any, params: dict | None = None) -> None:
    experiment.metrics["strategy_switch"] = {"triggered": True, "params": dict(_params(params))}
    detect_object(experiment, {"target_class": "apple"})
    create_grasp(experiment, params or {}, 0.127)
    _record(experiment, "strategy-switch", True, "detect_and_grasp_replanned")
