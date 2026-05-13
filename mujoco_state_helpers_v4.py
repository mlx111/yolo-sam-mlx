from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np


LEFT_PAD_GEOMS = {"left_pad1", "left_pad2"}
RIGHT_PAD_GEOMS = {"right_pad1", "right_pad2"}


def _safe_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    if obj_id < 0:
        return ""
    return mujoco.mj_id2name(model, obj_type, obj_id) or ""


def _pose_summary_from_body(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> Optional[Dict[str, Any]]:
    if body_id < 0:
        return None
    xpos = np.asarray(data.body(body_id).xpos, dtype=np.float64)
    xmat = np.asarray(data.body(body_id).xmat, dtype=np.float64).reshape(3, 3)
    return {
        "translation": np.round(xpos, 6).tolist(),
        "rotation": np.round(xmat, 6).tolist(),
        "z_axis": np.round(xmat[:, 2], 6).tolist(),
    }


def _pose_summary_from_se3(pose: Any) -> Optional[Dict[str, Any]]:
    if pose is None:
        return None
    try:
        translation = np.asarray(pose.t, dtype=np.float64)
        rotation = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
    except Exception:  # noqa: BLE001
        return None
    return {
        "translation": np.round(translation, 6).tolist(),
        "rotation": np.round(rotation, 6).tolist(),
        "z_axis": np.round(rotation[:, 2], 6).tolist(),
    }


def resolve_target_body_name(model: mujoco.MjModel, target: Optional[str]) -> Optional[str]:
    if model is None or not target:
        return None
    normalized = str(target).strip().lower()
    candidates: list[str] = []
    for body_id in range(model.nbody):
        name = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name:
            continue
        body_name = name.lower()
        if body_name == normalized or body_name == f"{normalized}0":
            return name
        if body_name.startswith(normalized):
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _rotation_error_deg(rotation_a: Optional[np.ndarray], rotation_b: Optional[np.ndarray]) -> Optional[float]:
    if rotation_a is None or rotation_b is None:
        return None
    delta = rotation_a.T @ rotation_b
    trace = float(np.trace(delta))
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cosine)))


def _contact_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_body_name: Optional[str],
) -> Dict[str, Any]:
    target_body_name = target_body_name or ""
    left_count = 0
    right_count = 0

    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_name = _safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
        geom2_name = _safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
        body1_name = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1]))
        body2_name = _safe_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2]))

        if target_body_name:
            if body1_name != target_body_name and body2_name != target_body_name:
                continue

        geom_names = {geom1_name, geom2_name}
        if geom_names & LEFT_PAD_GEOMS:
            left_count += 1
        if geom_names & RIGHT_PAD_GEOMS:
            right_count += 1

    return {
        "left_contact": left_count > 0,
        "right_contact": right_count > 0,
        "left_contact_count": int(left_count),
        "right_contact_count": int(right_count),
    }


def capture_world_state(
    env: Any,
    *,
    robot: Any = None,
    target: Optional[str] = None,
    gripper_signal: Optional[float] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "target_name": target or "",
        "gripper_signal": None if gripper_signal is None else float(gripper_signal),
    }
    if env is None or getattr(env, "mj_model", None) is None or getattr(env, "mj_data", None) is None:
        return state

    model = env.mj_model
    data = env.mj_data
    target_body_name = resolve_target_body_name(model, target)
    state["target_body_name"] = target_body_name

    if robot is not None:
        try:
            tcp_pose = robot.get_cartesian()
        except Exception:  # noqa: BLE001
            tcp_pose = None
        state["tcp_pose"] = _pose_summary_from_se3(tcp_pose)
        try:
            state["robot_joint"] = np.round(np.asarray(robot.get_joint(), dtype=np.float64), 6).tolist()
        except Exception:  # noqa: BLE001
            state["robot_joint"] = []
    else:
        state["tcp_pose"] = None
        state["robot_joint"] = []

    if target_body_name:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    else:
        body_id = -1
    state["target_pose"] = _pose_summary_from_body(model, data, body_id)
    state["contact"] = _contact_summary(model, data, target_body_name=target_body_name)
    state["sim_time"] = float(data.time)
    return state


def summarize_se3_pose(pose: Any) -> Optional[Dict[str, Any]]:
    return _pose_summary_from_se3(pose)


def position_error(before_pose: Optional[Dict[str, Any]], after_pose: Optional[Dict[str, Any]]) -> Optional[float]:
    if not before_pose or not after_pose:
        return None
    before_t = np.asarray(before_pose["translation"], dtype=np.float64)
    after_t = np.asarray(after_pose["translation"], dtype=np.float64)
    return float(np.linalg.norm(after_t - before_t))


def rotation_error_deg(before_pose: Optional[Dict[str, Any]], after_pose: Optional[Dict[str, Any]]) -> Optional[float]:
    if not before_pose or not after_pose:
        return None
    before_r = np.asarray(before_pose["rotation"], dtype=np.float64)
    after_r = np.asarray(after_pose["rotation"], dtype=np.float64)
    return _rotation_error_deg(before_r, after_r)


def target_xy_error(target_pose: Optional[Dict[str, Any]], place_target_xy: Optional[list[float]]) -> Optional[float]:
    if not target_pose or place_target_xy is None:
        return None
    target_xy = np.asarray(target_pose["translation"][:2], dtype=np.float64)
    place_xy = np.asarray(place_target_xy[:2], dtype=np.float64)
    return float(np.linalg.norm(target_xy - place_xy))


def relative_translation_error(
    tcp_pose: Optional[Dict[str, Any]],
    target_pose: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not tcp_pose or not target_pose:
        return None
    tcp_t = np.asarray(tcp_pose["translation"], dtype=np.float64)
    target_t = np.asarray(target_pose["translation"], dtype=np.float64)
    return float(np.linalg.norm(target_t - tcp_t))


def lift_delta_z(before_pose: Optional[Dict[str, Any]], after_pose: Optional[Dict[str, Any]]) -> Optional[float]:
    if not before_pose or not after_pose:
        return None
    return float(after_pose["translation"][2] - before_pose["translation"][2])


def pose_z(pose: Optional[Dict[str, Any]]) -> Optional[float]:
    if not pose:
        return None
    translation = pose.get("translation")
    if not translation or len(translation) < 3:
        return None
    return float(translation[2])


def relative_translation_delta(
    before_tcp_pose: Optional[Dict[str, Any]],
    before_target_pose: Optional[Dict[str, Any]],
    after_tcp_pose: Optional[Dict[str, Any]],
    after_target_pose: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not before_tcp_pose or not before_target_pose or not after_tcp_pose or not after_target_pose:
        return None
    before_tcp = np.asarray(before_tcp_pose["translation"], dtype=np.float64)
    before_target = np.asarray(before_target_pose["translation"], dtype=np.float64)
    after_tcp = np.asarray(after_tcp_pose["translation"], dtype=np.float64)
    after_target = np.asarray(after_target_pose["translation"], dtype=np.float64)
    before_rel = before_target - before_tcp
    after_rel = after_target - after_tcp
    return float(np.linalg.norm(after_rel - before_rel))


def joint_error_norm(current_joint: list[float], home_joint: Optional[np.ndarray]) -> Optional[float]:
    if not current_joint or home_joint is None:
        return None
    current = np.asarray(current_joint, dtype=np.float64)
    home = np.asarray(home_joint, dtype=np.float64)
    if current.shape != home.shape:
        return None
    return float(np.max(np.abs(current - home)))
