from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import mujoco
import numpy as np

from skills.base.arm_ik_skill import ARM_JOINTS
from skills.base.gripper_skill import R1ProGripperSkill
from skills.torso_frame.continuous_grasp_executor import R1ProTorsoFrameContinuousGraspExecutor
from skills.primitives.object_manipulation_skills import _move_tcp, _set_side_gripper

DEFAULT_VISUAL_GRASP_OFFSET_Z = 0.007


def load_position_payload(path: str | Path, key: str = "median_reference") -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if key not in payload:
        raise KeyError(f"Position key not found in {path}: {key}")
    return np.asarray(payload[key], dtype=np.float64).reshape(3)


def target_position_path(target_class: str, runtime_tmp_dir: str | Path | None = None) -> Path:
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(target_class).strip().lower()).strip("_")
    if runtime_tmp_dir:
        return Path(runtime_tmp_dir) / "object_positions" / (slug or "object") / "object_position.json"
    return Path("output/object_positions") / (slug or "object") / "object_position.json"


def resolve_torso_target(params: dict[str, Any]) -> np.ndarray:
    if "position_input_path" in params:
        return load_position_payload(params["position_input_path"], str(params.get("position_key", "median_reference")))
    if "target_class" in params:
        return load_position_payload(
            target_position_path(str(params["target_class"]), params.get("_runtime_tmp_dir")),
            str(params.get("position_key", "median_reference")),
        )
    if "target_torso" in params:
        return np.asarray(params["target_torso"], dtype=np.float64).reshape(3)
    if "center_reference" in params:
        return np.asarray(params["center_reference"], dtype=np.float64).reshape(3)
    if "median_reference" in params:
        return np.asarray(params["median_reference"], dtype=np.float64).reshape(3)
    return np.array([float(params["target_x"]), float(params["target_y"]), float(params["target_z"])], dtype=np.float64)


def torso_frame_pose(model: mujoco.MjModel, data: mujoco.MjData, torso_frame: str) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)
    name = str(torso_frame or "world").strip()
    if not name or name == "world":
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id >= 0:
        return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()
    raise ValueError(f"MuJoCo reference frame not found as body or site: {torso_frame}")


def world_to_local(frame_pos: np.ndarray, frame_xmat: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
    return np.asarray(frame_xmat, dtype=np.float64).reshape(3, 3).T @ (
        np.asarray(world_pos, dtype=np.float64).reshape(3) - np.asarray(frame_pos, dtype=np.float64).reshape(3)
    )


def local_to_world(frame_pos: np.ndarray, frame_xmat: np.ndarray, local_pos: np.ndarray) -> np.ndarray:
    return np.asarray(frame_pos, dtype=np.float64).reshape(3) + np.asarray(frame_xmat, dtype=np.float64).reshape(3, 3) @ np.asarray(local_pos, dtype=np.float64).reshape(3)


def tcp_pos_world(model: mujoco.MjModel, data: mujoco.MjData, side: str, frame: str = "grasp_tool") -> np.ndarray:
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_tool")
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {side}_grasp_tool")
    return data.site_xpos[site_id].copy()


def tcp_xmat_world(model: mujoco.MjModel, data: mujoco.MjData, side: str, frame: str = "grasp_tool") -> np.ndarray:
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_tool")
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {side}_grasp_tool")
    return data.site_xmat[site_id].reshape(3, 3).copy()


def side_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    command: str,
    *,
    gripper_steps: int = 240,
    direct_qpos: bool = False,
    step_callback=None,
    hold_arm: bool = False,
) -> float:
    return _set_side_gripper(
        model,
        data,
        side,
        command,
        steps=int(gripper_steps),
        direct_qpos=bool(direct_qpos),
        step_callback=step_callback,
        hold_joint_names=ARM_JOINTS[side] if hold_arm else (),
    )


def gripper_skill() -> R1ProGripperSkill:
    return R1ProGripperSkill.from_json()


def move_tcp(model: mujoco.MjModel, data: mujoco.MjData, side: str, target_pos: np.ndarray, params: dict[str, Any], *, step_callback=None):
    return _move_tcp(model, data, side, np.asarray(target_pos, dtype=np.float64).reshape(3), params, step_callback=step_callback)


def topdown_xmat_world(model: mujoco.MjModel, data: mujoco.MjData, side: str, params: dict[str, Any]) -> np.ndarray:
    executor = R1ProTorsoFrameContinuousGraspExecutor()
    return executor._topdown_xmat_world(
        model,
        data,
        side,
        str(params.get("topdown_mode", "palm_down")),
        str(params.get("control_frame", "grasp_tool")),
    )


def follow_line_torso(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    stage: str,
    start_torso: np.ndarray,
    target_torso: np.ndarray,
    torso_frame: str,
    params: dict[str, Any],
    *,
    hold_gripper_side: str | None = None,
) -> tuple[bool, dict[str, float], dict[str, float]]:
    executor = R1ProTorsoFrameContinuousGraspExecutor()
    stage_errors: dict[str, float] = {}
    stage_orientation_errors: dict[str, float] = {}
    target_xmat_world = executor._topdown_xmat_world(
        model,
        data,
        side,
        str(params.get("topdown_mode", "palm_down")),
        str(params.get("control_frame", "grasp_tool")),
    )
    ok = executor._follow_line_torso(
        model,
        data,
        side,
        stage,
        np.asarray(start_torso, dtype=np.float64).reshape(3),
        np.asarray(target_torso, dtype=np.float64).reshape(3),
        target_xmat_world,
        torso_frame,
        stage_errors,
        stage_orientation_errors,
        waypoint_count=int(params.get("waypoint_count", params.get("waypoints", 20))),
        waypoint_steps=int(params.get("waypoint_steps", 60)),
        solve_iterations=int(params.get("solve_iterations", 120)),
        fail_threshold=float(params.get("fail_threshold", 0.002)),
        orientation_weight=float(params.get("orientation_weight", 0.35)),
        orientation_threshold=float(params.get("orientation_threshold", 1.0)),
        max_joint_step=float(params.get("max_joint_step", 0.006)),
        control_mode=str(params.get("control_mode", "actuator")),
        settle_steps=int(params.get("settle_steps", 0)),
        step_callback=None,
        control_frame=str(params.get("control_frame", "grasp_tool")),
        hold_gripper_side=hold_gripper_side,
    )
    return bool(ok), stage_errors, stage_orientation_errors


def current_tcp_torso(model: mujoco.MjModel, data: mujoco.MjData, side: str, torso_frame: str) -> np.ndarray:
    frame_pos, frame_xmat = torso_frame_pose(model, data, torso_frame)
    return world_to_local(frame_pos, frame_xmat, tcp_pos_world(model, data, side))


def vertical_lift_from_current_tcp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    torso_frame: str,
    params: dict[str, Any],
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    executor = R1ProTorsoFrameContinuousGraspExecutor()
    start_torso = executor._tcp_pos_torso(model, data, side, str(params.get("control_frame", "grasp_tool")), torso_frame)
    target_torso = start_torso + np.array([0.0, 0.0, float(params.get("lift_height", params.get("lift_dz", 0.10)))], dtype=np.float64)
    target_xmat_world = executor._tcp_xmat_world(model, data, side, str(params.get("control_frame", "grasp_tool")))
    stage_errors: dict[str, float] = {}
    stage_orientation_errors: dict[str, float] = {}
    ok = executor._vertical_cartesian_lift_torso(
        model,
        data,
        side,
        f"{side}_vertical_lift",
        start_torso,
        target_xmat_world,
        torso_frame,
        stage_errors,
        stage_orientation_errors,
        lift_height=float(params.get("lift_height", params.get("lift_dz", 0.10))),
        segment_height=float(params.get("segment_height", 0.01)),
        segment_steps=int(params.get("segment_steps", max(int(params.get("waypoint_steps", 60)), 80))),
        fail_threshold=float(params.get("fail_threshold", 0.004)),
        orientation_weight=float(params.get("orientation_weight", 0.35)),
        orientation_threshold=float(params.get("orientation_threshold", 1.0)),
        max_joint_step=float(params.get("max_joint_step", 0.003)),
        step_callback=None,
        control_frame=str(params.get("control_frame", "grasp_tool")),
        hold_gripper_side=side,
    )
    return bool(ok), start_torso, target_torso, stage_errors, stage_orientation_errors


def grasp_target_with_correction(params: dict[str, Any]) -> np.ndarray:
    target_torso = resolve_torso_target(params)
    offset = np.array(
        [
            float(params.get("grasp_offset_x", 0.0)),
            float(params.get("grasp_offset_y", 0.0)),
            float(params.get("grasp_offset_z", 0.0)),
        ],
        dtype=np.float64,
    )
    if any(key in params for key in ("position_input_path", "center_reference", "median_reference")) and bool(params.get("enable_visual_grasp_correction", True)):
        offset[2] += float(params.get("visual_grasp_offset_z", DEFAULT_VISUAL_GRASP_OFFSET_Z))
    return target_torso + offset
