from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import mujoco
import numpy as np

from skills.base.arm_ik_skill import ARM_JOINTS, TORSO_JOINTS, ArmMotionResult, R1ProArmIKSkill
from skills.base.gripper_skill import load_skill as load_gripper_skill
from r1pro_control.arm import MuJoCoSiteServoResult, R1ProMuJoCoSiteServo
from skills.primitives.grasp_attachment import attach_object_to_hand, detach_object, update_attachments


ManipulationSide = Literal["left", "right"]


@dataclass(frozen=True)
class ManipulationSkillResult:
    name: str
    side: ManipulationSide
    success: bool
    target_pos: np.ndarray | None = None
    final_error: float | None = None
    arm_result: MuJoCoSiteServoResult | ArmMotionResult | None = None
    message: str = ""


def _side(params: dict) -> ManipulationSide:
    if "side" not in params:
        raise ValueError("Provide side")
    side = params["side"]
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")
    return side


def _require_keys(params: dict, keys: tuple[str, ...] | list[str]) -> None:
    missing = [key for key in keys if key not in params]
    if missing:
        raise ValueError(f"Provide required parameter(s): {', '.join(missing)}")


def _required_float(params: dict, key: str) -> float:
    if key not in params:
        raise ValueError(f"Provide {key}")
    return float(params[key])


def _tcp_site_name(side: ManipulationSide) -> str:
    return f"{side}_hand_tcp"


def _object_pos(model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> np.ndarray:
    mujoco.mj_forward(model, data)
    if all(key in params for key in ("object_x", "object_y", "object_z")):
        return np.array([params["object_x"], params["object_y"], params["object_z"]], dtype=np.float64)

    if "object_body" in params:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, params["object_body"])
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {params['object_body']}")
        return data.xpos[body_id].copy()

    if "object_site" in params:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, params["object_site"])
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {params['object_site']}")
        return data.site_xpos[site_id].copy()

    if "object_geom" in params:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, params["object_geom"])
        if geom_id < 0:
            raise ValueError(f"MuJoCo geom not found: {params['object_geom']}")
        return data.geom_xpos[geom_id].copy()

    raise ValueError("Provide object_x/object_y/object_z or object_body/object_site/object_geom")


def _tcp_pos(model: mujoco.MjModel, data: mujoco.MjData, side: ManipulationSide) -> np.ndarray:
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _tcp_site_name(side))
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {_tcp_site_name(side)}")
    return data.site_xpos[site_id].copy()


def _tcp_object_distance(model: mujoco.MjModel, data: mujoco.MjData, side: ManipulationSide, object_body: str) -> float:
    return float(np.linalg.norm(_tcp_pos(model, data, side) - _object_pos(model, data, {"object_body": object_body})))


def _tcp_xmat(model: mujoco.MjModel, data: mujoco.MjData, side: ManipulationSide) -> np.ndarray:
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _tcp_site_name(side))
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {_tcp_site_name(side)}")
    return data.site_xmat[site_id].reshape(3, 3).copy()


def _quat_wxyz_to_xmat(quat_wxyz: np.ndarray | list[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise ValueError("target_quat_wxyz must be non-zero")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _target_xmat(model: mujoco.MjModel, data: mujoco.MjData, side: ManipulationSide, params: dict) -> np.ndarray | None:
    if "target_xmat" in params:
        raise ValueError("target_xmat is not a supported external parameter; use target_quat_wxyz")
    if "target_quat_wxyz" in params:
        return _quat_wxyz_to_xmat(params["target_quat_wxyz"])
    if not bool(params.get("lock_orientation", True)):
        return None
    key = f"_locked_tcp_xmat_{side}"
    if key not in params:
        params[key] = _tcp_xmat(model, data, side)
    return params[key]


def _approach_dir(params: dict, side: ManipulationSide) -> np.ndarray:
    _require_keys(params, ("approach_dx", "approach_dy", "approach_dz"))
    direction = np.array([params["approach_dx"], params["approach_dy"], params["approach_dz"]], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("approach direction must be non-zero")
    return direction / norm


def _grasp_pos(model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> np.ndarray:
    pos = _object_pos(model, data, params)
    _require_keys(params, ("grasp_offset_x", "grasp_offset_y", "grasp_offset_z"))
    offset = np.array(
        [
            float(params["grasp_offset_x"]),
            float(params["grasp_offset_y"]),
            float(params["grasp_offset_z"]),
        ],
        dtype=np.float64,
    )
    return pos + offset


def _pregrasp_pos(model: mujoco.MjModel, data: mujoco.MjData, params: dict, side: ManipulationSide) -> np.ndarray:
    distance = _required_float(params, "pregrasp_distance")
    return _grasp_pos(model, data, params) - _approach_dir(params, side) * distance


def _select_physical_pregrasp_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: ManipulationSide,
    params: dict,
    original: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if not bool(params.get("adaptive_pregrasp_target", False)):
        return original, {"enabled": False}
    mode = str(params.get("control_mode", params.get("tcp_control_mode", "direct_qpos"))).lower()
    if mode not in {"actuator", "joint_servo", "ik_actuator", "joint_target_velocity_limited"}:
        return original, {"enabled": False, "reason": "non_physical_tcp_mode"}

    object_pos = _object_pos(model, data, params)
    grasp = _grasp_pos(model, data, params)
    approach = _approach_dir(params, side)
    base_distance = float(params.get("pregrasp_distance", 0.12))
    candidates = [original]
    for scale in (0.75, 0.5, 0.25):
        candidates.append(grasp - approach * base_distance * scale)
    for dz in (-0.04, -0.08):
        lowered = original.copy()
        lowered[2] = max(object_pos[2] + 0.04, lowered[2] + dz)
        candidates.append(lowered)
    return _rank_physical_tcp_candidates(model, data, side, candidates, params)


def _select_physical_approach_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: ManipulationSide,
    params: dict,
    original: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if not bool(params.get("adaptive_approach_target", False)):
        return original, {"enabled": False}
    mode = str(params.get("control_mode", params.get("tcp_control_mode", "direct_qpos"))).lower()
    if mode not in {"actuator", "joint_servo", "ik_actuator", "joint_target_velocity_limited"}:
        return original, {"enabled": False, "reason": "non_physical_tcp_mode"}
    approach = _approach_dir(params, side)
    candidates = [original]
    for offset in (0.02, 0.04, 0.06):
        candidates.append(original - approach * offset)
    return _rank_physical_tcp_candidates(model, data, side, candidates, params)


def _joint_cost_summary(ik_skill: R1ProArmIKSkill, side: ManipulationSide, q_seed: np.ndarray, q_target: np.ndarray) -> dict[str, float]:
    q_seed_values = ik_skill.arm_joint_values_from_pin_q(side, q_seed, joint_names=ARM_JOINTS[side])
    q_target_values = ik_skill.arm_joint_values_from_pin_q(side, q_target, joint_names=ARM_JOINTS[side])
    delta = np.abs(q_target_values - q_seed_values)
    wrist_names = ARM_JOINTS[side][3:]
    wrist_delta = delta[3:] if len(delta) > 3 else delta
    return {
        "joint_cost": float(np.linalg.norm(delta)),
        "joint_delta_max": float(np.max(delta)) if delta.size else 0.0,
        "wrist_cost": float(np.sum(wrist_delta)),
    }


def _rank_physical_tcp_candidates(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: ManipulationSide,
    candidates: list[np.ndarray],
    params: dict,
) -> tuple[np.ndarray, dict]:
    ik_skill = R1ProArmIKSkill(params.get("urdf_path", "urdf/r1_pro_with_gripper.urdf"))
    q_seed = ik_skill.sync_q_from_mujoco(model, data)
    ranked: list[dict] = []
    for index, candidate in enumerate(candidates):
        target_ik = ik_skill.world_to_pinocchio_position(model, data, candidate)
        q_target, _ik_pos, ik_error = ik_skill.solve_position_ik(
            side,
            q_seed,
            target_ik,
            joint_names=ARM_JOINTS[side],
            posture_reference=q_seed,
            posture_gain=float(params.get("posture_gain", 0.0)),
        )
        margin = ik_skill.joint_limit_margin_from_pin_q(side, q_target, joint_names=ARM_JOINTS[side])
        joint_summary = _joint_cost_summary(ik_skill, side, q_seed, q_target)
        score = (
            float(ik_error)
            + max(0.0, 0.02 - margin) * 10.0
            + joint_summary["joint_cost"] * 0.05
            + joint_summary["wrist_cost"] * 0.02
        )
        ranked.append({
            "index": index,
            "target": np.round(candidate, 6).tolist(),
            "ik_error": round(float(ik_error), 6),
            "joint_limit_margin": round(float(margin), 6),
            "joint_cost": round(joint_summary["joint_cost"], 6),
            "joint_delta_max": round(joint_summary["joint_delta_max"], 6),
            "wrist_cost": round(joint_summary["wrist_cost"], 6),
            "score": round(float(score), 6),
        })
    ranked.sort(key=lambda item: item["score"])
    return np.asarray(ranked[0]["target"], dtype=np.float64), {
        "enabled": True,
        "selected": ranked[0],
        "candidates": ranked,
    }


def _move_tcp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: ManipulationSide,
    target_pos: np.ndarray,
    params: dict,
    *,
    step_callback: Callable[[], None] | None = None,
) -> MuJoCoSiteServoResult | ArmMotionResult:
    mode = str(params.get("control_mode", params.get("tcp_control_mode", "direct_qpos"))).lower()
    if bool(params.get("conservative_cartesian_segments", False)) and mode in {
        "actuator",
        "joint_servo",
        "ik_actuator",
        "joint_target_velocity_limited",
    }:
        segment_count = max(int(params.get("segment_count", 3)), 1)
        start = _tcp_pos(model, data, side)
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        last: ArmMotionResult | None = None
        segment_params = dict(params)
        segment_params["conservative_cartesian_segments"] = False
        segment_params["steps"] = max(int(params.get("steps", params.get("servo_steps", 1500))) // segment_count, 1)
        segment_params["settle_steps"] = max(int(params.get("settle_steps", 0 if mode == "direct_qpos" else 1500)) // segment_count, 0)
        for index in range(1, segment_count + 1):
            alpha = index / segment_count
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            waypoint = (1.0 - alpha) * start + alpha * target
            last = _move_tcp(model, data, side, waypoint, segment_params, step_callback=step_callback)  # type: ignore[assignment]
            if not getattr(last, "success", False) and bool(params.get("stop_on_segment_failure", False)):
                return last
        if last is None:
            raise RuntimeError("conservative segmented TCP move did not execute")
        return last

    if mode in ("site_servo", "mujoco_site_servo", "servo"):
        skill = R1ProMuJoCoSiteServo()
        return skill.move_to_position(
            model,
            data,
            side,
            target_pos,
            target_xmat=_target_xmat(model, data, side, params),
            steps=int(params.get("servo_steps", params.get("steps", 2500))),
            settle_steps=int(params.get("settle_steps", 1500)),
            solve_iterations=int(params.get("solve_iterations", 1000)),
            fail_threshold=float(params.get("fail_threshold", 0.03)),
            orientation_threshold=float(params.get("orientation_threshold", 1.0)),
            orientation_weight=float(params.get("orientation_weight", 0.02)),
            damping=float(params.get("damping", 1e-3)),
            max_cart_step=float(params.get("max_cart_step", 0.006)),
            max_joint_step=float(params.get("max_joint_step", 0.012)),
            posture_gain=float(params.get("posture_gain", 0.02)),
            runtime_damping=float(params.get("runtime_damping", 50.0)),
            runtime_armature=float(params.get("runtime_armature", 0.05)),
            force_scale=float(params.get("force_scale", 1.0)),
            step_callback=step_callback,
        )

    if mode == "direct_qpos":
        skill = R1ProMuJoCoSiteServo()
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_mat = _target_xmat(model, data, side, params)
        site_id = skill._site_id(model, side)
        joint_names = skill.joint_names(side)
        joint_ids = tuple(skill._joint_id(model, name) for name in joint_names)
        qpos_ids = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        dof_ids = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        ctrl_ids = np.asarray([skill._actuator_id(model, name) for name in joint_names], dtype=np.int32)
        mujoco.mj_forward(model, data)
        start_pos = data.site_xpos[site_id].copy()
        q_start = data.qpos[qpos_ids].copy()
        q_target, _solved_pos, solve_error = skill._solve_site_q_target(
            model,
            data,
            site_id,
            joint_ids,
            qpos_ids,
            dof_ids,
            ctrl_ids,
            target,
            target_mat,
            iterations=int(params.get("solve_iterations", 1000)),
            damping=float(params.get("damping", 1e-3)),
            max_cart_step=float(params.get("max_cart_step", 0.006)),
            max_joint_step=float(params.get("max_joint_step", 0.012)),
            orientation_weight=float(params.get("orientation_weight", 0.02)),
            posture_gain=float(params.get("posture_gain", 0.02)),
        )
        steps = int(params.get("steps", params.get("servo_steps", 1500)))
        for step in range(max(steps, 1)):
            alpha = 1.0 if steps <= 1 else step / (steps - 1)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            q_command = (1.0 - alpha) * q_start + alpha * q_target
            data.ctrl[ctrl_ids] = q_command
            data.qpos[qpos_ids] = q_command
            data.qvel[dof_ids] = 0.0
            mujoco.mj_forward(model, data)
            update_attachments(model, data)
            if step_callback is not None:
                step_callback()
        data.ctrl[ctrl_ids] = q_target
        data.qpos[qpos_ids] = q_target
        data.qvel[dof_ids] = 0.0
        mujoco.mj_forward(model, data)
        update_attachments(model, data)
        final_pos = data.site_xpos[site_id].copy()
        final_error = float(np.linalg.norm(final_pos - target))
        threshold = float(params.get("fail_threshold", 0.03))
        return MuJoCoSiteServoResult(
            side=side,
            site_name=skill.site_name(side),
            start_pos=start_pos,
            target_pos=target,
            final_pos=final_pos,
            final_error=final_error,
            success=final_error <= threshold,
            steps_run=max(steps, 1),
            min_error=min(float(solve_error), final_error),
            stalled=False,
            contacts=skill._contact_summary(model, data),
        )

    if mode in ("actuator", "joint_servo", "ik_actuator", "joint_target_velocity_limited"):
        direct_qpos = False
    elif mode in ("snap", "ik_direct", "pinocchio_direct"):
        direct_qpos = True
    else:
        raise ValueError(f"Unsupported TCP control_mode: {mode!r}")

    ik_skill = R1ProArmIKSkill(params.get("urdf_path", "urdf/r1_pro_with_gripper.urdf"))
    return ik_skill.move_to_position(
        model,
        data,
        side,
        np.asarray(target_pos, dtype=np.float64),
        target_xmat=_target_xmat(model, data, side, params),
        steps=int(params.get("steps", params.get("servo_steps", 1500))),
        settle_steps=int(params.get("settle_steps", 0 if direct_qpos else 1500)),
        direct_qpos=bool(params.get("direct_qpos", direct_qpos)),
        stabilize=bool(params.get("stabilize", not direct_qpos)),
        closed_loop=bool(params.get("closed_loop", True)),
        cartesian_closed_loop=bool(params.get("cartesian_closed_loop", False)),
        lock_posture=bool(params.get("lock_posture", True)),
        control_mode=mode,
        velocity_limit=params.get("velocity_limit"),
        enforce_joint_limits=bool(params.get("enforce_joint_limits", False)),
        max_joint_step=float(params.get("max_joint_step", 0.006)),
        fail_threshold=float(params.get("fail_threshold", 0.03)),
        orientation_threshold=float(params.get("orientation_threshold", 1.0)),
        orientation_weight=float(params.get("orientation_weight", 0.02)),
        force_scale=float(params.get("force_scale", 1.0)),
        runtime_damping=float(params.get("runtime_damping", 50.0)),
        runtime_armature=float(params.get("runtime_armature", 0.05)),
        posture_gain=float(params.get("posture_gain", 0.0)),
        step_callback=step_callback,
    )


def _target_pos_from_params(params: dict, *, prefix: str = "target") -> np.ndarray:
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")
    if not all(key in params for key in keys):
        raise ValueError(f"Provide {keys[0]}/{keys[1]}/{keys[2]}")
    return np.array([params[keys[0]], params[keys[1]], params[keys[2]]], dtype=np.float64)


def _place_pos(params: dict) -> np.ndarray:
    if all(key in params for key in ("place_x", "place_y", "place_z")):
        return np.array([params["place_x"], params["place_y"], params["place_z"]], dtype=np.float64)
    return _target_pos_from_params(params)


def _set_joint_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: tuple[str, ...],
    q_values: np.ndarray,
) -> None:
    for name, value in zip(joint_names, q_values):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {name}_pos")
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(value, low, high)


def _joint_values(model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return np.asarray(values, dtype=np.float64)


def _move_joints_to_posture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: tuple[str, ...],
    target_qpos: np.ndarray,
    params: dict,
    *,
    step_callback: Callable[[], None] | None = None,
) -> tuple[np.ndarray, float, bool]:
    target = np.asarray(target_qpos, dtype=np.float64)
    if target.shape != (len(joint_names),):
        raise ValueError(f"Expected {len(joint_names)} joint targets, got shape {target.shape}")
    current = _joint_values(model, data, joint_names)
    max_step = float(params.get("max_joint_step", 0.006))
    steps = int(params.get("steps", 1500))
    settle_steps = int(params.get("settle_steps", 1000))
    fail_threshold = float(params.get("fail_threshold", 0.03))
    direct_qpos = bool(params.get("direct_qpos", False))
    locked_joint_ids = set()
    if direct_qpos:
        controlled = set(joint_names)
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name and name not in controlled:
                locked_joint_ids.add(joint_id)
        locked_qpos = data.qpos.copy()
        locked_qvel = data.qvel.copy()
    else:
        locked_qpos = None
        locked_qvel = None

    def apply_direct_values(values: np.ndarray) -> None:
        for name, value in zip(joint_names, values):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint_id]] = value
        if locked_qpos is not None and locked_qvel is not None:
            for joint_id in locked_joint_ids:
                qpos_id = model.jnt_qposadr[joint_id]
                dof_id = model.jnt_dofadr[joint_id]
                if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_size = 7
                    qvel_size = 6
                elif model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL:
                    qpos_size = 4
                    qvel_size = 3
                else:
                    qpos_size = 1
                    qvel_size = 1
                data.qpos[qpos_id : qpos_id + qpos_size] = locked_qpos[qpos_id : qpos_id + qpos_size]
                data.qvel[dof_id : dof_id + qvel_size] = locked_qvel[dof_id : dof_id + qvel_size]
            data.qvel[:] = 0.0

    for _ in range(max(steps, 0)):
        current = current + np.clip(target - current, -max_step, max_step)
        _set_joint_ctrl(model, data, joint_names, current)
        if direct_qpos:
            apply_direct_values(current)
            mujoco.mj_forward(model, data)
            update_attachments(model, data)
        else:
            mujoco.mj_step(model, data)
        if step_callback is not None:
            step_callback()
    _set_joint_ctrl(model, data, joint_names, target)
    if direct_qpos:
        apply_direct_values(target)
        mujoco.mj_forward(model, data)
        update_attachments(model, data)
    for _ in range(max(settle_steps, 0)):
        if direct_qpos:
            apply_direct_values(target)
            mujoco.mj_forward(model, data)
            update_attachments(model, data)
        else:
            mujoco.mj_step(model, data)
        if step_callback is not None:
            step_callback()
    final = _joint_values(model, data, joint_names)
    error = float(np.linalg.norm(final - target))
    return final, error, error <= fail_threshold


def _set_side_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: ManipulationSide,
    command: float | str,
    *,
    steps: int = 240,
    direct_qpos: bool = False,
    closure_bias: float = 0.0,
    hold_joint_names: tuple[str, ...] = (),
    step_callback: Callable[[], None] | None = None,
) -> float:
    gripper = load_gripper_skill()
    value = gripper.command_value(command)
    prefix = f"{side}_gripper_finger_joint"
    gripper_joint_ids = []
    gripper_actuator_ids = []
    split_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper_fingers_pos")
    if split_actuator_id >= 0:
        gripper_actuator_ids.append(split_actuator_id)
    for i in (1, 2):
        joint_name = f"{prefix}{i}"
        if split_actuator_id < 0:
            actuator_name = f"{joint_name}_pos"
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                raise ValueError(f"MuJoCo actuator not found: {actuator_name}")
            gripper_actuator_ids.append(actuator_id)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        gripper_joint_ids.append(joint_id)
    if isinstance(command, str) and command == "close":
        joint_highs = [
            float(model.jnt_range[joint_id][1])
            for joint_id in gripper_joint_ids
            if bool(model.jnt_limited[joint_id])
        ]
        actuator_highs = [
            float(model.actuator_ctrlrange[actuator_id][1])
            for actuator_id in gripper_actuator_ids
            if bool(model.actuator_ctrllimited[actuator_id])
        ]
        value = min([*joint_highs, *actuator_highs], default=value)
        value = value + float(closure_bias)
    start_values = {
        joint_id: float(data.qpos[model.jnt_qposadr[joint_id]])
        for joint_id in gripper_joint_ids
    }
    start_scalar = float(np.mean(tuple(start_values.values()))) if start_values else float(value)
    hold_joint_targets = None
    if hold_joint_names:
        hold_joint_targets = _joint_values(model, data, hold_joint_names)
    total_steps = max(steps, 1)
    for step in range(total_steps):
        alpha = 1.0 if total_steps <= 1 else (step + 1) / total_steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        command_value = (1.0 - alpha) * start_scalar + alpha * value
        for actuator_id in gripper_actuator_ids:
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(command_value, low, high)
        if hold_joint_targets is not None:
            _set_joint_ctrl(model, data, hold_joint_names, hold_joint_targets)
        if direct_qpos:
            for joint_id in gripper_joint_ids:
                qadr = model.jnt_qposadr[joint_id]
                dadr = model.jnt_dofadr[joint_id]
                joint_low, joint_high = model.jnt_range[joint_id]
                data.qpos[qadr] = np.clip(command_value, joint_low, joint_high)
                data.qvel[dadr] = 0.0
            mujoco.mj_forward(model, data)
        else:
            mujoco.mj_step(model, data)
        if step_callback is not None:
            step_callback()
    return value


class BaseObjectManipulationSkill:
    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            self.config = {}
        else:
            self.config = json.loads(Path(config_path).read_text())
        self.name = self.config.get("name", self.__class__.__name__)


class MoveToPregraspSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = _side(params)
        target = _pregrasp_pos(model, data, params, side)
        target, target_report = _select_physical_pregrasp_target(model, data, side, params, target)
        merged_params = dict(params)
        merged_params["adaptive_pregrasp_target"] = False
        merged_params["conservative_cartesian_segments"] = True
        merged_params["segment_count"] = int(params.get("pregrasp_segment_count", params.get("segment_count", 6)))
        merged_params["stop_on_segment_failure"] = True
        result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        message = (
            "cartesian_pregrasp="
            + str({
                "mode": "current_to_pregrasp",
                "target": np.round(target, 6).tolist(),
                "segment_count": int(merged_params["segment_count"]),
                "adaptive_pregrasp": target_report,
            })
        )
        return ManipulationSkillResult(self.name, side, result.success, target, result.final_error, result, message=message)


class ApproachObjectSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = _side(params)
        target = _grasp_pos(model, data, params)
        merged_params = dict(params)
        merged_params["adaptive_approach_target"] = False
        merged_params["conservative_cartesian_segments"] = True
        merged_params["segment_count"] = int(params.get("approach_segment_count", params.get("segment_count", 6)))
        merged_params["stop_on_segment_failure"] = True
        result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        start = getattr(result, "start_pos", None)
        message = (
            "cartesian_grasp_approach="
            + str({
                "mode": "pregrasp_to_grasp",
                "start": None if start is None else np.round(np.asarray(start, dtype=np.float64), 6).tolist(),
                "target": np.round(target, 6).tolist(),
                "segment_count": int(merged_params["segment_count"]),
            })
        )
        return ManipulationSkillResult(self.name, side, result.success, target, result.final_error, result, message=message)


class PlaceObjectSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = _side(params)
        place = _place_pos(params)
        _require_keys(params, ("place_offset_x", "place_offset_y", "place_offset_z"))
        target = place + np.array(
            [
                float(params["place_offset_x"]),
                float(params["place_offset_y"]),
                float(params["place_offset_z"]),
            ],
            dtype=np.float64,
        )
        merged_params = {**self.config.get("control_defaults", {}), **params}
        result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        return ManipulationSkillResult(self.name, side, result.success, target, result.final_error, result)


class FixedSideGripperCommandSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = self.config.get("side")
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported fixed gripper side: {side!r}")
        command = params.get("command", self.config.get("command", "close"))
        value = _set_side_gripper(
            model,
            data,
            side,
            command,
            steps=int(params.get("gripper_steps", self.config.get("gripper_steps", 240))),
            direct_qpos=bool(params.get("direct_qpos", self.config.get("direct_qpos", False))),
            step_callback=step_callback,
            closure_bias=float(params.get("closure_bias", 0.0)),
        )
        attached = ""
        direct_qpos = bool(params.get("direct_qpos", self.config.get("direct_qpos", False)))
        if str(command).lower() == "close" and "object_body" in params:
            object_body = str(params["object_body"])
            attach_on_close = bool(params.get("attach_on_close", self.config.get("attach_on_close", False)))
            if attach_on_close and direct_qpos:
                attach_object_to_hand(model, data, side, object_body)
                attached = f", attached_object={object_body}"
            elif attach_on_close and not direct_qpos:
                attached = ", attach_skipped=physical_mode"
        return ManipulationSkillResult(self.name, side, True, message=f"gripper_value={value:.6f}{attached}")


class OpenGripperReleaseSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = _side(params)
        if bool(params.get("detach_on_release", True)):
            detach_object(model, data, side)
        value = _set_side_gripper(
            model,
            data,
            side,
            "open",
            steps=int(params.get("gripper_steps", self.config.get("gripper_steps", 360))),
            direct_qpos=bool(params.get("direct_qpos", self.config.get("direct_qpos", False))),
            step_callback=step_callback,
        )
        settle_steps = int(params.get("settle_steps", self.config.get("settle_steps", 240)))
        direct_qpos = bool(params.get("direct_qpos", self.config.get("direct_qpos", False)))
        for _ in range(max(settle_steps, 0)):
            if direct_qpos:
                mujoco.mj_forward(model, data)
            else:
                mujoco.mj_step(model, data)
            if step_callback is not None:
                step_callback()
        return ManipulationSkillResult(self.name, side, True, message=f"released_gripper_value={value:.6f}")


class FixedSideVerticalLiftSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = self.config.get("side")
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported fixed lift side: {side!r}")
        current = _tcp_pos(model, data, side)
        _require_keys(params, ("lift_dx", "lift_dy"))
        if "lift_dz" not in params and "lift_height" not in params:
            raise ValueError("Provide lift_dz or lift_height")
        lift_vector = np.array(
            [
                float(params["lift_dx"]),
                float(params["lift_dy"]),
                float(params["lift_dz"] if "lift_dz" in params else params["lift_height"]),
            ],
            dtype=np.float64,
        )
        target = current + lift_vector
        control_defaults = self.config.get("control_defaults", {})
        fail_threshold = float(params.get("fail_threshold", control_defaults.get("fail_threshold", 0.02)))
        ik_skill = R1ProArmIKSkill(self.config.get("urdf_path", "urdf/r1_pro_with_gripper.urdf"))
        qpos_start = data.qpos.copy()
        qvel_start = data.qvel.copy()
        ctrl_start = data.ctrl.copy()
        def update_then_callback() -> None:
            update_attachments(model, data, side)
            if step_callback is not None:
                step_callback()

        result = ik_skill.move_to_position(
            model,
            data,
            side,
            target,
            target_xmat=_target_xmat(model, data, side, params),
            steps=int(params.get("steps", control_defaults.get("steps", 1500))),
            settle_steps=int(params.get("settle_steps", control_defaults.get("settle_steps", 3000))),
            direct_qpos=bool(params.get("direct_qpos", control_defaults.get("direct_qpos", False))),
            stabilize=bool(params.get("stabilize", not bool(params.get("direct_qpos", control_defaults.get("direct_qpos", False))))),
            closed_loop=bool(params.get("closed_loop", True)),
            cartesian_closed_loop=bool(params.get("cartesian_closed_loop", False)),
            lock_posture=bool(params.get("lock_posture", True)),
            max_joint_step=float(params.get("max_joint_step", control_defaults.get("max_joint_step", 0.006))),
            fail_threshold=fail_threshold,
            orientation_threshold=float(params.get("orientation_threshold", control_defaults.get("orientation_threshold", 1.0))),
            orientation_weight=float(params.get("orientation_weight", control_defaults.get("orientation_weight", 0.02))),
            force_scale=float(params.get("force_scale", control_defaults.get("force_scale", 1.0))),
            enforce_joint_limits=bool(params.get("enforce_joint_limits", control_defaults.get("enforce_joint_limits", False))),
            step_callback=update_then_callback,
        )
        update_attachments(model, data, side)
        achieved_lift = float(_tcp_pos(model, data, side)[2] - current[2])
        requested_lift = float(lift_vector[2])
        lift_tolerance = float(params.get("lift_tolerance", control_defaults.get("lift_tolerance", max(fail_threshold, 0.05))))
        if result.success or achieved_lift >= max(0.0, requested_lift - lift_tolerance):
            return ManipulationSkillResult(self.name, side, result.success, target, result.final_error, result)
        if bool(params.get("direct_qpos", control_defaults.get("direct_qpos", False))):
            return ManipulationSkillResult(self.name, side, result.success, target, result.final_error, result)

        data.qpos[:] = qpos_start
        data.qvel[:] = qvel_start
        data.ctrl[:] = ctrl_start
        mujoco.mj_forward(model, data)

        servo = R1ProMuJoCoSiteServo()
        waypoint_count = int(params.get("waypoints", control_defaults.get("waypoints", 20)))
        waypoint_steps = int(params.get("waypoint_steps", control_defaults.get("waypoint_steps", 80)))
        last_result: MuJoCoSiteServoResult | None = None
        for i in range(1, max(waypoint_count, 1) + 1):
            alpha = i / max(waypoint_count, 1)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            waypoint = (1.0 - alpha) * current + alpha * target
            last_result = servo.move_to_position(
                model,
                data,
                side,
                waypoint,
                target_xmat=_target_xmat(model, data, side, params),
                steps=waypoint_steps,
                settle_steps=0,
                solve_iterations=int(params.get("solve_iterations", control_defaults.get("solve_iterations", 120))),
                fail_threshold=fail_threshold,
                orientation_threshold=float(params.get("orientation_threshold", control_defaults.get("orientation_threshold", 1.0))),
                orientation_weight=float(params.get("orientation_weight", control_defaults.get("orientation_weight", 0.02))),
                max_joint_step=float(params.get("max_joint_step", control_defaults.get("max_joint_step", 0.012))),
                posture_gain=float(params.get("posture_gain", control_defaults.get("posture_gain", 0.12))),
                runtime_damping=float(params.get("runtime_damping", control_defaults.get("runtime_damping", 35.0))),
                runtime_armature=float(params.get("runtime_armature", control_defaults.get("runtime_armature", 0.05))),
                step_callback=step_callback,
            )
        if last_result is None:
            return ManipulationSkillResult(self.name, side, result.success, target, result.final_error, result)
        final_lift = float(_tcp_pos(model, data, side)[2] - current[2])
        success = last_result.success or final_lift >= max(0.0, requested_lift - lift_tolerance)
        if success and not last_result.success:
            last_result = MuJoCoSiteServoResult(
                side=last_result.side,
                site_name=last_result.site_name,
                start_pos=last_result.start_pos,
                target_pos=last_result.target_pos,
                final_pos=last_result.final_pos,
                final_error=last_result.final_error,
                success=True,
                steps_run=last_result.steps_run,
                min_error=last_result.min_error,
                stalled=last_result.stalled,
                contacts=last_result.contacts,
            )
        return ManipulationSkillResult(self.name, side, last_result.success, target, last_result.final_error, last_result)


class VerifyGraspSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = _side(params)
        object_pos = _object_pos(model, data, params)
        tcp_pos = _tcp_pos(model, data, side)
        max_distance = _required_float(params, "max_grasp_distance")
        min_lift = _required_float(params, "min_lift")
        initial_z = params.get("initial_object_z")
        distance = float(np.linalg.norm(object_pos - tcp_pos))
        lifted = True if initial_z is None else object_pos[2] - float(initial_z) >= min_lift
        success = distance <= max_distance and lifted
        message = f"distance={distance:.6f}"
        if initial_z is not None:
            message += f", lift={object_pos[2] - float(initial_z):.6f}"
        return ManipulationSkillResult(self.name, side, success, object_pos, distance, None, message)


class VerifyPlaceZoneSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = _side(params)
        object_pos = _object_pos(model, data, params)
        place = _place_pos(params)
        xy_error = float(np.linalg.norm(object_pos[:2] - place[:2]))
        z_error = abs(float(object_pos[2] - place[2]))
        max_xy_error = _required_float(params, "max_xy_error")
        max_z_error = _required_float(params, "max_z_error")
        success = xy_error <= max_xy_error and z_error <= max_z_error
        message = f"xy_error={xy_error:.6f}, z_error={z_error:.6f}"
        return ManipulationSkillResult(self.name, side, success, place, max(xy_error, z_error), None, message)


class RetreatHandSkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        side = self.config.get("side")
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported fixed retreat side: {side!r}")
        current = _tcp_pos(model, data, side)
        _require_keys(params, ("retreat_dx", "retreat_dy", "retreat_dz"))
        retreat = np.array(
            [
                float(params["retreat_dx"]),
                float(params["retreat_dy"]),
                float(params["retreat_dz"]),
            ],
            dtype=np.float64,
        )
        target = current + retreat
        merged_params = {**self.config.get("control_defaults", {}), **params}
        result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        return ManipulationSkillResult(self.name, side, result.success, target, result.final_error, result)


class GoHomeUpperBodySkill(BaseObjectManipulationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> ManipulationSkillResult:
        joint_names = TORSO_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"]
        if "target_qpos" not in params:
            raise ValueError("Provide target_qpos")
        target = np.asarray(params["target_qpos"], dtype=np.float64)
        merged_params = {**self.config.get("control_defaults", {}), **params}
        final, error, success = _move_joints_to_posture(
            model,
            data,
            joint_names,
            target,
            merged_params,
            step_callback=step_callback,
        )
        return ManipulationSkillResult(
            self.name,
            "left",
            success,
            final,
            error,
            None,
            message=f"upper_body_joint_error={error:.6f}",
        )
