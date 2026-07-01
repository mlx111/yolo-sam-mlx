"""Anomaly injection functions for MuJoCo UR5e experiments."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body {body_name!r} not found")
    return body_id


def _get_object_qpos_adr(model: mujoco.MjModel, body_name: str) -> tuple[int | None, int | None]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None, None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id:
            return int(model.jnt_qposadr[j]), int(body_id)
    return None, int(body_id)


def _get_object_dof_adr(model: mujoco.MjModel, body_id: int) -> int | None:
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == body_id and hasattr(model, "jnt_dofadr"):
            return int(model.jnt_dofadr[j])
    return None


def _set_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pos: np.ndarray,
    quat: np.ndarray | None = None,
) -> dict[str, Any]:
    qpos_adr, body_id = _get_object_qpos_adr(model, body_name)
    if qpos_adr is None or body_id is None:
        raise ValueError(f"body {body_name!r} has no free joint")
    before = data.body(body_id).xpos.copy()
    data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
    if quat is not None:
        data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quat, dtype=np.float64)
    dof_adr = _get_object_dof_adr(model, body_id)
    if dof_adr is not None:
        data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return {"body": body_name, "before_pos": before, "after_pos": data.body(body_id).xpos.copy()}


def inject_grasp_miss(model, data, apple_body_id, apple_initial_pos, apple_initial_quat):
    return {
        "type": "grasp_miss",
        **_set_body_pose(model, data, "apple0", np.asarray(apple_initial_pos), np.asarray(apple_initial_quat)),
    }


def inject_object_displaced(model, data, body_name, dx=0.05, dy=0.03, dz=0.0):
    body_id = _body_id(model, body_name)
    target = data.body(body_id).xpos.copy() + np.array([dx, dy, dz], dtype=np.float64)
    return _jsonable({
        "type": "object_displaced",
        "offset": [dx, dy, dz],
        **_set_body_pose(model, data, body_name, target),
    })


def inject_transport_displace(model, data, body_name, dx=0.06, dy=-0.04, dz=0.0):
    result = inject_object_displaced(model, data, body_name, dx=dx, dy=dy, dz=dz)
    result["type"] = "transport_displace"
    return result


def inject_collision(model, data, body_name, vx=1.2, vy=0.9, vz=2.5):
    body_id = _body_id(model, body_name)
    dof_adr = _get_object_dof_adr(model, body_id)
    if dof_adr is not None:
        data.qvel[dof_adr:dof_adr + 3] = np.array([vx, vy, vz], dtype=np.float64)
    mujoco.mj_forward(model, data)
    return _jsonable({"type": "collision", "object": body_name, "velocity": [vx, vy, vz]})


def inject_gripper_fail(model, data):
    for joint_name in ["right_driver_joint", "left_driver_joint"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = 0.0
    mujoco.mj_forward(model, data)
    return {"type": "gripper_fail", "detail": "fingers forced open"}


def inject_partial_close(model, data, close_ratio=0.15):
    close_ratio = max(0.0, min(float(close_ratio), 1.0))
    for joint_name in ["right_driver_joint", "left_driver_joint"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = close_ratio
    mujoco.mj_forward(model, data)
    return {"type": "partial_close", "close_ratio": close_ratio}


def inject_premature_close_push(model, data, body_name, dx=0.05, dy=0.025, dz=0.0):
    result = inject_object_displaced(model, data, body_name, dx=dx, dy=dy, dz=dz)
    result["type"] = "premature_close_push"
    return result


def inject_slip(model, data, apple_body_id, apple_initial_pos, apple_initial_quat):
    return _jsonable({
        "type": "slip",
        **_set_body_pose(model, data, "apple0", np.asarray(apple_initial_pos), np.asarray(apple_initial_quat)),
    })


def setup_incipient_slip(model, data, apple_body_id):
    apple_qpos_adr = None
    apple_dof_adr = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == apple_body_id:
            apple_qpos_adr = int(model.jnt_qposadr[j])
            if hasattr(model, "jnt_dofadr"):
                apple_dof_adr = int(model.jnt_dofadr[j])
            break
    if apple_qpos_adr is None:
        raise ValueError("apple body not found")
    initial_z = float(data.qpos[apple_qpos_adr + 2])
    return {
        "apple_qpos_adr": apple_qpos_adr,
        "apple_dof_adr": apple_dof_adr,
        "initial_z": initial_z,
        "table_z": initial_z,
    }


def apply_incipient_slip_step(model, data, apple_body_id, step, total_steps, slip_state):
    frac = float(step) / max(int(total_steps), 1)
    table_z = float(slip_state["table_z"])
    adr = int(slip_state["apple_qpos_adr"])
    if frac < 0.2:
        return 0.0
    if frac < 0.7:
        phase_frac = (frac - 0.2) / 0.5
        total_drift = 0.20 * phase_frac ** 2
        prev_frac = max(0.0, (float(step) - 1.0) / max(int(total_steps), 1))
        if prev_frac < 0.2:
            prev_drift = 0.0
        elif prev_frac < 0.7:
            prev_drift = 0.20 * ((prev_frac - 0.2) / 0.5) ** 2
        else:
            prev_drift = 0.20
        step_drift = total_drift - prev_drift
        data.qpos[adr + 2] = max(float(data.qpos[adr + 2]) - step_drift, table_z)
        mujoco.mj_forward(model, data)
        return float(step_drift)
    data.qpos[adr + 2] = table_z
    if slip_state.get("apple_dof_adr") is not None:
        dof_adr = int(slip_state["apple_dof_adr"])
        data.qvel[dof_adr:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return 0.0


def inject_incipient_slip(model, data, apple_body_id, step=0, total_steps=1, slip_state=None):
    state = slip_state or setup_incipient_slip(model, data, apple_body_id)
    drift = apply_incipient_slip_step(model, data, apple_body_id, step, total_steps, state)
    return _jsonable({"type": "incipient_slip", "drift": drift, "state": state})
