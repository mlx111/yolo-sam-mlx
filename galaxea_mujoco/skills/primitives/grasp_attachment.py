from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class GraspAttachment:
    side: str
    object_body: str
    joint_id: int
    qpos_id: int
    dof_id: int
    rel_pos: np.ndarray
    rel_xmat: np.ndarray


@dataclass
class DualGraspAttachment:
    left_side: str
    right_side: str
    object_body: str
    joint_id: int
    qpos_id: int
    dof_id: int
    rel_pos: np.ndarray
    rel_xmat: np.ndarray


_ATTACHMENTS: dict[int, dict[str, GraspAttachment]] = {}
_DUAL_ATTACHMENTS: dict[int, DualGraspAttachment] = {}


def _tcp_site_id(model: mujoco.MjModel, side: str) -> int:
    site_name = f"{side}_hand_tcp"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    return site_id


def _object_free_joint(model: mujoco.MjModel, object_body: str) -> tuple[int, int, int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {object_body}")
    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    for joint_id in range(jnt_adr, jnt_adr + jnt_num):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    raise ValueError(f"Body {object_body!r} does not have a free joint")


def _xmat_to_quat_wxyz(xmat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(xmat, dtype=np.float64).reshape(9))
    return quat


def _dual_hand_frame(model: mujoco.MjModel, data: mujoco.MjData, left_side: str, right_side: str) -> tuple[np.ndarray, np.ndarray]:
    left_site = _tcp_site_id(model, left_side)
    right_site = _tcp_site_id(model, right_side)
    left_pos = data.site_xpos[left_site].copy()
    right_pos = data.site_xpos[right_site].copy()
    origin = 0.5 * (left_pos + right_pos)
    y_axis = left_pos - right_pos
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-9:
        raise ValueError("Dual hand frame is degenerate: left/right TCPs overlap")
    y_axis = y_axis / y_norm
    z_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = z_hint - y_axis * float(np.dot(z_hint, y_axis))
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        z_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        z_axis = z_hint - y_axis * float(np.dot(z_hint, y_axis))
        z_norm = float(np.linalg.norm(z_axis))
    z_axis = z_axis / z_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    return origin, np.column_stack((x_axis, y_axis, z_axis))


def attach_object_to_hand(model: mujoco.MjModel, data: mujoco.MjData, side: str, object_body: str) -> GraspAttachment:
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")
    mujoco.mj_forward(model, data)
    site_id = _tcp_site_id(model, side)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {object_body}")
    joint_id, qpos_id, dof_id = _object_free_joint(model, object_body)
    tcp_pos = data.site_xpos[site_id].copy()
    tcp_xmat = data.site_xmat[site_id].reshape(3, 3).copy()
    object_pos = data.xpos[body_id].copy()
    object_xmat = data.xmat[body_id].reshape(3, 3).copy()
    attachment = GraspAttachment(
        side=side,
        object_body=object_body,
        joint_id=joint_id,
        qpos_id=qpos_id,
        dof_id=dof_id,
        rel_pos=tcp_xmat.T @ (object_pos - tcp_pos),
        rel_xmat=tcp_xmat.T @ object_xmat,
    )
    _ATTACHMENTS.setdefault(id(data), {})[side] = attachment
    apply_attachment(model, data, attachment)
    return attachment


def attach_object_to_hands(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body: str,
    *,
    left_side: str = "left",
    right_side: str = "right",
) -> DualGraspAttachment:
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {object_body}")
    joint_id, qpos_id, dof_id = _object_free_joint(model, object_body)
    frame_pos, frame_xmat = _dual_hand_frame(model, data, left_side, right_side)
    object_pos = data.xpos[body_id].copy()
    object_xmat = data.xmat[body_id].reshape(3, 3).copy()
    attachment = DualGraspAttachment(
        left_side=left_side,
        right_side=right_side,
        object_body=object_body,
        joint_id=joint_id,
        qpos_id=qpos_id,
        dof_id=dof_id,
        rel_pos=frame_xmat.T @ (object_pos - frame_pos),
        rel_xmat=frame_xmat.T @ object_xmat,
    )
    _DUAL_ATTACHMENTS[id(data)] = attachment
    apply_dual_attachment(model, data, attachment)
    return attachment


def apply_attachment(model: mujoco.MjModel, data: mujoco.MjData, attachment: GraspAttachment) -> None:
    site_id = _tcp_site_id(model, attachment.side)
    tcp_pos = data.site_xpos[site_id].copy()
    tcp_xmat = data.site_xmat[site_id].reshape(3, 3).copy()
    object_pos = tcp_pos + tcp_xmat @ attachment.rel_pos
    object_xmat = tcp_xmat @ attachment.rel_xmat
    data.qpos[attachment.qpos_id : attachment.qpos_id + 3] = object_pos
    data.qpos[attachment.qpos_id + 3 : attachment.qpos_id + 7] = _xmat_to_quat_wxyz(object_xmat)
    data.qvel[attachment.dof_id : attachment.dof_id + 6] = 0.0
    mujoco.mj_forward(model, data)


def apply_dual_attachment(model: mujoco.MjModel, data: mujoco.MjData, attachment: DualGraspAttachment) -> None:
    frame_pos, frame_xmat = _dual_hand_frame(model, data, attachment.left_side, attachment.right_side)
    object_pos = frame_pos + frame_xmat @ attachment.rel_pos
    object_xmat = frame_xmat @ attachment.rel_xmat
    data.qpos[attachment.qpos_id : attachment.qpos_id + 3] = object_pos
    data.qpos[attachment.qpos_id + 3 : attachment.qpos_id + 7] = _xmat_to_quat_wxyz(object_xmat)
    data.qvel[attachment.dof_id : attachment.dof_id + 6] = 0.0
    mujoco.mj_forward(model, data)


def update_attachments(model: mujoco.MjModel, data: mujoco.MjData, side: str | None = None) -> None:
    dual_attachment = _DUAL_ATTACHMENTS.get(id(data))
    if dual_attachment is not None and (
        side is None or side in (dual_attachment.left_side, dual_attachment.right_side)
    ):
        apply_dual_attachment(model, data, dual_attachment)
    attachments = _ATTACHMENTS.get(id(data), {})
    if side is not None:
        attachment = attachments.get(side)
        if attachment is not None:
            apply_attachment(model, data, attachment)
        return
    for attachment in tuple(attachments.values()):
        apply_attachment(model, data, attachment)


def detach_object(model: mujoco.MjModel, data: mujoco.MjData, side: str | None = None) -> None:
    dual_attachment = _DUAL_ATTACHMENTS.get(id(data))
    if dual_attachment is not None and (
        side is None or side in (dual_attachment.left_side, dual_attachment.right_side)
    ):
        _DUAL_ATTACHMENTS.pop(id(data), None)
    attachments = _ATTACHMENTS.get(id(data))
    if not attachments:
        mujoco.mj_forward(model, data)
        return
    if side is None:
        attachments.clear()
    else:
        attachments.pop(side, None)
    if not attachments:
        _ATTACHMENTS.pop(id(data), None)
    mujoco.mj_forward(model, data)


def attached_object_name(data: mujoco.MjData, side: str) -> str | None:
    dual_attachment = _DUAL_ATTACHMENTS.get(id(data))
    if dual_attachment is not None and side in (dual_attachment.left_side, dual_attachment.right_side):
        return dual_attachment.object_body
    attachment = _ATTACHMENTS.get(id(data), {}).get(side)
    return None if attachment is None else attachment.object_body
