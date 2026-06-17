from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import mujoco
import numpy as np


ArmSide = Literal["left", "right"]

ARM_JOINTS: dict[ArmSide, tuple[str, ...]] = {
    "left": tuple(f"left_arm_joint{i}" for i in range(1, 8)),
    "right": tuple(f"right_arm_joint{i}" for i in range(1, 8)),
}


@dataclass(frozen=True)
class MuJoCoSiteServoResult:
    side: ArmSide
    site_name: str
    start_pos: np.ndarray
    target_pos: np.ndarray
    final_pos: np.ndarray
    final_error: float
    success: bool
    steps_run: int
    min_error: float
    stalled: bool
    contacts: tuple[str, ...]


@dataclass(frozen=True)
class LockedJointState:
    qpos_adr: int
    qpos_size: int
    qvel_adr: int
    qvel_size: int
    qpos: np.ndarray
    qvel: np.ndarray


class R1ProMuJoCoSiteServo:
    """MuJoCo-native damped least-squares TCP position servo."""

    def site_name(self, side: ArmSide, frame: str = "tcp") -> str:
        if frame in {"grasp_tool", "tool", "grasp"}:
            return f"{side}_grasp_tool"
        return f"{side}_hand_tcp"

    def joint_names(self, side: ArmSide) -> tuple[str, ...]:
        return ARM_JOINTS[side]

    def move_to_position(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        target_pos: np.ndarray | list[float] | tuple[float, float, float],
        *,
        frame: str = "tcp",
        target_xmat: np.ndarray | None = None,
        steps: int = 2500,
        settle_steps: int = 1500,
        solve_iterations: int = 1000,
        fail_threshold: float = 0.02,
        orientation_threshold: float = 0.08,
        orientation_weight: float = 0.35,
        damping: float = 1e-3,
        max_cart_step: float = 0.006,
        max_joint_step: float = 0.012,
        stall_window: int = 250,
        stall_epsilon: float = 1e-4,
        posture_gain: float = 0.02,
        stabilize: bool = True,
        runtime_damping: float = 50.0,
        runtime_armature: float = 0.05,
        force_scale: float = 1.0,
        lock_uncontrolled: bool = True,
        step_callback: Callable[[], None] | None = None,
    ) -> MuJoCoSiteServoResult:
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_mat = None if target_xmat is None else np.asarray(target_xmat, dtype=np.float64).reshape(3, 3)
        site_id = self._site_id(model, side, frame=frame)
        joint_names = self.joint_names(side)
        joint_ids = tuple(self._joint_id(model, name) for name in joint_names)
        qpos_ids = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        dof_ids = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        ctrl_ids = np.asarray([self._actuator_id(model, name) for name in joint_names], dtype=np.int32)
        if stabilize:
            self._configure_runtime_damping(
                model,
                joint_ids,
                ctrl_ids,
                damping=runtime_damping,
                armature=runtime_armature,
                force_scale=force_scale,
            )

        mujoco.mj_forward(model, data)
        start_pos = data.site_xpos[site_id].copy()
        locked = self._capture_locked_joints(model, data, set(joint_names)) if lock_uncontrolled else {}
        q_start = data.qpos[qpos_ids].copy()
        q_target, _solved_pos, solve_error = self._solve_site_q_target(
            model,
            data,
            site_id,
            joint_ids,
            qpos_ids,
            dof_ids,
            ctrl_ids,
            target,
            target_mat,
            iterations=solve_iterations,
            damping=damping,
            max_cart_step=max_cart_step,
            max_joint_step=max_joint_step,
            orientation_weight=orientation_weight,
            posture_gain=posture_gain,
        )

        min_error = float(solve_error)
        recent_errors: list[float] = []
        stalled = False
        steps_run = 0

        for step in range(max(steps, 1)):
            alpha = 1.0 if steps <= 1 else step / (steps - 1)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            q_command = (1.0 - alpha) * q_start + alpha * q_target
            data.ctrl[ctrl_ids] = q_command
            mujoco.mj_step(model, data)
            self._apply_locked_joints(data, locked)
            mujoco.mj_forward(model, data)

            error = target - data.site_xpos[site_id]
            error_norm = float(np.linalg.norm(error))
            orientation_norm = 0.0
            if target_mat is not None:
                orientation_norm = float(np.linalg.norm(self._orientation_error(data.site_xmat[site_id].reshape(3, 3), target_mat)))
            min_error = min(min_error, error_norm)
            recent_errors.append(error_norm)
            if len(recent_errors) > stall_window:
                recent_errors.pop(0)

            if error_norm <= fail_threshold and orientation_norm <= orientation_threshold:
                steps_run = step
                break
            if step_callback is not None:
                step_callback()
            steps_run = step + 1

        for _ in range(max(settle_steps, 0)):
            data.ctrl[ctrl_ids] = q_command
            mujoco.mj_step(model, data)
            self._apply_locked_joints(data, locked)
            mujoco.mj_forward(model, data)
            error_norm = float(np.linalg.norm(data.site_xpos[site_id] - target))
            min_error = min(min_error, error_norm)
            if step_callback is not None:
                step_callback()

        recent_errors.append(float(np.linalg.norm(data.site_xpos[site_id] - target)))
        if len(recent_errors) >= stall_window:
            stalled = (recent_errors[0] - min(recent_errors)) < stall_epsilon

        mujoco.mj_forward(model, data)
        final_pos = data.site_xpos[site_id].copy()
        final_error = float(np.linalg.norm(final_pos - target))
        final_orientation_error = 0.0
        if target_mat is not None:
            final_orientation_error = float(np.linalg.norm(self._orientation_error(data.site_xmat[site_id].reshape(3, 3), target_mat)))
        return MuJoCoSiteServoResult(
            side=side,
            site_name=self.site_name(side, frame=frame),
            start_pos=start_pos,
            target_pos=target,
            final_pos=final_pos,
            final_error=final_error,
            success=final_error <= fail_threshold and final_orientation_error <= orientation_threshold,
            steps_run=steps_run,
            min_error=min(min_error, final_error),
            stalled=stalled,
            contacts=self._contact_summary(model, data),
        )

    def _solve_site_q_target(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_id: int,
        joint_ids: tuple[int, ...],
        qpos_ids: np.ndarray,
        dof_ids: np.ndarray,
        ctrl_ids: np.ndarray,
        target: np.ndarray,
        target_xmat: np.ndarray | None,
        *,
        iterations: int,
        damping: float,
        max_cart_step: float,
        max_joint_step: float,
        orientation_weight: float,
        posture_gain: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        solve_data = mujoco.MjData(model)
        solve_data.qpos[:] = data.qpos
        solve_data.qvel[:] = data.qvel
        q_reference = solve_data.qpos[qpos_ids].copy()

        for _ in range(max(iterations, 1)):
            mujoco.mj_forward(model, solve_data)
            pos_error = target - solve_data.site_xpos[site_id]
            ori_error = (
                np.zeros(3, dtype=np.float64)
                if target_xmat is None
                else self._orientation_error(solve_data.site_xmat[site_id].reshape(3, 3), target_xmat)
            )
            if float(np.linalg.norm(pos_error)) < 1e-4 and float(np.linalg.norm(ori_error)) < 1e-3:
                break

            cart_step = self._limit_norm(pos_error, max_cart_step)
            ori_step = self._limit_norm(ori_error, max_cart_step) * orientation_weight
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(model, solve_data, jacp, jacr, site_id)
            if target_xmat is None or orientation_weight <= 0.0:
                jac = jacp[:, dof_ids]
                task_step = cart_step
            else:
                jac = np.vstack((jacp[:, dof_ids], orientation_weight * jacr[:, dof_ids]))
                task_step = np.concatenate((cart_step, ori_step))
            lhs = jac @ jac.T + damping * np.eye(jac.shape[0])
            jac_pinv = jac.T @ np.linalg.inv(lhs)
            dq = jac_pinv @ task_step
            if posture_gain > 0.0:
                nullspace = np.eye(len(joint_ids)) - jac_pinv @ jac
                posture_error = solve_data.qpos[qpos_ids] - q_reference
                dq += nullspace @ (-posture_gain * posture_error)
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            next_q = solve_data.qpos[qpos_ids] + dq
            solve_data.qpos[qpos_ids] = self._clip_joint_values(model, joint_ids, ctrl_ids, next_q)

        mujoco.mj_forward(model, solve_data)
        final_pos = solve_data.site_xpos[site_id].copy()
        final_error = float(np.linalg.norm(final_pos - target))
        return solve_data.qpos[qpos_ids].copy(), final_pos, final_error

    def _site_id(self, model: mujoco.MjModel, side: ArmSide, frame: str = "tcp") -> int:
        site_name = self.site_name(side, frame=frame)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {site_name}")
        return site_id

    def _joint_id(self, model: mujoco.MjModel, name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Only hinge arm joints are supported: {name}")
        return joint_id

    def _actuator_id(self, model: mujoco.MjModel, joint_name: str) -> int:
        actuator_name = f"{joint_name}_pos"
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {actuator_name}")
        return actuator_id

    def _clip_joint_values(
        self,
        model: mujoco.MjModel,
        joint_ids: tuple[int, ...],
        ctrl_ids: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        clipped = values.copy()
        for i, joint_id in enumerate(joint_ids):
            if model.jnt_limited[joint_id]:
                clipped[i] = np.clip(clipped[i], *model.jnt_range[joint_id])
            if model.actuator_ctrllimited[ctrl_ids[i]]:
                clipped[i] = np.clip(clipped[i], *model.actuator_ctrlrange[ctrl_ids[i]])
        return clipped

    def _configure_runtime_damping(
        self,
        model: mujoco.MjModel,
        joint_ids: tuple[int, ...],
        ctrl_ids: np.ndarray,
        *,
        damping: float,
        armature: float,
        force_scale: float,
    ) -> None:
        for joint_id in joint_ids:
            dof_id = model.jnt_dofadr[joint_id]
            model.dof_damping[dof_id] = damping
            model.dof_armature[dof_id] = armature
        if force_scale != 1.0:
            model.actuator_forcerange[ctrl_ids] *= force_scale

    def _capture_locked_joints(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controlled_names: set[str],
    ) -> dict[int, LockedJointState]:
        locked: dict[int, LockedJointState] = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name or name in controlled_names:
                continue
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            qpos_size = 1 if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_BALL else 4
            qvel_size = 1 if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_BALL else 3
            qpos_adr = int(model.jnt_qposadr[joint_id])
            qvel_adr = int(model.jnt_dofadr[joint_id])
            locked[joint_id] = LockedJointState(
                qpos_adr=qpos_adr,
                qpos_size=qpos_size,
                qvel_adr=qvel_adr,
                qvel_size=qvel_size,
                qpos=data.qpos[qpos_adr : qpos_adr + qpos_size].copy(),
                qvel=data.qvel[qvel_adr : qvel_adr + qvel_size].copy(),
            )
        return locked

    def _apply_locked_joints(self, data: mujoco.MjData, locked: dict[int, LockedJointState]) -> None:
        for state in locked.values():
            data.qpos[state.qpos_adr : state.qpos_adr + state.qpos_size] = state.qpos
            data.qvel[state.qvel_adr : state.qvel_adr + state.qvel_size] = state.qvel

    def _contact_summary(self, model: mujoco.MjModel, data: mujoco.MjData, limit: int = 12) -> tuple[str, ...]:
        contacts = []
        for i in range(min(data.ncon, limit)):
            contact = data.contact[i]
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or str(body1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or str(body2)
            contacts.append(f"{name1} <-> {name2}: {contact.dist:.6f}")
        return tuple(contacts)

    def _limit_norm(self, vec: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= max_norm or norm == 0.0:
            return vec
        return vec * (max_norm / norm)

    def _orientation_error(self, current_xmat: np.ndarray, target_xmat: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.cross(current_xmat[:, 0], target_xmat[:, 0])
            + np.cross(current_xmat[:, 1], target_xmat[:, 1])
            + np.cross(current_xmat[:, 2], target_xmat[:, 2])
        )
