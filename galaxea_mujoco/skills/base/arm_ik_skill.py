from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import mujoco
import numpy as np
import pinocchio as pin


ArmSide = Literal["left", "right"]


ARM_JOINTS: dict[ArmSide, tuple[str, ...]] = {
    "left": tuple(f"left_arm_joint{i}" for i in range(1, 8)),
    "right": tuple(f"right_arm_joint{i}" for i in range(1, 8)),
}

TORSO_JOINTS: tuple[str, ...] = tuple(f"torso_joint{i}" for i in range(1, 5))
GRIPPER_JOINTS: tuple[str, ...] = tuple(
    f"{side}_gripper_finger_joint{i}"
    for side in ("left", "right")
    for i in range(1, 3)
)


@dataclass(frozen=True)
class ArmMotionResult:
    side: ArmSide
    frame_name: str
    start_pos: np.ndarray
    target_pos: np.ndarray
    ik_pos: np.ndarray
    final_site_pos: np.ndarray
    ik_error: float
    final_error: float
    q_target: np.ndarray
    success: bool
    control_mode: str
    velocity_limit_applied: bool = False
    commanded_steps: int = 0
    motor_control_like_command: dict | None = None
    q_start: np.ndarray | None = None
    q_final: np.ndarray | None = None
    max_joint_tracking_error: float | None = None
    mean_joint_tracking_error: float | None = None
    precheck: dict | None = None


@dataclass(frozen=True)
class LockedJointState:
    qpos_adr: int
    qpos_size: int
    qvel_adr: int
    qvel_size: int
    qpos: np.ndarray
    qvel: np.ndarray


class R1ProArmIKSkill:
    """Pinocchio IK + MuJoCo actuator control for one R1Pro arm."""

    def __init__(self, urdf_path: str | Path = "urdf/r1_pro_with_gripper.urdf"):
        self.urdf_path = Path(urdf_path)
        self.pin_model = pin.buildModelFromUrdf(str(self.urdf_path))

    def frame_name(self, side: ArmSide) -> str:
        return f"{side}_hand_tcp"

    def mobile_base_pose(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        base_xy = np.zeros(2, dtype=np.float64)
        base_yaw = 0.0
        for idx, name in enumerate(("base_x", "base_y")):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                base_xy[idx] = data.qpos[model.jnt_qposadr[joint_id]]
        yaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")
        if yaw_joint_id >= 0:
            base_yaw = float(data.qpos[model.jnt_qposadr[yaw_joint_id]])
        return base_xy, base_yaw

    def world_to_pinocchio_position(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        world_pos: np.ndarray,
    ) -> np.ndarray:
        base_xy, base_yaw = self.mobile_base_pose(model, data)
        c = np.cos(base_yaw)
        s = np.sin(base_yaw)
        translated = np.asarray(world_pos, dtype=np.float64).reshape(3).copy()
        translated[:2] -= base_xy
        return np.array(
            [
                c * translated[0] + s * translated[1],
                -s * translated[0] + c * translated[1],
                translated[2],
            ],
            dtype=np.float64,
        )

    def world_to_pinocchio_rotation(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        world_rot: np.ndarray,
    ) -> np.ndarray:
        _base_xy, base_yaw = self.mobile_base_pose(model, data)
        c = np.cos(base_yaw)
        s = np.sin(base_yaw)
        world_from_base = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return world_from_base.T @ np.asarray(world_rot, dtype=np.float64).reshape(3, 3)

    def joint_q_index(self, name: str) -> tuple[int, int]:
        if not self.pin_model.existJointName(name):
            raise ValueError(f"Pinocchio joint not found: {name}")
        joint_id = self.pin_model.getJointId(name)
        if self.pin_model.nqs[joint_id] != 1 or self.pin_model.nvs[joint_id] != 1:
            raise ValueError(f"Only 1-DoF joints are supported: {name}")
        return self.pin_model.idx_qs[joint_id], self.pin_model.idx_vs[joint_id]

    def sync_q_from_mujoco(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        q = pin.neutral(self.pin_model)
        for name in self.pin_model.names[1:]:
            joint_id = self.pin_model.getJointId(name)
            if self.pin_model.nqs[joint_id] != 1:
                continue
            mj_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if mj_joint_id < 0:
                continue
            q[self.pin_model.idx_qs[joint_id]] = data.qpos[model.jnt_qposadr[mj_joint_id]]
        return q

    def current_tcp_position(self, q: np.ndarray, side: ArmSide) -> np.ndarray:
        pin_data = self.pin_model.createData()
        frame_id = self.pin_model.getFrameId(self.frame_name(side))
        pin.forwardKinematics(self.pin_model, pin_data, q)
        pin.updateFramePlacements(self.pin_model, pin_data)
        return pin_data.oMf[frame_id].translation.copy()

    def current_tcp_rotation(self, q: np.ndarray, side: ArmSide) -> np.ndarray:
        pin_data = self.pin_model.createData()
        frame_id = self.pin_model.getFrameId(self.frame_name(side))
        pin.forwardKinematics(self.pin_model, pin_data, q)
        pin.updateFramePlacements(self.pin_model, pin_data)
        return pin_data.oMf[frame_id].rotation.copy()

    def _rotation_error(self, current_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        current = np.asarray(current_rot, dtype=np.float64).reshape(3, 3)
        target = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
        return 0.5 * (
            np.cross(current[:, 0], target[:, 0])
            + np.cross(current[:, 1], target[:, 1])
            + np.cross(current[:, 2], target[:, 2])
        )

    def solve_position_ik(
        self,
        side: ArmSide,
        q0: np.ndarray,
        target_pos: np.ndarray,
        *,
        iterations: int = 200,
        damping: float = 1e-4,
        step_scale: float = 0.5,
        tolerance: float = 1e-4,
        joint_names: tuple[str, ...] | None = None,
        posture_reference: np.ndarray | None = None,
        posture_weights: np.ndarray | None = None,
        posture_gain: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        pin_data = self.pin_model.createData()
        q = q0.copy()
        frame_id = self.pin_model.getFrameId(self.frame_name(side))
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        q_indices = []
        v_indices = []
        lower = []
        upper = []

        for name in controlled_joints:
            q_index, v_index = self.joint_q_index(name)
            joint_id = self.pin_model.getJointId(name)
            q_indices.append(q_index)
            v_indices.append(v_index)
            lower.append(self.pin_model.lowerPositionLimit[q_index])
            upper.append(self.pin_model.upperPositionLimit[q_index])

        q_indices_array = np.asarray(q_indices)
        v_indices_array = np.asarray(v_indices)
        lower_array = np.asarray(lower)
        upper_array = np.asarray(upper)
        reference_q = q0 if posture_reference is None else posture_reference
        weights = (
            np.ones(len(controlled_joints), dtype=np.float64)
            if posture_weights is None and posture_gain > 0.0
            else (
                np.zeros(len(controlled_joints), dtype=np.float64)
                if posture_weights is None
                else np.asarray(posture_weights, dtype=np.float64)
            )
        )
        if weights.shape != (len(controlled_joints),):
            raise ValueError("posture_weights must match the number of controlled joints")
        last_error = float("inf")

        for _ in range(iterations):
            pin.forwardKinematics(self.pin_model, pin_data, q)
            pin.updateFramePlacements(self.pin_model, pin_data)
            current_pos = pin_data.oMf[frame_id].translation.copy()
            error = target_pos - current_pos
            last_error = float(np.linalg.norm(error))
            if last_error < tolerance:
                break

            jacobian = pin.computeFrameJacobian(
                self.pin_model,
                pin_data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, v_indices_array]
            lhs = jacobian @ jacobian.T + damping * np.eye(3)
            jacobian_pinv = jacobian.T @ np.linalg.inv(lhs)
            dq = jacobian_pinv @ error
            if posture_gain > 0.0 and np.any(weights):
                nullspace = np.eye(len(controlled_joints)) - jacobian_pinv @ jacobian
                posture_error = q[q_indices_array] - reference_q[q_indices_array]
                dq += nullspace @ (-posture_gain * weights * posture_error)
            q_arm = q[q_indices_array] + step_scale * dq
            q[q_indices_array] = np.clip(q_arm, lower_array, upper_array)

        final_pos = self.current_tcp_position(q, side)
        return q, final_pos, last_error

    def solve_pose_ik(
        self,
        side: ArmSide,
        q0: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        *,
        iterations: int = 300,
        damping: float = 1e-4,
        step_scale: float = 0.5,
        tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        orientation_weight: float = 0.35,
        joint_names: tuple[str, ...] | None = None,
        posture_reference: np.ndarray | None = None,
        posture_weights: np.ndarray | None = None,
        posture_gain: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        pin_data = self.pin_model.createData()
        q = q0.copy()
        frame_id = self.pin_model.getFrameId(self.frame_name(side))
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        q_indices = []
        v_indices = []
        lower = []
        upper = []

        for name in controlled_joints:
            q_index, v_index = self.joint_q_index(name)
            joint_id = self.pin_model.getJointId(name)
            q_indices.append(q_index)
            v_indices.append(v_index)
            lower.append(self.pin_model.lowerPositionLimit[q_index])
            upper.append(self.pin_model.upperPositionLimit[q_index])

        q_indices_array = np.asarray(q_indices)
        v_indices_array = np.asarray(v_indices)
        lower_array = np.asarray(lower)
        upper_array = np.asarray(upper)
        reference_q = q0 if posture_reference is None else posture_reference
        weights = (
            np.ones(len(controlled_joints), dtype=np.float64)
            if posture_weights is None and posture_gain > 0.0
            else (
                np.zeros(len(controlled_joints), dtype=np.float64)
                if posture_weights is None
                else np.asarray(posture_weights, dtype=np.float64)
            )
        )
        if weights.shape != (len(controlled_joints),):
            raise ValueError("posture_weights must match the number of controlled joints")

        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
        last_pos_error = float("inf")
        last_ori_error = float("inf")

        for _ in range(iterations):
            pin.forwardKinematics(self.pin_model, pin_data, q)
            pin.updateFramePlacements(self.pin_model, pin_data)
            placement = pin_data.oMf[frame_id]
            pos_error = target_pos - placement.translation
            ori_error = self._rotation_error(placement.rotation, target_rot)
            last_pos_error = float(np.linalg.norm(pos_error))
            last_ori_error = float(np.linalg.norm(ori_error))
            if last_pos_error < tolerance and last_ori_error < orientation_tolerance:
                break

            jacobian = pin.computeFrameJacobian(
                self.pin_model,
                pin_data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, v_indices_array]
            task_jac = np.vstack((jacobian[:3, :], orientation_weight * jacobian[3:, :]))
            task_error = np.concatenate((pos_error, orientation_weight * ori_error))
            lhs = task_jac @ task_jac.T + damping * np.eye(6)
            jacobian_pinv = task_jac.T @ np.linalg.inv(lhs)
            dq = jacobian_pinv @ task_error
            if posture_gain > 0.0 and np.any(weights):
                nullspace = np.eye(len(controlled_joints)) - jacobian_pinv @ task_jac
                posture_error = q[q_indices_array] - reference_q[q_indices_array]
                dq += nullspace @ (-posture_gain * weights * posture_error)
            q_arm = q[q_indices_array] + step_scale * dq
            q[q_indices_array] = np.clip(q_arm, lower_array, upper_array)

        final_pos = self.current_tcp_position(q, side)
        final_rot = self.current_tcp_rotation(q, side)
        return q, final_pos, final_rot, last_pos_error, last_ori_error

    def make_joint_trajectory(
        self,
        side: ArmSide,
        q_start: np.ndarray,
        q_target: np.ndarray,
        steps: int,
        joint_names: tuple[str, ...] | None = None,
    ) -> list[np.ndarray]:
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        q_indices = np.asarray([self.joint_q_index(name)[0] for name in controlled_joints])
        trajectory = []
        for i in range(max(steps, 1)):
            alpha = 1.0 if steps <= 1 else i / (steps - 1)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            q = q_start.copy()
            q[q_indices] = (1.0 - alpha) * q_start[q_indices] + alpha * q_target[q_indices]
            trajectory.append(q)
        return trajectory

    def make_velocity_limited_joint_trajectory(
        self,
        side: ArmSide,
        q_start: np.ndarray,
        q_target: np.ndarray,
        velocity_limit: np.ndarray,
        timestep: float,
        *,
        min_steps: int = 1,
        joint_names: tuple[str, ...] | None = None,
    ) -> list[np.ndarray]:
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        q_indices = np.asarray([self.joint_q_index(name)[0] for name in controlled_joints])
        limits = np.asarray(velocity_limit, dtype=np.float64).reshape(-1)
        if limits.size == 1:
            limits = np.full(len(controlled_joints), float(limits[0]), dtype=np.float64)
        if limits.shape != (len(controlled_joints),):
            raise ValueError(f"velocity_limit must have {len(controlled_joints)} values, got {limits.shape}")
        limits = np.maximum(limits, 1e-6)
        delta = np.abs(q_target[q_indices] - q_start[q_indices])
        required_time = float(np.max(delta / limits)) if delta.size else 0.0
        steps = max(int(np.ceil(required_time / max(float(timestep), 1e-6))) + 1, int(min_steps), 1)
        return self.make_joint_trajectory(side, q_start, q_target, steps, joint_names=controlled_joints)

    def set_arm_ctrl(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        q_pin: np.ndarray,
        joint_names: tuple[str, ...] | None = None,
    ) -> None:
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        for name in controlled_joints:
            q_index, _ = self.joint_q_index(name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
            if actuator_id < 0:
                raise ValueError(f"MuJoCo actuator not found: {name}_pos")
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(q_pin[q_index], low, high)

    def arm_joint_values_from_mujoco(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        joint_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        values = []
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        for name in controlled_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            values.append(data.qpos[model.jnt_qposadr[joint_id]])
        return np.asarray(values, dtype=np.float64)

    def arm_joint_values_from_pin_q(
        self,
        side: ArmSide,
        q_pin: np.ndarray,
        joint_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        return np.asarray([q_pin[self.joint_q_index(name)[0]] for name in controlled_joints], dtype=np.float64)

    def joint_limit_margin_from_pin_q(
        self,
        side: ArmSide,
        q_pin: np.ndarray,
        joint_names: tuple[str, ...] | None = None,
    ) -> float:
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        margins = []
        for name in controlled_joints:
            q_index, _ = self.joint_q_index(name)
            lower = float(self.pin_model.lowerPositionLimit[q_index])
            upper = float(self.pin_model.upperPositionLimit[q_index])
            value = float(q_pin[q_index])
            margins.append(min(value - lower, upper - value))
        return float(min(margins)) if margins else 0.0

    def set_arm_ctrl_values(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        q_values: np.ndarray,
        joint_names: tuple[str, ...] | None = None,
    ) -> None:
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        for name, value in zip(controlled_joints, q_values):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
            if actuator_id < 0:
                raise ValueError(f"MuJoCo actuator not found: {name}_pos")
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(value, low, high)

    def set_arm_qpos_direct(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        q_pin: np.ndarray,
        joint_names: tuple[str, ...] | None = None,
    ) -> None:
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        for name in controlled_joints:
            q_index, _ = self.joint_q_index(name)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            data.qpos[model.jnt_qposadr[joint_id]] = q_pin[q_index]

    def set_joint_qpos_values(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: tuple[str, ...],
        q_values: np.ndarray,
    ) -> None:
        for name, value in zip(joint_names, q_values):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            data.qpos[model.jnt_qposadr[joint_id]] = value
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
            if actuator_id >= 0:
                low, high = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = np.clip(value, low, high)

    def clamp_joint_limits(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        joint_names: tuple[str, ...] | None = None,
    ) -> int:
        clamped = 0
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        for name in controlled_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
                continue
            qpos_id = int(model.jnt_qposadr[joint_id])
            dof_id = int(model.jnt_dofadr[joint_id])
            low, high = model.jnt_range[joint_id]
            value = float(data.qpos[qpos_id])
            clipped = float(np.clip(value, low, high))
            if clipped != value:
                data.qpos[qpos_id] = clipped
                if 0 <= dof_id < data.qvel.size:
                    data.qvel[dof_id] = 0.0
                clamped += 1
        if clamped:
            mujoco.mj_forward(model, data)
        return clamped

    def configure_runtime_damping(
        self,
        model: mujoco.MjModel,
        side: ArmSide,
        *,
        damping: float = 50.0,
        armature: float = 0.05,
        force_scale: float = 1.0,
        joint_names: tuple[str, ...] | None = None,
    ) -> None:
        """Optionally stabilize actuator tracking without editing model.xml."""
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        for name in controlled_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            dof_id = model.jnt_dofadr[joint_id]
            model.dof_damping[dof_id] = damping
            model.dof_armature[dof_id] = armature
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
            if actuator_id < 0:
                raise ValueError(f"MuJoCo actuator not found: {name}_pos")
            if force_scale != 1.0:
                model.actuator_forcerange[actuator_id] *= force_scale

    def capture_locked_posture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        joint_names: tuple[str, ...] | None = None,
    ) -> dict[int, LockedJointState]:
        controlled = set(ARM_JOINTS[side] if joint_names is None else joint_names)
        excluded = set(GRIPPER_JOINTS)
        locked: dict[int, LockedJointState] = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name or name in controlled or name in excluded:
                continue
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
            locked[joint_id] = LockedJointState(
                qpos_adr=int(qpos_id),
                qpos_size=qpos_size,
                qvel_adr=int(dof_id),
                qvel_size=qvel_size,
                qpos=data.qpos[qpos_id : qpos_id + qpos_size].copy(),
                qvel=data.qvel[dof_id : dof_id + qvel_size].copy(),
            )
        return locked

    def apply_locked_posture(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        locked: dict[int, LockedJointState],
    ) -> None:
        for state in locked.values():
            data.qpos[state.qpos_adr : state.qpos_adr + state.qpos_size] = state.qpos
            if state.qvel_adr >= 0:
                data.qvel[state.qvel_adr : state.qvel_adr + state.qvel_size] = state.qvel

    def move_to_position(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: ArmSide,
        target_pos: np.ndarray,
        *,
        target_xmat: np.ndarray | None = None,
        steps: int = 1500,
        settle_steps: int = 3000,
        direct_qpos: bool = False,
        stabilize: bool = True,
        closed_loop: bool = True,
        cartesian_closed_loop: bool = False,
        lock_posture: bool = True,
        max_joint_step: float = 0.006,
        fail_threshold: float = 0.02,
        orientation_threshold: float = 0.15,
        orientation_weight: float = 0.35,
        force_scale: float = 1.0,
        runtime_damping: float = 50.0,
        runtime_armature: float = 0.05,
        joint_names: tuple[str, ...] | None = None,
        direct_joint_names: tuple[str, ...] = (),
        control_mode: str = "",
        velocity_limit: tuple[float, ...] | list[float] | np.ndarray | None = None,
        enforce_joint_limits: bool = False,
        posture_weights: np.ndarray | None = None,
        posture_gain: float = 0.0,
        step_callback: Callable[[], None] | None = None,
    ) -> ArmMotionResult:
        target_world_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_world_rot = None if target_xmat is None else np.asarray(target_xmat, dtype=np.float64).reshape(3, 3)
        mujoco.mj_forward(model, data)
        start_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.frame_name(side))
        if start_site_id < 0:
            raise ValueError(f"MuJoCo site not found: {self.frame_name(side)}")
        start_world_pos = data.site_xpos[start_site_id].copy()
        target_ik_pos = self.world_to_pinocchio_position(model, data, target_world_pos)
        controlled_joints = ARM_JOINTS[side] if joint_names is None else joint_names
        direct_joint_set = set(direct_joint_names)
        actuator_joints = tuple(name for name in controlled_joints if name not in direct_joint_set)
        if stabilize and not direct_qpos:
            self.configure_runtime_damping(
                model,
                side,
                damping=runtime_damping,
                armature=runtime_armature,
                force_scale=force_scale,
                joint_names=controlled_joints,
            )

        q_start = self.sync_q_from_mujoco(model, data)
        start_ik_pos = self.current_tcp_position(q_start, side)
        target_ik_rot = None if target_world_rot is None else self.world_to_pinocchio_rotation(model, data, target_world_rot)
        ik_orientation_error = 0.0
        if target_ik_rot is None:
            q_target, ik_pos, ik_error = self.solve_position_ik(
                side,
                q_start,
                target_ik_pos,
                joint_names=controlled_joints,
                posture_reference=q_start,
                posture_weights=posture_weights,
                posture_gain=posture_gain,
            )
        else:
            q_target, ik_pos, ik_rot, ik_error, ik_orientation_error = self.solve_pose_ik(
                side,
                q_start,
                target_ik_pos,
                target_ik_rot,
                joint_names=controlled_joints,
                posture_reference=q_start,
                posture_weights=posture_weights,
                posture_gain=posture_gain,
                orientation_weight=float(orientation_weight),
            )
        q_start_values = self.arm_joint_values_from_pin_q(side, q_start, joint_names=controlled_joints)
        q_target_values = self.arm_joint_values_from_pin_q(side, q_target, joint_names=controlled_joints)
        precheck = {
            "target_distance_from_start": float(np.linalg.norm(target_world_pos - start_world_pos)),
            "ik_error": float(ik_error),
            "ik_orientation_error": float(ik_orientation_error),
            "target_joint_delta_max": float(np.max(np.abs(q_target_values - q_start_values))) if q_target_values.size else 0.0,
            "target_joint_delta_mean": float(np.mean(np.abs(q_target_values - q_start_values))) if q_target_values.size else 0.0,
            "target_joint_limit_margin_min": self.joint_limit_margin_from_pin_q(side, q_target, joint_names=controlled_joints),
        }
        requested_control_mode = str(control_mode or "").lower()
        velocity_limit_applied = requested_control_mode == "joint_target_velocity_limited"
        if velocity_limit_applied:
            limits = np.asarray(velocity_limit if velocity_limit is not None else [3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float64)
            trajectory = self.make_velocity_limited_joint_trajectory(
                side,
                q_start,
                q_target,
                limits,
                float(model.opt.timestep),
                min_steps=steps,
                joint_names=controlled_joints,
            )
        else:
            trajectory = self.make_joint_trajectory(side, q_start, q_target, steps, joint_names=controlled_joints)
        locked = self.capture_locked_posture(model, data, side, joint_names=controlled_joints) if lock_posture else {}

        if direct_qpos:
            control_mode = "direct_qpos"
            for q in trajectory:
                self.set_arm_qpos_direct(model, data, side, q, joint_names=controlled_joints)
                if enforce_joint_limits:
                    self.clamp_joint_limits(model, data, side, joint_names=controlled_joints)
                self.apply_locked_posture(model, data, locked)
                mujoco.mj_forward(model, data)
                if step_callback is not None:
                    step_callback()
        elif cartesian_closed_loop:
            control_mode = "cartesian_closed_loop"
            for i in range(max(steps, 1)):
                alpha = 1.0 if steps <= 1 else i / (steps - 1)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                waypoint = (1.0 - alpha) * start_ik_pos + alpha * target_ik_pos
                q_current = self.sync_q_from_mujoco(model, data)
                q_command, _, _ = self.solve_position_ik(
                    side,
                    q_current,
                    waypoint,
                    iterations=40,
                    tolerance=1e-4,
                    joint_names=controlled_joints,
                    posture_reference=q_start,
                    posture_weights=posture_weights,
                    posture_gain=posture_gain,
                )
                self.set_arm_ctrl(model, data, side, q_command, joint_names=controlled_joints)
                mujoco.mj_step(model, data)
                if enforce_joint_limits:
                    self.clamp_joint_limits(model, data, side, joint_names=controlled_joints)
                self.apply_locked_posture(model, data, locked)
                mujoco.mj_forward(model, data)
                if step_callback is not None:
                    step_callback()
            for _ in range(max(settle_steps, 0)):
                q_current = self.sync_q_from_mujoco(model, data)
                q_command, _, _ = self.solve_position_ik(
                    side,
                    q_current,
                    target_ik_pos,
                    iterations=40,
                    tolerance=1e-4,
                    joint_names=controlled_joints,
                    posture_reference=q_start,
                    posture_weights=posture_weights,
                    posture_gain=posture_gain,
                )
                self.set_arm_ctrl(model, data, side, q_command, joint_names=controlled_joints)
                mujoco.mj_step(model, data)
                if enforce_joint_limits:
                    self.clamp_joint_limits(model, data, side, joint_names=controlled_joints)
                self.apply_locked_posture(model, data, locked)
                mujoco.mj_forward(model, data)
                if step_callback is not None:
                    step_callback()
        elif closed_loop:
            control_mode = "joint_target_velocity_limited" if velocity_limit_applied else ("joint_servo" if not direct_joint_set else "joint_servo_with_direct_joints")
            q_command = self.arm_joint_values_from_mujoco(model, data, side, joint_names=controlled_joints)
            q_target_values = self.arm_joint_values_from_pin_q(side, q_target, joint_names=controlled_joints)
            command_values = [self.arm_joint_values_from_pin_q(side, q, joint_names=controlled_joints) for q in trajectory]
            for target_values in command_values:
                delta = np.clip(target_values - q_command, -max_joint_step, max_joint_step)
                q_command = q_command + delta
                if direct_joint_set:
                    direct_values = np.asarray(
                        [q_command[controlled_joints.index(name)] for name in direct_joint_names],
                        dtype=np.float64,
                    )
                    actuator_values = np.asarray(
                        [q_command[controlled_joints.index(name)] for name in actuator_joints],
                        dtype=np.float64,
                    )
                    self.set_joint_qpos_values(model, data, direct_joint_names, direct_values)
                    self.set_arm_ctrl_values(model, data, side, actuator_values, joint_names=actuator_joints)
                else:
                    self.set_arm_ctrl_values(model, data, side, q_command, joint_names=controlled_joints)
                mujoco.mj_step(model, data)
                if enforce_joint_limits:
                    self.clamp_joint_limits(model, data, side, joint_names=controlled_joints)
                self.apply_locked_posture(model, data, locked)
                mujoco.mj_forward(model, data)
                if step_callback is not None:
                    step_callback()
            for _ in range(max(settle_steps, 0)):
                if direct_joint_set:
                    direct_values = np.asarray(
                        [q_target_values[controlled_joints.index(name)] for name in direct_joint_names],
                        dtype=np.float64,
                    )
                    actuator_values = np.asarray(
                        [q_target_values[controlled_joints.index(name)] for name in actuator_joints],
                        dtype=np.float64,
                    )
                    self.set_joint_qpos_values(model, data, direct_joint_names, direct_values)
                    self.set_arm_ctrl_values(model, data, side, actuator_values, joint_names=actuator_joints)
                else:
                    current_values = self.arm_joint_values_from_mujoco(model, data, side, joint_names=controlled_joints)
                    correction = np.clip(q_target_values - current_values, -max_joint_step, max_joint_step)
                    self.set_arm_ctrl_values(model, data, side, current_values + correction, joint_names=controlled_joints)
                mujoco.mj_step(model, data)
                if enforce_joint_limits:
                    self.clamp_joint_limits(model, data, side, joint_names=controlled_joints)
                self.apply_locked_posture(model, data, locked)
                mujoco.mj_forward(model, data)
                if step_callback is not None:
                    step_callback()
        else:
            control_mode = "joint_trajectory"
            for q in trajectory:
                self.set_arm_ctrl(model, data, side, q, joint_names=controlled_joints)
                mujoco.mj_step(model, data)
                if enforce_joint_limits:
                    self.clamp_joint_limits(model, data, side, joint_names=controlled_joints)
                self.apply_locked_posture(model, data, locked)
                mujoco.mj_forward(model, data)
                if step_callback is not None:
                    step_callback()
            for _ in range(max(settle_steps, 0)):
                self.set_arm_ctrl(model, data, side, q_target, joint_names=controlled_joints)
                mujoco.mj_step(model, data)
                if enforce_joint_limits:
                    self.clamp_joint_limits(model, data, side, joint_names=controlled_joints)
                self.apply_locked_posture(model, data, locked)
                mujoco.mj_forward(model, data)
                if step_callback is not None:
                    step_callback()

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.frame_name(side))
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {self.frame_name(side)}")
        final_site_pos = data.site_xpos[site_id].copy()
        final_error = float(np.linalg.norm(final_site_pos - target_world_pos))
        final_orientation_error = 0.0
        if target_world_rot is not None:
            final_orientation_error = float(np.linalg.norm(self._rotation_error(data.site_xmat[site_id].reshape(3, 3), target_world_rot)))
        q_final = self.arm_joint_values_from_mujoco(model, data, side, joint_names=controlled_joints)
        tracking = np.abs(q_final - q_target_values)
        motor_control_like_command = {
            "joint_names": list(controlled_joints),
            "p_des": [float(x) for x in q_target_values],
            "v_des": [0.0 for _ in controlled_joints],
            "t_ff": [0.0 for _ in controlled_joints],
            "kp": [80.0 for _ in controlled_joints],
            "kd": [5.0 for _ in controlled_joints],
            "mode": "position",
        }
        return ArmMotionResult(
            side=side,
            frame_name=self.frame_name(side),
            start_pos=start_world_pos,
            target_pos=target_world_pos,
            ik_pos=ik_pos,
            final_site_pos=final_site_pos,
            ik_error=float(ik_error),
            final_error=final_error,
            q_target=q_target,
            success=final_error <= fail_threshold and final_orientation_error <= orientation_threshold,
            control_mode=control_mode,
            velocity_limit_applied=velocity_limit_applied,
            commanded_steps=len(trajectory),
            motor_control_like_command=motor_control_like_command,
            q_start=q_start_values,
            q_final=q_final,
            max_joint_tracking_error=float(np.max(tracking)) if tracking.size else 0.0,
            mean_joint_tracking_error=float(np.mean(tracking)) if tracking.size else 0.0,
            precheck={**precheck, "final_orientation_error": float(final_orientation_error)},
        )

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        direct_qpos: bool = False,
    ) -> ArmMotionResult:
        if "side" not in params:
            raise ValueError("Provide side")
        side = params["side"]
        if side not in ARM_JOINTS:
            raise ValueError(f"Unsupported arm side: {side!r}")
        target_pos = np.array([params["target_x"], params["target_y"], params["target_z"]], dtype=np.float64)
        return self.move_to_position(
            model,
            data,
            side,
            target_pos,
            steps=int(params.get("steps", 1500)),
            settle_steps=int(params.get("settle_steps", 3000)),
            direct_qpos=direct_qpos,
            stabilize=bool(params.get("stabilize", True)),
            closed_loop=bool(params.get("closed_loop", True)),
            cartesian_closed_loop=bool(params.get("cartesian_closed_loop", False)),
            lock_posture=bool(params.get("lock_posture", True)),
            max_joint_step=float(params.get("max_joint_step", 0.006)),
            fail_threshold=float(params.get("fail_threshold", 0.02)),
            force_scale=float(params.get("force_scale", 1.0)),
            runtime_damping=float(params.get("runtime_damping", 50.0)),
            runtime_armature=float(params.get("runtime_armature", 0.05)),
        )


def load_skill(urdf_path: str | Path = "urdf/r1_pro_with_gripper.urdf") -> R1ProArmIKSkill:
    return R1ProArmIKSkill(urdf_path)
