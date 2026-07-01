from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


@dataclass(frozen=True)
class FR5RobotIds:
    joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    arm_actuator_ids: np.ndarray
    finger_actuator_ids: tuple[int, int]
    tcp_site_id: int


class DiffIKController:
    """Small damped least-squares IK controller using MuJoCo site Jacobians."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "tool_center_point",
        joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES,
        damping: float = 1e-4,
        integration_dt: float = 0.15,
    ) -> None:
        self.model = model
        self.site_id = model.site(site_name).id
        self.joint_ids = np.array([model.joint(name).id for name in joint_names], dtype=np.int32)
        self.qpos_indices = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.damping = damping
        self.integration_dt = integration_dt

        self.jac = np.zeros((6, model.nv))
        self.error = np.zeros(6)
        self.diag = damping * np.eye(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.goal_quat = np.zeros(4)
        self.error_quat = np.zeros(4)

    def compute_q_des(
        self,
        data: mujoco.MjData,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        self.error[:3] = np.asarray(goal_pos, dtype=float) - data.site(self.site_id).xpos

        if goal_quat is None:
            self.error[3:] = 0.0
        else:
            self.goal_quat[:] = np.asarray(goal_quat, dtype=float)
            mujoco.mju_mat2Quat(self.site_quat, data.site(self.site_id).xmat)
            mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
            mujoco.mju_mulQuat(self.error_quat, self.goal_quat, self.site_quat_conj)
            mujoco.mju_quat2Vel(self.error[3:], self.error_quat, 1.0)

        mujoco.mj_jacSite(self.model, data, self.jac[:3], self.jac[3:], self.site_id)
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)

        q_full = data.qpos.copy()
        mujoco.mj_integratePos(self.model, q_full, dq, self.integration_dt)
        q_des = q_full[self.qpos_indices].copy()
        np.clip(
            q_des,
            self.model.jnt_range[self.joint_ids, 0],
            self.model.jnt_range[self.joint_ids, 1],
            out=q_des,
        )
        return q_des


class FR5MuJoCoController:
    """Minimal FR5 MuJoCo wrapper for joint, TCP, and PGI gripper control."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        kp: float = 800.0,
        kd: float = 80.0,
        joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES,
    ) -> None:
        self.model = model
        self.data = data
        self.kp = kp
        self.kd = kd

        for body_id in range(model.nbody):
            name = model.body(body_id).name
            if name and ("link" in name or "finger" in name):
                model.body_gravcomp[body_id] = 1.0
        self._configure_contact_properties()

        joint_ids = np.array([model.joint(name).id for name in joint_names], dtype=np.int32)
        qpos_indices = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
        qvel_indices = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)
        arm_actuator_ids = np.array([model.actuator(name).id for name in joint_names], dtype=np.int32)

        self.ids = FR5RobotIds(
            joint_names=joint_names,
            joint_ids=joint_ids,
            qpos_indices=qpos_indices,
            qvel_indices=qvel_indices,
            arm_actuator_ids=arm_actuator_ids,
            finger_actuator_ids=(model.actuator("act_finger1").id, model.actuator("act_finger2").id),
            tcp_site_id=model.site("tool_center_point").id,
        )
        self.ik = DiffIKController(model, joint_names=joint_names)

    def _configure_contact_properties(self) -> None:
        for geom_name in ("finger1_collision", "finger2_collision", "apple0"):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                continue
            self.model.geom_friction[geom_id] = np.array([3.0, 0.08, 0.008], dtype=float)
            self.model.geom_condim[geom_id] = 6
            self.model.geom_solref[geom_id] = np.array([0.008, 1.0], dtype=float)
            self.model.geom_solimp[geom_id] = np.array([0.95, 0.99, 0.001, 0.5, 2.0], dtype=float)

        for actuator_name in ("act_finger1", "act_finger2"):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                continue
            self.model.actuator_gainprm[actuator_id, 0] = 900.0
            self.model.actuator_biasprm[actuator_id, 1] = -900.0

    @classmethod
    def from_scene(cls, scene_path: str | Path | None = None) -> "FR5MuJoCoController":
        if scene_path is None:
            scene_path = Path(__file__).resolve().parents[2] / "assets" / "scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return cls(model, data)

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.ids.qpos_indices].copy()

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site(self.ids.tcp_site_id).xpos.copy()

    @property
    def tcp_quat(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site(self.ids.tcp_site_id).xmat)
        return quat

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.model.jnt_range[self.ids.joint_ids, 0].copy(),
            self.model.jnt_range[self.ids.joint_ids, 1].copy(),
        )

    def reset_home(self, key_name: str = "home") -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(key_name).id)
        mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, opening_m: float) -> None:
        opening = float(np.clip(opening_m, 0.0, 0.04))
        self.data.ctrl[self.ids.finger_actuator_ids[0]] = opening
        self.data.ctrl[self.ids.finger_actuator_ids[1]] = opening

    def apply_joint_pd(self, q_des: np.ndarray) -> None:
        q_target = np.asarray(q_des, dtype=float)
        q_curr = self.data.qpos[self.ids.qpos_indices]
        qvel_curr = self.data.qvel[self.ids.qvel_indices]
        torque = self.kp * (q_target - q_curr) - self.kd * qvel_curr
        torque += self.data.qfrc_bias[self.ids.qvel_indices]
        self.data.ctrl[self.ids.arm_actuator_ids] = torque

    def step_joint_target(self, q_des: np.ndarray, gripper_opening_m: float | None = None) -> None:
        self.apply_joint_pd(q_des)
        if gripper_opening_m is not None:
            self.set_gripper(gripper_opening_m)
        mujoco.mj_step(self.model, self.data)

    def step_tcp_target(
        self,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray | None = None,
        gripper_opening_m: float | None = None,
    ) -> np.ndarray:
        q_des = self.ik.compute_q_des(self.data, goal_pos=goal_pos, goal_quat=goal_quat)
        self.step_joint_target(q_des, gripper_opening_m=gripper_opening_m)
        return q_des
