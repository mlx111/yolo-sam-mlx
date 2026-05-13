from __future__ import annotations

import argparse
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SCENE_XML = Path(__file__).with_name("r1_pro_galaxea_scene.xml")
TARGET_BODY_CANDIDATES = ("follow_target", "goal_marker", "target_marker")

TORSO_JOINTS = ["torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"]
LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
LEFT_SIDE_JOINTS = TORSO_JOINTS + LEFT_ARM_JOINTS
RIGHT_SIDE_JOINTS = TORSO_JOINTS + RIGHT_ARM_JOINTS
SIDE_TO_JOINTS = {"left": LEFT_SIDE_JOINTS, "right": RIGHT_SIDE_JOINTS}
SIDE_TO_EE_BODY = {"left": "left_arm_link7", "right": "right_arm_link7"}
SIDE_TO_TCP_SITE = {"left": "left_hand_tcp", "right": "right_hand_tcp"}
SIDE_TO_FINGER_JOINTS = {
    "left": [
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
    ],
    "right": [
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    ],
}
FOLLOW_CANDIDATE_JOINT_SETS = {
    "left": [LEFT_ARM_JOINTS, LEFT_SIDE_JOINTS],
    "right": [RIGHT_ARM_JOINTS, RIGHT_SIDE_JOINTS],
}
TORSO_SOFT_LIMITS = {
    "torso_joint1": 0.30,
    "torso_joint2": 0.60,
    "torso_joint3": 0.35,
    "torso_joint4": 0.25,
}
FINGER_OPEN_Q = {
    "left": {
        "left_gripper_finger_joint1": 0.05,
        "left_gripper_finger_joint2": 0.05,
    },
    "right": {
        "right_gripper_finger_joint1": 0.05,
        "right_gripper_finger_joint2": 0.05,
    },
}
FINGER_CLOSE_Q = {
    "left": {
        "left_gripper_finger_joint1": 0.0,
        "left_gripper_finger_joint2": 0.0,
    },
    "right": {
        "right_gripper_finger_joint1": 0.0,
        "right_gripper_finger_joint2": 0.0,
    },
}
STABLE_Q = {
    "steer_motor_joint1": 0.0,
    "steer_motor_joint2": 0.0,
    "steer_motor_joint3": 0.0,
    "wheel_motor_joint1": 0.0,
    "wheel_motor_joint2": 0.0,
    "wheel_motor_joint3": 0.0,
    "torso_joint1": 0.25,
    "torso_joint2": -0.622,
    "torso_joint3": -0.532,
    "torso_joint4": 0.0,
    "left_arm_joint1": -0.13901277,
    "left_arm_joint2": 0.48248085,
    "left_arm_joint3": -0.0478383,
    "left_arm_joint4": -1.66591489,
    "left_arm_joint5": 0.90680851,
    "left_arm_joint6": -0.40558298,
    "left_arm_joint7": -0.73308511,
    "left_gripper_finger_joint1": 0.05,
    "left_gripper_finger_joint2": 0.05,
    "right_arm_joint1": -0.69229362,
    "right_arm_joint2": -0.99383404,
    "right_arm_joint3": 1.09085957,
    "right_arm_joint4": -1.74389787,
    "right_arm_joint5": -0.88277021,
    "right_arm_joint6": -0.59911489,
    "right_arm_joint7": 0.50699574,
    "right_gripper_finger_joint1": 0.05,
    "right_gripper_finger_joint2": 0.05,
}
ALL_CONTROL_JOINTS = [
    "steer_motor_joint1",
    "wheel_motor_joint1",
    "steer_motor_joint2",
    "wheel_motor_joint2",
    "steer_motor_joint3",
    "wheel_motor_joint3",
    *TORSO_JOINTS,
    *LEFT_ARM_JOINTS,
    *RIGHT_ARM_JOINTS,
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.copy()
    return v / n


def _rotation_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _top_down_grasp_rotation(yaw: float) -> np.ndarray:
    x_axis = _normalize(np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float))
    z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = _normalize(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _rotation_error_vector(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    R_err = R_des @ R_cur.T
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_theta))
    if angle < 1e-6:
        return np.zeros(3, dtype=float)
    denom = 2.0 * float(np.sin(angle))
    if abs(denom) < 1e-9:
        return np.zeros(3, dtype=float)
    axis = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ],
        dtype=float,
    ) / denom
    return axis * angle


def _quintic_blend(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _interpolate(q0: np.ndarray, q1: np.ndarray, t: float, duration: float) -> np.ndarray:
    if duration <= 0.0:
        return q1.copy()
    s = _quintic_blend(t / duration)
    return (1.0 - s) * q0 + s * q1


def _cartesian_path_points(points: list[np.ndarray], samples_per_segment: int) -> list[np.ndarray]:
    if len(points) < 2:
        return points[:]
    samples = max(2, int(samples_per_segment))
    result: list[np.ndarray] = []
    for idx in range(len(points) - 1):
        start = np.asarray(points[idx], dtype=float).reshape(3)
        end = np.asarray(points[idx + 1], dtype=float).reshape(3)
        for i in range(samples):
            if idx > 0 and i == 0:
                continue
            alpha = _quintic_blend(i / (samples - 1))
            result.append((1.0 - alpha) * start + alpha * end)
    return result


@dataclass
class JointMap:
    name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int
    qmin: float
    qmax: float


class R1ProReachEnv:
    def __init__(self, enable_viewer: bool = True) -> None:
        self.enable_viewer = enable_viewer
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer: Optional[mujoco.viewer.Handle] = None
        self.dt: float = 0.002
        self.ctrl_targets: Dict[str, float] = {}
        self.joint_maps: Dict[str, JointMap] = {}
        self.actuator_joint_names: list[str] = []
        self.ee_body_ids: Dict[str, int] = {}
        self.ee_site_ids: Dict[str, int] = {}
        self.target_body_id: int = -1
        self.target_mocap_id: int = -1
        self.last_move_joint_names: list[str] | None = None
        self.last_move_mode: str | None = None
        self._side_joint_ids: Dict[str, np.ndarray] = {}
        self._side_dof_ids: Dict[str, np.ndarray] = {}
        self._side_qmins: Dict[str, np.ndarray] = {}
        self._side_qmaxs: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        mujoco.mj_resetData(self.model, self.data)
        self._build_maps()

        for name in self.actuator_joint_names:
            self.ctrl_targets[name] = 0.0

        self._sync_qpos_from_targets(self.actuator_joint_names)
        self._apply_ctrl_targets()
        mujoco.mj_forward(self.model, self.data)

        if self.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.55, 0.0, 0.75]
            self.viewer.cam.azimuth = 145
            self.viewer.cam.elevation = -18
            self.viewer.cam.distance = 2.2
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _build_maps(self) -> None:
        assert self.model is not None
        self.joint_maps.clear()
        self.actuator_joint_names.clear()
        self.ee_body_ids.clear()
        self.ee_site_ids.clear()
        self.target_body_id = -1
        self.target_mocap_id = -1
        self.last_move_joint_names = None
        self.last_move_mode = None

        for side, joint_names in SIDE_TO_JOINTS.items():
            joint_ids = []
            dof_ids = []
            qmins = []
            qmaxs = []
            for joint_name in joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise ValueError(f"Joint not found: {joint_name}")
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                qmin, qmax = self.model.jnt_range[joint_id].astype(float)
                self.joint_maps[joint_name] = JointMap(joint_name, joint_id, qpos_adr, dof_adr, qmin, qmax)
                joint_ids.append(joint_id)
                dof_ids.append(dof_adr)
                qmins.append(qmin)
                qmaxs.append(qmax)
            self._side_joint_ids[side] = np.asarray(joint_ids, dtype=int)
            self._side_dof_ids[side] = np.asarray(dof_ids, dtype=int)
            self._side_qmins[side] = np.asarray(qmins, dtype=float)
            self._side_qmaxs[side] = np.asarray(qmaxs, dtype=float)
            self.ee_body_ids[side] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SIDE_TO_EE_BODY[side])
            self.ee_site_ids[side] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, SIDE_TO_TCP_SITE[side])

        for candidate in TARGET_BODY_CANDIDATES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if body_id >= 0:
                self.target_body_id = body_id
                self.target_mocap_id = int(self.model.body_mocapid[body_id])
                break

        for aid in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[aid, 0])
            if jnt_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if joint_name is None or joint_name not in self.joint_maps:
                continue
            self.actuator_joint_names.append(joint_name)

        for aid in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[aid, 0])
            if jnt_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if joint_name is None or joint_name in self.joint_maps:
                continue
            qpos_adr = int(self.model.jnt_qposadr[jnt_id])
            dof_adr = int(self.model.jnt_dofadr[jnt_id])
            qmin, qmax = self.model.jnt_range[jnt_id].astype(float)
            self.joint_maps[joint_name] = JointMap(joint_name, jnt_id, qpos_adr, dof_adr, qmin, qmax)

        self.actuator_joint_names.clear()
        for aid in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[aid, 0])
            if jnt_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if joint_name is None or joint_name not in self.joint_maps:
                continue
            self.actuator_joint_names.append(joint_name)

    def set_target_pose(self, target_xyz: np.ndarray) -> None:
        assert self.model is not None and self.data is not None
        if self.target_mocap_id < 0:
            return
        self.data.mocap_pos[self.target_mocap_id] = np.asarray(target_xyz, dtype=float).reshape(3)

    def _sync_qpos_from_targets(self, joint_names: Iterable[str]) -> None:
        assert self.model is not None and self.data is not None
        for joint_name in joint_names:
            joint_map = self.joint_maps[joint_name]
            self.data.qpos[joint_map.qpos_adr] = self.ctrl_targets[joint_name]

    def _get_joint_q(self, joint_names: Iterable[str]) -> np.ndarray:
        assert self.data is not None
        return np.asarray([self.data.qpos[self.joint_maps[n].qpos_adr] for n in joint_names], dtype=float)

    def _joint_qpos_adr(self, joint_name: str) -> int:
        assert self.model is not None
        if joint_name in self.joint_maps:
            return self.joint_maps[joint_name].qpos_adr
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(joint_name)
        return int(self.model.jnt_qposadr[joint_id])

    def _set_joint_q(self, joint_names: Iterable[str], q: np.ndarray) -> None:
        assert self.model is not None and self.data is not None
        self._set_joint_q_kinematic(joint_names, q, sync_ctrl=True)

    def _set_joint_q_kinematic(self, joint_names: Iterable[str], q: np.ndarray, sync_ctrl: bool = False) -> None:
        assert self.model is not None and self.data is not None
        for value, joint_name in zip(q, joint_names):
            if sync_ctrl:
                self.ctrl_targets[joint_name] = float(value)
            self.data.qpos[self.joint_maps[joint_name].qpos_adr] = float(value)
        if sync_ctrl:
            self._apply_ctrl_targets()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_joint_ctrl_targets(self, joint_names: Iterable[str], q: np.ndarray) -> None:
        assert self.model is not None and self.data is not None
        for value, joint_name in zip(q, joint_names):
            if joint_name in self.ctrl_targets:
                self.ctrl_targets[joint_name] = float(value)
        self._apply_ctrl_targets()

    def _step_physics(self, steps: int = 1) -> None:
        assert self.model is not None and self.data is not None
        for _ in range(max(1, int(steps))):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
            time.sleep(self.dt)

    def _step_physics_for(self, duration: float) -> None:
        steps = max(1, int(round(float(duration) / self.dt)))
        self._step_physics(steps)

    def _apply_ctrl_targets(self) -> np.ndarray:
        assert self.model is not None and self.data is not None
        ctrl = np.zeros(self.model.nu, dtype=float)
        for aid, joint_name in enumerate(self.actuator_joint_names):
            ctrl[aid] = self.ctrl_targets[joint_name]
        self.data.ctrl[:] = ctrl
        return ctrl

    def _build_full_target_map(self, overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        assert self.model is not None
        targets = {name: 0.0 for name in self.actuator_joint_names}
        targets.update(self.ctrl_targets)
        if overrides:
            targets.update(overrides)
        return targets

    def _ctrl_vector_from_map(self, targets: Dict[str, float]) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=float)
        for aid, joint_name in enumerate(self.actuator_joint_names):
            ctrl[aid] = float(targets.get(joint_name, 0.0))
        return ctrl

    def _run_full_segment(self, target_map: Dict[str, float], duration: float) -> None:
        assert self.model is not None and self.data is not None
        start_map = self._build_full_target_map()
        start_ctrl = self._ctrl_vector_from_map(start_map)
        end_ctrl = self._ctrl_vector_from_map(self._build_full_target_map(target_map))
        steps = max(2, int(round(duration / self.dt)))
        for i in range(1, steps + 1):
            alpha = _quintic_blend(i / steps)
            ctrl = (1.0 - alpha) * start_ctrl + alpha * end_ctrl
            for aid, joint_name in enumerate(self.actuator_joint_names):
                self.ctrl_targets[joint_name] = float(ctrl[aid])
            self._apply_ctrl_targets()
            self._step_physics(1)

    def move_to_stand(self, duration: float = 2.5) -> None:
        # In Isaac Sim this role is played by the stable home posture and a fixed base.
        # For MuJoCo we apply the stable posture directly to avoid a dynamic sweep
        # through an unstable configuration.
        self.apply_stable_pose(settle_seconds=duration)

    def apply_stable_pose(self, settle_seconds: float = 1.0) -> None:
        assert self.model is not None and self.data is not None
        for joint_name, value in STABLE_Q.items():
            if joint_name in self.joint_maps:
                self.ctrl_targets[joint_name] = float(value)
        self._sync_qpos_from_targets(self.ctrl_targets.keys())
        self._apply_ctrl_targets()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if settle_seconds > 0.0:
            self._step_physics_for(settle_seconds)
        if self.viewer is not None:
            self.viewer.sync()

    def _get_selected_q(self, side: str) -> np.ndarray:
        assert self.data is not None
        joint_names = SIDE_TO_JOINTS[side]
        return np.asarray([self.data.qpos[self.joint_maps[n].qpos_adr] for n in joint_names], dtype=float)

    def _set_selected_q(self, side: str, q: np.ndarray) -> None:
        assert self.model is not None and self.data is not None
        joint_names = SIDE_TO_JOINTS[side]
        for value, joint_name in zip(q, joint_names):
            self.ctrl_targets[joint_name] = float(value)
        self._apply_ctrl_targets()
        self._step_physics(1)

    def _set_finger_q(self, side: str, finger_targets: Dict[str, float]) -> None:
        assert self.model is not None and self.data is not None
        for joint_name, value in finger_targets.items():
            if joint_name in self.joint_maps:
                self.ctrl_targets[joint_name] = float(value)
        self._apply_ctrl_targets()
        self._step_physics(1)

    def _move_fingers(self, side: str, close: bool, duration: float = 0.5) -> None:
        joint_names = SIDE_TO_FINGER_JOINTS[side]
        q_start = np.asarray([self.data.qpos[self._joint_qpos_adr(name)] for name in joint_names], dtype=float)
        if close:
            q_end = np.asarray([FINGER_CLOSE_Q[side][name] for name in joint_names], dtype=float)
        else:
            q_end = np.asarray([FINGER_OPEN_Q[side][name] for name in joint_names], dtype=float)
        self._run_segment(joint_names, q_start, q_end, duration)

    def _move_fingers_locked(self, side: str, close: bool, duration: float = 0.5) -> None:
        assert self.model is not None and self.data is not None
        finger_joint_names = SIDE_TO_FINGER_JOINTS[side]
        locked_joint_names = [name for name in self.actuator_joint_names if name not in finger_joint_names]
        locked_q = np.asarray([self.data.qpos[self.joint_maps[name].qpos_adr] for name in locked_joint_names], dtype=float)
        q_start = np.asarray([self.data.qpos[self._joint_qpos_adr(name)] for name in finger_joint_names], dtype=float)
        if close:
            q_end = np.asarray([FINGER_CLOSE_Q[side][name] for name in finger_joint_names], dtype=float)
        else:
            q_end = np.asarray([FINGER_OPEN_Q[side][name] for name in finger_joint_names], dtype=float)

        steps = max(2, int(round(duration / self.dt)))
        self._set_joint_ctrl_targets(locked_joint_names, locked_q)
        for i in range(steps + 1):
            if i == 0:
                continue
            q = _interpolate(q_start, q_end, i * self.dt, duration)
            self._set_joint_ctrl_targets(finger_joint_names, q)
            self._set_joint_ctrl_targets(locked_joint_names, locked_q)
            self._step_physics(1)

    def _ee_pose(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        assert self.data is not None and self.model is not None
        site_id = self.ee_site_ids.get(side, -1)
        if site_id >= 0:
            p = self.data.site_xpos[site_id].copy()
            R = self.data.site_xmat[site_id].reshape(3, 3).copy()
            return p, R
        body_id = self.ee_body_ids[side]
        p = self.data.xpos[body_id].copy()
        R = self.data.xmat[body_id].reshape(3, 3).copy()
        return p, R

    def _solve_ik(
        self,
        joint_names: list[str],
        ref_name: str,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        seed: Optional[np.ndarray] = None,
        max_iter: int = 60,
        posture_ref: Optional[np.ndarray] = None,
        posture_weights: Optional[np.ndarray] = None,
        posture_weight: float = 0.25,
        orientation_weight: float = 0.0,
        qmin_override: Optional[np.ndarray] = None,
        qmax_override: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        assert self.model is not None and self.data is not None
        q = self._get_joint_q(joint_names) if seed is None else np.asarray(seed, dtype=float).copy()
        if qmin_override is None:
            qmin = np.asarray([self.joint_maps[n].qmin for n in joint_names], dtype=float)
        else:
            qmin = np.asarray(qmin_override, dtype=float).reshape(-1)
        if qmax_override is None:
            qmax = np.asarray([self.joint_maps[n].qmax for n in joint_names], dtype=float)
        else:
            qmax = np.asarray(qmax_override, dtype=float).reshape(-1)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ref_name)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ref_name)
        dof_ids = np.asarray([self.joint_maps[n].dof_adr for n in joint_names], dtype=int)
        use_site = site_id >= 0

        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        ctrl_backup = self.data.ctrl.copy()
        ctrl_targets_backup = self.ctrl_targets.copy()
        try:
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            for _ in range(max_iter):
                self._set_joint_q_kinematic(joint_names, q, sync_ctrl=False)
                if use_site:
                    p_cur = self.data.site_xpos[site_id].copy()
                    R_cur = self.data.site_xmat[site_id].reshape(3, 3).copy()
                else:
                    p_cur = self.data.xpos[body_id].copy()
                    R_cur = self.data.xmat[body_id].reshape(3, 3).copy()
                e_pos = target_pos - p_cur
                e_rot = np.zeros(3, dtype=float)
                if target_rot is not None:
                    e_rot = _rotation_error_vector(np.asarray(target_rot, dtype=float).reshape(3, 3), R_cur)
                if np.linalg.norm(e_pos) < 1e-3 and (
                    target_rot is None or orientation_weight <= 0.0 or np.linalg.norm(e_rot) < 1e-2
                ):
                    return q.copy()

                if use_site:
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                else:
                    mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                Jp = jacp[:, dof_ids]
                e = e_pos
                J = Jp
                if target_rot is not None and orientation_weight > 0.0:
                    Jr = jacr[:, dof_ids]
                    e = np.concatenate([e_pos, orientation_weight * e_rot], axis=0)
                    J = np.vstack([Jp, orientation_weight * Jr])
                lam = 0.04
                dq = J.T @ np.linalg.solve(J @ J.T + (lam * lam) * np.eye(J.shape[0]), e)
                if posture_ref is not None:
                    posture_delta = np.asarray(posture_ref, dtype=float).reshape(-1) - q
                    if posture_weights is not None:
                        posture_delta = posture_delta * np.asarray(posture_weights, dtype=float).reshape(-1)
                    dq += posture_weight * posture_delta
                step_norm = float(np.linalg.norm(dq))
                if step_norm > 0.18:
                    dq *= 0.18 / step_norm

                q = np.clip(q + dq, qmin, qmax)
        finally:
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            self.data.ctrl[:] = ctrl_backup
            self.ctrl_targets = ctrl_targets_backup
            mujoco.mj_forward(self.model, self.data)

        return None

    def _plan_cartesian_move(
        self,
        side: str,
        joint_candidates: list[list[str]],
        ref_name: str,
        cartesian_points: list[np.ndarray],
        posture_mode: str = "none",
        target_rot: Optional[np.ndarray] = None,
        orientation_weight: float = 0.0,
    ) -> Tuple[bool, Optional[list[str]], list[np.ndarray]]:
        assert self.model is not None
        q_path: list[np.ndarray] = []
        joint_names: list[str] | None = None

        for candidate in joint_candidates:
            q_path.clear()
            q_seed = self._get_joint_q(candidate)
            posture_ref = np.asarray([STABLE_Q[name] for name in candidate], dtype=float)
            posture_weights = np.ones(len(candidate), dtype=float)
            qmin_override = None
            qmax_override = None

            if posture_mode == "soft_torso" and candidate == SIDE_TO_JOINTS[side]:
                qmin_override = np.asarray([self.joint_maps[name].qmin for name in candidate], dtype=float)
                qmax_override = np.asarray([self.joint_maps[name].qmax for name in candidate], dtype=float)
                for idx, name in enumerate(candidate):
                    if name in TORSO_SOFT_LIMITS:
                        band = float(TORSO_SOFT_LIMITS[name])
                        center = float(STABLE_Q[name])
                        qmin_override[idx] = max(qmin_override[idx], center - band)
                        qmax_override[idx] = min(qmax_override[idx], center + band)

            ok = True
            for point in cartesian_points:
                q_sol = self._solve_ik(
                    joint_names=candidate,
                    ref_name=ref_name,
                    target_pos=np.asarray(point, dtype=float),
                    target_rot=target_rot,
                    seed=q_seed,
                    posture_ref=posture_ref if posture_mode != "none" else None,
                    posture_weights=posture_weights if posture_mode != "none" else None,
                    posture_weight=0.0,
                    orientation_weight=orientation_weight,
                    qmin_override=qmin_override,
                    qmax_override=qmax_override,
                )
                if q_sol is None:
                    ok = False
                    break
                q_path.append(q_sol)
                q_seed = q_sol

            if ok and q_path:
                joint_names = list(candidate)
                break

        return joint_names is not None, joint_names, q_path

    def _run_segment(self, joint_names: list[str], q_start: np.ndarray, q_end: np.ndarray, duration: float) -> None:
        assert self.model is not None and self.data is not None
        steps = max(2, int(round(duration / self.dt)))
        for i in range(steps + 1):
            if i == 0:
                continue
            q = _interpolate(q_start, q_end, i * self.dt, duration)
            self._set_joint_ctrl_targets(joint_names, q)
            self._step_physics(1)

    def _hold_selected_q(self, joint_names: list[str], q_hold: np.ndarray, duration: float) -> None:
        assert self.model is not None and self.data is not None
        steps = max(1, int(round(duration / self.dt)))
        self._set_joint_ctrl_targets(joint_names, q_hold)
        for _ in range(steps):
            self._step_physics(1)

    def move_to(
        self,
        side: str,
        target_xyz: np.ndarray,
        approach_height: float = 0.18,
        final_offset: float = 0.05,
        duration: float = 1.2,
        yaw: float = 0.0,
        execute: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        if side not in SIDE_TO_JOINTS:
            raise ValueError(f"Unknown side: {side}")

        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        self.set_target_pose(target_xyz)
        ee_now, _ = self._ee_pose(side)
        pre_xyz = target_xyz + np.array([0.0, 0.0, approach_height], dtype=float)
        goal_xyz = target_xyz + np.array([0.0, 0.0, final_offset], dtype=float)
        lift_z = max(float(ee_now[2]), float(pre_xyz[2])) + max(0.10, float(approach_height) * 0.5)
        lift_start = np.array([float(ee_now[0]), float(ee_now[1]), lift_z], dtype=float)
        lift_target = np.array([float(target_xyz[0]), float(target_xyz[1]), lift_z], dtype=float)
        cartesian_points = _cartesian_path_points([ee_now, lift_start, lift_target, pre_xyz, goal_xyz], samples_per_segment=16)
        grasp_rot = _top_down_grasp_rotation(yaw)

        preferred_ok, joint_names, q_path = self._plan_cartesian_move(
            side=side,
            joint_candidates=FOLLOW_CANDIDATE_JOINT_SETS[side],
            ref_name=SIDE_TO_TCP_SITE[side],
            cartesian_points=cartesian_points,
            posture_mode="soft_torso",
        )

        move_mode = "preferred_tcp"
        if not preferred_ok or joint_names is None or not q_path:
            legacy_ok, joint_names, q_path = self._plan_cartesian_move(
                side=side,
                joint_candidates=FOLLOW_CANDIDATE_JOINT_SETS[side],
                ref_name=SIDE_TO_EE_BODY[side],
                cartesian_points=cartesian_points,
                posture_mode="none",
            )
            if not legacy_ok or joint_names is None or not q_path:
                return False, None
            move_mode = "legacy_body"
        else:
            refined_goal = self._solve_ik(
                joint_names=joint_names,
                ref_name=SIDE_TO_TCP_SITE[side],
                target_pos=goal_xyz,
                target_rot=grasp_rot,
                seed=q_path[-1],
                posture_ref=np.asarray([STABLE_Q[name] for name in joint_names], dtype=float),
                posture_weights=np.ones(len(joint_names), dtype=float),
                posture_weight=0.0,
                orientation_weight=0.02,
            )
            if refined_goal is not None:
                q_path[-1] = refined_goal
                move_mode = "preferred_tcp+refine"

        q_goal = q_path[-1].copy()
        self.last_move_joint_names = joint_names
        self.last_move_mode = move_mode

        if execute:
            self._move_fingers_locked(side, close=False, duration=0.20)
            q_now = self._get_joint_q(joint_names)
            self._run_segment(joint_names, q_now, q_path[0], duration * 0.12)
            if len(q_path) > 1:
                per_step = max(0.0, float(duration) * 0.88 / max(1, len(q_path) - 1))
                for prev_q, next_q in zip(q_path[:-1], q_path[1:]):
                    self._run_segment(joint_names, prev_q, next_q, per_step)
            if move_mode.startswith("preferred_tcp"):
                q_stable = self._get_joint_q(joint_names)
                self._hold_selected_q(joint_names, q_stable, duration=0.15)
                self._move_fingers_locked(side, close=False, duration=0.20)
                self._move_fingers_locked(side, close=True, duration=0.45)
                self.last_move_mode = f"{move_mode}+close"

        return True, q_goal

    def print_state(self, side: str, joint_names: Optional[list[str]] = None) -> None:
        q = self._get_joint_q(joint_names or SIDE_TO_JOINTS[side])
        print(f"[{side}] q = {np.array2string(q, precision=4, suppress_small=True)}")

    def hold_until_closed(self) -> None:
        if self.viewer is None:
            return
        while self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.dt)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move the Galaxea R1Pro arm to a world coordinate.")
    parser.add_argument("--side", choices=["left", "right"], help="Arm side to move.")
    parser.add_argument("--x", type=float, help="Target x in world coordinates.")
    parser.add_argument("--y", type=float, help="Target y in world coordinates.")
    parser.add_argument("--z", type=float, help="Target z in world coordinates.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Optional fixed yaw in radians.")
    parser.add_argument("--approach-height", type=float, default=0.18, help="Height above target for pre-grasp pose.")
    parser.add_argument("--final-offset", type=float, default=0.05, help="Final offset above the target point.")
    parser.add_argument("--duration", type=float, default=1.2, help="Total motion duration in seconds.")
    parser.add_argument("--no-viewer", action="store_true", help="Disable interactive viewer.")
    parser.add_argument("--dry-run", action="store_true", help="Solve IK and print joint angles without stepping.")
    parser.add_argument("--once", action="store_true", help="Run one move from CLI arguments and exit.")
    parser.add_argument("--skip-stand", action="store_true", help="Skip the initial standing pose.")
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=-1,
        help="If negative, keep the viewer open until you close it. If non-negative, wait that many sync ticks and exit.",
    )
    return parser


def _prompt_optional_float(prompt: str, default: Optional[float] = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            if default is not None:
                return float(default)
            print("请输入数值。")
            continue
        try:
            return float(raw)
        except ValueError:
            print("输入格式错误，请输入数字。")


def _prompt_side(default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"请选择手臂 side(left/right)，输入 q 退出{suffix}: ").strip().lower()
        if not raw and default is not None:
            return default
        if raw in {"q", "quit", "exit"}:
            return "q"
        if raw in {"left", "right"}:
            return raw
        print("请输入 left / right / q。")


def _prompt_target(default: Optional[np.ndarray] = None) -> np.ndarray:
    default_text = None if default is None else np.array2string(default, precision=4, suppress_small=True)
    while True:
        raw = input(
            f"请输入目标坐标 x y z（空格分隔）{f'[{default_text}]' if default_text else ''}: "
        ).strip()
        if not raw and default is not None:
            return np.asarray(default, dtype=float).reshape(3)
        parts = raw.split()
        if len(parts) != 3:
            print("请输入 3 个数，例如：0.62 0.20 0.43")
            continue
        try:
            return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
        except ValueError:
            print("坐标格式错误，请重新输入。")


def _interactive_loop(env: R1ProReachEnv, defaults: argparse.Namespace) -> int:
    last_side = defaults.side if defaults.side in {"left", "right"} else "left"
    last_target = None

    print("\n进入交互模式。")
    print("输入 q 可以退出。")
    while True:
        if env.viewer is not None and not env.viewer.is_running():
            break

        side = _prompt_side(default=last_side)
        if side == "q":
            break
        last_side = side

        target = _prompt_target(default=last_target)
        last_target = target.copy()

        yaw = _prompt_optional_float("yaw（弧度）", defaults.yaw)
        approach_height = _prompt_optional_float("approach_height", defaults.approach_height)
        final_offset = _prompt_optional_float("final_offset", defaults.final_offset)
        duration = _prompt_optional_float("duration", defaults.duration)

        ok, q_goal = env.move_to(
            side=side,
            target_xyz=target,
            approach_height=approach_height,
            final_offset=final_offset,
            duration=duration,
            yaw=yaw,
            execute=not defaults.dry_run,
        )
        if not ok or q_goal is None:
            print("[ERROR] IK failed for the requested target.")
            continue

        env.print_state(side, env.last_move_joint_names)
        print(f"[{side}] target = {np.array2string(target, precision=4, suppress_small=True)}")
        print(f"[{side}] goal_q = {np.array2string(q_goal, precision=4, suppress_small=True)}")

        if defaults.dry_run:
            continue

    return 0


def _maybe_move_to_stand(env: R1ProReachEnv, args: argparse.Namespace) -> None:
    if args.skip_stand or args.dry_run:
        return
    print("[INFO] Moving to standing pose...")
    env.move_to_stand(duration=2.5)


def main() -> int:
    args = build_arg_parser().parse_args()
    env = R1ProReachEnv(enable_viewer=not args.no_viewer)
    try:
        env.reset()
        _maybe_move_to_stand(env, args)
        if args.once:
            if args.side is None or args.x is None or args.y is None or args.z is None:
                print("[ERROR] --once requires --side, --x, --y, and --z.")
                return 2
            target = np.array([args.x, args.y, args.z], dtype=float)
            ok, q_goal = env.move_to(
                side=args.side,
                target_xyz=target,
                approach_height=args.approach_height,
                final_offset=args.final_offset,
                duration=args.duration,
                yaw=args.yaw,
                execute=not args.dry_run,
            )
            if not ok or q_goal is None:
                print("[ERROR] IK failed for the requested target.")
                return 2

            env.print_state(args.side, env.last_move_joint_names)
            print(f"[{args.side}] target = {np.array2string(target, precision=4, suppress_small=True)}")
            print(f"[{args.side}] goal_q = {np.array2string(q_goal, precision=4, suppress_small=True)}")

            if args.dry_run:
                return 0
            if env.viewer is not None:
                if int(args.hold_steps) < 0:
                    env.hold_until_closed()
                else:
                    for _ in range(max(0, int(args.hold_steps))):
                        if not env.viewer.is_running():
                            break
                        env.viewer.sync()
                        time.sleep(env.dt)
            return 0

        return _interactive_loop(env, args)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
