"""
Candidate MuJoCo sandbox execution helpers.

Provides:
  1. State capture/restore for isolated candidate execution
  2. MuJoCo scene cloning/building for candidate plans
  3. Experiment-compatible methods used directly by recovery_steps skills
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco
import spatialmath as sm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from utils import mj as mj_utils
from arm.robot import UR5e
from arm.motion_planning import (
    JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner,
    LinePositionParameter, OneAttitudeParameter, CartesianParameter,
)
from experience_system.memory.calibration import apply_calibration_to_position, calibrated_attach_distance

SCENE_XML = str(ROOT / 'scene' / 'scene.xml')
SIM_TIMESTEP = 0.002
GRASP_ATTACH_MAX_DISTANCE = 0.045


def _env_headless() -> bool:
    return os.getenv("UR5E_HEADLESS", "").strip().lower() in {"1", "true", "yes", "y", "on"}


class CandidateSandbox:
    """Candidate MuJoCo sandbox execution helper."""

    def __init__(self, noise_scale=0.02, scene_xml: str | Path | None = None):
        self.noise_scale = noise_scale
        self.scene_xml = str(Path(scene_xml).resolve()) if scene_xml is not None else SCENE_XML

    # ── State management ────────────────────────────────────────────

    def capture_state(self, experiment):
        """Capture full MuJoCo + experiment state for later restore."""
        return {
            "qpos": experiment.data.qpos.copy(),
            "qvel": experiment.data.qvel.copy(),
            "ctrl": experiment.data.ctrl.copy(),
            "act": experiment.data.act.copy(),
            "time": float(experiment.data.time),
            "robot_q0": experiment.robot.get_joint(),
            "action": experiment.action.copy(),
        }

    def restore_state(self, experiment, state):
        """Restore experiment to a previously captured state."""
        experiment.data.qpos[:] = state["qpos"]
        experiment.data.qvel[:] = state["qvel"]
        experiment.data.ctrl[:] = state["ctrl"]
        experiment.data.act[:] = state["act"]
        experiment.data.time = state["time"]
        mujoco.mj_forward(experiment.model, experiment.data)

        experiment.robot.set_joint(state["robot_q0"])
        experiment.action[:] = state["action"]

    # ── Candidate MuJoCo scene ──────────────────────────────────────

    def build_virtual_scene(
        self,
        perceived_pos,
        perceived_quat=None,
        enable_viewer=False,
        initial_robot_q=None,
        initial_action=None,
        sandbox_calibration=None,
    ):
        """Create a fresh MuJoCo instance with apple at the PERCEIVED position.

        Parameters
        ----------
        perceived_pos : (3,) array — apple position from perception pipeline
        perceived_quat : (4,) array or None — wxyz quaternion (identity if None)
        enable_viewer : bool — if True, launch a passive viewer for the virtual scene
        initial_robot_q : (6,) array or None — current real-sim robot joint pose
        initial_action : array or None — current real-sim actuator command
        sandbox_calibration : dict or None — optional gap-derived scene calibration

        Returns
        -------
        VirtualScene handle with .model, .data, .robot, .action
        """
        model = mujoco.MjModel.from_xml_path(self.scene_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        if sandbox_calibration:
            calibrated_pos = np.asarray(
                apply_calibration_to_position(perceived_pos, sandbox_calibration),
                dtype=np.float64,
            )
        else:
            calibrated_pos = np.asarray(perceived_pos, dtype=np.float64)

        # Place apple at perceived/calibrated position (NOT ground truth)
        apple_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple0")
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == apple_body_id:
                adr = model.jnt_qposadr[j]
                data.qpos[adr:adr+3] = calibrated_pos
                q = np.asarray(perceived_quat, dtype=np.float64) if perceived_quat is not None else np.array([1, 0, 0, 0])
                data.qpos[adr+3:adr+7] = q
                break
        mujoco.mj_forward(model, data)

        # UR5e setup
        robot = UR5e()
        robot.set_base(mj_utils.get_body_pose(model, data, "ur5e_base").t)
        if initial_robot_q is None:
            robot_q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            robot_q_init = np.asarray(initial_robot_q, dtype=np.float64).reshape(6)
        robot.set_joint(robot_q_init)

        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for i, jn in enumerate(joint_names):
            mj_utils.set_joint_q(model, data, jn, robot_q_init[i])
        mujoco.mj_forward(model, data)

        mj_utils.attach(model, data, "attach", "2f85", robot.fkine(robot_q_init))
        tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        robot.set_tool(tool)

        action = np.zeros(model.nu)
        if initial_action is not None:
            src = np.asarray(initial_action, dtype=np.float64).reshape(-1)
            action[:min(model.nu, src.size)] = src[:min(model.nu, src.size)]
        action[:6] = robot_q_init

        for _ in range(500):
            data.ctrl[:] = action
            mujoco.mj_step(model, data)

        # Optional viewer for the virtual scene
        viewer = None
        if enable_viewer and not _env_headless():
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0.6, 0.25, 0.2]
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.2
            viewer.sync()

        attach_max_distance = calibrated_attach_distance(
            GRASP_ATTACH_MAX_DISTANCE,
            sandbox_calibration,
        ) if sandbox_calibration else GRASP_ATTACH_MAX_DISTANCE
        scene = VirtualScene(model=model, data=data, robot=robot,
                             joint_names=joint_names, apple_body_id=apple_body_id,
                             viewer=viewer, action=action)
        scene.sandbox_calibration = sandbox_calibration or {}
        scene.perceived_pos = np.asarray(perceived_pos, dtype=np.float64).tolist()
        scene.calibrated_pos = calibrated_pos.tolist()
        scene.attach_max_distance = float(attach_max_distance)
        return scene

    def clone_experiment_scene(self, experiment, enable_viewer=False):
        """Create a sandbox by cloning the current MuJoCo experiment state.

        This is used for candidate recovery validation.  Unlike
        build_virtual_scene(), it does not rebuild a perceived scene or reset the
        apple orientation; it copies the current qpos/qvel/ctrl/action/time from
        the real MuJoCo experiment into a separate model/data pair.
        """
        model = mujoco.MjModel.from_xml_path(self.scene_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        robot = UR5e()
        robot.set_base(mj_utils.get_body_pose(model, data, "ur5e_base").t)
        robot_q = np.asarray(experiment.robot.get_joint(), dtype=np.float64).reshape(6)
        robot.set_joint(robot_q)

        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for i, jn in enumerate(joint_names):
            mj_utils.set_joint_q(model, data, jn, robot_q[i])
        mujoco.mj_forward(model, data)

        mj_utils.attach(model, data, "attach", "2f85", robot.fkine(robot_q))
        tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        robot.set_tool(tool)

        if model.nq != experiment.model.nq or model.nv != experiment.model.nv or model.nu != experiment.model.nu:
            raise ValueError(
                "sandbox clone model shape mismatch: "
                f"clone(nq={model.nq}, nv={model.nv}, nu={model.nu}) "
                f"experiment(nq={experiment.model.nq}, nv={experiment.model.nv}, nu={experiment.model.nu})"
            )

        data.qpos[:] = np.asarray(experiment.data.qpos, dtype=np.float64)
        data.qvel[:] = np.asarray(experiment.data.qvel, dtype=np.float64)
        data.ctrl[:] = np.asarray(experiment.data.ctrl, dtype=np.float64)
        if data.act.shape == experiment.data.act.shape:
            data.act[:] = np.asarray(experiment.data.act, dtype=np.float64)
        if data.eq_active.shape == experiment.data.eq_active.shape:
            data.eq_active[:] = np.asarray(experiment.data.eq_active, dtype=np.uint8)
        if data.mocap_pos.shape == experiment.data.mocap_pos.shape:
            data.mocap_pos[:] = np.asarray(experiment.data.mocap_pos, dtype=np.float64)
        if data.mocap_quat.shape == experiment.data.mocap_quat.shape:
            data.mocap_quat[:] = np.asarray(experiment.data.mocap_quat, dtype=np.float64)
        data.time = float(experiment.data.time)
        mujoco.mj_forward(model, data)

        action = np.zeros(model.nu)
        src_action = np.asarray(experiment.action, dtype=np.float64).reshape(-1)
        action[:min(model.nu, src_action.size)] = src_action[:min(model.nu, src_action.size)]

        viewer = None
        if enable_viewer and not _env_headless():
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0.6, 0.25, 0.2]
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.2
            viewer.sync()

        apple_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple0")
        scene = VirtualScene(model=model, data=data, robot=robot,
                             joint_names=joint_names, apple_body_id=apple_body_id,
                             viewer=viewer, action=action)
        apple_pos = data.body(apple_body_id).xpos.copy()
        scene.sandbox_calibration = {}
        scene.perceived_pos = apple_pos.tolist()
        scene.calibrated_pos = apple_pos.tolist()
        scene.attach_max_distance = GRASP_ATTACH_MAX_DISTANCE
        scene.clone_source = "current_mujoco_state"
        return scene

    # ── MuJoCo motion backend used by recovery_steps skills ─────────

    @staticmethod
    def _v_move_joints(scene, q_target, duration=1.0):
        q0 = scene.robot.get_joint()
        param = JointParameter(q0, np.asarray(q_target))
        vel = QuinticVelocityParameter(duration)
        traj = TrajectoryParameter(param, vel)
        planner = TrajectoryPlanner(traj)
        for t in np.linspace(0, duration, int(duration / SIM_TIMESTEP)):
            interp = planner.interpolate(t)
            scene.robot.move_joint(interp)
            scene.action[:6] = interp
            scene.data.ctrl[:] = scene.action
            mujoco.mj_step(scene.model, scene.data)
            if scene.viewer:
                scene.viewer.sync()

    @staticmethod
    def _v_cartesian(scene, T_target, duration=1.0):
        T_cur = scene.robot.get_cartesian()
        pos_p = LinePositionParameter(T_cur.t, T_target.t)
        att_p = OneAttitudeParameter(sm.SO3(T_cur.R), sm.SO3(T_target.R))
        cart_p = CartesianParameter(pos_p, att_p)
        vel = QuinticVelocityParameter(duration)
        traj = TrajectoryParameter(cart_p, vel)
        planner = TrajectoryPlanner(traj)
        for t in np.linspace(0, duration, int(duration / SIM_TIMESTEP)):
            interp = planner.interpolate(t)
            scene.robot.move_cartesian(interp)
            scene.action[:6] = scene.robot.get_joint()
            scene.data.ctrl[:] = scene.action
            mujoco.mj_step(scene.model, scene.data)
            if scene.viewer:
                scene.viewer.sync()

    @staticmethod
    def _v_gripper_close(scene):
        for _ in range(1500):
            if float(scene.action[-1]) >= 254.0:
                break
            scene.action[-1] += 0.2
            scene.action[-1] = np.min([scene.action[-1], 255])
            scene.data.ctrl[:] = scene.action
            mujoco.mj_step(scene.model, scene.data)
            if scene.viewer:
                scene.viewer.sync()

    @staticmethod
    def _v_snap(scene, reset_gripper=False):
        scene.action[:6] = scene.robot.get_joint()
        if reset_gripper:
            scene.action[-1] = 0.0
        if scene.viewer:
            scene.viewer.sync()

    @staticmethod
    def _v_steps(scene, n):
        for _ in range(n):
            scene.data.ctrl[:] = scene.action
            mujoco.mj_step(scene.model, scene.data)
            if scene.viewer:
                scene.viewer.sync()



class VirtualScene:
    """Handle returned by CandidateSandbox.build_virtual_scene()."""
    def __init__(self, model, data, robot, joint_names, apple_body_id, viewer=None, action=None):
        self.model = model
        self.data = data
        self.robot = robot
        self.joint_names = joint_names
        self.apple_body_id = apple_body_id
        self.viewer = viewer
        self.action = np.zeros(model.nu)
        if action is not None:
            src = np.asarray(action, dtype=np.float64).reshape(-1)
            self.action[:min(model.nu, src.size)] = src[:min(model.nu, src.size)]
        self.pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.virtual_failure = None
        self.metrics = {"skill_results": []}
        self.recovery_plan = {"steps": []}
        self.perception = None
        self.T_wo = None
        self.T_pregrasp = None
        self._fixed_vertical_grasp_context = None
        self.anomaly_type = ""
        self.body_qpos_adr_cache = {}
        for j in range(model.njnt):
            body_id = model.jnt_bodyid[j]
            if body_id >= 0:
                self.body_qpos_adr_cache[body_id] = model.jnt_qposadr[j]

    def _step(self):
        self.data.ctrl[:] = self.action
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def _step_n(self, n):
        for _ in range(int(n)):
            self._step()

    def _move_joints(self, q_target, duration=1.0):
        CandidateSandbox._v_move_joints(self, q_target, duration)
        return {"skill": "joint_move", "success": True, "reason": "ok"}

    def _move_cartesian(self, T_target, duration=1.0):
        CandidateSandbox._v_cartesian(self, T_target, duration)
        final = np.asarray(self.robot.get_cartesian().t, dtype=np.float64)
        target = np.asarray(T_target.t, dtype=np.float64)
        pos_error = float(np.linalg.norm(final - target))
        return {
            "skill": "cartesian_move",
            "success": bool(pos_error <= 0.03),
            "reason": "ok" if pos_error <= 0.03 else "final_pose_error_exceeded",
            "target_pos": target.tolist(),
            "final_pos": final.tolist(),
            "pos_error": pos_error,
        }

    def _gripper_open(self):
        self.action[-1] = 0.0
        self._step_n(500)
        return {"skill": "gripper_open", "success": True, "reason": "ok"}

    def _gripper_close(self):
        CandidateSandbox._v_gripper_close(self)
        return {"skill": "gripper_close", "success": True, "reason": "ok"}

    def _contact_summary(self):
        contacts = {"left_contact": False, "right_contact": False}
        for ic in range(self.data.ncon):
            c = self.data.contact[ic]
            g1 = self.model.geom(c.geom1).name if c.geom1 < self.model.ngeom else ""
            g2 = self.model.geom(c.geom2).name if c.geom2 < self.model.ngeom else ""
            if "pad" in g1.lower() and "apple" in g2.lower():
                if "left" in g1.lower():
                    contacts["left_contact"] = True
                if "right" in g1.lower():
                    contacts["right_contact"] = True
            if "pad" in g2.lower() and "apple" in g1.lower():
                if "left" in g2.lower():
                    contacts["left_contact"] = True
                if "right" in g2.lower():
                    contacts["right_contact"] = True
        return contacts

    def _current_skill_observation(self):
        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        return {
            "contact": self._contact_summary(),
            "gripper_action": float(self.action[-1]),
            "tracked_body": "",
            "pinch_distance": float(np.linalg.norm(apple_pos - pinch_pos)),
            "object_pos": apple_pos.tolist(),
        }

    def _record_skill_result(self, result):
        record = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        self.metrics.setdefault("skill_results", []).append(record)
        return record

    def _record_basic_skill(self, skill, success=True, reason="ok", **extra):
        obs = self._current_skill_observation()
        record = {
            "skill": skill,
            "success": bool(success),
            "reason": reason,
            "contact": obs["contact"],
            "gripper_action": obs["gripper_action"],
            "pinch_distance": obs["pinch_distance"],
            "object_pos": obs["object_pos"],
            "extra": extra,
        }
        self.metrics.setdefault("skill_results", []).append(record)
        return record

    def _save_keyframe(self, *args, **kwargs):
        return None

    def _resolve_skill_target_position(self, skill):
        perceived = self.metrics.get("perceived_position") or self.metrics.get("observed_pos")
        if perceived is not None:
            return np.asarray(perceived, dtype=np.float64)
        raise RuntimeError("perception_target_unavailable")

    def close(self):
        if self.perception is not None and hasattr(self.perception, "close"):
            self.perception.close()
            self.perception = None
        if self.viewer:
            self.viewer.close()
            self.viewer = None
