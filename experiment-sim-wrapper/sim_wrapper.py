"""
Shadow simulation wrapper for verifying recovery strategies.

Provides:
  1. State capture/restore — run shadow sim without disturbing main sim
  2. Recovery verification — test candidate grasp positions in shadow sim
     using the same recovery logic as the real experiment (v4 flow)
  3. Virtual scene building (Condition A) — create a fresh MuJoCo instance
     with the apple at a PERCEIVED position (not ground truth), plan
     recovery inside it, and return a serialisable plan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco
import spatialmath as sm
from utils import mj as mj_utils

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'manipulator_grasp'))

from arm.robot import UR5e
from arm.motion_planning import (
    JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner,
    LinePositionParameter, OneAttitudeParameter, CartesianParameter,
)

SCENE_XML = str(ROOT / 'manipulator_grasp' / 'assets' / 'scenes' / 'apple_pear_runtime_refined.xml')
SIM_TIMESTEP = 0.002


class SimWrapper:
    """Shadow simulation bridge for condition B recovery verification."""

    def __init__(self, noise_scale=0.02):
        self.noise_scale = noise_scale

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

    # ── Recovery verification (Condition B) ──────────────────────────

    def verify_recovery(self, experiment, candidate_pos):
        """Test whether recovery succeeds at candidate_pos using the real recovery logic.

        Mirrors the exact flow from _execute_recovery:
          q1 joint move → snap+reattach+reset_gripper → T_pregrasp → T_wo → snap → close → lift.
        Returns True if apple Z change > 3cm.
        """
        # Build T_wo/T_pregrasp at candidate_pos with the experiment's rotation
        R = experiment.T_wo.R
        T_wo = sm.SE3.Trans(candidate_pos) * sm.SE3(sm.SO3(R, check=False))
        T_pre = sm.SE3.Trans(candidate_pos + np.array([0, 0, 0.127])) * sm.SE3(sm.SO3(R, check=False))

        apple_z_before = float(experiment.data.body(experiment.apple_body_id).xpos[2])

        # Step 1: Joint move to q1
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        experiment._move_joints(q1, 1.0)

        # Step 2: Snap arm + reset gripper + reattach at flange
        experiment._snap_to_robot_joints(reset_gripper=True)
        flange_pose = mj_utils.get_body_pose(experiment.model, experiment.data, "flange")
        mj_utils.attach(experiment.model, experiment.data, "attach", "2f85", flange_pose)
        experiment._step_n(500)

        # Step 3: Cartesian to pre-grasp
        experiment.robot.set_joint(q1)
        experiment._move_cartesian(T_pre, 1.0)

        # Step 4: Cartesian to grasp
        experiment._move_cartesian(T_wo, 1.0)
        experiment._step_n(500)

        # Step 5: Snap joints
        experiment._snap_to_robot_joints()

        # Step 6: Close gripper
        experiment._gripper_close()

        # Step 7: Lift
        T_lift = sm.SE3.Trans(0, 0, 0.3) * T_wo
        experiment._move_cartesian(T_lift, 1.0)

        # Check result
        apple_z_after = float(experiment.data.body(experiment.apple_body_id).xpos[2])
        return apple_z_after - apple_z_before > 0.03

    # ── Virtual scene (Condition A) ─────────────────────────────────

    def build_virtual_scene(self, perceived_pos, perceived_quat=None, enable_viewer=False):
        """Create a fresh MuJoCo instance with apple at the PERCEIVED position.

        Parameters
        ----------
        perceived_pos : (3,) array — apple position from perception pipeline
        perceived_quat : (4,) array or None — wxyz quaternion (identity if None)
        enable_viewer : bool — if True, launch a passive viewer for the virtual scene

        Returns
        -------
        VirtualScene handle with .model, .data, .robot, .action
        """
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        # Place apple at perceived position (NOT ground truth)
        apple_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple0")
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == apple_body_id:
                adr = model.jnt_qposadr[j]
                data.qpos[adr:adr+3] = np.asarray(perceived_pos, dtype=np.float64)
                q = np.asarray(perceived_quat, dtype=np.float64) if perceived_quat is not None else np.array([1, 0, 0, 0])
                data.qpos[adr+3:adr+7] = q
                break
        mujoco.mj_forward(model, data)

        # UR5e setup
        robot = UR5e()
        robot.set_base(mj_utils.get_body_pose(model, data, "ur5e_base").t)
        robot_q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.set_joint(robot_q_init)

        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for i, jn in enumerate(joint_names):
            mj_utils.set_joint_q(model, data, jn, robot_q_init[i])
        mujoco.mj_forward(model, data)

        mj_utils.attach(model, data, "attach", "2f85", robot.fkine(robot_q_init))
        tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        robot.set_tool(tool)

        for _ in range(500):
            data.ctrl[:] = np.zeros(model.nu)
            mujoco.mj_step(model, data)

        # Optional viewer for the virtual scene
        viewer = None
        if enable_viewer:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0.6, 0.25, 0.2]
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.2
            viewer.sync()

        return VirtualScene(model=model, data=data, robot=robot,
                            joint_names=joint_names, apple_body_id=apple_body_id,
                            viewer=viewer)

    def plan_recovery_in_virtual(self, virtual_scene, recovery_pos) -> tuple[bool, list[dict]]:
        """Run full recovery flow inside a virtual scene and record plan steps.

        Parameters
        ----------
        virtual_scene : VirtualScene — created by build_virtual_scene()
        recovery_pos : (3,) array — apple position to target

        Returns
        -------
        (success, steps) where steps is a list of dicts (the plan)
        """
        scene = virtual_scene
        steps = []

        # Determine T_wo/T_pregrasp at recovery_pos
        # (same rotation as original experiment: Rz(-π/2) @ Ry(π/2))
        R = (sm.SE3.Rz(-np.pi/2).R @ sm.SE3.Ry(np.pi/2).R) @ np.eye(3)
        T_wo = sm.SE3.Trans(recovery_pos) * sm.SE3(sm.SO3(R, check=False))
        T_pre = sm.SE3.Trans(recovery_pos + np.array([0, 0, 0.127])) * sm.SE3(sm.SO3(R, check=False))
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

        apple_z_before = float(scene.data.body(scene.apple_body_id).xpos[2])

        # Step 1: Joint move to q1
        self._v_move_joints(scene, q1, 1.0)
        steps.append({"type": "joint_move", "target": q1.tolist(), "duration": 1.0})

        # Step 2: Snap + reset gripper + reattach
        self._v_snap(scene, reset_gripper=True)
        flange_pose = mj_utils.get_body_pose(scene.model, scene.data, "flange")
        mj_utils.attach(scene.model, scene.data, "attach", "2f85", flange_pose)
        self._v_steps(scene, 500)
        steps.append({"type": "gripper", "command": "open"})

        # Step 3: Cartesian to pregrasp
        scene.robot.set_joint(q1)
        self._v_cartesian(scene, T_pre, 1.0)
        steps.append({"type": "cartesian_move", "target_pos": T_pre.t.tolist(),
                       "target_rot": T_pre.R.tolist(), "duration": 1.0, "label": "pregrasp"})

        # Step 4: Cartesian to grasp
        self._v_cartesian(scene, T_wo, 1.0)
        self._v_steps(scene, 500)
        steps.append({"type": "cartesian_move", "target_pos": T_wo.t.tolist(),
                       "target_rot": T_wo.R.tolist(), "duration": 1.0, "label": "grasp"})

        # Step 5: Snap
        self._v_snap(scene)

        # Step 6: Close gripper
        self._v_gripper_close(scene)
        steps.append({"type": "gripper", "command": "close"})

        # Step 7: Lift
        T_lift = sm.SE3.Trans(0, 0, 0.3) * T_wo
        self._v_cartesian(scene, T_lift, 1.0)
        steps.append({"type": "cartesian_move", "target_pos": T_lift.t.tolist(),
                       "target_rot": T_lift.R.tolist(), "duration": 1.0, "label": "lift"})

        apple_z_after = float(scene.data.body(scene.apple_body_id).xpos[2])
        success = (apple_z_after - apple_z_before) > 0.03
        return success, steps

    # ── Virtual scene motion helpers (mirror ExperimentV4) ───────────

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
            scene.data.ctrl[:6] = interp
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
            scene.data.ctrl[:6] = scene.action[:6]
            mujoco.mj_step(scene.model, scene.data)
            if scene.viewer:
                scene.viewer.sync()

    @staticmethod
    def _v_gripper_close(scene):
        for _ in range(1000):
            scene.action[-1] += 0.2
            scene.action[-1] = np.min([scene.action[-1], 255])
            scene.data.ctrl[:] = np.zeros(scene.model.nu)
            scene.data.ctrl[:6] = scene.action[:6]
            scene.data.ctrl[-1] = scene.action[-1]
            mujoco.mj_step(scene.model, scene.data)
            if scene.viewer:
                scene.viewer.sync()

    @staticmethod
    def _v_snap(scene, reset_gripper=False):
        for i, jn in enumerate(scene.joint_names):
            mj_utils.set_joint_q(scene.model, scene.data, jn, scene.robot.get_joint()[i])
        scene.action[:6] = scene.robot.get_joint()
        mujoco.mj_forward(scene.model, scene.data)
        flange_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "flange")
        for j in range(scene.model.njnt):
            name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and '2f85' in name.lower():
                adr = scene.model.jnt_qposadr[j]
                scene.data.qpos[adr:adr+3] = scene.data.body(flange_id).xpos.copy()
                scene.data.qpos[adr+3:adr+7] = scene.data.body(flange_id).xquat.copy()
                break
        if reset_gripper:
            for jn in ['right_driver_joint', 'left_driver_joint']:
                jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    scene.data.qpos[scene.model.jnt_qposadr[jid]] = 0.0
            scene.action[-1] = 0
        mujoco.mj_forward(scene.model, scene.data)
        if scene.viewer:
            scene.viewer.sync()

    @staticmethod
    def _v_steps(scene, n):
        for _ in range(n):
            scene.data.ctrl[:] = np.zeros(scene.model.nu)
            scene.data.ctrl[:6] = scene.action[:6]
            mujoco.mj_step(scene.model, scene.data)
            if scene.viewer:
                scene.viewer.sync()


class VirtualScene:
    """Handle returned by SimWrapper.build_virtual_scene()."""
    def __init__(self, model, data, robot, joint_names, apple_body_id, viewer=None):
        self.model = model
        self.data = data
        self.robot = robot
        self.joint_names = joint_names
        self.apple_body_id = apple_body_id
        self.viewer = viewer
        self.action = np.zeros(7)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
