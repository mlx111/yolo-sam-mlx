#!/usr/bin/env python3
"""
Recovery Plan Replayer — 加载恢复方案 JSON，在新 MuJoCo 实例中逐步骤执行。

用于验证仿真中生成的恢复方案是否可以回放（仿真回放和真机回放的前提）。

使用方式:
  conda run -n mujoco1 python replay_plan.py --plan /tmp/recovery_plan_test.json
  conda run -n mujoco1 python replay_plan.py --plan plan.json --no-viewer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import spatialmath as sm

ROOT_PROJECT = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp'
sys.path.insert(0, ROOT_PROJECT)
sys.path.insert(0, '/home/mlx/mujoco/YOLO_World-SAM-GraspNet')
sys.path.insert(0, '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/experiment-sim-wrapper')

from arm.robot import UR5e
from arm.motion_planning import (
    JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner,
    LinePositionParameter, OneAttitudeParameter, CartesianParameter,
)
from utils import mj as mj_utils

ROOT = Path(__file__).parent
SCENE_XML = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml'
SIM_TIMESTEP = 0.002


class PlanReplayer:
    """Load and execute a recovery plan in a MuJoCo simulation."""

    def __init__(self, plan_path: str, enable_viewer: bool = True, perturb_position: tuple = None):
        self.plan_path = Path(plan_path)
        with open(self.plan_path) as f:
            self.plan = json.load(f)

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.robot = UR5e()
        self.robot.set_base(mj_utils.get_body_pose(self.model, self.data, "ur5e_base").t)
        self.robot_q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.robot.set_joint(self.robot_q_init)

        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for i, jn in enumerate(self.joint_names):
            mj_utils.set_joint_q(self.model, self.data, jn, self.robot_q_init[i])
        mujoco.mj_forward(self.model, self.data)

        mj_utils.attach(self.model, self.data, "attach", "2f85",
                        self.robot.fkine(self.robot_q_init))
        tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(tool)

        self.action = np.zeros(7)
        self.apple_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple0")

        self.viewer = None
        if enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.6, 0.25, 0.2]
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -25
            self.viewer.cam.distance = 1.2
            self.viewer.sync()

        for _ in range(500):
            self._step()

        self.apple_initial_z = float(self.data.body(self.apple_body_id).xpos[2])
        print(f"Replayer 初始化完成。apple 初始 Z: {self.apple_initial_z:.4f}")

        if perturb_position is not None:
            self._apply_position_perturb(perturb_position)

    # ── MuJoCo helpers ──

    def _step(self):
        self.data.ctrl[:] = self.action
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def _step_n(self, n):
        for _ in range(n):
            self._step()

    def _apply_position_perturb(self, offset):
        """Move the apple to simulate a different scene (for robustness testing)."""
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == self.apple_body_id:
                adr = self.model.jnt_qposadr[j]
                self.data.qpos[adr:adr+3] += offset
                break
        mujoco.mj_forward(self.model, self.data)

    # ── Motion primitives ──

    def _move_joints(self, q_target, duration=1.0):
        q0 = self.robot.get_joint()
        param = JointParameter(q0, np.asarray(q_target))
        vel_param = QuinticVelocityParameter(duration)
        traj_param = TrajectoryParameter(param, vel_param)
        planner = TrajectoryPlanner(traj_param)
        for t in np.linspace(0, duration, int(duration / SIM_TIMESTEP)):
            interp = planner.interpolate(t)
            self.robot.move_joint(interp)
            self.action[:6] = interp
            self._step()

    def _move_cartesian(self, T_target, duration=1.0):
        T_current = self.robot.get_cartesian()
        pos_param = LinePositionParameter(T_current.t, T_target.t)
        att_param = OneAttitudeParameter(sm.SO3(T_current.R), sm.SO3(T_target.R))
        cart_param = CartesianParameter(pos_param, att_param)
        vel_param = QuinticVelocityParameter(duration)
        traj_param = TrajectoryParameter(cart_param, vel_param)
        planner = TrajectoryPlanner(traj_param)
        for t in np.linspace(0, duration, int(duration / SIM_TIMESTEP)):
            interp = planner.interpolate(t)
            self.robot.move_cartesian(interp)
            self.action[:6] = self.robot.get_joint()
            self._step()

    def _gripper_open(self):
        for _ in range(1000):
            self.action[-1] -= 0.2
            self.action[-1] = np.max([self.action[-1], 0])
            self._step()

    def _gripper_close(self):
        for _ in range(1000):
            self.action[-1] += 0.2
            self.action[-1] = np.min([self.action[-1], 255])
            self._step()

    def _snap_to_robot_joints(self, reset_gripper=False):
        for i, jn in enumerate(self.joint_names):
            mj_utils.set_joint_q(self.model, self.data, jn, self.robot.get_joint()[i])
        self.action[:6] = self.robot.get_joint()
        mujoco.mj_forward(self.model, self.data)
        flange_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flange")
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and '2f85' in name.lower():
                adr = self.model.jnt_qposadr[j]
                self.data.qpos[adr:adr+3] = self.data.body(flange_id).xpos.copy()
                self.data.qpos[adr+3:adr+7] = self.data.body(flange_id).xquat.copy()
                break
        if reset_gripper:
            for jn in ['right_driver_joint', 'left_driver_joint']:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0
            self.action[-1] = 0
        mujoco.mj_forward(self.model, self.data)

    # ── Plan execution ──

    def execute(self) -> dict:
        """Execute the recorded recovery plan step by step."""
        steps = self.plan.get("steps", [])
        if not steps:
            print("计划中无步骤。")
            return {"success": False, "reason": "empty plan"}

        print(f"\n正在执行恢复方案: {self.plan.get('plan_id', '(unknown)')}")
        print(f"条件: {self.plan.get('condition', 'unknown')}")
        print(f"步骤数: {len(steps)}")

        for i, step in enumerate(steps):
            stype = step.get("type", "unknown")
            label = step.get("label", stype)
            print(f"\n  [{i+1}/{len(steps)}] {label} ({stype})")

            if stype == "joint_move":
                target = np.asarray(step["target"], dtype=np.float64)
                duration = step.get("duration", 1.0)
                self._move_joints(target, duration)

            elif stype == "gripper":
                cmd = step.get("command", "")
                if cmd == "open":
                    self._snap_to_robot_joints(reset_gripper=True)
                    flange_pose = mj_utils.get_body_pose(self.model, self.data, "flange")
                    mj_utils.attach(self.model, self.data, "attach", "2f85", flange_pose)
                    self._step_n(500)
                elif cmd == "close":
                    self._snap_to_robot_joints()
                    self._gripper_close()
                else:
                    print(f"    未知夹爪命令: {cmd}")

            elif stype == "cartesian_move":
                pos = np.asarray(step["target_pos"], dtype=np.float64)
                rot = np.asarray(step["target_rot"], dtype=np.float64)
                duration = step.get("duration", 1.0)
                T_target = sm.SE3.Trans(pos) * sm.SE3(sm.SO3(rot, check=False))
                self._move_cartesian(T_target, duration)

            else:
                print(f"    未知步骤类型: {stype}")

        # Verify
        apple_z = float(self.data.body(self.apple_body_id).xpos[2])
        z_change = apple_z - self.apple_initial_z
        success = z_change > 0.03
        print(f"\n回放完成: apple Z = {apple_z:.4f}m (dz = {z_change:.4f}m)")
        print(f"结果: {'成功' if success else '失败'}")

        return {
            "success": success,
            "apple_z_after_replay": apple_z,
            "z_change": z_change,
            "plan_id": self.plan.get("plan_id"),
        }

    def close(self):
        if self.viewer:
            self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description="恢复方案回放器")
    parser.add_argument("--plan", type=str, required=True, help="恢复方案 JSON 文件路径")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--perturb", type=float, nargs=3, default=None,
                        help="苹果位置扰动 [x y z] (米)，用于测试方案鲁棒性")
    args = parser.parse_args()

    replayer = PlanReplayer(
        plan_path=args.plan,
        enable_viewer=not args.no_viewer,
        perturb_position=tuple(args.perturb) if args.perturb else None,
    )
    try:
        result = replayer.execute()
        print(f"\n回放结果: {json.dumps(result, indent=2, default=str)}")
    finally:
        replayer.close()


if __name__ == "__main__":
    main()
