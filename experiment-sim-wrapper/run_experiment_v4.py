#!/usr/bin/env python3
"""
抓取异常实验 v5 — 双层架构：真实仿真 + 感知层。

核心改进（v4 → v5）:
  - 异常检测不再使用 MuJoCo 真值，改为通过 YOLO+SAM2+点云 的感知管线
  - 条件 A (sim_wrapper): 从感知数据生成虚拟仿真 → 规划恢复 → 迁移回真实仿真
  - 条件 B (direct): 基于感知位置直接在真实仿真中恢复
  - 恢复方案序列化为 JSON，可回放到真机

流程:
  1. 加载场景，附接夹爪
  2. 关节运动到 q1
  3. 笛卡尔到预抓取位姿
  4. 笛卡尔到抓取位姿
  5. 闭合夹爪
  6. 笛卡尔提起
  7. 注入异常（grasp_miss）
  8. 感知管线检测异常
  9. 恢复抓取（条件 A/B）
  10. 记录恢复方案 + 结果

使用方式:
  conda run -n mujoco1 python run_experiment_v4.py [--no-viewer] [--no-inject] [--save-plan plan.json]
"""
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

import anomaly_injectors
from sim_wrapper import SimWrapper
from perception_pipeline import PerceptionPipeline, PerceivedScene

ROOT = Path(__file__).parent
# 使用根目录的 XML（与 v4 完全一致）
SCENE_XML = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml'
SIM_TIMESTEP = 0.002


class ExperimentV4:
    """基于 v4 控制方式的实验编排器."""

    def __init__(self, enable_viewer=True, condition="direct", noise_scale=0.02, save_plan=None, anomaly_type="grasp_miss"):
        # ── 渐发式异常 per-step 回调 (必须在任何 _step() 调用前初始化) ──
        self._anomaly_step_callback = None
        self._anomaly_step_state = None

        # ── 加载 MuJoCo 场景（与 v4 的 UR5GraspEnv.reset 一致）──
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # ── 条件配置 ──
        self.condition = condition
        self.noise_scale = noise_scale
        self.anomaly_type = anomaly_type
        self.sim_wrapper = SimWrapper(noise_scale) if condition == "sim_wrapper" else None
        self.save_plan_path = save_plan
        self.recovery_plan = None  # will hold serializable recovery plan dict

        # ── UR5e 初始化 ──
        self.robot = UR5e()
        self.robot.set_base(mj_utils.get_body_pose(self.model, self.data, "ur5e_base").t)
        # 初始关节 = [0,0,0,0,0,0]（与 v4 一致）
        self.robot_q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.robot.set_joint(self.robot_q_init)

        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for i, jn in enumerate(self.joint_names):
            mj_utils.set_joint_q(self.model, self.data, jn, self.robot_q_init[i])
        mujoco.mj_forward(self.model, self.data)

        # 附接夹爪
        mj_utils.attach(self.model, self.data, "attach", "2f85",
                        self.robot.fkine(self.robot_q_init))
        # 工具变换（与 v4 一致）
        tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(tool)

        # 控制 buffer
        self.action = np.zeros(7)

        # 物体信息
        self.apple_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple0")
        self.apple_initial_pos = self.data.body(self.apple_body_id).xpos.copy()
        self.apple_initial_quat = self.data.body(self.apple_body_id).xquat.copy()
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

        # ── Viewer ──
        self.viewer = None
        if enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.6, 0.25, 0.2]
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -25
            self.viewer.cam.distance = 1.2
            self.viewer.sync()

        # 稳定仿真
        for _ in range(500):
            self._step()

        # ── 感知管线（YOLO+SAM2+点云，替代 MuJoCo 真值）──
        self.perception = PerceptionPipeline(self.model, self.data)

        # ── 抓取位姿（从 v4 的 T_wo 获取）──
        self.T_wo = None
        self.T_pregrasp = None
        self._load_grasp_pose()

        # 指标
        self.metrics = self._init_metrics()

    def _load_grasp_pose(self):
        """加载 v4 保存的 T_wo 或使用默认位姿。"""
        try:
            t_v4 = np.load('/tmp/t_wo_translation.npy')
            R_v4 = np.load('/tmp/t_wo_rotation.npy')
            self.T_wo = sm.SE3.Trans(t_v4) * sm.SE3(sm.SO3(R_v4, check=False))
            self.T_pregrasp = sm.SE3.Trans(t_v4 + np.array([0, 0, 0.127])) * sm.SE3(sm.SO3(R_v4, check=False))
        except (FileNotFoundError, ValueError):
            # Fallback: 使用已知 apple 位置 + RPY(0, pi/2, -pi/2) 旋转
            pos = self.apple_initial_pos
            R = sm.SE3.Rz(-np.pi/2).R @ sm.SE3.Ry(np.pi/2).R
            self.T_wo = sm.SE3.Trans(pos) * sm.SE3(sm.SO3(R, check=False))
            self.T_pregrasp = sm.SE3.Trans(pos + np.array([0, 0, 0.12])) * sm.SE3(sm.SO3(R, check=False))

    def _init_metrics(self):
        return {
            "anomaly_type": self.anomaly_type,
            "condition": self.condition,
            "noise_scale": self.noise_scale,
            "anomaly_detected": False,
            "detection_method": "",
            "recovery_success": False,
            "apple_z_before_lift": 0.0,          # ground truth (for reference only)
            "apple_z_after_lift": 0.0,
            "apple_z_after_inject": 0.0,
            "apple_z_after_recovery": 0.0,
            "perceived_z_before_lift": None,     # perception-based
            "perceived_z_after_inject": None,
            "perceived_position": None,          # [x, y, z] perceived after injection
            "observed_pos": None,
            "contact_after_close": None,
            "contact_after_lift": None,
            "time_costs": {},
            "plan_saved": None,                  # path to saved recovery plan
        }

    # ── 步进 ──

    def _step(self):
        self.data.ctrl[:] = self.action
        mujoco.mj_step(self.model, self.data)
        if self._anomaly_step_callback:
            self._anomaly_step_callback(self, self._anomaly_step_state)
        if self.viewer:
            self.viewer.sync()

    def _step_n(self, n):
        for _ in range(n):
            self._step()

    # ── 轨迹执行（与 v4 一致）──

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

    # ── 夹爪控制 ──

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
        """Force MuJoCo qpos/ctrl to match robot internal joints (eliminate steady-state error).
        Also fix the 2f85 free joint qpos to match the flange (消除约束求解器累积漂移).
        If reset_gripper=True, also reset gripper driver qpos and action to 0."""
        for i, jn in enumerate(self.joint_names):
            mj_utils.set_joint_q(self.model, self.data, jn, self.robot.get_joint()[i])
        self.action[:6] = self.robot.get_joint()
        mujoco.mj_forward(self.model, self.data)
        # Snap 2f85 free joint qpos to match flange (消除 snap 后 flange 位置跳变导致的偏差)
        flange_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flange")
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and '2f85' in name.lower():
                adr = self.model.jnt_qposadr[j]
                self.data.qpos[adr:adr+3] = self.data.body(flange_id).xpos.copy()
                self.data.qpos[adr+3:adr+7] = self.data.body(flange_id).xquat.copy()
                break
        if reset_gripper:
            # Reset gripper driver qpos to 0 (fully open)
            for jn in ['right_driver_joint', 'left_driver_joint']:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0
            self.action[-1] = 0
        mujoco.mj_forward(self.model, self.data)

    # ── 恢复方案记录 ──

    def _init_recovery_plan(self):
        """Initialize a new recovery plan dict."""
        import datetime
        return {
            "plan_id": f"recovery_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
            "condition": self.condition,
            "anomaly_type": "grasp_miss",
            "detection_info": {},
            "steps": [],
            "result": {},
        }

    def _record_plan_step(self, step_type, **kwargs):
        if self.recovery_plan is None:
            return
        step = {"type": step_type}
        step.update(kwargs)
        self.recovery_plan["steps"].append(step)

    def _save_recovery_plan(self):
        if self.recovery_plan is None or self.save_plan_path is None:
            return None
        self.recovery_plan["result"] = {
            "success": bool(self.metrics.get("recovery_success", False)),
            "apple_z_after_recovery": self.metrics.get("apple_z_after_recovery", 0.0),
        }
        plan_path = Path(self.save_plan_path)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w") as f:
            json.dump(self.recovery_plan, f, indent=2, default=str)
        print(f"\n恢复方案已保存到: {plan_path}")
        return str(plan_path)

    # ── 接触检测 ──

    def _contact_summary(self):
        contacts = {"left_contact": False, "right_contact": False}
        for ic in range(self.data.ncon):
            c = self.data.contact[ic]
            g1 = self.model.geom(c.geom1).name if c.geom1 < self.model.ngeom else ""
            g2 = self.model.geom(c.geom2).name if c.geom2 < self.model.ngeom else ""
            if 'pad' in g1.lower() and 'apple' in g2.lower():
                if 'left' in g1.lower():
                    contacts["left_contact"] = True
                if 'right' in g1.lower():
                    contacts["right_contact"] = True
            if 'pad' in g2.lower() and 'apple' in g1.lower():
                if 'left' in g2.lower():
                    contacts["left_contact"] = True
                if 'right' in g2.lower():
                    contacts["right_contact"] = True
        return contacts

    # ── 核心实验流程 ──

    def run(self, inject_anomaly=True):
        """运行完整的抓取 → 异常注入 → 恢复实验。"""
        t0 = time.time()
        anomaly_type = self.anomaly_type
        print("=" * 50)
        print(f"实验开始（v4 控制方式，异常类型: {anomaly_type}）")
        print("=" * 50)

        # ── Step 1: 关节运动到 q1（与 v4 的 extute_pre 前半段一致）──
        print("\n[1/9] 关节运动到 q1...")
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._move_joints(q1, 1.0)
        self._report_state("after q1")

        # ── Step 2: 笛卡尔到预抓取位姿（与 v4 的 extute_pre 后半段一致）──
        print("\n[2/9] 移动到预抓取位姿...")
        self.robot.set_joint(q1)
        self._move_cartesian(self.T_pregrasp, 1.0)
        self._report_state("pregrasp")

        # ── Step 3: 笛卡尔到抓取位姿（与 v4 的 exeute_grasp 一致）──
        print("\n[3/9] 移动到抓取位姿...")
        if inject_anomaly and anomaly_type == "object_displaced":
            T_grasp_mid = sm.SE3.Trans(self.T_wo.t + np.array([0, 0, 0.04])) * sm.SE3(sm.SO3(self.T_wo.R, check=False))
            self._move_cartesian(T_grasp_mid, 0.4)
            anomaly_injectors.inject_object_displaced(self.model, self.data, "apple0", dx=0.06, dy=0.04, dz=0.0)
            print("  苹果被推离原位！")
            self._step_n(100)
            self._move_cartesian(self.T_wo, 0.6)
        elif inject_anomaly and anomaly_type == "collision":
            T_grasp_mid = sm.SE3.Trans(self.T_wo.t + np.array([0, 0, 0.04])) * sm.SE3(sm.SO3(self.T_wo.R, check=False))
            self._move_cartesian(T_grasp_mid, 0.4)
            anomaly_injectors.inject_object_displaced(self.model, self.data, "apple0", dx=0.06, dy=-0.04, dz=0.0)
            anomaly_injectors.inject_collision(self.model, self.data, "apple0", vx=0.6, vy=-0.4, vz=1.5)
            print("  夹爪碰到苹果，苹果被弹飞！")
            self._step_n(300)
            self._move_cartesian(self.T_wo, 0.6)
        else:
            self._move_cartesian(self.T_wo, 1.0)
        self._step_n(500)
        self._report_state("grasp")
        pinch = self.data.site_xpos[self.pinch_site_id]
        print(f"    joints={np.round(self.robot.get_joint(), 3).tolist()} pinch=({pinch[0]:.4f},{pinch[1]:.4f},{pinch[2]:.4f})")

        apple_before_lift = self.data.body(self.apple_body_id).xpos[2]
        self.metrics["apple_z_before_lift"] = float(apple_before_lift)

        # ── 提起前感知基线（苹果未被夹爪遮挡时获取）──
        self._perceive_before_lift()

        # ── Step 4: 同步 qpos 后闭合夹爪 ──
        print("\n[4/9] 同步关节位置 + 闭合夹爪...")
        self._snap_to_robot_joints()
        actual_q = [self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in self.joint_names]
        right_driver = self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")]
        left_driver = self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")]
        pinch = self.data.site_xpos[self.pinch_site_id]
        flange_pos = self.data.body(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flange")).xpos
        gripper_base_pos = self.data.body(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "2f85_base")).xpos
        attach_free_q = None
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and '2f85' in name.lower():
                adr = self.model.jnt_qposadr[j]
                attach_free_q = self.data.qpos[adr:adr+3].copy()
                break
        print(f"  [init-snap] joints={np.round(actual_q, 3).tolist()} gripper=({right_driver:.4f},{left_driver:.4f})")
        print(f"  [init-snap] pinch=({pinch[0]:.4f},{pinch[1]:.4f},{pinch[2]:.4f}) flange=({flange_pos[0]:.4f},{flange_pos[1]:.4f},{flange_pos[2]:.4f}) 2f85_base=({gripper_base_pos[0]:.4f},{gripper_base_pos[1]:.4f},{gripper_base_pos[2]:.4f})")
        if attach_free_q is not None:
            print(f"  [init-snap] 2f85_free_z={attach_free_q[2]:.4f}")
        self._gripper_close()
        if inject_anomaly and anomaly_type == "gripper_fail":
            anomaly_injectors.inject_gripper_fail(self.model, self.data)
            self.action[-1] = 0
            self.data.ctrl[-1] = 0
            print("  夹爪故障，未能闭合！")
        self._report_state("close")
        self.metrics["contact_after_close"] = self._contact_summary()

        # ── Step 5: 提起 ──
        print("\n[5/9] 提起物体...")
        T_lift = sm.SE3.Trans(0, 0, 0.3) * self.T_wo
        if inject_anomaly and anomaly_type == "incipient_slip":
            slip_state = anomaly_injectors.setup_incipient_slip(
                self.model, self.data, self.apple_body_id)
            total_steps = int(1.0 / SIM_TIMESTEP)
            def _incipient_slip_cb(exp, state):
                anomaly_injectors.apply_incipient_slip_step(
                    exp.model, exp.data, exp.apple_body_id,
                    _incipient_slip_cb.step, total_steps, slip_state)
                _incipient_slip_cb.step += 1
            _incipient_slip_cb.step = 0
            self._anomaly_step_callback = _incipient_slip_cb
            self._anomaly_step_state = slip_state
            print("  渐发式滑移已激活 (per-step drift)...")
            self._move_cartesian(T_lift, 1.0)
            self._anomaly_step_callback = None
            self._anomaly_step_state = None
        elif inject_anomaly and anomaly_type == "slip":
            T_lift_mid = sm.SE3.Trans(0, 0, 0.12) * self.T_wo
            self._move_cartesian(T_lift_mid, 0.4)
            anomaly_injectors.inject_slip(
                self.model, self.data,
                self.apple_body_id,
                self.apple_initial_pos, self.apple_initial_quat,
            )
            print("  苹果从夹爪滑落！")
            self._move_cartesian(T_lift, 0.6)
        else:
            self._move_cartesian(T_lift, 1.0)
        self._report_state("lift")
        self.metrics["contact_after_lift"] = self._contact_summary()
        self.metrics["apple_z_after_lift"] = float(self.data.body(self.apple_body_id).xpos[2])

        # ── Step 6: 注入异常（仅 grasp_miss 在此注入；slip 已在 Step 5 中途注入）──
        print(f"\n[6/9] 注入异常 ({anomaly_type})...")
        if inject_anomaly and anomaly_type == "grasp_miss":
            anomaly_injectors.inject_grasp_miss(
                self.model, self.data,
                self.apple_body_id,
                self.apple_initial_pos, self.apple_initial_quat,
            )
            self.metrics["apple_z_after_inject"] = float(self.data.body(self.apple_body_id).xpos[2])
            print(f"  注入后 apple Z: {self.metrics['apple_z_after_inject']:.4f}")
        else:
            self.metrics["apple_z_after_inject"] = float(self.data.body(self.apple_body_id).xpos[2])
            if anomaly_type != "grasp_miss":
                print(f"  (异常已在之前步骤注入)")
        self._step_n(50)

        # ── Step 7: 检测异常 ──
        print("\n[7/9] 检测异常...")
        self._detect_anomaly()

        t1 = time.time()
        self.metrics["time_costs"]["pre_recovery"] = round(t1 - t0, 3)

        # ── Step 8: 恢复 ──
        if self.metrics["anomaly_detected"]:
            print("\n[8/9] 执行恢复 (re-grasp)...")
            self._execute_recovery()
        else:
            print("\n[8/9] 未检测到异常，跳过恢复。")

        # ── Step 9: 放置苹果到盘子上 → 回初始位置 ──
        print("\n[9/9] 放置苹果到盘子 + 回到初始位置...")
        self._report_state("before_home")
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        plate_pos = self.data.body(plate_id).xpos.copy()
        R = self.T_wo.R
        T_above_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.15])) * sm.SE3(sm.SO3(R, check=False))
        self._move_cartesian(T_above_plate, 1.5)
        T_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.08])) * sm.SE3(sm.SO3(R, check=False))
        self._move_cartesian(T_plate, 1.0)
        self._gripper_open()         # 张开夹爪，苹果落到盘子上
        self._step_n(1000)           # 给重力时间让苹果落稳
        self._move_joints(q1, 1.0)   # 回初始关节位置
        self._report_state("home")

        self.metrics["time_costs"]["total"] = round(time.time() - t0, 3)

        self._print_metrics()
        return self.metrics

    def _report_state(self, label):
        apple = self.data.body(self.apple_body_id).xpos
        print(f"  [{label}] apple=({apple[0]:.4f},{apple[1]:.4f},{apple[2]:.4f})")

    def _detect_anomaly(self):
        """基于感知管线检测异常（不用 MuJoCo 真值）."""
        t0 = time.time()
        print("  正在通过 YOLO+SAM2+点云 感知管线检测物体位置...")
        scene = self.perception.detect(target_class="apple", work_dir=f"/tmp/perception_trial_{int(time.time()*1000)%100000}")
        t_perception = time.time() - t0
        print(f"  感知耗时: {t_perception:.1f}s")
        self.metrics.setdefault("time_costs", {})
        self.metrics["time_costs"]["detection_perception"] = round(t_perception, 2)

        if not scene.detection_ok:
            # 感知失败 — 回退到 MuJoCo 真值（仿真专用安全网）。
            # 在真机上，需要额外的传感器（力/接触）来区分"提起成功"和"感知失败"。
            apple_gt_z = self.metrics.get("apple_z_after_inject", 0.0)
            baseline_gt_z = self.metrics.get("apple_z_before_lift", 0.0)
            gt_z_change = apple_gt_z - baseline_gt_z
            print(f"  → 感知管线未检测到物体。回退真值: dz={gt_z_change:.4f}m")
            if gt_z_change < 0.03:
                self.metrics["anomaly_detected"] = True
                self.metrics["detection_method"] = "perception_failed"
                print("  → 真值显示苹果未被提起，判定为异常。")
            else:
                self.metrics["anomaly_detected"] = False
                self.metrics["detection_method"] = "perception_failed"
                print("  → 真值显示苹果已被提起，判定为无异常。")
            return

        perceived_z = float(scene.apple_pos[2])
        self.metrics["perceived_z_after_inject"] = perceived_z
        self.metrics["perceived_position"] = scene.apple_pos.tolist()
        z_before = self.metrics["perceived_z_before_lift"]
        z_change = perceived_z - z_before if z_before is not None else (perceived_z - 0.046)

        print(f"  感知 apple Z: {perceived_z:.4f}m (变化: {z_change:.4f}m, 阈值: 0.03m)")

        if z_change < 0.03:
            self.metrics["anomaly_detected"] = True
            self.metrics["detection_method"] = "perception_z_check"
            print("  → 检测到抓空异常！物体未被提起。")
        else:
            self.metrics["anomaly_detected"] = False
            self.metrics["detection_method"] = "perception_z_check"
            print("  → 未检测到异常。")

    def _perceive_before_lift(self):
        """在提起前获取感知 Z 高度（作为异常检测基线）."""
        print("  获取提起前感知基线...")
        scene = self.perception.detect(target_class="apple", work_dir=f"/tmp/perception_prelift_{int(time.time()*1000)%100000}")
        if scene.detection_ok:
            self.metrics["perceived_z_before_lift"] = float(scene.apple_pos[2])
            print(f"  感知基线 Z: {scene.apple_pos[2]:.4f}m")
        else:
            print("  [WARN] 感知基线获取失败，将使用地面高度作为参考。")

    def _execute_recovery(self):
        """执行恢复 — 根据条件分支选择策略。"""
        t0 = time.time()
        self.recovery_plan = self._init_recovery_plan()

        # 使用感知位置（在 _detect_anomaly 中获取），而非 data.body.xpos
        perceived = self.metrics.get("perceived_position")
        if perceived is not None:
            recovery_pos = np.asarray(perceived, dtype=np.float64)
            print(f"  感知位置用于恢复: ({recovery_pos[0]:.4f}, {recovery_pos[1]:.4f}, {recovery_pos[2]:.4f})")
        else:
            recovery_pos = self.data.body(self.apple_body_id).xpos.copy()
            print(f"  [WARN] 无感知数据，回退到真值: ({recovery_pos[0]:.4f}, {recovery_pos[1]:.4f}, {recovery_pos[2]:.4f})")

        self.metrics["observed_pos"] = recovery_pos.tolist()

        if self.condition == "sim_wrapper" and self.sim_wrapper is not None:
            # ── Condition A: 虚拟仿真 → 规划恢复 → 迁移 ──
            self._condition_a_recovery(recovery_pos)
        else:
            # ── Condition B: 直接恢复（基于感知位置）──
            self._direct_recovery(recovery_pos)

        self.metrics["apple_z_after_recovery"] = float(self.data.body(self.apple_body_id).xpos[2])
        z_change = self.metrics["apple_z_after_recovery"] - recovery_pos[2]
        self.metrics["recovery_success"] = z_change > 0.03
        self.metrics["time_costs"]["recovery"] = round(time.time() - t0, 3)

        # 将检测信息写入方案
        self.recovery_plan["detection_info"] = {
            "perceived_apple_z": self.metrics.get("perceived_z_after_inject"),
            "z_threshold": 0.03,
            "detection_method": self.metrics.get("detection_method"),
        }

        # 保存恢复方案
        saved = self._save_recovery_plan()
        if saved:
            self.metrics["plan_saved"] = saved

    # ── Condition A: 虚拟仿真迁移 ────────────────────────────────────

    def _condition_a_recovery(self, recovery_pos):
        """在虚拟仿真中规划恢复，迁移到真实仿真执行。

        流程:
          1. 从感知位置构建虚拟场景（apple 在 perceived pos）
          2. 在虚拟场景中规划完整恢复流程 → 得到方案步骤
          3. 方案步骤在 Layer 1 真实仿真中回放执行
        """
        enable_viewer = self.viewer is not None
        print(f"\n  ── [Condition A] 构建虚拟仿真场景 (viewer={enable_viewer}) ──")
        virtual_scene = self.sim_wrapper.build_virtual_scene(
            recovery_pos, enable_viewer=enable_viewer
        )

        try:
            print("  ── [Condition A] 在虚拟仿真中规划恢复 ──")
            success, plan_steps = self.sim_wrapper.plan_recovery_in_virtual(
                virtual_scene, recovery_pos
            )

            if success:
                print(f"  ── [Condition A] 虚拟规划成功 ({len(plan_steps)} 步)，迁移执行 ──")
                self.recovery_plan["steps"] = plan_steps
                self._execute_plan_steps(plan_steps)
            else:
                print("  ── [Condition A] 虚拟规划失败，回退到直接恢复 ──")
                self._direct_recovery(recovery_pos)
        finally:
            virtual_scene.close()

    def _execute_plan_steps(self, steps):
        """在 Layer 1 真实仿真中逐步骤执行恢复方案。

        将 plan 步骤 dict 转换为真实的 MuJoCo 运动命令。
        """
        for i, step in enumerate(steps):
            stype = step.get("type", "unknown")
            label = step.get("label", stype)
            print(f"    执行步骤 [{i+1}/{len(steps)}] {label}")

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

            elif stype == "cartesian_move":
                pos = np.asarray(step["target_pos"], dtype=np.float64)
                rot = np.asarray(step["target_rot"], dtype=np.float64)
                duration = step.get("duration", 1.0)
                T_target = sm.SE3.Trans(pos) * sm.SE3(sm.SO3(rot, check=False))
                self._move_cartesian(T_target, duration)

            else:
                print(f"    [WARN] 未知步骤类型: {stype}")

    # ── Condition B: 直接恢复 ───────────────────────────────────────

    def _direct_recovery(self, recovery_pos):
        """基于感知位置直接在真实仿真中恢复（Condition B）。"""
        # 调整抓取位姿到感知位置
        R = self.T_wo.R
        T_wo = sm.SE3.Trans(recovery_pos) * sm.SE3(sm.SO3(R, check=False))
        T_pregrasp = sm.SE3.Trans(recovery_pos + np.array([0, 0, 0.127])) * sm.SE3(sm.SO3(R, check=False))

        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._move_joints(q1, 1.0)
        self._record_plan_step("joint_move", target=q1.tolist(), duration=1.0)

        self._snap_to_robot_joints(reset_gripper=True)
        self._record_plan_step("gripper", command="open")

        flange_pose = mj_utils.get_body_pose(self.model, self.data, "flange")
        mj_utils.attach(self.model, self.data, "attach", "2f85", flange_pose)
        self._step_n(500)

        self.robot.set_joint(q1)
        self._move_cartesian(T_pregrasp, 1.0)
        self._record_plan_step("cartesian_move", target_pos=T_pregrasp.t.tolist(),
                               target_rot=T_pregrasp.R.tolist(), duration=1.0, label="pregrasp")

        self._move_cartesian(T_wo, 1.0)
        self._step_n(500)
        self._record_plan_step("cartesian_move", target_pos=T_wo.t.tolist(),
                               target_rot=T_wo.R.tolist(), duration=1.0, label="grasp")

        self._snap_to_robot_joints()
        self._gripper_close()
        self._record_plan_step("gripper", command="close")

        T_lift = sm.SE3.Trans(0, 0, 0.3) * T_wo
        self._move_cartesian(T_lift, 1.0)
        self._record_plan_step("cartesian_move", target_pos=T_lift.t.tolist(),
                               target_rot=T_lift.R.tolist(), duration=1.0, label="lift")

    def _print_metrics(self):
        print("\n" + "=" * 50)
        print("实验结果:")
        for k, v in self.metrics.items():
            print(f"  {k}: {v}")
        print("=" * 50)

    def close(self):
        if self.viewer:
            self.viewer.close()
        self.perception.close()


def _convert_numpy(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _convert_numpy(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_convert_numpy(x) for x in v]
    return v


def _aggregate_results(all_metrics):
    n = len(all_metrics)
    n_anomaly = sum(1 for m in all_metrics if m.get("anomaly_detected", False))
    n_recovery = sum(1 for m in all_metrics if m.get("recovery_success", False))
    recovery_times = [m["time_costs"]["recovery"] for m in all_metrics if "recovery" in m.get("time_costs", {})]

    return _convert_numpy({
        "n_trials": n,
        "anomaly_detection_rate": n_anomaly / n if n > 0 else 0.0,
        "recovery_success_rate": n_recovery / n if n > 0 else 0.0,
        "time_costs": {
            "pre_recovery_mean": float(np.mean([m["time_costs"]["pre_recovery"] for m in all_metrics])),
            "pre_recovery_std": float(np.std([m["time_costs"]["pre_recovery"] for m in all_metrics])),
            "recovery_mean": float(np.mean(recovery_times)) if recovery_times else None,
            "recovery_std": float(np.std(recovery_times)) if recovery_times else None,
            "total_mean": float(np.mean([m["time_costs"]["total"] for m in all_metrics])),
            "total_std": float(np.std([m["time_costs"]["total"] for m in all_metrics])),
        },
        "per_trial": all_metrics,
    })


def main():
    parser = argparse.ArgumentParser(description="基于 v4 的抓取异常实验")
    parser.add_argument("--no-viewer", action="store_true", help="不启动 viewer")
    parser.add_argument("--no-inject", action="store_true", help="不注入异常（对照组）")
    parser.add_argument("--save", type=str, default="results/result_v4.json", help="结果保存路径")
    parser.add_argument("--trials", type=int, default=1, help="重复实验次数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--perturb", type=float, default=0.01, help="每次试验的位置随机扰动幅度 (m)")
    parser.add_argument("--condition", type=str, default="direct",
                        choices=["direct", "sim_wrapper"],
                        help="恢复策略: direct=直接恢复, sim_wrapper=影子仿真验证后恢复")
    parser.add_argument("--noise-scale", type=float, default=0.0,
                        help="感知噪声幅度 (m)")
    parser.add_argument("--save-plan", type=str, default=None,
                        help="保存恢复方案到指定路径 (JSON)")
    parser.add_argument("--anomaly", type=str, default="grasp_miss",
                        choices=["grasp_miss", "slip", "incipient_slip", "object_displaced", "gripper_fail", "collision"],
                        help="异常类型: grasp_miss=提起后抓空, slip=提起中途滑落, incipient_slip=渐发式滑移, object_displaced=物体被推离, gripper_fail=夹爪故障, collision=碰撞推飞")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    all_metrics = []
    for trial in range(1, args.trials + 1):
        print(f"\n{'#' * 50}")
        print(f"# Trial {trial} / {args.trials}")
        print(f"{'#' * 50}")

        exp = ExperimentV4(
            enable_viewer=not args.no_viewer,
            condition=args.condition,
            noise_scale=args.noise_scale,
            save_plan=args.save_plan,
            anomaly_type=args.anomaly,
        )
        if args.trials > 1:
            perturb = np.random.uniform(-args.perturb, args.perturb, 3)
            exp.apple_initial_pos += perturb
            exp.T_wo = sm.SE3.Trans(exp.apple_initial_pos) * sm.SE3(sm.SO3(exp.T_wo.R, check=False))
            exp.T_pregrasp = sm.SE3.Trans(exp.apple_initial_pos + np.array([0, 0, 0.12])) * sm.SE3(sm.SO3(exp.T_wo.R, check=False))
        try:
            metrics = exp.run(inject_anomaly=not args.no_inject)
        finally:
            exp.close()
        all_metrics.append(metrics)

    if args.trials > 1:
        aggregated = _aggregate_results(all_metrics)
        print(f"\n{'=' * 50}")
        print(f"聚合结果 ({args.trials} trials):")
        print(f"  异常检测率: {aggregated['anomaly_detection_rate']:.1%}")
        print(f"  恢复成功率: {aggregated['recovery_success_rate']:.1%}")
        print("=" * 50)
        save_data = aggregated
    else:
        save_data = _convert_numpy(all_metrics[0])

    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n结果已保存到: {save_path}")


if __name__ == "__main__":
    main()
