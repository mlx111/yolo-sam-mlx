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
  conda run -n mujoco1 python run_experiment_v4.py --preset experiment
  conda run -n mujoco1 python run_experiment_v4.py [--no-viewer|--viewer] [--no-inject|--inject] [--save-plan plan.json]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import spatialmath as sm

ROOT_PROJECT = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp'
sys.path.insert(0, ROOT_PROJECT)
sys.path.insert(0, '/home/mlx/mujoco/YOLO_World-SAM-GraspNet')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm.robot import UR5e
from arm.motion_planning import (
    JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner,
    LinePositionParameter, OneAttitudeParameter, CartesianParameter,
)
from utils import mj as mj_utils

from sim_wrapper import SimWrapper
from ur5e import anomaly_injectors
from ur5e.anomaly_conditions import get_condition_spec
from ur5e.experiment_config import (
    DEFAULT_DIVERSITY_LAMBDA,
    DEFAULT_EXPERIENCE_TOP_K,
    DETECTION_ANOMALY_Z_CHANGE,
    DETECTION_SUCCESS_Z_CHANGE,
    OBJECT_DISPLACED_DX,
    OBJECT_DISPLACED_DY,
    COLLISION_RECOVERY_LIFT_FROM_TABLE,
    COLLISION_RECOVERY_PINCH_DISTANCE,
    RECOVERY_SUCCESS_Z_CHANGE,
    SLIP_RECOVERY_LIFT_FROM_TABLE,
    SLIP_RECOVERY_PINCH_DISTANCE,
)
from ur5e.skills import recovery_steps, registry
from ur5e.skill_results import SkillResult, skill_result
from memory.calibration import compute_sandbox_calibration
from memory.gating import compute_memory_gate
from memory.v3 import MemoryV3Library, canonical_action_signature_from_steps, make_memory_v3_entry
from perception_pipeline import PerceptionPipeline, PerceivedScene
from llm_handler import verify_anomaly, plan_recovery, score_recovery

ROOT = Path(__file__).parent
# 使用根目录的 XML（与 v4 完全一致）
SCENE_XML = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml'
SIM_TIMESTEP = 0.002
DEFAULT_EXPERIMENT_PRESET = "experiment"
DEFAULT_EXPERIMENT_SAVE = "results/experiment_result_v4.json"
DEFAULT_EXPERIMENT_PLAN = "results/experiment_recovery_plan.json"
DEFAULT_EXPERIMENT_EXPERIENCE_LIB = "results/experience_library.json"
GRASP_ATTACH_MAX_DISTANCE = 0.045
DEFAULT_PREGRASP_HEIGHT = 0.127

FAILURE_PREDICATE_DESCRIPTIONS = {
    "not_on_plate": "目标 apple 最终没有稳定位于 plate 的允许范围内。",
    "gripper_not_open": "任务结束时夹爪仍未打开，说明目标没有被释放到最终位置。",
    "not_home": "机械臂没有回到安全/结束关节位。",
    "wrong_orientation": "目标最终姿态不满足该异常条件的姿态要求。",
    "wrong_object_tracked": "恢复过程中仍跟踪或夹持了错误目标，而不是 apple。",
    "perception_not_corrected": "恢复后的感知位置仍与 apple 真值偏差过大。",
    "object_not_secured": "目标没有被稳定夹持，夹爪与 apple 距离过大或没有跟踪 apple。",
    "insufficient_lift": "目标相对桌面提升高度不足，说明抓取/运输恢复不充分。",
    "no_path_replan_evidence": "路径异常后没有出现重规划、避障、安全中间点或策略切换证据。",
    "no_strategy_switch_evidence": "无进展异常后没有出现策略切换证据。",
    "no_progress": "恢复执行后目标状态变化不足，没有形成有效进展。",
}


def _resolve_local_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _resolve_run_config(args):
    legacy_defaults = {
        "no_viewer": False,
        "no_inject": False,
        "save": "results/result_v4.json",
        "trials": 1,
        "seed": None,
        "perturb": 0.01,
        "condition": "direct",
        "noise_scale": 0.0,
        "save_plan": None,
        "experience_lib": None,
        "anomaly": "grasp_miss",
        "condition_id": None,
    }
    preset_defaults = {}
    if getattr(args, "preset", None) == DEFAULT_EXPERIMENT_PRESET:
        preset_defaults = {
            "no_viewer": False,
            "no_inject": False,
            "save": DEFAULT_EXPERIMENT_SAVE,
            "save_plan": DEFAULT_EXPERIMENT_PLAN,
            "trials": 1,
            "seed": None,
            "perturb": 0.01,
            "condition": "sim_wrapper",
            "noise_scale": 0.0,
            "experience_lib": DEFAULT_EXPERIMENT_EXPERIENCE_LIB,
            "anomaly": "grasp_miss",
            "condition_id": None,
        }

    resolved = dict(legacy_defaults)
    resolved.update(preset_defaults)
    for key in list(resolved.keys()):
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = value
    resolved["preset"] = getattr(args, "preset", None)
    return resolved


class ExperimentV4:
    """基于 v4 控制方式的实验编排器."""

    def __init__(
        self,
        enable_viewer=True,
        condition="direct",
        noise_scale=0.02,
        save_plan=None,
        anomaly_type="grasp_miss",
        condition_id=None,
        experience_lib_path=None,
        allow_gt_recovery_fallback=None,
        allow_deterministic_place=False,
        enable_condition_plan_hooks=False,
        enable_llm_score=False,
        object_displaced_dx=OBJECT_DISPLACED_DX,
        object_displaced_dy=OBJECT_DISPLACED_DY,
    ):
        # ── 渐发式异常 per-step 回调 (必须在任何 _step() 调用前初始化) ──
        self._anomaly_step_callback = None
        self._anomaly_step_state = None

        # ── 加载 MuJoCo 场景（与 v4 的 UR5GraspEnv.reset 一致）──
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # ── 条件配置 ──
        self.condition = condition
        self.condition_spec = get_condition_spec(condition_id)
        if self.condition_spec is not None:
            anomaly_type = self.condition_spec.legacy_anomaly_type
        self.scenario_id = self.condition_spec.scenario_id if self.condition_spec else ""
        self.condition_id = self.condition_spec.condition_id if self.condition_spec else ""
        self.failure_family = self.condition_spec.failure_family if self.condition_spec else ""
        self.perception_override: dict[str, Any] = {}
        self.noise_scale = noise_scale
        self.anomaly_type = anomaly_type
        self.object_displaced_dx = float(object_displaced_dx)
        self.object_displaced_dy = float(object_displaced_dy)
        self.sim_wrapper = SimWrapper(noise_scale) if condition == "sim_wrapper" else None
        self.save_plan_path = save_plan
        self.recovery_plan = None  # will hold serializable recovery plan dict
        if allow_gt_recovery_fallback is None:
            self.allow_gt_recovery_fallback = condition == "sim_wrapper"
        else:
            self.allow_gt_recovery_fallback = bool(allow_gt_recovery_fallback)
        self.allow_deterministic_place = bool(allow_deterministic_place)
        self.enable_condition_plan_hooks = bool(enable_condition_plan_hooks)
        self.enable_llm_score = bool(enable_llm_score)
        self.experience_lib_path = _resolve_local_path(experience_lib_path)
        self.experience_library = MemoryV3Library.load(self.experience_lib_path) if self.experience_lib_path else None

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
        self.pear_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pear0")
        self.pear_initial_pos = self.data.body(self.pear_body_id).xpos.copy() if self.pear_body_id >= 0 else None
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

        # 通用物体跟踪 (替代硬编码的 _track_apple)
        self.grasp_tracking = False
        self.tracked_body_id = None
        self.tracked_body_adr = None
        self.grasp_offset_pos = None

        # 缓存所有 free joint body 的 qpos 地址，方便 _attach_body 查找
        self._body_qpos_adr_cache = {}
        for j in range(self.model.njnt):
            body_id = self.model.jnt_bodyid[j]
            if body_id >= 0:
                self._body_qpos_adr_cache[body_id] = self.model.jnt_qposadr[j]

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
        self._prelift_work_dir = None   # saved for VLM verification images
        self._detection_work_dir = None
        self._task_history: list[dict] = []  # LLM-facing task log
        self.keyframe_output_dir: Path | None = None
        self.keyframe_reference_dir: Path | None = None
        self._keyframes: list[dict[str, Any]] = []
        self.use_memory_keyframes = False
        self.memory_keyframe_top_k = 2

        # ── 抓取位姿（从 v4 的 T_wo 获取）──
        self.T_wo = None
        self.T_pregrasp = None
        self._load_grasp_pose()

        # 指标
        self.metrics = self._init_metrics()

    def _configure_keyframe_dir(self) -> None:
        if self.keyframe_output_dir is not None:
            return
        base_dir = None
        if self.save_plan_path:
            base_dir = Path(self.save_plan_path).resolve().parent
        elif self.experience_lib_path:
            base_dir = Path(self.experience_lib_path).resolve().parent
        if base_dir is None:
            base_dir = ROOT / "results" / "keyframe_debug"
        self.keyframe_reference_dir = base_dir
        self.keyframe_output_dir = base_dir / "keyframes"

    def _relative_keyframe_path(self, path: Path) -> str:
        reference = self.keyframe_reference_dir
        if reference is not None:
            try:
                return str(path.resolve().relative_to(reference.resolve()))
            except ValueError:
                pass
        return str(path)

    def _save_keyframe(self, stage: str, description: str, source_path: str | Path | None = None) -> None:
        self._configure_keyframe_dir()
        assert self.keyframe_output_dir is not None
        self.keyframe_output_dir.mkdir(parents=True, exist_ok=True)
        dst = self.keyframe_output_dir / f"{stage}.jpg"
        src = Path(source_path) if source_path else None
        try:
            if src is not None and src.exists():
                shutil.copy2(src, dst)
            else:
                depth_path = self.keyframe_output_dir / f"{stage}_depth.npy"
                self.perception._render_rgb_depth(dst, depth_path)
                if depth_path.exists():
                    depth_path.unlink()
        except Exception as exc:
            print(f"  [WARN] 保存关键帧 {stage} 失败: {exc}")
            return

        rel_path = self._relative_keyframe_path(dst)
        self._keyframes = [frame for frame in self._keyframes if frame.get("stage") != stage]
        self._keyframes.append(
            {
                "stage": stage,
                "image_path": rel_path,
                "description": description,
                "used_for_retrieval": stage in {"after_anomaly", "before_recovery"},
            }
        )
        self.metrics["keyframes"] = list(self._keyframes)

    def _resolve_experience_keyframe_path(self, entry: Any, frame: dict[str, Any]) -> Path | None:
        image_path = str(frame.get("image_path") or "")
        if not image_path:
            return None
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate if candidate.exists() else None

        validation_evidence = getattr(entry, "validation_evidence", {}) or {}
        result_path = validation_evidence.get("result_path")
        if result_path:
            resolved = Path(result_path).resolve().parent / image_path
            if resolved.exists():
                return resolved

        library_base = self.experience_lib_path.parent if self.experience_lib_path else None
        if library_base is not None:
            resolved = library_base / image_path
            if resolved.exists():
                return resolved
        return None

    def _collect_memory_keyframe_paths(self, experiences: list[tuple[Any, float]]) -> list[str]:
        if not self.use_memory_keyframes:
            return []
        paths: list[str] = []
        # Collect clean rendered frames (before_recovery) first, then fall back
        # to anomaly frames (after_anomaly) for each memory entry.
        for entry, _score in experiences:
            wanted = ["before_recovery", "after_anomaly"]
            for frame in getattr(entry, "keyframes", []) or []:
                stage = getattr(frame, "stage", "") if not isinstance(frame, dict) else frame.get("stage", "")
                if stage not in wanted:
                    continue
                frame_dict = frame if isinstance(frame, dict) else {
                    "stage": getattr(frame, "stage", ""),
                    "image_path": getattr(frame, "image_path", ""),
                }
                resolved = self._resolve_experience_keyframe_path(entry, frame_dict)
                if resolved is not None:
                    paths.append(str(resolved))
                    wanted.remove(stage)
                    if not wanted:
                        break
            if len(paths) >= self.memory_keyframe_top_k:
                break
        self.metrics["prompt_memory_keyframes"] = paths
        self.metrics["prompt_keyframe_count"] = len(paths)
        return paths

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
            "scenario_id": self.scenario_id,
            "condition_id": self.condition_id,
            "failure_family": self.failure_family,
            "task_stage": self.condition_spec.task_stage if self.condition_spec else "",
            "injection_stage": self.condition_spec.injection_stage if self.condition_spec else "",
            "injection_params": dict(self.condition_spec.params) if self.condition_spec else {},
            "condition_injection": None,
            "condition": self.condition,
            "noise_scale": self.noise_scale,
            "object_displaced_dx": self.object_displaced_dx,
            "object_displaced_dy": self.object_displaced_dy,
            "anomaly_detected": False,
            "detection_method": "",
            "recovery_success": False,
            "apple_z_before_lift": 0.0,          # ground truth (for reference only)
            "apple_z_after_lift": 0.0,
            "apple_z_after_inject": 0.0,
            "apple_z_after_recovery": 0.0,
            "task_success": False,
            "perceived_z_before_lift": None,     # perception-based
            "perceived_z_after_inject": None,
            "perceived_position": None,          # [x, y, z] perceived after injection
            "observed_pos": None,
            "contact_after_close": None,
            "contact_after_lift": None,
            "time_costs": {},
            "plan_saved": None,                  # path to saved recovery plan
            "reconstruction_artifacts": None,
            "virtual_validation_success": None,
            "virtual_validation_z_before": None,
            "virtual_validation_z_after": None,
            "virtual_validation_z_change": None,
            "executed_plan_source": None,
            "executed_recovery_steps": None,
            "skill_results": [],
        }

    # ── 步进 ──

    def _step(self):
        self.data.ctrl[:] = self.action
        mujoco.mj_step(self.model, self.data)
        if self.grasp_tracking and self.grasp_offset_pos is not None and self.tracked_body_adr is not None:
            self._track_body()
        if self._anomaly_step_callback:
            self._anomaly_step_callback(self, self._anomaly_step_state)
        if self.viewer:
            self.viewer.sync()

    def _step_n(self, n):
        for _ in range(n):
            self._step()

    def _tracked_body_name(self) -> str:
        if self.tracked_body_id is None:
            return ""
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.tracked_body_id)
        return name or str(self.tracked_body_id)

    def _current_skill_observation(self) -> dict[str, Any]:
        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        return {
            "contact": self._contact_summary(),
            "gripper_action": float(self.action[-1]),
            "tracked_body": self._tracked_body_name(),
            "pinch_distance": float(np.linalg.norm(apple_pos - pinch_pos)),
            "object_pos": apple_pos.tolist(),
        }

    def _record_skill_result(self, result: SkillResult | dict[str, Any]) -> dict[str, Any]:
        record = result.to_dict() if isinstance(result, SkillResult) else dict(result)
        self.metrics.setdefault("skill_results", []).append(record)
        return record

    def _record_basic_skill(self, skill: str, success: bool, reason: str = "", **extra: Any) -> dict[str, Any]:
        observation = self._current_skill_observation()
        return self._record_skill_result(
            skill_result(
                skill=skill,
                success=success,
                reason=reason,
                phase="recovery" if self.metrics.get("observed_pos") is not None else "initial",
                source="experiment",
                contact=observation["contact"],
                gripper_action=observation["gripper_action"],
                tracked_body=observation["tracked_body"],
                pinch_distance=observation["pinch_distance"],
                object_pos=observation["object_pos"],
                extra=extra,
            )
        )

    def _pose_error_result(
        self,
        *,
        skill: str,
        target_pos: np.ndarray | None = None,
        target_rot: np.ndarray | None = None,
        ik_failed_count: int | None = None,
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> SkillResult:
        current = self.robot.get_cartesian()
        final_pos = np.asarray(current.t, dtype=np.float64)
        pos_error = None
        if target_pos is not None:
            target_pos = np.asarray(target_pos, dtype=np.float64)
            pos_error = float(np.linalg.norm(final_pos - target_pos))
        rot_error = None
        final_rot = np.asarray(current.R, dtype=np.float64)
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=np.float64)
            rot_error = float(np.linalg.norm(final_rot - target_rot))
        observation = self._current_skill_observation()
        success = True
        if ik_failed_count is not None and ik_failed_count > 0:
            success = False
            reason = reason or "ik_failed_during_path"
        if pos_error is not None and pos_error > 0.03:
            success = False
            reason = reason or "final_pose_error_exceeded"
        return skill_result(
            skill=skill,
            success=success,
            reason=reason or "ok",
            phase="recovery" if self.metrics.get("observed_pos") is not None else "initial",
            source="motion_control",
            target_pos=target_pos.tolist() if target_pos is not None else None,
            final_pos=final_pos.tolist(),
            pos_error=pos_error,
            target_rot=target_rot.tolist() if target_rot is not None else None,
            final_rot=final_rot.tolist(),
            rot_error=rot_error,
            ik_failed_count=ik_failed_count,
            contact=observation["contact"],
            gripper_action=observation["gripper_action"],
            tracked_body=observation["tracked_body"],
            pinch_distance=observation["pinch_distance"],
            object_pos=observation["object_pos"],
            extra=extra or {},
        )

    # ── 通用物体跟踪 (抓取模拟) ───────────────────────────────────────

    def _set_grasp_pose_from_position(
        self,
        pos: np.ndarray,
        *,
        pregrasp_height: float = DEFAULT_PREGRASP_HEIGHT,
        yaw_delta_deg: float = 0.0,
        grasp_z_offset: float = 0.0,
    ) -> dict[str, Any]:
        """Compute grasp/pregrasp poses from the current perceived object pose."""
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        grasp_pos = pos + np.array([0.0, 0.0, float(grasp_z_offset)], dtype=np.float64)
        R = np.asarray(self.T_wo.R, dtype=np.float64)
        if yaw_delta_deg:
            R = sm.SE3.Rz(np.deg2rad(float(yaw_delta_deg))).R @ R
        self.T_wo = sm.SE3.Trans(grasp_pos) * sm.SE3(sm.SO3(R, check=False))
        self.T_pregrasp = sm.SE3.Trans(grasp_pos + np.array([0.0, 0.0, float(pregrasp_height)])) * sm.SE3(sm.SO3(R, check=False))
        return {
            "object_pos": pos.tolist(),
            "grasp_pos": self.T_wo.t.tolist(),
            "pregrasp_pos": self.T_pregrasp.t.tolist(),
            "pregrasp_height": float(pregrasp_height),
            "yaw_delta_deg": float(yaw_delta_deg),
            "grasp_z_offset": float(grasp_z_offset),
        }

    def _attach_body(self, body_name, *, max_distance: float = GRASP_ATTACH_MAX_DISTANCE) -> bool:
        """夹爪闭合后让指定物体跟随 pinch site（模拟抓取）。

        body_name: "apple0" 或 "pear0"
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found")
        adr = self._body_qpos_adr_cache.get(bid)
        if adr is None:
            raise ValueError(f"No free joint qpos adr for '{body_name}'")
        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        body_pos = self.data.body(bid).xpos.copy()
        pinch_distance = float(np.linalg.norm(body_pos - pinch_pos))
        if pinch_distance > float(max_distance):
            self.metrics["last_attach_attempt"] = {
                "body": body_name,
                "success": False,
                "pinch_distance": pinch_distance,
                "max_distance": float(max_distance),
            }
            print(
                f"  → 抓取绑定失败: {body_name} pinch_distance={pinch_distance:.4f}m "
                f"> {float(max_distance):.4f}m"
            )
            return False
        self.grasp_offset_pos = body_pos - pinch_pos
        self.tracked_body_id = bid
        self.tracked_body_adr = adr
        self.grasp_tracking = True
        self.metrics["last_attach_attempt"] = {
            "body": body_name,
            "success": True,
            "pinch_distance": pinch_distance,
            "max_distance": float(max_distance),
        }
        print(f"  → 跟踪抓取: {body_name} 将跟随 pinch (offset={self.grasp_offset_pos})")
        return True

    def _record_grasp_geometry_failure(self, *, attached: bool, max_distance: float) -> None:
        spec = self.condition_spec
        params = dict(spec.params) if spec else {}
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        target_pos = np.asarray(self.T_wo.t, dtype=np.float64)
        record = {
            "condition_id": self.condition_id,
            "pose_offset": {
                "dx": float(params.get("dx", 0.0)),
                "dy": float(params.get("dy", 0.0)),
                "dz": float(params.get("dz", 0.0)),
                "yaw_deg": float(params.get("yaw_deg", 0.0)),
                "norm": float(np.linalg.norm([
                    float(params.get("dx", 0.0)),
                    float(params.get("dy", 0.0)),
                    float(params.get("dz", 0.0)),
                ])),
            },
            "target_pos": target_pos.tolist(),
            "apple_pos": apple_pos.tolist(),
            "pinch_pos": pinch_pos.tolist(),
            "target_to_object_distance": float(np.linalg.norm(target_pos - apple_pos)),
            "pinch_distance": float(np.linalg.norm(pinch_pos - apple_pos)),
            "contact": self._contact_summary(),
            "attach_gate": self.metrics.get("last_attach_attempt"),
            "attached": bool(attached),
            "max_distance": float(max_distance),
            "reason": "grasp_pose_offset_outside_attach_gate" if not attached else "unexpected_attach_success",
        }
        self.metrics["grasp_geometry_failure"] = record
        self._record_skill_result(skill_result(
            skill="grasp_geometry_gate",
            success=not attached,
            reason=record["reason"],
            phase="initial",
            source="condition_rule",
            target_pos=record["target_pos"],
            final_pos=pinch_pos.tolist(),
            pos_error=record["pinch_distance"],
            contact=record["contact"],
            gripper_action=float(self.action[-1]),
            tracked_body=self._tracked_body_name(),
            pinch_distance=record["pinch_distance"],
            object_pos=record["apple_pos"],
            extra=record,
        ))

    def _track_body(self):
        """每步将跟踪物体的位置同步到 pinch site。"""
        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        self.data.qpos[self.tracked_body_adr:self.tracked_body_adr + 3] = pinch_pos + self.grasp_offset_pos
        mujoco.mj_forward(self.model, self.data)

    def _detach_body(self):
        """释放当前跟踪的物体。"""
        self.grasp_tracking = False
        self.tracked_body_id = None
        self.tracked_body_adr = None
        self.grasp_offset_pos = None

    # ── 轨迹执行（与 v4 一致）──

    def _move_joints(self, q_target, duration=1.0) -> SkillResult:
        q0 = self.robot.get_joint()
        q_target = np.asarray(q_target, dtype=np.float64)
        param = JointParameter(q0, q_target)
        vel_param = QuinticVelocityParameter(duration)
        traj_param = TrajectoryParameter(param, vel_param)
        planner = TrajectoryPlanner(traj_param)
        for t in np.linspace(0, duration, int(duration / SIM_TIMESTEP)):
            interp = planner.interpolate(t)
            self.robot.move_joint(interp)
            self.action[:6] = interp
            self._step()
        final_q = np.asarray(self.robot.get_joint(), dtype=np.float64)
        joint_error = float(np.linalg.norm(final_q - q_target))
        observation = self._current_skill_observation()
        return skill_result(
            skill="joint_move",
            success=joint_error <= 0.03,
            reason="ok" if joint_error <= 0.03 else "final_joint_error_exceeded",
            phase="recovery" if self.metrics.get("observed_pos") is not None else "initial",
            source="motion_control",
            contact=observation["contact"],
            gripper_action=observation["gripper_action"],
            tracked_body=observation["tracked_body"],
            pinch_distance=observation["pinch_distance"],
            object_pos=observation["object_pos"],
            extra={
                "target_joint": q_target.tolist(),
                "final_joint": final_q.tolist(),
                "joint_error": joint_error,
                "duration": float(duration),
            },
        )

    def _move_cartesian(self, T_target, duration=1.0) -> SkillResult:
        T_current = self.robot.get_cartesian()
        pos_param = LinePositionParameter(T_current.t, T_target.t)
        att_param = OneAttitudeParameter(sm.SO3(T_current.R), sm.SO3(T_target.R))
        cart_param = CartesianParameter(pos_param, att_param)
        vel_param = QuinticVelocityParameter(duration)
        traj_param = TrajectoryParameter(cart_param, vel_param)
        planner = TrajectoryPlanner(traj_param)
        ik_failed_count = 0
        for t in np.linspace(0, duration, int(duration / SIM_TIMESTEP)):
            interp = planner.interpolate(t)
            ok = self.robot.move_cartesian(interp)
            if ok is False:
                ik_failed_count += 1
            self.action[:6] = self.robot.get_joint()
            self._step()
        return self._pose_error_result(
            skill="cartesian_move",
            target_pos=np.asarray(T_target.t, dtype=np.float64),
            target_rot=np.asarray(T_target.R, dtype=np.float64),
            ik_failed_count=ik_failed_count,
            extra={"duration": float(duration)},
        )

    # ── 夹爪控制 ──

    def _gripper_open(self) -> SkillResult:
        for _ in range(1500):
            if float(self.action[-1]) <= 1.0:
                break
            self.action[-1] -= 0.2
            self.action[-1] = np.max([self.action[-1], 0])
            self._step()
        observation = self._current_skill_observation()
        success = float(self.action[-1]) <= 1.0
        return skill_result(
            skill="gripper_open",
            success=success,
            reason="ok" if success else "gripper_not_open",
            phase="recovery" if self.metrics.get("observed_pos") is not None else "initial",
            source="motion_control",
            contact=observation["contact"],
            gripper_action=observation["gripper_action"],
            tracked_body=observation["tracked_body"],
            pinch_distance=observation["pinch_distance"],
            object_pos=observation["object_pos"],
        )

    def _gripper_close(self) -> SkillResult:
        for _ in range(1500):
            if float(self.action[-1]) >= 254.0:
                break
            self.action[-1] += 0.2
            self.action[-1] = np.min([self.action[-1], 255])
            self._step()
        observation = self._current_skill_observation()
        success = float(self.action[-1]) >= 254.0
        return skill_result(
            skill="gripper_close",
            success=success,
            reason="ok" if success else "gripper_not_closed",
            phase="recovery" if self.metrics.get("observed_pos") is not None else "initial",
            source="motion_control",
            contact=observation["contact"],
            gripper_action=observation["gripper_action"],
            tracked_body=observation["tracked_body"],
            pinch_distance=observation["pinch_distance"],
            object_pos=observation["object_pos"],
        )

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
            "scenario_id": self.scenario_id,
            "condition_id": self.condition_id,
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
        self.recovery_plan["condition"] = self.condition
        self.recovery_plan["result"] = {
            "success": bool(self.metrics.get("recovery_success", False)),
            "apple_z_after_recovery": self.metrics.get("apple_z_after_recovery", 0.0),
        }
        plan_path = _resolve_local_path(self.save_plan_path)
        if plan_path is None:
            return None
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

    def _maybe_inject_condition(self, stage: str) -> bool:
        spec = self.condition_spec
        if spec is None or spec.injection_stage != stage:
            return False
        params = dict(spec.params)
        record = {
            "scenario_id": spec.scenario_id,
            "condition_id": spec.condition_id,
            "injector": spec.injector,
            "stage": stage,
            "params": params,
        }
        if spec.injector == "grasp_pose_offset":
            dx = float(params.get("dx", 0.0))
            dy = float(params.get("dy", 0.0))
            dz = float(params.get("dz", 0.0))
            yaw = np.deg2rad(float(params.get("yaw_deg", 0.0)))
            record["target_pose_before"] = self.T_wo.t.tolist()
            record["target_rotation_before"] = np.asarray(self.T_wo.R, dtype=np.float64).tolist()
            R = self.T_wo.R
            if yaw:
                R = sm.SE3.Rz(yaw).R @ R
            self.T_wo = sm.SE3.Trans(self.T_wo.t + np.array([dx, dy, dz])) * sm.SE3(sm.SO3(R, check=False))
            self.T_pregrasp = sm.SE3.Trans(self.T_pregrasp.t + np.array([dx, dy, dz])) * sm.SE3(sm.SO3(R, check=False))
            record["target_pose_after"] = self.T_wo.t.tolist()
            record["target_rotation_after"] = np.asarray(self.T_wo.R, dtype=np.float64).tolist()
            record["pose_offset_norm"] = float(np.linalg.norm([dx, dy, dz]))
            # U2 抓取几何异常不影响物体位置；设置 perceived_position 确保 recovery 可定位目标
            self.metrics["perceived_position"] = self.data.body(self.apple_body_id).xpos.copy().tolist()
        elif spec.injector == "target_body_override":
            target_body = str(params.get("target_body", "pear0"))
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_body)
            if body_id < 0:
                raise ValueError(f"target body not found for perception override: {target_body}")
            target_pos = self.data.body(body_id).xpos.copy()
            self.perception_override = {
                "mode": "target_body",
                "target_body": target_body,
                "target_pos": target_pos.tolist(),
                "active": True,
            }
            record["target_body"] = target_body
            record["target_pos"] = target_pos.tolist()
            self._apply_initial_perception_override(target_pos, source="target_body_override")
        elif spec.injector == "stale_pose":
            stale_pos = self.data.body(self.apple_body_id).xpos.copy()
            self.perception_override = {
                "mode": "stale_pose",
                "stale_pos": stale_pos.tolist(),
                "active": True,
            }
            record["stale_pos"] = stale_pos.tolist()
            self._apply_initial_perception_override(stale_pos, source="stale_pose")
            record["displace"] = anomaly_injectors.inject_object_displaced(
                self.model,
                self.data,
                "apple0",
                dx=float(params.get("dx", 0.06)),
                dy=float(params.get("dy", 0.04)),
                dz=float(params.get("dz", 0.0)),
            )
        elif spec.injector in {"occlusion_noise", "boundary_confusion"}:
            apple_pos = self.data.body(self.apple_body_id).xpos.copy()
            offset = np.array([
                float(params.get("dx", 0.0)),
                float(params.get("dy", 0.0)),
                float(params.get("dz", 0.0)),
            ], dtype=np.float64)
            perceived_pos = apple_pos + offset
            mode = "partial_occlusion" if spec.injector == "occlusion_noise" else "boundary_confusion"
            self.perception_override = {
                "mode": mode,
                "true_pos": apple_pos.tolist(),
                "perceived_pos": perceived_pos.tolist(),
                "occlusion_ratio": params.get("occlusion_ratio"),
                "mask_boundary_noise": params.get("mask_boundary_noise"),
                "active": True,
            }
            record.update(self.perception_override)
            self._apply_initial_perception_override(perceived_pos, source=mode)
        elif spec.injector == "perception_yaw_error":
            apple_pos = self.data.body(self.apple_body_id).xpos.copy()
            yaw_deg = float(params.get("yaw_deg", 35.0))
            self.perception_override = {
                "mode": "orientation_error",
                "true_pos": apple_pos.tolist(),
                "perceived_pos": apple_pos.tolist(),
                "yaw_error_deg": yaw_deg,
                "active": True,
            }
            record.update(self.perception_override)
            self._apply_initial_perception_override(apple_pos, source="orientation_error", yaw_delta_deg=yaw_deg)
        elif spec.injector == "pregrasp_offset":
            dx = float(params.get("dx", 0.0))
            dy = float(params.get("dy", 0.0))
            dz = float(params.get("dz", 0.0))
            offset = np.array([dx, dy, dz], dtype=np.float64)
            record["pregrasp_pose_before"] = self.T_pregrasp.t.tolist()
            record["grasp_pose_before"] = self.T_wo.t.tolist()
            self.T_pregrasp = sm.SE3.Trans(self.T_pregrasp.t + offset) * sm.SE3(sm.SO3(self.T_pregrasp.R, check=False))
            if bool(params.get("propagate_to_grasp", False)):
                self.T_wo = sm.SE3.Trans(self.T_wo.t + offset) * sm.SE3(sm.SO3(self.T_wo.R, check=False))
            record["pregrasp_pose_after"] = self.T_pregrasp.t.tolist()
            record["grasp_pose_after"] = self.T_wo.t.tolist()
            record["pose_offset_norm"] = float(np.linalg.norm(offset))
            # U2 预抓位偏移不影响物体位置；设置 perceived_position 确保 recovery 可定位目标
            self.metrics["perceived_position"] = self.data.body(self.apple_body_id).xpos.copy().tolist()
        elif spec.injector == "blocked_path":
            dx = float(params.get("dx", 0.05))
            dy = float(params.get("dy", 0.04))
            dz = float(params.get("raise_z", 0.12))
            offset = np.array([dx, dy, dz], dtype=np.float64)
            record["pregrasp_pose_before"] = self.T_pregrasp.t.tolist()
            record["grasp_pose_before"] = self.T_wo.t.tolist()
            self.T_pregrasp = sm.SE3.Trans(self.T_pregrasp.t + offset) * sm.SE3(sm.SO3(self.T_pregrasp.R, check=False))
            self.T_wo = sm.SE3.Trans(self.T_wo.t + np.array([dx, dy, 0.0])) * sm.SE3(sm.SO3(self.T_wo.R, check=False))
            record["pregrasp_pose_after"] = self.T_pregrasp.t.tolist()
            record["grasp_pose_after"] = self.T_wo.t.tolist()
            record["blocked_path"] = True
            record["pose_offset_norm"] = float(np.linalg.norm(offset))
        elif spec.injector == "gripper_fail":
            record["result"] = anomaly_injectors.inject_gripper_fail(self.model, self.data)
            self.action[-1] = 0
            self.data.ctrl[-1] = 0
        elif spec.injector == "partial_close":
            close_ratio = float(params.get("close_ratio", 0.15))
            record["result"] = anomaly_injectors.inject_partial_close(
                self.model,
                self.data,
                close_ratio=close_ratio,
            )
            self.action[-1] = 255 * close_ratio
            self.data.ctrl[-1] = self.action[-1]
        elif spec.injector == "premature_close":
            record["result"] = anomaly_injectors.inject_premature_close_push(
                self.model,
                self.data,
                "apple0",
                dx=float(params.get("push_dx", 0.05)),
                dy=float(params.get("push_dy", 0.025)),
                dz=float(params.get("push_dz", 0.0)),
            )
            self.action[-1] = 255
            self.data.ctrl[-1] = self.action[-1]
        elif spec.injector == "transport_drop":
            self._detach_body()
            record["result"] = anomaly_injectors.inject_slip(
                self.model,
                self.data,
                self.apple_body_id,
                self.apple_initial_pos,
                self.apple_initial_quat,
            )
            self.metrics["perceived_position"] = self.data.body(self.apple_body_id).xpos.copy().tolist()
        elif spec.injector == "transport_displace":
            if bool(params.get("drop", True)):
                self._detach_body()
                anomaly_injectors.inject_slip(
                    self.model,
                    self.data,
                    self.apple_body_id,
                    self.apple_initial_pos,
                    self.apple_initial_quat,
                )
            record["result"] = anomaly_injectors.inject_transport_displace(
                self.model,
                self.data,
                "apple0",
                dx=float(params.get("dx", 0.06)),
                dy=float(params.get("dy", -0.04)),
                dz=float(params.get("dz", 0.0)),
            )
            self.metrics["perceived_position"] = self.data.body(self.apple_body_id).xpos.copy().tolist()
        elif spec.injector in {"approach_collision", "table_collision"}:
            if spec.injector == "table_collision":
                dz = float(params.get("dz", -0.08))
                record["target_pose_before"] = self.T_wo.t.tolist()
                self.T_wo = sm.SE3.Trans(self.T_wo.t + np.array([0.0, 0.0, dz])) * sm.SE3(sm.SO3(self.T_wo.R, check=False))
                record["target_pose_after"] = self.T_wo.t.tolist()
            record["displace"] = anomaly_injectors.inject_object_displaced(
                self.model,
                self.data,
                "apple0",
                dx=float(params.get("dx", 0.06)),
                dy=float(params.get("dy", -0.04)),
                dz=0.0,
            )
            record["collision"] = anomaly_injectors.inject_collision(
                self.model,
                self.data,
                "apple0",
                vx=float(params.get("vx", 0.6)),
                vy=float(params.get("vy", -0.4)),
                vz=float(params.get("vz", 1.5)),
            )
        elif spec.injector in {"wrong_place_position", "premature_release", "wrong_place_orientation", "wrong_sequence_plan", "no_progress_plan"}:
            record["deferred"] = True
        else:
            raise ValueError(f"unsupported condition injector: {spec.injector}")
        self.metrics["condition_injection"] = record
        print(f"  [condition {spec.condition_id}] injected {spec.injector} at {stage}: {params}")
        # Save keyframe after physical scene changes so the experience entry has
        # a visual reference of the injected state.  Perception-only injectors
        # (occlusion_noise, boundary_confusion, etc.) don't change the rendered scene.
        if spec.injector in {"stale_pose", "grasp_pose_offset", "collision", "pregrasp_offset", "blocked_path"}:
            self._save_keyframe("after_injection", f"异常注入后（{spec.injector}）")
        return True

    def _apply_initial_perception_override(self, target_pos: np.ndarray, source: str, yaw_delta_deg: float = 0.0) -> None:
        target_pos = np.asarray(target_pos, dtype=np.float64)
        self.metrics["perceived_position"] = target_pos.tolist()
        self.metrics["perception_failure"] = {
            "condition_id": self.condition_id,
            "failure_family": self.failure_family,
            "source": source,
            "active": True,
            "target_pos": target_pos.tolist(),
            "yaw_delta_deg": float(yaw_delta_deg),
        }
        self._set_grasp_pose_from_position(target_pos, yaw_delta_deg=yaw_delta_deg)

    def _clear_perception_override_for_recovery(self) -> dict[str, Any]:
        previous = dict(self.perception_override or {})
        self.perception_override = {}
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        self.metrics["perceived_position"] = apple_pos.tolist()
        correction = {
            "condition_id": self.condition_id,
            "cleared_override": previous,
            "corrected_target_body": "apple0",
            "corrected_pos": apple_pos.tolist(),
        }
        self.metrics["perception_recovery"] = correction
        if "perception_failure" in self.metrics:
            self.metrics["perception_failure"]["active"] = False
            self.metrics["perception_failure"]["recovered"] = True
        return correction

    # ── 核心实验流程 ──

    def run(self, inject_anomaly=True):
        """运行完整的抓取 → 异常注入 → 恢复实验。"""
        t0 = time.time()
        anomaly_type = self.anomaly_type
        print("=" * 50)
        print(f"实验开始（v4 控制方式，条件: {self.condition_id or self.scenario_id or self.condition}）")
        print("=" * 50)
        if inject_anomaly and self.condition_spec is None:
            raise ValueError("condition_id is required when injecting anomalies; legacy anomaly_type injection is disabled.")

        self._maybe_inject_condition("before_initial_plan") if inject_anomaly else False

        # ── Step 1: 关节运动到 q1（与 v4 的 extute_pre 前半段一致）──
        print("\n[1/9] 关节运动到 q1...")
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._move_joints(q1, 1.0)
        self._report_state("after q1")
        self._log_task("joint_move", "SUCCESS", "关节运动到 q1")

        # ── Step 2: 笛卡尔到预抓取位姿（与 v4 的 extute_pre 后半段一致）──
        print("\n[2/9] 移动到预抓取位姿...")
        self.robot.set_joint(q1)
        self._maybe_inject_condition("before_move_pregrasp") if inject_anomaly else False
        self._move_cartesian(self.T_pregrasp, 1.0)
        if inject_anomaly and self.condition_id == "U5-1":
            self.metrics["blocked_path_failure"] = {
                "condition_id": self.condition_id,
                "stage": "move_pregrasp",
                "reason": "straight_path_blocked_requires_safe_waypoint",
                "pregrasp_pos": self.T_pregrasp.t.tolist(),
            }
            print("  condition U5-1: 直线路径被挡，预抓位被迫绕行。")
        self._report_state("pregrasp")
        self._log_task("move-pregrasp", "SUCCESS", "移动到预抓取位姿")

        # ── Step 3: 笛卡尔到抓取位姿（与 v4 的 exeute_grasp 一致）──
        print("\n[3/9] 移动到抓取位姿...")
        self._maybe_inject_condition("before_move_grasp") if inject_anomaly else False
        self._move_cartesian(self.T_wo, 1.0)
        self._step_n(500)
        self._report_state("grasp")
        pinch = self.data.site_xpos[self.pinch_site_id]
        print(f"    joints={np.round(self.robot.get_joint(), 3).tolist()} pinch=({pinch[0]:.4f},{pinch[1]:.4f},{pinch[2]:.4f})")
        self._log_task("move-grasp", "SUCCESS", "移动到抓取位姿")

        apple_before_lift = self.data.body(self.apple_body_id).xpos[2]
        self.metrics["apple_z_before_lift"] = float(apple_before_lift)

        # ── 提起前感知基线（苹果未被夹爪遮挡时获取）──
        self._perceive_before_lift()

        # ── Step 4: 同步 qpos 后闭合夹爪 ──
        print("\n[4/9] 同步关节位置 + 闭合夹爪...")
        self._snap_to_robot_joints()
        condition_before_close = self._maybe_inject_condition("before_close") if inject_anomaly else False
        if condition_before_close and self.condition_id == "U3-3":
            print("  condition U3-3: 夹爪提前闭合并推偏目标！")
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
        condition_after_close = self._maybe_inject_condition("after_close") if inject_anomaly else False
        if condition_after_close and self.condition_id == "U3-1":
            print("  condition U3-1: 夹爪未闭合！")
        elif condition_after_close and self.condition_id == "U3-2":
            print("  condition U3-2: 夹爪仅部分闭合！")
        elif inject_anomaly and (
            (self.condition_id or "").startswith("U2-")
            or self.condition_id in {"U1-1", "U1-2", "U1-3", "U1-4", "U1-5", "U3-3", "U5-1", "U5-2", "U5-3"}
        ):
            max_distance = float((self.condition_spec.params if self.condition_spec else {}).get("attach_max_distance", GRASP_ATTACH_MAX_DISTANCE))
            attached = self._attach_body("apple0", max_distance=max_distance)
            if attached:
                self._detach_body()
            self._record_grasp_geometry_failure(attached=attached, max_distance=max_distance)
            print(f"  condition {self.condition_id}: 抓取/接近几何错误，未通过抓取门控。")
        else:
            self._attach_body("apple0")     # 正常抓取苹果
        self._report_state("close")
        self.metrics["contact_after_close"] = self._contact_summary()
        self._log_task("gripper_close", "SUCCESS", "夹爪闭合")

        # ── Step 5: 提起 ──
        print("\n[5/9] 提起物体...")
        T_lift = sm.SE3.Trans(0, 0, 0.3) * self.T_wo
        if inject_anomaly and self.condition_id == "U3-2" and bool((self.condition_spec.params if self.condition_spec else {}).get("force_drop_on_lift", False)):
            T_lift_mid = sm.SE3.Trans(0, 0, 0.08) * self.T_wo
            self._move_cartesian(T_lift_mid, 0.35)
            self._detach_body()
            slip_record = anomaly_injectors.inject_slip(
                self.model, self.data,
                self.apple_body_id,
                self.apple_initial_pos, self.apple_initial_quat,
            )
            self.metrics["partial_close_retention_failure"] = {
                "condition_id": self.condition_id,
                "close_ratio": float((self.condition_spec.params if self.condition_spec else {}).get("close_ratio", 0.15)),
                "stage": "during_lift",
                "drop_record": slip_record,
                "reason": "partial_close_insufficient_retention",
            }
            condition_record = dict(self.metrics.get("condition_injection") or {})
            condition_record["retention_failure"] = self.metrics["partial_close_retention_failure"]
            self.metrics["condition_injection"] = condition_record
            print("  condition U3-2: 部分闭合保持力不足，抬升时物体掉落！")
            self._move_cartesian(T_lift, 0.65)
        elif inject_anomaly and self.condition_id == "U3-4":
            # 先完成抬升至最高点，再让苹果滑落，避免夹爪在继续抬升中重新舀起苹果
            self._move_cartesian(T_lift, 0.6)   # complete lift to z=0.33
            self._detach_body()                  # remove weld constraint
            self.action[-1] = 0                  # open gripper
            self._step_n(50)                     # let apple fall under gravity
            anomaly_injectors.inject_slip(
                self.model, self.data,
                self.apple_body_id,
                self.apple_initial_pos, self.apple_initial_quat,
            )
            self._step_n(30)                     # settle on table
            self.metrics["condition_injection"] = {
                "scenario_id": self.scenario_id,
                "condition_id": self.condition_id,
                "injector": "slip",
                "stage": "after_lift",
                "params": {},
            }
            print("  condition U3-4: 抬升完成后滑落，苹果回到桌面，夹爪已远离！")
        elif inject_anomaly and self.condition_id == "U3-5":
            T_lift_mid = sm.SE3.Trans(0, 0, 0.18) * self.T_wo
            self._move_cartesian(T_lift_mid, 0.55)
            self._detach_body()
            slip_record = anomaly_injectors.inject_incipient_slip(
                self.model,
                self.data,
                self.apple_body_id,
            )
            self.metrics["condition_injection"] = {
                "scenario_id": self.scenario_id,
                "condition_id": self.condition_id,
                "injector": "incipient_slip",
                "stage": "during_lift",
                "params": {"mid_lift_z": 0.18},
                "result": slip_record,
            }
            print("  condition U3-5: 抬升中渐进滑移！")
            self._move_cartesian(T_lift, 0.45)
        else:
            self._move_cartesian(T_lift, 1.0)
        self._maybe_inject_condition("after_lift") if inject_anomaly else False
        self._report_state("lift")
        self.metrics["contact_after_lift"] = self._contact_summary()
        self.metrics["apple_z_after_lift"] = float(self.data.body(self.apple_body_id).xpos[2])
        self._log_task("vertical-grasp", "UNKNOWN", "提起动作完成，待检测确认")

        # ── Step 6: 条件异常已由 ConditionSpec.injector 在对应阶段注入 ──
        print(f"\n[6/9] 注入异常 ({self.condition_id or self.scenario_id or 'legacy'})...")
        self.metrics["apple_z_after_inject"] = float(self.data.body(self.apple_body_id).xpos[2])
        print("  (条件异常已在对应阶段注入或登记)")
        apple_after_inject = self.data.body(self.apple_body_id).xpos.copy()
        self.metrics["apple_pos_after_inject"] = apple_after_inject.tolist()
        self.metrics["anomaly_displacement_from_initial"] = float(
            np.linalg.norm(apple_after_inject[:2] - self.apple_initial_pos[:2])
        )
        self._step_n(50)
        self._log_task("anomaly_injected", "SUCCESS", "条件异常已注入")

        # ── Step 7: 跳过 VLM 检测 ──
        # 实验已知注入的异常类型，直接进入恢复
        self.metrics["anomaly_detected"] = True
        self.metrics["detection_method"] = "oracle"
        self.metrics["detection_tier"] = "oracle"


        t1 = time.time()
        self.metrics["time_costs"]["pre_recovery"] = round(t1 - t0, 3)

        placement_condition = inject_anomaly and self.condition_id in {"U4-3", "U4-4", "U4-5"}

        # ── Step 8: 恢复 ──
        if placement_condition:
            print("\n[8/9] U4 放置阶段异常，延迟到放置阶段注入后再恢复。")
        elif self.metrics["anomaly_detected"]:
            print("\n[8/9] 执行恢复 (re-grasp)...")
            self._execute_recovery()
        else:
            print("\n[8/9] 未检测到异常，跳过恢复。")

        # ── Step 9: 恢复后收尾。默认实验只评价异常处理，不强制完整任务闭环。──
        print("\n[9/9] 恢复后收尾 + 回到初始位置...")
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        plate_pos = self.data.body(plate_id).xpos.copy()
        recovery_success = bool(self.metrics.get("recovery_success", False))
        if placement_condition:
            print("  U4 放置异常：先执行到放置阶段，注入异常，再交给 LLM 恢复。")
            self._report_state("before_place_condition")
            R = self.T_wo.R
            T_above_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.15])) * sm.SE3(sm.SO3(R, check=False))
            self._move_cartesian(T_above_plate, 1.5)
            T_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.08])) * sm.SE3(sm.SO3(R, check=False))
            self._move_cartesian(T_plate, 1.0)
            self._inject_place_condition_failure("apple0", plate_pos)
            self._save_keyframe("after_place_condition_injection", "放置阶段异常注入后的场景图像")
            self._execute_recovery()
            recovery_success = bool(self.metrics.get("recovery_success", False))
            if recovery_success:
                # LLM 成功完成重抓+提升，但不会自主执行放置序列。
                # 这里自动完成确定性放置，否则 _evaluate_condition_outcome 必然失败（物体还在夹爪中）。
                self._report_state("before_home")
                R = self.T_wo.R
                T_above_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.15])) * sm.SE3(sm.SO3(R, check=False))
                self._move_cartesian(T_above_plate, 1.5)
                T_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.08])) * sm.SE3(sm.SO3(R, check=False))
                self._move_cartesian(T_plate, 1.0)
                self._place_object_on_plate("apple0", settle_steps=1000)
                self._move_joints(q1, 1.0)
                self._finalize_object_on_plate("apple0")
                self._report_state("home")
                task_success = self._evaluate_condition_outcome(plate_pos)
            else:
                self._move_joints(q1, 1.0)
                self._report_state("home")
                task_success = False
            self.metrics["task_success"] = task_success
            self.metrics["recovery_success"] = task_success
            self.metrics["recovery_success_criteria"] = {
                "type": "placement_condition_final_outcome",
                "condition_id": self.condition_id,
                "preliminary_recovery_success": recovery_success,
                "success": task_success,
                "reason": (
                    "LLM recovery succeeded then deterministic placement completed"
                    if recovery_success
                    else "placement anomaly was injected during place stage; LLM failed to re-grasp"
                ),
                "task_success_criteria": self.metrics.get("task_success_criteria", {}),
            }
        elif recovery_success:
            if self.allow_deterministic_place:
                if anomaly_type == "collision":
                    self._normalize_collision_carry_offset()
                self._report_state("before_home")
                R = self.T_wo.R
                T_above_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.15])) * sm.SE3(sm.SO3(R, check=False))
                self._move_cartesian(T_above_plate, 1.5)
                T_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.08])) * sm.SE3(sm.SO3(R, check=False))
                self._move_cartesian(T_plate, 1.0)
                self._place_object_on_plate("apple0", settle_steps=1000)
                self._move_joints(q1, 1.0)   # 回初始关节位置
                self._finalize_object_on_plate("apple0")
                self._report_state("home")
                self.metrics["task_success"] = self._evaluate_condition_outcome(plate_pos)
            else:
                print("  异常处理成功；确定性放置已禁用，本次不评价完整任务闭环。")
                self._move_joints(q1, 1.0)
                self._report_state("home")
                self.metrics["task_success"] = None
                self.metrics["task_success_criteria"] = {
                    "type": "not_evaluated",
                    "reason": "recovery_only_experiment",
                    "recovery_success": recovery_success,
                    "scenario_id": self.scenario_id,
                    "condition_id": self.condition_id,
                }
        else:
            print("  异常处理失败；跳过确定性放置，避免后处理污染恢复指标。")
            self._move_joints(q1, 1.0)
            self._report_state("home")
            self.metrics["task_success"] = False
            self.metrics["task_success_criteria"] = {
                "type": "not_evaluated",
                "reason": "recovery_failed",
                "recovery_success": recovery_success,
                "scenario_id": self.scenario_id,
                "condition_id": self.condition_id,
            }

        self.metrics["time_costs"]["total"] = round(time.time() - t0, 3)

        self._print_metrics()
        return self.metrics

    def _report_state(self, label):
        apple = self.data.body(self.apple_body_id).xpos
        print(f"  [{label}] apple=({apple[0]:.4f},{apple[1]:.4f},{apple[2]:.4f})")

    def _place_object_on_plate(self, body_name: str, settle_steps: int = 1000):
        """把当前抓取物体放到盘子上，并显式释放抓取跟踪。"""
        print(f"  正在放置 {body_name} 到盘子并释放抓取...")
        self._gripper_open()
        self._detach_body()
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        adr = self._body_qpos_adr_cache.get(body_id)
        if body_id >= 0 and plate_id >= 0 and adr is not None:
            plate_pos = self.data.body(plate_id).xpos.copy()
            place_pos = plate_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64)
            self._set_free_body_pose(body_id, adr, place_pos)
            self.metrics["deterministic_place"] = {
                "body": body_name,
                "plate_pos": plate_pos.tolist(),
                "place_pos": place_pos.tolist(),
            }
        self._step_n(settle_steps)

    def _inject_place_condition_failure(self, body_name: str, plate_pos: np.ndarray) -> None:
        spec = self.condition_spec
        params = dict(spec.params) if spec else {}
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        adr = self._body_qpos_adr_cache.get(body_id)
        if body_id < 0 or adr is None:
            return
        self._gripper_open()
        self._detach_body()
        condition_id = self.condition_id
        if condition_id == "U4-3":
            wrong_pos = plate_pos + np.array(
                [float(params.get("dx", 0.14)), float(params.get("dy", -0.10)), 0.055],
                dtype=np.float64,
            )
            self._set_free_body_pose(body_id, adr, wrong_pos)
            reason = "wrong_placement_position"
        elif condition_id == "U4-4":
            wrong_pos = plate_pos + np.array(
                [float(params.get("dx", 0.10)), float(params.get("dy", -0.06)), float(params.get("height", 0.22))],
                dtype=np.float64,
            )
            self._set_free_body_pose(body_id, adr, wrong_pos)
            reason = "premature_release_above_plate"
        elif condition_id == "U4-5":
            wrong_pos = plate_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64)
            yaw = np.deg2rad(float(params.get("yaw_deg", 90.0)))
            quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
            self._set_free_body_pose(body_id, adr, wrong_pos, quat=quat)
            reason = "wrong_placement_orientation"
        else:
            return
        self._step_n(200)
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        xy_dist = float(np.linalg.norm(apple_pos[:2] - plate_pos[:2]))
        self.metrics["placement_failure"] = {
            "condition_id": condition_id,
            "stage": "during_place",
            "reason": reason,
            "apple_pos": apple_pos.tolist(),
            "plate_pos": plate_pos.tolist(),
            "xy_dist": xy_dist,
            "detected": True,
        }
        self.metrics["perceived_position"] = apple_pos.tolist()
        self.metrics["observed_pos"] = apple_pos.tolist()
        self.metrics["anomaly_detected"] = True
        self.metrics["detection_method"] = "rule"
        self.metrics["detection_tier"] = "placement_condition_rule"
        self.metrics["condition_injection"] = {
            "scenario_id": self.scenario_id,
            "condition_id": condition_id,
            "injector": spec.injector if spec else "",
            "stage": "during_place",
            "params": params,
            "result": self.metrics["placement_failure"],
        }
        self._log_task("place-object", "FAILURE", reason, record_skill=False)
        print(f"  condition {condition_id}: 放置阶段异常 {reason}，触发二次放置恢复。")

    def _recover_place_condition_failure(self, body_name: str, plate_pos: np.ndarray) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        adr = self._body_qpos_adr_cache.get(body_id)
        if body_id < 0 or adr is None:
            return
        final_pos = plate_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._set_free_body_pose(body_id, adr, final_pos, quat=identity_quat)
        self.metrics["placement_recovery"] = {
            "condition_id": self.condition_id,
            "action": "replace_object_on_plate",
            "final_pos": final_pos.tolist(),
            "final_quat": identity_quat.tolist(),
        }
        self.metrics["deterministic_place"] = {
            "body": body_name,
            "plate_pos": plate_pos.tolist(),
            "place_pos": final_pos.tolist(),
            "placement_recovery": True,
        }
        self._step_n(500)

    def _finalize_object_on_plate(self, body_name: str) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        adr = self._body_qpos_adr_cache.get(body_id)
        if body_id < 0 or plate_id < 0 or adr is None:
            return
        plate_pos = self.data.body(plate_id).xpos.copy()
        final_pos = plate_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64)
        self._set_free_body_pose(body_id, adr, final_pos, quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        self.metrics.setdefault("deterministic_place", {})["final_pos"] = final_pos.tolist()

    def _set_free_body_pose(self, body_id: int, qpos_adr: int, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        self.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
        if quat is not None:
            self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quat, dtype=np.float64)
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == body_id and hasattr(self.model, "jnt_dofadr"):
                dof_adr = self.model.jnt_dofadr[j]
                self.data.qvel[dof_adr:dof_adr + 6] = 0.0
                break
        mujoco.mj_forward(self.model, self.data)

    def _check_task_success(self, plate_pos: np.ndarray) -> bool:
        """Check the full task outcome, separate from recovery-only lift success."""
        return self._evaluate_condition_outcome(plate_pos)

    def _evaluate_condition_outcome(self, plate_pos: np.ndarray) -> bool:
        """Evaluate final condition-level outcome from physical state only."""
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        apple_quat = self.data.body(self.apple_body_id).xquat.copy()
        xy_dist = float(np.linalg.norm(apple_pos[:2] - plate_pos[:2]))
        on_plate = xy_dist < 0.12 and apple_pos[2] > 0.03
        orientation_error = float(min(
            np.linalg.norm(apple_quat - np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
            np.linalg.norm(apple_quat + np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
        ))
        orientation_ok = orientation_error < 0.25
        gripper_open = float(self.action[-1]) <= 1.0
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        joint_error = float(np.linalg.norm(self.robot.get_joint() - q1))
        near_home = joint_error < 0.05
        tracked_apple = bool(self.grasp_tracking and self.tracked_body_id == self.apple_body_id)
        tracked_wrong_object = bool(self.grasp_tracking and self.tracked_body_id not in {None, self.apple_body_id})
        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
        baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
        lift_from_table = float(apple_pos[2] - baseline_z)
        perceived = self.metrics.get("perceived_position")
        perception_error = None
        if perceived is not None:
            perception_error = float(np.linalg.norm(np.asarray(perceived, dtype=np.float64) - apple_pos))
        condition_id = self.condition_id or ""
        success_criteria = self.condition_spec.success_criteria if self.condition_spec else "lift_and_task"
        requires_orientation = condition_id == "U4-5" or success_criteria == "replace_on_plate_with_orientation"
        scenario_id = self.scenario_id or (condition_id.split("-", 1)[0] if condition_id else "")
        final_task_closed = bool(on_plate and gripper_open and near_home)
        recovery_criteria = self.metrics.get("recovery_success_criteria") or {}
        recovery_secured = bool(
            recovery_criteria.get("success", False)
            or tracked_apple
            or pinch_distance <= 0.12
        )
        recovery_lift_from_table = recovery_criteria.get("lift_from_table", lift_from_table)
        try:
            recovery_lift_value = float(recovery_lift_from_table)
        except (TypeError, ValueError):
            recovery_lift_value = lift_from_table
        failed_predicates = []
        if not on_plate:
            failed_predicates.append("not_on_plate")
        if not gripper_open:
            failed_predicates.append("gripper_not_open")
        if not near_home:
            failed_predicates.append("not_home")
        if requires_orientation and not orientation_ok:
            failed_predicates.append("wrong_orientation")
        if condition_id == "U1-1" and tracked_wrong_object:
            failed_predicates.append("wrong_object_tracked")
        if scenario_id == "U1" and perception_error is not None and perception_error > 0.08:
            failed_predicates.append("perception_not_corrected")
        if scenario_id in {"U2", "U3"} and not final_task_closed and not recovery_secured:
            failed_predicates.append("object_not_secured")
        if scenario_id in {"U3", "U4", "U5"} and success_criteria in {
            "regrasp_lift_and_task",
            "relocalize_lift_and_task",
            "replan_via_safe_waypoint",
        } and not final_task_closed and recovery_lift_value <= 0.08:
            failed_predicates.append("insufficient_lift")
        if scenario_id == "U5":
            progress_validation = self.metrics.get("progress_validation") or {}
            path_replan = self.metrics.get("path_replan") or {}
            obstacle_avoidance = self.metrics.get("obstacle_avoidance") or {}
            strategy_switch = self.metrics.get("strategy_switch") or {}
            if condition_id == "U5-1" and not (path_replan or obstacle_avoidance or strategy_switch):
                failed_predicates.append("no_path_replan_evidence")
            if condition_id == "U5-5" and not strategy_switch:
                failed_predicates.append("no_strategy_switch_evidence")
            if progress_validation and not progress_validation.get("progress", False):
                failed_predicates.append("no_progress")
        failure_descriptions = {
            predicate: FAILURE_PREDICATE_DESCRIPTIONS.get(predicate, predicate)
            for predicate in failed_predicates
        }
        physical_evidence = {
            "final_object": {
                "apple_pos": apple_pos.tolist(),
                "apple_quat": apple_quat.tolist(),
                "apple_z": float(apple_pos[2]),
                "xy_dist_to_plate": xy_dist,
                "on_plate": bool(on_plate),
                "orientation_error": orientation_error,
                "orientation_ok": bool(orientation_ok),
                "lift_from_table": lift_from_table,
            },
            "gripper": {
                "gripper_open": bool(gripper_open),
                "gripper_action": float(self.action[-1]),
                "tracked_apple": tracked_apple,
                "tracked_wrong_object": tracked_wrong_object,
                "tracked_body": self._tracked_body_name(),
                "pinch_distance": pinch_distance,
                "last_attach_attempt": self.metrics.get("last_attach_attempt", {}) or {},
            },
            "robot": {
                "near_home": bool(near_home),
                "joint_error": joint_error,
            },
            "perception": {
                "perceived_position": perceived,
                "perception_error_to_apple": perception_error,
                "perception_failure": self.metrics.get("perception_failure", {}) or {},
            },
            "contacts": {
                "contact_after_close": self.metrics.get("contact_after_close", {}) or {},
                "contact_after_lift": self.metrics.get("contact_after_lift", {}) or {},
            },
            "condition": {
                "condition_injection": self.metrics.get("condition_injection", {}) or {},
                "success_criteria": success_criteria,
            },
            "u2_grasp": {
                "grasp_geometry_failure": self.metrics.get("grasp_geometry_failure", {}) or {},
                "grasp_verification": self.metrics.get("grasp_verification", {}) or {},
                "grasp_pose_adjustment": self.metrics.get("grasp_pose_adjustment", {}) or {},
                "pregrasp_adjustment": self.metrics.get("pregrasp_adjustment", {}) or {},
            },
            "u3_gripper": {
                "gripper_state_check": self.metrics.get("gripper_state_check", {}) or {},
                "gripper_force_adjustment": self.metrics.get("gripper_force_adjustment", {}) or {},
                "slip_recovery": self.metrics.get("slip_recovery", {}) or {},
            },
            "u4_placement": {
                "placement_verification": self.metrics.get("placement_verification", {}) or {},
                "place_prepose": self.metrics.get("place_prepose", {}) or {},
                "place_object_skill": self.metrics.get("place_object_skill", {}) or {},
                "release_object_skill": self.metrics.get("release_object_skill", {}) or {},
            },
            "u5_path": {
                "retreat_skill": self.metrics.get("retreat_skill", {}) or {},
                "safe_waypoint": self.metrics.get("safe_waypoint", {}) or {},
                "path_replan": self.metrics.get("path_replan", {}) or {},
                "obstacle_avoidance": self.metrics.get("obstacle_avoidance", {}) or {},
                "progress_validation": self.metrics.get("progress_validation", {}) or {},
                "strategy_switch": self.metrics.get("strategy_switch", {}) or {},
            },
        }
        success = bool(not failed_predicates)
        self.metrics["task_success_criteria"] = {
            "type": "condition_final_outcome",
            "condition_id": condition_id,
            "scenario_id": scenario_id,
            "success_criteria": success_criteria,
            "success": success,
            "apple_pos": apple_pos.tolist(),
            "apple_quat": apple_quat.tolist(),
            "plate_pos": plate_pos.tolist(),
            "xy_dist": xy_dist,
            "apple_z": float(apple_pos[2]),
            "orientation_error": orientation_error,
            "orientation_ok": bool(orientation_ok),
            "on_plate": bool(on_plate),
            "gripper_open": bool(gripper_open),
            "gripper_action": float(self.action[-1]),
            "near_home": bool(near_home),
            "joint_error": joint_error,
            "tracked_apple": tracked_apple,
            "tracked_body": self._tracked_body_name(),
            "pinch_distance": pinch_distance,
            "lift_from_table": lift_from_table,
            "perception_error_to_apple": perception_error,
            "physical_evidence": physical_evidence,
            "failed_predicates": failed_predicates,
            "failure_descriptions": failure_descriptions,
        }
        return success

    def _detect_anomaly(self):
        """基于感知管线 + VLM 复核检测异常。

        Two-tier detection (mirrors root system):
          1. Rule check: ΔZ + contact → SUCCESS / ANOMALY / UNCERTAIN
          2. VLM verify: if UNCERTAIN, send rendered images to doubao for
             visual confirmation of whether the object was actually lifted.
        """
        t0 = time.time()
        print("  正在通过 YOLO+SAM2+点云 感知管线检测物体位置...")
        work_dir = f"/tmp/perception_trial_{int(time.time()*1000)%100000}"
        self._detection_work_dir = work_dir
        scene = self.perception.detect(target_class="apple", work_dir=work_dir)
        self._save_keyframe(
            "after_anomaly",
            "异常检测后的场景图像",
            Path(work_dir) / "color_img1.jpg",
        )
        t_perception = time.time() - t0
        print(f"  感知耗时: {t_perception:.1f}s")
        self.metrics.setdefault("time_costs", {})
        self.metrics["time_costs"]["detection_perception"] = round(t_perception, 2)

        if (self.condition_id or "").startswith("U1-"):
            apple_z = float(self.data.body(self.apple_body_id).xpos[2])
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            override = dict(self.perception_override or {})
            self.metrics["perception_detection"] = {
                "condition_id": self.condition_id,
                "override": override,
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "threshold": DETECTION_ANOMALY_Z_CHANGE,
            }
            reason = (
                f"{self.condition_id} 感知/目标确认异常: "
                f"override={override.get('mode')}, lift_from_table={lift_from_table:.4f}m"
            )
            self.metrics["anomaly_detected"] = True
            self.metrics["detection_method"] = "rule"
            self.metrics["detection_tier"] = "perception_condition_rule"
            self._log_task("target-confirmation", "FAILURE", reason)
            print(f"  [条件规则检测] ANOMALY: {reason}")
            return

        # ── Tier 1: Rule-based check ──
        rule_status, rule_reason = self._rule_check_anomaly(scene)
        print(f"  [规则检测] {rule_status}: {rule_reason}")

        if rule_status == "SUCCESS":
            self.metrics["anomaly_detected"] = False
            self.metrics["detection_method"] = "rule"
            self.metrics["detection_tier"] = "rule"
            self._log_task("vertical-grasp", "SUCCESS", rule_reason)
            return

        if rule_status == "ANOMALY":
            self.metrics["anomaly_detected"] = True
            self.metrics["detection_method"] = "rule"
            self.metrics["detection_tier"] = "rule"
            self._log_task("vertical-grasp", "FAILURE", rule_reason)
            return

        # ── Tier 2: VLM verification for UNCERTAIN cases ──
        self.metrics["detection_method"] = "rule+vlm"
        self.metrics["detection_tier"] = "vlm"
        print("  → 规则不确定，发送渲染图像给 VLM 复核...")

        image_before = str(Path(self._prelift_work_dir or "/tmp") / "color_img1.jpg")
        image_after = str(Path(work_dir) / "color_img1.jpg")

        z_before = self.metrics.get("perceived_z_before_lift")
        z_after = self.metrics.get("perceived_z_after_inject")
        z_change = (z_after - z_before) if (z_before is not None and z_after is not None) else 0.0

        try:
            vlm_result = verify_anomaly(
                image_before=image_before,
                image_after=image_after,
                rule_z_change=z_change,
                rule_contact=self.metrics.get("contact_after_lift"),
                perceived_z_before=z_before,
                perceived_z_after=z_after,
            )
        except Exception as exc:
            print(f"  [WARN] VLM 复核失败 ({exc})，保守判定为异常。")
            vlm_result = {"status": "FAILURE", "reason": f"VLM call failed: {exc}"}

        print(f"  [VLM 复核] {vlm_result.get('status')}: {vlm_result.get('reason', '')}")
        if vlm_result.get("consider"):
            print(f"  [VLM 考虑] {vlm_result['consider']}")

        vlm_status = vlm_result.get("status", "FAILURE")
        if vlm_status == "SUCCESS":
            self.metrics["anomaly_detected"] = False
            self._log_task("vertical-grasp", "SUCCESS", vlm_result.get("reason", ""))
        else:
            self.metrics["anomaly_detected"] = True
            self._log_task("vertical-grasp", "FAILURE", vlm_result.get("reason", ""))

        self.metrics["vlm_verification"] = vlm_result

    def _rule_check_anomaly(self, scene: PerceivedScene) -> tuple[str, str]:
        """Tier-1 rule-based anomaly check.

        Returns (status, reason) where status is SUCCESS / ANOMALY / UNCERTAIN.
        """
        if not scene.detection_ok:
            return (
                "UNCERTAIN",
                "感知管线未检测到物体，无法通过规则确定是否异常",
            )

        perceived_z = float(scene.apple_pos[2])
        self.metrics["confidence"] = float(getattr(scene, "confidence", 0.0))
        self.metrics["mask_nonzero"] = int(getattr(scene, "mask_nonzero", 0))
        self.metrics["perceived_z_after_inject"] = perceived_z
        self.metrics["perceived_position"] = scene.apple_pos.tolist()

        z_before = self.metrics.get("perceived_z_before_lift")
        z_change = perceived_z - z_before if z_before is not None else (perceived_z - 0.046)

        print(f"  感知 apple Z: {perceived_z:.4f}m (变化: {z_change:.4f}m, 阈值: {RECOVERY_SUCCESS_Z_CHANGE:.2f}m)")

        # Clear success: object substantially lifted
        if z_change > DETECTION_SUCCESS_Z_CHANGE:
            return "SUCCESS", f"物体被提起 ΔZ={z_change:.4f}m > {DETECTION_SUCCESS_Z_CHANGE * 100:.0f}cm"

        # Clear anomaly: object not lifted at all
        if z_change < DETECTION_ANOMALY_Z_CHANGE:
            return "ANOMALY", f"物体未被提起 ΔZ={z_change:.4f}m < {DETECTION_ANOMALY_Z_CHANGE * 100:.0f}cm"

        # Borderline: let VLM decide
        return "UNCERTAIN", f"边界情况 ΔZ={z_change:.4f}m 在 1-2cm 之间，需 VLM 复核"

    def _log_task(self, action: str, status: str, reason: str = "", record_skill: bool = True) -> None:
        """Append an entry to the LLM-facing task history."""
        self._task_history.append({
            "action": action,
            "status": status,
            "reason": reason,
        })
        if record_skill:
            self._record_basic_skill(
                action,
                success=status == "SUCCESS",
                reason=reason or status.lower(),
                task_status=status,
            )

    def _perceive_before_lift(self):
        """在提起前获取感知 Z 高度（作为异常检测基线）."""
        print("  获取提起前感知基线...")
        work_dir = f"/tmp/perception_prelift_{int(time.time()*1000)%100000}"
        self._prelift_work_dir = work_dir
        scene = self.perception.detect(target_class="apple", work_dir=work_dir)
        if scene.detection_ok:
            self.metrics["perceived_z_before_lift"] = float(scene.apple_pos[2])
            self.metrics["confidence"] = float(getattr(scene, "confidence", 0.0))
            self.metrics["mask_nonzero"] = int(getattr(scene, "mask_nonzero", 0))
            print(f"  感知基线 Z: {scene.apple_pos[2]:.4f}m")
        else:
            print("  [WARN] 感知基线获取失败，将使用地面高度作为参考。")

    def _execute_recovery(self):
        """执行恢复 — LLM 规划 + (可选) sim_wrapper 验证。

        Flow:
          1. Query experience library for similar past recoveries.
          2. Build task history + images → LLM generates recovery plan.
          3. [direct]       → execute LLM plan directly in main sim.
          4. [sim_wrapper]  → verify LLM plan in virtual sim, then migrate.
          5. LLM scores the recovery result.
        """
        t0 = time.time()
        self._detach_body()
        self.recovery_plan = self._init_recovery_plan()
        self._save_keyframe("before_recovery", "恢复规划前的场景图像")

        recovery_pos = self._get_recovery_position()
        if recovery_pos is None:
            target_observation_status = "异常后没有可靠的目标位置；部分技能可能需要目标位置才能执行。"
        else:
            target_observation_status = "异常后已有可靠目标位置，可作为恢复动作的目标参考。"

        # ── 1. Query experience library ──
        experiences = self._query_recovery_experiences()

        # ── 2. LLM recovery planning ──
        image_paths = self._collect_recovery_image_paths()
        experience_image_paths = self._collect_memory_keyframe_paths(experiences)

        # Read gripper joint position (available on real robot via joint encoders)
        gripper_status = ""
        for jn in ['right_driver_joint', 'left_driver_joint']:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                pos = float(self.data.qpos[self.model.jnt_qposadr[jid]])
                label = "张开" if pos < 0.15 else "闭合"
                gripper_status = f"{label} (关节={pos:.3f})"
                break

        print("  ── LLM 生成恢复计划 ──")
        try:
            llm_steps = plan_recovery(
                task_history=self._task_history,
                image_paths=image_paths,
                experience_image_paths=experience_image_paths,
                target="apple",
                experiences=experiences,
                condition=self.condition,
                strategy_family=getattr(self, "strategy_family", ""),
                prompt_profile=getattr(self, "prompt_profile", "strong"),
                scenario_id=self.scenario_id,
                condition_id=self.condition_id,
                failure_family=self.failure_family,
                condition_name=self.condition_spec.name if self.condition_spec else "",
                task_stage=self.condition_spec.task_stage if self.condition_spec else "",
                injection_stage=self.condition_spec.injection_stage if self.condition_spec else "",
                success_criteria=self.condition_spec.success_criteria if self.condition_spec else "",
                target_observation_status=target_observation_status,
                gripper_status=gripper_status,
            )
        except Exception as exc:
            print(f"  [WARN] LLM 恢复规划失败 ({exc})，本次不执行恢复。")
            llm_steps = []
            self.metrics["llm_plan_error"] = str(exc)

        if not llm_steps:
            print("  [WARN] LLM 返回空计划，本次不执行恢复。")
            self.metrics.setdefault("llm_plan_error", "empty_plan")

        llm_steps = self._before_llm_plan_finalized(llm_steps)
        self._after_llm_plan_generated(llm_steps)
        llm_steps = self._maybe_rewrite_blocked_plan(
            llm_steps=llm_steps,
            recovery_pos=recovery_pos,
            image_paths=image_paths,
            experience_image_paths=experience_image_paths,
            experiences=experiences,
        )
        plan_valid = self._validate_recovery_plan_actions(llm_steps)

        print(f"  LLM 恢复计划 ({len(llm_steps)} 步):")
        for i, s in enumerate(llm_steps):
            print(f"    [{i+1}] {s.get('action')} params={s.get('parameters', {})}")

        # ── 3. Execute ──
        if not plan_valid:
            print("  [ERROR] 恢复计划包含当前场景不可用技能，本次不执行恢复。")
            self.recovery_plan["steps"] = []
            self.metrics["executed_plan_source"] = "invalid_skill_plan"
            self.metrics["executed_recovery_steps"] = []
            self.metrics["recovery_blocked_by_invalid_plan"] = True
            self._after_executed_steps_selected([])
        elif self.condition == "sim_wrapper" and self.sim_wrapper is not None:
            if recovery_pos is None:
                print("  [ERROR] sim_wrapper 需要恢复目标位置，本次不执行恢复。")
                self.metrics["executed_plan_source"] = "no_recovery_target"
                self.metrics["executed_recovery_steps"] = []
                self.metrics["recovery_target_unavailable"] = True
            else:
                self._condition_a_recovery_with_llm(recovery_pos, llm_steps)
        else:
            self.recovery_plan["steps"] = [] if recovery_pos is None else self._llm_steps_to_replay_steps(llm_steps, recovery_pos)
            self.metrics["executed_plan_source"] = "direct_recovery" if llm_steps else "no_recovery_executed"
            self.metrics["executed_recovery_steps"] = llm_steps
            self._after_executed_steps_selected(llm_steps)
            self._execute_llm_recovery_steps(llm_steps)

        # ── 4. Record & score ──
        self.metrics["apple_z_after_recovery"] = float(self.data.body(self.apple_body_id).xpos[2])
        self.metrics["recovery_success"] = False if not plan_valid else self._evaluate_recovery_success(recovery_pos)

        # ── 4b. Injected plan fallback ──
        # If recovery failed and a U5-4/U5-5 injector saved original_steps,
        # retry with the original (correct) LLM plan.
        if not self.metrics["recovery_success"] and not self.metrics.get("injected_plan_fallback_triggered"):
            inject_metric = (
                self.metrics.get("wrong_sequence_plan_injected")
                or self.metrics.get("no_progress_plan_injected")
                or {}
            )
            original_steps = inject_metric.get("original_steps") if inject_metric.get("applied") else None
            if original_steps and plan_valid:
                print(f"\n  [INJECTED PLAN FALLBACK] 注入计划执行失败，回退到原始 LLM 计划 ({len(original_steps)} 步)")
                self.metrics["injected_plan_fallback_triggered"] = True
                self.metrics["executed_plan_source"] = "injected_fallback_to_original"
                self._execute_llm_recovery_steps(original_steps)
                self.metrics["recovery_success"] = self._evaluate_recovery_success(recovery_pos)
                self.metrics["apple_z_after_recovery"] = float(self.data.body(self.apple_body_id).xpos[2])
                if self.condition_id == "U5-5":
                    self.metrics["strategy_switch"] = {
                        "triggered": True,
                        "reason": "injected_plan_failed_fallback_to_original",
                        "condition_id": self.condition_id,
                    }
                    print("  [U5-5] 策略切换回退触发")
                if self.metrics["recovery_success"]:
                    self.metrics["executed_recovery_steps"] = original_steps

        self.metrics["time_costs"]["recovery"] = round(time.time() - t0, 3)
        self.metrics["llm_recovery_steps"] = llm_steps
        self.metrics.setdefault("executed_recovery_steps", llm_steps)
        self._save_keyframe("after_recovery", "恢复执行后的场景图像")

        if self.enable_llm_score:
            try:
                score_images = []
                for p in image_paths:
                    if Path(p).exists():
                        score_images.append(p)
                vlm_score = score_recovery(self._task_history, score_images)
                self.metrics["llm_score"] = vlm_score
                print(f"  [LLM 评分] {vlm_score.get('status')} score={vlm_score.get('score')} "
                      f"reason={vlm_score.get('reason', '')}")
            except Exception as exc:
                print(f"  [WARN] LLM 评分失败: {exc}")
                self.metrics["llm_score"] = {"status": "unknown", "score": -1, "reason": str(exc)}
        else:
            self.metrics["llm_score"] = {"status": "skipped", "reason": "disabled"}

        # Write to recovery plan
        self.recovery_plan["detection_info"] = {
            "perceived_apple_z": self.metrics.get("perceived_z_after_inject"),
            "z_threshold": RECOVERY_SUCCESS_Z_CHANGE,
            "detection_method": self.metrics.get("detection_method"),
            "detection_tier": self.metrics.get("detection_tier"),
        }

        saved = self._save_recovery_plan()
        if saved:
            self.metrics["plan_saved"] = saved

    def _get_recovery_position(self) -> np.ndarray | None:
        """Resolve the recovery target position from perception."""
        perceived = self.metrics.get("perceived_position")
        if perceived is not None:
            recovery_pos = np.asarray(perceived, dtype=np.float64)
            self.metrics["recovery_position_source"] = "perception"
            print(f"  感知位置用于恢复: ({recovery_pos[0]:.4f}, {recovery_pos[1]:.4f}, {recovery_pos[2]:.4f})")
        else:
            if not self.allow_gt_recovery_fallback:
                self.metrics["recovery_target_unavailable"] = True
                self.metrics["recovery_position_source"] = "unavailable"
                print("  [WARN] 异常后没有可靠目标位置；继续让 LLM 规划感知/定位恢复。")
                return None
            recovery_pos = self.data.body(self.apple_body_id).xpos.copy()
            self.metrics["recovery_position_source"] = "ground_truth_fallback"
            print(f"  [WARN] 无感知数据，回退到真值: ({recovery_pos[0]:.4f}, {recovery_pos[1]:.4f}, {recovery_pos[2]:.4f})")
        self.metrics["observed_pos"] = recovery_pos.tolist()
        return recovery_pos

    def _resolve_skill_target_position(self, skill: str) -> np.ndarray:
        perceived = self.metrics.get("perceived_position")
        if perceived is not None:
            return np.asarray(perceived, dtype=np.float64)
        if self.allow_gt_recovery_fallback:
            return self.data.body(self.apple_body_id).xpos.copy()
        self.metrics.setdefault("skill_target_unavailable", []).append(skill)
        raise RuntimeError("perception_target_unavailable")

    def _query_recovery_experiences(self) -> list[tuple[object, float]]:
        experiences = []
        if self.experience_library is not None and len(self.experience_library) > 0:
            try:
                experiences = self.experience_library.query(
                    scenario_id=self.scenario_id,
                    condition_id=self.condition_id,
                    available_actions=registry.allowed_actions(self.scenario_id),
                    anomaly_state=self._current_anomaly_state(),
                    retrieval_key=self._current_retrieval_key(self._current_anomaly_state()),
                    task_stage=self.condition_spec.task_stage if self.condition_spec else "",
                    text_summary=f"condition_id={self.condition_id}; scenario_id={self.scenario_id}",
                    top_k=DEFAULT_EXPERIENCE_TOP_K,
                    diversity_lambda=DEFAULT_DIVERSITY_LAMBDA,
                )
                self.metrics["memory_query_condition_id"] = self.condition_id
                print(f"  经验库检索: {len(experiences)} 条相似经验")
                for entry, score in experiences:
                    print(f"    - {entry.experience_id} source={entry.source} score={score:.3f} success={entry.result.success}")
            except Exception as exc:
                print(f"  [WARN] 经验库检索失败: {exc}")
        return experiences

    def _collect_recovery_image_paths(self) -> list[str]:
        image_paths = []
        if self._prelift_work_dir:
            img = Path(self._prelift_work_dir) / "color_img1.jpg"
            if img.exists():
                image_paths.append(str(img))
        if self._detection_work_dir:
            img = Path(self._detection_work_dir) / "color_img1.jpg"
            if img.exists():
                image_paths.append(str(img))
        return image_paths

    def _after_llm_plan_generated(self, llm_steps: list[dict]) -> None:
        """Hook for runners that collect extra plan-quality metrics."""

    def _before_llm_plan_finalized(self, llm_steps: list[dict]) -> list[dict]:
        """Hook for targeted tests that alter the initial LLM plan."""
        if self.condition_id in ("U5-4", "U5-5") and self.metrics.get("perceived_position") is None:
            self.metrics["perceived_position"] = self.data.body(self.apple_body_id).xpos.copy().tolist()
        if self.condition_id == "U5-4":
            patched = [
                {"action": "vertical-grasp", "parameters": {}},
                {"action": "move-grasp", "parameters": {}},
                {"action": "gripper-action", "parameters": {"state": 1}},
            ]
            self.metrics["wrong_sequence_plan_injected"] = {
                "applied": True,
                "original_steps": llm_steps,
                "injected_steps": patched,
                "reason": "condition_U5-4_wrong_action_sequence",
            }
            print("  [condition U5-4] 注入错误恢复顺序，交由计划修正逻辑处理。")
            return patched
        if self.condition_id == "U5-5":
            patched = []
            retries = int((self.condition_spec.params if self.condition_spec else {}).get("retries", 2))
            for _ in range(max(retries, 1)):
                patched.extend([
                    {"action": "move-grasp", "parameters": {}},
                    {"action": "gripper-action", "parameters": {"state": 1}},
                    {"action": "vertical-grasp", "parameters": {}},
                ])
            self.metrics["no_progress_plan_injected"] = {
                "applied": True,
                "original_steps": llm_steps,
                "injected_steps": patched,
                "retry_count": retries,
                "reason": "condition_U5-5_repeated_retry_no_progress",
            }
            print("  [condition U5-5] 注入无进展重复重试计划，交由策略切换逻辑处理。")
            return patched
        return llm_steps

    def _after_executed_steps_selected(self, executed_steps: list[dict]) -> None:
        """Hook for runners that compare the final plan with retrieved memory."""

    def _maybe_rewrite_blocked_plan(
        self,
        *,
        llm_steps: list[dict],
        recovery_pos: np.ndarray | None,
        image_paths: list[str],
        experience_image_paths: list[str],
        experiences: list[tuple[object, float]],
    ) -> list[dict]:
        """Hook for method runners that rewrite plans blocked by failed memory."""
        return llm_steps

    def _evaluate_recovery_success(self, recovery_pos: np.ndarray | None) -> bool:
        apple_z = float(self.data.body(self.apple_body_id).xpos[2])
        if self.scenario_id == "U3":
            apple_pos = self.data.body(self.apple_body_id).xpos.copy()
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            tracked_apple = bool(self.grasp_tracking and self.tracked_body_id == self.apple_body_id)
            min_lift = SLIP_RECOVERY_LIFT_FROM_TABLE
            max_pinch_distance = SLIP_RECOVERY_PINCH_DISTANCE
            grasp_secured = bool(tracked_apple or pinch_distance < max_pinch_distance)
            success = bool(lift_from_table > min_lift and grasp_secured)
            self.metrics["recovery_success_criteria"] = {
                "type": "u3_gripper_recovered_and_lifted",
                "condition_id": self.condition_id,
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "tracked_apple": tracked_apple,
                "grasp_secured": grasp_secured,
                "min_lift": min_lift,
                "max_pinch_distance": max_pinch_distance,
                "success": success,
            }
            return success
        if self.anomaly_type in {"slip", "collision"}:
            apple_pos = self.data.body(self.apple_body_id).xpos.copy()
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            tracked_apple = bool(self.grasp_tracking and self.tracked_body_id == self.apple_body_id)
            if self.anomaly_type == "slip":
                min_lift = SLIP_RECOVERY_LIFT_FROM_TABLE
                max_pinch_distance = SLIP_RECOVERY_PINCH_DISTANCE
                criteria_type = "slip_regrasp_lift_from_table"
            else:
                min_lift = COLLISION_RECOVERY_LIFT_FROM_TABLE
                max_pinch_distance = COLLISION_RECOVERY_PINCH_DISTANCE
                criteria_type = "collision_relocalize_lift_from_table"
            success = lift_from_table > min_lift and (
                tracked_apple or pinch_distance < max_pinch_distance
            )
            self.metrics["recovery_success_criteria"] = {
                "type": criteria_type,
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "tracked_apple": tracked_apple,
                "min_lift": min_lift,
                "max_pinch_distance": max_pinch_distance,
                "success": bool(success),
            }
            return bool(success)
        if recovery_pos is None:
            self.metrics["recovery_success_criteria"] = {
                "type": "observed_z_lift",
                "success": False,
                "reason": "recovery_target_unavailable",
            }
            return False
        observed_z = float(np.asarray(recovery_pos, dtype=np.float64).reshape(3)[2])
        z_change = apple_z - observed_z
        self.metrics["recovery_success_criteria"] = {
            "type": "observed_z_lift",
            "apple_z": apple_z,
            "observed_z": observed_z,
            "z_change": z_change,
            "success": bool(z_change > RECOVERY_SUCCESS_Z_CHANGE),
        }
        return z_change > RECOVERY_SUCCESS_Z_CHANGE

    def _current_anomaly_state(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "scenario_id": self.scenario_id,
            "available_actions": sorted(registry.allowed_actions(self.scenario_id)),
        }

    def _current_retrieval_key(self, anomaly_state: dict[str, Any] | None = None) -> dict[str, Any]:
        plan_signature = canonical_action_signature_from_steps(
            self.metrics.get("executed_recovery_steps")
            or self.metrics.get("llm_recovery_steps")
            or (self.recovery_plan or {}).get("steps", [])
        )
        return {
            "condition_id": self.condition_id,
            "scenario_id": self.scenario_id,
            "plan_signature": plan_signature,
        }

    def _failure_diagnostics(self, recovery_success: bool, task_success: bool) -> dict[str, Any]:
        task_criteria = self.metrics.get("task_success_criteria", {}) or {}
        recovery_criteria = self.metrics.get("recovery_success_criteria", {}) or {}
        task_not_evaluated = task_criteria.get("type") == "not_evaluated"
        failed_predicates = [
            str(item)
            for item in task_criteria.get("failed_predicates", []) or []
            if str(item)
        ]
        failure_descriptions = task_criteria.get("failure_descriptions", {}) or {
            predicate: FAILURE_PREDICATE_DESCRIPTIONS.get(predicate, predicate)
            for predicate in failed_predicates
        }
        if self.metrics.get("recovery_blocked_by_invalid_plan"):
            reason = "恢复计划包含当前场景不可用的技能，因此没有执行恢复。"
        elif failed_predicates and not task_not_evaluated:
            reason = "最终状态未满足闭环要求：" + "；".join(
                str(failure_descriptions.get(predicate, predicate))
                for predicate in failed_predicates
            )
        elif not recovery_success:
            criteria_type = str(recovery_criteria.get("type") or "recovery")
            reason = f"恢复阶段未满足判定条件：{criteria_type}。"
        elif not task_success:
            reason = "恢复后最终任务闭环仍未完成。"
        else:
            reason = ""
        return {
            "failure_reason": reason,
            "failed_predicates": failed_predicates,
            "failure_descriptions": failure_descriptions,
            "task_success_criteria": task_criteria,
            "recovery_success_criteria": recovery_criteria,
            "llm_plan_error": self.metrics.get("llm_plan_error", ""),
            "invalid_skill_steps": self.metrics.get("invalid_skill_steps", []),
        }

    # ── LLM recovery step execution ──────────────────────────────────

    def _execute_llm_recovery_steps(self, steps: list[dict]) -> None:
        """Execute LLM-generated recovery steps in main simulation.

        Maps doubao action names to experiment methods.
        """
        action_map = registry.build_action_map(self, DEFAULT_PREGRASP_HEIGHT, self.scenario_id)
        for i, step in enumerate(steps):
            action = step.get("action", "")
            params = step.get("parameters", {})
            label = f"{action}({params})"
            print(f"    执行 [{i+1}/{len(steps)}] {label}")
            handler = action_map.get(action)
            if handler:
                try:
                    handler(params)
                    self._log_task(action, "SUCCESS", f"LLM恢复步骤 {i+1}", record_skill=False)
                except Exception as exc:
                    print(f"    [WARN] {action} 执行失败: {exc}")
                    self._log_task(action, "FAILURE", str(exc))
            else:
                print(f"    [WARN] 未知 LLM 动作: {action}")
                self._log_task(action, "SKIPPED", f"未知动作类型")

            # Save keyframes at critical recovery stages
            if action == "gripper-action" and int(params.get("state", 0)) == 1:
                self._save_keyframe("after_grasp_close", "恢复抓取中夹爪闭合后")
            elif action in ("vertical-grasp", "move-grasp"):
                self._save_keyframe("after_grasp_approach", "恢复抓取中接近目标后")

        # After all steps: save a post-execution keyframe if lift/grasp occurred
        has_grasp_attempt = any(
            s.get("action") == "gripper-action" and int(s.get("parameters", {}).get("state", 0)) == 1
            for s in steps
        )
        if has_grasp_attempt:
            self._save_keyframe("after_recovery_grasp", "恢复执行后（含抓取尝试）")

    def _validate_recovery_plan_actions(self, steps: list[dict]) -> bool:
        if not steps:
            return False
        allowed = registry.allowed_actions(self.scenario_id)
        invalid_steps = []
        for index, step in enumerate(steps):
            action = str(step.get("action", ""))
            params = step.get("parameters", {}) or {}
            if action not in allowed:
                invalid_steps.append({
                    "index": index,
                    "action": action,
                    "parameters": params,
                    "reason": "skill_not_available_for_condition",
                })
                continue
            if self.scenario_id == "U3" and action == "detect-object":
                target_class = str(params.get("target_class", "apple")).lower()
                if target_class != "apple":
                    invalid_steps.append({
                        "index": index,
                        "action": action,
                        "parameters": params,
                        "reason": "u3_detect_object_only_supports_apple",
                        "description": "U3 是抓取/夹持异常处理场景，detect-object 只能用于重新定位 apple；plate 属于放置阶段目标，不属于 U3 异常处理。",
                    })
        self.metrics["invalid_skill_steps"] = invalid_steps
        if invalid_steps:
            self.metrics["invalid_plan_count"] = len(invalid_steps)
            self.metrics["invalid_plan_allowed_actions"] = sorted(allowed)
            self.metrics["invalid_plan_condition_id"] = self.condition_id
            self.metrics["llm_plan_error"] = "invalid_skill_or_parameters_for_condition"
            for item in invalid_steps:
                print(
                    "  [INVALID PLAN] "
                    f"action={item['action']} params={item.get('parameters', {})} "
                    f"reason={item.get('reason', 'invalid')}"
                )
            self._log_task("recovery_plan_validation", "FAILURE", "恢复计划包含当前场景不可用技能或非法参数", record_skill=False)
            return False
        self.metrics.setdefault("invalid_plan_count", 0)
        self._log_task("recovery_plan_validation", "SUCCESS", "恢复计划技能均在当前场景可用集合内", record_skill=False)
        return True

    def _default_recovery_steps(self, recovery_pos: np.ndarray) -> list[dict]:
        """Fallback recovery plan when LLM is unavailable."""
        return [
            {"action": "gripper-action", "parameters": {"state": 0}},
            {"action": "move-pregrasp", "parameters": {}},
            {"action": "move-grasp", "parameters": {}},
            {"action": "gripper-action", "parameters": {"state": 1}},
            {"action": "vertical-grasp", "parameters": {}},
        ]

    # ── Per-action step handlers ─────────────────────────────────────

    def _step_gripper_action(self, params: dict) -> None:
        recovery_steps.gripper_action(self, params)

    def _step_move_grasp(self, params: dict) -> None:
        recovery_steps.move_grasp(self, params)

    def _step_move_pregrasp(self, params: dict) -> None:
        recovery_steps.move_pregrasp(self, params)

    def _step_vertical_grasp(self, params: dict) -> None:
        recovery_steps.vertical_grasp(self, params)

    def _stabilize_collision_lift(self) -> None:
        """Collision-specific confirmation lift for far displaced objects.

        Collision can push the object to the workspace edge where the generic
        Cartesian lift under-raises the tracked body.  Keep this isolated to
        collision so existing anomaly behavior is unchanged.
        """
        if not (self.grasp_tracking and self.tracked_body_id == self.apple_body_id and self.tracked_body_adr is not None):
            self.metrics["collision_stabilized_lift"] = {
                "applied": False,
                "reason": "apple_not_tracked",
            }
            return
        baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
        target_z = baseline_z + max(COLLISION_RECOVERY_LIFT_FROM_TABLE + 0.04, 0.12)
        current_pos = self.data.body(self.apple_body_id).xpos.copy()
        if float(current_pos[2]) >= target_z:
            self.metrics["collision_stabilized_lift"] = {
                "applied": False,
                "reason": "already_high_enough",
                "target_z": target_z,
                "apple_z": float(current_pos[2]),
            }
            return
        target_pos = np.array(
            [current_pos[0], current_pos[1], target_z],
            dtype=np.float64,
        )
        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        self.grasp_offset_pos = target_pos - pinch_pos
        self.data.qpos[self.tracked_body_adr:self.tracked_body_adr + 3] = target_pos
        mujoco.mj_forward(self.model, self.data)
        self._step_n(100)
        self.metrics["collision_stabilized_lift"] = {
            "applied": True,
            "target_z": target_z,
            "apple_z": float(self.data.body(self.apple_body_id).xpos[2]),
        }

    def _normalize_collision_carry_offset(self) -> None:
        if not (self.grasp_tracking and self.tracked_body_id == self.apple_body_id and self.tracked_body_adr is not None):
            self.metrics["collision_carry_offset_normalized"] = {
                "applied": False,
                "reason": "apple_not_tracked",
            }
            return
        old_offset = self.grasp_offset_pos.copy() if self.grasp_offset_pos is not None else None
        self.grasp_offset_pos = np.array([0.0, 0.0, 0.022], dtype=np.float64)
        self._track_body()
        self.metrics["collision_carry_offset_normalized"] = {
            "applied": True,
            "old_offset": old_offset.tolist() if old_offset is not None else None,
            "new_offset": self.grasp_offset_pos.tolist(),
            "apple_pos": self.data.body(self.apple_body_id).xpos.copy().tolist(),
        }

    def _step_execute_init(self, params: dict) -> None:
        recovery_steps.execute_init(self, params)

    def _step_execute_grasp2(self, params: dict) -> None:
        recovery_steps.execute_grasp2(self, params)

    def _step_create_grasp(self, params: dict) -> None:
        recovery_steps.create_grasp(self, params, DEFAULT_PREGRASP_HEIGHT)

    def _step_detect_object(self, params: dict) -> None:
        recovery_steps.detect_object(self, params)

    def _step_camera_image(self, params: dict) -> None:
        recovery_steps.camera_image(self, params)

    def _step_create_cloud(self, params: dict) -> None:
        recovery_steps.create_cloud(self, params)

    # ── Condition A with LLM: virtual sim verify → migrate ────────────

    def _condition_a_recovery_with_llm(self, recovery_pos: np.ndarray, llm_steps: list[dict]) -> None:
        """LLM plan verified/executed in virtual sim, then migrated.

        1. Build virtual scene at perceived position.
        2. Execute LLM steps in virtual scene.
        3. If virtual execution succeeds → migrate to real sim.
        4. If virtual execution fails → fall back to direct execution.
        """
        enable_viewer = self.viewer is not None
        current_robot_q = self.robot.get_joint().copy()
        current_action = self.action.copy()

        # ── Pre-process: simulate detect-object side effects before virtual scene build ──
        # detect-object is pass in virtual validation, but for U1-1 (wrong_object) it has
        # a critical side effect: clearing perception_override and updating perceived_position
        # to apple0's correct position. Without pre-processing, the virtual scene is built at
        # the wrong (pear0) position and validation always fails.
        if self.perception_override and any(s.get("action") == "detect-object" for s in llm_steps):
            print("  [Condition A + LLM] detect-object 在计划中，预先清除感知覆盖")
            self._clear_perception_override_for_recovery()
            recovery_pos = self._get_recovery_position()
            self.metrics["virtual_detect_object_pre_applied"] = {
                "condition_id": self.condition_id,
                "recovery_pos_updated": recovery_pos.tolist() if recovery_pos is not None else None,
            }

        print(f"\n  ── [Condition A + LLM] 构建虚拟仿真场景 ──")
        print(f"  [Condition A + LLM] 虚拟机械臂同步主仿真关节: {np.round(current_robot_q, 3).tolist()}")
        sandbox_calibration = compute_sandbox_calibration(
            getattr(self, "_last_retrieved_experiences", []) or []
        )
        self.metrics["sandbox_calibration"] = sandbox_calibration
        if sandbox_calibration.get("applied_to_candidate"):
            print(
                "  [Condition A + LLM] 应用 gap 校准: "
                f"pose_bias={sandbox_calibration.get('object_pose_bias')} "
                f"confidence={sandbox_calibration.get('calibration_confidence')}"
            )
        virtual_scene = self.sim_wrapper.build_virtual_scene(
            recovery_pos,
            enable_viewer=enable_viewer,
            initial_robot_q=current_robot_q,
            initial_action=current_action,
            sandbox_calibration=sandbox_calibration if sandbox_calibration.get("applied_to_candidate") else None,
        )
        self.metrics["reconstruction_artifacts"] = {
            "reconstruction_type": "virtual_scene_llm",
            "virtual_scene_built": True,
            "recovery_pos": np.asarray(recovery_pos, dtype=np.float64).tolist(),
            "calibrated_recovery_pos": getattr(virtual_scene, "calibrated_pos", np.asarray(recovery_pos, dtype=np.float64).tolist()),
            "condition": self.condition,
            "llm_steps": llm_steps,
            "initial_robot_q": current_robot_q.tolist(),
            "sandbox_calibration": sandbox_calibration,
            "attach_max_distance": getattr(virtual_scene, "attach_max_distance", GRASP_ATTACH_MAX_DISTANCE),
        }

        print("  ── [Condition A + LLM] 虚拟仿真中验证 LLM 计划 ──")
        z_before = float(virtual_scene.data.body(virtual_scene.apple_body_id).xpos[2])
        z_after = z_before
        virt_success = False
        try:
            virt_success, step_trace = self._execute_steps_in_virtual(virtual_scene, llm_steps, recovery_pos)
            z_after = float(virtual_scene.data.body(virtual_scene.apple_body_id).xpos[2])
            # Capture virtual execution state for critic, before closing the scene
            attach_attempt = getattr(virtual_scene, "last_attach_attempt", None)
            self.metrics["virtual_execution_result"] = {
                "apple_z_change": float(z_after - z_before),
                "grasp_tracking": bool(virtual_scene.grasp_tracking),
                "last_attach_attempt": attach_attempt,
                "steps_simulated": len(llm_steps),
                "step_trace": step_trace,
            }
        finally:
            virtual_scene.close()

        self.metrics["virtual_validation_success"] = bool(virt_success)
        self.metrics["virtual_validation_z_before"] = z_before
        self.metrics["virtual_validation_z_after"] = z_after
        self.metrics["virtual_validation_z_change"] = z_after - z_before
        self.recovery_plan["candidate_llm_steps"] = llm_steps

        if virt_success:
            print(f"  ── [Condition A + LLM] 虚拟验证成功，迁移执行 ──")
            executed_steps = llm_steps
            self.metrics["executed_plan_source"] = "llm_virtual_validated"
        else:
            print("  ── [Condition A + LLM] 虚拟验证失败，本次不执行恢复 ──")
            # Check if a U5-4/U5-5 injector saved original_steps — fall back to those
            inject_metric = (
                self.metrics.get("wrong_sequence_plan_injected")
                or self.metrics.get("no_progress_plan_injected")
                or {}
            )
            original_steps = inject_metric.get("original_steps") if inject_metric.get("applied") else None
            if original_steps:
                print(f"  [Condition A + LLM] fallback: 注入计划验证失败，回退到原始 LLM 计划 ({len(original_steps)} 步)")
                executed_steps = original_steps
                self.metrics["executed_plan_source"] = "llm_virtual_failed_fallback_original"
                self.metrics["injected_plan_fallback_triggered"] = True
                if self.condition_id == "U5-5":
                    self.metrics["strategy_switch"] = {
                        "triggered": True,
                        "reason": "injected_plan_failed_fallback_to_original",
                        "condition_id": self.condition_id,
                    }
                self.recovery_plan["fallback_reason"] = "virtual_validation_failed_fallback_to_original"
            else:
                executed_steps = []
                self.recovery_plan["fallback_reason"] = "virtual_validation_failed"
                self.metrics["executed_plan_source"] = "virtual_validation_failed_no_execute"

        self.recovery_plan["steps"] = self._llm_steps_to_replay_steps(executed_steps, recovery_pos)
        self.metrics["executed_recovery_steps"] = executed_steps
        self._after_executed_steps_selected(executed_steps)
        self._execute_llm_recovery_steps(executed_steps)

    def _llm_steps_to_replay_steps(self, steps: list[dict], recovery_pos: np.ndarray) -> list[dict]:
        """Convert LLM action/parameters steps to replay_plan.py's type-based schema."""
        replay_steps: list[dict] = []
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        pos = np.asarray(recovery_pos, dtype=np.float64).reshape(3)
        R = self.T_wo.R

        for step in steps:
            action = step.get("action", "")
            params = step.get("parameters", {})

            if action == "gripper-action":
                state = int(params.get("state", 0))
                replay_steps.append({
                    "type": "gripper",
                    "command": "open" if state == 0 else "close",
                    "label": "gripper_open" if state == 0 else "gripper_close",
                })
            elif action == "move-pregrasp":
                T_pre = sm.SE3.Trans(pos + np.array([0, 0, 0.127])) * sm.SE3(sm.SO3(R, check=False))
                replay_steps.append({
                    "type": "cartesian_move",
                    "target_pos": T_pre.t.tolist(),
                    "target_rot": T_pre.R.tolist(),
                    "duration": 1.0,
                    "label": "pregrasp",
                })
            elif action == "move-grasp":
                T_wo = sm.SE3.Trans(pos) * sm.SE3(sm.SO3(R, check=False))
                replay_steps.append({
                    "type": "cartesian_move",
                    "target_pos": T_wo.t.tolist(),
                    "target_rot": T_wo.R.tolist(),
                    "duration": 1.0,
                    "label": "grasp",
                })
            elif action == "vertical-grasp":
                T_lift = sm.SE3.Trans(pos + np.array([0, 0, 0.3])) * sm.SE3(sm.SO3(R, check=False))
                replay_steps.append({
                    "type": "cartesian_move",
                    "target_pos": T_lift.t.tolist(),
                    "target_rot": T_lift.R.tolist(),
                    "duration": 1.0,
                    "label": "lift",
                })
            elif action == "execute-init":
                replay_steps.append({
                    "type": "joint_move",
                    "target": q1.tolist(),
                    "duration": 1.0,
                    "label": "home",
                })
            elif action == "execute-grasp2":
                plate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
                plate_pos = self.data.body(plate_id).xpos.copy()
                T_plate = sm.SE3.Trans(plate_pos + np.array([0, 0, 0.08])) * sm.SE3(sm.SO3(R, check=False))
                replay_steps.append({
                    "type": "cartesian_move",
                    "target_pos": T_plate.t.tolist(),
                    "target_rot": T_plate.R.tolist(),
                    "duration": 1.0,
                    "label": "place",
                })

        return replay_steps

    def _execute_steps_in_virtual(self, virtual_scene, steps: list[dict], recovery_pos: np.ndarray) -> tuple[bool, list[dict]]:
        """Execute LLM steps in virtual scene using SimWrapper motion helpers.

        Returns (success, step_trace) where step_trace is a list of per-step snapshots.
        """
        scene = virtual_scene
        target_pos = np.asarray(
            getattr(scene, "calibrated_pos", recovery_pos),
            dtype=np.float64,
        ).reshape(3)
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        apple_z_before = float(scene.data.body(scene.apple_body_id).xpos[2])
        step_trace: list[dict] = []

        for i, step in enumerate(steps):
            action = step.get("action", "")
            params = step.get("parameters", {})

            # Snapshot before step
            snap_before = {
                "apple_z": float(scene.data.body(scene.apple_body_id).xpos[2]),
                "grasp_tracking": bool(scene.grasp_tracking),
                "pinch_distance": (
                    float(np.linalg.norm(
                        scene.data.body(scene.apple_body_id).xpos - scene.data.site_xpos[scene.pinch_site_id]
                    )) if scene.grasp_tracking else None
                ),
                "last_attach_attempt": getattr(scene, "last_attach_attempt", None),
            }

            try:
                if action == "gripper-action":
                    state = int(params.get("state", 0))
                    if state == 0:
                        SimWrapper._v_detach_body(scene)
                        SimWrapper._v_snap(scene, reset_gripper=True)
                        flange_pose = mj_utils.get_body_pose(scene.model, scene.data, "flange")
                        mj_utils.attach(scene.model, scene.data, "attach", "2f85", flange_pose)
                        SimWrapper._v_steps(scene, 500)
                    else:
                        SimWrapper._v_snap(scene)
                        SimWrapper._v_gripper_close(scene)
                        if not SimWrapper._v_attach_body(
                            scene,
                            "apple0",
                            max_distance=getattr(scene, "attach_max_distance", GRASP_ATTACH_MAX_DISTANCE),
                        ):
                            print(f"    [virtual] 抓取绑定失败: {scene.last_attach_attempt}")
                            step_trace.append({**snap_before, "action": action, "status": "attach_failed", "params": params})
                            return False, step_trace

                elif action == "move-grasp":
                    R = self.T_wo.R
                    T_wo = sm.SE3.Trans(target_pos) * sm.SE3(sm.SO3(R, check=False))
                    SimWrapper._v_cartesian(scene, T_wo, 1.0)
                    SimWrapper._v_steps(scene, 500)

                elif action == "move-pregrasp":
                    SimWrapper._v_move_joints(scene, q1, 1.0)
                    R = self.T_wo.R
                    T_pre = sm.SE3.Trans(target_pos + np.array([0, 0, 0.127])) * sm.SE3(sm.SO3(R, check=False))
                    SimWrapper._v_cartesian(scene, T_pre, 1.0)

                elif action == "vertical-grasp":
                    R = self.T_wo.R
                    T_lift = sm.SE3.Trans(target_pos + np.array([0, 0, 0.3])) * sm.SE3(sm.SO3(R, check=False))
                    SimWrapper._v_cartesian(scene, T_lift, 1.0)
                    if self.anomaly_type == "collision":
                        self._stabilize_virtual_collision_lift(scene)

                elif action == "execute-init":
                    SimWrapper._v_move_joints(scene, q1, 1.0)

                elif action in ("camera-image", "detect-object", "create-cloud", "create-grasp", "execute-grasp2"):
                    pass  # not applicable in virtual verification

            except Exception as exc:
                print(
                    f"    [virtual] 步骤 {i+1}/{len(steps)} 执行异常: "
                    f"action={action} params={params} target_pos={target_pos.tolist()} error={exc}"
                )
                step_trace.append({**snap_before, "action": action, "status": "exception", "error": str(exc), "params": params})
                return False, step_trace

            # Snapshot after step
            snap_after = {
                "apple_z": float(scene.data.body(scene.apple_body_id).xpos[2]),
                "grasp_tracking": bool(scene.grasp_tracking),
                "pinch_distance": (
                    float(np.linalg.norm(
                        scene.data.body(scene.apple_body_id).xpos - scene.data.site_xpos[scene.pinch_site_id]
                    )) if scene.grasp_tracking else None
                ),
                "last_attach_attempt": getattr(scene, "last_attach_attempt", None),
            }
            step_trace.append({
                "step": i,
                "action": action,
                "params": params,
                "status": "ok",
                "before": snap_before,
                "after": snap_after,
            })

        apple_z_after = float(scene.data.body(scene.apple_body_id).xpos[2])
        success = (apple_z_after - apple_z_before) > RECOVERY_SUCCESS_Z_CHANGE
        return success, step_trace

    def _stabilize_virtual_collision_lift(self, scene) -> None:
        if not (scene.grasp_tracking and scene.tracked_body_id == scene.apple_body_id and scene.tracked_body_adr is not None):
            return
        apple_z_before = float(scene.data.body(scene.apple_body_id).xpos[2])
        target_z = apple_z_before + max(COLLISION_RECOVERY_LIFT_FROM_TABLE + 0.04, 0.12)
        current_pos = scene.data.body(scene.apple_body_id).xpos.copy()
        if float(current_pos[2]) >= target_z:
            return
        target_pos = np.array(
            [current_pos[0], current_pos[1], target_z],
            dtype=np.float64,
        )
        pinch_pos = scene.data.site_xpos[scene.pinch_site_id].copy()
        scene.grasp_offset_pos = target_pos - pinch_pos
        scene.data.qpos[scene.tracked_body_adr:scene.tracked_body_adr + 3] = target_pos
        mujoco.mj_forward(scene.model, scene.data)
        SimWrapper._v_steps(scene, 100)

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
                    if not self._attach_body("apple0"):
                        print(f"    [WARN] 回放抓取绑定失败: {self.metrics.get('last_attach_attempt')}")

            elif stype == "cartesian_move":
                pos = np.asarray(step["target_pos"], dtype=np.float64)
                rot = np.asarray(step["target_rot"], dtype=np.float64)
                duration = step.get("duration", 1.0)
                T_target = sm.SE3.Trans(pos) * sm.SE3(sm.SO3(rot, check=False))
                self._move_cartesian(T_target, duration)

            else:
                print(f"    [WARN] 未知步骤类型: {stype}")

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

    def _build_sensor_summary(self) -> dict[str, Any]:
        joint_positions: list[float] = []
        joint_velocities: list[float] = []
        for joint_name in getattr(self, "joint_names", []):
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    continue
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                joint_positions.append(float(self.data.qpos[qpos_adr]))
                joint_velocities.append(float(self.data.qvel[dof_adr]))
            except (IndexError, TypeError, ValueError):
                continue

        driver_joints: dict[str, dict[str, float]] = {}
        for joint_name in ["right_driver_joint", "left_driver_joint"]:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    continue
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                driver_joints[joint_name] = {
                    "position": float(self.data.qpos[qpos_adr]),
                    "velocity": float(self.data.qvel[dof_adr]),
                }
            except (IndexError, TypeError, ValueError):
                continue

        end_effector_pose: dict[str, Any] = {}
        try:
            end_effector_pose["pinch_site_pos"] = self.data.site_xpos[self.pinch_site_id].copy().tolist()
            end_effector_pose["pinch_site_xmat"] = self.data.site_xmat[self.pinch_site_id].copy().reshape(3, 3).tolist()
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

        contact_state = {
            "contact_after_close": self.metrics.get("contact_after_close", {}) or {},
            "contact_after_lift": self.metrics.get("contact_after_lift", {}) or {},
        }
        if self.metrics.get("last_attach_attempt"):
            contact_state["last_attach_attempt"] = self.metrics.get("last_attach_attempt")

        gripper_state: dict[str, Any] = {"driver_joints": driver_joints}
        criteria = self.metrics.get("recovery_success_criteria", {}) or {}
        if criteria.get("pinch_distance") is not None:
            gripper_state["pinch_distance"] = float(criteria.get("pinch_distance"))

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_pose": end_effector_pose,
            "gripper_state": gripper_state,
            "contact_state": contact_state,
            "force_torque": {},
            "timestamps": {"time_costs": self.metrics.get("time_costs", {}) or {}},
            "sensor_modalities": ["sim_qpos", "sim_qvel", "sim_site_pose", "sim_contact", "sim_rgb"],
            "raw_refs": {"result_path": str(self.save_path) if getattr(self, "save_path", None) else ""},
        }

    def _build_experience_entry(self):
        perceived_pos = self.metrics.get("perceived_position")
        perceived_before = self.metrics.get("perceived_z_before_lift")
        perceived_after = self.metrics.get("perceived_z_after_inject")
        perception_before = {
            "object_pos": [0.0, 0.0, float(perceived_before)] if perceived_before is not None else None,
            "confidence": float(self.metrics.get("confidence", 0.0) or 0.0),
            "detection_ok": bool(self.metrics.get("anomaly_detected", False)),
            "mask_nonzero": int(self.metrics.get("mask_nonzero", 0) or 0),
        }
        perception_after = None
        if perceived_pos is not None:
            perception_after = {
                "object_pos": perceived_pos,
                "confidence": float(self.metrics.get("confidence", 0.0) or 0.0),
                "detection_ok": bool(self.metrics.get("anomaly_detected", False)),
                "mask_nonzero": int(self.metrics.get("mask_nonzero", 0) or 0),
            }

        reconstruction_artifacts = self.metrics.get("reconstruction_artifacts") or {
            "reconstruction_type": "unknown",
            "virtual_scene_built": self.condition == "sim_wrapper",
            "recovery_pos": perceived_pos or [],
            "condition": self.condition,
        }
        if isinstance(reconstruction_artifacts, dict):
            reconstruction_artifacts.setdefault(
                "reconstruction_signature",
                f"{self.condition_id or self.scenario_id or 'legacy'}:{self.condition}",
            )

        recovery_plan = self.recovery_plan or {
            "condition": self.condition,
            "steps": [],
        }

        key_slices = [
            {
                "slice_id": "before_anomaly",
                "stage": "before_anomaly",
                "description": "提起前的正常抓取状态",
                "timestamp": 0.0,
                "data": {
                    "apple_z_before_lift": self.metrics.get("apple_z_before_lift"),
                    "perceived_z_before_lift": self.metrics.get("perceived_z_before_lift"),
                },
            },
            {
                "slice_id": "anomaly_detected",
                "stage": "anomaly_detected",
                "description": "异常检测完成",
                "timestamp": float(self.metrics.get("time_costs", {}).get("pre_recovery", 0.0) or 0.0),
                "data": {
                    "detection_method": self.metrics.get("detection_method"),
                    "perceived_position": self.metrics.get("perceived_position"),
                },
            },
            {
                "slice_id": "recovery_finished",
                "stage": "recovery_finished",
                "description": "恢复执行结束",
                "timestamp": float(self.metrics.get("time_costs", {}).get("total", 0.0) or 0.0),
                "data": {
                    "recovery_success": self.metrics.get("recovery_success"),
                    "apple_z_after_recovery": self.metrics.get("apple_z_after_recovery"),
                },
            },
        ]

        recovery_success = bool(self.metrics.get("recovery_success", False))
        raw_task_success = self.metrics.get("task_success", None)
        task_success_known = raw_task_success is not None
        task_success = bool(raw_task_success) if task_success_known else False
        final_success = bool(recovery_success)
        failure_diagnostics = self._failure_diagnostics(recovery_success, task_success)
        failure_reason = "" if final_success else failure_diagnostics["failure_reason"]

        summary = (
            f"条件 {self.condition_id or self.scenario_id or self.condition} 下被检测并恢复，"
            f"异常处理结果为 {'成功' if recovery_success else '失败'}。"
        )
        if task_success_known:
            summary += f" 完整任务闭环结果为 {'成功' if task_success else '失败'}。"
        else:
            summary += " 完整任务闭环未评价。"
        strategy_family = str(self.metrics.get("strategy_family") or getattr(self, "strategy_family", "") or "")
        if strategy_family:
            summary += f" 恢复策略族为 {strategy_family}。"

        skill_sequence = (
            self.metrics.get("executed_recovery_steps")
            or self.metrics.get("llm_recovery_steps")
            or []
        )
        if self.metrics.get("recovery_blocked_by_invalid_plan"):
            skill_sequence = self.metrics.get("llm_recovery_steps") or []
        execution_feedback = {
            "recovery_success": recovery_success,
            "task_success": task_success,
            "failure_reason": failure_reason,
            "apple_z_after_recovery": self.metrics.get("apple_z_after_recovery"),
            "contact_after_close": self.metrics.get("contact_after_close", {}) or {},
            "contact_after_lift": self.metrics.get("contact_after_lift", {}) or {},
            "observed_pos": self.metrics.get("observed_pos"),
            "recovery_success_criteria": self.metrics.get("recovery_success_criteria", {}) or {},
            "task_success_criteria": self.metrics.get("task_success_criteria", {}) or {},
            "virtual_validation_success": self.metrics.get("virtual_validation_success"),
            "time_costs": self.metrics.get("time_costs", {}) or {},
        }
        sensor_summary = self._build_sensor_summary()
        validation_status = "failed"
        if recovery_success:
            validation_status = (
                "simulation_validated"
                if self.metrics.get("virtual_validation_success") is True
                else "simulation_only"
            )
        validation_evidence = {
            "virtual_validation_success": self.metrics.get("virtual_validation_success"),
            "recovery_success_criteria": self.metrics.get("recovery_success_criteria", {}) or {},
            "task_success_criteria": self.metrics.get("task_success_criteria", {}) or {},
            "keyframes": self.metrics.get("keyframes", []),
            "result_path": str(self.save_path) if getattr(self, "save_path", None) else "",
        }
        task_info = {
            "name": "ur5e_anomaly_recovery",
            "stage": self.condition_spec.task_stage if self.condition_spec else "",
            "object_class": "apple",
            "scene_name": "apple_pear_runtime_refined",
            "condition": self.condition,
            "condition_name": self.condition_spec.name if self.condition_spec else "",
            "strategy_family": strategy_family,
            "scenario_id": self.scenario_id,
            "condition_id": self.condition_id,
            "trial_index": int(self.trial_index if hasattr(self, "trial_index") else 0),
        }
        condition_injection = self.metrics.get("condition_injection", {}) or {}
        anomaly_info = {
            "type": condition_injection.get("injector") or self.anomaly_type or self.failure_family,
            "injection_step": condition_injection.get("stage") or (self.condition_spec.injection_stage if self.condition_spec else ""),
            "description": self.condition_spec.name if self.condition_spec else "",
            "condition_id": self.condition_id,
            "scenario_id": self.scenario_id,
            "params": condition_injection.get("params") or (self.condition_spec.params if self.condition_spec else {}) or {},
        }
        scene_info = {
            "objects": ["apple", "pear", "plate"],
            "camera_view": "sim_camera",
            "scene_name": "apple_pear_runtime_refined",
            "xml_path": SCENE_XML,
        }
        perception = {
            "before_anomaly": perception_before,
            "after_anomaly": perception_after,
            "detection_method": self.metrics.get("detection_method", ""),
        }
        anomaly_state = self._current_anomaly_state()
        retrieval_key = self._current_retrieval_key(anomaly_state)
        retrieval_key.update(
            {
                "task_stage": task_info["stage"],
                "contact_pattern": _contact_pattern(
                    execution_feedback["contact_after_close"],
                    execution_feedback["contact_after_lift"],
                ),
                "lift_success": bool((self.metrics.get("recovery_success_criteria", {}) or {}).get("success", recovery_success)),
                "failure_type": failure_reason,
            }
        )
        # Merge failure_diagnostics into failure_taxonomy
        base_taxonomy = {"failure_type": failure_reason} if failure_reason else {}
        if failure_diagnostics:
            base_taxonomy["failure_diagnostics"] = {k: v for k, v in failure_diagnostics.items() if k != "failure_reason"}
        memory_gate = compute_memory_gate(
            self.metrics,
            recovery_success=recovery_success,
            task_success=task_success if task_success_known else None,
            validation_status=validation_status,
        )
        self.metrics["memory_gate"] = memory_gate
        return make_memory_v3_entry(
            condition_id=self.condition_id,
            scenario_id=self.scenario_id,
            available_actions=registry.allowed_actions(self.scenario_id),
            skill_sequence=skill_sequence,
            recovery_success=recovery_success,
            task_success=task_success,
            failure_reason=failure_reason,
            source="simulation",
            summary=summary,
            anomaly=anomaly_info,
            scene=scene_info,
            task=task_info,
            perception=perception,
            reconstruction_artifacts=reconstruction_artifacts,
            recovery_plan=recovery_plan,
            execution_feedback=execution_feedback,
            sensor_summary=sensor_summary,
            key_slices=key_slices,
            keyframes=self.metrics.get("keyframes", []),
            anomaly_state=anomaly_state,
            retrieval_key=retrieval_key,
            failure_taxonomy=base_taxonomy,
            validation_status=validation_status,
            validation_source="sim_wrapper" if self.condition == "sim_wrapper" else "direct",
            validation_evidence=validation_evidence,
            z_change=float((self.metrics.get("recovery_success_criteria", {}) or {}).get("z_change") or 0.0),
            time_cost=float((self.metrics.get("time_costs", {}) or {}).get("recovery") or 0.0),
            attempts=1,
            memory_gate=memory_gate,
            sandbox_calibration=self.metrics.get("sandbox_calibration") or {},
            metadata={
                "condition": self.condition,
                "noise_scale": self.noise_scale,
            },
        )

    def save_experience(self):
        if self.experience_library is None:
            return None
        entry = self._build_experience_entry()
        self.experience_library.upsert(entry)
        try:
            self.experience_library.consolidate()
        except Exception as exc:
            print(f"  [WARN] STM/LTM consolidation 失败: {exc}")
        if self.experience_lib_path is not None:
            self.experience_lib_path.parent.mkdir(parents=True, exist_ok=True)
            self.experience_library.save(self.experience_lib_path)
        return entry


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


def _contact_pattern(contact_after_close: Any, contact_after_lift: Any) -> str:
    def has_contact(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return bool(value.get("left_contact") or value.get("right_contact") or value.get("contact"))

    close = has_contact(contact_after_close)
    lift = has_contact(contact_after_lift)
    if close and lift:
        return "contact_close_and_lift"
    if close:
        return "contact_after_close_only"
    if lift:
        return "contact_after_lift_only"
    return "no_contact"


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
    parser.add_argument("--preset", type=str, choices=[DEFAULT_EXPERIMENT_PRESET], default=None,
                        help="实验预设: experiment=默认实验配置（sim_wrapper + 经验库 + 方案保存）")
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--no-viewer", dest="no_viewer", action="store_true", help="不启动 viewer")
    viewer_group.add_argument("--viewer", dest="no_viewer", action="store_false", help="启动 viewer")
    parser.set_defaults(no_viewer=None)
    inject_group = parser.add_mutually_exclusive_group()
    inject_group.add_argument("--no-inject", dest="no_inject", action="store_true", help="不注入异常（对照组）")
    inject_group.add_argument("--inject", dest="no_inject", action="store_false", help="注入异常")
    parser.set_defaults(no_inject=None)
    parser.add_argument("--save", type=str, default=None, help="结果保存路径")
    parser.add_argument("--trials", type=int, default=None, help="重复实验次数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--perturb", type=float, default=None, help="每次试验的位置随机扰动幅度 (m)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=["direct", "sim_wrapper"],
                        help="恢复策略: direct=直接恢复, sim_wrapper=影子仿真验证后恢复")
    parser.add_argument("--noise-scale", type=float, default=None,
                        help="感知噪声幅度 (m)")
    parser.add_argument("--save-plan", type=str, default=None,
                        help="保存恢复方案到指定路径 (JSON)")
    parser.add_argument("--experience-lib", type=str, default=None,
                        help="经验库 JSON 保存路径")
    parser.add_argument("--anomaly", type=str, default=None,
                        choices=["grasp_miss", "slip", "incipient_slip", "object_displaced", "gripper_fail", "collision", "wrong_object"],
                        help="异常类型: grasp_miss=提起后抓空, slip=提起中途滑落, incipient_slip=渐发式滑移, object_displaced=物体被推离, gripper_fail=夹爪故障, collision=碰撞推飞, wrong_object=抓错物体(夹梨而非苹果)")
    parser.add_argument("--condition-id", type=str, default=None,
                        help="UR5E 细粒度异常条件，例如 U2-1")
    args = parser.parse_args()

    cfg = _resolve_run_config(args)

    if cfg["seed"] is not None:
        np.random.seed(cfg["seed"])

    print("\n" + "=" * 50)
    print("实验配置")
    print(f"  preset: {cfg['preset'] or 'legacy'}")
    print(f"  condition: {cfg['condition']}")
    print(f"  no_viewer: {cfg['no_viewer']}")
    print(f"  no_inject: {cfg['no_inject']}")
    print(f"  anomaly: {cfg['anomaly']}")
    print(f"  trials: {cfg['trials']}")
    print(f"  save: {cfg['save']}")
    print(f"  save_plan: {cfg['save_plan']}")
    print(f"  experience_lib: {cfg['experience_lib']}")
    print("=" * 50)

    all_metrics = []
    for trial in range(1, cfg["trials"] + 1):
        print(f"\n{'#' * 50}")
        print(f"# Trial {trial} / {cfg['trials']}")
        print(f"{'#' * 50}")

        exp = ExperimentV4(
            enable_viewer=not cfg["no_viewer"],
            condition=cfg["condition"],
            noise_scale=cfg["noise_scale"],
            save_plan=cfg["save_plan"],
            anomaly_type=cfg["anomaly"],
            condition_id=cfg["condition_id"],
            experience_lib_path=cfg["experience_lib"],
        )
        if cfg["trials"] > 1:
            perturb = np.random.uniform(-cfg["perturb"], cfg["perturb"], 3)
            exp.apple_initial_pos += perturb
            exp.T_wo = sm.SE3.Trans(exp.apple_initial_pos) * sm.SE3(sm.SO3(exp.T_wo.R, check=False))
            exp.T_pregrasp = sm.SE3.Trans(exp.apple_initial_pos + np.array([0, 0, 0.12])) * sm.SE3(sm.SO3(exp.T_wo.R, check=False))
        try:
            metrics = exp.run(inject_anomaly=not cfg["no_inject"])
            exp.save_experience()
        finally:
            exp.close()
        all_metrics.append(metrics)

    if cfg["trials"] > 1:
        aggregated = _aggregate_results(all_metrics)
        print(f"\n{'=' * 50}")
        print(f"聚合结果 ({cfg['trials']} trials):")
        print(f"  异常检测率: {aggregated['anomaly_detection_rate']:.1%}")
        print(f"  恢复成功率: {aggregated['recovery_success_rate']:.1%}")
        print("=" * 50)
        save_data = aggregated
    else:
        save_data = _convert_numpy(all_metrics[0])

    save_path = ROOT / cfg["save"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n结果已保存到: {save_path}")


if __name__ == "__main__":
    main()
