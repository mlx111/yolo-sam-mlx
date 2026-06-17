from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.base.arm_ik_skill import ARM_JOINTS, TORSO_JOINTS
from skills.base.torso_move_skill import R1ProTorsoMoveSkill
from skills.primitives.object_manipulation_skills import _move_joints_to_posture


@dataclass(frozen=True)
class WholeBodyPositioningResult:
    name: str
    success: bool
    target_qpos: np.ndarray
    final_qpos: np.ndarray
    final_error: float
    message: str = ""


class BaseWholeBodyPositioningSkill:
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        self.name = self.config.get("name", self.__class__.__name__)


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return np.asarray(values, dtype=np.float64)


def _clip_to_actuator_ranges(model: mujoco.MjModel, joint_names: tuple[str, ...], target: np.ndarray) -> np.ndarray:
    clipped = target.copy()
    for index, name in enumerate(joint_names):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        if actuator_id < 0:
            continue
        low, high = model.actuator_ctrlrange[actuator_id]
        clipped[index] = np.clip(clipped[index], low, high)
    return clipped


def _target_xyz(model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> np.ndarray:
    if all(key in params for key in ("target_x", "target_y", "target_z")):
        return np.array([params["target_x"], params["target_y"], params["target_z"]], dtype=np.float64)
    if "object_body" in params:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, params["object_body"])
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {params['object_body']}")
        return data.xpos[body_id].copy()
    if "object_geom" in params:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, params["object_geom"])
        if geom_id < 0:
            raise ValueError(f"MuJoCo geom not found: {params['object_geom']}")
        return data.geom_xpos[geom_id].copy()
    raise ValueError("Provide target_x/target_y/target_z or object_body/object_geom")


def _base_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if body_id < 0:
        base_xy = np.zeros(2, dtype=np.float64)
        for index, name in enumerate(("base_x", "base_y")):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                base_xy[index] = float(data.qpos[model.jnt_qposadr[joint_id]])
        return base_xy
    return data.xpos[body_id][:2].copy()


def _posture_from_level(config: dict, params: dict) -> np.ndarray:
    if "target_qpos" in params:
        return np.asarray(params["target_qpos"], dtype=np.float64)

    postures = config.get("height_postures", {})
    level = str(params.get("height_level", config.get("default_height_level", "mid")))
    if level in postures:
        return np.asarray(postures[level], dtype=np.float64)

    if "target_height" in params:
        low_height = float(config.get("low_height", 0.75))
        high_height = float(config.get("high_height", 1.15))
        low = np.asarray(postures.get("low", [0.0, -0.25, 0.20, 0.0]), dtype=np.float64)
        high = np.asarray(postures.get("high", [0.0, 0.35, -0.25, 0.0]), dtype=np.float64)
        alpha = (float(params["target_height"]) - low_height) / max(high_height - low_height, 1e-6)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (1.0 - alpha) * low + alpha * high

    raise ValueError("Provide target_qpos, height_level, or target_height")


class TorsoSetHeightSkill(BaseWholeBodyPositioningSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> WholeBodyPositioningResult:
        target = _posture_from_level(self.config, params)
        if target.shape != (len(TORSO_JOINTS),):
            raise ValueError(f"Expected {len(TORSO_JOINTS)} torso targets, got shape {target.shape}")
        target = _clip_to_actuator_ranges(model, TORSO_JOINTS, target)
        torso = R1ProTorsoMoveSkill.from_json(self.config.get("torso_skill_config", "skills/base/torso_move_skill.json"))
        result = torso.move_to_posture(
            model,
            data,
            target,
            steps=int(params.get("steps", self.config.get("steps", 900))),
            settle_steps=int(params.get("settle_steps", self.config.get("settle_steps", 900))),
            max_joint_step=float(params.get("max_joint_step", self.config.get("max_joint_step", 0.004))),
            fail_threshold=float(params.get("fail_threshold", self.config.get("fail_threshold", 0.02))),
            direct_qpos=bool(params.get("direct_qpos", self.config.get("direct_qpos", False))),
            lock_posture=bool(params.get("lock_posture", True)),
            step_callback=step_callback,
        )
        return WholeBodyPositioningResult(
            self.name,
            result.success,
            result.target_qpos,
            result.final_qpos,
            result.final_error,
            f"height_posture_error={result.final_error:.6f}",
        )


class TorsoTurnToTargetSkill(BaseWholeBodyPositioningSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> WholeBodyPositioningResult:
        mujoco.mj_forward(model, data)
        target_xyz = _target_xyz(model, data, params)
        base_xy = _base_xy(model, data)
        desired_yaw = math.atan2(float(target_xyz[1] - base_xy[1]), float(target_xyz[0] - base_xy[0]))
        yaw_offset = float(params.get("yaw_offset", self.config.get("yaw_offset", 0.0)))
        max_abs_yaw = params.get("max_abs_yaw", self.config.get("max_abs_yaw"))
        target = _joint_qpos(model, data, TORSO_JOINTS)
        target[3] = desired_yaw + yaw_offset
        if max_abs_yaw is not None:
            target[3] = float(np.clip(target[3], -float(max_abs_yaw), float(max_abs_yaw)))
        if "target_qpos" in params:
            explicit = np.asarray(params["target_qpos"], dtype=np.float64)
            if explicit.shape != target.shape:
                raise ValueError(f"Expected {len(TORSO_JOINTS)} torso targets, got shape {explicit.shape}")
            target[:3] = explicit[:3]
        target = _clip_to_actuator_ranges(model, TORSO_JOINTS, target)
        torso = R1ProTorsoMoveSkill.from_json(self.config.get("torso_skill_config", "skills/base/torso_move_skill.json"))
        result = torso.move_to_posture(
            model,
            data,
            target,
            steps=int(params.get("steps", self.config.get("steps", 900))),
            settle_steps=int(params.get("settle_steps", self.config.get("settle_steps", 900))),
            max_joint_step=float(params.get("max_joint_step", self.config.get("max_joint_step", 0.004))),
            fail_threshold=float(params.get("fail_threshold", self.config.get("fail_threshold", 0.02))),
            direct_qpos=bool(params.get("direct_qpos", self.config.get("direct_qpos", False))),
            lock_posture=bool(params.get("lock_posture", True)),
            step_callback=step_callback,
        )
        return WholeBodyPositioningResult(
            self.name,
            result.success,
            result.target_qpos,
            result.final_qpos,
            result.final_error,
            f"target_yaw={target[3]:.6f}, error={result.final_error:.6f}",
        )


class SafeTransportPoseSkill(BaseWholeBodyPositioningSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> WholeBodyPositioningResult:
        joint_names = TORSO_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"]
        if "target_qpos" in params:
            target = np.asarray(params["target_qpos"], dtype=np.float64)
        else:
            posture_name = str(params.get("posture", self.config.get("default_posture", "carry_center")))
            postures = self.config.get("postures", {})
            if posture_name not in postures:
                raise ValueError(f"Unknown safe transport posture: {posture_name}")
            target = np.asarray(postures[posture_name], dtype=np.float64)
        if target.shape != (len(joint_names),):
            raise ValueError(f"Expected {len(joint_names)} upper-body targets, got shape {target.shape}")
        target = _clip_to_actuator_ranges(model, joint_names, target)
        merged_params = {**self.config.get("control_defaults", {}), **params}
        final, error, success = _move_joints_to_posture(
            model,
            data,
            joint_names,
            target,
            merged_params,
            step_callback=step_callback,
        )
        return WholeBodyPositioningResult(
            self.name,
            success,
            target,
            final,
            error,
            f"safe_transport_joint_error={error:.6f}",
        )


class PreGraspSafePostureSkill(BaseWholeBodyPositioningSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> WholeBodyPositioningResult:
        joint_names = TORSO_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"]
        if "target_qpos" in params:
            target = np.asarray(params["target_qpos"], dtype=np.float64)
        else:
            posture_name = str(params.get("posture", self.config.get("default_posture", "left_pregrasp_seed")))
            postures = self.config.get("postures", {})
            if posture_name not in postures:
                raise ValueError(f"Unknown pregrasp safe posture: {posture_name}")
            target = np.asarray(postures[posture_name], dtype=np.float64)
        if target.shape != (len(joint_names),):
            raise ValueError(f"Expected {len(joint_names)} upper-body targets, got shape {target.shape}")
        target = _clip_to_actuator_ranges(model, joint_names, target)
        merged_params = {**self.config.get("control_defaults", {}), **params}
        final, error, success = _move_joints_to_posture(
            model,
            data,
            joint_names,
            target,
            merged_params,
            step_callback=step_callback,
        )
        return WholeBodyPositioningResult(
            self.name,
            success,
            target,
            final,
            error,
            f"pregrasp_seed_joint_error={error:.6f}",
        )
