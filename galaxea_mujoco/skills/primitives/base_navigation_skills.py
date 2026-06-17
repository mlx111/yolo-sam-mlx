from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.base.base_motion_skill import BASE_JOINTS, R1ProBaseMotionSkill


@dataclass(frozen=True)
class BaseNavigationResult:
    name: str
    success: bool
    target_qpos: np.ndarray
    final_qpos: np.ndarray
    final_error: float
    waypoints: tuple[np.ndarray, ...] = field(default_factory=tuple)
    message: str = ""


class BaseNavigationSkill:
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        self.name = self.config.get("name", self.__class__.__name__)
        self.base_skill_config = self.config.get("base_skill_config", "skills/base/base_motion_skill.json")

    def _base_skill(self) -> R1ProBaseMotionSkill:
        return R1ProBaseMotionSkill.from_json(self.base_skill_config)


def _base_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = []
    for name in BASE_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return np.asarray(values, dtype=np.float64)


def _object_xy(model: mujoco.MjModel, data: mujoco.MjData, params: dict) -> np.ndarray:
    if all(key in params for key in ("target_x", "target_y")):
        return np.array([params["target_x"], params["target_y"]], dtype=np.float64)
    if "object_body" in params:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, params["object_body"])
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {params['object_body']}")
        return data.xpos[body_id][:2].copy()
    if "object_geom" in params:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, params["object_geom"])
        if geom_id < 0:
            raise ValueError(f"MuJoCo geom not found: {params['object_geom']}")
        return data.geom_xpos[geom_id][:2].copy()
    raise ValueError("Provide target_x/target_y or object_body/object_geom")


def _yaw_to_target(base_xy: np.ndarray, target_xy: np.ndarray, yaw_offset: float = 0.0) -> float:
    return math.atan2(float(target_xy[1] - base_xy[1]), float(target_xy[0] - base_xy[0])) + yaw_offset


def _region_pose(config: dict, params: dict) -> np.ndarray:
    if "target_qpos" in params:
        target = np.asarray(params["target_qpos"], dtype=np.float64)
    elif all(key in params for key in ("base_x", "base_y", "base_yaw")):
        target = np.array([params["base_x"], params["base_y"], params["base_yaw"]], dtype=np.float64)
    else:
        region_name = params.get("region", params.get("region_name", config.get("default_region")))
        regions = config.get("regions", {})
        if region_name is None or region_name not in regions:
            raise ValueError(f"Unknown base region: {region_name!r}")
        target = np.asarray(regions[region_name], dtype=np.float64)
    if target.shape != (3,):
        raise ValueError(f"Expected 3-element base target, got shape {target.shape}")
    return target


def _motion_params(config: dict, params: dict) -> dict:
    defaults = config.get("control_defaults", {})
    return {
        "steps": int(params.get("steps", defaults.get("steps", 900))),
        "settle_steps": int(params.get("settle_steps", defaults.get("settle_steps", 120))),
        "max_joint_step": float(params.get("max_joint_step", defaults.get("max_joint_step", 0.01))),
        "fail_threshold": float(params.get("fail_threshold", defaults.get("fail_threshold", 0.02))),
        "direct_qpos": bool(params.get("direct_qpos", defaults.get("direct_qpos", False))),
    }


class BaseMoveToRegionSkill(BaseNavigationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> BaseNavigationResult:
        target = _region_pose(self.config, params)
        result = self._base_skill().move_to_pose(
            model,
            data,
            target,
            **_motion_params(self.config, params),
            step_callback=step_callback,
        )
        return BaseNavigationResult(
            self.name,
            result.success,
            result.target_qpos,
            result.final_qpos,
            result.final_error,
            (target,),
            f"base_region_error={result.final_error:.6f}",
        )


class BaseRepositionLateralSkill(BaseNavigationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> BaseNavigationResult:
        current = _base_qpos(model, data)
        lateral_offset = float(params.get("lateral_offset", params.get("delta_y", 0.0)))
        forward_offset = float(params.get("forward_offset", params.get("delta_x", 0.0)))
        yaw = current[2]
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        left = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        delta_xy = forward * forward_offset + left * lateral_offset
        target = current.copy()
        target[:2] = target[:2] + delta_xy
        target[2] = target[2] + float(params.get("yaw_delta", 0.0))
        result = self._base_skill().move_to_pose(
            model,
            data,
            target,
            **_motion_params(self.config, params),
            step_callback=step_callback,
        )
        return BaseNavigationResult(
            self.name,
            result.success,
            result.target_qpos,
            result.final_qpos,
            result.final_error,
            (target,),
            f"lateral_offset={lateral_offset:.6f}, error={result.final_error:.6f}",
        )


class BaseReplanPathSkill(BaseNavigationSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> BaseNavigationResult:
        mujoco.mj_forward(model, data)
        current = _base_qpos(model, data)
        if "waypoints" in params:
            waypoints = tuple(np.asarray(point, dtype=np.float64) for point in params["waypoints"])
        else:
            target_xy = _object_xy(model, data, params)
            standoff = float(params.get("standoff_distance", self.config.get("standoff_distance", 0.65)))
            yaw_offset = float(params.get("yaw_offset", self.config.get("yaw_offset", 0.0)))
            direction = target_xy - current[:2]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                direction = np.array([1.0, 0.0], dtype=np.float64)
            else:
                direction = direction / norm
            final_xy = target_xy - direction * standoff
            final_yaw = _yaw_to_target(final_xy, target_xy, yaw_offset)
            lateral_clearance = float(params.get("lateral_clearance", self.config.get("lateral_clearance", 0.25)))
            side = np.array([-direction[1], direction[0]], dtype=np.float64)
            midpoint = (current[:2] + final_xy) * 0.5 + side * lateral_clearance
            waypoints = (
                np.array([midpoint[0], midpoint[1], final_yaw], dtype=np.float64),
                np.array([final_xy[0], final_xy[1], final_yaw], dtype=np.float64),
            )
        if not waypoints:
            raise ValueError("Provide at least one waypoint")
        base = self._base_skill()
        last_result = None
        for waypoint in waypoints:
            if waypoint.shape != (3,):
                raise ValueError(f"Expected 3-element waypoint, got shape {waypoint.shape}")
            last_result = base.move_to_pose(
                model,
                data,
                waypoint,
                **_motion_params(self.config, params),
                step_callback=step_callback,
            )
            if not last_result.success and bool(params.get("stop_on_failure", True)):
                break
        if last_result is None:
            raise RuntimeError("No base path waypoint executed")
        return BaseNavigationResult(
            self.name,
            last_result.success,
            last_result.target_qpos,
            last_result.final_qpos,
            last_result.final_error,
            waypoints,
            f"waypoints={len(waypoints)}, final_error={last_result.final_error:.6f}",
        )
