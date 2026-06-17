from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.primitives.object_manipulation_skills import ManipulationSide, _move_tcp, _set_side_gripper, _side


@dataclass(frozen=True)
class DrawerDoorSkillResult:
    name: str
    success: bool
    position: np.ndarray | None = None
    joint_qpos: float | None = None
    joint_axis: np.ndarray | None = None
    joint_type: str | None = None
    message: str = ""


class BaseDrawerDoorSkill:
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        self.name = self.config.get("name", self.__class__.__name__)


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    return data.site_xpos[site_id].copy()


def _geom_pos(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> np.ndarray:
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"MuJoCo geom not found: {geom_name}")
    return data.geom_xpos[geom_id].copy()


def _target_pos_from_site_or_geom(model: mujoco.MjModel, data: mujoco.MjData, params: dict, config: dict, site_key: str, geom_key: str) -> np.ndarray:
    site_name = str(params.get(site_key, config.get(site_key, "")))
    if site_name and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) >= 0:
        return _site_pos(model, data, site_name)
    geom_name = str(params.get(geom_key, config.get(geom_key, "")))
    if geom_name:
        return _geom_pos(model, data, geom_name)
    raise ValueError(f"Provide {site_key} or {geom_key}")


def _joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {joint_name}")
    return joint_id


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = _joint_id(model, joint_name)
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _joint_axis(model: mujoco.MjModel, joint_name: str) -> np.ndarray:
    joint_id = _joint_id(model, joint_name)
    return model.jnt_axis[joint_id].copy()


def _joint_type_name(model: mujoco.MjModel, joint_name: str) -> str:
    joint_type = model.jnt_type[_joint_id(model, joint_name)]
    if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        return "slide"
    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return "hinge"
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return "free"
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return "ball"
    return str(int(joint_type))


def _set_joint_actuator(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_name: str,
    target: float,
    *,
    steps: int,
    settle_steps: int,
    step_callback: Callable[[], None] | None,
) -> float:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"MuJoCo actuator not found: {actuator_name}")
    low, high = model.actuator_ctrlrange[actuator_id]
    target = float(np.clip(target, low, high))
    start = float(data.ctrl[actuator_id])
    for index in range(max(steps, 1)):
        alpha = (index + 1) / max(steps, 1)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        data.ctrl[actuator_id] = (1.0 - alpha) * start + alpha * target
        mujoco.mj_step(model, data)
        if step_callback is not None:
            step_callback()
    data.ctrl[actuator_id] = target
    for _ in range(max(settle_steps, 0)):
        mujoco.mj_step(model, data)
        if step_callback is not None:
            step_callback()
    return target


def _offset(params: dict, prefix: str, default: tuple[float, float, float]) -> np.ndarray:
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")
    if all(key in params for key in keys):
        return np.array([params[keys[0]], params[keys[1]], params[keys[2]]], dtype=np.float64)
    return np.asarray(default, dtype=np.float64)


class DetectHandleSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        del step_callback
        site_name = str(params.get("handle_site", self.config.get("handle_site", "handle_site")))
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) >= 0:
            pos = _site_pos(model, data, site_name)
            return DrawerDoorSkillResult(self.name, True, position=pos, message=f"detected site {site_name}")
        geom_name = str(params.get("handle_geom", self.config.get("handle_geom", "drawer_handle_collision_geom")))
        pos = _geom_pos(model, data, geom_name)
        return DrawerDoorSkillResult(self.name, True, position=pos, message=f"detected geom {geom_name}")


class RedetectHandleSkill(DetectHandleSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        result = super().execute_recovery_action(model, data, params, step_callback=step_callback)
        return DrawerDoorSkillResult(
            self.name,
            result.success,
            position=result.position,
            message=result.message.replace("detected", "redetected", 1),
        )


class InferHingeOrSlideDirectionSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        del data, step_callback
        joint_name = str(params.get("joint_name", self.config.get("joint_name", "drawer_slide_joint")))
        joint_type = _joint_type_name(model, joint_name)
        axis = _joint_axis(model, joint_name)
        expected = str(params.get("expected_type", self.config.get("expected_type", "slide")))
        success = joint_type == expected
        return DrawerDoorSkillResult(
            self.name,
            success,
            joint_axis=axis,
            joint_type=joint_type,
            message=f"joint={joint_name}, type={joint_type}, axis={np.round(axis, 6).tolist()}",
        )


class VerifyDrawerOpenSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        del step_callback
        joint_name = str(params.get("joint_name", self.config.get("joint_name", "drawer_slide_joint")))
        qpos = _joint_qpos(model, data, joint_name)
        min_open = float(params.get("min_open", self.config.get("min_open", 0.18)))
        success = qpos >= min_open
        return DrawerDoorSkillResult(
            self.name,
            success,
            joint_qpos=qpos,
            message=f"{joint_name}={qpos:.6f}, min_open={min_open:.6f}",
        )


class PullDrawerSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        joint_name = str(params.get("joint_name", self.config.get("joint_name", "drawer_slide_joint")))
        actuator_name = str(params.get("actuator_name", self.config.get("actuator_name", "drawer_slide_joint_pos")))
        target_open = float(params.get("target_open", self.config.get("target_open", 0.25)))
        steps = int(params.get("steps", self.config.get("steps", 500)))
        settle_steps = int(params.get("settle_steps", self.config.get("settle_steps", 120)))
        target = _set_joint_actuator(
            model,
            data,
            actuator_name,
            target_open,
            steps=steps,
            settle_steps=settle_steps,
            step_callback=step_callback,
        )
        qpos = _joint_qpos(model, data, joint_name)
        tolerance = float(params.get("tolerance", self.config.get("tolerance", 0.02)))
        success = abs(qpos - target) <= tolerance
        return DrawerDoorSkillResult(
            self.name,
            success,
            joint_qpos=qpos,
            message=f"target_open={target:.6f}, {joint_name}={qpos:.6f}",
        )


class GraspHandleSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        side = _side(params)
        handle_pos = _target_pos_from_site_or_geom(
            model,
            data,
            params,
            self.config,
            "handle_site",
            "handle_geom",
        )
        target = handle_pos + _offset(params, "handle_offset", tuple(self.config.get("handle_offset", [0.0, 0.0, 0.0])))
        merged_params = {**self.config.get("control_defaults", {}), **params}
        move_result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        gripper_value = _set_side_gripper(
            model,
            data,
            side,
            "close",
            steps=int(params.get("gripper_steps", self.config.get("gripper_steps", 240))),
            direct_qpos=bool(params.get("direct_qpos", self.config.get("direct_qpos", False))),
            step_callback=step_callback,
        )
        return DrawerDoorSkillResult(
            self.name,
            move_result.success,
            position=target,
            message=f"side={side}, handle_target_error={move_result.final_error:.6f}, gripper_value={gripper_value:.6f}",
        )


class PushOrPullDoorSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        joint_name = str(params.get("joint_name", self.config.get("joint_name", "drawer_slide_joint")))
        actuator_name = str(params.get("actuator_name", self.config.get("actuator_name", f"{joint_name}_pos")))
        joint_type = _joint_type_name(model, joint_name)
        if "target_qpos" in params:
            target = float(params["target_qpos"])
        elif joint_type == "slide":
            target = float(params.get("target_open", self.config.get("target_open", 0.25)))
        elif joint_type == "hinge":
            target = float(params.get("target_angle", self.config.get("target_angle", 0.8)))
        else:
            raise ValueError(f"Unsupported door/drawer joint type: {joint_type}")
        target = _set_joint_actuator(
            model,
            data,
            actuator_name,
            target,
            steps=int(params.get("steps", self.config.get("steps", 500))),
            settle_steps=int(params.get("settle_steps", self.config.get("settle_steps", 120))),
            step_callback=step_callback,
        )
        qpos = _joint_qpos(model, data, joint_name)
        tolerance = float(params.get("tolerance", self.config.get("tolerance", 0.02)))
        success = abs(qpos - target) <= tolerance
        return DrawerDoorSkillResult(
            self.name,
            success,
            joint_qpos=qpos,
            joint_type=joint_type,
            message=f"joint={joint_name}, type={joint_type}, target={target:.6f}, qpos={qpos:.6f}",
        )


class InsertHandIntoDrawerSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        side = _side(params)
        inside_pos = _target_pos_from_site_or_geom(
            model,
            data,
            params,
            self.config,
            "drawer_inside_site",
            "drawer_inside_geom",
        )
        target = inside_pos + _offset(params, "inside_offset", tuple(self.config.get("inside_offset", [0.0, 0.0, 0.0])))
        merged_params = {**self.config.get("control_defaults", {}), **params}
        move_result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        return DrawerDoorSkillResult(
            self.name,
            move_result.success,
            position=target,
            message=f"side={side}, insert_error={move_result.final_error:.6f}",
        )


class ExtractObjectFromDrawerSkill(BaseDrawerDoorSkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> DrawerDoorSkillResult:
        side = _side(params)
        if all(key in params for key in ("target_x", "target_y", "target_z")):
            target = np.array([params["target_x"], params["target_y"], params["target_z"]], dtype=np.float64)
        else:
            exit_pos = _target_pos_from_site_or_geom(
                model,
                data,
                params,
                self.config,
                "drawer_open_site",
                "drawer_open_geom",
            )
            target = exit_pos + _offset(params, "extract_offset", tuple(self.config.get("extract_offset", [-0.12, 0.0, 0.02])))
        merged_params = {**self.config.get("control_defaults", {}), **params}
        move_result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        return DrawerDoorSkillResult(
            self.name,
            move_result.success,
            position=target,
            message=f"side={side}, extract_error={move_result.final_error:.6f}",
        )
