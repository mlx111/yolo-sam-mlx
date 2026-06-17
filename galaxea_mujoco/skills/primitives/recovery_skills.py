from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.base.arm_ik_skill import ARM_JOINTS
from skills.primitives.object_manipulation_skills import (
    ManipulationSide,
    _move_tcp,
    _object_pos,
    _set_side_gripper,
    _side,
    _tcp_pos,
)


@dataclass(frozen=True)
class RecoverySkillResult:
    name: str
    success: bool
    side: ManipulationSide | None = None
    detected: bool = False
    target_pos: np.ndarray | None = None
    final_error: float | None = None
    contacts: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    message: str = ""


class BaseRecoverySkill:
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        self.name = self.config.get("name", self.__class__.__name__)


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom:{geom_id}"


def _joint_value(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {joint_name}")
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _hold_arm_joints(model: mujoco.MjModel, data: mujoco.MjData, side: ManipulationSide) -> np.ndarray:
    values = []
    for joint_name in ARM_JOINTS[side]:
        value = _joint_value(model, data, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_pos")
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {joint_name}_pos")
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(value, low, high)
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _as_vector(params: dict, keys: tuple[str, str, str], default: tuple[float, float, float] | None = None) -> np.ndarray:
    if all(key in params for key in keys):
        return np.array([params[keys[0]], params[keys[1]], params[keys[2]]], dtype=np.float64)
    if default is None:
        raise ValueError(f"Provide {keys[0]}/{keys[1]}/{keys[2]}")
    return np.asarray(default, dtype=np.float64)


class CheckCollisionSkill(BaseRecoverySkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> RecoverySkillResult:
        del step_callback
        mujoco.mj_forward(model, data)
        ignored_prefixes = tuple(params.get("ignored_geom_prefixes", self.config.get("ignored_geom_prefixes", [])))
        allowed_pairs = {
            tuple(sorted(pair))
            for pair in params.get("allowed_geom_pairs", self.config.get("allowed_geom_pairs", []))
        }
        blocked_prefixes = tuple(params.get("blocked_geom_prefixes", self.config.get("blocked_geom_prefixes", [])))
        contacts: list[tuple[str, str]] = []
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1 = _geom_name(model, int(contact.geom1))
            geom2 = _geom_name(model, int(contact.geom2))
            if ignored_prefixes and (geom1.startswith(ignored_prefixes) or geom2.startswith(ignored_prefixes)):
                continue
            pair = tuple(sorted((geom1, geom2)))
            if pair in allowed_pairs:
                continue
            if blocked_prefixes:
                if geom1.startswith(blocked_prefixes) or geom2.startswith(blocked_prefixes):
                    contacts.append((geom1, geom2))
            else:
                contacts.append((geom1, geom2))
        collision = bool(contacts)
        return RecoverySkillResult(
            self.name,
            success=not collision,
            detected=collision,
            contacts=tuple(contacts),
            message=f"collision_contacts={len(contacts)}",
        )


class StopLiftSkill(BaseRecoverySkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> RecoverySkillResult:
        side = _side(params)
        held = _hold_arm_joints(model, data, side)
        settle_steps = int(params.get("settle_steps", self.config.get("settle_steps", 120)))
        for _ in range(max(settle_steps, 0)):
            _hold_arm_joints(model, data, side)
            mujoco.mj_step(model, data)
            if step_callback is not None:
                step_callback()
        return RecoverySkillResult(
            self.name,
            success=True,
            side=side,
            target_pos=held,
            message=f"held_{side}_arm_joints",
        )


class CheckSlipSkill(BaseRecoverySkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> RecoverySkillResult:
        del step_callback
        side = _side(params)
        object_pos = _object_pos(model, data, params)
        tcp_pos = _tcp_pos(model, data, side)
        max_drop = float(params.get("max_drop", self.config.get("max_drop", 0.03)))
        max_tcp_distance = float(params.get("max_tcp_distance", self.config.get("max_tcp_distance", 0.10)))
        previous_z = params.get("previous_object_z")
        if previous_z is None and "previous_object_pos" in params:
            previous_z = np.asarray(params["previous_object_pos"], dtype=np.float64)[2]
        drop = 0.0 if previous_z is None else float(float(previous_z) - object_pos[2])
        distance = float(np.linalg.norm(object_pos - tcp_pos))
        slip = drop > max_drop or distance > max_tcp_distance
        return RecoverySkillResult(
            self.name,
            success=not slip,
            side=side,
            detected=slip,
            target_pos=object_pos,
            final_error=max(drop, distance),
            message=f"drop={drop:.6f}, tcp_distance={distance:.6f}",
        )


class RecoverFromContactSkill(BaseRecoverySkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> RecoverySkillResult:
        side = _side(params)
        current = _tcp_pos(model, data, side)
        retreat = _as_vector(
            params,
            ("retreat_dx", "retreat_dy", "retreat_dz"),
            tuple(self.config.get("default_retreat", [0.0, 0.0, 0.08])),
        )
        target = current + retreat
        merged_params = {**self.config.get("control_defaults", {}), **params}
        result = _move_tcp(model, data, side, target, merged_params, step_callback=step_callback)
        return RecoverySkillResult(
            self.name,
            success=result.success,
            side=side,
            target_pos=target,
            final_error=result.final_error,
            message=f"retreat_error={result.final_error:.6f}",
        )


class RegraspDeeperSkill(BaseRecoverySkill):
    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> RecoverySkillResult:
        side = _side(params)
        current = _tcp_pos(model, data, side)
        approach = _as_vector(params, ("approach_dx", "approach_dy", "approach_dz"))
        norm = float(np.linalg.norm(approach))
        if norm < 1e-9:
            raise ValueError("approach direction must be non-zero")
        direction = approach / norm
        distance = float(params.get("deeper_distance", self.config.get("deeper_distance", 0.025)))
        target = current + direction * distance
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
        return RecoverySkillResult(
            self.name,
            success=move_result.success,
            side=side,
            target_pos=target,
            final_error=move_result.final_error,
            message=f"deeper_distance={distance:.6f}, gripper_value={gripper_value:.6f}",
        )
