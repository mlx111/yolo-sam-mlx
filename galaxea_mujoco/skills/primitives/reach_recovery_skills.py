from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from skills.primitives.approach_object_skill import load_skill as load_approach
from skills.primitives.base_reposition_lateral_skill import load_skill as load_base_reposition
from skills.primitives.left_vertical_lift_skill import load_skill as load_left_lift
from skills.primitives.move_to_pregrasp_skill import load_skill as load_pregrasp
from skills.primitives.pre_grasp_safe_posture_skill import load_skill as load_pregrasp_safe_posture
from skills.primitives.right_vertical_lift_skill import load_skill as load_right_lift
from skills.primitives.torso_set_height_skill import load_skill as load_torso_height
from skills.primitives.torso_turn_to_target_skill import load_skill as load_torso_turn
from skills.primitives.verify_grasp_skill import load_skill as load_verify_grasp


@dataclass(frozen=True)
class CompositeRecoveryResult:
    name: str
    success: bool
    substeps: list[dict] = field(default_factory=list)
    final_error: float = 0.0
    message: str = ""


def _control_defaults(params: dict) -> dict:
    return {
        "control_mode": params.get("control_mode", params.get("tcp_control_mode", "joint_target_velocity_limited")),
        "direct_qpos": bool(params.get("direct_qpos", False)),
        "steps": int(params.get("steps", 600)),
        "settle_steps": int(params.get("settle_steps", 180)),
        "fail_threshold": float(params.get("fail_threshold", 0.04)),
        "max_joint_step": float(params.get("max_joint_step", 0.004)),
        "velocity_limit": params.get("velocity_limit", 0.45),
        "force_scale": float(params.get("force_scale", 0.65)),
        "adaptive_pregrasp_target": bool(params.get("adaptive_pregrasp_target", True)),
        "adaptive_approach_target": bool(params.get("adaptive_approach_target", True)),
    }


def _grasp_params(params: dict, object_body: str) -> dict:
    return {
        "side": str(params.get("side", "left")),
        "object_body": object_body,
        "approach_dx": float(params.get("approach_dx", 0.0)),
        "approach_dy": float(params.get("approach_dy", 0.0)),
        "approach_dz": float(params.get("approach_dz", -1.0)),
        "pregrasp_distance": float(params.get("pregrasp_distance", 0.025)),
        "grasp_offset_x": float(params.get("grasp_offset_x", 0.0)),
        "grasp_offset_y": float(params.get("grasp_offset_y", 0.0)),
        "grasp_offset_z": float(params.get("grasp_offset_z", 0.0)),
        **_control_defaults(params),
    }


def _object_pos(model: mujoco.MjModel, data: mujoco.MjData, object_body: str) -> np.ndarray:
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {object_body}")
    return data.xpos[body_id].copy()


def _step(name: str, result: object) -> dict:
    return {
        "skill": name,
        "success": bool(getattr(result, "success", False)),
        "final_error": float(getattr(result, "final_error", 0.0) or 0.0),
        "message": str(getattr(result, "message", "") or ""),
    }


def _finish(name: str, substeps: list[dict]) -> CompositeRecoveryResult:
    success = all(bool(item.get("success")) for item in substeps)
    failed = [item["skill"] for item in substeps if not item.get("success")]
    final_error = max((float(item.get("final_error") or 0.0) for item in substeps), default=0.0)
    message = "ok" if success else "failed_substeps=" + ",".join(failed)
    return CompositeRecoveryResult(name=name, success=success, substeps=substeps, final_error=final_error, message=message)


class RepositionBaseForReachSkill:
    name = "reposition_base_for_reach"

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> CompositeRecoveryResult:
        object_body = str(params.get("object_body") or "")
        if not object_body:
            raise ValueError("Provide object_body")
        target = _object_pos(model, data, object_body)
        lateral = float(params.get("lateral_offset", -0.04 if target[1] >= 0.0 else 0.04))
        forward = float(params.get("forward_offset", -0.04))
        yaw_delta = float(params.get("yaw_delta", 0.0))
        result = load_base_reposition().execute_recovery_action(
            model,
            data,
            {
                "lateral_offset": lateral,
                "forward_offset": forward,
                "yaw_delta": yaw_delta,
                "steps": int(params.get("base_steps", 360)),
                "settle_steps": int(params.get("base_settle_steps", 80)),
                "max_joint_step": float(params.get("base_max_joint_step", 0.004)),
                "fail_threshold": float(params.get("base_fail_threshold", 0.05)),
                "direct_qpos": bool(params.get("direct_qpos", False)),
            },
            step_callback=step_callback,
        )
        return _finish(self.name, [_step("base_reposition_lateral", result)])


class AdjustTorsoForReachSkill:
    name = "adjust_torso_for_reach"

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> CompositeRecoveryResult:
        object_body = str(params.get("object_body") or "")
        if not object_body:
            raise ValueError("Provide object_body")
        target = _object_pos(model, data, object_body)
        level = str(params.get("height_level") or ("high" if target[2] > 0.88 else "mid"))
        height = load_torso_height().execute_recovery_action(
            model,
            data,
            {
                "height_level": level,
                "steps": int(params.get("torso_steps", 420)),
                "settle_steps": int(params.get("torso_settle_steps", 120)),
                "max_joint_step": float(params.get("torso_max_joint_step", 0.004)),
                "fail_threshold": float(params.get("torso_fail_threshold", 0.05)),
                "direct_qpos": bool(params.get("direct_qpos", False)),
            },
            step_callback=step_callback,
        )
        turn = load_torso_turn().execute_recovery_action(
            model,
            data,
            {
                "object_body": object_body,
                "steps": int(params.get("torso_turn_steps", 300)),
                "settle_steps": int(params.get("torso_turn_settle_steps", 80)),
                "max_joint_step": float(params.get("torso_max_joint_step", 0.004)),
                "fail_threshold": float(params.get("torso_fail_threshold", 0.05)),
                "direct_qpos": bool(params.get("direct_qpos", False)),
            },
            step_callback=step_callback,
        )
        return _finish(self.name, [_step("torso_set_height", height), _step("torso_turn_to_target", turn)])


class RetryPregraspWithSaferOffsetSkill:
    name = "retry_pregrasp_with_safer_offset"

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> CompositeRecoveryResult:
        object_body = str(params.get("object_body") or "")
        if not object_body:
            raise ValueError("Provide object_body")
        posture = load_pregrasp_safe_posture().execute_recovery_action(
            model,
            data,
            {
                "posture": str(params.get("posture", "left_pregrasp_seed")),
                "steps": int(params.get("posture_steps", 360)),
                "settle_steps": int(params.get("posture_settle_steps", 120)),
                "max_joint_step": float(params.get("posture_max_joint_step", 0.004)),
                "fail_threshold": float(params.get("posture_fail_threshold", 0.10)),
                "direct_qpos": bool(params.get("direct_qpos", False)),
            },
            step_callback=step_callback,
        )
        pregrasp_params = _grasp_params(params, object_body)
        pregrasp_params["pregrasp_distance"] = float(params.get("safe_pregrasp_distance", 0.035))
        pregrasp_params["steps"] = int(params.get("pregrasp_steps", 900))
        pregrasp_params["settle_steps"] = int(params.get("pregrasp_settle_steps", 220))
        pregrasp_params["segment_count"] = int(params.get("pregrasp_segment_count", 8))
        pregrasp = load_pregrasp().execute_recovery_action(model, data, pregrasp_params, step_callback=step_callback)
        return _finish(self.name, [_step("pre_grasp_safe_posture", posture), _step("move_to_pregrasp", pregrasp)])


class SlowCartesianApproachSkill:
    name = "slow_cartesian_approach"

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> CompositeRecoveryResult:
        object_body = str(params.get("object_body") or "")
        if not object_body:
            raise ValueError("Provide object_body")
        approach_params = _grasp_params(params, object_body)
        approach_params["steps"] = int(params.get("approach_steps", 1200))
        approach_params["settle_steps"] = int(params.get("approach_settle_steps", 260))
        approach_params["segment_count"] = int(params.get("approach_segment_count", 10))
        approach_params["velocity_limit"] = params.get("approach_velocity_limit", params.get("velocity_limit", 0.25))
        approach_params["force_scale"] = float(params.get("approach_force_scale", 0.45))
        result = load_approach().execute_recovery_action(model, data, approach_params, step_callback=step_callback)
        return _finish(self.name, [_step("approach_object", result)])


class RecoverFromJointLimitSkill:
    name = "recover_from_joint_limit"

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> CompositeRecoveryResult:
        object_body = str(params.get("object_body") or "")
        if not object_body:
            raise ValueError("Provide object_body")
        substeps: list[dict] = []
        base = RepositionBaseForReachSkill().execute_recovery_action(model, data, params, step_callback=step_callback)
        substeps.extend(base.substeps)
        torso = AdjustTorsoForReachSkill().execute_recovery_action(model, data, params, step_callback=step_callback)
        substeps.extend(torso.substeps)
        retry = RetryPregraspWithSaferOffsetSkill().execute_recovery_action(model, data, params, step_callback=step_callback)
        substeps.extend(retry.substeps)
        return _finish(self.name, substeps)


class RetryLiftAfterGraspCheckSkill:
    name = "retry_lift_after_grasp_check"

    def execute_recovery_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        params: dict,
        *,
        step_callback: Callable[[], None] | None = None,
    ) -> CompositeRecoveryResult:
        object_body = str(params.get("object_body") or "")
        if not object_body:
            raise ValueError("Provide object_body")
        side = str(params.get("side", "left"))
        verify = load_verify_grasp().execute_recovery_action(
            model,
            data,
            {
                "side": side,
                "object_body": object_body,
                "max_grasp_distance": float(params.get("max_grasp_distance", 0.08)),
                "min_lift": 0.0,
                "initial_object_z": params.get("initial_object_z"),
            },
            step_callback=step_callback,
        )
        lift_loader = load_left_lift if side == "left" else load_right_lift
        lift = lift_loader().execute_recovery_action(
            model,
            data,
            {
                "lift_dx": 0.0,
                "lift_dy": 0.0,
                "lift_dz": float(params.get("retry_lift_dz", 0.12)),
                "steps": int(params.get("lift_steps", 1200)),
                "settle_steps": int(params.get("lift_settle_steps", 160)),
                "fail_threshold": float(params.get("lift_fail_threshold", 0.04)),
                "lift_tolerance": float(params.get("lift_tolerance", 0.04)),
                **_control_defaults(params),
            },
            step_callback=step_callback,
        )
        return _finish(self.name, [_step("verify_grasp", verify), _step(f"{side}_vertical_lift", lift)])
