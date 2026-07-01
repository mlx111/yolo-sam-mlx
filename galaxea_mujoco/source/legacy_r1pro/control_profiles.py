"""Sandbox control profiles for R1Pro MuJoCo execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SITE_SERVO_SAFE_PROFILE: dict[str, Any] = {
    "profile_id": "r1pro_physical_actuator_site_servo_safe_v1",
    "arm": {
        "control_mode": "site_servo",
        "direct_qpos": False,
        "stabilize": True,
        "servo_steps": 2200,
        "steps": 2200,
        "settle_steps": 900,
        "solve_iterations": 1400,
        "fail_threshold": 0.075,
        "orientation_threshold": 1.0,
        "orientation_weight": 0.0,
        "damping": 0.002,
        "max_cart_step": 0.003,
        "max_joint_step": 0.004,
        "posture_gain": 0.04,
        "runtime_damping": 110.0,
        "runtime_armature": 0.12,
        "force_scale": 2.0,
    },
    "base": {
        "steps": 600,
        "settle_steps": 180,
        "max_joint_step": 0.004,
        "fail_threshold": 0.04,
        "direct_qpos": False,
    },
    "torso": {
        "steps": 600,
        "settle_steps": 180,
        "max_joint_step": 0.003,
        "fail_threshold": 0.04,
        "direct_qpos": False,
    },
    "gripper": {
        "direct_qpos": False,
        "gripper_steps": 240,
        "settle_steps": 80,
    },
}


REAL_DRIVER_LIKE_PROFILE: dict[str, Any] = {
    "profile_id": "r1pro_real_driver_like_v1",
    "arm": {
        "control_mode": "joint_target_velocity_limited",
        "direct_qpos": False,
        "stabilize": True,
        "closed_loop": True,
        "lock_posture": False,
        "servo_steps": 1800,
        "steps": 1800,
        "settle_steps": 1200,
        "fail_threshold": 0.075,
        "velocity_limit": [3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0],
        "acceleration_limit_scale": 1.5,
        "jerk_limit_scale": 1.5,
        "posture_gain": 0.08,
        "max_joint_step": 0.006,
        "force_scale": 2.0,
        "runtime_damping": 110.0,
        "runtime_armature": 0.12,
        "conservative_cartesian_segments": True,
        "segment_count": 4,
        "stop_on_segment_failure": False,
        "adaptive_pregrasp_target": True,
        "adaptive_approach_target": True,
        "pregrasp_safe_posture": False,
        "pregrasp_safe_posture_name": "left_pregrasp_seed",
        "pregrasp_safe_posture_steps": 1800,
        "pregrasp_safe_posture_settle_steps": 800,
        "pregrasp_safe_posture_direct_qpos": False,
        "whole_body_prepare_grasp": True,
        "whole_body_prepare_direct_qpos": False,
        "whole_body_prepare_arm_seed": True,
        "whole_body_prepare_max_base_delta": 0.12,
        "whole_body_prepare_max_torso_yaw": 0.30,
        "pregrasp_base_align": False,
        "pregrasp_torso_align": False,
        "pregrasp_torso_align_direct_qpos": False,
        "pregrasp_torso_max_yaw": 0.35,
        "enforce_joint_limits": True,
    },
    "base": {
        "control_mode": "joint_servo",
        "steps": 600,
        "settle_steps": 180,
        "max_joint_step": 0.004,
        "fail_threshold": 0.04,
        "direct_qpos": False,
    },
    "torso": {
        "control_mode": "joint_servo",
        "steps": 600,
        "settle_steps": 180,
        "max_joint_step": 0.003,
        "fail_threshold": 0.04,
        "direct_qpos": False,
        "velocity_limit": [1.5, 1.5, 1.5, 1.5],
    },
    "gripper": {
        "control_mode": "position_target",
        "opening_range_mm": [0.0, 100.0],
        "direct_qpos": False,
        "gripper_steps": 240,
        "settle_steps": 80,
    },
}


REAL_DRIVER_LIKE_SLOW_PROFILE: dict[str, Any] = {
    "profile_id": "r1pro_real_driver_like_slow_v1",
    "arm": {
        "control_mode": "joint_target_velocity_limited",
        "direct_qpos": False,
        "stabilize": True,
        "closed_loop": True,
        "lock_posture": False,
        "servo_steps": 2600,
        "steps": 2600,
        "settle_steps": 1600,
        "fail_threshold": 0.075,
        "velocity_limit": [1.2, 1.2, 1.2, 1.2, 2.0, 2.0, 2.0],
        "acceleration_limit_scale": 1.5,
        "jerk_limit_scale": 1.5,
        "posture_gain": 0.12,
        "max_joint_step": 0.002,
        "runtime_damping": 180.0,
        "runtime_armature": 0.25,
        "force_scale": 1.2,
        "conservative_cartesian_segments": True,
        "segment_count": 6,
        "stop_on_segment_failure": False,
        "adaptive_pregrasp_target": True,
        "adaptive_approach_target": True,
        "pregrasp_safe_posture": True,
        "pregrasp_safe_posture_name": "left_pregrasp_seed",
        "pregrasp_safe_posture_steps": 2600,
        "pregrasp_safe_posture_settle_steps": 1200,
        "pregrasp_safe_posture_direct_qpos": False,
        "enforce_joint_limits": True,
    },
    "base": REAL_DRIVER_LIKE_PROFILE["base"],
    "torso": REAL_DRIVER_LIKE_PROFILE["torso"],
    "gripper": REAL_DRIVER_LIKE_PROFILE["gripper"],
}


CONTROL_PROFILES: dict[str, dict[str, Any]] = {
    "site_servo_safe_v1": SITE_SERVO_SAFE_PROFILE,
    "r1pro_physical_actuator_safe_v1": SITE_SERVO_SAFE_PROFILE,
    "real_driver_like_v1": REAL_DRIVER_LIKE_PROFILE,
    "r1pro_real_driver_like_v1": REAL_DRIVER_LIKE_PROFILE,
    "real_driver_like_slow_v1": REAL_DRIVER_LIKE_SLOW_PROFILE,
    "r1pro_real_driver_like_slow_v1": REAL_DRIVER_LIKE_SLOW_PROFILE,
}


DEFAULT_PHYSICAL_CONTROL_PROFILE_ID = "real_driver_like_v1"


def get_control_profile(profile_id: str | None = None) -> dict[str, Any]:
    resolved = profile_id or DEFAULT_PHYSICAL_CONTROL_PROFILE_ID
    if resolved not in CONTROL_PROFILES:
        valid = ", ".join(sorted(CONTROL_PROFILES))
        raise ValueError(f"Unsupported control profile: {resolved!r}. Valid profiles: {valid}")
    return deepcopy(CONTROL_PROFILES[resolved])
