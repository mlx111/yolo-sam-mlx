"""Condition registry for UR5e wrapper1 anomaly benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    scenario_id: str
    name: str
    failure_family: str
    legacy_anomaly_type: str
    task_stage: str
    injection_stage: str
    injector: str
    params: dict[str, Any] = field(default_factory=dict)
    success_criteria: str = "lift_and_task"


def _spec(
    condition_id: str,
    scenario_id: str,
    name: str,
    family: str,
    legacy: str,
    stage: str,
    injection_stage: str,
    injector: str,
    params: dict[str, Any] | None = None,
    criteria: str = "lift_and_task",
) -> ConditionSpec:
    return ConditionSpec(condition_id, scenario_id, name, family, legacy, stage, injection_stage, injector, params or {}, criteria)


CONDITIONS: dict[str, ConditionSpec] = {
    "U1-1": _spec("U1-1", "U1", "wrong_object_selected", "perception_target_error", "wrong_object", "detect", "before_recovery", "target_body_override", {"target_body": "pear0"}, "correct_target_recovered_and_task"),
    "U1-2": _spec("U1-2", "U1", "stale_pose_after_object_moved", "perception_pose_error", "object_displaced", "detect", "before_recovery", "stale_pose", {"dx": 0.08, "dy": 0.04, "attach_max_distance": 0.045}, "correct_target_recovered_and_task"),
    "U1-3": _spec("U1-3", "U1", "partial_occlusion_pose_bias", "perception_pose_error", "perception_noise", "detect", "before_recovery", "occlusion_noise", {"dx": 0.035, "dy": -0.025, "occlusion_ratio": 0.45}, "correct_target_recovered_and_task"),
    "U1-4": _spec("U1-4", "U1", "mask_boundary_confusion", "perception_pose_error", "perception_noise", "detect", "before_recovery", "boundary_confusion", {"dx": -0.035, "dy": 0.03, "mask_boundary_noise": 0.35}, "correct_target_recovered_and_task"),
    "U1-5": _spec("U1-5", "U1", "perception_orientation_error", "perception_orientation_error", "perception_noise", "detect", "before_recovery", "perception_yaw_error", {"yaw_deg": 45.0}, "correct_target_recovered_and_task"),
    "U2-1": _spec("U2-1", "U2", "grasp_lateral_offset_x", "grasp_geometry_error", "grasp_miss", "move_grasp", "before_move_grasp", "grasp_pose_offset", {"dx": 0.055, "dy": 0.0, "attach_max_distance": 0.045}, "object_secured_and_lifted"),
    "U2-2": _spec("U2-2", "U2", "grasp_lateral_offset_y", "grasp_geometry_error", "grasp_miss", "move_grasp", "before_move_grasp", "grasp_pose_offset", {"dx": 0.0, "dy": 0.055, "attach_max_distance": 0.045}, "object_secured_and_lifted"),
    "U2-3": _spec("U2-3", "U2", "grasp_height_offset", "grasp_geometry_error", "grasp_miss", "move_grasp", "before_move_grasp", "grasp_pose_offset", {"dz": 0.055, "attach_max_distance": 0.045}, "object_secured_and_lifted"),
    "U2-4": _spec("U2-4", "U2", "pregrasp_too_far", "pregrasp_geometry_error", "grasp_miss", "move_pregrasp", "before_move_pregrasp", "pregrasp_offset", {"dx": 0.08, "dy": 0.04, "propagate_to_grasp": True}, "object_secured_and_lifted"),
    "U2-5": _spec("U2-5", "U2", "grasp_yaw_error", "grasp_orientation_error", "grasp_miss", "move_grasp", "before_move_grasp", "grasp_pose_offset", {"yaw_deg": 35.0, "attach_max_distance": 0.045}, "object_secured_and_lifted"),
    "U3-1": _spec("U3-1", "U3", "gripper_not_closing", "gripper_failure", "gripper_fail", "gripper_close", "after_gripper_close", "gripper_fail", {}, "u3_gripper_recovered_and_lifted"),
    "U3-2": _spec("U3-2", "U3", "partial_gripper_close", "gripper_failure", "partial_close", "gripper_close", "after_gripper_close", "partial_close", {"close_ratio": 0.15, "force_drop_on_lift": True, "attach_max_distance": 0.035}, "u3_gripper_recovered_and_lifted"),
    "U3-3": _spec("U3-3", "U3", "premature_gripper_close", "gripper_timing_error", "premature_close", "gripper_close", "before_gripper_close", "premature_close", {"push_dx": 0.05, "push_dy": 0.025, "attach_max_distance": 0.025}, "u3_gripper_recovered_and_lifted"),
    "U3-4": _spec("U3-4", "U3", "early_lift_slip", "slip_failure", "slip", "lift", "during_lift", "transport_drop", {}, "u3_gripper_recovered_and_lifted"),
    "U3-5": _spec("U3-5", "U3", "incipient_slip", "slip_failure", "incipient_slip", "lift", "during_lift", "transport_drop", {"incipient": True}, "u3_gripper_recovered_and_lifted"),
    "U4-1": _spec("U4-1", "U4", "transport_drop", "transport_failure", "transport_drop", "transport", "after_lift", "transport_drop", {}, "replace_on_plate"),
    "U4-2": _spec("U4-2", "U4", "transport_displace", "transport_failure", "transport_displace", "transport", "after_lift", "transport_displace", {"dx": 0.06, "dy": -0.04}, "replace_on_plate"),
    "U4-3": _spec("U4-3", "U4", "wrong_placement_position", "placement_failure", "wrong_place", "place", "during_place", "wrong_place_position", {"dx": 0.14, "dy": -0.10}, "replace_on_plate"),
    "U4-4": _spec("U4-4", "U4", "premature_release_above_plate", "placement_failure", "premature_release", "place", "during_place", "premature_release", {"dx": 0.10, "dy": -0.06, "height": 0.22}, "replace_on_plate"),
    "U4-5": _spec("U4-5", "U4", "wrong_placement_orientation", "placement_orientation_failure", "wrong_orientation", "place", "during_place", "wrong_place_orientation", {"yaw_deg": 90.0}, "replace_on_plate_with_orientation"),
    "U5-1": _spec("U5-1", "U5", "blocked_path_requires_replan", "path_failure", "blocked_path", "move_pregrasp", "before_move_pregrasp", "blocked_path", {"dx": 0.05, "dy": 0.04, "raise_z": 0.12}, "path_replanned_and_task"),
    "U5-2": _spec("U5-2", "U5", "approach_collision_neighbor", "collision_failure", "collision", "move_grasp", "before_move_grasp", "approach_collision", {"dx": 0.06, "dy": -0.04, "vx": 0.6, "vy": -0.4, "vz": 1.5}, "collision_recovered_and_task"),
    "U5-3": _spec("U5-3", "U5", "table_collision_low_grasp", "collision_failure", "collision", "move_grasp", "before_move_grasp", "table_collision", {"dz": -0.08, "dx": 0.04, "dy": -0.03}, "collision_recovered_and_task"),
    "U5-4": _spec("U5-4", "U5", "wrong_sequence_plan", "plan_sequence_failure", "wrong_sequence", "recovery_plan", "before_recovery", "wrong_sequence_plan", {}, "strategy_switch_and_task"),
    "U5-5": _spec("U5-5", "U5", "no_progress_requires_strategy_switch", "no_progress_failure", "no_progress", "recovery_plan", "before_recovery", "no_progress_plan", {"retries": 2}, "strategy_switch_and_task"),
}


def get_condition_spec(condition_id: str | None) -> ConditionSpec | None:
    if not condition_id:
        return None
    key = str(condition_id).upper()
    if key not in CONDITIONS:
        available = ", ".join(sorted(CONDITIONS))
        raise KeyError(f"unknown UR5e condition_id {condition_id!r}; available: {available}")
    return CONDITIONS[key]
