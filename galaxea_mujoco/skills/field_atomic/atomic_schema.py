from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldAtomicSkillSpec:
    name: str
    action: str
    side: str = ""
    description: str = ""
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    source_skill: str = ""


@dataclass(frozen=True)
class FieldAtomicAction:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class FieldAtomicResult:
    action: str
    success: bool
    status: str
    message: str
    parameters: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)


def default_field_atomic_skill_specs() -> list[FieldAtomicSkillSpec]:
    return [
        FieldAtomicSkillSpec(
            name="left_arm_move_to_position",
            action="left_arm_move_to_position",
            side="left",
            description="Move the left arm TCP to a target position with LLM-supplied parameters.",
            parameter_schema={
                "target_x": "float",
                "target_y": "float",
                "target_z": "float",
                "target_quat_wxyz": "list[float]",
                "control_frame": "str",
                "orientation_weight": "float",
                "orientation_threshold": "float",
                "steps": "int",
                "settle_steps": "int",
                "max_joint_step": "float",
                "fail_threshold": "float",
                "direct_qpos": "bool",
            },
            source_skill="skills.base.left_arm_move_skill.R1ProLeftArmMoveSkill",
        ),
        FieldAtomicSkillSpec(
            name="right_arm_move_to_position",
            action="right_arm_move_to_position",
            side="right",
            description="Move the right arm TCP to a target position with LLM-supplied parameters.",
            parameter_schema={
                "target_x": "float",
                "target_y": "float",
                "target_z": "float",
                "target_quat_wxyz": "list[float]",
                "control_frame": "str",
                "orientation_weight": "float",
                "orientation_threshold": "float",
                "steps": "int",
                "settle_steps": "int",
                "max_joint_step": "float",
                "fail_threshold": "float",
                "direct_qpos": "bool",
            },
            source_skill="skills.base.right_arm_move_skill.R1ProRightArmMoveSkill",
        ),
        FieldAtomicSkillSpec(
            name="left_gripper_set",
            action="left_gripper_set",
            side="left",
            description="Open or close the left gripper by target value or state.",
            parameter_schema={
                "state": "int",
                "gripper_value": "float",
                "gripper_steps": "int",
                "direct_qpos": "bool",
            },
            source_skill="skills.base.gripper_skill.R1ProGripperSkill",
        ),
        FieldAtomicSkillSpec(
            name="right_gripper_set",
            action="right_gripper_set",
            side="right",
            description="Open or close the right gripper by target value or state.",
            parameter_schema={
                "state": "int",
                "gripper_value": "float",
                "gripper_steps": "int",
                "direct_qpos": "bool",
            },
            source_skill="skills.base.gripper_skill.R1ProGripperSkill",
        ),
        FieldAtomicSkillSpec(
            name="torso_move_to_posture",
            action="torso_move_to_posture",
            description="Move torso joints to a parameterized posture.",
            parameter_schema={
                "target_qpos": "list[float]",
                "torso_joint1": "float",
                "torso_joint2": "float",
                "torso_joint3": "float",
                "torso_joint4": "float",
                "steps": "int",
                "settle_steps": "int",
                "max_joint_step": "float",
                "fail_threshold": "float",
                "closed_loop_gain": "float",
                "direct_qpos": "bool",
                "lock_posture": "bool",
            },
            source_skill="skills.base.torso_move_skill.R1ProTorsoMoveSkill",
        ),
        FieldAtomicSkillSpec(
            name="base_move_to_pose",
            action="base_move_to_pose",
            description="Move the mobile base to a parameterized pose.",
            parameter_schema={
                "target_qpos": "list[float]",
                "base_x": "float",
                "base_y": "float",
                "base_yaw": "float",
                "steps": "int",
                "settle_steps": "int",
                "max_joint_step": "float",
                "fail_threshold": "float",
                "direct_qpos": "bool",
            },
            source_skill="skills.base.base_motion_skill.R1ProBaseMotionSkill",
        ),
        FieldAtomicSkillSpec(
            name="head_camera_capture",
            action="head_camera_capture",
            description="Capture RGB-D from the head-mounted camera.",
            parameter_schema={
                "width": "int",
                "height": "int",
                "include_depth": "bool",
            },
            source_skill="skills.base.head_camera_skill.R1ProHeadCameraSkill",
        ),
        FieldAtomicSkillSpec(
            name="base_lidar_scan",
            action="base_lidar_scan",
            description="Capture a planar lidar scan from the base lidar site.",
            parameter_schema={
                "ray_count": "int",
                "horizontal_fov_deg": "float",
                "min_range": "float",
                "max_range": "float",
                "exclude_sensor_body": "bool",
            },
            source_skill="skills.base.base_lidar_skill.R1ProBaseLidarSkill",
        ),
    ]
