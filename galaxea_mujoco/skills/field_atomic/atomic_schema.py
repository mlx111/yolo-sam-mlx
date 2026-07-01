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
            name="move_base_relative",
            action="move_base_relative",
            description="Move the mobile base by a relative x/y offset in the current base joint frame.",
            parameter_schema={
                "x": "float meters. Relative displacement along current base +X.",
                "y": "float meters. Relative displacement along current base +Y.",
            },
            source_skill="skills.base.base_motion_skill.BaseMotionSkill",
        ),
        FieldAtomicSkillSpec(
            name="set_torso_posture",
            action="set_torso_posture",
            description="Move torso joints to a fixed vertical torso height level.",
            parameter_schema={
                "level": "str enum: mid | high. Discrete torso height preset.",
            },
            source_skill="skills.base.torso_move_skill.TorsoMoveSkill",
        ),
        FieldAtomicSkillSpec(
            name="head_camera_rgbd_save",
            action="head_camera_rgbd_save",
            description="Capture MuJoCo head-camera RGB-D for torso-frame perception.",
            parameter_schema={},
            source_skill="skills.torso_frame.head_camera_rgbd_save_skill.HeadCameraRGBDSaveSkill",
        ),
        FieldAtomicSkillSpec(
            name="head_camera_grounded_sam2_pose",
            action="head_camera_grounded_sam2_pose",
            description="Detect an object from saved head-camera RGB-D and write torso_link4 object position.",
            parameter_schema={
                "target_class": "str. Object class to detect; output is used by later target_class-linked skills.",
            },
            source_skill="skills.torso_frame.head_camera_grounded_sam2_pose_skill.HeadCameraGroundedSAM2PoseSkill",
        ),
        FieldAtomicSkillSpec(
            name="move_to_pregrasp",
            action="move_to_pregrasp",
            description="Move the TCP to a torso-frame pregrasp pose.",
            parameter_schema={
                "side": "str enum: left | right. Arm used for the move.",
                "target_class": "str. Object class whose latest perceived pose is used as the grasp anchor.",
                "pregrasp_offset_x": "float meters in torso_link4. Offset from grasp pose to pregrasp pose.",
                "pregrasp_offset_y": "float meters in torso_link4. Offset from grasp pose to pregrasp pose.",
                "pregrasp_offset_z": "float meters in torso_link4. Positive z moves above the grasp pose.",
                "topdown_mode": "str enum: palm_down | vertical_down | forward_parallel | current. TCP orientation preset.",
            },
            source_skill="skills.torso_frame.torso_frame_move_to_pregrasp_skill.TorsoFrameMoveToPregraspSkill",
        ),
        FieldAtomicSkillSpec(
            name="plan_cartesian_trajectory",
            action="plan_cartesian_trajectory",
            description="Plan torso-frame Cartesian waypoints before a pregrasp move.",
            parameter_schema={
                "side": "str enum: left | right. Arm used for the planned trajectory.",
                "target_class": "str. Object class whose latest perceived pose is used as the grasp anchor.",
                "pregrasp_offset_x": "float meters in torso_link4. Offset from grasp pose to pregrasp pose.",
                "pregrasp_offset_y": "float meters in torso_link4. Offset from grasp pose to pregrasp pose.",
                "pregrasp_offset_z": "float meters in torso_link4. Positive z moves above the grasp pose.",
                "mode": "str enum: straight | top_then_down | side_then_in. straight directly interpolates to pregrasp; top_then_down rises by clearance_z, translates above pregrasp, then descends; side_then_in moves to a side entry point then enters pregrasp.",
                "side_offset_x": "float meters in torso_link4. Used only by side_then_in: entry point = pregrasp + [side_offset_x, side_offset_y, 0].",
                "side_offset_y": "float meters in torso_link4. Used only by side_then_in: entry point = pregrasp + [side_offset_x, side_offset_y, 0].",
                "clearance_z": "float meters in torso_link4. Used only by top_then_down to set the lifted clearance height.",
                "topdown_mode": "str enum: palm_down | vertical_down | forward_parallel | current. TCP orientation preset.",
            },
            source_skill="skills.torso_frame.torso_frame_plan_cartesian_trajectory_skill.TorsoFramePlanCartesianTrajectorySkill",
        ),
        FieldAtomicSkillSpec(
            name="approach_object",
            action="approach_object",
            description="Move the TCP from pregrasp to grasp in torso coordinates.",
            parameter_schema={
                "side": "str enum: left | right. Arm used for final approach.",
                "target_class": "str. Object class whose latest perceived pose is used as the grasp anchor.",
                "visual_grasp_offset_z": "float meters in torso_link4. Extra z correction added to the perceived grasp point.",
                "topdown_mode": "str enum: palm_down | vertical_down | forward_parallel | current. TCP orientation preset.",
            },
            source_skill="skills.torso_frame.torso_frame_approach_object_skill.TorsoFrameApproachObjectSkill",
        ),
        FieldAtomicSkillSpec(
            name="close_gripper",
            action="close_gripper",
            description="Close the selected side gripper.",
            parameter_schema={"side": "str enum: left | right. Gripper to close."},
            source_skill="skills.torso_frame.torso_frame_close_gripper_skill.TorsoFrameCloseGripperSkill",
        ),
        FieldAtomicSkillSpec(
            name="open_gripper",
            action="open_gripper",
            description="Open the selected side gripper.",
            parameter_schema={"side": "str enum: left | right. Gripper to open."},
            source_skill="skills.torso_frame.torso_frame_open_gripper_skill.TorsoFrameOpenGripperSkill",
        ),
        FieldAtomicSkillSpec(
            name="lift",
            action="lift",
            description="Lift the TCP upward in torso coordinates.",
            parameter_schema={
                "side": "str enum: left | right. Arm used for lift.",
                "target_class": "str. Object class used to resolve the grasped object for lift evaluation.",
                "lift_height": "float meters. Vertical lift distance along torso +Z.",
            },
            source_skill="skills.torso_frame.torso_frame_lift_skill.TorsoFrameLiftSkill",
        ),
        FieldAtomicSkillSpec(
            name="lower_held_object",
            action="lower_held_object",
            description="Lower the currently held object by moving the grasping TCP along torso -Z while keeping the gripper closed.",
            parameter_schema={
                "side": "str enum: left | right. Arm carrying the object.",
                "lower_distance": "float meters, limited to [0.0, 0.08]. Downward distance along torso -Z.",
            },
            source_skill="skills.torso_frame.torso_frame_lower_held_object_skill.TorsoFrameLowerHeldObjectSkill",
        ),
        FieldAtomicSkillSpec(
            name="transport_to_detected_target",
            action="transport_to_detected_target",
            description="Move the grasping TCP to a previously detected target_class position with an optional small XY placement offset while keeping the gripper closed. Fails without moving if the target position JSON is missing.",
            parameter_schema={
                "side": "str enum: left | right. Arm carrying the object.",
                "target_class": "str. Destination/reference object class; previous perception output is used as transport target.",
                "place_offset_x": "optional float meters in torso_link4, limited to [-0.02, 0.02]. Small X placement offset from detected target pose.",
                "place_offset_y": "optional float meters in torso_link4, limited to [-0.02, 0.02]. Small Y placement offset from detected target pose.",
            },
            source_skill="skills.torso_frame.torso_frame_transport_object_skill.TorsoFrameTransportObjectSkill",
        ),
    ]
