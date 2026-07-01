from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FR5FieldAtomicSkillSpec:
    name: str
    action: str
    description: str = ""
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    parameter_enums: dict[str, list[str]] = field(default_factory=dict)
    parameter_vector_lengths: dict[str, int] = field(default_factory=dict)
    required_any: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    source_skill: str = ""
    llm_visible: bool = True


@dataclass(frozen=True)
class FR5FieldAtomicResult:
    action: str
    success: bool
    status: str
    message: str
    parameters: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)


def default_field_atomic_skill_specs() -> list[FR5FieldAtomicSkillSpec]:
    return [
        FR5FieldAtomicSkillSpec(
            name="camera_rgbd_save",
            action="camera_rgbd_save",
            description="Render and save the current FR5 ee_camera RGB-D observation for recovery.",
            parameter_schema={},
            source_skill="skills.primitives.camera_rgbd_save_skill.CameraRGBDSaveSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="detect_object_pose",
            action="detect_object_pose",
            description="Detect the target object pose from saved RGB-D and record its observed position.",
            parameter_schema={"target_class": "required str. Object class to detect, e.g. apple, pear, or plate."},
            parameter_enums={"target_class": ["apple", "pear", "plate"]},
            required_any=(("target_class",),),
            source_skill="skills.primitives.detect_object_pose_skill.DetectObjectPoseSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="go_camera_ready",
            action="go_camera_ready",
            description="Move FR5 to the initial ee_camera observation posture.",
            parameter_schema={},
            source_skill="skills.primitives.go_camera_ready_skill.GoCameraReadySkill",
            llm_visible=False,
        ),
        FR5FieldAtomicSkillSpec(
            name="create_fixed_vertical_grasp",
            action="create_fixed_vertical_grasp",
            description="Create fixed vertical-down grasp/pregrasp poses from the latest detected grasp target.",
            parameter_schema={},
            source_skill="skills.primitives.create_fixed_vertical_grasp_skill.CreateFixedVerticalGraspSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="move_to_pregrasp",
            action="move_to_pregrasp",
            description="Move FR5 TCP to the fixed vertical pregrasp pose.",
            parameter_schema={
                "dx": "required float meters. Pregrasp offset from grasp point in world x.",
                "dy": "required float meters. Pregrasp offset from grasp point in world y.",
                "dz": "required float meters. Pregrasp offset from grasp point in world z.",
            },
            parameter_ranges={"dx": (-0.05, 0.05), "dy": (-0.05, 0.05), "dz": (0.02, 0.10)},
            required_any=(("dx", "dy", "dz"),),
            source_skill="skills.primitives.move_to_pregrasp_skill.MoveToPregraspSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="approach_object",
            action="approach_object",
            description="Move FR5 TCP from pregrasp to the fixed vertical grasp pose.",
            parameter_schema={
                "dx": "required float meters. Final grasp point offset from detected object center in world x.",
                "dy": "required float meters. Final grasp point offset from detected object center in world y.",
                "dz": "required float meters. Final grasp point offset from detected object center in world z.",
            },
            parameter_ranges={"dx": (-0.02, 0.05), "dy": (-0.02, 0.05), "dz": (-0.02, 0.05)},
            required_any=(("dx", "dy", "dz"),),
            source_skill="skills.primitives.approach_object_skill.ApproachObjectSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="close_gripper",
            action="close_gripper",
            description="Close the FR5 gripper and rely on MuJoCo contact/friction to hold the target.",
            parameter_schema={},
            source_skill="skills.primitives.close_gripper_skill.CloseGripperSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="open_gripper",
            action="open_gripper",
            description="Open the FR5 gripper and release the physically held target.",
            parameter_schema={},
            source_skill="skills.primitives.open_gripper_skill.OpenGripperSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="lift",
            action="lift",
            description="Lift the grasped object vertically while keeping the gripper orientation downward.",
            parameter_schema={"lift_height": "required float meters. Vertical lift height from the detected object position."},
            parameter_ranges={"lift_height": (0.02, 0.20)},
            required_any=(("lift_height",),),
            source_skill="skills.primitives.lift_skill.LiftSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="move_lifted_object_to",
            action="move_lifted_object_to",
            description="Move the lifted object to a previously detected placement target while keeping the gripper closed.",
            parameter_schema={
                "target": "required detected placement target class/name. The executor reads x/y from the previous detect_object_pose result and uses fixed z/height internally.",
            },
            required_any=(("target",),),
            source_skill="skills.primitives.move_lifted_object_to_skill.MoveLiftedObjectToSkill",
        ),
        FR5FieldAtomicSkillSpec(
            name="go_home",
            action="go_home",
            description="Return FR5 to home posture.",
            parameter_schema={},
            source_skill="skills.primitives.go_home_skill.GoHomeSkill",
        ),
    ]
