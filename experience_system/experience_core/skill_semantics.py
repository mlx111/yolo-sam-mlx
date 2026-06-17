"""Skill precondition/effect semantics for recovery-plan validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillSemantics:
    name: str
    requires: frozenset[str] = field(default_factory=frozenset)
    effects: frozenset[str] = field(default_factory=frozenset)
    consumes: frozenset[str] = field(default_factory=frozenset)
    optional_requires: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requires": sorted(self.requires),
            "effects": sorted(self.effects),
            "consumes": sorted(self.consumes),
            "optional_requires": sorted(self.optional_requires),
            "description": self.description,
        }


def _semantics(
    name: str,
    *,
    requires: list[str] | None = None,
    effects: list[str] | None = None,
    consumes: list[str] | None = None,
    optional_requires: list[str] | None = None,
    description: str = "",
) -> SkillSemantics:
    return SkillSemantics(
        name=name,
        requires=frozenset(requires or []),
        effects=frozenset(effects or []),
        consumes=frozenset(consumes or []),
        optional_requires=frozenset(optional_requires or []),
        description=description,
    )


def default_r1pro_skill_semantics() -> dict[str, SkillSemantics]:
    """Return skill semantics independent of any fixed G3/G4 scenario order."""

    entries = [
        _semantics(
            "detect_multiple_objects",
            effects=["object_candidates_known"],
            description="Detect candidate objects for manipulation.",
        ),
        _semantics(
            "select_correct_object",
            requires=["object_candidates_known"],
            effects=["target_object_selected"],
            description="Select the target object from detected candidates.",
        ),
        _semantics(
            "move_to_pregrasp",
            requires=["target_object_selected"],
            effects=["single_arm_pregrasp_reached"],
            description="Move one arm to a pre-grasp pose.",
        ),
        _semantics(
            "approach_object",
            requires=["single_arm_pregrasp_reached"],
            effects=["single_arm_grasp_pose_reached"],
            description="Approach the selected object before closing the gripper.",
        ),
        _semantics(
            "left_gripper_close",
            requires=["single_arm_grasp_pose_reached"],
            effects=["object_grasped", "left_gripper_closed"],
            description="Close the left gripper on the selected object.",
        ),
        _semantics(
            "verify_grasp",
            requires=["object_grasped"],
            effects=["grasp_verified"],
            description="Check that the grasp is still valid.",
        ),
        _semantics(
            "left_vertical_lift",
            requires=["object_grasped"],
            effects=["object_lifted"],
            description="Lift the grasped object vertically.",
        ),
        _semantics(
            "reposition_base_for_reach",
            requires=["target_object_selected"],
            effects=["base_repositioned_for_reach"],
            description="Move the base slightly to improve single-arm reachability.",
        ),
        _semantics(
            "adjust_torso_for_reach",
            requires=["target_object_selected"],
            effects=["torso_adjusted_for_reach"],
            description="Adjust torso height/yaw toward the selected target.",
        ),
        _semantics(
            "retry_pregrasp_with_safer_offset",
            requires=["target_object_selected"],
            effects=["single_arm_pregrasp_reached"],
            optional_requires=["base_repositioned_for_reach", "torso_adjusted_for_reach"],
            description="Retry pregrasp after moving to a safer posture and offset.",
        ),
        _semantics(
            "slow_cartesian_approach",
            requires=["single_arm_pregrasp_reached"],
            effects=["single_arm_grasp_pose_reached"],
            description="Approach the target with slower segmented Cartesian motion.",
        ),
        _semantics(
            "recover_from_joint_limit",
            requires=["target_object_selected"],
            effects=["base_repositioned_for_reach", "torso_adjusted_for_reach", "single_arm_pregrasp_reached"],
            description="Composite recovery for joint-limit or reach failures before grasp.",
        ),
        _semantics(
            "retry_lift_after_grasp_check",
            requires=["object_grasped"],
            effects=["grasp_verified", "object_lifted"],
            description="Verify grasp and retry a conservative lift.",
        ),
        _semantics(
            "detect_place_occupancy",
            effects=["place_occupancy_known"],
            description="Sense whether the nominal place region is occupied.",
        ),
        _semantics(
            "choose_alternate_place",
            effects=["place_site_selected"],
            optional_requires=["place_occupancy_known"],
            description="Choose a usable place site.",
        ),
        _semantics(
            "place_object",
            requires=["object_lifted", "place_site_selected"],
            effects=["object_placed"],
            consumes=["object_grasped", "object_lifted"],
            description="Place the grasped object at the selected site.",
        ),
        _semantics(
            "open_gripper_release",
            requires=["object_placed"],
            effects=["object_released"],
            consumes=["left_gripper_closed"],
            description="Open the gripper after placing the object.",
        ),
        _semantics(
            "base_move_to_region",
            effects=["base_at_pick_region"],
            description="Move the mobile base to the pick region.",
        ),
        _semantics(
            "torso_set_height",
            effects=["torso_height_set"],
            description="Set torso height for dual-arm manipulation.",
        ),
        _semantics(
            "dual_arm_pregrasp",
            requires=["base_at_pick_region"],
            effects=["dual_arm_pregrasp_reached"],
            optional_requires=["torso_height_set"],
            description="Move both arms to dual-arm pre-grasp poses.",
        ),
        _semantics(
            "dual_arm_approach",
            requires=["dual_arm_pregrasp_reached"],
            effects=["dual_arm_grasp_pose_reached"],
            description="Approach the object with both arms.",
        ),
        _semantics(
            "dual_gripper_close",
            requires=["dual_arm_grasp_pose_reached"],
            effects=["dual_object_grasped", "dual_grippers_closed"],
            description="Close both grippers on the object.",
        ),
        _semantics(
            "dual_arm_synchronized_lift",
            requires=["dual_object_grasped"],
            effects=["dual_object_lifted"],
            description="Lift the object with both arms.",
        ),
        _semantics(
            "dual_arm_level_object",
            requires=["dual_object_grasped"],
            effects=["dual_object_leveled"],
            description="Level the object while it is held by both arms.",
        ),
        _semantics(
            "safe_transport_pose",
            requires=["dual_object_lifted"],
            effects=["transport_pose_safe"],
            description="Move to a safer pose before transport.",
        ),
        _semantics(
            "segmented_transport",
            requires=["dual_object_lifted"],
            effects=["object_transported"],
            optional_requires=["transport_pose_safe", "dual_object_leveled"],
            description="Transport the lifted object in controlled segments.",
        ),
        _semantics(
            "segmented_transport_fast",
            requires=["dual_object_lifted"],
            effects=["object_transported"],
            optional_requires=["dual_object_leveled"],
            description="Transport the lifted object with fewer segments; sandbox critic should check speed/slip risk.",
        ),
        _semantics(
            "base_move_to_place",
            requires=["place_site_selected"],
            effects=["base_at_place_region"],
            description="Move the base to the selected place site.",
        ),
        _semantics(
            "dual_arm_place",
            requires=["dual_object_lifted", "place_site_selected"],
            effects=["object_placed"],
            optional_requires=["base_at_place_region", "object_transported", "dual_object_leveled"],
            consumes=["dual_object_lifted"],
            description="Place the dual-arm object at the selected site.",
        ),
        _semantics(
            "dual_gripper_release",
            requires=["object_placed"],
            effects=["object_released"],
            consumes=["dual_object_grasped", "dual_grippers_closed"],
            description="Release the object after dual-arm placement.",
        ),
        _semantics(
            "verify_place_zone",
            requires=["object_placed"],
            effects=["place_verified"],
            description="Verify the object is in the place zone.",
        ),
        _semantics(
            "head_camera_capture",
            effects=["visual_observation_available"],
            description="Capture image evidence from the head-mounted RGB-D camera.",
        ),
        _semantics(
            "base_lidar_scan",
            effects=["lidar_observation_available"],
            description="Capture base LiDAR evidence for nearby obstacles.",
        ),
        _semantics(
            "base_move_to_pose",
            effects=["base_pose_commanded"],
            optional_requires=["lidar_observation_available"],
            description="Move the mobile base to an explicit pose generated by the planner.",
        ),
        _semantics(
            "torso_move_to_posture",
            effects=["torso_posture_commanded"],
            description="Move torso joints to an explicit posture generated by the planner.",
        ),
        _semantics(
            "left_arm_move_to_position",
            effects=["left_arm_position_commanded"],
            optional_requires=["visual_observation_available", "base_pose_commanded", "torso_posture_commanded"],
            description="Move the left arm TCP to an explicit position generated by the planner.",
        ),
        _semantics(
            "right_arm_move_to_position",
            effects=["right_arm_position_commanded"],
            optional_requires=["visual_observation_available", "base_pose_commanded", "torso_posture_commanded"],
            description="Move the right arm TCP to an explicit position generated by the planner.",
        ),
        _semantics(
            "left_gripper_set",
            effects=["left_gripper_commanded"],
            optional_requires=["left_arm_position_commanded"],
            description="Open or close the left gripper using an explicit planner command.",
        ),
        _semantics(
            "right_gripper_set",
            effects=["right_gripper_commanded"],
            optional_requires=["right_arm_position_commanded"],
            description="Open or close the right gripper using an explicit planner command.",
        ),
    ]
    return {entry.name: entry for entry in entries}


def validate_skill_semantic_plan(
    plan: dict[str, Any],
    *,
    skill_semantics: dict[str, SkillSemantics] | None = None,
    initial_facts: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a plan by simulating skill preconditions/effects over facts."""

    semantics = skill_semantics or default_r1pro_skill_semantics()
    facts = set(initial_facts or [])
    steps = plan.get("steps") if isinstance(plan, dict) else []
    actions = [str(step.get("action") or "") for step in steps or [] if isinstance(step, dict)]
    issues: list[dict[str, Any]] = []
    fact_trace: list[dict[str, Any]] = []

    if not actions:
        issues.append({
            "severity": "fatal",
            "code": "empty_plan",
            "message": "plan has no executable actions",
        })

    for index, action in enumerate(actions):
        semantic = semantics.get(action)
        before = sorted(facts)
        if semantic is None:
            issues.append({
                "severity": "warning",
                "code": "missing_skill_semantics",
                "message": f"no precondition/effect semantics registered for {action}; only executor support can be checked",
                "action": action,
                "index": index,
            })
            fact_trace.append({
                "index": index,
                "action": action,
                "facts_before": before,
                "missing_requires": [],
                "facts_after": sorted(facts),
            })
            continue

        missing = sorted(semantic.requires - facts)
        if missing:
            issues.append({
                "severity": "fatal",
                "code": "missing_precondition",
                "message": f"{action} requires missing facts: {', '.join(missing)}",
                "action": action,
                "index": index,
                "missing_requires": missing,
            })
        missing_optional = sorted(semantic.optional_requires - facts)
        if missing_optional:
            issues.append({
                "severity": "warning",
                "code": "missing_optional_precondition",
                "message": f"{action} can run, but missing optional facts: {', '.join(missing_optional)}",
                "action": action,
                "index": index,
                "missing_optional_requires": missing_optional,
            })
        if index > 0 and actions[index - 1] == action:
            issues.append({
                "severity": "warning",
                "code": "consecutive_duplicate_step",
                "message": f"{action} appears consecutively; remove duplicate unless intentional",
                "action": action,
                "index": index,
            })

        facts.difference_update(semantic.consumes)
        facts.update(semantic.effects)
        fact_trace.append({
            "index": index,
            "action": action,
            "facts_before": before,
            "missing_requires": missing,
            "effects": sorted(semantic.effects),
            "consumes": sorted(semantic.consumes),
            "facts_after": sorted(facts),
        })

    fatal_count = sum(1 for issue in issues if issue.get("severity") == "fatal")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return {
        "schema_version": "skill_semantic_plan_validation_v1",
        "status": "fail" if fatal_count else "warn" if warning_count else "pass",
        "action_count": len(actions),
        "fatal_count": fatal_count,
        "warning_count": warning_count,
        "issues": issues,
        "final_facts": sorted(facts),
        "fact_trace": fact_trace,
        "registered_skill_count": len(semantics),
    }
