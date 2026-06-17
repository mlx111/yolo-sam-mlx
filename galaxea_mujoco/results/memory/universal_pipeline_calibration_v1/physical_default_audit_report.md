# Physical Default Audit Report

- File count: 26
- direct_qpos=true total: 0
- direct_qpos=false total: 20

| File | direct_qpos=true count | direct_qpos=false count |
|---|---:|---:|
| skills/base/base_motion_skill.json | 0 | 1 |
| skills/base/torso_move_skill.json | 0 | 1 |
| skills/base/left_arm_move_skill.json | 0 | 1 |
| skills/base/right_arm_move_skill.json | 0 | 1 |
| skills/primitives/left_gripper_close_skill.json | 0 | 1 |
| skills/primitives/right_gripper_close_skill.json | 0 | 1 |
| skills/primitives/left_gripper_open_skill.json | 0 | 1 |
| skills/primitives/right_gripper_open_skill.json | 0 | 1 |
| skills/primitives/open_gripper_release_skill.json | 0 | 1 |
| skills/primitives/resync_grippers_skill.json | 0 | 1 |
| skills/primitives/base_move_to_region_skill.json | 0 | 1 |
| skills/primitives/base_reposition_lateral_skill.json | 0 | 1 |
| skills/primitives/base_replan_path_skill.json | 0 | 1 |
| skills/primitives/torso_turn_to_target_skill.json | 0 | 1 |
| skills/primitives/torso_set_height_skill.json | 0 | 1 |
| skills/primitives/pre_grasp_safe_posture_skill.json | 0 | 1 |
| skills/primitives/safe_transport_pose_skill.json | 0 | 1 |
| skills/primitives/go_home_upper_body_skill.json | 0 | 1 |
| skills/primitives/grasp_handle_skill.json | 0 | 1 |
| skills/primitives/regrasp_deeper_skill.json | 0 | 1 |
| skills/primitives/object_manipulation_skills.py | 0 | 0 |
| skills/base/arm_ik_skill.py | 0 | 0 |
| skills/base/base_motion_skill.py | 0 | 0 |
| skills/base/torso_move_skill.py | 0 | 0 |
| skills/base/gripper_skill.py | 0 | 0 |
| skills/primitives/recovery_skills.py | 0 | 0 |

## Paper Wording

- Safe claim: Core base, torso, arm, gripper, and field atomic defaults have been audited to prefer physical control over direct-qpos shortcuts.
- Avoid claim: Do not claim that every explicit debug or test path is physical-only; the audit concerns defaults, not every possible parameter override.
