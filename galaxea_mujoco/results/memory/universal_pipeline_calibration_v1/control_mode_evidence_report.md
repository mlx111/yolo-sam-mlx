# Control Mode Evidence Report

This report compares direct-position and physical-actuator sandbox execution without requiring a real robot.

## Summary

- Rollouts: 4
- Completed runners: 4
- Successful task chains: 2
- Task-success count: 2
- Execution types: `{"direct_position": 2, "physical_actuator": 2}`
- Failed skills: `{"approach_object": 1, "detect_place_occupancy": 1, "dual_arm_approach": 1, "dual_arm_level_object": 2, "dual_arm_place": 1, "dual_arm_pregrasp": 1, "dual_arm_synchronized_lift": 1, "left_vertical_lift": 1, "move_to_pregrasp": 1, "place_object": 1, "segmented_transport": 1, "verify_grasp": 1, "verify_place_zone": 2}`

## Rollouts

| Scenario | Condition | Candidate | Mode | Execution | Success | Task success | Failure | Failed skills |
|---|---|---|---|---|---:|---:|---|---|
| G3 | clean | g3_cautious_place | ideal | direct_position | True | True |  |  |
| G3 | clean | g3_cautious_place | physical | physical_actuator | False | False | move_to_pregrasp,approach_object,verify_grasp,left_vertical_lift,place_object,verify_place_zone | move_to_pregrasp, approach_object, verify_grasp, left_vertical_lift, place_object, verify_place_zone |
| G4 | place_occupied | g4_relevel_before_place | ideal | direct_position | True | True |  | detect_place_occupancy |
| G4 | place_occupied | g4_relevel_before_place | physical | physical_actuator | False | False | dual_arm_pregrasp,dual_arm_approach,dual_arm_synchronized_lift,dual_arm_level_object,segmented_transport,dual_arm_level_object,dual_arm_place,verify_place_zone | dual_arm_pregrasp, dual_arm_approach, dual_arm_synchronized_lift, dual_arm_level_object, segmented_transport, dual_arm_level_object, dual_arm_place, verify_place_zone |

## Paper Wording

- Safe claim: The implementation supports separate direct-position and physical-actuator sandbox execution modes, and reports which mode was used together with task outcome, failed skills, and motion/contact critic evidence.
- Avoid claim: Do not claim the physical-actuator sandbox matches real robot IK/control without real FK/IK alignment and response calibration data.
