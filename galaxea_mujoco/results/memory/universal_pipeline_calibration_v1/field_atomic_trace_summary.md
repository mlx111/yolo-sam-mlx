# Field Atomic Trace Summary

This report aggregates field_atomic execution traces for debugging and paper evidence.

## Summary

- Inputs: 2
- Traces: 10
- Actions: 5
- Success / failure: 10 / 0
- Action kinds: `{"base": 2, "gripper": 2, "sensor": 4, "torso": 2}`
- direct_qpos true / false: 0 / 10
- Final error: `{"count": 4, "max": 0.002026, "mean": 0.001998, "median": 0.001998, "min": 0.001971}`

## By Action

| Action | Kind | Count | Success | Failure | Final error | Gripper command | direct_qpos true/false | Control modes |
|---|---|---:|---:|---:|---|---|---|---|
| base_lidar_scan | sensor | 2 | 2 | 0 | `{}` | `{}` | 0/2 | `{"unknown": 2}` |
| base_move_to_pose | base | 2 | 2 | 0 | `{"count": 2, "max": 0.001971, "mean": 0.001971, "median": 0.001971, "min": 0.001971}` | `{}` | 0/2 | `{"unknown": 2}` |
| head_camera_capture | sensor | 2 | 2 | 0 | `{}` | `{}` | 0/2 | `{"unknown": 2}` |
| left_gripper_set | gripper | 2 | 2 | 0 | `{}` | `{"count": 2, "max": 0.025, "mean": 0.025, "median": 0.025, "min": 0.025}` | 0/2 | `{"unknown": 2}` |
| torso_move_to_posture | torso | 2 | 2 | 0 | `{"count": 2, "max": 0.002026, "mean": 0.002026, "median": 0.002026, "min": 0.002026}` | `{}` | 0/2 | `{"actuator_joint_servo": 2}` |

## Paper Wording

- Safe claim: Field-atomic executions expose action-level trace summaries, including final tracking error, gripper command, control mode, and direct-qpos usage for onsite debugging.
- Avoid claim: Do not claim these summaries prove real-robot tracking accuracy; they summarize MuJoCo or stored experience traces only.
