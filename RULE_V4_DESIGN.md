# V4 Rule Detector

V4 uses a two-stage decision flow:

1. Rule detector runs first using MuJoCo truth state.
2. VLM recheck runs only for `UNCERTAIN` or `FAILURE_SOFT`.

Hard failures never go through the VLM recheck path.

Tracked metrics:

- `expected_pos_error`
- `expected_rot_error_deg`
- `target_motion`
- `post_relative_error`
- `target_lift_dz`
- `place_xy_error`
- `home_joint_error`
- `left_contact`
- `right_contact`
- `gripper_signal`

Default thresholds are implemented in `rule_action_detector_v4.py`.
