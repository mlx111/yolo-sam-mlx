# R1Pro Whole-Body Grasp Control Diagnosis

## Observed Issues

During live MuJoCo viewer inspection of `G3 clean physical`:

1. The robot had an unnatural whole-body motion before grasping.
2. The target object appears difficult to reach with the arm alone.
3. The pregrasp target does not visually behave like a clean top-down pregrasp over the object.

## Current Findings

The unnatural whole-body motion came from `pre_grasp_safe_posture`, which reset torso and both arms before grasp. This is useful as a sandbox seed reset, but it is not a realistic robot execution primitive. It is now disabled by default in `real_driver_like_v1`.

Experiments with `base_align_to_target` and `torso_turn_to_target` showed that simply adding base/torso motion before arm IK is not safe yet:

- `torso_turn_to_target` produced a large yaw request and made subsequent arm IK inconsistent.
- `torso_set_height` with direct posture must also synchronize actuator `ctrl`, otherwise later physical stepping pulls the torso back.
- Moving base/torso changes the kinematic frame, but the current arm IK path only partially accounts for mobile base and does not robustly account for torso yaw/whole-body state during target conversion and execution.

## Current Safe Default

The default physical profile is kept conservative:

- `pregrasp_safe_posture = false`
- `pregrasp_base_align = false`
- `pregrasp_torso_align = false`

This avoids defaulting to visually misleading whole-body behavior.

The successful part of the current physical path remains arm-level tracking after a stable initial posture:

- `move_to_pregrasp` can reach centimeter-level error in the stable arm-only path.
- `approach_object` can reach millimeter-level error in the stable arm-only path.

## Required Fix Before Enabling Whole-Body Grasp

To make whole-body grasp control credible, implement these before enabling base/torso prealignment by default:

1. Whole-body FK/IK frame consistency:
   - Convert world target into the current base and torso frame, not only base x/y/yaw.
   - Include torso joints in the IK seed and forward kinematics consistently.

2. Whole-body candidate scoring:
   - Score candidate base offset, torso height, torso yaw, and arm target together.
   - Reject candidates that increase arm IK error or joint-limit risk.

3. Physical actuator consistency:
   - Whenever direct initialization is used for torso/base, synchronize actuator `ctrl` to the same target.
   - Otherwise later `mj_step` pulls joints back toward stale actuator commands.

4. Visual top-down pregrasp constraint:
   - Keep pregrasp XY aligned to object XY unless an explicit side-grasp mode is selected.
   - Report `pregrasp_xy_error_to_object` and `pregrasp_height_above_object`.

5. Viewer validation:
   - Use `--viewer` to inspect G3 after each whole-body change.
   - Do not accept a change only because JSON metrics improve; the motion must be visually plausible.

## Suggested Next Implementation

Create a dedicated whole-body pregrasp planner:

```text
object pose
-> generate base/torso candidates
-> solve arm IK for each candidate
-> rank by IK error, joint margin, TCP/object top-down alignment, motion size
-> apply selected base/torso candidate
-> execute arm pregrasp/approach
```

Until that exists, whole-body prealignment should remain opt-in only.
