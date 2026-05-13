from __future__ import annotations

import argparse
import time

import numpy as np

from .r1_pro_reach import R1ProReachEnv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move the R1Pro right arm to a world coordinate in MuJoCo.",
    )
    parser.add_argument("--x", type=float, required=True, help="Target x in world coordinates.")
    parser.add_argument("--y", type=float, required=True, help="Target y in world coordinates.")
    parser.add_argument("--z", type=float, required=True, help="Target z in world coordinates.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Optional fixed yaw in radians.")
    parser.add_argument("--approach-height", type=float, default=0.18, help="Height above target for the pre-grasp pose.")
    parser.add_argument("--final-offset", type=float, default=0.05, help="Final offset above the target point.")
    parser.add_argument("--duration", type=float, default=1.2, help="Total motion duration in seconds.")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the interactive viewer.")
    parser.add_argument("--dry-run", action="store_true", help="Solve IK and print joint angles without stepping.")
    parser.add_argument("--skip-stand", action="store_true", help="Skip the initial standing pose.")
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=-1,
        help="If negative, keep the viewer open until you close it. If non-negative, wait that many sync ticks and exit.",
    )
    return parser


def _run_hold(env: R1ProReachEnv, hold_steps: int) -> None:
    if env.viewer is None:
        return

    if hold_steps >= 0:
        for _ in range(hold_steps):
            if not env.viewer.is_running():
                break
            env.viewer.sync()
            time.sleep(env.dt)
        return

    env.hold_until_closed()


def main() -> int:
    args = build_arg_parser().parse_args()
    target = np.array([args.x, args.y, args.z], dtype=float)

    env = R1ProReachEnv(enable_viewer=not args.no_viewer)
    try:
        env.reset()
        if not args.skip_stand and not args.dry_run:
            print("[follow] applying stable pose")
            env.apply_stable_pose(settle_seconds=1.0)

        ok, q_goal = env.move_to(
            side="right",
            target_xyz=target,
            approach_height=args.approach_height,
            final_offset=args.final_offset,
            duration=args.duration,
            yaw=args.yaw,
            execute=not args.dry_run,
        )
        if not ok or q_goal is None:
            print("[follow] IK failed for the requested target.")
            return 2

        motion_mode = env.last_move_mode or ("arm-only" if env.last_move_joint_names is not None and len(env.last_move_joint_names) == 7 else "torso+arm")
        print(f"[follow] motion_mode = {motion_mode}")
        env.print_state("right", env.last_move_joint_names)
        print(f"[follow] target = {np.array2string(target, precision=4, suppress_small=True)}")
        print(f"[follow] goal_q = {np.array2string(q_goal, precision=4, suppress_small=True)}")

        if not args.dry_run:
            _run_hold(env, int(args.hold_steps))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
