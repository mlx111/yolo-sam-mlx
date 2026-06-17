import argparse
import time

import mujoco
import numpy as np

from skills.base.arm_ik_skill import load_skill


def parse_args():
    parser = argparse.ArgumentParser(description="Move an R1Pro hand TCP to a target position.")
    parser.add_argument("--model", default="model.xml")
    parser.add_argument("--urdf", default="urdf/r1_pro_with_gripper.urdf")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--direct-qpos", action="store_true")
    parser.add_argument("--open-loop", action="store_true", help="Use one-shot IK joint trajectory instead of closed-loop IK.")
    parser.add_argument(
        "--cartesian-closed-loop",
        action="store_true",
        help="Debug mode: re-solve Cartesian IK every step. This can saturate joints on large moves.",
    )
    parser.add_argument("--no-stabilize", action="store_true", help="Do not add runtime damping/armature.")
    parser.add_argument("--no-lock-posture", action="store_true", help="Allow non-target joints to move under dynamics.")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=None)
    parser.add_argument("--dx", type=float, default=0.04)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.04)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--settle-steps", type=int, default=3000)
    parser.add_argument("--max-joint-step", type=float, default=0.006)
    parser.add_argument("--fail-threshold", type=float, default=0.02)
    parser.add_argument(
        "--force-scale",
        type=float,
        default=1.0,
        help="Runtime multiplier for controlled arm actuator force ranges. Use >1 only for stress tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    arm_skill = load_skill(args.urdf)

    q0 = arm_skill.sync_q_from_mujoco(model, data)
    start_pos = arm_skill.current_tcp_position(q0, args.side)
    if args.target_x is None and args.target_y is None and args.target_z is None:
        target_pos = start_pos + np.array([args.dx, args.dy, args.dz], dtype=np.float64)
    elif None in (args.target_x, args.target_y, args.target_z):
        raise ValueError("Use --target-x, --target-y, and --target-z together.")
    else:
        target_pos = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)

    result = arm_skill.move_to_position(
        model,
        data,
        args.side,
        target_pos,
        steps=args.steps,
        settle_steps=args.settle_steps,
        direct_qpos=args.direct_qpos,
        stabilize=not args.no_stabilize,
        closed_loop=not args.open_loop,
        cartesian_closed_loop=args.cartesian_closed_loop,
        lock_posture=not args.no_lock_posture,
        max_joint_step=args.max_joint_step,
        fail_threshold=args.fail_threshold,
        force_scale=args.force_scale,
    )

    print("side:", result.side)
    print("frame:", result.frame_name)
    print("control_mode:", result.control_mode)
    print("start_pos:", np.round(result.start_pos, 6))
    print("target_pos:", np.round(result.target_pos, 6))
    print("ik_pos:", np.round(result.ik_pos, 6))
    print("final_site_pos:", np.round(result.final_site_pos, 6))
    print("ik_error:", result.ik_error)
    print("final_error:", result.final_error)
    print("success:", result.success)

    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            locked = arm_skill.capture_locked_posture(model, data, args.side) if not args.no_lock_posture else {}
            while viewer.is_running():
                if args.direct_qpos:
                    arm_skill.apply_locked_posture(model, data, locked)
                    mujoco.mj_forward(model, data)
                elif args.cartesian_closed_loop:
                    q_current = arm_skill.sync_q_from_mujoco(model, data)
                    q_command, _, _ = arm_skill.solve_position_ik(
                        args.side,
                        q_current,
                        target_pos,
                        iterations=40,
                        tolerance=1e-4,
                    )
                    arm_skill.set_arm_ctrl(model, data, args.side, q_command)
                    mujoco.mj_step(model, data)
                    arm_skill.apply_locked_posture(model, data, locked)
                    mujoco.mj_forward(model, data)
                elif not args.open_loop:
                    q_target_values = arm_skill.arm_joint_values_from_pin_q(args.side, result.q_target)
                    arm_skill.set_arm_ctrl_values(model, data, args.side, q_target_values)
                    mujoco.mj_step(model, data)
                    arm_skill.apply_locked_posture(model, data, locked)
                    mujoco.mj_forward(model, data)
                else:
                    arm_skill.set_arm_ctrl(model, data, args.side, result.q_target)
                    mujoco.mj_step(model, data)
                    arm_skill.apply_locked_posture(model, data, locked)
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
