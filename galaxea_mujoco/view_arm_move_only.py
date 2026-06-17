import argparse
import time

import mujoco
import numpy as np

from skills.base.left_arm_move_skill import load_skill as load_left_arm
from skills.base.right_arm_move_skill import load_skill as load_right_arm


def parse_args():
    parser = argparse.ArgumentParser(description="View arm-only TCP move without torso fallback.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=None)
    parser.add_argument("--object-body", default=None)
    parser.add_argument("--object-offset-x", type=float, default=0.0)
    parser.add_argument("--object-offset-y", type=float, default=0.0)
    parser.add_argument("--object-offset-z", type=float, default=0.145)
    parser.add_argument("--dx", type=float, default=0.04)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.04)
    parser.add_argument(
        "--target-quat-wxyz",
        type=float,
        nargs=4,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Optional target TCP orientation quaternion in wxyz order.",
    )
    parser.add_argument("--orientation-weight", type=float, default=None)
    parser.add_argument("--orientation-threshold", type=float, default=1.0)
    parser.add_argument("--control-frame", choices=["grasp_tool"], default="grasp_tool")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--settle-steps", type=int, default=3000)
    parser.add_argument("--fail-threshold", type=float, default=0.02)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def load_arm(side):
    return load_left_arm() if side == "left" else load_right_arm()


def tcp_pos(model, data, side):
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_tcp")
    if site_id < 0:
        raise ValueError(f"site not found: {side}_hand_tcp")
    return data.site_xpos[site_id].copy()


def site_pos(model, data, side, frame):
    mujoco.mj_forward(model, data)
    if frame != "grasp_tool":
        raise ValueError(f"Unsupported control frame: {frame!r}; use 'grasp_tool'")
    site_name = f"{side}_grasp_tool"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"site not found: {site_name}")
    return data.site_xpos[site_id].copy()


def body_pos(model, data, body_name):
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body not found: {body_name}")
    return data.xpos[body_id].copy()


def target_pos(model, data, args):
    if args.object_body is not None:
        return body_pos(model, data, args.object_body) + np.array(
            [args.object_offset_x, args.object_offset_y, args.object_offset_z],
            dtype=np.float64,
        )
    if None in (args.target_x, args.target_y, args.target_z):
        start = site_pos(model, data, args.side, args.control_frame)
        return start + np.array([args.dx, args.dy, args.dz], dtype=np.float64)
    return np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)


def print_result(result, model, data, side):
    print("side:", result.side)
    print("frame:", result.frame_name)
    print("control_mode:", result.control_mode)
    print("target_pos:", np.round(result.target_pos, 6))
    print("ik_pos:", np.round(result.ik_pos, 6))
    print("final_site_pos:", np.round(result.final_site_pos, 6))
    print("actual_tcp:", np.round(tcp_pos(model, data, side), 6))
    print("ik_error:", result.ik_error)
    print("final_error:", result.final_error)
    if hasattr(result, "final_orientation_error"):
        print("final_orientation_error:", result.final_orientation_error)
    print("success:", result.success)


def run(model, data, args, step_callback=None):
    arm = load_arm(args.side)
    target = target_pos(model, data, args)
    print("initial_tcp:", np.round(tcp_pos(model, data, args.side), 6))
    print("initial_control_site:", np.round(site_pos(model, data, args.side, args.control_frame), 6))
    print("target:", np.round(target, 6))
    result = arm.move_to_pose(
        model,
        data,
        target,
        target_quat_wxyz=args.target_quat_wxyz,
        steps=args.steps,
        settle_steps=args.settle_steps,
        fail_threshold=args.fail_threshold,
        orientation_weight=args.orientation_weight,
        orientation_threshold=args.orientation_threshold,
        control_frame=args.control_frame,
        step_callback=step_callback,
    )
    print_result(result, model, data, args.side)


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    if args.headless:
        run(model, data, args)
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        def sync_viewer():
            if viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)

        start = time.time()
        while viewer.is_running() and time.time() - start < args.delay:
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        run(model, data, args, step_callback=sync_viewer)

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
