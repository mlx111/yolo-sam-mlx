from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from r1pro_control.arm import R1ProMuJoCoSiteServo
from r1pro_control.env import R1ProEnv


def parse_args():
    parser = argparse.ArgumentParser(description="View MuJoCo-native R1Pro TCP site servo.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=None)
    parser.add_argument("--object-body", default=None)
    parser.add_argument("--object-offset-x", type=float, default=0.0)
    parser.add_argument("--object-offset-y", type=float, default=0.0)
    parser.add_argument("--object-offset-z", type=float, default=0.145)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--fail-threshold", type=float, default=0.02)
    parser.add_argument("--max-cart-step", type=float, default=0.006)
    parser.add_argument("--max-joint-step", type=float, default=0.012)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def site_pos(model, data, side):
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_tcp")
    if site_id < 0:
        raise ValueError(f"site not found: {side}_hand_tcp")
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
        raise ValueError("Use either --object-body or --target-x/--target-y/--target-z")
    return np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)


def print_result(result):
    print("side:", result.side)
    print("site:", result.site_name)
    print("start_pos:", np.round(result.start_pos, 6))
    print("target_pos:", np.round(result.target_pos, 6))
    print("final_pos:", np.round(result.final_pos, 6))
    print("final_error:", result.final_error)
    print("min_error:", result.min_error)
    print("success:", result.success)
    print("stalled:", result.stalled)
    print("steps_run:", result.steps_run)
    if result.contacts:
        print("contacts:")
        for contact in result.contacts:
            print(" ", contact)


def run(model, data, args, step_callback=None):
    servo = R1ProMuJoCoSiteServo()
    target = target_pos(model, data, args)
    print("initial_tcp:", np.round(site_pos(model, data, args.side), 6))
    print("target:", np.round(target, 6))
    result = servo.move_to_position(
        model,
        data,
        args.side,
        target,
        steps=args.steps,
        fail_threshold=args.fail_threshold,
        max_cart_step=args.max_cart_step,
        max_joint_step=args.max_joint_step,
        step_callback=step_callback,
    )
    print_result(result)


def main():
    args = parse_args()
    env = R1ProEnv(args.model)
    model, data = env.reset()

    if args.headless:
        run(model, data, args)
        return

    with env.launch_viewer() as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < args.delay:
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        def sync_viewer():
            if viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)

        run(model, data, args, step_callback=sync_viewer)

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
