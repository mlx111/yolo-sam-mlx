import argparse
import time

import mujoco
import numpy as np

from skills.primitives.approach_object_skill import load_skill as load_approach
from skills.primitives.move_to_pregrasp_skill import load_skill as load_pregrasp


def parse_args():
    parser = argparse.ArgumentParser(description="View only pregrasp and approach skills.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--object-body", default="target_cube")
    parser.add_argument("--approach-dx", type=float, default=0.0)
    parser.add_argument("--approach-dy", type=float, default=0.0)
    parser.add_argument("--approach-dz", type=float, default=-1.0)
    parser.add_argument("--pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.025)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=500)
    parser.add_argument("--fail-threshold", type=float, default=0.02)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def tcp_pos(model, data, side):
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_tcp")
    if site_id < 0:
        raise ValueError(f"site not found: {side}_hand_tcp")
    return data.site_xpos[site_id].copy()


def object_pos(model, data, body_name):
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body not found: {body_name}")
    return data.xpos[body_id].copy()


def make_params(args):
    return {
        "side": args.side,
        "object_body": args.object_body,
        "approach_dx": args.approach_dx,
        "approach_dy": args.approach_dy,
        "approach_dz": args.approach_dz,
        "pregrasp_distance": args.pregrasp_distance,
        "grasp_offset_x": args.grasp_offset_x,
        "grasp_offset_y": args.grasp_offset_y,
        "grasp_offset_z": args.grasp_offset_z,
        "steps": args.steps,
        "settle_steps": args.settle_steps,
        "fail_threshold": args.fail_threshold,
    }


def print_result(label, result, model, data, side):
    print(label)
    print("  target:", np.round(result.target_pos, 6))
    print("  tcp:", np.round(tcp_pos(model, data, side), 6))
    print("  error:", result.final_error)
    print("  success:", result.success)
    if result.arm_result is not None:
        print("  control_mode:", getattr(result.arm_result, "control_mode", "direct_qpos"))


def run(model, data, args, step_callback=None):
    params = make_params(args)
    print("initial_tcp:", np.round(tcp_pos(model, data, args.side), 6))
    print("object:", np.round(object_pos(model, data, args.object_body), 6))

    pregrasp = load_pregrasp()
    approach = load_approach()

    pre_result = pregrasp.execute_recovery_action(model, data, params, step_callback=step_callback)
    print_result("move_to_pregrasp", pre_result, model, data, args.side)
    if not pre_result.success:
        print("stop: move_to_pregrasp failed")
        return

    approach_result = approach.execute_recovery_action(model, data, params, step_callback=step_callback)
    print_result("approach_object", approach_result, model, data, args.side)


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
