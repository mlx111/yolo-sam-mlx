import argparse
import time

import mujoco
import numpy as np

from r1pro_control.grasp import R1ProContinuousGraspExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="View continuous R1Pro grasp execution.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--object-body", default="target_cube")
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--lift-height", type=float, default=0.30)
    parser.add_argument("--waypoints", type=int, default=20)
    parser.add_argument("--waypoint-steps", type=int, default=60)
    parser.add_argument("--fail-threshold", type=float, default=0.025)
    parser.add_argument("--orientation-weight", type=float, default=0.02)
    parser.add_argument("--lift-orientation-weight", type=float, default=0.0)
    parser.add_argument("--orientation-threshold", type=float, default=1.0)
    parser.add_argument("--topdown-mode", choices=["palm_down", "x_forward", "x_side", "current"], default="palm_down")
    parser.add_argument("--direct-gripper-qpos", action="store_true", default=True)
    parser.add_argument("--actuator-gripper", action="store_true")
    parser.add_argument("--max-object-displacement-before-close", type=float, default=0.015)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def settle(model, data, steps):
    for _ in range(max(steps, 0)):
        mujoco.mj_step(model, data)


def run(model, data, args, step_callback=None):
    settle(model, data, 1500)
    executor = R1ProContinuousGraspExecutor()
    result = executor.execute(
        model,
        data,
        side=args.side,
        object_body=args.object_body,
        grasp_offset=np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=float),
        pregrasp_distance=args.pregrasp_distance,
        lift_height=args.lift_height,
        waypoint_count=args.waypoints,
        waypoint_steps=args.waypoint_steps,
        fail_threshold=args.fail_threshold,
        orientation_weight=args.orientation_weight,
        lift_orientation_weight=args.lift_orientation_weight,
        orientation_threshold=args.orientation_threshold,
        topdown_mode=args.topdown_mode,
        close_direct_qpos=args.direct_gripper_qpos and not args.actuator_gripper,
        max_object_displacement_before_close=args.max_object_displacement_before_close,
        step_callback=step_callback,
    )
    print("success:", result.success)
    print("object_start:", np.round(result.object_start, 6))
    print("object_final:", np.round(result.object_final, 6))
    print("tcp_final:", np.round(result.tcp_final, 6))
    print("lift:", result.lift)
    print("stage_errors:", {k: round(v, 6) for k, v in result.stage_errors.items()})
    print("message:", result.message)


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
