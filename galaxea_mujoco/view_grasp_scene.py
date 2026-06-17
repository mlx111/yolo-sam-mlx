import argparse
import time

import mujoco
import numpy as np

from r1pro_control.grasp import R1ProContinuousGraspExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="Run continuous R1Pro grasp execution in MuJoCo.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--object-body", default="target_cube")
    parser.add_argument("--pregrasp-distance", type=float, default=0.12)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=0.0)
    parser.add_argument("--no-auto-grasp-table-clearance", action="store_true")
    parser.add_argument("--pad-table-clearance", type=float, default=0.010)
    parser.add_argument("--lift-height", type=float, default=0.30)
    parser.add_argument("--waypoints", type=int, default=20)
    parser.add_argument("--waypoint-steps", type=int, default=60)
    parser.add_argument("--fail-threshold", type=float, default=0.002)
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--lift-orientation-weight", type=float, default=0.0)
    parser.add_argument("--orientation-threshold", type=float, default=1.0)
    parser.add_argument("--topdown-mode", choices=["palm_down", "x_forward", "x_side", "current"], default="palm_down")
    parser.add_argument("--control-mode", choices=["actuator", "site_servo", "direct_qpos"], default="actuator")
    parser.add_argument("--control-frame", choices=["grasp_tool"], default="grasp_tool")
    parser.add_argument("--gripper-steps", type=int, default=500)
    parser.add_argument("--direct-gripper-qpos", action="store_true")
    parser.add_argument("--max-object-displacement-before-close", type=float, default=0.015)
    parser.add_argument("--max-approach-error-before-close", type=float, default=0.012)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def settle(model, data, steps):
    for _ in range(max(steps, 0)):
        mujoco.mj_step(model, data)


def print_result(result):
    print("success:", result.success)
    print("object_start:", np.round(result.object_start, 6))
    print("object_final:", np.round(result.object_final, 6))
    print("tcp_final:", np.round(result.tcp_final, 6))
    print("lift:", result.lift)
    print("stage_errors:", {k: round(v, 6) for k, v in result.stage_errors.items()})
    print("stage_orientation_errors:", {k: round(v, 6) for k, v in result.stage_orientation_errors.items()})
    print("message:", result.message)


def run_sequence(model, data, args, step_callback=None):
    settle(model, data, 1500)
    result = R1ProContinuousGraspExecutor().execute(
        model,
        data,
        side=args.side,
        object_body=args.object_body,
        grasp_offset=np.array([args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z], dtype=np.float64),
        auto_grasp_table_clearance=not args.no_auto_grasp_table_clearance,
        pad_table_clearance=args.pad_table_clearance,
        pregrasp_distance=args.pregrasp_distance,
        lift_height=args.lift_height,
        waypoint_count=args.waypoints,
        waypoint_steps=args.waypoint_steps,
        fail_threshold=args.fail_threshold,
        orientation_weight=args.orientation_weight,
        lift_orientation_weight=args.lift_orientation_weight,
        orientation_threshold=args.orientation_threshold,
        topdown_mode=args.topdown_mode,
        control_mode=args.control_mode,
        control_frame=args.control_frame,
        close_steps=args.gripper_steps,
        close_direct_qpos=args.direct_gripper_qpos,
        max_object_displacement_before_close=args.max_object_displacement_before_close,
        max_approach_error_before_close=args.max_approach_error_before_close,
        step_callback=step_callback,
    )
    print_result(result)
    return result


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    if args.headless:
        run_sequence(model, data, args)
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

        run_sequence(model, data, args, step_callback=sync_viewer)

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
