import argparse

import mujoco
import numpy as np

from skills.primitives.check_collision_skill import load_skill as load_check_collision
from skills.primitives.check_slip_skill import load_skill as load_check_slip
from skills.primitives.recover_from_contact_skill import load_skill as load_recover_contact
from skills.primitives.regrasp_deeper_skill import load_skill as load_regrasp_deeper
from skills.primitives.stop_lift_skill import load_skill as load_stop_lift


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test Phase 4 recovery and safety skills.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--object-body", default="target_cube")
    return parser.parse_args()


def print_result(label, result):
    print(f"{label}: success={result.success}, detected={result.detected}, message={result.message}")
    if result.target_pos is not None:
        print("  target_pos:", np.round(result.target_pos, 6).tolist())
    if result.final_error is not None:
        print("  final_error:", round(result.final_error, 6))
    if result.contacts:
        print("  contacts:", result.contacts[:5])


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    check_collision = load_check_collision()
    collision = check_collision.execute_recovery_action(model, data, {})
    print_result("check_collision", collision)

    stop_lift = load_stop_lift()
    stopped = stop_lift.execute_recovery_action(model, data, {"side": args.side, "settle_steps": 10})
    print_result("stop_lift", stopped)

    check_slip = load_check_slip()
    slip = check_slip.execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "object_body": args.object_body,
            "previous_object_z": 0.78,
            "max_drop": 0.05,
            "max_tcp_distance": 10.0,
        },
    )
    print_result("check_slip", slip)

    recover = load_recover_contact()
    recovered = recover.execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "retreat_dx": 0.0,
            "retreat_dy": 0.0,
            "retreat_dz": 0.02,
            "servo_steps": 20,
            "settle_steps": 5,
            "fail_threshold": 1.0,
        },
    )
    print_result("recover_from_contact", recovered)

    regrasp = load_regrasp_deeper()
    deeper = regrasp.execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "approach_dx": 0.0,
            "approach_dy": 0.0,
            "approach_dz": -1.0,
            "deeper_distance": 0.005,
            "servo_steps": 20,
            "settle_steps": 5,
            "gripper_steps": 10,
            "fail_threshold": 1.0,
        },
    )
    print_result("regrasp_deeper", deeper)


if __name__ == "__main__":
    main()
