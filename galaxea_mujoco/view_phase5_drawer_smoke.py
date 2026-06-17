import argparse

import mujoco
import numpy as np

from skills.base.base_motion_skill import load_skill as load_base_motion
from skills.primitives.detect_handle_skill import load_skill as load_detect_handle
from skills.primitives.extract_object_from_drawer_skill import load_skill as load_extract_object
from skills.primitives.grasp_handle_skill import load_skill as load_grasp_handle
from skills.primitives.infer_hinge_or_slide_direction_skill import load_skill as load_infer_direction
from skills.primitives.insert_hand_into_drawer_skill import load_skill as load_insert_hand
from skills.primitives.pull_drawer_skill import load_skill as load_pull_drawer
from skills.primitives.push_or_pull_door_skill import load_skill as load_push_or_pull
from skills.primitives.redetect_handle_skill import load_skill as load_redetect_handle
from skills.primitives.torso_set_height_skill import load_skill as load_torso_height
from skills.primitives.verify_drawer_open_skill import load_skill as load_verify_drawer


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test Phase 5 drawer state and actuation skills.")
    parser.add_argument("--model", default="r1pro_drawer_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--target-open", type=float, default=0.25)
    return parser.parse_args()


def print_result(label, result):
    print(f"{label}: success={result.success}, message={result.message}")
    if result.position is not None:
        print("  position:", np.round(result.position, 6).tolist())
    if result.joint_qpos is not None:
        print("  joint_qpos:", round(result.joint_qpos, 6))
    if result.joint_axis is not None:
        print("  joint_axis:", np.round(result.joint_axis, 6).tolist())
    if result.joint_type is not None:
        print("  joint_type:", result.joint_type)


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base = load_base_motion()
    base.move_to_pose(model, data, [-0.45, 0.0, 0.0], steps=1, settle_steps=0, direct_qpos=True)

    torso = load_torso_height()
    torso.execute_recovery_action(
        model,
        data,
        {"height_level": "high", "steps": 1, "settle_steps": 0, "direct_qpos": True},
    )

    detect = load_detect_handle()
    detected = detect.execute_recovery_action(model, data, {})
    print_result("detect_handle", detected)

    infer = load_infer_direction()
    inferred = infer.execute_recovery_action(model, data, {})
    print_result("infer_hinge_or_slide_direction", inferred)

    verify = load_verify_drawer()
    closed_check = verify.execute_recovery_action(model, data, {"min_open": 0.18})
    print_result("verify_drawer_open_before_pull", closed_check)

    pull = load_pull_drawer()
    pulled = pull.execute_recovery_action(
        model,
        data,
        {"target_open": args.target_open, "steps": 300, "settle_steps": 80},
    )
    print_result("pull_drawer", pulled)

    push_or_pull = load_push_or_pull()
    pushed = push_or_pull.execute_recovery_action(
        model,
        data,
        {"target_open": args.target_open, "steps": 120, "settle_steps": 40},
    )
    print_result("push_or_pull_door", pushed)

    redetect = load_redetect_handle()
    redetected = redetect.execute_recovery_action(model, data, {})
    print_result("redetect_handle", redetected)

    open_check = verify.execute_recovery_action(model, data, {"min_open": 0.18})
    print_result("verify_drawer_open_after_pull", open_check)

    grasp = load_grasp_handle()
    grasped = grasp.execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "servo_steps": 800,
            "settle_steps": 100,
            "gripper_steps": 10,
            "fail_threshold": 0.10,
        },
    )
    print_result("grasp_handle", grasped)

    insert = load_insert_hand()
    inserted = insert.execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "servo_steps": 800,
            "settle_steps": 100,
            "fail_threshold": 0.12,
        },
    )
    print_result("insert_hand_into_drawer", inserted)

    extract = load_extract_object()
    extracted = extract.execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "servo_steps": 800,
            "settle_steps": 100,
            "fail_threshold": 0.12,
        },
    )
    print_result("extract_object_from_drawer", extracted)


if __name__ == "__main__":
    main()
