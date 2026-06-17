import argparse

import mujoco
import numpy as np

from skills.primitives.safe_transport_pose_skill import load_skill as load_safe_transport
from skills.primitives.torso_set_height_skill import load_skill as load_torso_height
from skills.primitives.torso_turn_to_target_skill import load_skill as load_torso_turn


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test Phase 3A whole-body positioning skills.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--object-body", default="target_cube")
    parser.add_argument("--height-level", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--safe-posture", choices=["carry_center", "home"], default="home")
    return parser.parse_args()


def print_result(label, result):
    print(f"{label}: success={result.success}, error={result.final_error:.6f}, message={result.message}")
    print("  target:", np.round(result.target_qpos, 4).tolist())
    print("  final:", np.round(result.final_qpos, 4).tolist())


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    torso_height = load_torso_height()
    height_result = torso_height.execute_recovery_action(
        model,
        data,
        {"height_level": args.height_level, "steps": 120, "settle_steps": 20},
    )
    print_result("torso_set_height", height_result)

    torso_turn = load_torso_turn()
    turn_result = torso_turn.execute_recovery_action(
        model,
        data,
        {"object_body": args.object_body, "steps": 120, "settle_steps": 20},
    )
    print_result("torso_turn_to_target", turn_result)

    safe_transport = load_safe_transport()
    transport_result = safe_transport.execute_recovery_action(
        model,
        data,
        {"posture": args.safe_posture, "steps": 120, "settle_steps": 20, "direct_qpos": True},
    )
    print_result("safe_transport_pose", transport_result)


if __name__ == "__main__":
    main()
