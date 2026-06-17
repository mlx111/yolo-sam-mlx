import argparse

import mujoco
import numpy as np

from skills.primitives.base_move_to_region_skill import load_skill as load_move_region
from skills.primitives.base_replan_path_skill import load_skill as load_replan_path
from skills.primitives.base_reposition_lateral_skill import load_skill as load_reposition_lateral


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test Phase 3B base navigation primitives.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--region", default="table_front")
    parser.add_argument("--object-body", default="target_cube")
    return parser.parse_args()


def print_result(label, result):
    print(f"{label}: success={result.success}, error={result.final_error:.6f}, message={result.message}")
    print("  target:", np.round(result.target_qpos, 6).tolist())
    print("  final:", np.round(result.final_qpos, 6).tolist())
    if result.waypoints:
        print("  waypoints:", [np.round(point, 6).tolist() for point in result.waypoints])


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    move_region = load_move_region()
    region_result = move_region.execute_recovery_action(
        model,
        data,
        {"region": args.region, "steps": 250, "settle_steps": 50},
    )
    print_result("base_move_to_region", region_result)

    lateral = load_reposition_lateral()
    lateral_result = lateral.execute_recovery_action(
        model,
        data,
        {"lateral_offset": 0.08, "forward_offset": 0.03, "steps": 160, "settle_steps": 40},
    )
    print_result("base_reposition_lateral", lateral_result)

    replan = load_replan_path()
    replan_result = replan.execute_recovery_action(
        model,
        data,
        {"object_body": args.object_body, "standoff_distance": 0.55, "steps": 180, "settle_steps": 40},
    )
    print_result("base_replan_path", replan_result)


if __name__ == "__main__":
    main()
