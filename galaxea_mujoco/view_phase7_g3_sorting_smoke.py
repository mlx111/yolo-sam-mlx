import argparse

import mujoco
import numpy as np

from skills.primitives.approach_object_skill import load_skill as load_approach
from skills.primitives.choose_alternate_place_skill import load_skill as load_choose_place
from skills.primitives.detect_multiple_objects_skill import load_skill as load_detect_multiple
from skills.primitives.detect_place_occupancy_skill import load_skill as load_detect_occupancy
from skills.primitives.left_gripper_close_skill import load_skill as load_left_close
from skills.primitives.left_vertical_lift_skill import load_skill as load_left_lift
from skills.primitives.move_to_pregrasp_skill import load_skill as load_pregrasp
from skills.primitives.open_gripper_release_skill import load_skill as load_release
from skills.primitives.place_object_skill import load_skill as load_place
from skills.primitives.select_correct_object_skill import load_skill as load_select_correct
from skills.primitives.verify_place_zone_skill import load_skill as load_verify_place


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test G3 multi-object sorting task chain.")
    parser.add_argument("--model", default="r1pro_g3_sorting_scene.xml")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--target-name", default="target_cube")
    parser.add_argument("--object-bodies", nargs="+", default=["target_cube", "distractor_cylinder", "distractor_box"])
    parser.add_argument("--occupancy-bodies", nargs="+", default=["place_obstacle_body", "target_cube"])
    return parser.parse_args()


def body_pos(model, data, body_name):
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    return data.xpos[body_id].copy()


def site_pos(model, data, site_name):
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    return data.site_xpos[site_id].copy()


def print_bool(label, result):
    print(f"{label}: success={result.success}, message={result.message}")


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    detect = load_detect_multiple().execute_recovery_action(model, data, {"object_bodies": args.object_bodies})
    print_bool("detect_multiple_objects", detect)
    print("  objects:", [obj.name for obj in detect.objects])

    selected = load_select_correct().execute_recovery_action(
        model,
        data,
        {"objects": detect.objects, "target_name": args.target_name, "require_unique": True},
    )
    print_bool("select_correct_object", selected)
    if selected.selected_object is None or selected.selected_object.body_name is None:
        raise RuntimeError("No selected object body available")
    target_body = selected.selected_object.body_name
    initial_object_z = float(body_pos(model, data, target_body)[2])
    print("  selected:", target_body)
    print("  object_start:", np.round(body_pos(model, data, target_body), 6).tolist())

    grasp_params = {
        "side": args.side,
        "object_body": target_body,
        "approach_dx": 0.0,
        "approach_dy": 0.0,
        "approach_dz": -1.0,
        "pregrasp_distance": 0.12,
        "grasp_offset_x": 0.0,
        "grasp_offset_y": 0.0,
        "grasp_offset_z": 0.025,
        "steps": 300,
        "settle_steps": 500,
        "fail_threshold": 0.02,
    }

    pregrasp = load_pregrasp().execute_recovery_action(model, data, grasp_params)
    print_bool("move_to_pregrasp", pregrasp)
    print("  error:", round(float(pregrasp.final_error), 6))

    approach = load_approach().execute_recovery_action(model, data, grasp_params)
    print_bool("approach_object", approach)
    print("  error:", round(float(approach.final_error), 6))

    closed = load_left_close().execute_recovery_action(model, data, {"object_body": target_body, "gripper_steps": 60})
    print_bool("left_gripper_close", closed)

    lifted = load_left_lift().execute_recovery_action(
        model,
        data,
        {
            "lift_dx": 0.0,
            "lift_dy": 0.0,
            "lift_dz": 0.18,
            "steps": 900,
            "settle_steps": 100,
            "fail_threshold": 0.03,
            "lift_tolerance": 0.03,
        },
    )
    print_bool("left_vertical_lift", lifted)
    print("  error:", round(float(lifted.final_error), 6))
    print("  object_after_lift:", np.round(body_pos(model, data, target_body), 6).tolist())

    occupancy = load_detect_occupancy().execute_recovery_action(
        model,
        data,
        {"candidate_bodies": args.occupancy_bodies, "exclude_bodies": [target_body]},
    )
    print_bool("detect_place_occupancy", occupancy)
    print("  occupied:", occupancy.occupied, "objects:", list(occupancy.occupied_objects))

    chosen = load_choose_place().execute_recovery_action(
        model,
        data,
        {"candidate_bodies": args.occupancy_bodies, "exclude_bodies": [target_body]},
    )
    print_bool("choose_alternate_place", chosen)
    if chosen.selected_site is None:
        raise RuntimeError("No free place site selected")
    place = site_pos(model, data, chosen.selected_site)
    print("  selected_site:", chosen.selected_site)
    print("  selected_site_pos:", np.round(place, 6).tolist())

    placed = load_place().execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "place_x": float(place[0]),
            "place_y": float(place[1]),
            "place_z": float(place[2]),
            "place_offset_x": 0.0,
            "place_offset_y": 0.0,
            "place_offset_z": 0.10,
            "steps": 900,
            "settle_steps": 100,
            "fail_threshold": 0.05,
            "orientation_weight": 0.0,
        },
    )
    print_bool("place_object", placed)
    print("  error:", round(float(placed.final_error), 6))

    released = load_release().execute_recovery_action(
        model,
        data,
        {"side": args.side, "gripper_steps": 60, "settle_steps": 20},
    )
    print_bool("open_gripper_release", released)

    final_pos = body_pos(model, data, target_body)
    verified = load_verify_place().execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "object_body": target_body,
            "place_x": float(place[0]),
            "place_y": float(place[1]),
            "place_z": float(place[2]),
            "max_xy_error": 0.025,
            "max_z_error": 0.08,
        },
    )
    print_bool("verify_place_zone", verified)
    print("  object_final:", np.round(final_pos, 6).tolist())
    print("  lift_total:", round(float(final_pos[2] - initial_object_z), 6))

    results = (detect, selected, pregrasp, approach, closed, lifted, occupancy, chosen, placed, released, verified)
    failed = [result.name for result in results if not result.success and result.name != "detect_place_occupancy"]
    if failed:
        raise SystemExit(f"G3 sorting smoke failed: {failed}")


if __name__ == "__main__":
    main()
