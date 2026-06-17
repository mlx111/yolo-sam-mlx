import argparse

import mujoco
import numpy as np

from skills.primitives.classify_target_object_skill import load_skill as load_classify_target
from skills.primitives.detect_multiple_objects_skill import load_skill as load_detect_multiple
from skills.primitives.detect_object_pose_skill import load_skill as load_detect_object
from skills.primitives.multi_view_redetect_skill import load_skill as load_multi_view
from skills.primitives.redetect_target_pose_skill import load_skill as load_redetect
from skills.primitives.verify_grasped_object_skill import load_skill as load_verify_grasped


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test Phase 2 MuJoCo perception skills.")
    parser.add_argument("--model", default="r1pro_grasp_scene.xml")
    parser.add_argument("--target-body", default="target_cube")
    parser.add_argument("--target-label", default="target_cube")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    return parser.parse_args()


def print_object(prefix, obj):
    if obj is None:
        print(f"{prefix}: None")
        return
    print(
        f"{prefix}: name={obj.name}, body={obj.body_name}, "
        f"pos={np.round(obj.position, 6).tolist()}, label={obj.label}"
    )


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    detect = load_detect_object()
    one = detect.execute_recovery_action(model, data, {"object_body": args.target_body})
    print("detect_object_pose:", one.success, one.message)
    print_object("  selected", one.selected_object)

    detect_many = load_detect_multiple()
    many = detect_many.execute_recovery_action(model, data, {"movable_only": True})
    print("detect_multiple_objects:", many.success, many.message)
    for obj in many.objects:
        print_object("  object", obj)

    classify = load_classify_target()
    selected = classify.execute_recovery_action(
        model,
        data,
        {"objects": many.objects, "target_label": args.target_label},
    )
    print("classify_target_object:", selected.success, selected.message)
    print_object("  selected", selected.selected_object)

    redetect = load_redetect()
    again = redetect.execute_recovery_action(model, data, {"target_name": args.target_body})
    print("redetect_target_pose:", again.success, again.message)
    print_object("  selected", again.selected_object)

    multi_view = load_multi_view()
    mv = multi_view.execute_recovery_action(
        model,
        data,
        {"object_bodies": [obj.body_name for obj in many.objects if obj.body_name], "target_label": args.target_label},
    )
    print("multi_view_redetect:", mv.success, mv.message)
    print_object("  selected", mv.selected_object)

    verify = load_verify_grasped()
    verified = verify.execute_recovery_action(
        model,
        data,
        {
            "side": args.side,
            "expected_object_body": args.target_body,
            "object_bodies": [obj.body_name for obj in many.objects if obj.body_name],
            "max_grasp_distance": 0.08,
        },
    )
    print("verify_grasped_object:", verified.success, verified.message)
    print_object("  selected", verified.selected_object)


if __name__ == "__main__":
    main()
