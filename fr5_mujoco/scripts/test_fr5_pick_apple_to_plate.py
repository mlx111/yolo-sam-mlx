from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_mujoco import FR5MotionRuntime
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = FR5MotionRuntime.from_scene(str(root / "assets" / "scene.xml"))
    runtime.reset_home()
    executor = FR5FieldAtomicSkillExecutor(runtime)
    plan = [
        {"action": "go_camera_ready", "parameters": {"duration": 1.5}},
        {
            "action": "camera_rgbd_save",
            "parameters": {
                "camera_name": "ee_camera",
                "width": 640,
                "height": 480,
                "prefix": "pick_apple_to_plate",
            },
        },
        {
            "action": "detect_object_pose",
            "parameters": {
                "target_class": "apple",
                "output_dir": str(root / "output" / "grounded_sam2_pick_apple_to_plate_apple"),
            },
        },
        {
            "action": "detect_object_pose",
            "parameters": {
                "target_class": "plate",
                "output_dir": str(root / "output" / "grounded_sam2_pick_apple_to_plate_plate"),
            },
        },
        {"action": "create_fixed_vertical_grasp", "parameters": {"target_class": "apple", "pregrasp_height": 0.08}},
        {"action": "move_to_pregrasp", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.08, "duration": 1.2}},
        {"action": "approach_object", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "duration": 1.2, "settle_steps": 100}},
        {"action": "close_gripper", "parameters": {"duration": 1.5}},
        {"action": "lift", "parameters": {"lift_height": 0.10, "duration": 1.3}},
        {"action": "move_lifted_object_to", "parameters": {"target": "plate", "height": 0.13, "duration": 1.8}},
        {"action": "open_gripper", "parameters": {"duration": 0.8}},
    ]
    for step in plan:
        result = executor.execute(step["action"], step.get("parameters", {}))
        print(result)
        if not result.success:
            raise SystemExit(1)

    runtime._step_n(800)
    apple_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "apple0")
    plate_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "plate_geom")
    apple_pos = runtime.data.geom_xpos[apple_id].copy()
    plate_pos = runtime.data.geom_xpos[plate_id].copy()
    xy_error = float(np.linalg.norm(apple_pos[:2] - plate_pos[:2]))
    print("FINAL_APPLE", apple_pos.tolist())
    print("PLATE", plate_pos.tolist())
    print("XY_ERROR", xy_error)
    print("CONTACTS", runtime._contact_summary())
    if xy_error > 0.08:
        raise SystemExit("apple is outside plate radius")


if __name__ == "__main__":
    main()
