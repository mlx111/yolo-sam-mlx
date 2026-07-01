from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from fr5_mujoco import FR5MotionRuntime
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = FR5MotionRuntime.from_scene(str(root / "assets" / "scene.xml"), realtime=True)
    runtime.reset_home()
    executor = FR5FieldAtomicSkillExecutor(runtime)
    speed_scale = 2.5

    perception_plan = [
        {"action": "go_camera_ready", "parameters": {"duration": 1.5}},
        {
            "action": "camera_rgbd_save",
            "parameters": {
                "camera_name": "ee_camera",
                "width": 640,
                "height": 480,
                "prefix": "play_pick_apple_to_plate",
            },
        },
        {
            "action": "detect_object_pose",
            "parameters": {
                "target_class": "apple",
                "output_dir": str(root / "output" / "grounded_sam2_play_pick_apple_to_plate_apple"),
            },
        },
        {
            "action": "detect_object_pose",
            "parameters": {
                "target_class": "plate",
                "output_dir": str(root / "output" / "grounded_sam2_play_pick_apple_to_plate_plate"),
            },
        },
    ]

    print("Running perception before opening viewer...")
    for step in perception_plan:
        result = executor.execute(step["action"], step.get("parameters", {}))
        print(result)
        if not result.success:
            raise SystemExit(1)

    motion_plan = [
        {"action": "create_fixed_vertical_grasp", "parameters": {"target_class": "apple", "pregrasp_height": 0.08}},
        {"action": "move_to_pregrasp", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.08, "duration": 1.2 * speed_scale}},
        {"action": "approach_object", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "duration": 1.2 * speed_scale, "settle_steps": 250}},
        {"action": "close_gripper", "parameters": {"duration": 1.5 * speed_scale}},
        {"action": "lift", "parameters": {"lift_height": 0.10, "duration": 2.0 * speed_scale}},
        {"action": "move_lifted_object_to", "parameters": {"target": "plate", "height": 0.13, "duration": 1.8 * speed_scale}},
        {"action": "open_gripper", "parameters": {"duration": 0.8 * speed_scale}},
    ]

    with mujoco.viewer.launch_passive(runtime.model, runtime.data) as viewer:
        runtime.set_viewer(viewer)
        viewer.cam.lookat[:] = [0.35, 0.3, 0.18]
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -35
        viewer.sync()
        time.sleep(1.0)

        for step in motion_plan:
            result = executor.execute(step["action"], step.get("parameters", {}))
            print(result)
            if not result.success:
                break
            time.sleep(0.35)

        runtime._step_n(800)
        end_time = time.time() + 20.0
        while viewer.is_running() and time.time() < end_time:
            viewer.sync()
            time.sleep(0.02)

        runtime.set_viewer(None)


if __name__ == "__main__":
    main()
