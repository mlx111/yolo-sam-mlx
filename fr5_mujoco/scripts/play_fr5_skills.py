from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer

from fr5_mujoco import FR5MotionRuntime
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor


def main() -> None:
    runtime = FR5MotionRuntime.from_scene(realtime=True)
    runtime.reset_home()
    executor = FR5FieldAtomicSkillExecutor(runtime)

    target_pos = (runtime.tcp_pos + [0.0, 0.0, 0.03]).tolist()
    plan = [
        {"action": "go_home", "parameters": {"duration": 1.0}},
        {
            "action": "create_fixed_vertical_grasp",
            "parameters": {"target_pos": target_pos, "pregrasp_height": 0.05},
        },
        {
            "action": "move_to_pregrasp",
            "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.05, "duration": 1.0},
        },
        {
            "action": "approach_object",
            "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "duration": 1.0},
        },
        {"action": "close_gripper", "parameters": {"duration": 0.8}},
        {"action": "lift", "parameters": {"lift_height": 0.05, "duration": 1.0}},
        {"action": "open_gripper", "parameters": {"duration": 0.8}},
        {"action": "go_home", "parameters": {"duration": 1.0}},
    ]

    with mujoco.viewer.launch_passive(runtime.model, runtime.data) as viewer:
        runtime.set_viewer(viewer)
        viewer.cam.lookat[:] = [-0.45, -0.2, 0.2]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25
        viewer.sync()
        time.sleep(1.0)

        for step in plan:
            result = executor.execute(step["action"], step.get("parameters", {}))
            print(result)
            time.sleep(0.35)

        end_time = time.time() + 10.0
        while viewer.is_running() and time.time() < end_time:
            viewer.sync()
            time.sleep(0.02)

        runtime.set_viewer(None)


if __name__ == "__main__":
    main()
