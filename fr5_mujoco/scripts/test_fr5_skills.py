from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_mujoco import FR5MotionRuntime
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor


def main() -> None:
    runtime = FR5MotionRuntime.from_scene()
    runtime.reset_home()
    executor = FR5FieldAtomicSkillExecutor(runtime)

    target_pos = (runtime.tcp_pos + [0.0, 0.0, 0.03]).tolist()
    plan = [
        {"action": "go_home", "parameters": {"duration": 1.0}},
        {"action": "create_fixed_vertical_grasp", "parameters": {"target_pos": target_pos, "pregrasp_height": 0.05}},
        {"action": "move_to_pregrasp", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.05, "duration": 1.0}},
        {"action": "approach_object", "parameters": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "duration": 1.0}},
        {"action": "close_gripper", "parameters": {"duration": 0.5}},
        {"action": "lift", "parameters": {"lift_height": 0.05, "duration": 1.0}},
        {"action": "open_gripper", "parameters": {"duration": 0.5}},
        {"action": "go_home", "parameters": {"duration": 1.0}},
    ]
    results = executor.execute_plan(plan)
    for result in results:
        print(result)
    failed = [result for result in results if not result.success]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
