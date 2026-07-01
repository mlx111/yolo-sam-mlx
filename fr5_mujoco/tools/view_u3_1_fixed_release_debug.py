from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from run_experiment_v4 import DEFAULT_PREGRASP_HEIGHT, ExperimentV4
from skills import registry


FIXED_STEPS = [
    {"action": "camera_rgbd_save", "parameters": {}},
    {"action": "detect_object_pose", "parameters": {"target_class": "apple"}},
    {"action": "create_fixed_vertical_grasp", "parameters": {}},
    {"action": "move_to_pregrasp", "parameters": {"dx": 0.03, "dy": -0.01, "dz": 0.09}},
    {"action": "approach_object", "parameters": {"dx": 0.01, "dy": 0.02, "dz": -0.02}},
    {"action": "close_gripper", "parameters": {}},
    {"action": "lift", "parameters": {"lift_height": 0.13}},
    {"action": "camera_rgbd_save", "parameters": {}},
    {"action": "detect_object_pose", "parameters": {"target_class": "plate"}},
    {"action": "move_lifted_object_to", "parameters": {"target_pos": [0.39, 0.41, 0.04]}},
    {"action": "open_gripper", "parameters": {}},
    {"action": "go_home", "parameters": {}},
]


def _snap(exp: ExperimentV4) -> dict:
    apple = exp.data.body(exp.apple_body_id).xpos.copy()
    pinch = exp.data.site_xpos[exp.pinch_site_id].copy()
    tcp = np.asarray(exp.robot.get_cartesian().t, dtype=float)
    plate_id = mujoco.mj_name2id(exp.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    plate = exp.data.body(plate_id).xpos.copy()
    return {
        "apple": [round(float(x), 4) for x in apple],
        "pinch": [round(float(x), 4) for x in pinch],
        "tcp": [round(float(x), 4) for x in tcp],
        "plate": [round(float(x), 4) for x in plate],
        "pinch_dist": round(float(np.linalg.norm(apple - pinch)), 4),
        "apple_plate_xy": round(float(np.linalg.norm(apple[:2] - plate[:2])), 4),
        "contacts": exp._contact_summary(),
    }


def _hold(exp: ExperimentV4, label: str, seconds: float) -> None:
    print(f"\n[HOLD] {label}: {seconds:.1f}s")
    print("[HOLD]", json.dumps(_snap(exp), ensure_ascii=False))
    end = time.time() + seconds
    while time.time() < end:
        if exp.viewer:
            exp.viewer.sync()
        time.sleep(0.03)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", type=float, default=20.0)
    parser.add_argument("--scene-xml", default=str(ROOT / "scene" / "scene.xml"))
    args = parser.parse_args()

    exp = ExperimentV4(
        enable_viewer=True,
        condition="direct",
        noise_scale=0.0,
        condition_id="U3-1",
        scene_xml=args.scene_xml,
    )

    def fixed_recovery(self: ExperimentV4) -> None:
        print("\n[FIXED-VIEW] 开始固定恢复流程，不调用 LLM")
        self.metrics["llm_recovery_steps"] = FIXED_STEPS
        self.metrics["executed_recovery_steps"] = FIXED_STEPS
        self.metrics["executed_plan_source"] = "fixed_view_debug_flow"
        action_map = registry.build_action_map(self, DEFAULT_PREGRASP_HEIGHT, self.scenario_id)
        _hold(self, "恢复开始前", args.hold)

        for i, step in enumerate(FIXED_STEPS, 1):
            action = step["action"]
            params = step.get("parameters") or {}
            print(f"\n[FIXED-VIEW {i}/{len(FIXED_STEPS)}] {action} params={params}")
            result = action_map[action](params)
            print(
                "[FIXED-VIEW]",
                "success=", getattr(result, "success", None),
                "status=", getattr(result, "status", None),
                "snap=", json.dumps(_snap(self), ensure_ascii=False),
            )
            if action == "move_lifted_object_to":
                _hold(self, "move_lifted_object_to 后，开夹前", args.hold)
            if action == "open_gripper":
                _hold(self, "open_gripper 后", args.hold)

        self.metrics["task_success"] = False

    exp._execute_recovery = types.MethodType(fixed_recovery, exp)
    try:
        exp.run(inject_anomaly=True)
        _hold(exp, "实验结束", args.hold)
    finally:
        exp.close()


if __name__ == "__main__":
    main()
