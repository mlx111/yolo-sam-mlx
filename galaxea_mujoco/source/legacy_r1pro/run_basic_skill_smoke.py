"""Run isolated smoke tests for core R1Pro MuJoCo skills."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from source.legacy_r1pro.run_r1pro_task_chain import _load_model, _site_pos
from skills.base.base_lidar_skill import load_skill as load_base_lidar
from skills.base.base_motion_skill import load_skill as load_base_motion
from skills.base.head_camera_skill import load_skill as load_head_camera
from skills.base.left_arm_move_skill import load_skill as load_left_arm_move
from skills.base.right_arm_move_skill import load_skill as load_right_arm_move
from skills.base.torso_move_skill import load_skill as load_torso_move
from skills.base.torso_move_skill import load_skill as load_torso_turn_loader
from skills.base.wrist_force_skill import load_skill as load_wrist_force
from skills.primitives.base_move_to_region_skill import load_skill as load_base_move_to_region
from skills.primitives.base_reposition_lateral_skill import load_skill as load_base_reposition_lateral
from skills.primitives.base_replan_path_skill import load_skill as load_base_replan_path
from skills.primitives.left_retreat_hand_skill import load_skill as load_left_retreat
from skills.primitives.detect_multiple_objects_skill import load_skill as load_detect_multiple
from skills.primitives.detect_place_occupancy_skill import load_skill as load_detect_occupancy
from skills.primitives.left_gripper_close_skill import load_skill as load_left_close
from skills.primitives.left_gripper_open_skill import load_skill as load_left_open
from skills.primitives.right_gripper_close_skill import load_skill as load_right_close
from skills.primitives.right_gripper_open_skill import load_skill as load_right_open
from skills.primitives.right_retreat_hand_skill import load_skill as load_right_retreat
from skills.primitives.safe_transport_pose_skill import load_skill as load_safe_transport_pose
from skills.primitives.pre_grasp_safe_posture_skill import load_skill as load_pregrasp_safe_posture
from skills.primitives.select_correct_object_skill import load_skill as load_select_correct
from skills.primitives.torso_set_height_skill import load_skill as load_torso_set_height
from skills.primitives.torso_turn_to_target_skill import load_skill as load_torso_turn


def _round(values: Any, digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(values, dtype=np.float64).reshape(-1)]


def _result_ok(result: Any) -> bool:
    if hasattr(result, "success"):
        return bool(result.success)
    return True


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _round(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summarize_result(result: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "success",
        "final_error",
        "force_norm",
        "torque_norm",
        "contact_count",
        "min_range",
        "max_range",
        "message",
    ):
        if hasattr(result, key):
            summary[key] = _scalar(getattr(result, key))
    if hasattr(result, "objects"):
        summary["object_count"] = len(getattr(result, "objects") or [])
    if hasattr(result, "selected_object"):
        selected = getattr(result, "selected_object")
        summary["selected_object"] = getattr(selected, "name", "") if selected is not None else ""
    if hasattr(result, "rgb"):
        summary["rgb_shape"] = list(result.rgb.shape)
    if hasattr(result, "depth") and result.depth is not None:
        summary["depth_shape"] = list(result.depth.shape)
    if hasattr(result, "ranges"):
        ranges = np.asarray(result.ranges, dtype=np.float64)
        summary["range_count"] = int(ranges.size)
        summary["range_min_observed"] = round(float(np.min(ranges)), 6) if ranges.size else None
    return summary


def _run_case(name: str, fn: Callable[[], Any], *, drive_mode: str) -> dict[str, Any]:
    try:
        result = fn()
        semantic_success = _result_ok(result)
        return {
            "name": name,
            "drive_mode": drive_mode,
            "execution_success": True,
            "semantic_success": semantic_success,
            "success": True,
            "summary": _summarize_result(result),
        }
    except Exception as exc:
        return {
            "name": name,
            "drive_mode": drive_mode,
            "execution_success": False,
            "semantic_success": False,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_smoke(model_path: str, *, physical_drive: bool = True) -> dict[str, Any]:
    model, data = _load_model(model_path)
    mujoco.mj_forward(model, data)
    cases: list[dict[str, Any]] = []

    def fresh() -> tuple[mujoco.MjModel, mujoco.MjData]:
        m, d = _load_model(model_path)
        mujoco.mj_forward(m, d)
        return m, d

    cases.append(_run_case(
        "base_move_to_pose",
        lambda: load_base_motion().move_to_pose(*fresh(), [0.03, -0.02, 0.05], steps=300, settle_steps=80, max_joint_step=0.004, fail_threshold=0.04, direct_qpos=not physical_drive),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    cases.append(_run_case(
        "base_move_to_region",
        lambda: load_base_move_to_region().execute_recovery_action(*fresh(), {"region": "table_front", "steps": 300, "settle_steps": 80, "max_joint_step": 0.004, "fail_threshold": 0.04, "direct_qpos": not physical_drive}),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    cases.append(_run_case(
        "base_reposition_lateral",
        lambda: load_base_reposition_lateral().execute_recovery_action(*fresh(), {"lateral_offset": 0.02, "forward_offset": -0.01, "yaw_delta": 0.02, "steps": 240, "settle_steps": 60, "max_joint_step": 0.004, "fail_threshold": 0.04, "direct_qpos": not physical_drive}),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    cases.append(_run_case(
        "base_replan_path",
        lambda: load_base_replan_path().execute_recovery_action(*fresh(), {"waypoints": [[0.02, 0.01, 0.02], [0.04, 0.0, 0.0]], "steps": 240, "settle_steps": 60, "max_joint_step": 0.004, "fail_threshold": 0.05, "direct_qpos": not physical_drive}),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    cases.append(_run_case(
        "torso_move_to_posture",
        lambda: load_torso_move().move_to_posture(*fresh(), [0.0, 0.04, -0.02, 0.06], steps=300, settle_steps=120, max_joint_step=0.004, fail_threshold=0.04, direct_qpos=not physical_drive),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    cases.append(_run_case(
        "torso_set_height",
        lambda: load_torso_set_height().execute_recovery_action(*fresh(), {"height_level": "mid", "steps": 300, "settle_steps": 120, "max_joint_step": 0.004, "fail_threshold": 0.04, "direct_qpos": not physical_drive}),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    cases.append(_run_case(
        "torso_turn_to_target",
        lambda: load_torso_turn().execute_recovery_action(*fresh(), {"object_body": "target_cube", "steps": 300, "settle_steps": 120, "max_joint_step": 0.004, "fail_threshold": 0.04, "direct_qpos": not physical_drive}),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))

    for side, loader in (("left", load_left_arm_move), ("right", load_right_arm_move)):
        def arm_case(side: str = side, loader: Callable[..., Any] = loader) -> Any:
            m, d = fresh()
            start = _site_pos(m, d, f"{side}_hand_tcp")
            target = start + np.array([0.0, 0.0, 0.015], dtype=np.float64)
            return loader().move_to_position(
                m,
                d,
                target,
                steps=600,
                settle_steps=160,
                direct_qpos=not physical_drive,
                stabilize=physical_drive,
                fail_threshold=0.04,
            )

        cases.append(_run_case(
            f"{side}_arm_tcp_small_lift",
            arm_case,
            drive_mode="physical_actuator" if physical_drive else "direct_qpos",
        ))
        cases.append(_run_case(
            f"{side}_retreat_hand",
            lambda side=side: (load_left_retreat() if side == "left" else load_right_retreat()).execute_recovery_action(
                *fresh(),
                {
                    "retreat_dx": 0.0,
                    "retreat_dy": 0.0,
                    "retreat_dz": 0.01,
                    "steps": 120,
                    "settle_steps": 20,
                    "fail_threshold": 0.04,
                    "direct_qpos": not physical_drive,
                },
            ),
            drive_mode="physical_actuator" if physical_drive else "direct_qpos",
        ))

    for side, close_loader, open_loader in (
        ("left", load_left_close, load_left_open),
        ("right", load_right_close, load_right_open),
    ):
        cases.append(_run_case(
            f"{side}_gripper_close",
            lambda close_loader=close_loader: close_loader().execute_recovery_action(*fresh(), {"gripper_steps": 120, "direct_qpos": not physical_drive, "attach_on_close": False}),
            drive_mode="physical_actuator" if physical_drive else "direct_qpos",
        ))
        cases.append(_run_case(
            f"{side}_gripper_open",
            lambda open_loader=open_loader: open_loader().execute_recovery_action(*fresh(), {"gripper_steps": 120, "direct_qpos": not physical_drive}),
            drive_mode="physical_actuator" if physical_drive else "direct_qpos",
        ))

    cases.append(_run_case(
        "detect_multiple_objects",
        lambda: load_detect_multiple().execute_recovery_action(*fresh(), {"object_bodies": ["target_cube", "distractor_cylinder", "distractor_box"]}),
        drive_mode="sensor_read",
    ))

    def select_case() -> Any:
        m, d = fresh()
        detected = load_detect_multiple().execute_recovery_action(m, d, {"object_bodies": ["target_cube", "distractor_cylinder", "distractor_box"]})
        return load_select_correct().execute_recovery_action(m, d, {"objects": detected.objects, "target_name": "target_cube", "require_unique": True})

    cases.append(_run_case("select_correct_object", select_case, drive_mode="sensor_read"))
    cases.append(_run_case(
        "detect_place_occupancy",
        lambda: load_detect_occupancy().execute_recovery_action(*fresh(), {"candidate_bodies": ["place_obstacle_body", "target_cube"], "exclude_bodies": ["target_cube"]}),
        drive_mode="sensor_read",
    ))
    cases.append(_run_case(
        "base_lidar_scan",
        lambda: load_base_lidar().execute_recovery_action(*fresh(), {"ray_count": 45, "max_range": 3.0}),
        drive_mode="sensor_read",
    ))
    cases.append(_run_case(
        "head_camera_capture",
        lambda: load_head_camera(width=80, height=60).execute_recovery_action(*fresh(), {"width": 80, "height": 60, "include_depth": True}),
        drive_mode="sensor_read",
    ))
    cases.append(_run_case(
        "pre_grasp_safe_posture",
        lambda: load_pregrasp_safe_posture().execute_recovery_action(*fresh(), {"posture": "neutral_seed", "steps": 300, "settle_steps": 120, "max_joint_step": 0.004, "fail_threshold": 0.08, "direct_qpos": not physical_drive}),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    cases.append(_run_case(
        "safe_transport_pose",
        lambda: load_safe_transport_pose().execute_recovery_action(*fresh(), {"posture": "carry_center", "steps": 300, "settle_steps": 120, "max_joint_step": 0.004, "fail_threshold": 0.08, "direct_qpos": not physical_drive}),
        drive_mode="physical_actuator" if physical_drive else "direct_qpos",
    ))
    for side in ("left", "right"):
        cases.append(_run_case(
            f"{side}_wrist_force_read",
            lambda side=side: load_wrist_force().execute_recovery_action(*fresh(), {"side": side}),
            drive_mode="sensor_read",
        ))

    return {
        "schema_version": "r1pro_basic_skill_smoke_v1",
        "model_path": model_path,
        "physical_drive": bool(physical_drive),
        "case_count": len(cases),
        "execution_success_count": sum(1 for item in cases if item["execution_success"]),
        "execution_failure_count": sum(1 for item in cases if not item["execution_success"]),
        "semantic_success_count": sum(1 for item in cases if item["semantic_success"]),
        "semantic_failure_count": sum(1 for item in cases if not item["semantic_success"]),
        "success_count": sum(1 for item in cases if item["success"]),
        "failure_count": sum(1 for item in cases if not item["success"]),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated smoke tests for core R1Pro skills.")
    parser.add_argument("--model-path", default="r1pro_g3_sorting_scene.xml")
    parser.add_argument("--save", type=Path, default=Path("/tmp/r1pro_basic_skill_smoke.json"))
    parser.add_argument("--direct-qpos", action="store_true", help="Use direct qpos writes for motion skills instead of physical actuator stepping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_smoke(args.model_path, physical_drive=not args.direct_qpos)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
