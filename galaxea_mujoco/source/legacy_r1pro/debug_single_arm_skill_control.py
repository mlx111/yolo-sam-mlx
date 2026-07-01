"""Debug one R1Pro arm manipulation skill under a selected control profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from source.legacy_r1pro.control_profiles import DEFAULT_PHYSICAL_CONTROL_PROFILE_ID, get_control_profile
from source.legacy_r1pro.grasp_defaults import DEFAULT_GRASP_OFFSET_Z, DEFAULT_PREGRASP_DISTANCE
from source.legacy_r1pro.run_r1pro_task_chain import (
    _body_pos,
    _control_trace,
    _load_model,
    _motion_snapshot,
    _motion_summary,
    _normalize_control_mode,
    _site_pos,
    _set_freejoint_body_pose,
)
from skills.primitives.approach_object_skill import load_skill as load_approach
from skills.primitives.move_to_pregrasp_skill import load_skill as load_pregrasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a single R1Pro arm skill with physical/direct control.")
    parser.add_argument("--skill", choices=["move_to_pregrasp", "approach_object"], default="move_to_pregrasp")
    parser.add_argument("--model-path", default="r1pro_g3_sorting_scene.xml")
    parser.add_argument("--condition", choices=["clean", "place_occupied"], default="clean")
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--control-profile", default=DEFAULT_PHYSICAL_CONTROL_PROFILE_ID)
    parser.add_argument("--object-body", default="target_cube")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def _round(values: Any, digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(values, dtype=np.float64).reshape(-1)]


def _failure_diagnosis(result: Any, motion: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    ik_error = float(control.get("ik_error", 0.0) or 0.0)
    final_error = float(control.get("final_error", 0.0) or 0.0)
    tracking_error = float(control.get("max_joint_tracking_error", 0.0) or 0.0)
    joint_margin = motion.get("joint_limit_margin_min")
    workspace_radius = float(motion.get("workspace_radius_after", 0.0) or 0.0)
    joint_limit_violation = joint_margin is not None and float(joint_margin) < 0.0
    primary = "success"
    if not bool(getattr(result, "success", False)):
        if ik_error > 0.08:
            primary = "ik_unreachable_or_bad_target"
        elif tracking_error > 0.08:
            primary = "actuator_tracking_error"
        elif joint_limit_violation:
            primary = "joint_limit_violation"
        elif workspace_radius > 1.2:
            primary = "workspace_exceeded"
        else:
            primary = "residual_pose_error"
    return {
        "primary_reason": primary,
        "ik_error": round(ik_error, 6),
        "final_error": round(final_error, 6),
        "max_joint_tracking_error": round(tracking_error, 6),
        "mean_joint_tracking_error": control.get("mean_joint_tracking_error"),
        "joint_limit_violation": bool(joint_limit_violation),
        "joint_limit_margin_min": joint_margin,
        "workspace_radius_after": round(workspace_radius, 6),
        "max_tcp_delta": motion.get("max_tcp_delta"),
    }


def _prepare_scene(model: mujoco.MjModel, data: mujoco.MjData, condition: str) -> None:
    if condition == "clean":
        _set_freejoint_body_pose(model, data, "place_obstacle_body", [0.54, 0.32, 0.895])
    mujoco.mj_forward(model, data)


def _target_report(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace, object_pos: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    tcp_pos = _site_pos(model, data, f"{args.side}_hand_tcp")
    approach = np.array([params["approach_dx"], params["approach_dy"], params["approach_dz"]], dtype=np.float64)
    approach = approach / max(float(np.linalg.norm(approach)), 1e-9)
    grasp = object_pos + np.array(
        [params["grasp_offset_x"], params["grasp_offset_y"], params["grasp_offset_z"]],
        dtype=np.float64,
    )
    pregrasp = grasp - approach * float(params["pregrasp_distance"])
    target = pregrasp if args.skill == "move_to_pregrasp" else grasp
    return {
        "tcp_start": _round(tcp_pos),
        "grasp_target": _round(grasp),
        "pregrasp_target": _round(pregrasp),
        "selected_target": _round(target),
        "target_distance_from_tcp_start": round(float(np.linalg.norm(target - tcp_pos)), 6),
        "pregrasp_distance": float(params["pregrasp_distance"]),
        "approach_direction": _round(approach),
    }


def main() -> None:
    args = parse_args()
    control_mode = _normalize_control_mode(args.control_mode)
    profile = get_control_profile(args.control_profile)
    model, data = _load_model(args.model_path)
    _prepare_scene(model, data, args.condition)

    object_pos = _body_pos(model, data, args.object_body)
    params = {
        "side": args.side,
        "object_body": args.object_body,
        "approach_dx": 0.0,
        "approach_dy": 0.0,
        "approach_dz": -1.0,
        "pregrasp_distance": DEFAULT_PREGRASP_DISTANCE,
        "grasp_offset_x": 0.0,
        "grasp_offset_y": 0.0,
        "grasp_offset_z": DEFAULT_GRASP_OFFSET_Z,
        "steps": 300,
        "settle_steps": 500,
        "fail_threshold": 0.02,
    }
    if control_mode == "physical":
        params.update(profile.get("arm", {}))

    target_report = _target_report(model, data, args, object_pos, params)
    before = _motion_snapshot(model, data)
    skill = load_pregrasp() if args.skill == "move_to_pregrasp" else load_approach()
    result = skill.execute_recovery_action(model, data, params)
    after = _motion_snapshot(model, data)
    motion = _motion_summary(before, after)
    control = _control_trace(result)
    payload = {
        "schema_version": "single_arm_skill_control_debug_v1",
        "skill": args.skill,
        "model_path": args.model_path,
        "condition": args.condition,
        "object_body": args.object_body,
        "object_position": _round(object_pos),
        "target_report": target_report,
        "control_mode": control_mode,
        "control_profile_id": profile.get("profile_id", ""),
        "arm_control_mode": profile.get("arm", {}).get("control_mode", "") if control_mode == "physical" else "direct_qpos",
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "motion": motion,
        "control": control,
        "failure_diagnosis": _failure_diagnosis(result, motion, control),
    }
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
