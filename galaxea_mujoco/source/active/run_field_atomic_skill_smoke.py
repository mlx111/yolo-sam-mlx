from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_perception.warnings_filter import suppress_grounded_sam2_warnings
from source.active.keyframe_recorder import KeyframeRecorder

suppress_grounded_sam2_warnings()

EXPERIENCE_ROOT = ROOT.parent / "experience_system"
for path in (ROOT, EXPERIENCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from skills.field_atomic import FieldAtomicSkillExecutor, build_atomic_experience_entry
from skills.field_atomic.atomic_executor import _canonical_action
from skills.field_atomic.atomic_executor import public_field_atomic_parameters
from experience_core import (
    CriticResult,
    ExperienceEntry,
    ExperienceLibrary,
    FailureClusterer,
    MemoryGate,
    ObjectState,
    SensorEvidence,
    SensorSummary,
    SkillTraceItem,
    VisualRetrievalIndex,
    build_critic_result,
    compute_memory_gate,
    consolidate_memory_lifecycle,
    critique_field_atomic_failure,
    field_atomic_memory_lesson,
    image_paths_from_entry,
)
from experience_core.schema import build_retrieval_key


DEFAULT_ACTIONS = [
    {
        "action": "base_move_to_pose",
        "parameters": {"base_x": 0.02, "base_y": 0.0, "base_yaw": 0.0, "steps": 120, "settle_steps": 20, "max_joint_step": 0.004, "direct_qpos": False},
    },
    {
        "action": "torso_move_to_posture",
        "parameters": {"target_qpos": [0.0, 0.02, 0.0, 0.02], "steps": 120, "settle_steps": 20, "max_joint_step": 0.004, "direct_qpos": False},
    },
    {
        "action": "left_gripper_set",
        "parameters": {"state": 1, "direct_qpos": False},
    },
    {
        "action": "head_camera_capture",
        "parameters": {"width": 80, "height": 60, "include_depth": True},
    },
    {
        "action": "base_lidar_scan",
        "parameters": {"ray_count": 45, "horizontal_fov_deg": 180.0, "max_range": 3.0},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test field atomic parameterized skills.")
    parser.add_argument("--model-path", default="r1pro_g3_sorting_scene.xml")
    parser.add_argument("--actions", type=Path, default=None, help='JSON file containing {"steps": [{"action": ..., "parameters": {...}}]}')
    parser.add_argument("--scenario-id", default="field_atomic_smoke")
    parser.add_argument("--condition-id", default="default")
    parser.add_argument("--goal", default="execute field atomic skill sequence and record outcome")
    parser.add_argument("--initial-state", type=Path, default=None, help="Optional replay_state JSON whose qpos/qvel are restored before execution.")
    parser.add_argument("--settle-before-steps", type=int, default=0)
    parser.add_argument("--save-report", type=Path, default=None)
    parser.add_argument("--save-experience-library", type=Path, default=None)
    parser.add_argument("--experience-read", type=Path, default=None)
    parser.add_argument("--experience-save-mode", choices=["all", "success_only", "failure_only", "none"], default="all")
    parser.add_argument("--episode-role", choices=["execution", "recovery"], default="execution")
    parser.add_argument("--source-failure-report", type=Path, default=None)
    parser.add_argument("--source-failure-experience-id", default="")
    parser.add_argument("--source-recovery-plan", type=Path, default=None)
    parser.add_argument("--visual-index-dir", type=Path, default=None)
    parser.add_argument("--skip-visual-index", action="store_true")
    parser.add_argument("--llm-critic", dest="llm_critic", action="store_true", default=True, help="Use an LLM critic to enrich failed episode memories. This is the default.")
    parser.add_argument("--no-llm-critic", dest="llm_critic", action="store_false", help="Disable LLM critic enrichment for failed episode memories.")
    parser.add_argument("--llm-critic-provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--llm-critic-model", default="")
    parser.add_argument("--apply-memory-lifecycle", action="store_true")
    parser.add_argument("--stm-capacity", type=int, default=30)
    parser.add_argument("--viewer", action="store_true", help="Show a MuJoCo passive viewer while running the action sequence.")
    parser.add_argument("--viewer-sync-every", type=int, default=4, help="Sync viewer every N mj_step calls.")
    parser.add_argument("--viewer-hold-seconds", type=float, default=-1.0, help="Hold viewer after execution. Negative means hold until the window is closed.")
    parser.add_argument("--stop-on-failure", action="store_true", default=True, help="Stop the action sequence after the first failed action. This is the default.")
    parser.add_argument("--continue-on-failure", dest="stop_on_failure", action="store_false", help="Continue running later actions after a failed action.")
    parser.add_argument("--cleanup-runtime-tmp", action="store_true", help="Delete per-run tmp files after execution.")
    parser.add_argument("--keyframe-dir", type=Path, default=None, help="Directory for rendered keyframe png files.")
    parser.add_argument("--keyframe-camera", default="workspace_overview_camera", help="MuJoCo camera name for keyframe rendering.")
    return parser.parse_args()


def _load_actions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_ACTIONS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError('--actions must be a JSON object with a "steps" list')
    return [item for item in payload["steps"] if isinstance(item, dict)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _task_summary(action_reports: list[dict[str, Any]]) -> dict[str, Any]:
    all_action_success = all(bool(item["success"]) for item in action_reports)
    lift_reports = [item for item in action_reports if _canonical_action(item.get("action")) == "lift"]
    criteria = _build_task_success_criteria(action_reports, all_action_success=all_action_success)
    if not lift_reports:
        return {
            "task_success": False,
            "all_action_success": all_action_success,
            "task_success_reason": "no_lift_action_executed",
            "task_success_criteria": criteria,
            "recovery_success_criteria": criteria,
        }
    lift = lift_reports[-1]
    raw = lift.get("raw_result") if isinstance(lift.get("raw_result"), dict) else {}
    object_lift_world = raw.get("object_lift_world")
    min_object_lift = lift.get("parameters", {}).get("min_object_lift", 0.0)
    object_lift_success = raw.get("object_lift_success")
    task_success = bool(object_lift_success) if object_lift_success is not None else all_action_success
    return {
        "task_success": task_success,
        "all_action_success": all_action_success,
        "lift_action_success": bool(lift.get("success", False)),
        "object_lift_success": object_lift_success,
        "object_body": raw.get("object_body"),
        "object_lift_world": object_lift_world,
        "min_object_lift": min_object_lift,
        "object_start_world": raw.get("object_start_world"),
        "object_final_world": raw.get("object_final_world"),
        "task_success_criteria": criteria,
        "recovery_success_criteria": criteria,
    }


def _build_task_success_criteria(action_reports: list[dict[str, Any]], *, all_action_success: bool) -> dict[str, Any]:
    actions = [_canonical_action(item.get("action")) for item in action_reports]
    failed_predicates: list[str] = []
    descriptions: dict[str, str] = {}

    def require(predicate: str, ok: bool, description: str) -> None:
        if not ok:
            failed_predicates.append(predicate)
            descriptions[predicate] = description

    require("has_perception", "head_camera_grounded_sam2_pose" in actions, "没有完成目标物体位置识别。")
    require("has_pregrasp", "move_to_pregrasp" in actions, "没有执行到达预抓取位置。")
    require("has_approach", "approach_object" in actions, "没有执行接近抓取点。")
    require("has_close_gripper", "close_gripper" in actions, "没有闭合夹爪。")
    require("has_lift", "lift" in actions, "没有执行提升动作。")

    first_failed = _first_failed_action(action_reports)
    if first_failed is not None:
        action = _canonical_action(first_failed.get("action"))
        failed_predicates.append(f"{action}_success")
        descriptions[f"{action}_success"] = str(first_failed.get("message") or f"{action} 执行失败。")

    lift_reports = [item for item in action_reports if _canonical_action(item.get("action")) == "lift"]
    object_lift_world = None
    min_object_lift = None
    object_lift_success = None
    if lift_reports:
        lift = lift_reports[-1]
        raw = lift.get("raw_result") if isinstance(lift.get("raw_result"), dict) else {}
        object_lift_world = raw.get("object_lift_world")
        min_object_lift = raw.get("min_object_lift", lift.get("parameters", {}).get("min_object_lift", 0.0))
        object_lift_success = raw.get("object_lift_success")
        require("object_lift_success", bool(object_lift_success), f"物体提升高度不足：object_lift_world={object_lift_world}, min_object_lift={min_object_lift}。")

    success = all_action_success and not failed_predicates
    if lift_reports and object_lift_success is not None:
        success = all_action_success and bool(object_lift_success) and not [p for p in failed_predicates if p != "object_lift_success"]
    return {
        "type": "field_atomic_grasp_lift_task",
        "success": bool(success),
        "all_action_success": bool(all_action_success),
        "required_phases": ["perception", "pregrasp", "approach", "close_gripper", "lift"],
        "executed_actions": actions,
        "failed_predicates": failed_predicates,
        "failure_descriptions": descriptions,
        "object_lift_world": object_lift_world,
        "min_object_lift": min_object_lift,
        "object_lift_success": object_lift_success,
    }


def _success_episode(task_summary: dict[str, Any]) -> bool:
    if not task_summary:
        return False
    if task_summary.get("object_lift_success") is not None:
        return bool(task_summary.get("object_lift_success"))
    return bool(task_summary.get("task_success", False))


def _first_failed_action(action_reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in action_reports:
        if not bool(item.get("success", False)):
            return item
    return None


def _infer_failure_taxonomy(failed_action: dict[str, Any] | None, task_summary: dict[str, Any]) -> dict[str, Any]:
    if failed_action is None:
        if task_summary.get("task_success") is False and task_summary.get("object_lift_success") is False:
            return {
                "failure_type": "object_not_lifted",
                "standard_failure_type": "object_not_lifted",
                "failure_stage": "task_summary",
                "failure_reason": "lift action completed but object lift check failed",
            }
        return {}

    action = _canonical_action(failed_action.get("action"))
    raw = failed_action.get("raw_result") if isinstance(failed_action.get("raw_result"), dict) else {}
    message = str(failed_action.get("message") or "")
    failure_type = "unknown_failure"
    reason = message or f"{action} failed"

    if action == "head_camera_grounded_sam2_pose":
        failure_type = "perception_miss"
    elif action in {"move_to_pregrasp", "approach_object"}:
        failure_type = "actuation_limit"
        final_error = raw.get("final_error")
        reason = f"{action} final_error={final_error}"
    elif action in {"close_gripper"}:
        failure_type = "grasp_miss"
    elif action == "lift":
        if raw.get("object_lift_success") is False or raw.get("object_lift_world") is not None:
            failure_type = "object_not_lifted"
            reason = f"object_lift_world={raw.get('object_lift_world')}, min_object_lift={raw.get('min_object_lift')}"
        else:
            failure_type = "actuation_limit"
    elif action == "transport_to_detected_target":
        if raw.get("object_follow_error") is not None:
            failure_type = "transport_collision"
            reason = f"object_follow_error={raw.get('object_follow_error')}"
        else:
            failure_type = "place_error"
    elif action == "lower_held_object":
        failure_type = "place_error"
    elif action == "plan_cartesian_trajectory":
        failure_type = "actuation_limit"
    elif action == "move_base_relative":
        failure_type = "actuation_limit"

    return {
        "failure_type": failure_type,
        "standard_failure_type": failure_type,
        "failure_stage": action,
        "failure_action_index": failed_action.get("index"),
        "failure_action": action,
        "failure_status": failed_action.get("status"),
        "failure_reason": reason,
        "final_error": raw.get("final_error"),
        "stage_errors": raw.get("stage_errors"),
        "stage_orientation_errors": raw.get("stage_orientation_errors"),
    }


def _extract_path_refs(value: Any, refs: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and (key.endswith("_path") or key.endswith("_dir") or key.endswith("_file")):
                refs[str(key)] = item
            else:
                _extract_path_refs(item, refs)
    elif isinstance(value, list):
        for item in value:
            _extract_path_refs(item, refs)


def _object_state_from_actions(action_reports: list[dict[str, Any]], task_summary: dict[str, Any]) -> ObjectState:
    objects: dict[str, Any] = {}
    target_object = ""
    object_class = ""
    for item in action_reports:
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        target_class = str(raw.get("target_class") or params.get("target_class") or "")
        object_body = str(raw.get("object_body") or params.get("object_body") or "")
        if target_class and not object_class:
            object_class = target_class
        if object_body and not target_object:
            target_object = object_body
        name = target_class or object_body
        if name:
            objects[name] = {
                "target_class": target_class,
                "object_body": object_body,
                "median_world": raw.get("median_world"),
                "center_world": raw.get("center_world"),
                "object_start_world": raw.get("object_start_world"),
                "object_final_world": raw.get("object_final_world"),
                "object_lift_world": raw.get("object_lift_world"),
            }
    if task_summary.get("object_body") and not target_object:
        target_object = str(task_summary.get("object_body"))
    return ObjectState(objects=objects, target_object=target_object, object_class=object_class)


def _load_source_failure(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.source_failure_report is not None and args.source_failure_report.exists():
        try:
            payload = json.loads(args.source_failure_report.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"load_error": str(exc), "path": str(args.source_failure_report)}
    if args.source_failure_experience_id:
        payload["source_failure_experience_id"] = args.source_failure_experience_id
    return payload


def _load_source_recovery_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_recovery_plan is None:
        return {}
    payload: dict[str, Any] = {"path": str(args.source_recovery_plan)}
    if args.source_recovery_plan.exists():
        try:
            data = json.loads(args.source_recovery_plan.read_text(encoding="utf-8"))
            payload["plan"] = data
        except Exception as exc:
            payload["load_error"] = str(exc)
    else:
        payload["missing"] = True
    return payload


def _restore_initial_state(model: mujoco.MjModel, data: mujoco.MjData, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"enabled": False}
    if not path.exists():
        return {"enabled": True, "path": str(path), "success": False, "error": "state_file_missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"enabled": True, "path": str(path), "success": False, "error": str(exc)}
    robot_state = payload.get("robot_state") if isinstance(payload, dict) else {}
    qpos = robot_state.get("qpos") if isinstance(robot_state, dict) else None
    qvel = robot_state.get("qvel") if isinstance(robot_state, dict) else None
    restored: dict[str, Any] = {"enabled": True, "path": str(path), "success": True, "restored_robot_qpos": False, "restored_robot_qvel": False, "restored_objects": []}
    try:
        if isinstance(qpos, list) and len(qpos) == model.nq:
            data.qpos[:] = np.asarray(qpos, dtype=float)
            restored["restored_robot_qpos"] = True
        if isinstance(qvel, list) and len(qvel) == model.nv:
            data.qvel[:] = np.asarray(qvel, dtype=float)
            restored["restored_robot_qvel"] = True
        object_state = payload.get("object_state") if isinstance(payload, dict) else {}
        if isinstance(object_state, dict):
            for name, state in object_state.items():
                if not isinstance(state, dict) or not bool(state.get("found", False)):
                    continue
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
                if body_id < 0:
                    continue
                freejoint = state.get("freejoint") if isinstance(state.get("freejoint"), dict) else {}
                joint_name = str(freejoint.get("joint_name") or "")
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) if joint_name else -1
                if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                qpos_values = freejoint.get("qpos")
                qvel_values = freejoint.get("qvel")
                if isinstance(qpos_values, list) and len(qpos_values) == 7:
                    qpos_adr = int(model.jnt_qposadr[joint_id])
                    data.qpos[qpos_adr : qpos_adr + 7] = np.asarray(qpos_values, dtype=float)
                if isinstance(qvel_values, list) and len(qvel_values) == 6:
                    qvel_adr = int(model.jnt_dofadr[joint_id])
                    data.qvel[qvel_adr : qvel_adr + 6] = np.asarray(qvel_values, dtype=float)
                restored["restored_objects"].append(str(name))
        mujoco.mj_forward(model, data)
    except Exception as exc:
        restored["success"] = False
        restored["error"] = str(exc)
    return restored


def _source_recovery_plan_validation(source_recovery_plan: dict[str, Any]) -> dict[str, Any]:
    plan = source_recovery_plan.get("plan") if isinstance(source_recovery_plan, dict) else {}
    if not isinstance(plan, dict):
        return {}
    if isinstance(plan.get("semantic_validation"), dict):
        return dict(plan.get("semantic_validation") or {})
    if isinstance(plan.get("field_atomic_plan"), dict) and isinstance(plan["field_atomic_plan"].get("semantic_validation"), dict):
        return dict(plan["field_atomic_plan"].get("semantic_validation") or {})
    return {}


def _extract_galaxea_retrieval_fields(action_reports: list[dict[str, Any]], task_summary: dict[str, Any], failure_taxonomy: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "failure_stage": failure_taxonomy.get("failure_stage", ""),
        "task_success": bool(task_summary.get("task_success", False)),
        "object_lift_success": task_summary.get("object_lift_success"),
        "object_body": task_summary.get("object_body") or "",
    }
    preferred_actions = {"move_to_pregrasp", "approach_object", "plan_cartesian_trajectory", "lift", "transport_to_detected_target"}
    for item in action_reports:
        action = _canonical_action(item.get("action"))
        params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        if action in preferred_actions or not fields.get("target_class"):
            for key in (
                "side",
                "target_class",
                "object_body",
                "topdown_mode",
                "mode",
                "pregrasp_offset_x",
                "pregrasp_offset_y",
                "pregrasp_offset_z",
                "visual_grasp_offset_z",
            ):
                value = params.get(key, raw.get(key))
                if value not in (None, "", [], {}):
                    out_key = "trajectory_mode" if key == "mode" else key
                    fields[out_key] = value
    return fields


def _build_episode_text_summary(task_summary: dict[str, Any], failure_taxonomy: dict[str, Any], action_reports: list[dict[str, Any]]) -> str:
    action_names = "->".join(_canonical_action(item.get("action")) for item in action_reports if item.get("action"))
    bits = [
        f"scenario={task_summary.get('scenario_id', '')}" if task_summary.get("scenario_id") else "",
        f"condition={task_summary.get('condition_id', '')}" if task_summary.get("condition_id") else "",
        f"role={task_summary.get('episode_role', 'execution')}",
        f"actions={action_names}",
        f"task_success={task_summary.get('task_success', False)}",
    ]
    failure_type = str(failure_taxonomy.get("failure_type") or "")
    failure_reason = str(failure_taxonomy.get("failure_reason") or "")
    if failure_type:
        bits.append(f"failure_type={failure_type}")
    if failure_reason:
        bits.append(f"failure_reason={failure_reason}")
    object_body = str(task_summary.get("object_body") or "")
    if object_body:
        bits.append(f"object={object_body}")
    source_failure = str(task_summary.get("source_failure_experience_id") or "")
    if source_failure:
        bits.append(f"source_failure={source_failure}")
    return "; ".join(bit for bit in bits if bit)


def _episode_parameter_summary(action_reports: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"actions": []}
    for item in action_reports:
        action = _canonical_action(item.get("action"))
        params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
        row = {
            "index": item.get("index"),
            "action": action,
            "success": bool(item.get("success", False)),
            "parameters": _jsonable({key: value for key, value in params.items() if not str(key).startswith("_")}),
            "final_error": raw.get("final_error"),
            "debug_tcp_minus_target_world": raw.get("debug_tcp_minus_target_world"),
            "debug_tcp_target_error_norm": raw.get("debug_tcp_target_error_norm"),
            "object_lift_world": raw.get("object_lift_world"),
        }
        summary["actions"].append({key: value for key, value in row.items() if value not in (None, "", [], {})})
        if action == "move_base_relative":
            summary["last_base_move_relative"] = row["parameters"]
        elif action == "set_torso_posture":
            summary["last_torso_posture"] = row["parameters"]
        elif action == "plan_cartesian_trajectory":
            summary["last_trajectory_plan"] = row["parameters"]
        elif action == "move_to_pregrasp":
            summary["last_pregrasp"] = row
        elif action == "approach_object":
            summary["last_approach"] = row
    return summary


def _compact_action_result(item: dict[str, Any]) -> dict[str, Any]:
    action = _canonical_action(item.get("action"))
    params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
    raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
    row: dict[str, Any] = {
        "index": item.get("index"),
        "action": action,
        "success": bool(item.get("success", False)),
        "status": item.get("status"),
        "message": item.get("message"),
        "parameters": _jsonable(params),
    }
    if action == "move_base_relative":
        row.update({
            "start_qpos": raw.get("start_qpos"),
            "target_qpos": raw.get("target_qpos"),
            "final_qpos": raw.get("final_qpos"),
            "final_error": raw.get("final_error"),
        })
    elif action == "set_torso_posture":
        row.update({
            "target_qpos": raw.get("target_qpos"),
            "final_qpos": raw.get("final_qpos"),
            "final_error": raw.get("final_error"),
        })
    elif action == "head_camera_grounded_sam2_pose":
        row.update({
            "target_class": raw.get("target_class"),
            "reference_frame": raw.get("reference_frame"),
            "median_reference": raw.get("median_reference"),
            "center_reference": raw.get("center_reference"),
            "median_world": raw.get("median_world"),
            "center_world": raw.get("center_world"),
            "bbox_xyxy": raw.get("bbox_xyxy"),
            "valid_depth_count": raw.get("valid_depth_count"),
            "position_output_path": raw.get("position_output_path"),
        })
    elif action == "plan_cartesian_trajectory":
        row.update({
            "target_torso": raw.get("target_torso"),
            "pregrasp_torso": raw.get("pregrasp_torso"),
            "start_torso": raw.get("start_torso"),
            "mode": raw.get("mode"),
            "waypoint_count": raw.get("waypoint_count"),
            "output_path": raw.get("output_path"),
        })
    elif action == "move_to_pregrasp":
        row.update({
            "target_torso": raw.get("target_torso"),
            "target_world": raw.get("target_world"),
            "pregrasp_torso": raw.get("pregrasp_torso"),
            "pregrasp_world": raw.get("pregrasp_world"),
            "final_tcp_torso": raw.get("final_tcp_torso"),
            "final_tcp_world": raw.get("final_tcp_world"),
            "final_tcp_minus_pregrasp_torso": raw.get("final_tcp_minus_pregrasp_torso"),
            "final_tcp_minus_pregrasp_world": raw.get("final_tcp_minus_pregrasp_world"),
            "final_tcp_pregrasp_error_norm": raw.get("final_tcp_pregrasp_error_norm"),
            "final_error": raw.get("final_error"),
            "stage_errors": raw.get("stage_errors"),
            "stage_orientation_errors": raw.get("stage_orientation_errors"),
            "debug_tcp_minus_target_world": raw.get("debug_tcp_minus_target_world"),
            "debug_tcp_target_error_norm": raw.get("debug_tcp_target_error_norm"),
        })
    elif action == "approach_object":
        row.update({
            "target_torso": raw.get("target_torso"),
            "target_world": raw.get("target_world"),
            "grasp_torso": raw.get("grasp_torso"),
            "grasp_world": raw.get("grasp_world"),
            "final_error": raw.get("final_error"),
            "stage_errors": raw.get("stage_errors"),
            "stage_orientation_errors": raw.get("stage_orientation_errors"),
        })
    elif action == "lift":
        row.update({
            "start_torso": raw.get("start_torso"),
            "target_torso": raw.get("target_torso"),
            "start_world": raw.get("start_world"),
            "target_world": raw.get("target_world"),
            "final_error": raw.get("final_error"),
            "object_body": raw.get("object_body"),
            "object_start_world": raw.get("object_start_world"),
            "object_final_world": raw.get("object_final_world"),
            "object_lift_world": raw.get("object_lift_world"),
            "stage_errors": raw.get("stage_errors"),
        })
    elif action in {"transport_to_detected_target", "transport_to_detected_target_with_offset"}:
        row.update({
            "start_torso": raw.get("start_torso"),
            "target_source_torso": raw.get("target_source_torso"),
            "target_torso": raw.get("target_torso"),
            "target_world": raw.get("target_world"),
            "target_z_mode": raw.get("target_z_mode"),
            "place_offset_torso": raw.get("place_offset_torso"),
            "final_error": raw.get("final_error"),
            "object_follow_error": raw.get("object_follow_error"),
            "stage_errors": raw.get("stage_errors"),
        })
    elif raw:
        row["result"] = _jsonable(raw)
    return {key: _jsonable(value) for key, value in row.items() if not _empty_summary_value(value)}


def _empty_summary_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _execution_summary_report(report: dict[str, Any], action_reports: list[dict[str, Any]]) -> dict[str, Any]:
    failed_action = next((item for item in action_reports if not bool(item.get("success", False))), None)
    return {
        "schema_version": "field_atomic_execution_summary_v1",
        "model_path": report.get("model_path"),
        "scenario_id": report.get("scenario_id"),
        "condition_id": report.get("condition_id"),
        "action_count": report.get("action_count"),
        "success_count": report.get("success_count"),
        "failure_count": report.get("failure_count"),
        "task_success": report.get("task_success"),
        "object_lift_success": report.get("object_lift_success"),
        "object_lift_world": report.get("object_lift_world"),
        "stopped_on_failure": report.get("stopped_on_failure"),
        "failed_action_index": failed_action.get("index") if isinstance(failed_action, dict) else None,
        "failed_action": _canonical_action(failed_action.get("action")) if isinstance(failed_action, dict) else "",
        "skills": [_compact_action_result(item) for item in action_reports],
    }


def _build_episode_critic(
    task_summary: dict[str, Any],
    failure_taxonomy: dict[str, Any],
    failed_action: dict[str, Any] | None,
    *,
    llm_critic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    success = _success_episode(task_summary)
    if success:
        return {
            "overall_status": "pass",
            "critic_risk_score": 0.0,
            "rule_flags": [],
            "feedback_for_rewrite": "",
            "evidence": {
                "task_success": True,
                "object_lift_success": task_summary.get("object_lift_success"),
            },
        }
    rule_flags = []
    failure_type = str(failure_taxonomy.get("failure_type") or "unknown_failure")
    failure_stage = str(failure_taxonomy.get("failure_stage") or "")
    rule_flags.append({
        "rule": failure_type,
        "stage": failure_stage,
        "severity": "block" if failure_type in {"actuation_limit", "grasp_miss", "object_not_lifted", "transport_collision"} else "warn",
        "evidence": str(failure_taxonomy.get("failure_reason") or ""),
        "action": str(failure_taxonomy.get("failure_action") or ""),
    })
    result = build_critic_result(
        rule_result={"enabled": True, "rule_flags": rule_flags},
        llm_result=llm_critic or {},
        is_failure=True,
    )
    if not result.get("feedback_for_rewrite"):
        result["feedback_for_rewrite"] = str(failure_taxonomy.get("failure_reason") or failure_type)
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    evidence.update({
        "failed_action": failed_action or {},
        "failure_taxonomy": failure_taxonomy,
    })
    result["evidence"] = evidence
    return result


def _merge_llm_critic_into_failure_taxonomy(failure_taxonomy: dict[str, Any], llm_critic: dict[str, Any]) -> dict[str, Any]:
    return dict(failure_taxonomy)


def _build_episode_gate(task_summary: dict[str, Any], failure_taxonomy: dict[str, Any], action_reports: list[dict[str, Any]]) -> dict[str, Any]:
    success = _success_episode(task_summary)
    has_failure = not success
    trigger_events = []
    if has_failure:
        trigger_events.extend(["skill_failure", failure_taxonomy.get("failure_type", "unknown_failure")])
    else:
        trigger_events.append("task_success")
    if task_summary.get("object_lift_success") is False:
        trigger_events.append("object_not_lifted")
    criteria = task_summary.get("task_success_criteria") if isinstance(task_summary.get("task_success_criteria"), dict) else {}
    if criteria.get("failed_predicates"):
        trigger_events.append("failed_predicates")
    gate = compute_memory_gate(
        {
            "condition_id": str(task_summary.get("condition_id") or ""),
            "anomaly_detected": has_failure,
            "skill_trace": action_reports,
            "critic_warnings": bool(failure_taxonomy.get("llm_critic")),
        },
        recovery_success=success,
        task_success=success,
        validation_status="simulation_validated" if success else "simulation_failed",
        sim_real_gap=None,
    )
    explanation = dict(gate.get("explanation") or {})
    explanation.update({
        "task_success": success,
        "action_count": len(action_reports),
        "failure_type": failure_taxonomy.get("failure_type", ""),
        "failed_predicates": criteria.get("failed_predicates", []),
        "has_structured_evidence": bool(criteria),
        "galaxea_field_atomic_gate": True,
    })
    gate["trigger_events"] = list(dict.fromkeys([*gate.get("trigger_events", []), *trigger_events]))
    gate["explanation"] = explanation
    return gate


def _build_anomaly_state(action_reports: list[dict[str, Any]], task_summary: dict[str, Any], failure_taxonomy: dict[str, Any]) -> dict[str, Any]:
    failed = _first_failed_action(action_reports) or {}
    raw = failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {}
    params = failed.get("parameters") if isinstance(failed.get("parameters"), dict) else {}
    target_torso = raw.get("target_torso") if isinstance(raw.get("target_torso"), list) else None
    final_error = raw.get("final_error")
    return {
        "failure_stage": failure_taxonomy.get("failure_stage", ""),
        "failure_type": failure_taxonomy.get("failure_type", ""),
        "target_class": params.get("target_class") or raw.get("target_class") or "",
        "side": params.get("side") or raw.get("side") or "",
        "target_torso_y_sign": _sign_bucket(target_torso[1]) if target_torso and len(target_torso) > 1 else "",
        "final_error_bucket": _error_bucket(final_error),
        "object_lift_bucket": _lift_bucket(task_summary.get("object_lift_world")),
        "task_success": bool(task_summary.get("task_success", False)),
        "failed_predicates": (task_summary.get("task_success_criteria") or {}).get("failed_predicates", []) if isinstance(task_summary.get("task_success_criteria"), dict) else [],
    }


def _sign_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric > 0.03:
        return "positive"
    if numeric < -0.03:
        return "negative"
    return "center"


def _error_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric < 0.02:
        return "low"
    if numeric < 0.10:
        return "medium"
    return "high"


def _lift_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric >= 0.095:
        return "success"
    if numeric > 0.01:
        return "partial"
    return "none"


def _build_episode_experience_entry(
    *,
    args: argparse.Namespace,
    runtime_tmp_dir: Path,
    action_reports: list[dict[str, Any]],
    task_summary: dict[str, Any],
    keyframes: list[dict[str, Any]],
    keyframe_errors: list[dict[str, Any]],
    stopped_on_failure: bool,
) -> ExperienceEntry | None:
    failed_action = _first_failed_action(action_reports)
    if failed_action is None and bool(task_summary.get("task_success", True)):
        return None

    failure_taxonomy = _infer_failure_taxonomy(failed_action, task_summary)
    raw_refs: dict[str, Any] = {
        "runtime_tmp_dir": str(runtime_tmp_dir),
        "actions_path": str(args.actions) if args.actions else "",
        "model_path": str(args.model_path),
        "keyframe_dir": str(args.keyframe_dir) if args.keyframe_dir else "",
        "initial_state": str(args.initial_state) if args.initial_state else "",
    }
    source_failure = _load_source_failure(args)
    if source_failure:
        raw_refs["source_failure"] = source_failure
    if args.source_failure_report is not None:
        raw_refs["source_failure_report"] = str(args.source_failure_report)
    source_recovery_plan = _load_source_recovery_plan(args)
    if source_recovery_plan:
        raw_refs["source_recovery_plan"] = source_recovery_plan
    source_recovery_plan_validation = _source_recovery_plan_validation(source_recovery_plan)
    path_refs: dict[str, str] = {}
    _extract_path_refs(action_reports, path_refs)
    if path_refs:
        raw_refs["artifact_paths"] = path_refs

    failed_index = int(failed_action.get("index", -1)) if failed_action is not None else -1
    relevant_keyframes = [
        item for item in keyframes
        if isinstance(item, dict) and (failed_index < 0 or item.get("index") in {None, failed_index})
    ]
    skill_sequence = [
        SkillTraceItem(
            name=_canonical_action(item.get("action")),
            primitive_type="field_atomic",
            phase="execution",
            inputs={"parameters": _jsonable(item.get("parameters", {}))},
            outputs={"result": _jsonable(item.get("raw_result", {}))},
            success=bool(item.get("success", False)),
            message=str(item.get("message") or ""),
            error=(item.get("raw_result") or {}).get("final_error") if isinstance(item.get("raw_result"), dict) else None,
            raw={"index": item.get("index"), "status": item.get("status")},
        )
        for item in action_reports
    ]
    task_summary = dict(task_summary)
    task_summary["scenario_id"] = args.scenario_id
    task_summary["condition_id"] = args.condition_id
    task_summary["episode_role"] = args.episode_role
    if args.source_failure_experience_id:
        task_summary["source_failure_experience_id"] = args.source_failure_experience_id
    success = _success_episode(task_summary)
    llm_critic: dict[str, Any] = {}
    if not success and bool(args.llm_critic):
        llm_critic = critique_field_atomic_failure(
            goal=str(args.goal or ""),
            scenario_id=str(args.scenario_id or ""),
            episode_role=str(args.episode_role or ""),
            action_reports=action_reports,
            task_summary=task_summary,
            failure_taxonomy=failure_taxonomy,
            image_paths=_critic_image_paths(relevant_keyframes),
            provider=str(args.llm_critic_provider or "doubao"),
            model=str(args.llm_critic_model or ""),
        )
    critic_result = _build_episode_critic(task_summary, failure_taxonomy, failed_action, llm_critic=llm_critic)
    memory_gate = _build_episode_gate(task_summary, failure_taxonomy, action_reports)
    text_summary = _build_episode_text_summary(task_summary, failure_taxonomy, action_reports)
    memory_lesson = field_atomic_memory_lesson(llm_critic)
    anomaly_state = _build_anomaly_state(action_reports, task_summary, failure_taxonomy)
    sensor_summary = SensorSummary(
        sensor_modalities=["rgb", "depth", "mujoco_state"],
        raw_refs=raw_refs,
        timestamps={"episode": args.scenario_id},
    )
    sensor_evidence = SensorEvidence(
        modalities=["rgb", "depth", "mujoco_state"],
        evidence_refs=raw_refs,
        summary={
            "failed_action": failure_taxonomy.get("failure_action", ""),
            "failure_reason": failure_taxonomy.get("failure_reason", ""),
            "final_error": failure_taxonomy.get("final_error", ""),
            "failed_predicates": _jsonable((task_summary.get("task_success_criteria") or {}).get("failed_predicates", []) if isinstance(task_summary.get("task_success_criteria"), dict) else []),
            "anomaly_state": {
                "failure_stage": anomaly_state.get("failure_stage", ""),
                "failure_type": anomaly_state.get("failure_type", ""),
                "target_torso_y_sign": anomaly_state.get("target_torso_y_sign", ""),
                "final_error_bucket": anomaly_state.get("final_error_bucket", ""),
            },
        },
    )
    result = {
        "success": success,
        "task_success": success,
        "failure_reason": failure_taxonomy.get("failure_reason", ""),
        "failed_action": failure_taxonomy.get("failure_action", ""),
        "failed_action_index": failed_index,
        "stopped_on_failure": bool(stopped_on_failure),
        "task_success_criteria": _jsonable(task_summary.get("task_success_criteria") or {}),
        "recovery_success_criteria": _jsonable(task_summary.get("recovery_success_criteria") or {}),
        **_jsonable(task_summary),
    }
    if args.episode_role == "recovery":
        memory_role = "field_atomic_recovery_success_episode" if success else "field_atomic_recovery_failure_episode"
        task_name = "field_atomic_recovery_execution"
        anomaly_trigger = "recovery_success" if success else "recovery_failure"
    else:
        memory_role = "field_atomic_success_episode" if success else "field_atomic_failure_episode"
        task_name = "field_atomic_execution"
        anomaly_trigger = "skill_failure" if failed_action is not None else ("task_success" if success else "task_failure")
    source_failed_action = str(source_failure.get("failed_action") or "")

    entry = ExperienceEntry(
        source="simulation",
        backend="mujoco",
        validation_status="simulation_validated" if success else "simulation_failed",
        skill_namespace="galaxea_r1pro_torso",
        skill_catalog_version="v1",
        scenario={"scenario_id": args.scenario_id},
        condition={"condition_id": args.condition_id},
        task={
            "name": task_name,
            "stage": failure_taxonomy.get("failure_stage", "task") if not success else "task_success",
            "goal": str(args.goal or "execute field atomic skill sequence and record outcome"),
        },
        anomaly={
            "is_anomaly": not success,
            "detected_by": "field_atomic_executor",
            "trigger": anomaly_trigger,
            "failed_action_index": failed_index,
            "failed_action": failure_taxonomy.get("failure_action", ""),
            "stopped_on_failure": bool(stopped_on_failure),
            "source_failure_experience_id": args.source_failure_experience_id,
            "source_failed_action": source_failed_action,
        },
        robot={"robot_type": "mobile_dual_arm", "backend": "mujoco", "embodiment_tags": ["galaxea", "r1pro", "torso_frame"]},
        object_state=_object_state_from_actions(action_reports, task_summary),
        skill_sequence=skill_sequence,
        action_trace=_jsonable(action_reports),
        sensor_summary=sensor_summary,
        sensor_evidence=sensor_evidence,
        result=result,
        execution_feedback={
            "failed_action": _jsonable(failed_action or {}),
            "episode_parameter_summary": _episode_parameter_summary(action_reports),
            "action_count": len(action_reports),
            "success_count": sum(1 for item in action_reports if item.get("success")),
            "failure_count": sum(1 for item in action_reports if not item.get("success")),
            "keyframe_errors": _jsonable(keyframe_errors),
            "task_summary": _jsonable(task_summary),
            "task_success_criteria": _jsonable(task_summary.get("task_success_criteria") or {}),
            "recovery_success_criteria": _jsonable(task_summary.get("recovery_success_criteria") or {}),
            "source_failure": _jsonable(source_failure),
            "source_recovery_plan": _jsonable(source_recovery_plan),
            "source_recovery_plan_validation": _jsonable(source_recovery_plan_validation),
            "recovery_plan_signature": "->".join(_canonical_action(item.get("action")) for item in action_reports if item.get("action")),
            "recovery_success": success if args.episode_role == "recovery" else None,
            "llm_critic": _jsonable(llm_critic),
        },
        keyframes=_jsonable(relevant_keyframes),
        failure_taxonomy=failure_taxonomy,
        raw_refs=raw_refs,
        memory_tags={
            "memory_type": "field_atomic_episode",
            "memory_role": memory_role,
            "robot": "galaxea_r1pro",
            "skill_namespace": "galaxea_r1pro_torso",
            "failure_type": failure_taxonomy.get("failure_type", ""),
            "episode_role": args.episode_role,
        },
        memory_gate=MemoryGate(**memory_gate),
        critic_result=CriticResult(**critic_result),
        retrieval_key={},
        state_before={},
        state_after={},
        observation_trace=[],
        spatial_state={"anomaly_state": _jsonable(anomaly_state)},
        metadata={
            "field_atomic": True,
            "episode_level": True,
            "model_path": str(args.model_path),
            "runtime_tmp_dir": str(runtime_tmp_dir),
            "stop_on_failure": bool(args.stop_on_failure),
            "text_summary": text_summary,
            "source_failure_experience_id": args.source_failure_experience_id,
            "source_recovery_plan": _jsonable(source_recovery_plan),
            "source_recovery_plan_validation": _jsonable(source_recovery_plan_validation),
            "anomaly_state": _jsonable(anomaly_state),
        },
    )
    galaxea_fields = _extract_galaxea_retrieval_fields(action_reports, task_summary, failure_taxonomy)
    if args.source_failure_experience_id:
        galaxea_fields["source_failure_experience_id"] = args.source_failure_experience_id
    galaxea_fields["episode_role"] = args.episode_role
    episode_retrieval_key = {
        "scenario_id": args.scenario_id,
        "condition_id": args.condition_id,
        "skill_namespace": "galaxea_r1pro_torso",
        "task_stage": str(entry.task.get("stage") or ""),
        "plan_signature": "->".join(_canonical_action(item.get("action")) for item in action_reports if item.get("action")),
        "failure_type": failure_taxonomy.get("failure_type", ""),
        "critic_status": entry.critic_result.overall_status,
        "memory_type": entry.memory_tags.get("memory_type", ""),
        "memory_role": entry.memory_tags.get("memory_role", ""),
        "object_class": entry.object_state.object_class,
        "target_object": entry.object_state.target_object,
        "task_success": bool(entry.result.get("task_success", False)),
        "failure_stage": failure_taxonomy.get("failure_stage", ""),
        "episode_role": args.episode_role,
        "source_failure_experience_id": args.source_failure_experience_id,
        "source_recovery_plan_valid": source_recovery_plan_validation.get("valid") if source_recovery_plan_validation else None,
        "source_recovery_plan_failed_checks": source_recovery_plan_validation.get("failed_checks", []) if source_recovery_plan_validation else [],
        "anomaly_state": _jsonable(anomaly_state),
        "failed_predicates": anomaly_state.get("failed_predicates", []),
        "target_torso_y_sign": anomaly_state.get("target_torso_y_sign", ""),
        "final_error_bucket": anomaly_state.get("final_error_bucket", ""),
        **galaxea_fields,
    }
    entry.retrieval_key = {**build_retrieval_key(entry), **episode_retrieval_key}
    entry.metadata["text_summary"] = text_summary
    entry.metadata["galaxea_retrieval_fields"] = _jsonable(galaxea_fields)
    return entry


def _keyframes_for_action(keyframes: list[dict[str, Any]], index: int, *, base_dir: Path | None = None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in keyframes:
        if isinstance(frame, dict) and frame.get("index") == index:
            item = dict(frame)
            image_path = str(item.get("image_path") or "")
            if image_path and base_dir is not None:
                try:
                    item["image_path"] = str(Path(image_path).resolve().relative_to(base_dir.resolve()))
                except ValueError:
                    item["image_path"] = image_path
            selected.append(item)
    return selected


def _critic_image_paths(keyframes: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    preferred: list[str] = []
    fallback: list[str] = []
    for frame in keyframes or []:
        if not isinstance(frame, dict):
            continue
        path = str(frame.get("image_path") or "")
        if not path or not Path(path).exists():
            continue
        stage = str(frame.get("stage") or "").lower()
        if "failed" in stage or "after" in stage or "before" in stage:
            preferred.append(path)
        else:
            fallback.append(path)
    selected = list(dict.fromkeys([*preferred, *fallback]))
    return selected[: max(0, int(limit))]


def _build_visual_index_for_library(library: ExperienceLibrary, *, index_dir: Path, base_dir: Path) -> dict[str, Any]:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index = VisualRetrievalIndex()
    added_entry_count = 0
    added_image_count = 0
    entries = []
    for entry in library.entries:
        if not _is_galaxea_episode_entry(entry):
            continue
        image_paths = image_paths_from_entry(entry, base_dir=base_dir)
        image_paths = _select_visual_index_images(image_paths)
        added = index.add(entry.experience_id, image_paths)
        if added:
            added_entry_count += 1
            added_image_count += added
        entries.append({
            "experience_id": entry.experience_id,
            "memory_role": entry.memory_tags.get("memory_role", ""),
            "image_count": len(image_paths),
            "indexed_image_count": added,
        })
    index.save(index_dir)
    return {
        "visual_index_dir": str(index_dir),
        "visual_index_added_entry_count": added_entry_count,
        "visual_index_added_image_count": added_image_count,
        "visual_index_faiss_size": index.size,
        "entries": entries,
    }


def _is_galaxea_episode_entry(entry: ExperienceEntry) -> bool:
    if str(entry.skill_namespace or "") != "galaxea_r1pro_torso":
        return False
    if str(entry.memory_tags.get("memory_type") or "") != "field_atomic_episode":
        return False
    text = json.dumps(entry.to_dict(), ensure_ascii=False).lower()
    legacy_markers = ("g3", "g4", "pseudo_real_g3", "pseudo_real_g4", "r1pro_g3", "r1pro_g4")
    return not any(marker in text for marker in legacy_markers)


def _select_visual_index_images(image_paths: list[str], *, limit: int = 5) -> list[str]:
    if not image_paths:
        return []
    preferred_tokens = (
        "failed_step",
        "after_step",
        "final_state",
        "scene_initial",
        "before_step",
    )
    selected: list[str] = []
    for token in preferred_tokens:
        for path in image_paths:
            if token in str(path) and path not in selected:
                selected.append(path)
            if len(selected) >= limit:
                return selected
    for path in image_paths:
        if path not in selected:
            selected.append(path)
        if len(selected) >= limit:
            break
    return selected


def _should_write_experience_entry(entry: ExperienceEntry, *, mode: str) -> bool:
    if mode == "none":
        return False
    memory_type = str(entry.memory_tags.get("memory_type") or "")
    memory_role = str(entry.memory_tags.get("memory_role") or "")
    is_episode = memory_type == "field_atomic_episode"
    is_atomic_failure = memory_type == "field_atomic_experience" and memory_role == "field_atomic_failure"
    if not is_episode and not is_atomic_failure:
        return False
    is_success = bool(entry.result.get("success", entry.result.get("recovery_success", False)))
    if mode == "success_only" and not is_success:
        return False
    if mode == "failure_only" and is_success:
        return False
    return True


def _restore_galaxea_retrieval_keys(library: ExperienceLibrary) -> None:
    for entry in library.entries:
        fields = entry.metadata.get("galaxea_retrieval_fields") if isinstance(entry.metadata, dict) else {}
        if not isinstance(fields, dict):
            continue
        extra = {
            "episode_role": entry.memory_tags.get("episode_role", ""),
            "source_failure_experience_id": entry.metadata.get("source_failure_experience_id", ""),
            "failure_stage": entry.failure_taxonomy.get("failure_stage", ""),
            "task_success": bool(entry.result.get("task_success", False)),
            **fields,
        }
        entry.retrieval_key = {**entry.retrieval_key, **{key: value for key, value in extra.items() if value not in (None, "", [], {})}}


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            _run_sequence(model, data, args, viewer=viewer)
        return
    _run_sequence(model, data, args, viewer=None)


def _run_sequence(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace, *, viewer: Any | None) -> None:
    original_mj_step = mujoco.mj_step
    step_count = 0

    def sync_viewer(force: bool = False) -> None:
        if viewer is None or not viewer.is_running():
            return
        sync_every = max(1, int(args.viewer_sync_every))
        if force or step_count % sync_every == 0:
            viewer.sync()
            time.sleep(float(model.opt.timestep))

    def patched_mj_step(step_model: mujoco.MjModel, step_data: mujoco.MjData, *step_args: Any, **step_kwargs: Any) -> Any:
        nonlocal step_count
        result = original_mj_step(step_model, step_data, *step_args, **step_kwargs)
        step_count += 1
        sync_viewer()
        return result

    if viewer is not None:
        mujoco.mj_step = patched_mj_step
    try:
        _execute_sequence(model, data, args, sync_viewer, viewer=viewer)
    finally:
        if viewer is not None:
            mujoco.mj_step = original_mj_step


def _execute_sequence(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace, sync_viewer, *, viewer: Any | None) -> None:
    runtime_tmp_dir = _make_runtime_tmp_dir()
    recorder = KeyframeRecorder(args.keyframe_dir, camera=args.keyframe_camera)
    mujoco.mj_forward(model, data)
    initial_state_restore = _restore_initial_state(model, data, args.initial_state)
    sync_viewer(force=True)
    try:
        for _ in range(max(0, int(args.settle_before_steps))):
            mujoco.mj_step(model, data)
        recorder.capture(model, data, "scene_initial", description="Initial scene before field atomic execution")
        executor = FieldAtomicSkillExecutor()
        action_reports: list[dict[str, Any]] = []
        entries = []
        stopped_on_failure = False

        for index, item in enumerate(_load_actions(args.actions)):
            action = str(item.get("action") or "")
            parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
            parameters = dict(parameters)
            parameters.setdefault("_runtime_tmp_dir", str(runtime_tmp_dir))
            action_name = _canonical_action(action) or action or "unknown"
            recorder.capture(
                model,
                data,
                f"before_step_{index:03d}_{action_name}",
                description=f"Before step {index}: {action_name}",
                action=action_name,
                index=index,
            )
            try:
                result = executor.execute(model, data, action, parameters)
            except Exception as exc:
                from skills.field_atomic.atomic_schema import FieldAtomicResult

                result = FieldAtomicResult(action=action, success=False, status="exception", message=str(exc), parameters=dict(parameters))
            recorder.capture(
                model,
                data,
                f"after_step_{index:03d}_{action_name}",
                description=f"After step {index}: {action_name}",
                action=action_name,
                index=index,
            )
            if not bool(result.success):
                recorder.capture(
                    model,
                    data,
                    f"failed_step_{index:03d}_{action_name}",
                    description=f"Failed step {index}: {action_name}",
                    action=action_name,
                    index=index,
                )
            action_reports.append({
                "index": index,
                "action": action,
                "success": bool(result.success),
                "status": result.status,
                "message": result.message,
                "parameters": public_field_atomic_parameters(dict(result.parameters)),
                "raw_result": dict(result.raw_result),
            })
            _print_action_trace(model, data, index, action, result)
            sync_viewer(force=True)
            entry = build_atomic_experience_entry(
                scenario_id=args.scenario_id,
                condition_id=args.condition_id,
                robot_type="mobile_dual_arm",
                action=_canonical_action(action),
                result=result,
            )
            entry.keyframes = _keyframes_for_action(
                recorder.keyframes,
                index,
                base_dir=ROOT.parent,
            )
            entries.append(entry)
            if args.stop_on_failure and not bool(result.success):
                stopped_on_failure = True
                break

        recorder.capture(model, data, "final_state", description="Final state after field atomic execution")
        task_summary = _task_summary(action_reports)
        report = {
            "schema_version": "field_atomic_skill_smoke_report_v1",
            "model_path": args.model_path,
            "scenario_id": args.scenario_id,
            "condition_id": args.condition_id,
            "settle_before_steps": max(0, int(args.settle_before_steps)),
            "initial_state_restore": _jsonable(initial_state_restore),
            "runtime_tmp_dir": str(runtime_tmp_dir),
            "action_count": len(action_reports),
            "success_count": sum(1 for item in action_reports if item["success"]),
            "failure_count": sum(1 for item in action_reports if not item["success"]),
            "stopped_on_failure": stopped_on_failure,
            **task_summary,
            "keyframes": list(recorder.keyframes),
            "keyframe_errors": list(recorder.errors),
            "actions": action_reports,
        }
        episode_entry = _build_episode_experience_entry(
            args=args,
            runtime_tmp_dir=runtime_tmp_dir,
            action_reports=action_reports,
            task_summary=task_summary,
            keyframes=list(recorder.keyframes),
            keyframe_errors=list(recorder.errors),
            stopped_on_failure=stopped_on_failure,
        )
        if episode_entry is not None:
            entries.append(episode_entry)
            report["experience_failure_episode_id"] = episode_entry.experience_id
            report["failure_taxonomy"] = dict(episode_entry.failure_taxonomy)
        if args.save_experience_library is not None:
            existing = ExperienceLibrary.load(args.experience_read) if args.experience_read else ExperienceLibrary()
            library = existing
            write_decisions = []
            episode_entry_ids = []
            failure_cluster_report = {"enabled": True, "assigned": []}
            clusterer: FailureClusterer | None = None
            for entry in entries:
                if not _should_write_experience_entry(entry, mode=args.experience_save_mode):
                    continue
                decision = library.add_with_policy(entry, strict_quality=False, merge_duplicates=True)
                if not bool(entry.result.get("success", False)) or "failure" in str(entry.memory_tags.get("memory_role") or ""):
                    try:
                        if clusterer is None:
                            clusterer = FailureClusterer()
                            existing_failures = [
                                item for item in library.entries
                                if (not bool(item.result.get("success", False)) or "failure" in str(item.memory_tags.get("memory_role") or ""))
                            ]
                            if len(existing_failures) >= 2:
                                clusterer.cluster(existing_failures)
                        cid = clusterer.assign_new(entry, library.entries) if clusterer is not None else "noise"
                        failure_cluster_report["assigned"].append({"experience_id": entry.experience_id, "cluster_id": cid})
                    except Exception as exc:
                        failure_cluster_report["error"] = str(exc)
                entry.metadata = dict(entry.metadata or {})
                entry.metadata["write_policy"] = decision
                write_decisions.append(decision)
                if entry.memory_tags.get("memory_type") == "field_atomic_episode":
                    episode_entry_ids.append(entry.experience_id)
            if args.apply_memory_lifecycle:
                library.entries, lifecycle_report = consolidate_memory_lifecycle(
                    library.entries,
                    stm_capacity=max(1, int(args.stm_capacity)),
                    promote_failures=True,
                    promote_validated_success=True,
                )
                report["memory_lifecycle_report"] = lifecycle_report
                report["stm_count"] = lifecycle_report.get("stm_count")
                report["ltm_count"] = lifecycle_report.get("ltm_count")
            _restore_galaxea_retrieval_keys(library)
            _write_json(args.save_experience_library, library.to_dict())
            report["experience_library_output"] = str(args.save_experience_library)
            report["experience_write_decisions"] = write_decisions
            report["failure_cluster_report"] = failure_cluster_report
            report["experience_saved_entry_count"] = len(library.entries)
            report["experience_episode_entry_ids"] = episode_entry_ids
            if not args.skip_visual_index:
                visual_index_dir = args.visual_index_dir or (args.save_experience_library.parent / "visual_index")
                try:
                    visual_report = _build_visual_index_for_library(
                        library,
                        index_dir=visual_index_dir,
                        base_dir=ROOT.parent,
                    )
                    report["visual_index_report"] = visual_report
                except Exception as exc:
                    report["visual_index_report"] = {
                        "visual_index_dir": str(visual_index_dir),
                        "error": str(exc),
                    }
        summary_path = None
        if args.save_report is not None:
            summary_path = args.save_report.with_name(f"{args.save_report.stem}_execution_summary.json")
            _write_json(summary_path, _execution_summary_report(report, action_reports))
            report["execution_summary_report"] = str(summary_path)
        if args.save_report is not None:
            _write_json(args.save_report, report)
        print(json.dumps({
            "action_count": report["action_count"],
            "success_count": report["success_count"],
            "failure_count": report["failure_count"],
            "task_success": report.get("task_success"),
            "lift_action_success": report.get("lift_action_success"),
            "object_lift_success": report.get("object_lift_success"),
            "object_body": report.get("object_body"),
            "object_lift_world": report.get("object_lift_world"),
            "min_object_lift": report.get("min_object_lift"),
            "runtime_tmp_dir": str(runtime_tmp_dir),
            "stopped_on_failure": report.get("stopped_on_failure"),
            "save_report": str(args.save_report) if args.save_report else "",
            "execution_summary_report": str(summary_path) if summary_path else "",
            "save_experience_library": str(args.save_experience_library) if args.save_experience_library else "",
        }, ensure_ascii=False))
        if args.viewer and args.viewer_hold_seconds != 0.0:
            if float(args.viewer_hold_seconds) < 0.0:
                while viewer is not None and viewer.is_running():
                    mujoco.mj_step(model, data)
                    sync_viewer(force=True)
                    time.sleep(0.03)
            else:
                end_time = time.time() + float(args.viewer_hold_seconds)
                while time.time() < end_time and (viewer is None or viewer.is_running()):
                    mujoco.mj_step(model, data)
                    sync_viewer(force=True)
                    time.sleep(0.03)
    finally:
        recorder.close()
        if bool(args.cleanup_runtime_tmp):
            shutil.rmtree(runtime_tmp_dir, ignore_errors=True)


def _make_runtime_tmp_dir() -> Path:
    root = ROOT / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = f"{stamp}_{index:03d}"
        path = root / f"field_atomic_run_{suffix}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Failed to create runtime tmp directory")


def _print_action_trace(model: mujoco.MjModel, data: mujoco.MjData, index: int, action: str, result: Any) -> None:
    action = _canonical_action(action)
    raw = dict(getattr(result, "raw_result", {}) or {})
    parameters = dict(getattr(result, "parameters", {}) or {})
    payload: dict[str, Any] = {
        "index": index,
        "action": action,
        "success": bool(getattr(result, "success", False)),
        "status": getattr(result, "status", ""),
    }
    if action == "head_camera_grounded_sam2_pose":
        payload.update({
            "target_class": raw.get("target_class"),
            "reference_frame": raw.get("reference_frame"),
            "median_reference": raw.get("median_reference"),
            "center_reference": raw.get("center_reference"),
            "median_world": raw.get("median_world"),
            "center_world": raw.get("center_world"),
            "center_camera_cv": raw.get("center_camera_cv"),
            "bbox_xyxy": raw.get("bbox_xyxy"),
            "valid_depth_count": raw.get("valid_depth_count"),
            "position_output_path": raw.get("position_output_path"),
        })
        _add_object_position_debug(model, data, payload, parameters, raw, detected_key="median_world")
    elif action == "plan_cartesian_trajectory":
        payload.update({
            "target_torso": raw.get("target_torso"),
            "pregrasp_torso": raw.get("pregrasp_torso"),
            "start_torso": raw.get("start_torso"),
            "mode": raw.get("mode"),
            "waypoint_count": raw.get("waypoint_count"),
            "output_path": raw.get("output_path"),
        })
    elif action == "move_to_pregrasp":
        payload.update({
            "target_torso": raw.get("target_torso"),
            "target_world": raw.get("target_world"),
            "pregrasp_torso": raw.get("pregrasp_torso"),
            "pregrasp_world": raw.get("pregrasp_world"),
            "final_error": raw.get("final_error"),
            "stage_errors": raw.get("stage_errors"),
        })
        _add_tcp_debug(model, data, payload, parameters, raw, target_key="pregrasp_world")
        _add_object_position_debug(model, data, payload, parameters, raw, detected_key="target_world")
    elif action == "approach_object":
        payload.update({
            "target_torso": raw.get("target_torso"),
            "target_world": raw.get("target_world"),
            "grasp_torso": raw.get("grasp_torso"),
            "grasp_world": raw.get("grasp_world"),
            "final_error": raw.get("final_error"),
            "stage_errors": raw.get("stage_errors"),
        })
        _add_tcp_debug(model, data, payload, parameters, raw, target_key="grasp_world")
        _add_object_position_debug(model, data, payload, parameters, raw, detected_key="target_world")
    elif action == "lift":
        payload.update({
            "start_torso": raw.get("start_torso"),
            "target_torso": raw.get("target_torso"),
            "start_world": raw.get("start_world"),
            "target_world": raw.get("target_world"),
            "final_error": raw.get("final_error"),
            "object_body": raw.get("object_body"),
            "object_start_world": raw.get("object_start_world"),
            "object_final_world": raw.get("object_final_world"),
            "object_lift_world": raw.get("object_lift_world"),
            "stage_errors": raw.get("stage_errors"),
        })
        _add_tcp_debug(model, data, payload, parameters, raw, target_key="target_world")
        _add_object_position_debug(model, data, payload, parameters, raw, detected_key="start_world")
    elif action in {"transport_to_detected_target", "transport_to_detected_target_with_offset"}:
        payload.update({
            "start_torso": raw.get("start_torso"),
            "target_source_torso": raw.get("target_source_torso"),
            "target_torso": raw.get("target_torso"),
            "target_z_mode": raw.get("target_z_mode"),
            "place_offset_torso": raw.get("place_offset_torso"),
            "start_world": raw.get("start_world"),
            "target_world": raw.get("target_world"),
            "final_error": raw.get("final_error"),
            "object_follow_error": raw.get("object_follow_error"),
            "stage_errors": raw.get("stage_errors"),
        })
        _add_tcp_debug(model, data, payload, parameters, raw, target_key="target_world")
        _add_object_position_debug(model, data, payload, parameters, raw, detected_key="target_world")
    elif action == "frame_alignment_debug":
        payload.update(raw)
    else:
        message = getattr(result, "message", "")
        if message:
            payload["message"] = message
    print("TRACE " + json.dumps(_jsonable(payload), ensure_ascii=False))



def _add_object_position_debug(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload: dict[str, Any],
    parameters: dict[str, Any],
    raw: dict[str, Any],
    *,
    detected_key: str,
) -> None:
    object_body = str(parameters.get("object_body") or raw.get("object_body") or "")
    if not object_body:
        return
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if body_id < 0:
        payload["debug_object_body"] = object_body
        payload["debug_object_error"] = "body_not_found"
        return
    mujoco.mj_forward(model, data)
    body_world = data.xpos[body_id].copy()
    payload["debug_object_body"] = object_body
    payload["debug_object_world"] = np.round(body_world, 6).tolist()
    detected = raw.get(detected_key)
    if detected is None and detected_key != "median_world":
        detected = raw.get("median_world")
    if detected is None:
        return
    try:
        detected_world = np.asarray(detected, dtype=float).reshape(3)
    except Exception:
        return
    delta = detected_world - body_world
    payload["debug_detected_minus_object_world"] = np.round(delta, 6).tolist()
    payload["debug_detected_object_error_norm"] = float(np.linalg.norm(delta))


def _add_tcp_debug(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload: dict[str, Any],
    parameters: dict[str, Any],
    raw: dict[str, Any],
    *,
    target_key: str,
) -> None:
    side = str(parameters.get("side") or raw.get("side") or "")
    if side not in {"left", "right"}:
        return
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_tool")
    if site_id < 0:
        payload["debug_tcp_error"] = f"site_not_found:{side}_grasp_tool"
        return
    mujoco.mj_forward(model, data)
    tcp_world = data.site_xpos[site_id].copy()
    tcp_xmat = data.site_xmat[site_id].reshape(3, 3).copy()
    payload["debug_tcp_world"] = np.round(tcp_world, 6).tolist()
    payload["debug_tcp_x_axis_world"] = np.round(tcp_xmat[:, 0], 6).tolist()
    payload["debug_tcp_y_axis_world"] = np.round(tcp_xmat[:, 1], 6).tolist()
    payload["debug_tcp_z_axis_world"] = np.round(tcp_xmat[:, 2], 6).tolist()
    target = raw.get(target_key)
    if target is None:
        return
    try:
        target_world = np.asarray(target, dtype=float).reshape(3)
    except Exception:
        return
    delta = tcp_world - target_world
    payload["debug_tcp_minus_target_world"] = np.round(delta, 6).tolist()
    payload["debug_tcp_target_error_norm"] = float(np.linalg.norm(delta))


if __name__ == "__main__":
    main()
