from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE_ROOT = ROOT.parent / "experience_system"
for path in (ROOT, EXPERIENCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from skills.field_atomic import FieldAtomicSkillExecutor, build_atomic_experience_entry


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
    parser.add_argument("--actions", type=Path, default=None, help="JSON file containing a list of {action, parameters}")
    parser.add_argument("--scenario-id", default="field_atomic_smoke")
    parser.add_argument("--condition-id", default="default")
    parser.add_argument("--save-report", type=Path, default=None)
    parser.add_argument("--save-experience-library", type=Path, default=None)
    return parser.parse_args()


def _load_actions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_ACTIONS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--actions must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


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


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    executor = FieldAtomicSkillExecutor()
    action_reports: list[dict[str, Any]] = []
    entries = []

    for index, item in enumerate(_load_actions(args.actions)):
        action = str(item.get("action") or "")
        parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        try:
            result = executor.execute(model, data, action, parameters)
        except Exception as exc:
            from skills.field_atomic.atomic_schema import FieldAtomicResult

            result = FieldAtomicResult(action=action, success=False, status="exception", message=str(exc), parameters=dict(parameters))
        action_reports.append({
            "index": index,
            "action": action,
            "success": bool(result.success),
            "status": result.status,
            "message": result.message,
            "parameters": dict(result.parameters),
            "raw_result": dict(result.raw_result),
        })
        entries.append(build_atomic_experience_entry(
            scenario_id=args.scenario_id,
            condition_id=args.condition_id,
            robot_type="mobile_dual_arm",
            action=action,
            result=result,
        ))

    report = {
        "schema_version": "field_atomic_skill_smoke_report_v1",
        "model_path": args.model_path,
        "scenario_id": args.scenario_id,
        "condition_id": args.condition_id,
        "action_count": len(action_reports),
        "success_count": sum(1 for item in action_reports if item["success"]),
        "failure_count": sum(1 for item in action_reports if not item["success"]),
        "actions": action_reports,
    }
    if args.save_experience_library is not None:
        library = {
            "schema_version": "universal_experience_library_v1",
            "entry_count": len(entries),
            "entries": [entry.to_dict() for entry in entries],
        }
        _write_json(args.save_experience_library, library)
        report["experience_library_output"] = str(args.save_experience_library)
    if args.save_report is not None:
        _write_json(args.save_report, report)
    print(json.dumps({
        "action_count": report["action_count"],
        "success_count": report["success_count"],
        "failure_count": report["failure_count"],
        "save_report": str(args.save_report) if args.save_report else "",
        "save_experience_library": str(args.save_experience_library) if args.save_experience_library else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
