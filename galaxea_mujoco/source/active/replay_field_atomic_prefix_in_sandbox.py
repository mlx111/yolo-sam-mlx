from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIENCE_ROOT = ROOT.parent / "experience_system"
if str(EXPERIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIENCE_ROOT))

from runtime_perception.warnings_filter import suppress_grounded_sam2_warnings
from source.active.keyframe_recorder import KeyframeRecorder
from skills.field_atomic import FieldAtomicSkillExecutor
from skills.field_atomic.atomic_executor import _canonical_action
from skills.field_atomic.atomic_schema import FieldAtomicResult


suppress_grounded_sam2_warnings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a field atomic action prefix in a runtime sandbox scene.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace", type=Path, required=True, help="field_atomic report JSON or JSON action list")
    parser.add_argument("--stop-index", type=int, default=-1, help="last action index to replay; negative replays all")
    parser.add_argument("--settle-before-steps", type=int, default=0)
    parser.add_argument("--save-state", type=Path, required=True)
    parser.add_argument("--save-report", type=Path, default=None)
    parser.add_argument("--object-bodies", nargs="*", default=[], help="specific object body names to record")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--keyframe-dir", type=Path, default=None, help="Directory for rendered replay keyframe png files.")
    parser.add_argument("--keyframe-camera", default="workspace_overview_camera", help="MuJoCo camera name for keyframe rendering.")
    return parser.parse_args()


def _load_trace(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("actions"), list):
        return [item for item in payload["actions"] if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        return [item for item in payload["steps"] if isinstance(item, dict)]
    raise ValueError("--trace must be a JSON action list or a report with actions/steps")


def _trace_action(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    action = str(item.get("action") or "")
    parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
    return action, dict(parameters)


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> dict[str, Any]:
    return {
        "position": np.round(data.xpos[body_id], 9).tolist(),
        "xmat": np.round(data.xmat[body_id].reshape(3, 3), 9).tolist(),
    }


def _freejoint_qpos(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> dict[str, Any]:
    body_jntadr = int(model.body_jntadr[body_id])
    body_jntnum = int(model.body_jntnum[body_id])
    for offset in range(body_jntnum):
        joint_id = body_jntadr + offset
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = int(model.jnt_qposadr[joint_id])
            qvel_adr = int(model.jnt_dofadr[joint_id])
            return {
                "joint_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "qpos": np.round(data.qpos[qpos_adr : qpos_adr + 7], 9).tolist(),
                "qvel": np.round(data.qvel[qvel_adr : qvel_adr + 6], 9).tolist(),
            }
    return {}


def _object_state(model: mujoco.MjModel, data: mujoco.MjData, requested: list[str]) -> dict[str, Any]:
    names = list(requested)
    if not names:
        for body_id in range(model.nbody):
            if int(model.body_jntnum[body_id]) <= 0:
                continue
            if any(model.jnt_type[int(model.body_jntadr[body_id]) + offset] == mujoco.mjtJoint.mjJNT_FREE for offset in range(int(model.body_jntnum[body_id]))):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name and name not in {"world"}:
                    names.append(name)
    state: dict[str, Any] = {}
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        if body_id < 0:
            state[str(name)] = {"found": False}
            continue
        state[str(name)] = {
            "found": True,
            **_body_pose(model, data, body_id),
            "freejoint": _freejoint_qpos(model, data, body_id),
        }
    return state


def _robot_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    joint_values: dict[str, float] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_values[name] = float(data.qpos[qpos_adr])
    return {
        "qpos": np.round(data.qpos, 9).tolist(),
        "qvel": np.round(data.qvel, 9).tolist(),
        "joint_qpos": joint_values,
    }


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    recorder = KeyframeRecorder(args.keyframe_dir, camera=args.keyframe_camera)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, int(args.settle_before_steps))):
        mujoco.mj_step(model, data)
    recorder.capture(model, data, "replay_initial", description="Initial state before sandbox replay")

    actions = _load_trace(args.trace)
    if args.stop_index >= 0:
        actions = [item for item in actions if int(item.get("index", actions.index(item))) <= args.stop_index]

    executor = FieldAtomicSkillExecutor()
    step_reports: list[dict[str, Any]] = []
    for replay_index, item in enumerate(actions):
        action, parameters = _trace_action(item)
        action_name = _canonical_action(action) or action or "unknown"
        recorder.capture(
            model,
            data,
            f"replay_before_step_{replay_index:03d}_{action_name}",
            description=f"Before replay step {replay_index}: {action_name}",
            action=action_name,
            index=replay_index,
        )
        try:
            result = executor.execute(model, data, action, parameters)
        except Exception as exc:
            result = FieldAtomicResult(action=action, success=False, status="exception", message=str(exc), parameters=dict(parameters))
        recorder.capture(
            model,
            data,
            f"replay_after_step_{replay_index:03d}_{action_name}",
            description=f"After replay step {replay_index}: {action_name}",
            action=action_name,
            index=replay_index,
        )
        step_reports.append({
            "replay_index": replay_index,
            "source_index": item.get("index", replay_index),
            "action": action,
            "canonical_action": _canonical_action(action),
            "parameters": dict(parameters),
            "success": bool(result.success),
            "status": result.status,
            "message": result.message,
            "raw_result": dict(result.raw_result),
        })
        if not bool(result.success) and not bool(args.continue_on_failure):
            break

    mujoco.mj_forward(model, data)
    recorder.capture(model, data, "replay_state", description="Sandbox state after replayed anomaly prefix")
    state = {
        "schema_version": "field_atomic_sandbox_replay_state_v1",
        "model_path": args.model_path,
        "trace": str(args.trace),
        "stop_index": args.stop_index,
        "settle_before_steps": max(0, int(args.settle_before_steps)),
        "replayed_action_count": len(step_reports),
        "success_count": sum(1 for item in step_reports if item["success"]),
        "failure_count": sum(1 for item in step_reports if not item["success"]),
        "robot_state": _robot_state(model, data),
        "object_state": _object_state(model, data, args.object_bodies),
        "keyframes": list(recorder.keyframes),
        "keyframe_errors": list(recorder.errors),
        "step_reports": step_reports,
    }
    _write_json(args.save_state, state)
    if args.save_report is not None:
        _write_json(args.save_report, {
            "schema_version": "field_atomic_sandbox_replay_report_v1",
            "save_state": str(args.save_state),
            "replayed_action_count": state["replayed_action_count"],
            "success_count": state["success_count"],
            "failure_count": state["failure_count"],
            "object_names": sorted(state["object_state"]),
            "keyframes": list(recorder.keyframes),
            "keyframe_errors": list(recorder.errors),
        })
    recorder.close()
    print(json.dumps({
        "replayed_action_count": state["replayed_action_count"],
        "success_count": state["success_count"],
        "failure_count": state["failure_count"],
        "save_state": str(args.save_state),
        "save_report": str(args.save_report) if args.save_report else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
