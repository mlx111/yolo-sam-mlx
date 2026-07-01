from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[2]
EXPERIENCE_ROOT = ROOT.parent / "experience_system"
for path in (ROOT, EXPERIENCE_ROOT):
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)

from experience_core import (  # noqa: E402
    ExperienceLibrary,
    consolidate_memory_lifecycle,
    is_field_atomic_entry,
    score_field_atomic_recovery_quality,
    verify_field_atomic_anomaly,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-click Galaxea tabletop anomaly experiment.")
    parser.add_argument("--color", default="camera/color_2026-06-25T00_40_21.png")
    parser.add_argument("--depth", default="camera/depth_2026-06-25T00_40_21.png")
    parser.add_argument("--objects", nargs="+", default=["apple", "red box", "green box"])
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--recovery-actions", type=Path, default=None)
    parser.add_argument("--generate-recovery-plan", action="store_true", help="Generate recovery actions with multimodal LLM after sandbox replay.")
    parser.add_argument("--dry-run-recovery-llm", action="store_true")
    parser.add_argument("--goal", default="恢复异常并完成目标物体抓取、搬运或放置任务")
    parser.add_argument("--target-class", default="apple")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--condition-id", default="", help="Deprecated; scenario_id is the primary experiment grouping key.")
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-recovery-steps", type=int, default=8)
    parser.add_argument("--recovery-candidate-count", type=int, default=1)
    parser.add_argument("--recovery-rewrite-attempts", type=int, default=1)
    parser.add_argument("--allow-invalid-recovery-plan", action="store_true")
    parser.add_argument("--execute-recovery-candidate-validation", action="store_true")
    parser.add_argument("--recovery-validation-limit", type=int, default=3)
    parser.add_argument("--universal-experience-lib", type=Path, default=None, help="Universal experience library path. Defaults to <save-dir>/memory/universal_experience_library.json.")
    parser.add_argument("--visual-index-dir", type=Path, default=None)
    parser.add_argument("--use-visual-retrieval", action="store_true")
    parser.add_argument("--anomaly-stop-index", type=int, default=None)
    parser.add_argument("--settle-before-steps", type=int, default=1500)
    parser.add_argument("--coordinate-system", choices=["camera", "base", "world"], default="world")
    parser.add_argument("--grounded-sam2-root", default="../Grounded-SAM-2")
    parser.add_argument("--model-path", type=Path, default=None, help="Use an existing MuJoCo scene XML instead of building from camera images.")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-hold-seconds", type=float, default=0.0)
    parser.add_argument("--keep-runtime-tmp", action="store_true", help="Deprecated no-op; runtime tmp is kept by default.")
    parser.add_argument("--skip-scene-build", action="store_true")
    parser.add_argument("--skip-nominal-execution", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--recovery-only", action="store_true", help="Skip nominal execution and replay; only run recovery actions.")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _resolve_workspace_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    repo_root = ROOT.parent
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate.resolve()
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate.resolve()
    return repo_candidate.resolve()


def _default_save_dir(args: argparse.Namespace) -> Path:
    return ROOT / "output" / "tabletop_anomaly" / str(args.scenario_id)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _rewrite_archived_scene_xml(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    model_mesh_dir = (ROOT / "model" / "meshes").resolve()
    text = text.replace('meshdir="../model/meshes/"', f'meshdir="{model_mesh_dir}/"')
    text = text.replace('meshdir="../model/meshes"', f'meshdir="{model_mesh_dir}"')
    text = text.replace('meshdir="../meshes/"', f'meshdir="{model_mesh_dir}/"')
    text = text.replace('meshdir="../meshes"', f'meshdir="{model_mesh_dir}"')
    text = text.replace('include file="../model/model.xml"', f'include file="{(ROOT / "model" / "model.xml").resolve()}"')
    text = text.replace('include file="model/model.xml"', f'include file="{(ROOT / "model" / "model.xml").resolve()}"')
    old_runtime_mesh_root = str((ROOT / "meshes").resolve()) + "/"
    new_runtime_mesh_root = str(model_mesh_dir) + "/"
    text = text.replace(old_runtime_mesh_root, new_runtime_mesh_root)
    for resource_dir in ("fruit", "stl"):
        text = text.replace(f'file="../{resource_dir}/', f'file="{(ROOT / resource_dir).resolve()}/')
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _log_stage(message: str) -> None:
    print(f"[tabletop_anomaly] {message}", flush=True)


def _run_step(name: str, cmd: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    _log_stage(f"start {name}")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    _log_stage(f"finish {name} returncode={int(proc.returncode)}")
    return {
        "name": name,
        "command": cmd,
        "cwd": str(cwd),
        "returncode": int(proc.returncode),
        "success": int(proc.returncode) == 0,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def _plan_actions(plan_path: Path) -> list[str]:
    payload = _read_json(plan_path)
    if not isinstance(payload, dict):
        return []
    return [
        str(item.get("action") or "")
        for item in payload.get("steps") or []
        if isinstance(item, dict) and str(item.get("action") or "")
    ]


def _print_plan_summary(plan_path: Path, report_path: Path, *, prefix: str) -> None:
    actions = _plan_actions(plan_path)
    report = _read_json(report_path) or {}
    if actions:
        _log_stage(f"{prefix} plan_actions={actions}")
    if isinstance(report, dict):
        _log_stage(
            f"{prefix} result success={report.get('task_success', report.get('success', False))} "
            f"action_count={report.get('action_count')} "
            f"success_count={report.get('success_count')} "
            f"failure_count={report.get('failure_count')} "
            f"failed_action={report.get('failed_action', {}).get('action', '') if isinstance(report.get('failed_action'), dict) else ''}"
        )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _merge_experience_artifacts(
    *,
    universal_library_path: Path | None,
    artifact_paths: list[Path],
) -> dict[str, Any]:
    if universal_library_path is None:
        return {
            "enabled": False,
            "library_path": "",
            "before_count": 0,
            "candidate_count": 0,
            "written_count": 0,
            "skipped_count": 0,
            "decisions": [],
        }

    library = ExperienceLibrary.load(universal_library_path)
    before_count = len(library)
    decisions: list[dict[str, Any]] = []
    candidate_count = 0
    written_count = 0
    skipped_count = 0

    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            continue
        artifact = ExperienceLibrary.load(artifact_path)
        for entry in artifact.entries:
            if not is_field_atomic_entry(entry):
                continue
            candidate_count += 1
            decision = library.add_with_policy(entry, strict_quality=False)
            action = ""
            feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
            action = str(feedback.get("field_atomic_action") or entry.metadata.get("field_atomic_action") or "")
            written = str(decision.get("decision") or "") in {"write", "merge"}
            if written:
                written_count += 1
            else:
                skipped_count += 1
            decisions.append({
                "artifact": str(artifact_path),
                "experience_id": entry.experience_id,
                "stored_experience_id": decision.get("stored_experience_id", ""),
                "action": action,
                "memory_role": entry.memory_tags.get("memory_role", ""),
                "decision": decision.get("decision", ""),
                "reason": decision.get("reason", ""),
            })

    if candidate_count:
        library.entries, lifecycle_report = consolidate_memory_lifecycle(library.entries)
        library.save(universal_library_path)
    else:
        lifecycle_report = {"skipped": True, "reason": "no_candidate_experiences"}

    return {
        "enabled": True,
        "library_path": str(universal_library_path),
        "before_count": before_count,
        "after_count": len(library),
        "candidate_count": candidate_count,
        "written_count": written_count,
        "skipped_count": skipped_count,
        "memory_lifecycle_report": lifecycle_report,
        "decisions": decisions,
    }


def _summary_from_report(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"report": str(path), "available": False}
    actions = payload.get("actions")
    executed_actions = [
        str(item.get("action") or "")
        for item in actions or []
        if isinstance(item, dict) and str(item.get("action") or "")
    ]
    failed_action = next(
        (
            {
                "index": item.get("index"),
                "action": item.get("action"),
                "status": item.get("status"),
                "message": item.get("message"),
            }
            for item in actions or []
            if isinstance(item, dict) and not bool(item.get("success", False))
        ),
        {},
    )
    return {
        "report": str(path),
        "available": True,
        "action_count": payload.get("action_count"),
        "success_count": payload.get("success_count"),
        "failure_count": payload.get("failure_count"),
        "task_success": payload.get("task_success"),
        "object_lift_success": payload.get("object_lift_success"),
        "object_lift_world": payload.get("object_lift_world"),
        "object_body": payload.get("object_body"),
        "executed_actions": executed_actions,
        "failed_action": failed_action,
    }


def _plan_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"report": str(path), "available": False}
    steps = [
        {
            "action": str(item.get("action") or ""),
            "parameters": item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
        }
        for item in payload.get("steps") or []
        if isinstance(item, dict) and str(item.get("action") or "")
    ]
    return {
        "report": str(path),
        "available": True,
        "step_count": len(steps),
        "steps": steps,
        "action_sequence": [str(item.get("action") or "") for item in steps],
    }


def _keyframe_images(path: Path, *, limit: int = 6) -> list[str]:
    if not path.exists():
        return []
    images = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    return [str(item) for item in images[: max(0, int(limit))]]


def _first_last_images(path: Path) -> tuple[str, str] | tuple[None, None]:
    images = _keyframe_images(path, limit=100)
    if not images:
        return None, None
    return images[0], images[-1]


def _action_history_from_report(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return []
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        rows.append({
            "index": item.get("index"),
            "action": item.get("action"),
            "status": item.get("status"),
            "success": bool(item.get("success", False)),
            "message": item.get("message", ""),
        })
    return rows


def _rule_summary_from_report(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"available": False, "report": str(path)}
    return {
        "available": True,
        "report": str(path),
        "action_count": payload.get("action_count"),
        "success_count": payload.get("success_count"),
        "failure_count": payload.get("failure_count"),
        "task_success": payload.get("task_success"),
        "object_lift_success": payload.get("object_lift_success"),
        "object_lift_world": payload.get("object_lift_world"),
        "failed_action": next(
            (
                {
                    "index": item.get("index"),
                    "action": item.get("action"),
                    "message": item.get("message"),
                }
                for item in payload.get("actions") or []
                if isinstance(item, dict) and not bool(item.get("success", False))
            ),
            {},
        ),
    }


def _select_anomaly_stop_index(nominal_report_path: Path, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    payload = _read_json(nominal_report_path)
    if not isinstance(payload, dict):
        return -1
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        return -1
    for item in actions:
        if isinstance(item, dict) and not bool(item.get("success", False)):
            return int(item.get("index", 0))
    last = actions[-1]
    return int(last.get("index", len(actions) - 1)) if isinstance(last, dict) else len(actions) - 1


def _scene_paths(save_dir: Path) -> dict[str, Path]:
    return {
        "scene_xml": ROOT / "scence" / "initial_runtime_scene.xml",
        "scene_report": ROOT / "scence" / "initial_runtime_scene_report.json",
        "observation": ROOT / "scence" / "initial_runtime_observation.json",
        "archive_scene_xml": save_dir / "scene" / "initial_runtime_scene.xml",
        "archive_scene_report": save_dir / "scene" / "initial_runtime_scene_report.json",
        "archive_observation": save_dir / "scene" / "initial_runtime_observation.json",
    }


def _experiment_paths(save_dir: Path) -> dict[str, Path]:
    return {
        "nominal_report": save_dir / "nominal" / "execution_report.json",
        "nominal_summary": save_dir / "nominal" / "execution_report_execution_summary.json",
        "nominal_library": save_dir / "memory" / "nominal_experience_library.json",
        "replay_report": save_dir / "replay" / "replay_report.json",
        "replay_state": save_dir / "replay" / "state.json",
        "recovery_context": save_dir / "plans" / "recovery_context.json",
        "recovery_plan": save_dir / "plans" / "recovery_plan.json",
        "recovery_plan_report": save_dir / "plans" / "recovery_plan_report.json",
        "recovery_report": save_dir / "recovery" / "execution_report.json",
        "recovery_summary": save_dir / "recovery" / "execution_report_execution_summary.json",
        "universal_library": save_dir / "memory" / "universal_experience_library.json",
        "recovery_library": save_dir / "memory" / "recovery_experience_library.json",
        "nominal_keyframes": save_dir / "keyframes" / "nominal",
        "replay_keyframes": save_dir / "keyframes" / "replay",
        "recovery_keyframes": save_dir / "keyframes" / "recovery",
        "candidate_validation": save_dir / "plans" / "candidate_execution_validation",
    }


def _resolve_recovery_actions_arg(path: Path | None, *, save_dir: Path) -> Path:
    exp_paths = _experiment_paths(save_dir)
    if path is None:
        return exp_paths["recovery_plan"]
    resolved = _resolve_workspace_path(path)
    if resolved.exists():
        return resolved
    legacy_name = Path(path).name
    if legacy_name == "recovery_plan.json" and exp_paths["recovery_plan"].exists():
        return exp_paths["recovery_plan"]
    return resolved


def _build_report(
    *,
    args: argparse.Namespace,
    save_dir: Path,
    paths: dict[str, Path],
    anomaly_stop_index: int | None,
    final_status: str,
    experience_writeback: dict[str, Any],
    anomaly_verification: dict[str, Any],
    recovery_quality_score: dict[str, Any],
    command_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    exp_paths = _experiment_paths(save_dir)
    nominal_report = exp_paths["nominal_report"]
    nominal_summary = exp_paths["nominal_summary"]
    recovery_report = exp_paths["recovery_report"]
    recovery_summary = exp_paths["recovery_summary"]
    replay_report = exp_paths["replay_report"]
    replay_payload = _read_json(replay_report)
    nominal_payload = _summary_from_report(nominal_report)
    if nominal_summary.exists():
        nominal_payload["execution_summary_report"] = str(nominal_summary)
    recovery_payload = _summary_from_report(recovery_report) if (args.recovery_actions is not None or bool(args.generate_recovery_plan)) else {}
    if recovery_summary.exists():
        recovery_payload["execution_summary_report"] = str(recovery_summary)
    recovery_plan_path = exp_paths["recovery_plan"] if bool(args.generate_recovery_plan) else (
        _resolve_workspace_path(args.recovery_actions) if args.recovery_actions is not None else exp_paths["recovery_plan"]
    )
    return {
        "schema_version": "tabletop_anomaly_experiment_report_v1",
        "scenario_id": args.scenario_id,
        "objects": list(args.objects),
        "final_status": final_status,
        "save_dir": str(save_dir),
        "scene": {
            "xml": str(paths["archive_scene_xml"]),
            "report": str(paths["archive_scene_report"]),
            "observation": str(paths["archive_observation"]),
            "current_scene_xml": str(paths["scene_xml"]),
        },
        "command_trace": command_trace,
        "nominal_execution": nominal_payload,
        "anomaly_replay": {
            "enabled": not bool(args.skip_replay),
            "stop_index": anomaly_stop_index,
            "summary": {
                "report": str(replay_report),
                "available": isinstance(replay_payload, dict),
                "replayed_action_count": (replay_payload or {}).get("replayed_action_count") if isinstance(replay_payload, dict) else None,
                "success_count": (replay_payload or {}).get("success_count") if isinstance(replay_payload, dict) else None,
                "failure_count": (replay_payload or {}).get("failure_count") if isinstance(replay_payload, dict) else None,
            },
            "vlm_verification": anomaly_verification,
        },
        "recovery_execution": {
            "enabled": args.recovery_actions is not None or bool(args.generate_recovery_plan),
            **recovery_payload,
        },
        "recovery_plan": {
            "generated": bool(args.generate_recovery_plan),
            "context": str(exp_paths["recovery_context"]),
            "plan": str(recovery_plan_path),
            "report": str(exp_paths["recovery_plan_report"]),
            **_plan_summary(recovery_plan_path),
        },
        "recovery_quality_score": recovery_quality_score,
        "experience_writeback": experience_writeback,
    }


def main() -> None:
    args = parse_args()
    save_dir = (args.save_dir.resolve() if args.save_dir is not None else _default_save_dir(args).resolve())
    if bool(args.recovery_only):
        args.skip_nominal_execution = True
        args.skip_replay = True
        if not bool(args.generate_recovery_plan):
            args.recovery_actions = _resolve_recovery_actions_arg(args.recovery_actions, save_dir=save_dir)
            if not args.recovery_actions.exists():
                raise FileNotFoundError(f"--recovery-actions not found: {args.recovery_actions}")
    save_dir.mkdir(parents=True, exist_ok=True)
    _log_stage(f"save_dir={save_dir}")
    exp_paths = _experiment_paths(save_dir)
    universal_experience_lib = (
        _resolve_workspace_path(args.universal_experience_lib)
        if args.universal_experience_lib is not None
        else exp_paths["universal_library"]
    )
    stale_outputs = [
        exp_paths["nominal_report"],
        exp_paths["nominal_summary"],
        exp_paths["nominal_library"],
        exp_paths["replay_report"],
        exp_paths["replay_state"],
        exp_paths["recovery_library"],
        exp_paths["recovery_report"],
        exp_paths["recovery_summary"],
        exp_paths["recovery_plan"],
        exp_paths["recovery_plan_report"],
        exp_paths["recovery_context"],
    ]
    if bool(args.recovery_only):
        stale_outputs = [
            item for item in stale_outputs
            if item not in {
                exp_paths["nominal_report"],
                exp_paths["nominal_summary"],
                exp_paths["nominal_library"],
                exp_paths["replay_report"],
                exp_paths["replay_state"],
            }
        ]
        recovery_actions_path = _resolve_workspace_path(args.recovery_actions) if args.recovery_actions is not None else None
        if recovery_actions_path is not None:
            stale_outputs = [
                item for item in stale_outputs
                if item.resolve() != recovery_actions_path
            ]
    for stale_path in stale_outputs:
        if stale_path.exists():
            stale_path.unlink()
    stale_validation = exp_paths["candidate_validation"]
    if stale_validation.exists():
        shutil.rmtree(stale_validation)
    paths = _scene_paths(save_dir)
    command_trace: list[dict[str, Any]] = []
    anomaly_stop_index: int | None = None
    final_status = "not_started"
    execution_model_path = paths["scene_xml"]
    experience_writeback: dict[str, Any] = {"enabled": True, "library_path": str(universal_experience_lib)}
    experience_artifacts: list[Path] = []
    anomaly_verification: dict[str, Any] = {"enabled": False}
    recovery_quality_score: dict[str, Any] = {"enabled": False}

    try:
        if not args.skip_scene_build:
            _log_stage("stage scene_build")
            if args.model_path is not None:
                model_path = _resolve_workspace_path(args.model_path)
                execution_model_path = model_path
                _copy_if_exists(model_path, paths["scene_xml"])
                _write_json(paths["scene_report"], {
                    "schema_version": "existing_mujoco_scene_reference_v1",
                    "source_model_path": str(model_path),
                    "scene_out": str(paths["scene_xml"]),
                })
                _write_json(paths["observation"], {
                    "schema_version": "existing_mujoco_scene_observation_v1",
                    "source_model_path": str(model_path),
                    "objects": list(args.objects),
                })
            else:
                scene_cmd = [
                    sys.executable,
                    "-B",
                    "build_runtime_scene_from_real_camera.py",
                    "--color",
                    str(_resolve_workspace_path(args.color)),
                    "--depth",
                    str(_resolve_workspace_path(args.depth)),
                    "--objects",
                    *[str(item) for item in args.objects],
                    "--scene-out",
                    str(paths["scene_xml"]),
                    "--report-out",
                    str(paths["scene_report"]),
                    "--observation-out",
                    str(paths["observation"]),
                    "--grounded-sam2-root",
                    str(_resolve_workspace_path(args.grounded_sam2_root)),
                    "--coordinate-system",
                    str(args.coordinate_system),
                    "--settle-steps",
                    str(int(args.settle_before_steps)),
                ]
                result = _run_step("build_initial_runtime_scene", scene_cmd)
                command_trace.append(result)
                if not result["success"]:
                    final_status = "scene_build_failed"
                    _log_stage("stage scene_build failed")
                    return
        elif not paths["scene_xml"].exists():
            final_status = "scene_missing"
            _log_stage("stage scene missing")
            return

        _log_stage("stage archive_scene")
        _copy_if_exists(paths["scene_xml"], paths["archive_scene_xml"])
        _rewrite_archived_scene_xml(paths["archive_scene_xml"])
        _copy_if_exists(paths["scene_report"], paths["archive_scene_report"])
        _copy_if_exists(paths["observation"], paths["archive_observation"])

        nominal_report = exp_paths["nominal_report"]
        nominal_library = exp_paths["nominal_library"]
        nominal_keyframes = exp_paths["nominal_keyframes"]
        if not args.skip_nominal_execution:
            _log_stage("stage nominal_execution")
            nominal_cmd = [
                sys.executable,
                "-B",
                "source/active/run_field_atomic_skill_smoke.py",
                "--model-path",
                str(execution_model_path),
                "--actions",
                str(_resolve_workspace_path(args.actions)),
                "--scenario-id",
                str(args.scenario_id),
                "--condition-id",
                "",
                "--goal",
                str(args.goal),
                "--settle-before-steps",
                str(int(args.settle_before_steps)),
                "--save-report",
                str(nominal_report),
                "--save-experience-library",
                str(nominal_library),
                "--experience-save-mode",
                "failure_only",
                "--llm-critic",
                "--llm-critic-provider",
                str(args.provider),
                "--llm-critic-model",
                str(args.model),
                "--apply-memory-lifecycle",
                "--skip-visual-index",
                "--keyframe-dir",
                str(nominal_keyframes),
                "--keyframe-camera",
                "workspace_overview_camera",
            ]
            if args.viewer:
                nominal_cmd.extend(["--viewer", "--viewer-hold-seconds", str(float(args.viewer_hold_seconds))])
            if args.stop_on_failure:
                nominal_cmd.append("--stop-on-failure")
            result = _run_step("nominal_execution", nominal_cmd)
            command_trace.append(result)
            if not result["success"] and not nominal_report.exists():
                final_status = "nominal_execution_command_failed"
                _log_stage("stage nominal_execution failed")
                return
            if not result["success"]:
                _log_stage("stage nominal_execution returned nonzero but report exists; continuing")
        elif not nominal_report.exists():
            final_status = "nominal_report_missing"
            _log_stage("stage nominal report missing")
            return

        anomaly_stop_index = _select_anomaly_stop_index(nominal_report, args.anomaly_stop_index)
        if not args.skip_replay:
            _log_stage(f"stage anomaly_replay stop_index={anomaly_stop_index}")
            replay_keyframes = exp_paths["replay_keyframes"]
            replay_cmd = [
                sys.executable,
                "-B",
                "source/active/replay_field_atomic_prefix_in_sandbox.py",
                "--model-path",
                str(execution_model_path),
                "--trace",
                str(nominal_report),
                "--stop-index",
                str(anomaly_stop_index),
                "--settle-before-steps",
                str(int(args.settle_before_steps)),
                "--save-state",
                str(exp_paths["replay_state"]),
                "--save-report",
                str(exp_paths["replay_report"]),
                "--keyframe-dir",
                str(replay_keyframes),
                "--keyframe-camera",
                "workspace_overview_camera",
                "--continue-on-failure",
            ]
            result = _run_step("anomaly_replay", replay_cmd)
            command_trace.append(result)
            if not result["success"]:
                final_status = "anomaly_replay_failed"
                _log_stage("stage anomaly_replay failed")
                return
            before_image, after_image = _first_last_images(replay_keyframes)
            if before_image and after_image:
                anomaly_verification = verify_field_atomic_anomaly(
                    goal=str(args.goal),
                    image_before=before_image,
                    image_after=after_image,
                    rule_summary=_rule_summary_from_report(exp_paths["replay_report"]),
                    provider=str(args.provider),
                    model=str(args.model),
                )
            else:
                anomaly_verification = {
                    "enabled": True,
                    "status": "FAILURE",
                    "reason": "replay keyframes missing",
                    "consider": "missing_visual_evidence",
                }

        if nominal_library.exists() and not bool(args.recovery_only):
            experience_artifacts.append(nominal_library)

        recovery_actions = args.recovery_actions
        if bool(args.generate_recovery_plan):
            _log_stage("stage generate_recovery_plan")
            recovery_actions = exp_paths["recovery_plan"]
            recovery_plan_cmd = [
                sys.executable,
                "-B",
                "source/active/run_field_atomic_recovery_vlm_plan.py",
                "--goal",
                str(args.goal),
                "--scenario-id",
                str(args.scenario_id),
                "--target-class",
                str(args.target_class),
                "--nominal-report",
                str(nominal_report),
                "--replay-state",
                str(exp_paths["replay_state"]),
                "--replay-report",
                str(exp_paths["replay_report"]),
                "--scene-observation",
                str(paths["archive_observation"]),
                "--provider",
                str(args.provider),
                "--max-steps",
                str(int(args.max_recovery_steps)),
                "--candidate-count",
                str(max(1, int(args.recovery_candidate_count))),
                "--rewrite-attempts",
                str(max(0, int(args.recovery_rewrite_attempts))),
                "--save-context",
                str(exp_paths["recovery_context"]),
                "--save-plan",
                str(recovery_actions),
                "--save-report",
                str(exp_paths["recovery_plan_report"]),
            ]
            if bool(args.execute_recovery_candidate_validation):
                recovery_plan_cmd.extend([
                    "--execute-candidate-validation",
                    "--validation-model-path",
                    str(execution_model_path),
                    "--validation-output-dir",
                    str(exp_paths["candidate_validation"]),
                    "--validation-settle-before-steps",
                    str(int(args.settle_before_steps)),
                    "--validation-limit",
                    str(max(1, int(args.recovery_validation_limit))),
                ])
            if args.model:
                recovery_plan_cmd.extend(["--model", str(args.model)])
            recovery_plan_cmd.extend(["--universal-experience-lib", str(universal_experience_lib)])
            if args.visual_index_dir is not None:
                recovery_plan_cmd.extend(["--visual-index-dir", str(_resolve_workspace_path(args.visual_index_dir))])
            if bool(args.use_visual_retrieval):
                recovery_plan_cmd.append("--use-visual-retrieval")
            if args.dry_run_recovery_llm:
                recovery_plan_cmd.append("--dry-run-llm")
            result = _run_step("generate_recovery_plan", recovery_plan_cmd)
            command_trace.append(result)
            if not result["success"]:
                final_status = "recovery_plan_generation_failed"
                _log_stage("stage generate_recovery_plan failed")
                return
            recovery_plan_report = _read_json(exp_paths["recovery_plan_report"])
            recovery_plan_payload = _read_json(recovery_actions)
            _print_plan_summary(recovery_actions, exp_paths["recovery_plan_report"], prefix="recovery_plan")
        if recovery_actions is not None:
            _log_stage("stage recovery_execution")
            nominal_payload = _read_json(nominal_report)
            recovery_keyframes = exp_paths["recovery_keyframes"]
            recovery_cmd = [
                sys.executable,
                "-B",
                "source/active/run_field_atomic_skill_smoke.py",
                "--model-path",
                str(execution_model_path),
                "--actions",
                str(_resolve_workspace_path(recovery_actions)),
                "--scenario-id",
                str(args.scenario_id),
                "--condition-id",
                "recovery",
                "--goal",
                str(args.goal),
                "--settle-before-steps",
                str(int(args.settle_before_steps)),
                "--save-report",
                str(exp_paths["recovery_report"]),
                "--save-experience-library",
                str(exp_paths["recovery_library"]),
                "--experience-save-mode",
                "all",
                "--llm-critic",
                "--llm-critic-provider",
                str(args.provider),
                "--llm-critic-model",
                str(args.model),
                "--episode-role",
                "recovery",
                "--source-recovery-plan",
                str(_resolve_workspace_path(recovery_actions)),
                "--apply-memory-lifecycle",
                "--keyframe-dir",
                str(recovery_keyframes),
                "--keyframe-camera",
                "workspace_overview_camera",
            ]
            if nominal_report.exists() and not bool(args.recovery_only):
                source_failure_experience_id = ""
                if isinstance(nominal_payload, dict):
                    source_failure_experience_id = str(nominal_payload.get("experience_failure_episode_id") or "")
                recovery_cmd.extend([
                    "--source-failure-report",
                    str(nominal_report),
                    "--source-failure-experience-id",
                    source_failure_experience_id,
                ])
            if args.visual_index_dir is not None:
                recovery_cmd.extend(["--visual-index-dir", str(_resolve_workspace_path(args.visual_index_dir))])
            result = _run_step("recovery_execution", recovery_cmd)
            command_trace.append(result)
            if exp_paths["recovery_library"].exists():
                experience_artifacts.append(exp_paths["recovery_library"])
            recovery_quality_score = score_field_atomic_recovery_quality(
                goal=str(args.goal),
                task_history=_action_history_from_report(exp_paths["recovery_report"]),
                image_paths=_keyframe_images(recovery_keyframes, limit=3),
                provider=str(args.provider),
                model=str(args.model),
            )
            _print_plan_summary(_resolve_workspace_path(recovery_actions), exp_paths["recovery_report"], prefix="recovery_execution")
            final_status = "recovery_executed" if result["success"] else "recovery_execution_failed"
            _log_stage(f"stage recovery_execution finished status={final_status}")
            return

        nominal = _read_json(nominal_report)
        if isinstance(nominal, dict) and int(nominal.get("failure_count") or 0) > 0:
            final_status = "anomaly_replayed" if not args.skip_replay else "nominal_failed"
        else:
            final_status = "nominal_completed"
        _log_stage(f"stage finished status={final_status}")
    finally:
        try:
            experience_writeback = _merge_experience_artifacts(
                universal_library_path=(
                    universal_experience_lib
                ),
                artifact_paths=[
                    *experience_artifacts,
                ],
            )
        except Exception as exc:
            experience_writeback = {
                "enabled": True,
                "library_path": str(universal_experience_lib),
                "error": str(exc),
            }
        report = _build_report(
            args=args,
            save_dir=save_dir,
            paths=paths,
            anomaly_stop_index=anomaly_stop_index,
            final_status=final_status,
            experience_writeback=experience_writeback,
            anomaly_verification=anomaly_verification,
            recovery_quality_score=recovery_quality_score,
            command_trace=command_trace,
        )
        _write_json(save_dir / "experiment_report.json", report)
        _log_stage(f"wrote experiment_report={save_dir / 'experiment_report.json'}")
        print(json.dumps({
            "final_status": final_status,
            "save_dir": str(save_dir),
            "experiment_report": str(save_dir / "experiment_report.json"),
            "command_count": len(command_trace),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
