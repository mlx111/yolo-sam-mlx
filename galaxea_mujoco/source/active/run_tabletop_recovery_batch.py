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
    build_galaxea_recovery_rules,
    is_field_atomic_entry,
    update_galaxea_failure_rule_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one nominal tabletop episode, replay the anomaly state once, then run repeated recovery planning/execution rounds."
    )
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--goal", default="恢复异常并完成目标物体抓取、搬运或放置任务")
    parser.add_argument("--target-class", default="apple")
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, help="Number of recovery planning/execution rounds after nominal replay.")
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "output" / "exp_recovery_batch")
    parser.add_argument("--model-path", type=Path, default=None, help="Existing MuJoCo scene XML. Defaults to galaxea_mujoco/scence/initial_runtime_scene.xml.")
    parser.add_argument("--settle-before-steps", type=int, default=1500)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-hold-seconds", type=float, default=0.0)
    parser.add_argument("--max-recovery-steps", type=int, default=8)
    parser.add_argument("--recovery-candidate-count", type=int, default=1)
    parser.add_argument("--recovery-rewrite-attempts", type=int, default=1)
    parser.add_argument("--execute-recovery-candidate-validation", action="store_true")
    parser.add_argument("--recovery-validation-limit", type=int, default=3)
    parser.add_argument("--dry-run-recovery-llm", action="store_true")
    parser.add_argument("--use-visual-retrieval", action="store_true")
    parser.add_argument("--visual-index-dir", type=Path, default=None)
    parser.add_argument("--anomaly-stop-index", type=int, default=None)
    parser.add_argument("--skip-nominal", action="store_true", help="Reuse existing nominal/replay files in save-dir.")
    parser.add_argument("--skip-replay", action="store_true", help="Reuse existing replay files in save-dir.")
    parser.add_argument("--append-rounds", action="store_true", help="Continue from the next available round_NNN directory instead of starting at round_000.")
    parser.add_argument("--continue-on-round-failure", action="store_true", help="Run later recovery rounds even if a round command fails.")
    return parser.parse_args()


def _resolve_workspace_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    repo_candidate = ROOT.parent / path
    if repo_candidate.exists():
        return repo_candidate.resolve()
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate.resolve()
    return repo_candidate.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def _tail(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _log(message: str) -> None:
    print(f"[tabletop_recovery_batch] {message}", flush=True)


def _run_step(name: str, cmd: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    _log(f"start {name}")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    _log(f"finish {name} returncode={int(proc.returncode)}")
    return {
        "name": name,
        "command": cmd,
        "cwd": str(cwd),
        "returncode": int(proc.returncode),
        "success": int(proc.returncode) == 0,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def _paths(save_dir: Path) -> dict[str, Path]:
    return {
        "scene_xml": save_dir / "scene" / "initial_runtime_scene.xml",
        "scene_report": save_dir / "scene" / "initial_runtime_scene_report.json",
        "scene_observation": save_dir / "scene" / "initial_runtime_observation.json",
        "nominal_report": save_dir / "nominal" / "execution_report.json",
        "nominal_library": save_dir / "memory" / "nominal_experience_library.json",
        "nominal_keyframes": save_dir / "keyframes" / "nominal",
        "replay_state": save_dir / "replay" / "state.json",
        "replay_report": save_dir / "replay" / "replay_report.json",
        "replay_keyframes": save_dir / "keyframes" / "replay",
        "universal_library": save_dir / "memory" / "universal_experience_library.json",
        "batch_report": save_dir / "batch_report.json",
        "rounds": save_dir / "rounds",
    }


def _round_paths(save_dir: Path, index: int) -> dict[str, Path]:
    root = save_dir / "rounds" / f"round_{index:03d}"
    return {
        "root": root,
        "context": root / "recovery_context.json",
        "plan": root / "recovery_plan.json",
        "plan_report": root / "recovery_plan_report.json",
        "execution_report": root / "recovery_execution_report.json",
        "execution_summary": root / "recovery_execution_report_execution_summary.json",
        "experience_library": root / "recovery_experience_library.json",
        "memory_rule_report": root / "memory_rule_report.json",
        "keyframes": root / "keyframes",
        "validation": root / "candidate_execution_validation",
        "round_report": root / "round_report.json",
    }


def _next_round_index(save_dir: Path) -> int:
    rounds_dir = save_dir / "rounds"
    if not rounds_dir.exists():
        return 0
    indices: list[int] = []
    for item in rounds_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if not name.startswith("round_"):
            continue
        try:
            indices.append(int(name.removeprefix("round_")))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


def _copy_scene_files(args: argparse.Namespace, paths: dict[str, Path]) -> Path:
    model_path = _resolve_workspace_path(args.model_path) if args.model_path is not None else ROOT / "scence" / "initial_runtime_scene.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"scene model not found: {model_path}")
    paths["scene_xml"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, paths["scene_xml"])
    _rewrite_archived_scene_xml(paths["scene_xml"])
    _write_json(paths["scene_report"], {
        "schema_version": "batch_existing_mujoco_scene_reference_v1",
        "source_model_path": str(model_path),
        "scene_out": str(paths["scene_xml"]),
    })
    source_observation = model_path.with_name("initial_runtime_observation.json")
    if source_observation.exists():
        shutil.copyfile(source_observation, paths["scene_observation"])
    else:
        _write_json(paths["scene_observation"], {
            "schema_version": "batch_existing_mujoco_scene_observation_v1",
            "source_model_path": str(model_path),
        })
    return paths["scene_xml"]


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


def _plan_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"available": False, "path": str(path), "steps": [], "action_sequence": []}
    steps = [
        {
            "action": str(item.get("action") or ""),
            "parameters": item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
        }
        for item in payload.get("steps") or []
        if isinstance(item, dict) and str(item.get("action") or "")
    ]
    return {
        "available": True,
        "path": str(path),
        "step_count": len(steps),
        "steps": steps,
        "action_sequence": [item["action"] for item in steps],
    }


def _execution_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"available": False, "path": str(path)}
    failed_action = next(
        (
            {
                "index": item.get("index"),
                "action": item.get("action"),
                "status": item.get("status"),
                "message": item.get("message"),
                "parameters": item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
                "raw_result": item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {},
            }
            for item in payload.get("actions") or []
            if isinstance(item, dict) and not bool(item.get("success", False))
        ),
        {},
    )
    return {
        "available": True,
        "path": str(path),
        "action_count": payload.get("action_count"),
        "success_count": payload.get("success_count"),
        "failure_count": payload.get("failure_count"),
        "task_success": payload.get("task_success"),
        "object_lift_success": payload.get("object_lift_success"),
        "object_lift_world": payload.get("object_lift_world"),
        "object_body": payload.get("object_body"),
        "stopped_on_failure": payload.get("stopped_on_failure"),
        "experience_episode_entry_ids": payload.get("experience_episode_entry_ids", []),
        "execution_summary_report": payload.get("execution_summary_report", ""),
        "failed_action": failed_action,
    }


def _merge_experience_artifacts(universal_library_path: Path, artifact_paths: list[Path]) -> dict[str, Any]:
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
                "episode_role": entry.memory_tags.get("episode_role", ""),
                "memory_role": entry.memory_tags.get("memory_role", ""),
                "decision": decision.get("decision", ""),
                "reason": decision.get("reason", ""),
            })

    if candidate_count:
        rule_update_report = update_galaxea_failure_rule_entries(library.entries)
        library.entries, lifecycle_report = consolidate_memory_lifecycle(library.entries)
        library.save(universal_library_path)
    else:
        lifecycle_report = {"skipped": True, "reason": "no_candidate_experiences"}
        rule_update_report = {"skipped": True, "reason": "no_candidate_experiences"}

    return {
        "library_path": str(universal_library_path),
        "before_count": before_count,
        "after_count": len(library),
        "candidate_count": candidate_count,
        "written_count": written_count,
        "skipped_count": skipped_count,
        "galaxea_failure_rule_update": rule_update_report,
        "memory_lifecycle_report": lifecycle_report,
        "decisions": decisions,
    }


def _write_memory_rule_report(universal_library_path: Path, report_path: Path, *, scenario_id: str, condition_id: str = "") -> dict[str, Any]:
    library = ExperienceLibrary.load(universal_library_path)
    rules = build_galaxea_recovery_rules(
        library.entries,
        scenario_id=scenario_id,
        condition_id=condition_id,
    )
    report = {
        "schema_version": "galaxea_memory_rule_report_v1",
        "library_path": str(universal_library_path),
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "galaxea_recovery_rules": rules,
    }
    _write_json(report_path, report)
    return report


def _run_nominal(args: argparse.Namespace, paths: dict[str, Path], model_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-B",
        "source/active/run_field_atomic_skill_smoke.py",
        "--model-path",
        str(model_path),
        "--actions",
        str(_resolve_workspace_path(args.actions)),
        "--scenario-id",
        str(args.scenario_id),
        "--condition-id",
        "",
        "--goal",
        str(args.goal),
        "--settle-before-steps",
        str(max(0, int(args.settle_before_steps))),
        "--save-report",
        str(paths["nominal_report"]),
        "--save-experience-library",
        str(paths["nominal_library"]),
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
        str(paths["nominal_keyframes"]),
        "--keyframe-camera",
        "workspace_overview_camera",
        "--stop-on-failure",
    ]
    if args.viewer:
        cmd.extend(["--viewer", "--viewer-hold-seconds", str(float(args.viewer_hold_seconds))])
    return _run_step("nominal_execution", cmd)


def _run_replay(args: argparse.Namespace, paths: dict[str, Path], model_path: Path, stop_index: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-B",
        "source/active/replay_field_atomic_prefix_in_sandbox.py",
        "--model-path",
        str(model_path),
        "--trace",
        str(paths["nominal_report"]),
        "--stop-index",
        str(int(stop_index)),
        "--settle-before-steps",
        str(max(0, int(args.settle_before_steps))),
        "--save-state",
        str(paths["replay_state"]),
        "--save-report",
        str(paths["replay_report"]),
        "--keyframe-dir",
        str(paths["replay_keyframes"]),
        "--keyframe-camera",
        "workspace_overview_camera",
        "--continue-on-failure",
    ]
    return _run_step("anomaly_replay", cmd)


def _run_plan_round(args: argparse.Namespace, paths: dict[str, Path], round_paths: dict[str, Path], model_path: Path) -> dict[str, Any]:
    cmd = [
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
        str(paths["nominal_report"]),
        "--replay-state",
        str(paths["replay_state"]),
        "--replay-report",
        str(paths["replay_report"]),
        "--scene-observation",
        str(paths["scene_observation"]),
        "--provider",
        str(args.provider),
        "--max-steps",
        str(max(1, int(args.max_recovery_steps))),
        "--candidate-count",
        str(max(1, int(args.recovery_candidate_count))),
        "--rewrite-attempts",
        str(max(0, int(args.recovery_rewrite_attempts))),
        "--save-context",
        str(round_paths["context"]),
        "--save-plan",
        str(round_paths["plan"]),
        "--save-report",
        str(round_paths["plan_report"]),
        "--universal-experience-lib",
        str(paths["universal_library"]),
    ]
    if args.model:
        cmd.extend(["--model", str(args.model)])
    if args.visual_index_dir is not None:
        cmd.extend(["--visual-index-dir", str(_resolve_workspace_path(args.visual_index_dir))])
    if bool(args.use_visual_retrieval):
        cmd.append("--use-visual-retrieval")
    if bool(args.dry_run_recovery_llm):
        cmd.append("--dry-run-llm")
    if bool(args.execute_recovery_candidate_validation):
        cmd.extend([
            "--execute-candidate-validation",
            "--validation-model-path",
            str(model_path),
            "--validation-output-dir",
            str(round_paths["validation"]),
            "--validation-settle-before-steps",
            str(max(0, int(args.settle_before_steps))),
            "--validation-limit",
            str(max(1, int(args.recovery_validation_limit))),
        ])
    return _run_step(f"round_{round_paths['root'].name}_generate_recovery_plan", cmd)


def _run_recovery_round(args: argparse.Namespace, paths: dict[str, Path], round_paths: dict[str, Path], model_path: Path) -> dict[str, Any]:
    nominal_payload = _read_json(paths["nominal_report"])
    source_failure_experience_id = ""
    if isinstance(nominal_payload, dict):
        source_failure_experience_id = str(nominal_payload.get("experience_failure_episode_id") or "")
    cmd = [
        sys.executable,
        "-B",
        "source/active/run_field_atomic_skill_smoke.py",
        "--model-path",
        str(model_path),
        "--actions",
        str(round_paths["plan"]),
        "--initial-state",
        str(paths["replay_state"]),
        "--scenario-id",
        str(args.scenario_id),
        "--condition-id",
        f"recovery_{round_paths['root'].name}",
        "--goal",
        str(args.goal),
        "--settle-before-steps",
        str(max(0, int(args.settle_before_steps))),
        "--save-report",
        str(round_paths["execution_report"]),
        "--save-experience-library",
        str(round_paths["experience_library"]),
        "--experience-read",
        str(paths["universal_library"]),
        "--experience-save-mode",
        "all",
        "--llm-critic",
        "--llm-critic-provider",
        str(args.provider),
        "--llm-critic-model",
        str(args.model),
        "--episode-role",
        "recovery",
        "--source-failure-report",
        str(paths["nominal_report"]),
        "--source-failure-experience-id",
        source_failure_experience_id,
        "--source-recovery-plan",
        str(round_paths["plan"]),
        "--apply-memory-lifecycle",
        "--skip-visual-index",
        "--keyframe-dir",
        str(round_paths["keyframes"]),
        "--keyframe-camera",
        "workspace_overview_camera",
    ]
    if args.viewer:
        cmd.extend(["--viewer", "--viewer-hold-seconds", str(float(args.viewer_hold_seconds))])
    return _run_step(f"round_{round_paths['root'].name}_recovery_execution", cmd)


def _round_report(
    *,
    index: int,
    round_paths: dict[str, Path],
    plan_command: dict[str, Any],
    execution_command: dict[str, Any] | None,
    memory_writeback: dict[str, Any],
    memory_rule_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "tabletop_recovery_batch_round_report_v1",
        "round_index": index,
        "round_dir": str(round_paths["root"]),
        "plan": _plan_summary(round_paths["plan"]),
        "plan_report": str(round_paths["plan_report"]),
        "execution": _execution_summary(round_paths["execution_report"]),
        "experience_library": str(round_paths["experience_library"]),
        "memory_rule_report": str(round_paths["memory_rule_report"]),
        "memory_rules": memory_rule_report or {},
        "memory_writeback": memory_writeback,
        "commands": {
            "generate_recovery_plan": plan_command,
            "recovery_execution": execution_command,
        },
    }


def main() -> None:
    args = parse_args()
    save_dir = _resolve_workspace_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(save_dir)
    for item in paths.values():
        if isinstance(item, Path):
            item.parent.mkdir(parents=True, exist_ok=True)

    _log(f"save_dir={save_dir}")
    model_path = _copy_scene_files(args, paths)
    commands: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    final_status = "not_started"

    if not args.skip_nominal:
        result = _run_nominal(args, paths, model_path)
        commands.append(result)
        if not result["success"] and not paths["nominal_report"].exists():
            final_status = "nominal_execution_command_failed"
            _write_json(paths["batch_report"], {
                "schema_version": "tabletop_recovery_batch_report_v1",
                "scenario_id": args.scenario_id,
                "final_status": final_status,
                "save_dir": str(save_dir),
                "commands": commands,
                "rounds": rounds,
            })
            print(json.dumps({"final_status": final_status, "save_dir": str(save_dir), "batch_report": str(paths["batch_report"])}, ensure_ascii=False))
            return
        nominal_merge = _merge_experience_artifacts(paths["universal_library"], [paths["nominal_library"]])
        _log(f"nominal memory merged candidate_count={nominal_merge['candidate_count']} after_count={nominal_merge['after_count']}")
    elif not paths["nominal_report"].exists():
        raise FileNotFoundError(f"nominal report missing: {paths['nominal_report']}")

    stop_index = _select_anomaly_stop_index(paths["nominal_report"], args.anomaly_stop_index)
    if not args.skip_replay:
        result = _run_replay(args, paths, model_path, stop_index)
        commands.append(result)
        if not result["success"]:
            final_status = "anomaly_replay_failed"
            _write_json(paths["batch_report"], {
                "schema_version": "tabletop_recovery_batch_report_v1",
                "scenario_id": args.scenario_id,
                "final_status": final_status,
                "save_dir": str(save_dir),
                "anomaly_stop_index": stop_index,
                "commands": commands,
                "rounds": rounds,
            })
            print(json.dumps({"final_status": final_status, "save_dir": str(save_dir), "batch_report": str(paths["batch_report"])}, ensure_ascii=False))
            return
    elif not paths["replay_state"].exists():
        raise FileNotFoundError(f"replay state missing: {paths['replay_state']}")

    start_round_index = _next_round_index(save_dir) if bool(args.append_rounds) else 0
    if bool(args.append_rounds):
        _log(f"append_rounds enabled start_round_index={start_round_index}")

    for index in range(start_round_index, start_round_index + max(0, int(args.rounds))):
        rp = _round_paths(save_dir, index)
        rp["root"].mkdir(parents=True, exist_ok=True)
        _log(f"stage recovery_round index={index}")

        plan_result = _run_plan_round(args, paths, rp, model_path)
        commands.append(plan_result)
        execution_result: dict[str, Any] | None = None
        memory_writeback = {
            "library_path": str(paths["universal_library"]),
            "candidate_count": 0,
            "written_count": 0,
            "skipped_count": 0,
            "decisions": [],
        }
        memory_rule_report: dict[str, Any] | None = None
        if plan_result["success"]:
            execution_result = _run_recovery_round(args, paths, rp, model_path)
            commands.append(execution_result)
            if rp["experience_library"].exists():
                memory_writeback = _merge_experience_artifacts(paths["universal_library"], [rp["experience_library"]])
                _log(
                    f"round {index} memory merged candidate_count={memory_writeback['candidate_count']} "
                    f"after_count={memory_writeback.get('after_count')}"
                )
        memory_rule_report = _write_memory_rule_report(
            paths["universal_library"],
            rp["memory_rule_report"],
            scenario_id=str(args.scenario_id),
        )
        report = _round_report(
            index=index,
            round_paths=rp,
            plan_command=plan_result,
            execution_command=execution_result,
            memory_writeback=memory_writeback,
            memory_rule_report=memory_rule_report,
        )
        _write_json(rp["round_report"], report)
        rounds.append(report)

        plan_actions = report["plan"].get("action_sequence", [])
        execution = report["execution"]
        _log(
            f"round {index} plan_actions={plan_actions} "
            f"task_success={execution.get('task_success')} failure_count={execution.get('failure_count')}"
        )
        if (not plan_result["success"] or not (execution_result or {}).get("success", False)) and not bool(args.continue_on_round_failure):
            final_status = "round_failed"
            break
    else:
        final_status = "batch_completed"

    batch_report = {
        "schema_version": "tabletop_recovery_batch_report_v1",
        "scenario_id": args.scenario_id,
        "goal": args.goal,
        "target_class": args.target_class,
        "final_status": final_status,
        "save_dir": str(save_dir),
        "scene": {
            "xml": str(paths["scene_xml"]),
            "report": str(paths["scene_report"]),
            "observation": str(paths["scene_observation"]),
        },
        "nominal": _execution_summary(paths["nominal_report"]),
        "replay": {
            "state": str(paths["replay_state"]),
            "report": str(paths["replay_report"]),
            "anomaly_stop_index": stop_index,
        },
        "memory": {
            "universal_library": str(paths["universal_library"]),
            "nominal_library": str(paths["nominal_library"]),
        },
        "round_count": len(rounds),
        "round_start_index": start_round_index,
        "append_rounds": bool(args.append_rounds),
        "rounds": [
            {
                "round_index": item.get("round_index"),
                "round_report": str(save_dir / "rounds" / f"round_{int(item.get('round_index', 0)):03d}" / "round_report.json"),
                "plan_actions": item.get("plan", {}).get("action_sequence", []),
                "execution": item.get("execution", {}),
                "memory_writeback": item.get("memory_writeback", {}),
            }
            for item in rounds
        ],
        "commands": commands,
    }
    _write_json(paths["batch_report"], batch_report)
    print(json.dumps({
        "final_status": final_status,
        "save_dir": str(save_dir),
        "batch_report": str(paths["batch_report"]),
        "round_count": len(rounds),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
