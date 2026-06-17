#!/usr/bin/env python3
"""
Batch runner for experiment_method_runner.py.

This script is intentionally separate from run_experiment_v4.py. It launches
each trial as a subprocess so failures are isolated and every trial keeps its
own result, plan, experience snapshot, and stdout/stderr log.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
METHOD_RUNNER = ROOT / "experiment_method_runner.py"
sys.path.insert(0, str(ROOT))
from experiment_method_runner import METHOD_DEFAULTS


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _default_experience_write(output_dir: Path, method: str, anomaly: str, trial_index: int) -> Path:
    return output_dir / anomaly / method / f"trial_{trial_index:03d}" / "experience_after.json"


def _trial_dir(output_dir: Path, anomaly: str, method: str, trial_index: int, strategy_family: str = "") -> Path:
    if strategy_family:
        return output_dir / anomaly / strategy_family / method / f"trial_{trial_index:03d}"
    return output_dir / anomaly / method / f"trial_{trial_index:03d}"


def _condition_trial_dir(output_dir: Path, scenario_id: str, condition_id: str, method: str, trial_index: int, strategy_family: str = "") -> Path:
    if strategy_family:
        return output_dir / scenario_id / condition_id / strategy_family / method / f"trial_{trial_index:03d}"
    return output_dir / scenario_id / condition_id / method / f"trial_{trial_index:03d}"


def _build_command(
    *,
    method: str,
    anomaly: str,
    condition_id: str = "",
    scenario_id: str = "",
    trial_index: int,
    seed: int,
    output_dir: Path,
    experience_read: str | None,
    no_viewer: bool,
    no_inject: bool,
    noise_scale: float,
    object_displaced_dx: float | None,
    object_displaced_dy: float | None,
    use_memory_keyframes: bool,
    memory_keyframe_top_k: int,
    enable_failed_plan_rewrite: bool,
    inject_failed_plan_for_test: bool,
    memory_index_dir: str | None,
    experience_save_mode: str = "all",
    strategy_family: str = "",
    experience_write_path: str | None = None,
) -> tuple[list[str], Path]:
    trial_dir = (
        _condition_trial_dir(output_dir, scenario_id, condition_id, method, trial_index, strategy_family)
        if condition_id
        else _trial_dir(output_dir, anomaly, method, trial_index, strategy_family)
    )
    result_path = trial_dir / "result.json"
    plan_path = trial_dir / "plan.json"
    experience_write = Path(experience_write_path) if experience_write_path else trial_dir / "experience_after.json"
    trial_prefix = condition_id or anomaly
    trial_id = f"{trial_prefix}_{strategy_family + '_' if strategy_family else ''}{method}_{trial_index:03d}"

    cmd = [
        sys.executable,
        str(METHOD_RUNNER),
        "--method",
        method,
        "--anomaly",
        anomaly,
        "--seed",
        str(seed),
        "--trial-id",
        trial_id,
        "--save",
        str(result_path),
        "--save-plan",
        str(plan_path),
        "--experience-write",
        str(experience_write),
        "--noise-scale",
        str(noise_scale),
        "--experience-save-mode",
        experience_save_mode,
    ]
    if condition_id:
        cmd.extend(["--condition-id", condition_id])
    if object_displaced_dx is not None:
        cmd.extend(["--object-displaced-dx", str(object_displaced_dx)])
    if object_displaced_dy is not None:
        cmd.extend(["--object-displaced-dy", str(object_displaced_dy)])
    if experience_read:
        cmd.extend(["--experience-read", experience_read])
    if no_viewer:
        cmd.append("--no-viewer")
    if no_inject:
        cmd.append("--no-inject")
    if use_memory_keyframes:
        cmd.append("--use-memory-keyframes")
        cmd.extend(["--memory-keyframe-top-k", str(memory_keyframe_top_k)])
    if enable_failed_plan_rewrite:
        cmd.append("--enable-failed-plan-rewrite")
    if inject_failed_plan_for_test:
        cmd.append("--inject-failed-plan-for-test")
    if memory_index_dir:
        cmd.extend(["--memory-index-dir", memory_index_dir])
    if strategy_family:
        cmd.extend(["--strategy-family", strategy_family])
    return cmd, trial_dir


def _method_uses_memory(method: str) -> bool:
    """Check if a method uses an experience library (memory_policy != 'none')."""
    config = METHOD_DEFAULTS.get(method, {})
    return config.get("memory_policy", "none") != "none"


def _method_rolling_save_mode(method: str) -> str:
    """Default write policy for online-memory growth experiments."""
    if method in {"sim_memory_weak", "hierarchical_no_failed"}:
        return "success_only"
    if method in {"hierarchical_memory_weak", "direct_memory"}:
        return "all"
    return "success_only"


def _copy_memory_snapshot(src: Path | None, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is not None and src.exists():
        shutil.copy2(src, dst)
    else:
        _write_json(dst, {})


def _resolve_root_relative(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _get_method_library_path(config_lib: Any, method: str) -> str | None:
    """Resolve experience library path for a given method.

    Supports both:
      - string: shared across all methods
      - dict[str, str]: per-method paths, falls back to None if method missing
    """
    if isinstance(config_lib, dict):
        return config_lib.get(method)
    if isinstance(config_lib, str):
        return config_lib
    return None


def _rolling_cell_dir(output_dir: Path, anomaly: str, method: str, strategy_family: str = "") -> Path:
    if strategy_family:
        return output_dir / anomaly / strategy_family / method
    return output_dir / anomaly / method


def _summarize_trial(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        return {"result_exists": False}
    try:
        data = _load_json(result_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "result_exists": True,
            "result_readable": False,
            "read_error": str(exc),
        }
    return {
        "result_exists": True,
        "result_readable": True,
        "method": data.get("method"),
        "memory_policy": data.get("memory_policy"),
        "trial_id": data.get("trial_id"),
        "seed": data.get("seed"),
        "anomaly_detected": data.get("anomaly_detected"),
        "recovery_success": data.get("recovery_success"),
        "task_success": data.get("task_success"),
        "virtual_validation_success": data.get("virtual_validation_success"),
        "virtual_validation_z_change": data.get("virtual_validation_z_change"),
        "executed_plan_source": data.get("executed_plan_source"),
        "retrieved_count": len(data.get("retrieved_memories") or []),
        "retrieved_positive_count": data.get("retrieved_positive_count"),
        "retrieved_failed_count": data.get("retrieved_failed_count"),
        "experience_save_mode": data.get("experience_save_mode"),
        "experience_saved": data.get("experience_saved"),
        "experience_save_skipped_reason": data.get("experience_save_skipped_reason"),
        "invalid_plan_count": data.get("invalid_plan_count"),
        "unsafe_gripper_action_count": data.get("unsafe_gripper_action_count"),
        "recovery_time": (data.get("time_costs") or {}).get("recovery"),
        "total_time": (data.get("time_costs") or {}).get("total"),
        "prompt_keyframe_count": data.get("prompt_keyframe_count", 0),
        "strategy_family": data.get("strategy_family", ""),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = f"{rec.get('anomaly')}::{rec.get('method')}"
        grouped.setdefault(key, []).append(rec)

    cells = []
    for key, items in sorted(grouped.items()):
        ok = [r for r in items if r.get("returncode") == 0 and r.get("summary", {}).get("result_exists")]
        n = len(ok)
        if n == 0:
            cells.append({"cell": key, "n": 0, "runs": len(items)})
            continue
        summaries = [r["summary"] for r in ok]
        recovery = [bool(s.get("recovery_success")) for s in summaries]
        task = [bool(s.get("task_success")) for s in summaries]
        retrieval = [float(s.get("retrieved_count") or 0) for s in summaries]
        recovery_times = [float(s["recovery_time"]) for s in summaries if s.get("recovery_time") is not None]
        cells.append(
            {
                "cell": key,
                "n": n,
                "runs": len(items),
                "recovery_success_rate": sum(recovery) / n,
                "task_success_rate": sum(task) / n,
                "retrieved_count_mean": sum(retrieval) / n,
                "recovery_time_mean": sum(recovery_times) / len(recovery_times) if recovery_times else None,
            }
        )
    return {"cells": cells, "records": records}


def _should_skip_trial(
    *,
    result_path: Path,
    trial_dir: Path,
    resume: bool,
    skip_existing: bool,
) -> tuple[bool, str]:
    """Decide whether a trial should be skipped before launching subprocess."""
    if resume:
        summary = _summarize_trial(result_path)
        if summary.get("result_exists") and summary.get("result_readable", True):
            return True, "resume_existing_result"
    if skip_existing and trial_dir.exists() and any(trial_dir.iterdir()):
        return True, "skip_existing_trial_dir"
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch runner for anomaly recovery method experiments")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file")
    parser.add_argument("--anomaly", action="append", default=None, help="Anomaly type; repeatable")
    parser.add_argument("--condition-id", action="append", default=None, help="UR5E condition id; repeatable, e.g. U2-1")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--trials-per-method", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--experience-read", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--object-displaced-dx", type=float, default=None)
    parser.add_argument("--object-displaced-dy", type=float, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-inject", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--use-memory-keyframes", action="store_true")
    parser.add_argument("--memory-keyframe-top-k", type=int, default=None)
    parser.add_argument("--enable-failed-plan-rewrite", action="store_true")
    parser.add_argument("--inject-failed-plan-for-test", action="store_true")
    parser.add_argument("--memory-index-dir", type=str, default=None)
    parser.add_argument("--strategy-family", action="append", default=None, help="Strategy family; repeatable")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trials with an existing readable result.json",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials whose trial directory already contains files",
    )
    parser.add_argument(
        "--rolling-memory",
        action="store_true",
        help="Feed each trial's experience_after.json into the next trial",
    )
    parser.add_argument(
        "--rolling-memory-scope",
        choices=["cell", "scenario", "global"],
        default=None,
        help="cell keeps condition/method memories separate; scenario shares across conditions in a scenario; global shares across the whole batch",
    )
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.config:
        config = _load_json(Path(args.config))

    conditions = args.condition_id or config.get("conditions") or []
    anomalies = args.anomaly or config.get("anomalies") or ["grasp_miss"]
    methods = args.methods or config.get("methods") or [
        "direct_llm_weak",
        "sim_only_weak",
        "sim_memory_weak",
        "hierarchical_memory_weak",
    ]
    trials_per_method = args.trials_per_method if args.trials_per_method is not None else int(config.get("trials_per_method", 1))
    seed_start = args.seed_start if args.seed_start is not None else int(config.get("seed_start", 1000))
    raw_experience_library = config.get("experience_library")
    experience_read = (
        raw_experience_library
        if isinstance(raw_experience_library, str)
        else (args.experience_read if args.experience_read is not None else config.get("experience_read"))
    )
    output_dir = Path(args.output_dir or config.get("output_dir") or "results/batch_experiment").resolve()
    noise_scale = args.noise_scale if args.noise_scale is not None else float(config.get("noise_scale", 0.0))
    object_displaced_dx = args.object_displaced_dx if args.object_displaced_dx is not None else config.get("object_displaced_dx")
    object_displaced_dy = args.object_displaced_dy if args.object_displaced_dy is not None else config.get("object_displaced_dy")
    no_viewer = bool(args.no_viewer or config.get("no_viewer", True))
    no_inject = bool(args.no_inject or config.get("no_inject", False))
    resume = bool(args.resume or config.get("resume", False))
    skip_existing = bool(args.skip_existing or config.get("skip_existing", False))
    stop_on_failure = bool(args.stop_on_failure or config.get("stop_on_failure", False))
    use_memory_keyframes = bool(args.use_memory_keyframes or config.get("use_memory_keyframes", False))
    memory_keyframe_top_k = args.memory_keyframe_top_k if args.memory_keyframe_top_k is not None else int(config.get("memory_keyframe_top_k", 2))
    enable_failed_plan_rewrite = bool(args.enable_failed_plan_rewrite or config.get("enable_failed_plan_rewrite", False))
    inject_failed_plan_for_test = bool(args.inject_failed_plan_for_test or config.get("inject_failed_plan_for_test", False))
    memory_index_dir = args.memory_index_dir if args.memory_index_dir is not None else config.get("memory_index_dir")
    strategy_families = args.strategy_family if args.strategy_family is not None else config.get("strategy_families")
    if not strategy_families:
        strategy_families = [""]
    rolling_memory = bool(args.rolling_memory or config.get("rolling_memory", False))
    rolling_memory_scope = args.rolling_memory_scope or config.get("rolling_memory_scope", "scenario")
    rolling_memory_save_modes = config.get("rolling_memory_save_modes") or {}

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "batch_config.json",
        {
            "anomalies": anomalies,
            "conditions": conditions,
            "methods": methods,
            "trials_per_method": trials_per_method,
            "seed_start": seed_start,
            "experience_read": experience_read,
            "output_dir": str(output_dir),
            "noise_scale": noise_scale,
            "object_displaced_dx": object_displaced_dx,
            "object_displaced_dy": object_displaced_dy,
            "no_viewer": no_viewer,
            "no_inject": no_inject,
            "resume": resume,
            "skip_existing": skip_existing,
            "stop_on_failure": stop_on_failure,
            "use_memory_keyframes": use_memory_keyframes,
            "memory_keyframe_top_k": memory_keyframe_top_k,
            "enable_failed_plan_rewrite": enable_failed_plan_rewrite,
            "inject_failed_plan_for_test": inject_failed_plan_for_test,
            "memory_index_dir": memory_index_dir,
            "strategy_families": strategy_families,
            "rolling_memory": rolling_memory,
            "rolling_memory_scope": rolling_memory_scope,
            "rolling_memory_save_modes": rolling_memory_save_modes,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    records: list[dict[str, Any]] = []
    rolling_memory_paths: dict[str, Path] = {}
    using_unified_library = bool(config.get("experience_library"))
    if using_unified_library:
        rolling_memory = False
    initial_experience_path = _resolve_root_relative(
        experience_read if isinstance(raw_experience_library, str) else None
    )
    if rolling_memory and rolling_memory_scope == "global":
        global_rolling_path = output_dir / "rolling_memory.json"
        _copy_memory_snapshot(initial_experience_path, global_rolling_path)
        rolling_memory_paths["global"] = global_rolling_path
    condition_specs: dict[str, dict[str, str]] = {}
    if conditions:
        from ur5e.anomaly_conditions import get_condition_spec
        for condition_id in conditions:
            spec = get_condition_spec(condition_id)
            condition_specs[condition_id] = {
                "scenario_id": spec.scenario_id,
                "anomaly": spec.legacy_anomaly_type,
            }
    run_items = [
        {
            "anomaly": condition_specs[c]["anomaly"],
            "condition_id": c,
            "scenario_id": condition_specs[c]["scenario_id"],
        }
        for c in conditions
    ] if conditions else [
        {"anomaly": a, "condition_id": "", "scenario_id": ""}
        for a in anomalies
    ]
    total = len(run_items) * len(strategy_families) * len(methods) * trials_per_method
    run_index = 0
    for run_item in run_items:
        anomaly = run_item["anomaly"]
        condition_id = run_item["condition_id"]
        scenario_id = run_item["scenario_id"]
        for strategy_family in strategy_families:
            for method in methods:
                if rolling_memory and rolling_memory_scope == "scenario":
                    cell_key = f"scenario::{scenario_id or anomaly}::{strategy_family}::{method}"
                else:
                    cell_key = f"{condition_id or anomaly}::{strategy_family}::{method}"
                if rolling_memory and rolling_memory_scope == "cell":
                    cell_dir = _rolling_cell_dir(output_dir, anomaly, method, strategy_family)
                    initial_path = cell_dir / "rolling_memory_initial.json"
                    current_path = cell_dir / "rolling_memory.json"
                    _copy_memory_snapshot(initial_experience_path, initial_path)
                    _copy_memory_snapshot(initial_path, current_path)
                    rolling_memory_paths[cell_key] = current_path
                elif rolling_memory and rolling_memory_scope == "scenario" and cell_key not in rolling_memory_paths:
                    scenario_dir = output_dir / "rolling" / cell_key
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    current_path = scenario_dir / "rolling_memory.json"
                    _copy_memory_snapshot(initial_experience_path, current_path)
                    rolling_memory_paths[cell_key] = current_path
                for trial_index in range(trials_per_method):
                    run_index += 1
                    seed = seed_start + trial_index
                    rolling_key = "global" if rolling_memory_scope == "global" else cell_key
                    rolling_read_path = rolling_memory_paths.get(rolling_key)
                    trial_experience_read = (
                        str(rolling_read_path)
                        if rolling_memory and rolling_read_path is not None
                        else _get_method_library_path(raw_experience_library, method)
                        or experience_read
                    )
                    # Auto-enable memory keyframes for any method that uses an experience library
                    method_use_keyframes = use_memory_keyframes or _method_uses_memory(method)
                    experience_save_mode = str(
                        rolling_memory_save_modes.get(method)
                        or (_method_rolling_save_mode(method) if rolling_memory else "all")
                    )
                    cmd, trial_dir = _build_command(
                        method=method,
                        anomaly=anomaly,
                        condition_id=condition_id,
                        scenario_id=scenario_id,
                        trial_index=trial_index,
                        seed=seed,
                        output_dir=output_dir,
                        experience_read=trial_experience_read,
                        no_viewer=no_viewer,
                        no_inject=no_inject,
                        noise_scale=noise_scale,
                        object_displaced_dx=object_displaced_dx,
                        object_displaced_dy=object_displaced_dy,
                        use_memory_keyframes=method_use_keyframes,
                        memory_keyframe_top_k=memory_keyframe_top_k,
                        enable_failed_plan_rewrite=enable_failed_plan_rewrite,
                        inject_failed_plan_for_test=inject_failed_plan_for_test,
                        memory_index_dir=memory_index_dir,
                        experience_save_mode=experience_save_mode,
                        experience_write_path=_get_method_library_path(raw_experience_library, method),
                        strategy_family=strategy_family,
                    )
                    result_path = trial_dir / "result.json"
                    log_path = trial_dir / "stdout.log"
                    should_skip, skip_reason = _should_skip_trial(
                        result_path=result_path,
                        trial_dir=trial_dir,
                        resume=resume,
                        skip_existing=skip_existing,
                    )
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"\n[{run_index}/{total}] anomaly={anomaly} condition={condition_id or '-'} strategy={strategy_family or '-'} "
                        f"method={method} trial={trial_index:03d} seed={seed}"
                    )
                    print(" ".join(cmd))
                    if should_skip:
                        summary = _summarize_trial(result_path)
                        experience_after_path = trial_dir / "experience_after.json"
                        if rolling_memory and experience_after_path.exists():
                            target_path = rolling_memory_paths.get(rolling_key)
                            if target_path is not None:
                                shutil.copy2(experience_after_path, target_path)
                        record = {
                            "anomaly": anomaly,
                            "scenario_id": scenario_id,
                            "condition_id": condition_id,
                            "method": method,
                            "strategy_family": strategy_family,
                            "trial_index": trial_index,
                            "seed": seed,
                            "returncode": 0 if summary.get("result_exists") and summary.get("result_readable", True) else None,
                            "elapsed": 0.0,
                            "trial_dir": str(trial_dir),
                            "log_path": str(log_path),
                            "summary": summary,
                            "skipped": True,
                            "skip_reason": skip_reason,
                            "rolling_memory": rolling_memory,
                            "rolling_memory_scope": rolling_memory_scope,
                            "rolling_memory_read": str(rolling_read_path) if rolling_read_path else None,
                            "rolling_memory_write": str(experience_after_path) if rolling_memory else None,
                            "experience_save_mode": experience_save_mode,
                        }
                        records.append(record)
                        _write_json(output_dir / "batch_progress.json", _aggregate(records))
                        print(f"  skipped={skip_reason} summary={summary}")
                        continue

                    started = time.time()
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    elapsed = round(time.time() - started, 3)
                    log_path.write_text(proc.stdout or "")
                    summary = _summarize_trial(result_path)
                    experience_after_path = trial_dir / "experience_after.json"
                    if rolling_memory and proc.returncode == 0 and experience_after_path.exists():
                        target_path = rolling_memory_paths.get(rolling_key)
                        if target_path is not None:
                            shutil.copy2(experience_after_path, target_path)
                    record = {
                        "anomaly": anomaly,
                        "scenario_id": scenario_id,
                        "condition_id": condition_id,
                        "method": method,
                        "strategy_family": strategy_family,
                        "trial_index": trial_index,
                        "seed": seed,
                        "returncode": proc.returncode,
                        "elapsed": elapsed,
                        "trial_dir": str(trial_dir),
                        "log_path": str(log_path),
                        "summary": summary,
                        "rolling_memory": rolling_memory,
                        "rolling_memory_scope": rolling_memory_scope,
                        "rolling_memory_read": str(rolling_read_path) if rolling_read_path else None,
                        "rolling_memory_write": str(experience_after_path) if rolling_memory else None,
                        "experience_save_mode": experience_save_mode,
                    }
                    records.append(record)
                    _write_json(output_dir / "batch_progress.json", _aggregate(records))
                    print(f"  returncode={proc.returncode} elapsed={elapsed}s summary={summary}")
                    if proc.returncode != 0 and stop_on_failure:
                        _write_json(output_dir / "batch_summary.json", _aggregate(records))
                        return proc.returncode

    _write_json(output_dir / "batch_summary.json", _aggregate(records))
    print(f"\nBatch finished. Summary: {output_dir / 'batch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
