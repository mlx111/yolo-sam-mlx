#!/usr/bin/env python3
"""Batch runner for FR5 experience method trials."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
METHOD_RUNNER = ROOT / "experiment_method_runner.py"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_memory_snapshot(src: Path | None, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is not None and src.exists():
        shutil.copy2(src, dst)
    else:
        _write_json(dst, {})


def _trial_dir(output_dir: Path, scenario_id: str, condition_id: str, method: str, trial_index: int) -> Path:
    return output_dir / scenario_id / condition_id / method / f"trial_{trial_index:03d}"


def _build_command(
    *,
    method: str,
    trial_index: int,
    seed: int,
    trial_dir: Path,
    config: dict[str, Any],
    experience_read: Path | None,
    experience_write: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(METHOD_RUNNER),
        "--method",
        method,
        "--trial-id",
        f"{config['condition_id']}_{method}_{trial_index:03d}",
        "--seed",
        str(seed),
        "--save",
        str(trial_dir / "result.json"),
        "--save-plan",
        str(trial_dir / "plan.json"),
        "--experience-write",
        str(experience_write),
        "--experience-save-mode",
        str(config.get("experience_save_mode", "all")),
        "--scene-xml",
        str(config["scene_xml"]),
        "--scenario-id",
        str(config["scenario_id"]),
        "--condition-id",
        str(config["condition_id"]),
        "--target-class",
        str(config.get("target_class", "apple")),
        "--place-target",
        str(config.get("place_target", "plate")),
        "--recovery-candidate-count",
        str(int(config.get("recovery_candidate_count", 1))),
    ]
    if experience_read is not None:
        cmd.extend(["--experience-read", str(experience_read)])
    if bool(config.get("no_viewer", True)):
        cmd.append("--no-viewer")
    if bool(config.get("no_llm", False)):
        cmd.append("--no-llm")
    if bool(config.get("continue_on_failure", False)):
        cmd.append("--continue-on-failure")
    return cmd


def _summarize_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"result_exists": False}
    try:
        data = _load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {"result_exists": True, "result_readable": False, "read_error": str(exc)}
    return {
        "result_exists": True,
        "result_readable": True,
        "method": data.get("method"),
        "trial_id": data.get("trial_id"),
        "task_success": data.get("task_success"),
        "executed_plan_source": data.get("executed_plan_source"),
        "retrieved_count": len(data.get("retrieved_memories") or []),
        "candidate_count": len(data.get("candidate_plans") or []),
        "failure_reason": data.get("failure_reason", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FR5 direct-memory batch experiments.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = _load_json(args.config)
    output_dir = Path(raw.get("output_dir") or ROOT / "results" / "fr5_direct_memory_batch").resolve()
    scenario_id = str(raw.get("scenario_id") or "fr5_direct_recovery")
    condition_id = str(raw.get("condition_id") or "direct")
    methods = [str(item) for item in (raw.get("methods") or ["direct_memory"])]
    trials = int(raw.get("trials_per_method") or raw.get("trials") or 1)
    seed_start = int(raw.get("seed_start") or 0)
    scene_xml = Path(raw.get("scene_xml") or ROOT / "assets" / "scene.xml").resolve()
    experience_read = Path(raw["experience_read"]).resolve() if raw.get("experience_read") else None
    rolling_memory = bool(raw.get("rolling_memory", True))
    config = {
        **raw,
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "scene_xml": str(scene_xml),
    }
    _write_json(output_dir / "effective_batch_config.json", config)
    rolling_path = output_dir / "memory" / scenario_id / "rolling_memory.json"
    _copy_memory_snapshot(experience_read, rolling_path)
    cells: list[dict[str, Any]] = []
    failures = 0
    for method in methods:
        summaries: list[dict[str, Any]] = []
        for trial_index in range(trials):
            trial_dir = _trial_dir(output_dir, scenario_id, condition_id, method, trial_index)
            trial_dir.mkdir(parents=True, exist_ok=True)
            exp_write = rolling_path if rolling_memory else trial_dir / "experience_after.json"
            exp_read = rolling_path if rolling_memory else experience_read
            cmd = _build_command(
                method=method,
                trial_index=trial_index,
                seed=seed_start + trial_index,
                trial_dir=trial_dir,
                config=config,
                experience_read=exp_read,
                experience_write=exp_write,
            )
            env = os.environ.copy()
            env.setdefault("MUJOCO_GL", "egl")
            env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            env.setdefault("EXPERIENCE_RUNTIME_SKILLS_ROOT", str(ROOT))
            env.setdefault("EXPERIENCE_RUNTIME_SKILLS_MODULE", "skills.registry")
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (trial_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            if proc.returncode != 0:
                failures += 1
            summaries.append(_summarize_result(trial_dir / "result.json"))
        success_count = sum(1 for item in summaries if item.get("task_success"))
        cells.append(
            {
                "cell": method,
                "n": len(summaries),
                "runs": trials,
                "success_count": success_count,
                "success_rate": success_count / max(1, len(summaries)),
                "trials": summaries,
            }
        )
    summary = {"schema_version": "fr5_batch_summary_v1", "cells": cells, "failure_process_count": failures}
    _write_json(output_dir / "batch_summary.json", summary)
    _write_json(output_dir / "batch_progress.json", {"completed": True, "cells": len(cells)})
    print(f"saved batch summary: {output_dir / 'batch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
