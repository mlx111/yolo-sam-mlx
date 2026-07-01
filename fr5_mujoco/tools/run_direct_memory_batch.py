#!/usr/bin/env python3
"""Convenience launcher for FR5 direct-memory batch runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


FR5_ROOT = Path(__file__).resolve().parents[1]
BATCH_RUNNER = FR5_ROOT / "run_experiment_batch.py"
DEFAULT_OUTPUT_DIR = FR5_ROOT / "results" / "fr5_direct_memory_batch"
DEFAULT_EXPERIENCE_READ = FR5_ROOT / "results" / "memory" / "fr5_experience_library.json"
DEFAULT_SCENE_XML = FR5_ROOT / "assets" / "scene.xml"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_batch_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scenario_id": args.scenario_id,
        "condition_id": args.condition_id,
        "methods": args.methods,
        "trials_per_method": args.trials,
        "seed_start": args.seed_start,
        "output_dir": str(args.output_dir),
        "experience_read": str(args.experience_read) if args.experience_read else "",
        "rolling_memory": args.rolling_memory,
        "experience_save_mode": args.experience_save_mode,
        "scene_xml": str(args.scene_xml),
        "target_class": args.target_class,
        "place_target": args.place_target,
        "no_viewer": args.no_viewer,
        "no_llm": args.no_llm,
        "recovery_candidate_count": args.recovery_candidate_count,
        "continue_on_failure": args.continue_on_failure,
    }


def run_batch(config_path: Path, *, dry_run: bool = False) -> int:
    cmd = [sys.executable, str(BATCH_RUNNER), "--config", str(config_path)]
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("EXPERIENCE_RUNTIME_SKILLS_ROOT", str(FR5_ROOT))
    env.setdefault("EXPERIENCE_RUNTIME_SKILLS_MODULE", "skills.registry")
    print("Running batch:")
    print(" ".join(cmd))
    print(f"cwd={FR5_ROOT}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(FR5_ROOT), env=env).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated FR5 direct-memory experiments.")
    parser.add_argument("--scenario-id", default="fr5_direct_recovery")
    parser.add_argument("--condition-id", default="direct")
    parser.add_argument("--methods", nargs="+", default=["direct_llm_weak", "direct_memory", "hierarchical_no_failed"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=3100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experience-read", type=Path, default=DEFAULT_EXPERIENCE_READ)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--target-class", default="apple")
    parser.add_argument("--place-target", default="plate")
    parser.add_argument("--experience-save-mode", choices=["all", "success_only", "failure_only", "none"], default="all")
    parser.add_argument("--rolling-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-llm", action="store_true", default=False)
    parser.add_argument("--recovery-candidate-count", type=int, default=1)
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.experience_read = args.experience_read.resolve() if args.experience_read else None
    args.scene_xml = args.scene_xml.resolve()
    return args


def main() -> int:
    args = parse_args()
    config = build_batch_config(args)
    config_path = args.output_dir / "effective_batch_config.json"
    _write_json(config_path, config)
    return run_batch(config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
