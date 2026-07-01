#!/usr/bin/env python3
"""Run repeated UR5e memory experiments with the current direct MuJoCo flow.

This is a small, explicit entry point for the current workflow:

1. Run an anomaly condition in the generated MuJoCo scene.
2. Query/use the experience library.
3. Let the LLM generate a recovery skill plan.
4. Execute that plan in the same MuJoCo scene.
5. Save per-trial results and print success rates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


UR5E_ROOT = Path(__file__).resolve().parents[1]
BATCH_RUNNER = UR5E_ROOT / "run_experiment_batch.py"
DEFAULT_OUTPUT_DIR = UR5E_ROOT / "results" / "ur5e_direct_memory_batch"
DEFAULT_EXPERIENCE_READ = UR5E_ROOT / "results" / "memory" / "field_atomic_smoke_experience.json"
DEFAULT_SCENE_XML = UR5E_ROOT / "scene" / "scene.xml"
DEFAULT_COLOR_IMAGE = UR5E_ROOT / "inputs" / "cleft001.png"
DEFAULT_DEPTH_IMAGE = UR5E_ROOT / "inputs" / "dleft001.png"
DEFAULT_SCENE_TEMPLATE = UR5E_ROOT / "assets" / "scenes" / "scene2.xml"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_batch_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the config consumed by run_experiment_batch.py."""
    return {
        "conditions": args.condition_id,
        "methods": args.methods,
        "trials_per_method": args.trials,
        "seed_start": args.seed_start,
        "output_dir": str(args.output_dir),
        "experience_read": str(args.experience_read),
        "rolling_memory": True,
        "rolling_memory_scope": args.rolling_memory_scope,
        "rolling_memory_save_modes": {
            "direct_llm_weak": "all",
            "direct_memory": "all",
            "hierarchical_memory_weak": "all",
            "hierarchical_no_failed": "success_only",
            "dual_source_gap_memory": "all",
            "dual_source_gap_critic": "all",
        },
        "noise_scale": args.noise_scale,
        "scene_xml": str(args.scene_xml) if args.scene_xml else None,
        "auto_build_runtime_scene": args.auto_build_runtime_scene,
        "runtime_scene_color": str(args.runtime_scene_color),
        "runtime_scene_depth": str(args.runtime_scene_depth),
        "runtime_scene_objects": args.runtime_scene_objects,
        "runtime_scene_template": str(args.runtime_scene_template),
        "runtime_scene_camera": args.runtime_scene_camera,
        "runtime_scene_downsample_scale": args.runtime_scene_downsample_scale,
        "no_viewer": args.no_viewer,
        "no_inject": args.no_inject,
        "use_memory_keyframes": args.use_memory_keyframes,
        "memory_keyframe_top_k": args.memory_keyframe_top_k,
        "enable_failed_plan_rewrite": args.enable_failed_plan_rewrite,
        "recovery_candidate_count": args.recovery_candidate_count,
        "execute_recovery_candidate_validation": args.execute_recovery_candidate_validation,
        "failed_memory_hard_block": args.failed_memory_hard_block,
        "dedupe_failure_memory": args.dedupe_failure_memory,
        "resume": args.resume,
        "stop_on_failure": args.stop_on_failure,
    }


def run_batch(config_path: Path, *, dry_run: bool = False) -> int:
    cmd = [
        sys.executable,
        str(BATCH_RUNNER),
        "--config",
        str(config_path),
    ]
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("UR5E_HEADLESS", "1")

    print("Running batch:")
    print(" ".join(cmd))
    print(f"cwd={UR5E_ROOT}")
    print(
        f"MUJOCO_GL={env['MUJOCO_GL']} "
        f"MPLCONFIGDIR={env['MPLCONFIGDIR']} "
        f"MPLBACKEND={env['MPLBACKEND']} "
        f"UR5E_HEADLESS={env['UR5E_HEADLESS']}"
    )

    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(UR5E_ROOT), env=env).returncode


def print_success_rates(summary_path: Path) -> None:
    if not summary_path.exists():
        print(f"No summary found: {summary_path}")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    print("\nSuccess rates:")
    for cell in data.get("cells", []):
        name = cell.get("cell", "")
        n = cell.get("n", 0)
        runs = cell.get("runs", 0)
        success = cell.get("success_rate")
        retrieved = cell.get("retrieved_count_mean")
        print(
            f"- {name}: valid={n}/{runs}, "
            f"success_rate={success}, "
            f"retrieved_count_mean={retrieved}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated direct UR5e memory experiments.")
    parser.add_argument(
        "--condition-id",
        action="append",
        default=None,
        help="Repeatable UR5e condition id, e.g. --condition-id U3-1 --condition-id U3-2",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["direct_llm_weak", "direct_memory", "hierarchical_no_failed"],
        help="Experiment methods. sim_* methods are kept only for explicit comparison.",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=3100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experience-read", type=Path, default=DEFAULT_EXPERIENCE_READ)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument(
        "--auto-build-runtime-scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="At batch startup, build one runtime scene from RGB-D inputs and reuse it for all trials.",
    )
    parser.add_argument("--runtime-scene-color", type=Path, default=DEFAULT_COLOR_IMAGE)
    parser.add_argument("--runtime-scene-depth", type=Path, default=DEFAULT_DEPTH_IMAGE)
    parser.add_argument("--runtime-scene-objects", nargs="+", default=["apple"])
    parser.add_argument("--runtime-scene-template", type=Path, default=DEFAULT_SCENE_TEMPLATE)
    parser.add_argument("--runtime-scene-camera", default="cam1")
    parser.add_argument("--runtime-scene-downsample-scale", type=float, default=1.0)
    parser.add_argument("--rolling-memory-scope", choices=["cell", "scenario", "global"], default="scenario")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--memory-keyframe-top-k", type=int, default=2)
    parser.add_argument("--no-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-inject", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-memory-keyframes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-failed-plan-rewrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recovery-candidate-count", type=int, default=1)
    parser.add_argument("--execute-recovery-candidate-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--failed-memory-hard-block", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dedupe-failure-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.condition_id:
        args.condition_id = ["U3-1"]
    args.output_dir = args.output_dir.resolve()
    args.experience_read = args.experience_read.resolve()
    args.scene_xml = args.scene_xml.resolve() if args.scene_xml else None
    args.runtime_scene_color = args.runtime_scene_color.resolve()
    args.runtime_scene_depth = args.runtime_scene_depth.resolve()
    args.runtime_scene_template = args.runtime_scene_template.resolve()
    return args


def main() -> int:
    args = parse_args()
    config = build_batch_config(args)
    config_path = args.output_dir / "effective_batch_config.json"
    _write_json(config_path, config)

    returncode = run_batch(config_path, dry_run=args.dry_run)
    if returncode == 0 and not args.dry_run:
        print_success_rates(args.output_dir / "batch_summary.json")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
