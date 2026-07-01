#!/usr/bin/env python3
"""FR5 method-level runner matching the UR5e experience experiment interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experience_system"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("EXPERIENCE_RUNTIME_SKILLS_ROOT", str(ROOT))
os.environ.setdefault("EXPERIENCE_RUNTIME_SKILLS_MODULE", "skills.registry")

from run_experiment_v4 import ExperimentV4, _convert_numpy, _method_policy


METHOD_DEFAULTS = {
    "direct_llm_weak": {"condition": "direct", "memory_policy": "none"},
    "direct_memory": {"condition": "direct", "memory_policy": "hierarchical"},
    "hierarchical_memory_weak": {"condition": "direct", "memory_policy": "hierarchical"},
    "hierarchical_no_failed": {"condition": "direct", "memory_policy": "no_failed"},
    "dual_source_gap_memory": {"condition": "direct", "memory_policy": "dual_source_gap"},
    "dual_source_gap_critic": {"condition": "direct", "memory_policy": "dual_source_gap_critic"},
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_convert_numpy(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FR5 experience method trial.")
    parser.add_argument("--method", default="direct_memory", choices=sorted(METHOD_DEFAULTS))
    parser.add_argument("--trial-id", default="fr5_trial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--save-plan", type=Path, required=True)
    parser.add_argument("--experience-read", type=Path, default=None)
    parser.add_argument("--experience-write", type=Path, default=None)
    parser.add_argument("--experience-save-mode", choices=["all", "success_only", "failure_only", "none"], default="all")
    parser.add_argument("--scene-xml", type=Path, default=ROOT / "assets" / "scene.xml")
    parser.add_argument("--scenario-id", default="fr5_direct_recovery")
    parser.add_argument("--condition-id", default="direct")
    parser.add_argument("--target-class", default="apple")
    parser.add_argument("--place-target", default="plate")
    parser.add_argument("--no-viewer", action="store_true", default=True)
    parser.add_argument("--viewer", dest="no_viewer", action="store_false")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--recovery-candidate-count", type=int, default=1)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--anomaly", default="direct")
    parser.add_argument("--condition", default="direct")
    # Compatibility flags accepted but intentionally unused in the no-U1/U5 FR5 flow.
    parser.add_argument("--no-inject", action="store_true", default=True)
    parser.add_argument("--use-memory-keyframes", action="store_true")
    parser.add_argument("--memory-keyframe-top-k", type=int, default=2)
    parser.add_argument("--enable-failed-plan-rewrite", action="store_true")
    parser.add_argument("--execute-recovery-candidate-validation", action="store_true")
    parser.add_argument("--failed-memory-hard-block", action="store_true")
    parser.add_argument("--dedupe-failure-memory", action="store_true")
    parser.add_argument("--memory-index-dir", default="")
    parser.add_argument("--strategy-family", default="")
    return parser.parse_args()


def _should_write_experience(mode: str, task_success: bool) -> bool:
    if mode == "none":
        return False
    if mode == "success_only":
        return bool(task_success)
    if mode == "failure_only":
        return not bool(task_success)
    return True


def main() -> int:
    args = parse_args()
    defaults = METHOD_DEFAULTS.get(args.method, {})
    memory_policy = str(defaults.get("memory_policy") or _method_policy(args.method))
    experience_write = args.experience_write
    exp = ExperimentV4(
        enable_viewer=not args.no_viewer,
        scene_xml=args.scene_xml,
        save_plan=args.save_plan,
        experience_lib_path=args.experience_read,
        experience_write_path=experience_write,
        condition=args.condition,
        condition_id=args.condition_id,
        scenario_id=args.scenario_id,
        target_class=args.target_class,
        place_target=args.place_target,
    )
    try:
        result = exp.run_recovery(
            method=args.method,
            memory_policy=memory_policy,
            no_llm=args.no_llm,
            candidate_count=args.recovery_candidate_count,
            stop_on_failure=not args.continue_on_failure,
            camera_ready=True,
            include_place=bool(args.place_target),
            save_experience_entry=False,
        )
        should_write = _should_write_experience(args.experience_save_mode, bool(result.get("task_success")))
        if should_write and exp.experience_library is not None and getattr(exp, "_last_experience_entry", None) is not None:
            exp.save_experience(exp._last_experience_entry)
            result["experience_saved"] = True
            result["experience_id"] = getattr(exp._last_experience_entry, "experience_id", "")
        else:
            result["experience_saved"] = False
            result["experience_save_skipped_reason"] = args.experience_save_mode
        result.update(
            {
                "trial_id": args.trial_id,
                "seed": args.seed,
                "method": args.method,
                "memory_policy": memory_policy,
                "experience_save_mode": args.experience_save_mode,
                "strategy_family": args.strategy_family,
            }
        )
        _write_json(args.save, result)
        _write_json(args.save_plan, exp.recovery_plan or {"steps": []})
        print(f"saved result: {args.save}")
        print(f"saved plan: {args.save_plan}")
        return 0
    finally:
        exp.close()


if __name__ == "__main__":
    raise SystemExit(main())
