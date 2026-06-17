"""Build a sandbox initial-state JSON from an existing experience entry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, build_sandbox_initial_state, choose_sandbox_state_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sandbox_initial_state from universal experience memory.")
    parser.add_argument("--input", type=Path, required=True, help="Universal experience library JSON")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--experience-id", default="", help="Optional explicit source experience id")
    parser.add_argument("--include-failures", action="store_true", help="do not prefer successful episodes when auto-selecting")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    if args.experience_id:
        entry = next((item for item in library.entries if item.experience_id == args.experience_id), None)
    else:
        entry = choose_sandbox_state_entry(
            library.entries,
            scenario=args.scenario,
            condition=args.condition,
            prefer_success=not args.include_failures,
        )
    if entry is None:
        raise SystemExit(f"No matching experience for scenario={args.scenario} condition={args.condition}")

    state = build_sandbox_initial_state(entry)
    payload = state.to_dict()
    report = {
        "input": str(args.input),
        "scenario": args.scenario,
        "condition": args.condition,
        "selected_experience_id": entry.experience_id,
        "selected_source": entry.source,
        "selected_success": bool(entry.result.get("success")),
        "state_confidence": state.confidence,
        "missing_fields": list(state.missing_fields),
        "object_pose_count": len(state.object_poses),
        "obstacle_pose_count": len(state.obstacle_poses),
        "robot_qpos_count": len(state.robot_qpos),
        "robot_qvel_count": len(state.robot_qvel),
        "gripper_state_count": len(state.gripper_state),
        "contact_state_count": len(state.contact_state),
        "save": str(args.save),
    }
    payload["build_report"] = report
    _write_json(args.save, payload)
    if args.report is not None:
        _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
