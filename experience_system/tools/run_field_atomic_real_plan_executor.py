"""Guarded real-robot executor entry for field atomic validated plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import execute_field_atomic_validated_plan_on_robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a field_atomic validated_robot_plan before real robot dispatch.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = execute_field_atomic_validated_plan_on_robot(
        plan,
        adapter=None,
        stop_on_failure=bool(args.stop_on_failure),
    ).to_dict()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "plan_id": report["plan_id"],
        "success": report["success"],
        "status": report["status"],
        "unsupported_actions": report["unsupported_actions"],
        "save": str(args.save),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
