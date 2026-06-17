"""Dry-run a validated_robot_plan without calling robot hardware."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import DryRunSkillExecutor, default_r1pro_skill_registry, execute_validated_robot_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and dry-run dispatch a validated_robot_plan_v1.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--allow-skill", action="append", default=[], help="additional skill name allowed by the dry-run registry")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    registry = default_r1pro_skill_registry()
    registry.update(str(item) for item in args.allow_skill if str(item))
    report = execute_validated_robot_plan(
        plan,
        DryRunSkillExecutor(registry),
        mode="dry_run",
        stop_on_failure=args.stop_on_failure,
    ).to_dict()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "plan_id": report["plan_id"],
        "success": report["success"],
        "status": report["status"],
        "step_count": len(report["step_reports"]),
        "unsupported_actions": report["unsupported_actions"],
        "save": str(args.save),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
