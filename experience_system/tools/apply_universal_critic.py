"""Apply universal rule critic to an experience library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, apply_critic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach universal critic_result to every experience entry.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--min-object-lift", type=float, default=0.05)
    parser.add_argument("--max-place-xy-error", type=float, default=0.05)
    parser.add_argument("--max-place-z-error", type=float, default=0.08)
    parser.add_argument("--max-dual-arm-height-mismatch", type=float, default=0.02)
    parser.add_argument("--high-sim-real-gap", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = {
        "min_object_lift": args.min_object_lift,
        "max_place_xy_error": args.max_place_xy_error,
        "max_place_z_error": args.max_place_z_error,
        "max_dual_arm_height_mismatch": args.max_dual_arm_height_mismatch,
        "high_sim_real_gap": args.high_sim_real_gap,
    }
    library = ExperienceLibrary.load(args.input)
    summary = {"pass": 0, "warn": 0, "block": 0, "unknown": 0}
    entries = []
    for entry in library.entries:
        apply_critic(entry, thresholds=thresholds)
        status = entry.critic_result.overall_status or "unknown"
        summary[status] = summary.get(status, 0) + 1
        entries.append({
            "experience_id": entry.experience_id,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "source": entry.source,
            "success": bool(entry.result.get("success", False)),
            "failure_type": entry.failure_taxonomy.get("failure_type", ""),
            "raw_failure_type": entry.failure_taxonomy.get("raw_failure_type", ""),
            "standard_failure_type": entry.failure_taxonomy.get("standard_failure_type", ""),
            "critic_status": status,
            "critic_risk_score": entry.critic_result.critic_risk_score,
            "rule_flags": [flag.get("rule") for flag in entry.critic_result.rule_flags],
        })

    library.save(args.output)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "entry_count": len(library.entries),
        "summary": summary,
        "entries": entries,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"entry_count": len(library.entries), "summary": summary, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
