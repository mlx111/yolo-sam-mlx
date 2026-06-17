"""Build a field-atomic memory and parameter-prior report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import (
    ExperienceEntry,
    ExperienceLibrary,
    build_field_atomic_parameter_priors,
    build_field_atomic_planner_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a field_atomic memory and parameter prior report.")
    parser.add_argument(
        "--universal-experience-lib",
        type=Path,
        default=Path("results/memory/universal_pipeline_calibration_v1/universal_experience_library.json"),
    )
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--condition-id", default="")
    parser.add_argument("--action", default="")
    parser.add_argument("--goal", default="")
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_library(path: Path) -> ExperienceLibrary:
    return ExperienceLibrary.load(path)


def _field_atomic_entries(entries: list[ExperienceEntry]) -> list[ExperienceEntry]:
    return [
        entry
        for entry in entries
        if str(entry.memory_tags.get("memory_type") or "") == "field_atomic_experience"
        or str(entry.memory_tags.get("memory_role") or "").startswith("field_atomic_")
        or bool(entry.metadata.get("field_atomic"))
    ]


def _summary(entries: list[ExperienceEntry]) -> dict[str, Any]:
    by_role: dict[str, int] = {}
    by_action: dict[str, int] = {}
    for entry in entries:
        role = str(entry.memory_tags.get("memory_role") or "")
        by_role[role] = by_role.get(role, 0) + 1
        feedback = entry.execution_feedback if isinstance(entry.execution_feedback, dict) else {}
        action = str(feedback.get("field_atomic_action") or entry.metadata.get("field_atomic_action") or "")
        if action:
            by_action[action] = by_action.get(action, 0) + 1
    return {
        "field_atomic_entry_count": len(entries),
        "field_atomic_success_count": sum(1 for entry in entries if str(entry.memory_tags.get("memory_role") or "") == "field_atomic_success"),
        "field_atomic_failure_count": sum(1 for entry in entries if str(entry.memory_tags.get("memory_role") or "") == "field_atomic_failure"),
        "memory_role_distribution": by_role,
        "action_distribution": by_action,
    }


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Field Atomic Memory Report",
        "",
        f"- Input: `{report['input']}`",
        f"- Entry count: {report['summary']['field_atomic_entry_count']}",
        f"- Success count: {report['summary']['field_atomic_success_count']}",
        f"- Failure count: {report['summary']['field_atomic_failure_count']}",
        "",
        "## Action Distribution",
    ]
    for action, count in sorted(report["summary"]["action_distribution"].items()):
        priors = report["parameter_priors"]["by_action"].get(action, {})
        lines.extend([
            f"- `{action}`: {count}",
            f"  - success_count: {priors.get('success_count', 0)}",
            f"  - failure_count: {priors.get('failure_count', 0)}",
            f"  - success_rate: {priors.get('success_rate', 0.0)}",
        ])
    lines.extend([
        "",
        "## Planner Input Preview",
        "```json",
        json.dumps(report["planner_input_preview"], indent=2, ensure_ascii=False),
        "```",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    library = _load_library(args.universal_experience_lib)
    entries = _field_atomic_entries(library.entries)
    priors = build_field_atomic_parameter_priors(
        entries,
        scenario_id=args.scenario_id,
        condition_id=args.condition_id,
        action=args.action,
    )
    planner_input = build_field_atomic_planner_input(
        entries,
        scenario_id=args.scenario_id,
        condition_id=args.condition_id,
        goal=args.goal,
    )
    report = {
        "schema_version": "field_atomic_memory_report_v1",
        "input": str(args.universal_experience_lib),
        "filters": {
            "scenario_id": args.scenario_id,
            "condition_id": args.condition_id,
            "action": args.action,
            "goal": args.goal,
        },
        "summary": _summary(entries),
        "parameter_priors": priors,
        "planner_input_preview": planner_input,
        "paper_wording": {
            "safe_claim": "The system stores successful and failed field atomic experiences and uses them as explicit parameter priors for later planning.",
            "avoid_claim": "Do not claim this proves improved real-robot success without a real-robot ablation.",
        },
    }
    _write_json(args.save_json, report)
    _write_md(args.save_md, _render_md(report))
    print(json.dumps({
        "field_atomic_entry_count": report["summary"]["field_atomic_entry_count"],
        "field_atomic_success_count": report["summary"]["field_atomic_success_count"],
        "field_atomic_failure_count": report["summary"]["field_atomic_failure_count"],
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
