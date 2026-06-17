"""Construct write-policy cases that exercise write/merge/skip/reject decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core.library import ExperienceLibrary
from experience_core.schema import (
    CriticResult,
    ExperienceEntry,
    MemoryGate,
    ObjectState,
    RobotState,
    SkillTraceItem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic write-policy pressure test.")
    parser.add_argument("--save-library", type=Path, required=True)
    parser.add_argument("--save-report", type=Path, required=True)
    return parser.parse_args()


def _skill(name: str = "field_atomic_base_move") -> SkillTraceItem:
    return SkillTraceItem(name=name, primitive_type="field_atomic", phase="test", success=True)


def _entry(
    experience_id: str,
    *,
    success: bool,
    write_score: float,
    memory_role: str = "",
    failure_type: str = "",
    skill_name: str = "common_skill",
    scenario_id: str = "write_policy_pressure",
    condition_id: str = "default",
    object_class: str = "test_object",
    robot_type: str = "mobile_dual_arm",
    backend: str = "mujoco",
    critic_status: str = "pass",
) -> ExperienceEntry:
    return ExperienceEntry(
        experience_id=experience_id,
        source="simulation",
        domain="manipulation",
        backend=backend,
        validation_status="simulation_validated",
        robot=RobotState(robot_id="r1pro_sim", robot_type=robot_type, backend=backend),
        scenario={"scenario_id": scenario_id},
        condition={"condition_id": condition_id},
        task={"name": "write_policy_pressure_test", "stage": "writeback"},
        skill_sequence=[_skill(skill_name)],
        object_state=ObjectState(target_object="target_cube", object_class=object_class),
        result={"success": bool(success), "task_success": bool(success)},
        memory_tags={"memory_type": "write_policy_test", "memory_role": memory_role} if memory_role else {"memory_type": "write_policy_test"},
        memory_gate=MemoryGate(
            anomaly_score=0.0 if success else 1.0,
            failure_score=0.0 if success else 1.0,
            recovery_utility_score=write_score,
            write_score=write_score,
            trigger_events=["pressure_test"],
        ),
        critic_result=CriticResult(overall_status=critic_status, critic_risk_score=0.0 if success else 0.8),
        failure_taxonomy={"failure_type": failure_type} if failure_type else {},
    )


def _bad_entry() -> ExperienceEntry:
    return ExperienceEntry(
        experience_id="case_reject_missing_required_fields",
        source="simulation",
        domain="manipulation",
        backend="",
        scenario={},
        condition={},
        robot=RobotState(robot_id="r1pro_sim", robot_type="", backend=""),
        result={},
        skill_sequence=[],
        object_state=ObjectState(object_class=""),
        memory_gate=MemoryGate(write_score=0.9),
    )


def build_cases() -> list[tuple[str, ExperienceEntry]]:
    return [
        (
            "write_preserve_failure",
            _entry(
                "case_write_preserve_failure",
                success=False,
                write_score=0.95,
                failure_type="grasp_slip",
                skill_name="failed_grasp",
                critic_status="warn",
            ),
        ),
        (
            "write_field_atomic_success",
            _entry(
                "case_write_field_atomic_success",
                success=True,
                write_score=0.35,
                memory_role="field_atomic_success",
                skill_name="base_move_to_pose",
            ),
        ),
        (
            "write_duplicate_seed",
            _entry(
                "case_write_duplicate_seed",
                success=True,
                write_score=0.35,
                skill_name="duplicate_low_risk_skill",
            ),
        ),
        (
            "merge_duplicate_low_risk_success",
            _entry(
                "case_merge_duplicate_low_risk_success",
                success=True,
                write_score=0.35,
                skill_name="duplicate_low_risk_skill",
            ),
        ),
        (
            "skip_low_value_success",
            _entry(
                "case_skip_low_value_success",
                success=True,
                write_score=0.05,
                skill_name="unique_low_value_success",
                condition_id="low_value",
            ),
        ),
        ("reject_missing_required_fields", _bad_entry()),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Write Policy Pressure Test",
        "",
        f"- Case count: {report['case_count']}",
        f"- Stored library entry count: {report['stored_library_entry_count']}",
        f"- Decision counts: `{json.dumps(report['decision_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Reason counts: `{json.dumps(report['reason_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "| Case | Experience | Decision | Reason | Stored/Target |",
        "|---|---|---|---|---|",
    ]
    for item in report["cases"]:
        decision = item["write_policy"]
        lines.append(
            f"| {item['case_id']} | {item['experience_id']} | "
            f"{decision.get('decision', '')} | {decision.get('reason', '')} | "
            f"{decision.get('stored_experience_id') or decision.get('target_experience_id') or ''} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This pressure test verifies that writeback is not a plain append-only log.",
        "Important failures and field-atomic memories are preserved, duplicate low-risk successes can merge, low-value successes can skip, and malformed entries can be rejected.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary()
    cases = []
    decision_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for case_id, entry in build_cases():
        decision = library.add_with_policy(entry, strict_quality=True, merge_duplicates=True)
        decision_name = str(decision.get("decision") or "")
        reason = str(decision.get("reason") or "")
        decision_counts[decision_name] = decision_counts.get(decision_name, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        cases.append({
            "case_id": case_id,
            "experience_id": entry.experience_id,
            "write_policy": decision,
        })
    report = {
        "schema_version": "write_policy_pressure_test_v1",
        "case_count": len(cases),
        "stored_library_entry_count": len(library.entries),
        "decision_counts": decision_counts,
        "reason_counts": reason_counts,
        "cases": cases,
        "expected_decisions_present": {
            name: decision_counts.get(name, 0) > 0
            for name in ("write", "merge", "skip", "reject")
        },
    }
    args.save_library.parent.mkdir(parents=True, exist_ok=True)
    args.save_report.parent.mkdir(parents=True, exist_ok=True)
    library.save(args.save_library)
    args.save_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_report.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "case_count": report["case_count"],
        "stored_library_entry_count": report["stored_library_entry_count"],
        "decision_counts": report["decision_counts"],
        "reason_counts": report["reason_counts"],
        "save_library": str(args.save_library),
        "save_report": str(args.save_report),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
