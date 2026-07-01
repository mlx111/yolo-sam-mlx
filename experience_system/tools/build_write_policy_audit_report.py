"""Audit universal-memory write policy decisions on a library."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core.library import ExperienceLibrary
from experience_core.schema import ExperienceEntry
from experience_core.write_policy import should_write_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a write-policy audit report for an experience library.")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--merge-duplicates", action="store_true")
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _entry_role(entry: ExperienceEntry) -> str:
    return str(entry.memory_tags.get("memory_role") or entry.memory_tags.get("memory_type") or "")


def _audit_entries(entries: list[ExperienceEntry], *, merge_duplicates: bool) -> list[dict[str, Any]]:
    previous: list[ExperienceEntry] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        decision = should_write_entry(entry, previous, strict_quality=True, merge_duplicates=merge_duplicates)
        rows.append({
            "experience_id": entry.experience_id,
            "source": entry.source,
            "backend": entry.backend,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "success": bool(entry.result.get("success", entry.result.get("task_success", False))),
            "memory_role": _entry_role(entry),
            "critic_status": entry.critic_result.overall_status,
            "write_score": float(entry.memory_gate.write_score or 0.0),
            "decision": decision,
        })
        if decision.get("decision") in {"write", "merge"}:
            previous.append(entry)
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    library = ExperienceLibrary.load(args.universal_experience_lib)
    rows = _audit_entries(library.entries, merge_duplicates=bool(args.merge_duplicates))
    decision_counts = Counter(str(row["decision"].get("decision") or "") for row in rows)
    reason_counts = Counter(str(row["decision"].get("reason") or "") for row in rows)
    role_counts = Counter(str(row.get("memory_role") or "") for row in rows)
    source_counts = Counter(str(row.get("source") or "") for row in rows)
    preserved_reasons = {
        reason: count
        for reason, count in reason_counts.items()
        if reason.startswith("preserve_")
    }
    merged_metadata_count = 0
    merged_support_count = 0
    for entry in library.entries:
        merged_ids = entry.metadata.get("write_policy_merged_experience_ids") if isinstance(entry.metadata, dict) else []
        if isinstance(merged_ids, list) and merged_ids:
            merged_metadata_count += 1
            merged_support_count += len(merged_ids)
    return {
        "schema_version": "write_policy_audit_report_v1",
        "library_path": str(args.universal_experience_lib),
        "entry_count": len(library.entries),
        "merge_duplicates": bool(args.merge_duplicates),
        "summary": {
            "decision_counts": dict(decision_counts),
            "reason_counts": dict(reason_counts),
            "preserved_reason_counts": preserved_reasons,
            "memory_role_counts": dict(role_counts),
            "source_counts": dict(source_counts),
            "write_count": decision_counts.get("write", 0),
            "skip_count": decision_counts.get("skip", 0),
            "merge_count": decision_counts.get("merge", 0),
            "reject_count": decision_counts.get("reject", 0),
            "merged_representative_count_from_metadata": merged_metadata_count,
            "merged_support_count_from_metadata": merged_support_count,
        },
        "entries": rows,
        "paper_wording": {
            "safe_claim": "The write policy is auditable: each candidate entry receives an explicit write, skip, merge, or reject decision with a reason.",
            "avoid_claim": "Do not claim the write policy is learned or globally optimal; it is an explicit engineering gate.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Write Policy Audit Report",
        "",
        f"- Library: `{report['library_path']}`",
        f"- Entries: {report['entry_count']}",
        f"- Merge duplicates: {report['merge_duplicates']}",
        "",
        "## Summary",
        "",
        f"- Decision counts: `{json.dumps(summary['decision_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Reason counts: `{json.dumps(summary['reason_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Preserved reasons: `{json.dumps(summary['preserved_reason_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Memory roles: `{json.dumps(summary['memory_role_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Paper Wording",
        "",
        f"- Safe claim: {report['paper_wording']['safe_claim']}",
        f"- Avoid claim: {report['paper_wording']['avoid_claim']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "schema_version": report["schema_version"],
        "entry_count": report["entry_count"],
        "summary": report["summary"],
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
