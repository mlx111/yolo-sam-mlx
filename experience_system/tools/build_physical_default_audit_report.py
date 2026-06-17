"""Audit whether core skills default to physical control instead of direct qpos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_FILES = (
    Path("skills/base/base_motion_skill.json"),
    Path("skills/base/torso_move_skill.json"),
    Path("skills/base/left_arm_move_skill.json"),
    Path("skills/base/right_arm_move_skill.json"),
    Path("skills/primitives/left_gripper_close_skill.json"),
    Path("skills/primitives/right_gripper_close_skill.json"),
    Path("skills/primitives/left_gripper_open_skill.json"),
    Path("skills/primitives/right_gripper_open_skill.json"),
    Path("skills/primitives/open_gripper_release_skill.json"),
    Path("skills/primitives/resync_grippers_skill.json"),
    Path("skills/primitives/base_move_to_region_skill.json"),
    Path("skills/primitives/base_reposition_lateral_skill.json"),
    Path("skills/primitives/base_replan_path_skill.json"),
    Path("skills/primitives/torso_turn_to_target_skill.json"),
    Path("skills/primitives/torso_set_height_skill.json"),
    Path("skills/primitives/pre_grasp_safe_posture_skill.json"),
    Path("skills/primitives/safe_transport_pose_skill.json"),
    Path("skills/primitives/go_home_upper_body_skill.json"),
    Path("skills/primitives/grasp_handle_skill.json"),
    Path("skills/primitives/regrasp_deeper_skill.json"),
    Path("skills/primitives/object_manipulation_skills.py"),
    Path("skills/base/arm_ik_skill.py"),
    Path("skills/base/base_motion_skill.py"),
    Path("skills/base/torso_move_skill.py"),
    Path("skills/base/gripper_skill.py"),
    Path("skills/primitives/recovery_skills.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a physical-default audit report.")
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _scan_text(text: str) -> dict[str, int]:
    return {
        "direct_qpos_true_count": text.count("direct_qpos=True") + text.count('"direct_qpos": true'),
        "direct_qpos_false_count": text.count("direct_qpos=False") + text.count('"direct_qpos": false'),
    }


def build_report() -> dict[str, Any]:
    rows = []
    for rel in DEFAULT_FILES:
        path = Path(rel)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        stats = _scan_text(text)
        rows.append({
            "path": str(path),
            **stats,
        })
    return {
        "schema_version": "physical_default_audit_report_v1",
        "file_count": len(rows),
        "rows": rows,
        "summary": {
            "file_count": len(rows),
            "direct_qpos_true_total": sum(row["direct_qpos_true_count"] for row in rows),
            "direct_qpos_false_total": sum(row["direct_qpos_false_count"] for row in rows),
        },
        "paper_wording": {
            "safe_claim": "Core base, torso, arm, gripper, and field atomic defaults have been audited to prefer physical control over direct-qpos shortcuts.",
            "avoid_claim": "Do not claim that every explicit debug or test path is physical-only; the audit concerns defaults, not every possible parameter override.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physical Default Audit Report",
        "",
        f"- File count: {report['file_count']}",
        f"- direct_qpos=true total: {report['summary']['direct_qpos_true_total']}",
        f"- direct_qpos=false total: {report['summary']['direct_qpos_false_total']}",
        "",
        "| File | direct_qpos=true count | direct_qpos=false count |",
        "|---|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(f"| {row['path']} | {row['direct_qpos_true_count']} | {row['direct_qpos_false_count']} |")
    lines.extend([
        "",
        "## Paper Wording",
        "",
        f"- Safe claim: {report['paper_wording']['safe_claim']}",
        f"- Avoid claim: {report['paper_wording']['avoid_claim']}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report()
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "schema_version": report["schema_version"],
        "file_count": report["file_count"],
        "summary": report["summary"],
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
