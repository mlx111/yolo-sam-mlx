"""Summarize field-atomic execution traces for debugging and paper evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build action-level trace/error summaries from field_atomic reports and experience libraries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Input JSON report/library. Can be repeated.",
    )
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_final_error(raw_result: dict[str, Any]) -> float | None:
    for key in ("final_error", "left_error", "right_error", "error"):
        value = _as_float(raw_result.get(key))
        if value is not None:
            return value
    arm_result = _as_dict(raw_result.get("arm_result"))
    for key in ("final_error", "left_error", "right_error", "error"):
        value = _as_float(arm_result.get(key))
        if value is not None:
            return value
    return None


def _extract_control_mode(raw_result: dict[str, Any]) -> str:
    for key in ("control_mode", "drive_mode", "execution_type"):
        value = raw_result.get(key)
        if isinstance(value, str) and value:
            return value
    arm_result = _as_dict(raw_result.get("arm_result"))
    value = arm_result.get("control_mode")
    return value if isinstance(value, str) else ""


def _extract_gripper_value(raw_result: dict[str, Any], parameters: dict[str, Any]) -> float | None:
    value = _as_float(raw_result.get("value"))
    if value is not None:
        return value
    value = _as_float(parameters.get("gripper_value"))
    if value is not None:
        return value
    if "state" in parameters:
        try:
            return 0.025 if int(parameters["state"]) == 1 else 0.0
        except Exception:
            return None
    return None


def _action_kind(action: str) -> str:
    if "arm_move" in action:
        return "arm"
    if action.startswith("base_move"):
        return "base"
    if action.startswith("torso"):
        return "torso"
    if "gripper" in action:
        return "gripper"
    if "camera" in action or "lidar" in action:
        return "sensor"
    return "other"


def _iter_from_field_atomic_report(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action_items = payload.get("actions")
    if not isinstance(action_items, list):
        action_items = payload.get("step_reports")
    if not isinstance(action_items, list):
        return rows
    for item in action_items:
        item_dict = _as_dict(item)
        action = str(item_dict.get("action") or "")
        parameters = _as_dict(item_dict.get("parameters"))
        raw_result = _as_dict(item_dict.get("raw_result"))
        rows.append(
            {
                "source_file": str(path),
                "source_kind": str(payload.get("schema_version") or "field_atomic_report"),
                "index": item_dict.get("index"),
                "action": action,
                "success": bool(item_dict.get("success")),
                "status": str(item_dict.get("status") or ""),
                "message": str(item_dict.get("message") or ""),
                "parameters": parameters,
                "raw_result": raw_result,
            }
        )
    return rows


def _iter_from_library(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return rows
    for entry in entries:
        entry_dict = _as_dict(entry)
        metadata = _as_dict(entry_dict.get("metadata"))
        feedback = _as_dict(entry_dict.get("execution_feedback"))
        action = str(
            feedback.get("field_atomic_action")
            or metadata.get("field_atomic_action")
            or _as_dict(entry_dict.get("result")).get("field_atomic_action")
            or ""
        )
        if not action:
            continue
        parameters = _as_dict(feedback.get("field_atomic_parameters"))
        raw_result = _as_dict(feedback.get("field_atomic_result"))
        result = _as_dict(entry_dict.get("result"))
        rows.append(
            {
                "source_file": str(path),
                "source_kind": str(payload.get("schema_version") or "experience_library"),
                "experience_id": entry_dict.get("experience_id"),
                "action": action,
                "success": bool(result.get("success", result.get("task_success", False))),
                "status": str(result.get("field_atomic_status") or ""),
                "message": "",
                "parameters": parameters,
                "raw_result": raw_result,
            }
        )
    return rows


def _extract_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        rows.extend(_iter_from_field_atomic_report(path, payload))
        rows.extend(_iter_from_library(path, payload))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        action = str(row.get("action") or "")
        parameters = _as_dict(row.get("parameters"))
        raw_result = _as_dict(row.get("raw_result"))
        direct_qpos = parameters.get("direct_qpos")
        normalized.append(
            {
                **row,
                "action_kind": _action_kind(action),
                "final_error": _extract_final_error(raw_result),
                "control_mode": _extract_control_mode(raw_result),
                "direct_qpos_used": bool(direct_qpos) if isinstance(direct_qpos, bool) else False,
                "gripper_command": _extract_gripper_value(raw_result, parameters),
                "target_xyz": [
                    parameters.get("target_x"),
                    parameters.get("target_y"),
                    parameters.get("target_z"),
                ]
                if all(key in parameters for key in ("target_x", "target_y", "target_z"))
                else None,
                "target_qpos": parameters.get("target_qpos"),
            }
        )
    return normalized


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "max": round(max(values), 6),
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows = _extract_rows(paths)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[str(row["action"])].append(row)

    action_summaries: dict[str, dict[str, Any]] = {}
    for action, items in sorted(by_action.items()):
        errors = [value for value in (_as_float(item.get("final_error")) for item in items) if value is not None]
        gripper_values = [
            value for value in (_as_float(item.get("gripper_command")) for item in items) if value is not None
        ]
        action_summaries[action] = {
            "action_kind": _action_kind(action),
            "count": len(items),
            "success_count": sum(1 for item in items if item["success"]),
            "failure_count": sum(1 for item in items if not item["success"]),
            "success_rate": round(sum(1 for item in items if item["success"]) / len(items), 6) if items else 0.0,
            "final_error": _metric_summary(errors),
            "gripper_command": _metric_summary(gripper_values),
            "direct_qpos_true_count": sum(1 for item in items if item["direct_qpos_used"]),
            "direct_qpos_false_count": sum(1 for item in items if not item["direct_qpos_used"]),
            "control_mode_counts": dict(Counter(str(item.get("control_mode") or "unknown") for item in items)),
            "status_counts": dict(Counter(str(item.get("status") or "unknown") for item in items)),
            "failed_examples": [
                {
                    "source_file": item["source_file"],
                    "experience_id": item.get("experience_id", ""),
                    "status": item.get("status", ""),
                    "message": item.get("message", ""),
                    "parameters": item.get("parameters", {}),
                    "final_error": item.get("final_error"),
                }
                for item in items
                if not item["success"]
            ][:5],
        }

    final_errors = [value for value in (_as_float(row.get("final_error")) for row in rows) if value is not None]
    direct_true = sum(1 for row in rows if row["direct_qpos_used"])
    report = {
        "schema_version": "field_atomic_trace_summary_v1",
        "inputs": [str(path) for path in paths],
        "input_count": len(paths),
        "trace_count": len(rows),
        "summary": {
            "action_count": len(action_summaries),
            "success_count": sum(1 for row in rows if row["success"]),
            "failure_count": sum(1 for row in rows if not row["success"]),
            "action_kind_counts": dict(Counter(str(row["action_kind"]) for row in rows)),
            "direct_qpos_true_count": direct_true,
            "direct_qpos_false_count": len(rows) - direct_true,
            "final_error": _metric_summary(final_errors),
        },
        "by_action": action_summaries,
        "traces": rows,
        "paper_wording": {
            "safe_claim": (
                "Field-atomic executions expose action-level trace summaries, including final tracking error, "
                "gripper command, control mode, and direct-qpos usage for onsite debugging."
            ),
            "avoid_claim": (
                "Do not claim these summaries prove real-robot tracking accuracy; they summarize MuJoCo or stored experience traces only."
            ),
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Field Atomic Trace Summary",
        "",
        "This report aggregates field_atomic execution traces for debugging and paper evidence.",
        "",
        "## Summary",
        "",
        f"- Inputs: {report['input_count']}",
        f"- Traces: {report['trace_count']}",
        f"- Actions: {summary['action_count']}",
        f"- Success / failure: {summary['success_count']} / {summary['failure_count']}",
        f"- Action kinds: `{json.dumps(summary['action_kind_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- direct_qpos true / false: {summary['direct_qpos_true_count']} / {summary['direct_qpos_false_count']}",
        f"- Final error: `{json.dumps(summary['final_error'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## By Action",
        "",
        "| Action | Kind | Count | Success | Failure | Final error | Gripper command | direct_qpos true/false | Control modes |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for action, item in report["by_action"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    action,
                    str(item["action_kind"]),
                    str(item["count"]),
                    str(item["success_count"]),
                    str(item["failure_count"]),
                    f"`{json.dumps(item['final_error'], ensure_ascii=False, sort_keys=True)}`",
                    f"`{json.dumps(item['gripper_command'], ensure_ascii=False, sort_keys=True)}`",
                    f"{item['direct_qpos_true_count']}/{item['direct_qpos_false_count']}",
                    f"`{json.dumps(item['control_mode_counts'], ensure_ascii=False, sort_keys=True)}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paper Wording",
            "",
            f"- Safe claim: {report['paper_wording']['safe_claim']}",
            f"- Avoid claim: {report['paper_wording']['avoid_claim']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.input:
        raise SystemExit("--input is required")
    report = build_report(args.input)
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "save_json": str(args.save_json),
                "save_md": str(args.save_md),
                "trace_count": report["trace_count"],
                "action_count": report["summary"]["action_count"],
                "direct_qpos_true_count": report["summary"]["direct_qpos_true_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
