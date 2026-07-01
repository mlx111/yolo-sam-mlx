#!/usr/bin/env python3
"""Promote simulation experiences using batch execution evidence (memory_v3_plus)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from experience_system.memory.v3 import (
    MemoryV3Entry,
    MemoryV3Library,
    entry_from_dict,
    canonical_action_signature_from_steps,
    infer_memory_partition,
    build_text_summary,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _normalize_signature(sig: Any) -> str:
    """Normalize executed_plan_signature using the canonical pipeline.

    New-format results may store the signature as a list of action dicts
    (``[{"action": "approach_object"}, {"action": "close_gripper"}, ...]``) or as a JSON string.
    The canonical function expects ``parameters`` as a sub-dict, so trial
    top-level keys (e.g. ``state``) are wrapped before canonicalization.
    """
    if isinstance(sig, str):
        try:
            parsed = json.loads(sig)
        except (TypeError, json.JSONDecodeError):
            return sig
        if isinstance(parsed, list):
            return _canonicalize_steps(parsed)
        return sig
    if isinstance(sig, (list, tuple)):
        return _canonicalize_steps(sig)
    return ""


def _canonicalize_steps(steps: list[Any]) -> str:
    """Canonicalize steps, wrapping trial-style top-level keys into parameters."""
    wrapped: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        if not action:
            continue
        params = step.get("parameters")
        if not isinstance(params, dict):
            params = {}
        # Lift top-level keys that belong in parameters
        for key in ("state", "target_class", "command"):
            if key in step and key not in params:
                params[key] = step[key]
        wrapped.append({"action": action, "parameters": params})
    return canonical_action_signature_from_steps(wrapped)


def _signature_actions(signature: str) -> list[str]:
    try:
        payload = json.loads(signature)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item.get("action", "")) for item in payload if isinstance(item, dict)]


def _lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, left_item in enumerate(left, 1):
        for j, right_item in enumerate(right, 1):
            if left_item == right_item:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(left), len(right))


def _entry_anomaly(entry: MemoryV3Entry) -> str:
    return entry.anomaly.type or entry.condition_id or ""


def _entry_signature(entry: MemoryV3Entry) -> str:
    return entry.plan_signature  # property, derived from skill_sequence


def _trial_results(results_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Walk current batch experiment result trees."""
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(results_dir.glob("*/*/*/trial_*/result.json")):
        rows.append((path, _load_json(path)))
    return rows


def _promotion_eligible(data: dict[str, Any]) -> bool:
    """Check if a trial result qualifies for promotion.

    Complete task success and virtual validation are required.
    """
    return bool(
        data.get("virtual_validation_success") is True
        and (data.get("success") is True or data.get("task_success") is True)
        and data.get("repeated_failure_detected") is not True
    )


def _evidence(path: Path, data: dict[str, Any], match_type: str, overlap: float) -> dict[str, Any]:
    return {
        "trial_id": data.get("trial_id", ""),
        "method": data.get("method", ""),
        "condition_id": data.get("condition_id", ""),
        "scenario_id": data.get("scenario_id", ""),
        "memory_policy": data.get("memory_policy", ""),
        "seed": data.get("seed"),
        "virtual_validation_success": data.get("virtual_validation_success"),
        "task_success": data.get("task_success"),
        "success": data.get("success"),
        "executed_plan_source": data.get("executed_plan_source", ""),
        "executed_plan_signature": _normalize_signature(data.get("executed_plan_signature", "")),
        "keyframes": data.get("keyframes") or [],
        "match_type": match_type,
        "action_overlap": overlap,
        "result_path": str(path),
    }


def _best_trial_for_entry(
    entry: MemoryV3Entry,
    trials: list[tuple[Path, dict[str, Any]]],
    weak_match_threshold: float,
) -> tuple[Path, dict[str, Any], str, float] | None:
    entry_cid = entry.condition_id
    entry_sig = _entry_signature(entry)
    entry_actions = _signature_actions(entry_sig)
    best: tuple[Path, dict[str, Any], str, float] | None = None

    for path, data in trials:
        if not _promotion_eligible(data):
            continue
        # condition_id filter when the entry has one
        trial_cid = data.get("condition_id") or ""
        if entry_cid and trial_cid and trial_cid != entry_cid:
            continue
        trial_sig = _normalize_signature(data.get("executed_plan_signature") or "")
        if not trial_sig:
            continue
        if trial_sig == entry_sig:
            return path, data, "exact_signature", 1.0
        overlap = _lcs_ratio(entry_actions, _signature_actions(trial_sig))
        if overlap >= weak_match_threshold:
            if best is None or overlap > best[3]:
                best = (path, data, "weak_action_overlap", overlap)
    return best


def _promote_entry(entry: MemoryV3Entry, evidence: dict[str, Any], now: str) -> None:
    previous = {
        "validation_status": entry.validation_status,
        "memory_partition": entry.memory_partition,
        "source": entry.source,
    }
    entry.validation_status = "simulation_validated"
    entry.validation_source = "batch_experiment"
    entry.validation_evidence = evidence
    if evidence.get("keyframes"):
        entry.keyframes = evidence["keyframes"]
    entry.promotion_history.append(
        {
            "timestamp": now,
            "event": "simulation_to_validated",
            "previous": previous,
            "evidence": evidence,
        }
    )


def _mark_failed_entry(entry: MemoryV3Entry) -> None:
    entry.validation_status = "failed"
    if not entry.validation_source:
        entry.validation_source = "existing_failed_experience"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, *,
                  input_path: Path, results_dir: Path, output_path: Path,
                  rows: list[dict[str, Any]], total: int) -> None:
    promoted = [r for r in rows if r["promoted"]]
    weak = [r for r in promoted if r["match_type"] == "weak_action_overlap"]
    exact = [r for r in promoted if r["match_type"] == "exact_signature"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experience Promotion Report",
        "",
        f"- input: `{input_path}`",
        f"- results: `{results_dir}`",
        f"- output: `{output_path}`",
        f"- entries: `{total}`",
        f"- promoted: `{len(promoted)}`",
        f"- exact signature matches: `{len(exact)}`",
        f"- weak action-overlap matches: `{len(weak)}`",
        "",
        "| experience_id | condition_id | anomaly | from_partition | to_partition | match_type | action_overlap | trial_id |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not row["promoted"]:
            continue
        lines.append(
            "| {experience_id} | {condition_id} | {anomaly_type} | {from_partition} | {to_partition} | {match_type} | {action_overlap:.3f} | {trial_id} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def promote(
    input_path: Path,
    results_dir: Path,
    output_path: Path,
    report_path: Path,
    csv_path: Path,
    weak_match_threshold: float,
) -> int:
    lib = MemoryV3Library.load(input_path)
    trials = _trial_results(results_dir)
    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    promoted_ids: set[str] = set()

    for entry in lib:
        partition = infer_memory_partition(entry)
        row: dict[str, Any] = {
            "experience_id": entry.experience_id,
            "condition_id": entry.condition_id,
            "anomaly_type": _entry_anomaly(entry),
            "from_partition": partition,
            "to_partition": partition,
            "promoted": False,
            "match_type": "",
            "action_overlap": 0.0,
            "trial_id": "",
            "result_path": "",
        }

        entry_success = bool(getattr(entry.result, "success", False) or getattr(entry.result, "task_success", False))
        if partition == "failed_memory" or not entry_success:
            from_partition = partition
            _mark_failed_entry(entry)
            row.update(
                {
                    "from_partition": from_partition,
                    "to_partition": infer_memory_partition(entry),
                }
            )
        elif partition == "simulation_memory" and entry_success:
            match = _best_trial_for_entry(entry, trials, weak_match_threshold)
            if match is not None:
                path, data, match_type, overlap = match
                ev = _evidence(path, data, match_type, overlap)
                _promote_entry(entry, ev, now)
                promoted_ids.add(entry.experience_id)
                row.update(
                    {
                        "to_partition": infer_memory_partition(entry),
                        "promoted": True,
                        "match_type": match_type,
                        "action_overlap": overlap,
                        "trial_id": data.get("trial_id", ""),
                        "result_path": str(path),
                    }
                )
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lib.save(output_path)
    _write_csv(csv_path, rows)
    _write_report(
        report_path,
        input_path=input_path,
        results_dir=results_dir,
        output_path=output_path,
        rows=rows,
        total=len(lib),
    )
    print(f"Loaded entries: {len(lib)}")
    print(f"Promoted entries: {len(promoted_ids)}")
    print(f"Wrote promoted snapshot: {output_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote table: {csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote simulation experiences to validated_memory using batch results (memory_v3_plus)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input experience snapshot JSON (memory_v3_plus format)")
    parser.add_argument("--results", required=True, type=Path, help="Batch results directory")
    parser.add_argument("--output", required=True, type=Path, help="Output promoted snapshot JSON")
    parser.add_argument("--report", type=Path, help="Markdown promotion report")
    parser.add_argument("--csv", type=Path, help="CSV promotion table")
    parser.add_argument("--weak-match-threshold", type=float, default=0.8)
    args = parser.parse_args()

    report = args.report or args.output.with_suffix(".promotion_report.md")
    csv_path = args.csv or args.output.with_suffix(".promotion_table.csv")
    return promote(
        input_path=args.input,
        results_dir=args.results,
        output_path=args.output,
        report_path=report,
        csv_path=csv_path,
        weak_match_threshold=args.weak_match_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
