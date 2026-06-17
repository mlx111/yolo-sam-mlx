"""Policy risk-transfer calibration from universal experience memory."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import ExperienceEntry, utc_now


DEFAULT_POLICY_RISK_CALIBRATION = {
    "schema_version": "policy_risk_calibration_v1",
    "default_weights": {
        "gap_condition_mismatch_scale": 0.30,
        "failure_condition_mismatch_scale": 0.50,
        "gap_task_stage_mismatch_scale": 0.60,
        "gap_object_class_mismatch_scale": 0.50,
        "min_nonzero_risk_transfer": 0.05,
    },
    "groups": [],
}


def _success(entry: ExperienceEntry) -> bool:
    return bool(entry.result.get("success", entry.result.get("recovery_success", False)))


def _outcome_type(entry: ExperienceEntry) -> str:
    return str(entry.sim_real_gap.outcome_gap.get("type") or "")


def _group_key(entry: ExperienceEntry) -> tuple[str, str, str, str, str]:
    return (
        entry.robot.robot_type,
        entry.scenario_id,
        entry.condition_id,
        str(entry.task.get("stage") or ""),
        entry.object_state.object_class,
    )


def _rates(entries: list[ExperienceEntry]) -> dict[str, Any]:
    total = len(entries)
    failures = sum(1 for entry in entries if not _success(entry))
    sim_success_real_fail = sum(1 for entry in entries if _outcome_type(entry) == "sim_success_real_fail")
    gap_count = sum(1 for entry in entries if _outcome_type(entry))
    confidence = min(total, 10) / 10.0
    return {
        "entry_count": total,
        "failure_count": failures,
        "gap_count": gap_count,
        "sim_success_real_fail_count": sim_success_real_fail,
        "failure_rate": round(failures / total, 4) if total else 0.0,
        "sim_success_real_fail_rate": round(sim_success_real_fail / total, 4) if total else 0.0,
        "evidence_confidence": round(confidence, 4),
    }


def _condition_mismatch_scale(rates: dict[str, Any]) -> float:
    # More evidence for condition-specific sim-real failures means cross-condition transfer should be weaker.
    evidence_confidence = float(rates.get("evidence_confidence", 0.0))
    gap_rate = float(rates.get("sim_success_real_fail_rate", 0.0))
    return round(max(0.12, min(0.50, 0.35 - 0.18 * gap_rate * evidence_confidence)), 4)


def build_policy_risk_calibration(entries: list[ExperienceEntry]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], list[ExperienceEntry]] = defaultdict(list)
    for entry in entries:
        if not entry.scenario_id or not entry.condition_id:
            continue
        if not entry.skill_sequence:
            continue
        if not _outcome_type(entry) and _success(entry):
            continue
        groups[_group_key(entry)].append(entry)

    group_payloads = []
    for key, group_entries in sorted(groups.items(), key=lambda item: item[0]):
        rates = _rates(group_entries)
        group_payloads.append({
            "robot_type": key[0],
            "scenario_id": key[1],
            "condition_id": key[2],
            "task_stage": key[3],
            "object_class": key[4],
            **rates,
            "weights": {
                "gap_condition_mismatch_scale": _condition_mismatch_scale(rates),
            },
            "evidence_ids": [entry.experience_id for entry in group_entries],
        })

    return {
        **DEFAULT_POLICY_RISK_CALIBRATION,
        "created_at": utc_now(),
        "group_count": len(group_payloads),
        "groups": group_payloads,
    }


def load_policy_risk_calibration(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_POLICY_RISK_CALIBRATION)
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"policy calibration not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"policy calibration must be a JSON object: {source}")
    defaults = dict(DEFAULT_POLICY_RISK_CALIBRATION["default_weights"])
    defaults.update(payload.get("default_weights") or {})
    return {**payload, "default_weights": defaults}


def find_policy_group(calibration: dict[str, Any] | None, entry: ExperienceEntry) -> dict[str, Any]:
    if not isinstance(calibration, dict):
        return {}
    groups = calibration.get("groups")
    if not isinstance(groups, list):
        return {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        if str(group.get("robot_type") or "") != entry.robot.robot_type:
            continue
        if str(group.get("scenario_id") or "") != entry.scenario_id:
            continue
        if str(group.get("condition_id") or "") != entry.condition_id:
            continue
        task_stage = str(entry.task.get("stage") or "")
        if str(group.get("task_stage") or "") != task_stage:
            continue
        if str(group.get("object_class") or "") != entry.object_state.object_class:
            continue
        return group
    return {}


def policy_weight(calibration: dict[str, Any] | None, group: dict[str, Any], name: str, default: float) -> float:
    if isinstance(group.get("weights"), dict) and group["weights"].get(name) is not None:
        return float(group["weights"][name])
    if isinstance(calibration, dict) and isinstance(calibration.get("default_weights"), dict):
        return float(calibration["default_weights"].get(name, default))
    return float(default)
