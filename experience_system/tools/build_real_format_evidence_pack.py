"""Build an evidence pack for real-format and pseudo-real memory support."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real-format, pseudo-real, sim-real gap, and calibration evidence.")
    parser.add_argument("--input", type=Path, required=True, help="universal experience library JSON")
    parser.add_argument("--sandbox-report", type=Path, action="append", default=[], help="optional policy/sandbox rollout report JSON")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _counter_dict(counter: Counter[str], *, top_k: int | None = None) -> dict[str, int]:
    items = counter.most_common(top_k) if top_k else counter.most_common()
    return {key: value for key, value in items}


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scenario_condition(entry: Any) -> str:
    return f"{entry.scenario_id}/{entry.condition_id}"


def _has_nonzero_vector(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return any(abs(_float(item)) > 0.0 for item in value)


def _evidence_ref(entry: Any) -> dict[str, Any]:
    return {
        "experience_id": entry.experience_id,
        "source": entry.source,
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "object_class": entry.object_state.object_class,
        "success": bool(entry.result.get("success", False)),
        "memory_role": str(entry.memory_tags.get("memory_role") or ""),
        "pair_id": str(entry.sim_real_pair.get("pair_id") or ""),
        "gap_id": entry.sim_real_gap.gap_id,
        "gap_type": str(entry.sim_real_gap.outcome_gap.get("type") or ""),
        "gap_score": round(float(entry.sim_real_gap.gap_score or 0.0), 4),
        "calibration_id": entry.sandbox_calibration.calibration_id,
        "object_pose_bias": list(entry.sandbox_calibration.object_pose_bias or []),
        "calibration_confidence": round(float(entry.sandbox_calibration.calibration_confidence or 0.0), 4),
        "real_episode_ref": dict(entry.real_episode_ref or {}),
    }


def build_library_evidence(library: ExperienceLibrary, *, top_k: int) -> dict[str, Any]:
    source = Counter(entry.source or "" for entry in library.entries)
    backend = Counter(entry.backend or "" for entry in library.entries)
    validation = Counter(entry.validation_status or "" for entry in library.entries)
    memory_partition = Counter(entry.memory_partition or "" for entry in library.entries)
    memory_role = Counter(str(entry.memory_tags.get("memory_role") or "") for entry in library.entries)
    scenario_condition = Counter(_scenario_condition(entry) for entry in library.entries)

    real_entries = [entry for entry in library.entries if entry.source == "real"]
    pseudo_real_entries = [entry for entry in library.entries if entry.source == "pseudo_real"]
    sim_entries = [entry for entry in library.entries if entry.source in {"simulation", "sim"}]
    real_format_entries = [
        entry
        for entry in library.entries
        if entry.source in {"real", "pseudo_real"}
        or entry.validation_status in {"real_executed", "real_validated", "pseudo_real_executed"}
        or bool(entry.real_episode_ref)
    ]
    paired_entries = [
        entry
        for entry in library.entries
        if entry.sim_real_pair.get("validation_status") == "paired" and entry.sim_real_pair.get("pair_id")
    ]
    gap_entries = [entry for entry in library.entries if entry.sim_real_gap.gap_id]
    sensor_gap_entries = [entry for entry in gap_entries if str(entry.sim_real_gap.evidence.get("method") or "") == "sensor_derived_gap_v1"]
    calibration_entries = [entry for entry in library.entries if entry.sandbox_calibration.calibration_id]
    calibration_ids_with_pose_bias = {
        entry.sandbox_calibration.calibration_id
        for entry in calibration_entries
        if entry.sandbox_calibration.calibration_id and _has_nonzero_vector(entry.sandbox_calibration.object_pose_bias)
    }
    pair_ids = {str(entry.sim_real_pair.get("pair_id")) for entry in paired_entries if entry.sim_real_pair.get("pair_id")}
    gap_ids = {entry.sim_real_gap.gap_id for entry in gap_entries if entry.sim_real_gap.gap_id}
    calibration_ids = {
        entry.sandbox_calibration.calibration_id
        for entry in calibration_entries
        if entry.sandbox_calibration.calibration_id
    }
    gap_scores = [float(entry.sim_real_gap.gap_score or 0.0) for entry in gap_entries]
    gap_uncertainties = [float(entry.sim_real_gap.uncertainty or 0.0) for entry in gap_entries]
    calibration_confidences = [float(entry.sandbox_calibration.calibration_confidence or 0.0) for entry in calibration_entries]
    outcome_gap_type = Counter(str(entry.sim_real_gap.outcome_gap.get("type") or "") for entry in gap_entries)
    sensor_evidence_entries = [entry for entry in library.entries if getattr(entry, "sensor_evidence", None) and entry.sensor_evidence.modalities]
    rgb_entries = [entry for entry in library.entries if "rgb" in set(entry.sensor_evidence.modalities or []) or "rgbd" in set(entry.sensor_evidence.modalities or [])]
    rgbd_entries = [entry for entry in library.entries if "rgbd" in set(entry.sensor_evidence.modalities or [])]
    lidar_entries = [entry for entry in library.entries if "lidar" in set(entry.sensor_evidence.modalities or [])]
    wrist_force_entries = [entry for entry in library.entries if "wrist_force" in set(entry.sensor_evidence.modalities or [])]
    sensor_modality = Counter()
    max_wrist_force_norms = []
    lidar_ray_counts = []
    for entry in sensor_evidence_entries:
        for modality in entry.sensor_evidence.modalities:
            sensor_modality[str(modality)] += 1
        summary = entry.sensor_evidence.summary or {}
        if summary.get("max_wrist_force_norm") is not None:
            max_wrist_force_norms.append(_float(summary.get("max_wrist_force_norm")))
        if summary.get("lidar_ray_count") is not None:
            lidar_ray_counts.append(_float(summary.get("lidar_ray_count")))

    source_gap_ids = Counter()
    for entry in calibration_entries:
        for gap_id in entry.sandbox_calibration.source_gap_ids:
            if gap_id:
                source_gap_ids[str(gap_id)] += 1

    return {
        "entry_count": len(library.entries),
        "real_format_load_ok": True,
        "real_format_schema_valid": True,
        "real_format_entry_count": len(real_format_entries),
        "real_entry_count": len(real_entries),
        "pseudo_real_entry_count": len(pseudo_real_entries),
        "sim_entry_count": len(sim_entries),
        "paired_gap_count": len(pair_ids),
        "sim_real_gap_count": len(gap_ids),
        "sensor_gap_entry_count": len(sensor_gap_entries),
        "calibration_id_count": len(calibration_ids),
        "calibration_with_object_pose_bias_count": len(calibration_ids_with_pose_bias),
        "sensor_evidence_entry_count": len(sensor_evidence_entries),
        "rgb_evidence_entry_count": len(rgb_entries),
        "rgbd_evidence_entry_count": len(rgbd_entries),
        "lidar_evidence_entry_count": len(lidar_entries),
        "wrist_force_evidence_entry_count": len(wrist_force_entries),
        "sensor_modality_distribution": _counter_dict(sensor_modality),
        "max_wrist_force_norm_avg": _avg(max_wrist_force_norms),
        "lidar_ray_count_avg": _avg(lidar_ray_counts),
        "source_distribution": _counter_dict(source),
        "backend_distribution": _counter_dict(backend),
        "validation_status_distribution": _counter_dict(validation),
        "memory_partition_distribution": _counter_dict(memory_partition),
        "memory_role_distribution": _counter_dict(memory_role),
        "scenario_condition_distribution": _counter_dict(scenario_condition),
        "gap_type_distribution": _counter_dict(outcome_gap_type),
        "score_summary": {
            "gap_score_avg": _avg(gap_scores),
            "gap_uncertainty_avg": _avg(gap_uncertainties),
            "calibration_confidence_avg": _avg(calibration_confidences),
        },
        "paired_gap_ids": sorted(pair_ids),
        "sim_real_gap_ids": sorted(gap_ids),
        "calibration_ids": sorted(calibration_ids),
        "calibration_source_gap_reuse_count": _counter_dict(source_gap_ids),
        "sample_real_format_entries": [_evidence_ref(entry) for entry in real_format_entries[:top_k]],
        "sample_gap_entries": [_evidence_ref(entry) for entry in gap_entries[:top_k]],
        "sample_sensor_gap_entries": [_evidence_ref(entry) for entry in sensor_gap_entries[:top_k]],
        "sample_calibration_entries": [_evidence_ref(entry) for entry in calibration_entries[:top_k]],
    }


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object report: {path}")
    return payload


def _candidate_sandbox_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in report.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        sandbox = candidate.get("sandbox")
        if isinstance(sandbox, dict):
            item = dict(sandbox)
            item.setdefault("candidate_id", candidate.get("candidate_id", ""))
            items.append(item)
    for sandbox in report.get("sandbox_reports") or []:
        if isinstance(sandbox, dict):
            items.append(dict(sandbox))
    return items


def _calibration_applied(sandbox: dict[str, Any]) -> bool:
    if sandbox.get("calibration_applied") is True:
        return True
    effect = sandbox.get("sandbox_calibration_effect")
    if isinstance(effect, dict) and effect.get("applied") is True:
        return True
    if sandbox.get("calibration_risk_penalty", 0.0):
        return True
    return False


def build_sandbox_application_evidence(paths: list[Path]) -> dict[str, Any]:
    reports = []
    candidate_items = []
    enabled_count = 0
    applied_count = 0
    calibration_ids = Counter()
    risk_penalties: list[float] = []
    object_start_deltas: list[float] = []

    for path in paths:
        report = _load_report(path)
        enabled = bool(report.get("sandbox_calibration_enabled"))
        enabled_count += int(enabled)
        sandbox_items = _candidate_sandbox_items(report)
        report_applied = 0
        for item in sandbox_items:
            applied = _calibration_applied(item)
            applied_count += int(applied)
            report_applied += int(applied)
            penalty = _float(item.get("calibration_risk_penalty"))
            if penalty:
                risk_penalties.append(penalty)
            effect = item.get("sandbox_calibration_effect")
            if isinstance(effect, dict):
                cal_id = str(effect.get("calibration_id") or "")
                if cal_id:
                    calibration_ids[cal_id] += 1
                before = effect.get("object_position_before")
                after = effect.get("object_position_after")
                if isinstance(before, list) and isinstance(after, list):
                    n = min(len(before), len(after), 3)
                    delta = sum(abs(_float(after[i]) - _float(before[i])) for i in range(n))
                    if delta:
                        object_start_deltas.append(round(delta, 6))
        reports.append({
            "path": str(path),
            "sandbox_calibration_enabled": enabled,
            "candidate_sandbox_count": len(sandbox_items),
            "candidate_calibration_applied_count": report_applied,
            "top_level_calibration_id": str((report.get("sandbox_calibration") or {}).get("calibration_id") or ""),
        })

    return {
        "sandbox_report_count": len(paths),
        "sandbox_calibration_enabled_report_count": enabled_count,
        "sandbox_calibration_application_count": applied_count,
        "applied_calibration_id_distribution": _counter_dict(calibration_ids),
        "calibration_risk_penalty_avg": _avg(risk_penalties),
        "object_start_delta_l1_avg": _avg(object_start_deltas),
        "reports": reports,
    }


def safe_paper_wording(library_evidence: dict[str, Any], sandbox_evidence: dict[str, Any]) -> dict[str, str]:
    if int(library_evidence.get("real_entry_count") or 0) > 0:
        real_clause = "real-format and real-source episodes"
    else:
        real_clause = "real-format episode import"
    if int(sandbox_evidence.get("sandbox_calibration_application_count") or 0) > 0:
        calibration_clause = "and shows calibration consumed by sandbox rollout reports"
    else:
        calibration_clause = "and stores gap-derived sandbox calibration records"
    return {
        "safe_claim": (
            f"The implementation supports {real_clause} and uses pseudo-real evidence "
            f"to exercise sim-real pairing, gap extraction, and sandbox calibration, {calibration_clause}."
        ),
        "avoid_claim": "Do not claim real-robot validation or real-robot success-rate improvement unless true real-source execution reports are added.",
    }


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    library_evidence = build_library_evidence(library, top_k=args.top_k)
    sandbox_evidence = build_sandbox_application_evidence(list(args.sandbox_report or []))
    report = {
        "schema_version": "real_format_evidence_pack_v1",
        "input": str(args.input),
        "sandbox_reports": [str(path) for path in args.sandbox_report or []],
        "library_evidence": library_evidence,
        "sandbox_application_evidence": sandbox_evidence,
        "core_metrics": {
            "real_format_entry_count": library_evidence["real_format_entry_count"],
            "pseudo_real_entry_count": library_evidence["pseudo_real_entry_count"],
            "sim_entry_count": library_evidence["sim_entry_count"],
            "paired_gap_count": library_evidence["paired_gap_count"],
            "sim_real_gap_count": library_evidence["sim_real_gap_count"],
            "sensor_gap_entry_count": library_evidence["sensor_gap_entry_count"],
            "calibration_id_count": library_evidence["calibration_id_count"],
            "calibration_with_object_pose_bias_count": library_evidence["calibration_with_object_pose_bias_count"],
            "sandbox_calibration_application_count": sandbox_evidence["sandbox_calibration_application_count"],
        },
        "paper_wording": safe_paper_wording(library_evidence, sandbox_evidence),
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "save": str(args.save),
        "core_metrics": report["core_metrics"],
        "safe_claim": report["paper_wording"]["safe_claim"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
