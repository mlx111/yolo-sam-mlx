"""Consolidate experience-system reports into a paper claim evidence table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_REPORTS = {
    "library_summary": "summary_report.json",
    "real_format": "real_format_evidence_pack.json",
    "visual_g3": "visual_retrieval_ablation_g3_clean.json",
    "visual_g4": "visual_retrieval_ablation_g4_place_occupied.json",
    "calibration_ablation": "sandbox_calibration_ablation_g4_place_occupied.json",
    "lesson_quality": "lesson_quality_report_g4_place_occupied.json",
    "visual_index": "visual_index_report.json",
    "sandbox_g4": "sandbox_rollout_g4_place_occupied.json",
    "sandbox_g3": "sandbox_rollout_g3_clean.json",
    "stage_context": "stage_planner_context_g4.json",
    "safety_stress": "safety_stress_g4_place_occupied.json",
    "harder_safety_stress": "harder_safety_stress_g4_place_occupied.json",
    "writeback_benchmark": "writeback_benchmark_g3_clean.json",
    "memory_type_coverage": "memory_type_coverage_report.json",
    "text_semantic": "text_semantic_memory_report.json",
    "llm_plan_search_real": "llm_plan_candidate_search_g3_clean_real_llm_report.json",
    "llm_plan_search_ablation": "llm_plan_search_ablation_g3_clean_real_llm.json",
    "field_atomic_memory": "field_atomic_memory_report.json",
    "field_atomic_ablation": "field_atomic_memory_ablation.json",
    "field_atomic_trace": "field_atomic_trace_summary.json",
    "rewrite_loop_ablation": "rewrite_loop_ablation_report.json",
    "physical_perturbation": "physical_sandbox_perturbation_report.json",
    "physical_default_audit": "physical_default_audit_report.json",
    "write_policy_audit": "write_policy_audit_report.json",
    "write_policy_pressure": "write_policy_pressure_report.json",
}

DEFAULT_TMP_REPORTS = {
    "stage_policy_context": "/tmp/policy_stage_context_g4.json",
    "stage_context_fallback": "/tmp/stage_planner_context_g4.json",
    "safety_stress_fallback": "/tmp/safety_stress_g4_place_occupied.json",
    "writeback_benchmark_fallback": "/tmp/writeback_benchmark_g3_clean.json",
    "field_atomic_memory_fallback": "/tmp/field_atomic_memory_report.json",
    "field_atomic_ablation_fallback": "/tmp/field_atomic_ablation.json",
    "field_atomic_trace_fallback": "/tmp/field_atomic_trace_summary.json",
    "rewrite_loop_ablation_fallback": "/tmp/rewrite_loop_ablation_report.json",
    "physical_perturbation_fallback": "/tmp/physical_sandbox_perturbation_report.json",
    "physical_default_audit_fallback": "/tmp/physical_default_audit_report.json",
    "write_policy_audit_fallback": "/tmp/write_policy_audit_report.json",
    "write_policy_pressure_fallback": "/tmp/write_policy_pressure_report.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSON/Markdown paper evidence summary from existing reports.")
    parser.add_argument("--report-dir", type=Path, default=Path("results/memory/universal_pipeline_calibration_v1"))
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _get(payload: dict[str, Any] | None, path: str, default: Any = "") -> Any:
    if payload is None:
        return default
    value: Any = payload
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part, default)
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return value


def _metric(metrics: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        metrics[key] = value


def _status(supported: bool, report: dict[str, Any] | None) -> str:
    if report is None:
        return "missing_report"
    return "supported" if supported else "partial"


def _source(path: Path) -> str:
    return str(path)


def _claim(
    *,
    claim: str,
    status: str,
    primary_report: str,
    key_metrics: dict[str, Any],
    safe_wording: str,
    avoid_wording: str,
) -> dict[str, Any]:
    return {
        "claim": claim,
        "status": status,
        "primary_report": primary_report,
        "key_metrics": key_metrics,
        "safe_wording": safe_wording,
        "avoid_wording": avoid_wording,
    }


def _summarize_safety_stress(report: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in (report or {}).get("summary") or []:
        if not isinstance(row, dict):
            continue
        variant = str(row.get("variant_id") or "")
        if variant in {"memory_sandbox_critic", "full_stage_lesson_sandbox", "full"}:
            for key in (
                "risky_candidate_selected_rate",
                "safe_candidate_selected_rate",
                "risky_warn_or_block_rate",
                "critic_warn_rate_avg",
                "score_margin_selected_vs_best_risky",
            ):
                if key in row:
                    out[f"{variant}.{key}"] = row[key]
    if not out and report:
        out["summary_row_count"] = len(report.get("summary") or [])
    return out


def _sandbox_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = {
        "candidate_count": report.get("candidate_count"),
        "selected_before_sandbox": report.get("selected_before_sandbox"),
        "selected_after_sandbox": report.get("selected_after_sandbox"),
        "candidate_changed_by_sandbox": report.get("candidate_changed_by_sandbox"),
    }
    candidates = report.get("candidates") or []
    critic_statuses: dict[str, int] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        sandbox = item.get("sandbox") or {}
        status = str(sandbox.get("critic_status") or "")
        if status:
            critic_statuses[status] = critic_statuses.get(status, 0) + 1
    if critic_statuses:
        metrics["critic_status_counts"] = critic_statuses
    return metrics


def _llm_plan_search_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    candidates = report.get("candidate_reports") or []
    status_counts: dict[str, int] = {}
    semantic_status_counts: dict[str, int] = {}
    critic_status_counts: dict[str, int] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        status = str(item.get("search_status") or "")
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        semantic_status = str((item.get("plan_semantic_validation") or {}).get("status") or "")
        if semantic_status:
            semantic_status_counts[semantic_status] = semantic_status_counts.get(semantic_status, 0) + 1
        critic_status = str((item.get("sandbox_result") or {}).get("critic_status") or "")
        if critic_status:
            critic_status_counts[critic_status] = critic_status_counts.get(critic_status, 0) + 1
    return {
        "dry_run_llm": report.get("dry_run_llm"),
        "llm_provider": report.get("llm_provider"),
        "num_plans_requested": report.get("num_plans_requested"),
        "num_plans_normalized": report.get("num_plans_normalized"),
        "sandboxed_plan_count": report.get("sandboxed_plan_count"),
        "failed_worker_count": report.get("failed_worker_count"),
        "final_sandbox_status": report.get("final_sandbox_status"),
        "selected_plan_index": report.get("selected_plan_index"),
        "rollouts_per_minute": report.get("rollouts_per_minute"),
        "search_status_counts": status_counts,
        "semantic_status_counts": semantic_status_counts,
        "critic_status_counts": critic_status_counts,
        "dry_run_executor_status": _get(report, "dry_run_execution_report.status"),
        "dry_run_executor_success": _get(report, "dry_run_execution_report.success"),
    }


def _llm_plan_search_ablation_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    variants = report.get("variants") or []
    metrics = dict(report.get("comparison") or {})
    for item in variants:
        if not isinstance(item, dict):
            continue
        prefix = str(item.get("variant") or "variant")
        for key in (
            "plan_count",
            "sandboxed_plan_count",
            "accepted_plan_count",
            "review_plan_count",
            "rejected_plan_count",
            "best_sandbox_score",
            "candidate_diversity_unique_step_sequences",
            "failed_worker_count",
            "final_sandbox_status",
        ):
            metrics[f"{prefix}.{key}"] = item.get(key)
    return metrics


def _field_atomic_memory_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    priors = report.get("field_atomic_parameter_priors") if isinstance(report.get("field_atomic_parameter_priors"), dict) else {}
    return {
        "field_atomic_entry_count": summary.get("field_atomic_entry_count", priors.get("field_atomic_entry_count")),
        "field_atomic_success_count": summary.get("field_atomic_success_count", priors.get("field_atomic_success_count")),
        "field_atomic_failure_count": summary.get("field_atomic_failure_count", priors.get("field_atomic_failure_count")),
        "prior_action_count": len((priors.get("by_action") or {}) if isinstance(priors.get("by_action"), dict) else {}),
    }


def _field_atomic_ablation_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        "dry_run_llm": report.get("dry_run_llm"),
        "cold_memory_count": _get(report, "cold_start.planner_metrics.field_atomic_memory_count"),
        "warm_memory_count": _get(report, "warm_start.planner_metrics.field_atomic_memory_count"),
        "warm_prior_action_count": _get(report, "warm_start.planner_metrics.prior_action_count"),
        "parameter_changed_count": _get(report, "comparison.parameter_changed_count"),
        "action_set_changed": _get(report, "comparison.action_set_changed"),
        "cold_write_count": _get(report, "cold_start.write_count"),
        "warm_write_count": _get(report, "warm_start.write_count"),
    }


def _field_atomic_trace_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "trace_count": report.get("trace_count"),
        "action_count": summary.get("action_count"),
        "success_count": summary.get("success_count"),
        "failure_count": summary.get("failure_count"),
        "direct_qpos_true_count": summary.get("direct_qpos_true_count"),
        "direct_qpos_false_count": summary.get("direct_qpos_false_count"),
        "final_error": summary.get("final_error"),
        "action_kind_counts": summary.get("action_kind_counts"),
    }


def _rewrite_loop_ablation_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metrics = {
        "variant_count": report.get("variant_count"),
        "status_counts": report.get("status_counts"),
        "sequential_rewrite_report_count": report.get("sequential_rewrite_report_count"),
        "has_single_plan_baseline": summary.get("has_single_plan_baseline"),
        "has_multi_candidate_critic_ranking": summary.get("has_multi_candidate_critic_ranking"),
        "has_sequential_critic_feedback_rewrite": summary.get("has_sequential_critic_feedback_rewrite"),
        "has_failure_memory_ranking_evidence": summary.get("has_failure_memory_ranking_evidence"),
        "has_parameter_prior_evidence": summary.get("has_parameter_prior_evidence"),
        "has_true_sequential_rewrite_evidence": summary.get("has_true_sequential_rewrite_evidence"),
        "has_successful_sequential_rewrite_evidence": summary.get("has_successful_sequential_rewrite_evidence"),
    }
    for item in report.get("variants") or []:
        if not isinstance(item, dict):
            continue
        variant_id = str(item.get("variant_id") or "")
        item_metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        if variant_id:
            for key in (
                "rewrite_rounds",
                "final_sandbox_status",
                "sandboxed_plan_count",
                "critic_feedback_history_count",
                "sandbox_status_changed",
                "changed_rate",
                "failure_evidence_count",
                "warm_memory_count",
                "warm_prior_action_count",
                "parameter_changed_count",
            ):
                if key in item_metrics:
                    metrics[f"{variant_id}.{key}"] = item_metrics.get(key)
    return metrics


def _physical_perturbation_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metrics = {
        "rollout_count": report.get("rollout_count"),
        "success_count": summary.get("success_count"),
        "task_success_count": summary.get("task_success_count"),
        "failure_label_counts": summary.get("failure_label_counts"),
    }
    for row in summary.get("variant_summary") or []:
        if not isinstance(row, dict):
            continue
        variant_id = str(row.get("variant_id") or "")
        if variant_id:
            metrics[f"{variant_id}.success_rate"] = row.get("success_rate")
            metrics[f"{variant_id}.success_rate_delta_vs_nominal"] = row.get("success_rate_delta_vs_nominal")
    return metrics


def _physical_default_audit_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "file_count": report.get("file_count"),
        "direct_qpos_true_total": summary.get("direct_qpos_true_total"),
        "direct_qpos_false_total": summary.get("direct_qpos_false_total"),
    }


def _write_policy_audit_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "entry_count": report.get("entry_count"),
        "decision_counts": summary.get("decision_counts"),
        "reason_counts": summary.get("reason_counts"),
        "preserved_reason_counts": summary.get("preserved_reason_counts"),
        "memory_role_counts": summary.get("memory_role_counts"),
    }


def _write_policy_pressure_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        "case_count": report.get("case_count"),
        "stored_library_entry_count": report.get("stored_library_entry_count"),
        "decision_counts": report.get("decision_counts"),
        "reason_counts": report.get("reason_counts"),
        "expected_decisions_present": report.get("expected_decisions_present"),
    }


def build_summary(report_dir: Path) -> dict[str, Any]:
    paths = {key: report_dir / rel for key, rel in DEFAULT_REPORTS.items()}
    paths.update({key: Path(rel) for key, rel in DEFAULT_TMP_REPORTS.items()})
    reports = {key: _load_json(path) for key, path in paths.items()}

    claims: list[dict[str, Any]] = []

    library = reports["library_summary"]
    library_metrics = {}
    _metric(library_metrics, "entry_count", _get(library, "summary.entry_count"))
    _metric(library_metrics, "source_distribution", _get(library, "summary.source_distribution"))
    _metric(library_metrics, "gap_count", _get(library, "summary.coverage.gap_count"))
    _metric(library_metrics, "calibration_count", _get(library, "summary.coverage.calibration_count"))
    claims.append(_claim(
        claim="Unified experience memory stores simulation and pseudo-real episodes with gap/calibration fields.",
        status=_status(bool(library_metrics.get("entry_count")), library),
        primary_report=_source(paths["library_summary"]),
        key_metrics=library_metrics,
        safe_wording="The memory library stores simulation and pseudo-real experiences using a unified schema with gap and calibration fields.",
        avoid_wording="Do not claim true real-robot validation from this report.",
    ))

    real = reports["real_format"]
    claims.append(_claim(
        claim="Real-format import and pseudo-real evidence exercise sim-real pair, gap, and sandbox calibration paths.",
        status=_status(bool(_get(real, "core_metrics.pseudo_real_entry_count", 0)), real),
        primary_report=_source(paths["real_format"]),
        key_metrics=dict(_get(real, "core_metrics", {})),
        safe_wording=_get(real, "paper_wording.safe_claim", "Real-format import and pseudo-real evidence exercise pairing, gap extraction, and calibration."),
        avoid_wording=_get(real, "paper_wording.avoid_claim", "Do not claim real-robot validation."),
    ))

    sensor_metrics = {
        "sensor_evidence_entry_count": _get(real, "library_evidence.sensor_evidence_entry_count"),
        "rgbd_evidence_entry_count": _get(real, "library_evidence.rgbd_evidence_entry_count"),
        "lidar_evidence_entry_count": _get(real, "library_evidence.lidar_evidence_entry_count"),
        "wrist_force_evidence_entry_count": _get(real, "library_evidence.wrist_force_evidence_entry_count"),
        "sensor_modality_distribution": _get(real, "library_evidence.sensor_modality_distribution"),
        "sensor_gap_entry_count": _get(real, "library_evidence.sensor_gap_entry_count"),
    }
    claims.append(_claim(
        claim="Shared real-format memory stores RGB-D, lidar, and wrist-force evidence and can derive conservative sensor-gap summaries.",
        status="partial" if real is not None else "missing_report",
        primary_report=_source(paths["real_format"]),
        key_metrics=sensor_metrics,
        safe_wording=(
            "The implementation supports real-format sensor evidence fields for RGB-D, lidar, and wrist-force observations, "
            "and can derive conservative sensor-gap summaries from those stored observations."
        ),
        avoid_wording="Do not claim real-robot success-rate improvement or validated sensor-derived calibration effects from this row alone.",
    ))

    sandbox = reports["sandbox_g4"]
    claims.append(_claim(
        claim="Candidate plans can be shadow-rolled out in sandbox and scored before selection.",
        status=_status(bool(_get(sandbox, "candidate_count", 0)), sandbox),
        primary_report=_source(paths["sandbox_g4"]),
        key_metrics=_sandbox_summary(sandbox),
        safe_wording="Candidate plans are shadow-executed in MuJoCo and fused with memory scores before selection.",
        avoid_wording="Do not claim a full digital twin or real-world execution success.",
    ))

    stage = reports["stage_context"] or reports["stage_context_fallback"] or reports["stage_policy_context"]
    if reports["stage_context"]:
        stage_path = paths["stage_context"]
    elif reports["stage_context_fallback"]:
        stage_path = paths["stage_context_fallback"]
    else:
        stage_path = paths["stage_policy_context"]
    stage_metrics = dict(_get(stage, "summary", _get(stage, "stage_context_summary", {})))
    claims.append(_claim(
        claim="Stage-aware retrieval renders planner context separating generation, ranking, rewrite, and writeback evidence.",
        status=_status(bool(stage_metrics.get("context_count")), stage),
        primary_report=_source(stage_path),
        key_metrics=stage_metrics,
        safe_wording="Planner context is assembled from stage-specific memory evidence separating positive examples, risk priors, critic evidence, and writeback histories.",
        avoid_wording="Do not claim a learned multi-stage planner.",
    ))

    visual_g3 = reports["visual_g3"]
    visual_g4 = reports["visual_g4"]
    visual_metrics = {
        "visual_index.indexed_entry_count": _get(reports["visual_index"], "indexed_entry_count"),
        "visual_index.indexed_image_count": _get(reports["visual_index"], "indexed_image_count"),
        "g3.selected_candidate_before_visual": _get(visual_g3, "comparison.selected_candidate_before_visual"),
        "g3.selected_candidate_after_visual": _get(visual_g3, "comparison.selected_candidate_after_visual"),
        "g3.selected_candidate_change": _get(visual_g3, "comparison.selected_candidate_change"),
        "g3.retrieval_changed_rate": _get(visual_g3, "comparison.retrieval_changed_rate"),
        "g4.candidate_score_delta_avg": _get(visual_g4, "comparison.candidate_score_delta_avg"),
        "g4.candidate_score_delta_max": _get(visual_g4, "comparison.candidate_score_delta_max"),
    }
    claims.append(_claim(
        claim="Visual keyframes are used as an auxiliary retrieval signal during candidate ranking.",
        status=_status(bool(_get(visual_g3, "comparison.visual_score_count", 0)), visual_g3),
        primary_report=f"{_source(paths['visual_g3'])}; {_source(paths['visual_g4'])}",
        key_metrics=visual_metrics,
        safe_wording="Visual keyframes are indexed and used as an auxiliary retrieval signal during candidate ranking.",
        avoid_wording="Do not claim broad multimodal semantic reasoning.",
    ))

    cal = reports["calibration_ablation"]
    cal_summary = dict(_get(cal, "candidates.0.summary", {}))
    claims.append(_claim(
        claim="Gap-derived calibration changes sandbox initial state and contributes an explicit risk penalty.",
        status=_status(bool(cal_summary.get("object_start_delta_full_vs_no_calibration")), cal),
        primary_report=_source(paths["calibration_ablation"]),
        key_metrics=cal_summary,
        safe_wording=_get(cal, "safe_paper_wording", "Gap-derived calibration changes sandbox initialization and contributes an explicit risk penalty."),
        avoid_wording=_get(cal, "claim_boundary", "Do not claim dynamics/friction/contact calibration."),
    ))

    lesson = reports["lesson_quality"]
    claims.append(_claim(
        claim="LLM-generated lessons are audited for grounding, validity, concision, and conflicts before policy use.",
        status=_status(bool(_get(lesson, "quality_pass", False)), lesson),
        primary_report=_source(paths["lesson_quality"]),
        key_metrics=dict(_get(lesson, "metrics", {})),
        safe_wording=_get(lesson, "safe_paper_wording", "Generated lessons are checked before policy adjustment."),
        avoid_wording=_get(lesson, "claim_boundary", "Do not claim learned policy rules."),
    ))

    safety = reports["safety_stress"] or reports["safety_stress_fallback"]
    safety_path = paths["safety_stress"] if reports["safety_stress"] else paths["safety_stress_fallback"]
    claims.append(_claim(
        claim="Sandbox critic identifies risky candidates and increases safety evidence under stress variants.",
        status=_status(bool(safety), safety),
        primary_report=_source(safety_path),
        key_metrics=_summarize_safety_stress(safety),
        safe_wording="The sandbox critic identifies unsafe candidates and increases the safety margin in stress reports.",
        avoid_wording="Do not claim selection changed if memory-only already selected a safe candidate.",
    ))

    harder = reports["harder_safety_stress"]
    claims.append(_claim(
        claim="Under adversarial ranking stress, sandbox critic redirects selection away from artificially boosted risky candidates.",
        status=_status(bool(_get(harder, "summary.sandbox_prevented_risky_selection_count", 0)), harder),
        primary_report=_source(paths["harder_safety_stress"]),
        key_metrics=dict(_get(harder, "summary", {})),
        safe_wording=_get(harder, "safe_paper_wording", "Under adversarial ranking stress, sandbox critic can redirect selection away from artificially boosted risky candidates."),
        avoid_wording=_get(harder, "claim_boundary", "Do not claim this selection change occurs in the unperturbed policy."),
    ))

    writeback = reports["writeback_benchmark"] or reports["writeback_benchmark_fallback"]
    writeback_path = paths["writeback_benchmark"] if reports["writeback_benchmark"] else paths["writeback_benchmark_fallback"]
    writeback_metrics = {
        "round_count": _get(writeback, "round_count"),
        "write_count": _get(writeback, "write_count"),
        "entry_count_delta": _get(writeback, "entry_count_delta"),
        "new_memory_retrieval_rate": _get(writeback, "new_memory_retrieval_rate"),
        "selected_candidate_change_rate": _get(writeback, "selected_candidate_change_rate"),
        "score_delta_after_writeback_avg": _get(writeback, "score_delta_after_writeback_avg"),
        "risk_delta_after_writeback_avg": _get(writeback, "risk_delta_after_writeback_avg"),
    }
    claims.append(_claim(
        claim="Closed-loop writeback stores executed experiences and retrieves them in later ranking passes.",
        status=_status(bool(_get(writeback, "write_count", 0)), writeback),
        primary_report=_source(writeback_path),
        key_metrics=writeback_metrics,
        safe_wording="Newly executed experiences can be written back and retrieved by later candidate ranking passes.",
        avoid_wording="Do not claim statistically proven long-horizon success-rate improvement.",
    ))

    coverage = reports["memory_type_coverage"]
    coverage_metrics = {
        "entry_count": _get(coverage, "entry_count"),
        "covered_memory_type_count_avg": _get(coverage, "summary.covered_memory_type_count_avg"),
        "entries_covering_all_memory_types": _get(coverage, "summary.entries_covering_all_memory_types"),
        "entries_covering_all_memory_types_rate": _get(coverage, "summary.entries_covering_all_memory_types_rate"),
        "temporal_memory.coverage_rate": _get(coverage, "coverage.temporal_memory.coverage_rate"),
        "spatial_memory.coverage_rate": _get(coverage, "coverage.spatial_memory.coverage_rate"),
        "episodic_memory.coverage_rate": _get(coverage, "coverage.episodic_memory.coverage_rate"),
        "semantic_memory.coverage_rate": _get(coverage, "coverage.semantic_memory.coverage_rate"),
        "perceptual_memory.coverage_rate": _get(coverage, "coverage.perceptual_memory.coverage_rate"),
        "sim_real_gap_memory.coverage_rate": _get(coverage, "coverage.sim_real_gap_memory.coverage_rate"),
    }
    claims.append(_claim(
        claim="The experience library represents multiple robot memory types rather than only flat episode logs.",
        status=_status(bool(_get(coverage, "entry_count", 0)), coverage),
        primary_report=_source(paths["memory_type_coverage"]),
        key_metrics=coverage_metrics,
        safe_wording=_get(
            coverage,
            "paper_wording.safe_claim",
            "The library stores temporal, spatial, episodic, semantic, perceptual, and sim-real gap evidence in explicit fields.",
        ),
        avoid_wording=_get(
            coverage,
            "paper_wording.avoid_claim",
            "Do not claim broad benchmark-scale memory coverage or real-robot sensor validation.",
        ),
    ))

    text_semantic = reports["text_semantic"]
    text_metrics = {
        "entry_count": _get(text_semantic, "entry_count"),
        "semantic_summary_nonempty_count": _get(text_semantic, "summary.semantic_summary_nonempty_count"),
        "semantic_summary_nonempty_rate": _get(text_semantic, "summary.semantic_summary_nonempty_rate"),
        "avg_token_count": _get(text_semantic, "summary.avg_token_count"),
        "semantic_signal_rate": _get(text_semantic, "summary.semantic_signal_rate"),
        "query_count": _get(text_semantic, "summary.query_count"),
        "same_scenario_topk_match_count": _get(text_semantic, "summary.same_scenario_topk_match_count"),
        "cross_condition_topk_match_count": _get(text_semantic, "summary.cross_condition_topk_match_count"),
        "query_token_coverage": _get(text_semantic, "summary.query_token_coverage"),
    }
    claims.append(_claim(
        claim="Text-semantic summaries provide an auxiliary retrieval signal over structured experience fields.",
        status=_status(bool(_get(text_semantic, "summary.semantic_summary_nonempty_count", 0)), text_semantic),
        primary_report=_source(paths["text_semantic"]),
        key_metrics=text_metrics,
        safe_wording=_get(
            text_semantic,
            "paper_wording.safe_claim",
            "The system constructs text-semantic summaries from structured experience fields and uses lightweight retrieval as an auxiliary signal.",
        ),
        avoid_wording=_get(
            text_semantic,
            "paper_wording.avoid_claim",
            "Do not claim this is a learned language encoder or FAISS benchmark result.",
        ),
    ))

    llm_plan_search = reports["llm_plan_search_real"]
    llm_plan_metrics = _llm_plan_search_metrics(llm_plan_search)
    claims.append(_claim(
        claim="Real LLM-generated recovery-plan candidates can be semantically validated, sandboxed in parallel, critic-ranked, and exported as a validated robot plan.",
        status=_status(
            bool(llm_plan_search)
            and _get(llm_plan_search, "dry_run_llm") is False
            and bool(_get(llm_plan_search, "sandboxed_plan_count", 0))
            and _get(llm_plan_search, "failed_worker_count", 1) == 0
            and bool(_get(llm_plan_search, "validated_robot_plan.plan_id", "")),
            llm_plan_search,
        ),
        primary_report=_source(paths["llm_plan_search_real"]),
        key_metrics=llm_plan_metrics,
        safe_wording=(
            "A configured external LLM generated multiple recovery-plan candidates; the system normalized them, "
            "validated skill preconditions/effects, ran parallel MuJoCo sandbox rollouts, ranked by critic/sandbox status, "
            "and exported a validated_robot_plan for dry-run dispatch."
        ),
        avoid_wording=(
            "Do not claim real-robot execution success, arbitrary unseen-skill planning, or statistically proven superiority over all single-plan baselines."
        ),
    ))

    llm_plan_ablation = reports["llm_plan_search_ablation"]
    claims.append(_claim(
        claim="Multi-candidate LLM plan search evaluates more recovery alternatives than a single-plan LLM baseline before selecting a validated plan.",
        status=_status(bool(_get(llm_plan_ablation, "comparison.plan_count_delta", 0)), llm_plan_ablation),
        primary_report=_source(paths["llm_plan_search_ablation"]),
        key_metrics=_llm_plan_search_ablation_metrics(llm_plan_ablation),
        safe_wording=_get(
            llm_plan_ablation,
            "paper_wording.safe_claim",
            "Multi-candidate search evaluates more LLM-generated recovery alternatives than a single-plan baseline before sandbox-backed selection.",
        ),
        avoid_wording=_get(
            llm_plan_ablation,
            "paper_wording.avoid_claim",
            "Do not claim a statistically meaningful success-rate gain from this small ablation alone.",
        ),
    ))

    field_memory = reports["field_atomic_memory"] or reports["field_atomic_memory_fallback"]
    field_memory_path = paths["field_atomic_memory"] if reports["field_atomic_memory"] else paths["field_atomic_memory_fallback"]
    claims.append(_claim(
        claim="Field-atomic execution experiences are converted into parameter priors for later LLM planning.",
        status=_status(bool(_field_atomic_memory_metrics(field_memory).get("field_atomic_entry_count", 0)), field_memory),
        primary_report=_source(field_memory_path),
        key_metrics=_field_atomic_memory_metrics(field_memory),
        safe_wording=_get(
            field_memory,
            "paper_wording.safe_claim",
            "Field-atomic success and failure writebacks can be summarized as recommended and avoid parameter priors for planner_input.",
        ),
        avoid_wording=_get(
            field_memory,
            "paper_wording.avoid_claim",
            "Do not claim the parameter priors are learned optimal controls or real-robot validated.",
        ),
    ))

    field_ablation = reports["field_atomic_ablation"] or reports["field_atomic_ablation_fallback"]
    field_ablation_path = paths["field_atomic_ablation"] if reports["field_atomic_ablation"] else paths["field_atomic_ablation_fallback"]
    claims.append(_claim(
        claim="Cold-start and writeback rounds show whether field-atomic memory is exposed to the planner as explicit context.",
        status=_status(bool(_get(field_ablation, "warm_start.planner_metrics.field_atomic_memory_count", 0)), field_ablation),
        primary_report=_source(field_ablation_path),
        key_metrics=_field_atomic_ablation_metrics(field_ablation),
        safe_wording=_get(
            field_ablation,
            "paper_wording.safe_claim",
            "Field atomic writeback can be reloaded as explicit planner_input priors in a subsequent planning round.",
        ),
        avoid_wording=_get(
            field_ablation,
            "paper_wording.avoid_claim",
            "Do not claim parameter improvement from dry-run LLM if the mock plan is deterministic.",
        ),
    ))

    field_trace = reports["field_atomic_trace"] or reports["field_atomic_trace_fallback"]
    field_trace_path = paths["field_atomic_trace"] if reports["field_atomic_trace"] else paths["field_atomic_trace_fallback"]
    claims.append(_claim(
        claim="Field-atomic execution traces expose action-level tracking error, gripper commands, and direct-qpos usage for debugging.",
        status=_status(bool(_get(field_trace, "trace_count", 0)), field_trace),
        primary_report=_source(field_trace_path),
        key_metrics=_field_atomic_trace_metrics(field_trace),
        safe_wording=_get(
            field_trace,
            "paper_wording.safe_claim",
            "Field-atomic executions expose action-level trace summaries, including final tracking error, gripper command, control mode, and direct-qpos usage for onsite debugging.",
        ),
        avoid_wording=_get(
            field_trace,
            "paper_wording.avoid_claim",
            "Do not claim these summaries prove real-robot tracking accuracy; they summarize MuJoCo or stored experience traces only.",
        ),
    ))

    rewrite_ablation = reports["rewrite_loop_ablation"] or reports["rewrite_loop_ablation_fallback"]
    rewrite_ablation_path = paths["rewrite_loop_ablation"] if reports["rewrite_loop_ablation"] else paths["rewrite_loop_ablation_fallback"]
    claims.append(_claim(
        claim="Rewrite-loop ablation reports separate supported component evidence from missing true sequential LLM rewrite evidence.",
        status=_status(bool(_get(rewrite_ablation, "variant_count", 0)), rewrite_ablation),
        primary_report=_source(rewrite_ablation_path),
        key_metrics=_rewrite_loop_ablation_metrics(rewrite_ablation),
        safe_wording=_get(
            rewrite_ablation,
            "paper_wording.safe_claim",
            "Existing reports support single-plan validation, multi-candidate critic ranking, memory-backed risk evidence, and field-atomic parameter-prior exposure; true sequential rewrite requires rewrite_rounds > 0.",
        ),
        avoid_wording=_get(
            rewrite_ablation,
            "paper_wording.avoid_claim",
            "Do not claim critic feedback caused an LLM rewrite unless the underlying report contains multiple attempts and a non-empty critic_feedback_history.",
        ),
    ))

    perturbation = reports["physical_perturbation"] or reports["physical_perturbation_fallback"]
    perturbation_path = paths["physical_perturbation"] if reports["physical_perturbation"] else paths["physical_perturbation_fallback"]
    claims.append(_claim(
        claim="Physical sandbox rollouts can sweep control and scene perturbations to expose sensitivity to pose, delay, gain, and gripper effects.",
        status=_status(bool(_get(perturbation, "rollout_count", 0)), perturbation),
        primary_report=_source(perturbation_path),
        key_metrics=_physical_perturbation_metrics(perturbation),
        safe_wording=_get(
            perturbation,
            "paper_wording.safe_claim",
            "The physical MuJoCo sandbox can sweep pose, delay, gain, and gripper perturbations and report which failures are sensitive to those controls.",
        ),
        avoid_wording=_get(
            perturbation,
            "paper_wording.avoid_claim",
            "Do not claim real-driver calibration or real-robot robustness without measured hardware response data.",
        ),
    ))

    physical_defaults = reports["physical_default_audit"] or reports["physical_default_audit_fallback"]
    physical_defaults_path = paths["physical_default_audit"] if reports["physical_default_audit"] else paths["physical_default_audit_fallback"]
    claims.append(_claim(
        claim="Core skill defaults are audited to prefer physical actuator execution over direct-qpos shortcuts.",
        status=_status(_get(physical_defaults, "summary.direct_qpos_true_total", 1) == 0, physical_defaults),
        primary_report=_source(physical_defaults_path),
        key_metrics=_physical_default_audit_metrics(physical_defaults),
        safe_wording=_get(
            physical_defaults,
            "paper_wording.safe_claim",
            "Core base, torso, arm, gripper, and field atomic defaults have been audited to prefer physical control over direct-qpos shortcuts.",
        ),
        avoid_wording=_get(
            physical_defaults,
            "paper_wording.avoid_claim",
            "Do not claim that every explicit debug or test path is physical-only; the audit concerns defaults, not every possible parameter override.",
        ),
    ))

    write_audit = reports["write_policy_audit"] or reports["write_policy_audit_fallback"]
    write_audit_path = paths["write_policy_audit"] if reports["write_policy_audit"] else paths["write_policy_audit_fallback"]
    claims.append(_claim(
        claim="Memory writeback is auditable through explicit write, skip, merge, and reject decisions.",
        status=_status(bool(_get(write_audit, "entry_count", 0)), write_audit),
        primary_report=_source(write_audit_path),
        key_metrics=_write_policy_audit_metrics(write_audit),
        safe_wording=_get(
            write_audit,
            "paper_wording.safe_claim",
            "The write policy is auditable: each candidate entry receives an explicit write, skip, merge, or reject decision with a reason.",
        ),
        avoid_wording=_get(
            write_audit,
            "paper_wording.avoid_claim",
            "Do not claim the write policy is learned or globally optimal; it is an explicit engineering gate.",
        ),
    ))

    write_pressure = reports["write_policy_pressure"] or reports["write_policy_pressure_fallback"]
    write_pressure_path = paths["write_policy_pressure"] if reports["write_policy_pressure"] else paths["write_policy_pressure_fallback"]
    pressure_decisions = _get(write_pressure, "expected_decisions_present", {})
    pressure_supported = bool(write_pressure) and all(bool((pressure_decisions or {}).get(name)) for name in ("write", "merge", "skip", "reject"))
    claims.append(_claim(
        claim="Write-policy pressure tests exercise all lifecycle decisions: write, merge, skip, and reject.",
        status=_status(pressure_supported, write_pressure),
        primary_report=_source(write_pressure_path),
        key_metrics=_write_policy_pressure_metrics(write_pressure),
        safe_wording=(
            "A deterministic pressure test constructs representative experiences that trigger write, merge, skip, "
            "and reject decisions, showing that writeback is not a plain append-only log."
        ),
        avoid_wording="Do not claim the handcrafted pressure cases prove optimal memory lifecycle policy.",
    ))

    runtime_schema = ROOT / "templates" / "field_runtime_scene_observation_template.json"
    claims.append(_claim(
        claim="Runtime scene observations have a fixed schema for converting onsite RGB-D/LiDAR outputs into sandbox scene construction inputs.",
        status="supported" if runtime_schema.exists() else "missing_report",
        primary_report=str(runtime_schema),
        key_metrics={"schema_exists": runtime_schema.exists()},
        safe_wording="The implementation defines a field runtime observation template for converting onsite perception outputs into sandbox scene inputs.",
        avoid_wording="Do not claim automatic high-fidelity scene reconstruction until real perception outputs are mapped and validated onsite.",
    ))

    return {
        "schema_version": "paper_evidence_summary_v1",
        "report_dir": str(report_dir),
        "claim_count": len(claims),
        "supported_claim_count": sum(1 for claim in claims if claim["status"] == "supported"),
        "missing_report_count": sum(1 for claim in claims if claim["status"] == "missing_report"),
        "claims": claims,
    }


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            compact = json.dumps(value, ensure_ascii=False, sort_keys=True)
            parts.append(f"`{key}`={compact}")
        else:
            parts.append(f"`{key}`={value}")
    return "<br>".join(parts)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Paper Evidence Summary",
        "",
        f"- Claim count: {summary['claim_count']}",
        f"- Supported claim count: {summary['supported_claim_count']}",
        f"- Missing report count: {summary['missing_report_count']}",
        "",
        "| Claim | Status | Primary report | Key metrics | Safe wording | Avoid wording |",
        "|---|---|---|---|---|---|",
    ]
    for claim in summary["claims"]:
        lines.append(
            "| "
            + " | ".join([
                str(claim["claim"]).replace("|", "\\|"),
                str(claim["status"]),
                f"`{claim['primary_report']}`",
                _format_metrics(claim["key_metrics"]).replace("|", "\\|"),
                str(claim["safe_wording"]).replace("|", "\\|"),
                str(claim["avoid_wording"]).replace("|", "\\|"),
            ])
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.report_dir)
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
        "claim_count": summary["claim_count"],
        "supported_claim_count": summary["supported_claim_count"],
        "missing_report_count": summary["missing_report_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
