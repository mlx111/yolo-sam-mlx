"""Build a rewrite-loop ablation evidence report from existing planner/sandbox reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("results/memory/universal_pipeline_calibration_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize no-feedback, critic-feedback, failure-memory, and parameter-prior rewrite evidence."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--single-plan-report", type=Path, default=None)
    parser.add_argument("--multi-plan-report", type=Path, default=None)
    parser.add_argument("--sequential-rewrite-report", type=Path, default=None)
    parser.add_argument("--policy-memory-report", type=Path, default=None)
    parser.add_argument("--field-atomic-ablation", type=Path, default=None)
    parser.add_argument("--field-atomic-trace", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _get(payload: dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = payload
    for part in dotted.split("."):
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _sandbox_metrics_from_single(report: dict[str, Any]) -> dict[str, Any]:
    attempts = report.get("attempts") if isinstance(report.get("attempts"), list) else []
    final_attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    sandbox = final_attempt.get("sandbox_result") if isinstance(final_attempt.get("sandbox_result"), dict) else {}
    critic = final_attempt.get("critic_feedback") if isinstance(final_attempt.get("critic_feedback"), dict) else {}
    planner_input = final_attempt.get("planner_input") if isinstance(final_attempt.get("planner_input"), dict) else {}
    return {
        "report_present": bool(report),
        "dry_run_llm": report.get("dry_run_llm", ""),
        "attempt_count": report.get("attempt_count", len(attempts) if attempts else 0),
        "rewrite_rounds": report.get("rewrite_rounds", max(0, len(attempts) - 1)),
        "critic_feedback_count": len(report.get("critic_feedback_history") or []),
        "final_sandbox_status": report.get("final_sandbox_status", ""),
        "sandbox_score": sandbox.get("sandbox_score"),
        "critic_status": sandbox.get("critic_status", critic.get("critic_status", "")),
        "critic_risk_score": sandbox.get("critic_risk_score", critic.get("critic_risk_score", "")),
        "critic_flag_count": len(sandbox.get("critic_flags") or critic.get("critic_flags") or []),
        "semantic_status": _get(final_attempt, "plan_semantic_validation.status"),
        "planner_has_rewrite_guidance": bool(planner_input.get("rewrite_guidance")),
        "planner_has_recovery_parameter_priors": bool(planner_input.get("recovery_parameter_priors")),
    }


def _sandbox_metrics_from_multi(report: dict[str, Any]) -> dict[str, Any]:
    candidates = report.get("candidate_reports") if isinstance(report.get("candidate_reports"), list) else []
    critic_status_counts = Counter()
    search_status_counts = Counter()
    semantic_status_counts = Counter()
    scores: list[float] = []
    risk_scores: list[float] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        search_status_counts[str(item.get("search_status") or "unknown")] += 1
        semantic_status_counts[str(_get(item, "plan_semantic_validation.status", "unknown"))] += 1
        sandbox = item.get("sandbox_result") if isinstance(item.get("sandbox_result"), dict) else {}
        critic_status_counts[str(sandbox.get("critic_status") or "unknown")] += 1
        if "sandbox_score" in sandbox:
            scores.append(_safe_float(sandbox.get("sandbox_score")))
        if "critic_risk_score" in sandbox:
            risk_scores.append(_safe_float(sandbox.get("critic_risk_score")))
    return {
        "report_present": bool(report),
        "dry_run_llm": report.get("dry_run_llm", ""),
        "num_plans_requested": report.get("num_plans_requested", 0),
        "num_plans_normalized": report.get("num_plans_normalized", 0),
        "sandboxed_plan_count": report.get("sandboxed_plan_count", len(candidates)),
        "failed_worker_count": report.get("failed_worker_count", 0),
        "final_sandbox_status": report.get("final_sandbox_status", ""),
        "selected_plan_index": report.get("selected_plan_index", ""),
        "rollouts_per_minute": report.get("rollouts_per_minute", ""),
        "search_status_counts": dict(search_status_counts),
        "semantic_status_counts": dict(semantic_status_counts),
        "critic_status_counts": dict(critic_status_counts),
        "best_sandbox_score": max(scores) if scores else "",
        "avg_critic_risk_score": round(sum(risk_scores) / len(risk_scores), 6) if risk_scores else "",
    }


def _policy_memory_metrics(report: dict[str, Any]) -> dict[str, Any]:
    comparisons = report.get("comparisons") if isinstance(report.get("comparisons"), list) else []
    failure_evidence_count = 0
    success_evidence_count = 0
    risk_delta_values: list[float] = []
    score_delta_values: list[float] = []
    decisions = Counter()
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        decisions[str(item.get("memory_decision") or "unknown")] += 1
        if isinstance(item.get("risk_delta"), (int, float)):
            risk_delta_values.append(float(item["risk_delta"]))
        if isinstance(item.get("score_delta"), (int, float)):
            score_delta_values.append(float(item["score_delta"]))
        for evidence in item.get("memory_risk_evidence") or []:
            if isinstance(evidence, dict) and evidence.get("success") is False:
                failure_evidence_count += 1
            elif isinstance(evidence, dict) and evidence.get("success") is True:
                success_evidence_count += 1
    return {
        "report_present": bool(report),
        "comparison_count": report.get("comparison_count", len(comparisons)),
        "changed_count": report.get("changed_count", 0),
        "changed_rate": round(float(report.get("changed_count", 0)) / max(len(comparisons), 1), 6) if comparisons else 0.0,
        "memory_decision_counts": dict(decisions),
        "failure_evidence_count": failure_evidence_count,
        "success_evidence_count": success_evidence_count,
        "avg_risk_delta": round(sum(risk_delta_values) / len(risk_delta_values), 6) if risk_delta_values else "",
        "avg_score_delta": round(sum(score_delta_values) / len(score_delta_values), 6) if score_delta_values else "",
    }


def _field_atomic_metrics(report: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_present": bool(report),
        "dry_run_llm": report.get("dry_run_llm", ""),
        "cold_memory_count": _get(report, "cold_start.planner_metrics.field_atomic_memory_count", ""),
        "warm_memory_count": _get(report, "warm_start.planner_metrics.field_atomic_memory_count", ""),
        "warm_prior_action_count": _get(report, "warm_start.planner_metrics.prior_action_count", ""),
        "parameter_changed_count": _get(report, "comparison.parameter_changed_count", ""),
        "action_set_changed": _get(report, "comparison.action_set_changed", ""),
        "trace_count": trace.get("trace_count", ""),
        "trace_direct_qpos_true_count": _get(trace, "summary.direct_qpos_true_count", ""),
        "trace_final_error": _get(trace, "summary.final_error", {}),
    }


def _sequential_rewrite_metrics(report: dict[str, Any]) -> dict[str, Any]:
    attempts = report.get("attempts") if isinstance(report.get("attempts"), list) else []
    status_counts = Counter()
    semantic_status_counts = Counter()
    critic_status_counts = Counter()
    sandbox_status_changed = False
    previous_status = ""
    for item in attempts:
        if not isinstance(item, dict):
            continue
        round_status = str(item.get("round_status") or "unknown")
        status_counts[round_status] += 1
        semantic_status_counts[str(_get(item, "plan_semantic_validation.status", "unknown"))] += 1
        critic_status = str(_get(item, "sandbox_result.critic_status", "unknown"))
        critic_status_counts[critic_status] += 1
        if previous_status and round_status != previous_status:
            sandbox_status_changed = True
        previous_status = round_status
    return {
        "report_present": bool(report),
        "dry_run_llm": report.get("dry_run_llm", ""),
        "attempt_count": report.get("attempt_count", len(attempts)),
        "rewrite_rounds": report.get("rewrite_rounds", max(0, len(attempts) - 1)),
        "critic_feedback_history_count": len(report.get("critic_feedback_history") or []),
        "final_sandbox_status": report.get("final_sandbox_status", ""),
        "round_status_counts": dict(status_counts),
        "semantic_status_counts": dict(semantic_status_counts),
        "critic_status_counts": dict(critic_status_counts),
        "sandbox_status_changed": sandbox_status_changed,
        "first_round_status": str(_get(report, "attempts.0.round_status", "")),
        "last_round_status": str(_get(report, f"attempts.{max(len(attempts) - 1, 0)}.round_status", "")) if attempts else "",
    }


def _status_for(metrics: dict[str, Any], required: list[str]) -> str:
    if not metrics.get("report_present"):
        return "missing_report"
    missing = [key for key in required if metrics.get(key) in ("", None, 0, False)]
    return "partial" if missing else "supported"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = args.report_dir
    single_path = args.single_plan_report or report_dir / "single_plan_real_llm_g3_clean_report.json"
    multi_path = args.multi_plan_report or report_dir / "llm_plan_candidate_search_g3_clean_real_llm_report.json"
    sequential_path = args.sequential_rewrite_report or report_dir / "rewrite_loop_g3_clean_dryrun_report.json"
    policy_path = args.policy_memory_report or report_dir / "policy_baseline_vs_memory_mitigation_report.json"
    field_ablation_path = args.field_atomic_ablation or report_dir / "field_atomic_memory_ablation.json"
    if not field_ablation_path.exists():
        field_ablation_path = Path("/tmp/field_atomic_ablation.json")
    field_trace_path = args.field_atomic_trace or report_dir / "field_atomic_trace_summary.json"

    single = _load(single_path)
    multi = _load(multi_path)
    sequential = _load(sequential_path)
    policy = _load(policy_path)
    field_ablation = _load(field_ablation_path)
    field_trace = _load(field_trace_path)

    variants = [
        {
            "variant_id": "single_plan_no_feedback",
            "description": "Single LLM plan with no rewrite rounds; sandbox critic is used only for final validation.",
            "source_report": str(single_path),
            "status": _status_for(_sandbox_metrics_from_single(single), ["attempt_count", "final_sandbox_status"]),
            "metrics": _sandbox_metrics_from_single(single),
            "supported_claim": "A single LLM plan can be normalized, semantically checked, sandboxed, and exported when accepted.",
            "claim_boundary": "This variant does not test critic-driven rewriting because rewrite_rounds is zero.",
        },
        {
            "variant_id": "sequential_critic_feedback_rewrite",
            "description": "A first failed plan feeds critic feedback into a later rewrite attempt.",
            "source_report": str(sequential_path),
            "status": _status_for(_sequential_rewrite_metrics(sequential), ["rewrite_rounds", "critic_feedback_history_count"]),
            "metrics": _sequential_rewrite_metrics(sequential),
            "supported_claim": "The loop can run multiple attempts and inject critic feedback into a rewrite round.",
            "claim_boundary": "If final_sandbox_status is not accept, this proves rewrite-loop mechanics, not successful recovery.",
        },
        {
            "variant_id": "multi_candidate_critic_ranking",
            "description": "Multiple LLM candidates are sandboxed and critic-ranked before selecting a validated plan.",
            "source_report": str(multi_path),
            "status": _status_for(_sandbox_metrics_from_multi(multi), ["sandboxed_plan_count", "final_sandbox_status"]),
            "metrics": _sandbox_metrics_from_multi(multi),
            "supported_claim": "Multi-candidate search exposes more than one LLM plan to semantic validation, sandbox rollout, and critic ranking.",
            "claim_boundary": "This is not a sequential rewrite loop unless a report contains multiple rewrite attempts.",
        },
        {
            "variant_id": "critic_plus_failure_memory",
            "description": "Memory-backed ranking exposes success/failure evidence and risk deltas that can be included in planner context.",
            "source_report": str(policy_path),
            "status": _status_for(_policy_memory_metrics(policy), ["comparison_count"]),
            "metrics": _policy_memory_metrics(policy),
            "supported_claim": "Failure/success memory affects candidate ranking evidence and can change selected candidates.",
            "claim_boundary": "This report proves memory-aware ranking evidence, not a full LLM rewrite response to that evidence.",
        },
        {
            "variant_id": "critic_failure_memory_parameter_priors",
            "description": "Field-atomic writeback is reloaded as explicit parameter priors and trace summaries.",
            "source_report": f"{field_ablation_path}; {field_trace_path}",
            "status": _status_for(_field_atomic_metrics(field_ablation, field_trace), ["warm_memory_count", "warm_prior_action_count"]),
            "metrics": _field_atomic_metrics(field_ablation, field_trace),
            "supported_claim": "Writeback can expose prior parameter ranges and execution-error summaries to later planner inputs.",
            "claim_boundary": "Dry-run field_atomic ablation does not prove parameter improvement unless a real LLM changes parameters.",
        },
    ]

    statuses = Counter(str(item["status"]) for item in variants)
    sequential_reports = [
        item
        for item in variants
        if int(_safe_float(item["metrics"].get("rewrite_rounds"), 0.0)) > 0
    ]
    return {
        "schema_version": "rewrite_loop_ablation_report_v1",
        "report_dir": str(report_dir),
        "variant_count": len(variants),
        "status_counts": dict(statuses),
        "sequential_rewrite_report_count": len(sequential_reports),
        "variants": variants,
        "summary": {
            "has_single_plan_baseline": variants[0]["status"] in {"supported", "partial"},
            "has_sequential_critic_feedback_rewrite": variants[1]["status"] in {"supported", "partial"},
            "has_multi_candidate_critic_ranking": variants[2]["status"] in {"supported", "partial"},
            "has_failure_memory_ranking_evidence": variants[3]["status"] in {"supported", "partial"},
            "has_parameter_prior_evidence": variants[4]["status"] in {"supported", "partial"},
            "has_true_sequential_rewrite_evidence": len(sequential_reports) > 0,
            "has_successful_sequential_rewrite_evidence": any(
                item["metrics"].get("final_sandbox_status") == "accept" for item in sequential_reports
            ),
        },
        "paper_wording": {
            "safe_claim": (
                "Existing reports support single-plan validation, multi-candidate critic ranking, memory-backed risk evidence, "
                "field-atomic parameter-prior exposure, and a dry-run sequential critic-feedback rewrite attempt."
            ),
            "avoid_claim": (
                "Do not claim critic-feedback rewriting improves recovery unless the underlying report has rewrite_rounds > 0 "
                "and final_sandbox_status=accept under a real LLM or clearly stated dry-run setting."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rewrite Loop Ablation Report",
        "",
        "This report consolidates existing evidence for rewrite-loop components and explicitly marks missing sequential rewrite evidence.",
        "",
        "## Summary",
        "",
        f"- Variant count: {report['variant_count']}",
        f"- Status counts: `{json.dumps(report['status_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Sequential rewrite reports: {report['sequential_rewrite_report_count']}",
        f"- True sequential rewrite evidence: {report['summary']['has_true_sequential_rewrite_evidence']}",
        "",
        "## Variants",
        "",
        "| Variant | Status | Source | Key metrics | Boundary |",
        "|---|---|---|---|---|",
    ]
    for item in report["variants"]:
        metrics = item.get("metrics") or {}
        compact_keys = [
            "attempt_count",
            "rewrite_rounds",
            "final_sandbox_status",
            "sandboxed_plan_count",
            "changed_rate",
            "failure_evidence_count",
            "warm_memory_count",
            "warm_prior_action_count",
            "parameter_changed_count",
            "trace_count",
        ]
        compact = {key: metrics.get(key) for key in compact_keys if key in metrics}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["variant_id"]),
                    str(item["status"]),
                    f"`{item['source_report']}`",
                    f"`{json.dumps(compact, ensure_ascii=False, sort_keys=True)}`",
                    str(item["claim_boundary"]).replace("|", "\\|"),
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
    report = build_report(args)
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "save_json": str(args.save_json),
                "save_md": str(args.save_md),
                "variant_count": report["variant_count"],
                "status_counts": report["status_counts"],
            "has_true_sequential_rewrite_evidence": report["summary"]["has_true_sequential_rewrite_evidence"],
            "has_successful_sequential_rewrite_evidence": report["summary"]["has_successful_sequential_rewrite_evidence"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
