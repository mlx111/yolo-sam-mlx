"""Compare single-plan LLM validation with multi-plan candidate search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact ablation report for LLM plan candidate search.")
    parser.add_argument("--single-plan-report", type=Path, required=True)
    parser.add_argument("--multi-plan-report", type=Path, required=True)
    parser.add_argument("--save-json", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"report must be a JSON object: {path}")
    return payload


def _plan_steps(plan: dict[str, Any]) -> list[str]:
    return [str(item) for item in plan.get("candidate_steps") or [] if str(item)]


def _single_summary(report: dict[str, Any]) -> dict[str, Any]:
    attempts = [item for item in report.get("attempts") or [] if isinstance(item, dict)]
    attempt = attempts[-1] if attempts else {}
    plan = attempt.get("recovery_plan") if isinstance(attempt.get("recovery_plan"), dict) else {}
    sandbox = attempt.get("sandbox_result") if isinstance(attempt.get("sandbox_result"), dict) else {}
    validation = attempt.get("plan_semantic_validation") if isinstance(attempt.get("plan_semantic_validation"), dict) else {}
    return {
        "variant": "single_plan_real_llm",
        "dry_run_llm": bool(report.get("dry_run_llm")),
        "llm_provider": report.get("llm_provider"),
        "plan_count": 1 if plan else 0,
        "sandboxed_plan_count": 1 if sandbox else 0,
        "accepted_plan_count": 1 if str(report.get("final_sandbox_status") or "") == "accept" else 0,
        "review_plan_count": 1 if str(report.get("final_sandbox_status") or "") == "review" else 0,
        "rejected_plan_count": 1 if str(report.get("final_sandbox_status") or "") == "reject" else 0,
        "final_sandbox_status": report.get("final_sandbox_status"),
        "best_sandbox_score": float(sandbox.get("sandbox_score") or 0.0),
        "critic_status_counts": {str(sandbox.get("critic_status") or "unknown"): 1} if sandbox else {},
        "semantic_status_counts": {str(validation.get("status") or "unknown"): 1} if validation else {},
        "candidate_diversity_unique_step_sequences": 1 if plan else 0,
        "candidate_diversity_unique_action_sets": 1 if plan else 0,
        "selected_plan_index": 0 if plan else -1,
        "selected_plan_steps": _plan_steps(plan),
        "elapsed_s": None,
        "rollouts_per_minute": None,
        "failed_worker_count": 0,
        "dry_run_executor_success": bool((report.get("dry_run_execution_report") or {}).get("success")),
    }


def _multi_summary(report: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for item in report.get("candidate_reports") or [] if isinstance(item, dict)]
    status_counts: dict[str, int] = {}
    critic_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    step_sequences: set[tuple[str, ...]] = set()
    action_sets: set[tuple[str, ...]] = set()
    best_score = 0.0
    for item in candidates:
        status = str(item.get("search_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        sandbox = item.get("sandbox_result") if isinstance(item.get("sandbox_result"), dict) else {}
        critic = str(sandbox.get("critic_status") or "unknown")
        critic_counts[critic] = critic_counts.get(critic, 0) + 1
        semantic = str((item.get("plan_semantic_validation") or {}).get("status") or "unknown")
        semantic_counts[semantic] = semantic_counts.get(semantic, 0) + 1
        steps = tuple(_plan_steps(item.get("recovery_plan") if isinstance(item.get("recovery_plan"), dict) else {}))
        if steps:
            step_sequences.add(steps)
            action_sets.add(tuple(sorted(set(steps))))
        best_score = max(best_score, float(sandbox.get("sandbox_score") or 0.0))
    selected = candidates[0] if candidates else {}
    return {
        "variant": "multi_plan_real_llm_parallel_sandbox",
        "dry_run_llm": bool(report.get("dry_run_llm")),
        "llm_provider": report.get("llm_provider"),
        "plan_count": int(report.get("num_plans_normalized") or len(candidates)),
        "sandboxed_plan_count": int(report.get("sandboxed_plan_count") or 0),
        "accepted_plan_count": status_counts.get("accept", 0),
        "review_plan_count": status_counts.get("review", 0),
        "rejected_plan_count": status_counts.get("reject", 0),
        "final_sandbox_status": report.get("final_sandbox_status"),
        "best_sandbox_score": best_score,
        "critic_status_counts": critic_counts,
        "semantic_status_counts": semantic_counts,
        "candidate_diversity_unique_step_sequences": len(step_sequences),
        "candidate_diversity_unique_action_sets": len(action_sets),
        "selected_plan_index": report.get("selected_plan_index"),
        "selected_plan_steps": _plan_steps(selected.get("recovery_plan") if isinstance(selected.get("recovery_plan"), dict) else {}),
        "elapsed_s": report.get("elapsed_s"),
        "rollouts_per_minute": report.get("rollouts_per_minute"),
        "failed_worker_count": report.get("failed_worker_count"),
        "dry_run_executor_success": bool((report.get("dry_run_execution_report") or {}).get("success")),
    }


def build_report(single: dict[str, Any], multi: dict[str, Any], *, single_path: Path, multi_path: Path) -> dict[str, Any]:
    single_summary = _single_summary(single)
    multi_summary = _multi_summary(multi)
    return {
        "schema_version": "llm_plan_search_ablation_v1",
        "single_plan_report": str(single_path),
        "multi_plan_report": str(multi_path),
        "variants": [single_summary, multi_summary],
        "comparison": {
            "plan_count_delta": multi_summary["plan_count"] - single_summary["plan_count"],
            "sandboxed_plan_count_delta": multi_summary["sandboxed_plan_count"] - single_summary["sandboxed_plan_count"],
            "accepted_plan_count_delta": multi_summary["accepted_plan_count"] - single_summary["accepted_plan_count"],
            "best_sandbox_score_delta": round(multi_summary["best_sandbox_score"] - single_summary["best_sandbox_score"], 4),
            "unique_step_sequence_delta": multi_summary["candidate_diversity_unique_step_sequences"] - single_summary["candidate_diversity_unique_step_sequences"],
            "single_final_status": single_summary["final_sandbox_status"],
            "multi_final_status": multi_summary["final_sandbox_status"],
        },
        "paper_wording": {
            "safe_claim": (
                "The multi-candidate search evaluates more LLM-generated recovery alternatives than a single-plan baseline, "
                "runs valid candidates through parallel sandbox rollout, and selects a critic-approved validated plan."
            ),
            "avoid_claim": (
                "Do not claim a success-rate improvement from this small G3 clean ablation alone; both variants succeeded in the current smoke."
            ),
        },
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# LLM Plan Search Ablation",
        "",
        f"- Single-plan report: `{report['single_plan_report']}`",
        f"- Multi-plan report: `{report['multi_plan_report']}`",
        "",
        "| Variant | Plans | Sandboxed | Accepted | Review | Rejected | Best score | Unique step sequences | Failed workers | Final status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["variants"]:
        lines.append(
            "| "
            + " | ".join([
                str(item["variant"]),
                str(item["plan_count"]),
                str(item["sandboxed_plan_count"]),
                str(item["accepted_plan_count"]),
                str(item["review_plan_count"]),
                str(item["rejected_plan_count"]),
                str(item["best_sandbox_score"]),
                str(item["candidate_diversity_unique_step_sequences"]),
                str(item["failed_worker_count"]),
                str(item["final_sandbox_status"]),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Comparison",
        "",
        "```json",
        json.dumps(report["comparison"], indent=2, ensure_ascii=False),
        "```",
        "",
        f"Safe claim: {report['paper_wording']['safe_claim']}",
        "",
        f"Avoid claim: {report['paper_wording']['avoid_claim']}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(
        _load(args.single_plan_report),
        _load(args.multi_plan_report),
        single_path=args.single_plan_report,
        multi_path=args.multi_plan_report,
    )
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.save_md.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({
        "save_json": str(args.save_json),
        "save_md": str(args.save_md),
        "plan_count_delta": report["comparison"]["plan_count_delta"],
        "accepted_plan_count_delta": report["comparison"]["accepted_plan_count_delta"],
        "best_sandbox_score_delta": report["comparison"]["best_sandbox_score_delta"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
