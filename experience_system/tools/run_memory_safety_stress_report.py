"""Generate a safety-focused stress report for memory and sandbox modules."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import (
    ExperienceLibrary,
    apply_stage_score_adjustment,
    load_lesson_library,
    load_policy_risk_calibration,
    run_stage_retrieval,
    summarize_stage_retrieval,
)
from source.candidate_sandbox import evaluate_candidate_in_sandbox, fuse_memory_and_sandbox, select_sandbox_calibration, summarize_sandbox_fusion
from source.run_r1pro_memory_policy_smoke import (
    adjust_candidate_with_lessons,
    candidates_for_scenario,
    evaluate_candidate,
    object_class_for_scenario,
    sandbox_selection_rank,
    select_candidate,
    select_sandbox_candidate,
    selection_rank,
)


STRESS_VARIANTS = {
    "memory_only": {
        "label": "memory only",
        "use_lessons": False,
        "use_stage": False,
        "use_sandbox": False,
    },
    "memory_stage": {
        "label": "memory + stage retrieval",
        "use_lessons": False,
        "use_stage": True,
        "use_sandbox": False,
    },
    "memory_sandbox_critic": {
        "label": "memory + sandbox critic",
        "use_lessons": False,
        "use_stage": False,
        "use_sandbox": True,
    },
    "full_stage_lesson_sandbox": {
        "label": "memory + lessons + stage + sandbox critic",
        "use_lessons": True,
        "use_stage": True,
        "use_sandbox": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a safety stress report for risky candidate selection.")
    parser.add_argument("--scenario", choices=["G3"], default="G3")
    parser.add_argument("--condition", choices=["clean", "place_occupied"], default="place_occupied")
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--policy-calibration", type=Path, default=None)
    parser.add_argument("--lesson-lib", type=Path, default=None)
    parser.add_argument("--variant", choices=sorted(STRESS_VARIANTS), action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--execute-on", choices=["accept", "review", "always"], default="review")
    parser.add_argument("--sandbox-weight", type=float, default=0.45)
    parser.add_argument("--lesson-weight", type=float, default=0.08)
    parser.add_argument("--stage-top-k", type=int, default=None)
    parser.add_argument("--stage-support-weight", type=float, default=0.08)
    parser.add_argument("--stage-risk-weight", type=float, default=0.12)
    parser.add_argument("--use-sandbox-calibration", action="store_true")
    parser.add_argument("--risky-candidate-id", action="append", default=["g3_place_first"])
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def _score(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    return candidate.get("fused_score") or candidate.get("candidate_score") or {}


def _score_value(candidate: dict[str, Any] | None) -> float:
    score = _score(candidate)
    return float(score.get("combined_score", score.get("candidate_score", 0.0)) or 0.0)


def _decision(candidate: dict[str, Any] | None) -> str:
    return str(_score(candidate).get("decision") or "")


def _candidate_summary(candidate: dict[str, Any] | None, risky_ids: set[str]) -> dict[str, Any]:
    if not candidate:
        return {}
    score = _score(candidate)
    sandbox = candidate.get("sandbox") or {}
    stage = candidate.get("stage_retrieval") or {}
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "is_risky": str(candidate.get("candidate_id") or "") in risky_ids,
        "decision": score.get("decision", ""),
        "score": round(_score_value(candidate), 4),
        "memory_score": score.get("memory_score", (candidate.get("candidate_score") or {}).get("candidate_score", 0.0)),
        "risk_score": (candidate.get("candidate_score") or {}).get("risk_score", score.get("memory_risk_score", 0.0)),
        "stage_support_score": stage.get("stage_support_score", 0.0),
        "stage_risk_score": stage.get("stage_risk_score", 0.0),
        "sandbox_score": sandbox.get("sandbox_score", None),
        "critic_status": sandbox.get("critic_status", ""),
        "critic_risk_score": sandbox.get("critic_risk_score", 0.0),
        "calibration_risk_penalty": sandbox.get("calibration_risk_penalty", 0.0),
    }


def _rank_of(candidates: list[dict[str, Any]], candidate_id: str) -> int | None:
    for index, candidate in enumerate(candidates, 1):
        if str(candidate.get("candidate_id") or "") == candidate_id:
            return index
    return None


def _warn_or_block(candidate: dict[str, Any] | None) -> bool:
    status = str(((candidate or {}).get("sandbox") or {}).get("critic_status") or "")
    return status in {"warn", "block"}


def _evaluate_variant(
    base_library: ExperienceLibrary,
    *,
    variant_id: str,
    scenario: str,
    condition: str,
    top_k: int,
    execute_on: str,
    control_mode: str,
    policy_calibration: dict[str, Any] | None,
    lessons: list[dict[str, Any]],
    lesson_weight: float,
    sandbox_weight: float,
    use_sandbox_calibration: bool,
    stage_top_k: int | None,
    stage_support_weight: float,
    stage_risk_weight: float,
    risky_ids: set[str],
) -> dict[str, Any]:
    variant = STRESS_VARIANTS[variant_id]
    library = ExperienceLibrary(deepcopy(base_library.entries))
    object_class = object_class_for_scenario(scenario)
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=scenario,
        condition=condition,
        object_class=object_class,
    ) if use_sandbox_calibration else None
    candidates = [
        evaluate_candidate(
            library,
            candidate,
            scenario=scenario,
            condition=condition,
            object_class=object_class,
            top_k=top_k,
            risk_aware=True,
            policy_calibration=policy_calibration,
        )
        for candidate in candidates_for_scenario(scenario, include_risky=True)
    ]
    if variant["use_lessons"] and lessons:
        for report in candidates:
            adjust_candidate_with_lessons(
                report,
                lessons,
                scenario=scenario,
                condition=condition,
                lesson_weight=lesson_weight,
            )
    if variant["use_stage"]:
        for report in candidates:
            stage_report = run_stage_retrieval(
                library,
                scenario=scenario,
                condition=condition,
                object_class=object_class,
                candidate_id=str(report["candidate_id"]),
                candidate_steps=list(report["candidate_steps"]),
                top_k=stage_top_k,
            )
            apply_stage_score_adjustment(
                report,
                stage_report,
                support_weight=stage_support_weight,
                risk_weight=stage_risk_weight,
            )
    ranked_before_sandbox = sorted(candidates, key=selection_rank, reverse=True)
    selected_before_sandbox = select_candidate(candidates, execute_on) or ranked_before_sandbox[0]
    selected = selected_before_sandbox
    sandbox_summary: dict[str, Any] = {}
    if variant["use_sandbox"]:
        for report in candidates:
            sandbox = evaluate_candidate_in_sandbox(
                scenario=scenario,
                condition=condition,
                candidate_id=str(report["candidate_id"]),
                control_mode=control_mode,
                sandbox_calibration=sandbox_calibration,
            )
            report["sandbox"] = sandbox
            report["fused_score"] = fuse_memory_and_sandbox(report, sandbox, sandbox_weight=sandbox_weight)
        candidates = sorted(candidates, key=sandbox_selection_rank, reverse=True)
        selected = select_sandbox_candidate(candidates, execute_on) or candidates[0]
        sandbox_summary = summarize_sandbox_fusion(candidates)
    else:
        candidates = ranked_before_sandbox

    risky_candidates = [item for item in candidates if str(item.get("candidate_id") or "") in risky_ids]
    risky_selected = str(selected.get("candidate_id") or "") in risky_ids
    safe_selected = bool(selected and not risky_selected)
    warn_or_block_candidates = [item for item in candidates if _warn_or_block(item)]
    risky_warn_or_block = [item for item in risky_candidates if _warn_or_block(item)]
    best_risky_rank = min((_rank_of(candidates, rid) for rid in risky_ids if _rank_of(candidates, rid) is not None), default=None)
    return {
        "variant_id": variant_id,
        "variant_label": variant["label"],
        "scenario": scenario,
        "condition": condition,
        "selected_candidate_id": str(selected.get("candidate_id") or ""),
        "selected_before_sandbox_id": str(selected_before_sandbox.get("candidate_id") or ""),
        "risky_candidate_ids": sorted(risky_ids),
        "risky_candidate_selected": risky_selected,
        "safe_candidate_selected": safe_selected,
        "best_risky_rank": best_risky_rank,
        "selected_warn_or_block": _warn_or_block(selected),
        "warn_or_block_candidate_count": len(warn_or_block_candidates),
        "warn_or_block_candidate_rate": round(len(warn_or_block_candidates) / len(candidates), 4) if candidates else 0.0,
        "risky_warn_or_block_rate": round(len(risky_warn_or_block) / len(risky_candidates), 4) if risky_candidates else 0.0,
        "candidate_changed_by_sandbox": bool(
            variant["use_sandbox"]
            and selected_before_sandbox.get("candidate_id") != selected.get("candidate_id")
        ),
        "score_margin_selected_vs_best_risky": round(
            _score_value(selected) - max((_score_value(item) for item in risky_candidates), default=0.0),
            4,
        ),
        "stage_retrieval_enabled": bool(variant["use_stage"]),
        "stage_retrieval_summary": summarize_stage_retrieval(candidates) if variant["use_stage"] else {},
        "sandbox_enabled": bool(variant["use_sandbox"]),
        "sandbox_summary": sandbox_summary,
        "sandbox_calibration_enabled": use_sandbox_calibration,
        "sandbox_calibration": sandbox_calibration or {},
        "selected": _candidate_summary(selected, risky_ids),
        "selected_before_sandbox": _candidate_summary(selected_before_sandbox, risky_ids),
        "candidate_ranking": [_candidate_summary(item, risky_ids) for item in candidates],
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for variant_id in [item for item in STRESS_VARIANTS if any(run["variant_id"] == item for run in runs)]:
        group = [run for run in runs if run["variant_id"] == variant_id]
        summaries.append({
            "variant_id": variant_id,
            "variant_label": STRESS_VARIANTS[variant_id]["label"],
            "run_count": len(group),
            "risky_candidate_selected_rate": round(sum(1 for item in group if item["risky_candidate_selected"]) / len(group), 4),
            "safe_candidate_selected_rate": round(sum(1 for item in group if item["safe_candidate_selected"]) / len(group), 4),
            "selected_warn_or_block_rate": round(sum(1 for item in group if item["selected_warn_or_block"]) / len(group), 4),
            "warn_or_block_candidate_rate": round(mean(float(item["warn_or_block_candidate_rate"]) for item in group), 4),
            "risky_warn_or_block_rate": round(mean(float(item["risky_warn_or_block_rate"]) for item in group), 4),
            "candidate_changed_by_sandbox_rate": round(sum(1 for item in group if item["candidate_changed_by_sandbox"]) / len(group), 4),
            "score_margin_selected_vs_best_risky_avg": round(mean(float(item["score_margin_selected_vs_best_risky"]) for item in group), 4),
            "stage_specificity_score_avg": round(mean(float((item.get("stage_retrieval_summary") or {}).get("stage_specificity_score") or 0.0) for item in group), 4),
            "critic_warn_rate_avg": round(mean(float((item.get("sandbox_summary") or {}).get("critic_warn_rate") or 0.0) for item in group), 4),
            "critic_block_rate_avg": round(mean(float((item.get("sandbox_summary") or {}).get("critic_block_rate") or 0.0) for item in group), 4),
        })
    return summaries


def _csv_rows(runs: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_by_variant = {item["variant_id"]: item for item in summaries}
    rows = []
    for run in runs:
        summary = summary_by_variant[run["variant_id"]]
        rows.append({
            "variant_id": run["variant_id"],
            "variant_label": run["variant_label"],
            "scenario": run["scenario"],
            "condition": run["condition"],
            "selected_candidate_id": run["selected_candidate_id"],
            "selected_before_sandbox_id": run["selected_before_sandbox_id"],
            "risky_candidate_selected": run["risky_candidate_selected"],
            "safe_candidate_selected": run["safe_candidate_selected"],
            "selected_warn_or_block": run["selected_warn_or_block"],
            "warn_or_block_candidate_rate": run["warn_or_block_candidate_rate"],
            "risky_warn_or_block_rate": run["risky_warn_or_block_rate"],
            "candidate_changed_by_sandbox": run["candidate_changed_by_sandbox"],
            "score_margin_selected_vs_best_risky": run["score_margin_selected_vs_best_risky"],
            "summary_risky_candidate_selected_rate": summary["risky_candidate_selected_rate"],
            "summary_safe_candidate_selected_rate": summary["safe_candidate_selected_rate"],
            "summary_critic_warn_rate_avg": summary["critic_warn_rate_avg"],
            "summary_stage_specificity_score_avg": summary["stage_specificity_score_avg"],
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_library = ExperienceLibrary.load(args.universal_experience_lib)
    policy_calibration = load_policy_risk_calibration(args.policy_calibration) if args.policy_calibration else None
    lessons = load_lesson_library(args.lesson_lib) if args.lesson_lib else []
    variants = args.variant or list(STRESS_VARIANTS)
    risky_ids = {str(item) for item in args.risky_candidate_id if str(item)}
    runs = [
        _evaluate_variant(
            base_library,
            variant_id=variant_id,
            scenario=args.scenario,
            condition=args.condition,
            top_k=args.top_k,
            execute_on=args.execute_on,
            control_mode=args.control_mode,
            policy_calibration=policy_calibration,
            lessons=lessons,
            lesson_weight=args.lesson_weight,
            sandbox_weight=args.sandbox_weight,
            use_sandbox_calibration=args.use_sandbox_calibration,
            stage_top_k=args.stage_top_k,
            stage_support_weight=args.stage_support_weight,
            stage_risk_weight=args.stage_risk_weight,
            risky_ids=risky_ids,
        )
        for variant_id in variants
    ]
    summaries = _summarize_runs(runs)
    report = {
        "schema_version": "memory_safety_stress_report_v1",
        "scenario": args.scenario,
        "condition": args.condition,
        "experience_library": str(args.universal_experience_lib),
        "policy_calibration": str(args.policy_calibration) if args.policy_calibration else "",
        "lesson_lib": str(args.lesson_lib) if args.lesson_lib else "",
        "risky_candidate_ids": sorted(risky_ids),
        "sandbox_weight": args.sandbox_weight,
        "stage_support_weight": args.stage_support_weight,
        "stage_risk_weight": args.stage_risk_weight,
        "summary": summaries,
        "runs": runs,
    }
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_csv is not None:
        _write_csv(args.save_csv, _csv_rows(runs, summaries))
    print(json.dumps({
        "scenario": args.scenario,
        "condition": args.condition,
        "variant_count": len(variants),
        "summary": summaries,
        "save": str(args.save) if args.save else "",
        "save_csv": str(args.save_csv) if args.save_csv else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
