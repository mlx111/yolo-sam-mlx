"""Run ablation variants for the R1Pro experience-memory policy."""

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

from experience_core import ExperienceLibrary, load_lesson_library, load_policy_risk_calibration, retrieval_count
from source.candidate_sandbox import evaluate_candidate_in_sandbox, fuse_memory_and_sandbox, select_sandbox_calibration, summarize_sandbox_fusion
from source.run_r1pro_memory_policy_smoke import (
    adjust_candidate_with_lessons,
    candidates_for_scenario,
    evaluate_candidate,
    load_visual_scores,
    object_class_for_scenario,
    sandbox_selection_rank,
    select_candidate,
    select_sandbox_candidate,
    selection_rank,
)


VARIANTS = {
    "baseline_no_memory": {
        "label": "baseline/no memory",
        "use_memory": False,
        "use_lifecycle": False,
        "use_visual": False,
        "use_lessons": False,
        "use_sandbox": False,
    },
    "memory_only": {
        "label": "memory only",
        "use_memory": True,
        "use_lifecycle": False,
        "use_visual": False,
        "use_lessons": False,
        "use_sandbox": False,
    },
    "memory_lifecycle": {
        "label": "memory + lifecycle",
        "use_memory": True,
        "use_lifecycle": True,
        "use_visual": False,
        "use_lessons": False,
        "use_sandbox": False,
    },
    "full_visual_sandbox_critic": {
        "label": "memory + lifecycle + visual + sandbox critic",
        "use_memory": True,
        "use_lifecycle": True,
        "use_visual": True,
        "use_lessons": True,
        "use_sandbox": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ablation report for the experience-memory policy.")
    parser.add_argument("--config", type=Path, default=None, help="optional JSON config for reproducible ablation runs")
    parser.add_argument("--scenario", choices=["G3"], action="append", default=[])
    parser.add_argument("--condition", choices=["clean", "place_occupied"], action="append", default=[])
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append", default=[])
    parser.add_argument("--universal-experience-lib", type=Path, default=None)
    parser.add_argument("--policy-calibration", type=Path, default=None)
    parser.add_argument("--lesson-lib", type=Path, default=None)
    parser.add_argument("--visual-index-dir", type=Path, default=None)
    parser.add_argument("--query-image", type=Path, action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--execute-on", choices=["accept", "review", "always"], default="review")
    parser.add_argument("--control-mode", choices=["ideal", "physical"], default="physical")
    parser.add_argument("--sandbox-weight", type=float, default=0.45)
    parser.add_argument("--lesson-weight", type=float, default=0.08)
    parser.add_argument("--visual-top-k", type=int, default=10)
    parser.add_argument("--visual-weight", type=float, default=0.12)
    parser.add_argument("--include-risky-candidates", action="store_true")
    parser.add_argument("--use-sandbox-calibration", action="store_true")
    parser.add_argument("--execute-selected", action="store_true", help="execute the final selected candidate for each run to populate success_rate")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"ablation config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ablation config must be a JSON object: {path}")
    return payload


def _path_arg(args: argparse.Namespace, config: dict[str, Any], name: str) -> Path | None:
    value = getattr(args, name)
    if value is not None:
        return value
    raw = config.get(name.replace("_", "-"), config.get(name))
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT.parent / path
    return path


def _list_arg(args: argparse.Namespace, config: dict[str, Any], name: str, default: list[str], *, config_name: str | None = None) -> list[str]:
    value = getattr(args, name)
    if value:
        return list(value)
    raw = config.get(config_name or name, config.get(name, default))
    return [str(item) for item in raw]


def _scalar_arg(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if name in config:
        return config[name]
    return value if value is not None else default


def _bool_arg(args: argparse.Namespace, config: dict[str, Any], name: str, default: bool = False) -> bool:
    return bool(getattr(args, name) or config.get(name, default))


def _default_candidate_id(scenario: str) -> str:
    return f"{scenario.lower()}_default"


def _score(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    return candidate.get("fused_score") or candidate.get("candidate_score") or {}


def _score_value(candidate: dict[str, Any] | None) -> float:
    score = _score(candidate)
    return float(score.get("combined_score", score.get("candidate_score", 0.0)) or 0.0)


def _risk_value(candidate: dict[str, Any] | None) -> float:
    score = candidate.get("candidate_score", {}) if candidate else {}
    return float(score.get("risk_score", score.get("memory_risk_score", 0.0)) or 0.0)


def _decision(candidate: dict[str, Any] | None) -> str:
    return str(_score(candidate).get("decision") or "")


def _candidate_summary(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    score = _score(candidate)
    retrieval = candidate.get("retrieval") or {}
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "decision": score.get("decision", ""),
        "score": round(_score_value(candidate), 4),
        "risk_score": round(_risk_value(candidate), 4),
        "match_count": int(retrieval.get("match_count") or 0),
        "support_match_count": int(retrieval.get("support_match_count") or 0),
        "risk_match_count": int(retrieval.get("risk_match_count") or 0),
        "executable": bool(candidate.get("executable", False)),
    }


def _memory_snapshot(library: ExperienceLibrary) -> dict[str, Any]:
    counts = [retrieval_count(entry) for entry in library.entries]
    tiers: dict[str, int] = {}
    partitions: dict[str, int] = {}
    for entry in library.entries:
        lifecycle = entry.metadata.get("memory_lifecycle") if isinstance(entry.metadata, dict) else {}
        tier = str((lifecycle or {}).get("memory_tier") or "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
        partition = str(entry.memory_partition or "unknown")
        partitions[partition] = partitions.get(partition, 0) + 1
    return {
        "entry_count": len(library.entries),
        "retrieval_count_distribution": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(mean(counts), 4) if counts else 0.0,
            "nonzero_count": sum(1 for item in counts if item > 0),
        },
        "memory_tier_counts": dict(sorted(tiers.items())),
        "memory_partition_counts": dict(sorted(partitions.items())),
    }


def _repeated_failure_rate(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    risks = (candidate.get("candidate_score") or {}).get("top_failure_risks") or []
    if not risks:
        return 0.0
    repeated = [
        item for item in risks
        if float(item.get("failed_action_overlap", 0.0) or 0.0) >= 0.8
        and float(item.get("terminal_risk_score", 0.0) or 0.0) >= 0.35
    ]
    return round(len(repeated) / len(risks), 4)


def _evaluate_baseline(
    *,
    scenario: str,
    condition: str,
) -> dict[str, Any]:
    default_id = _default_candidate_id(scenario)
    default_plan = next(candidate for candidate in candidates_for_scenario(scenario) if candidate.candidate_id == default_id)
    return {
        "candidate_id": default_plan.candidate_id,
        "description": default_plan.description,
        "executable": default_plan.executable,
        "candidate_steps": list(default_plan.steps),
        "retrieval": {
            "support_match_count": 0,
            "risk_match_count": 0,
            "match_count": 0,
            "matches": [],
        },
        "candidate_score": {
            "decision": "accept",
            "candidate_score": 0.5,
            "risk_score": 0.5,
            "support_score": 0.0,
            "risk_penalty": 0.0,
            "top_failure_risks": [],
            "failure_risk_penalty": 0.0,
        },
    }


def _evaluate_variant_run(
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
    visual_scores: dict[str, float],
    visual_weight: float,
    sandbox_weight: float,
    lesson_weight: float,
    include_risky_candidates: bool,
    execute_selected: bool,
    use_sandbox_calibration: bool,
) -> dict[str, Any]:
    variant = VARIANTS[variant_id]
    library = ExperienceLibrary(deepcopy(base_library.entries))
    before_snapshot = _memory_snapshot(library)
    object_class = object_class_for_scenario(scenario)
    sandbox_calibration = select_sandbox_calibration(
        library,
        scenario=scenario,
        condition=condition,
        object_class=object_class,
    ) if use_sandbox_calibration else None

    if not variant["use_memory"]:
        selected = _evaluate_baseline(
            scenario=scenario,
            condition=condition,
        )
        candidates = [selected]
        baseline = selected
        selected_before_sandbox = selected
        sandbox_summary = {}
    else:
        active_visual_scores = visual_scores if variant["use_visual"] else {}
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
                visual_scores=active_visual_scores,
                visual_weight=visual_weight,
            )
            for candidate in candidates_for_scenario(scenario, include_risky=include_risky_candidates)
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
        baseline = next(item for item in candidates if item["candidate_id"] == _default_candidate_id(scenario))
        selected_before_sandbox = select_candidate(candidates, execute_on) or max(candidates, key=selection_rank)
        selected = selected_before_sandbox
        sandbox_summary = {}
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
            candidates = sorted(candidates, key=selection_rank, reverse=True)

    after_snapshot = _memory_snapshot(library)
    selected_id = str(selected.get("candidate_id", "")) if selected else ""
    default_id = _default_candidate_id(scenario)
    risk_delta = round(_risk_value(selected) - _risk_value(baseline), 4)
    score_delta = round(_score_value(selected) - _score_value(baseline), 4)
    sandbox = selected.get("sandbox") if selected else {}
    execution_success = None
    if sandbox:
        execution_success = bool(sandbox.get("task_success", sandbox.get("success", False)))
    execution_report: dict[str, Any] | None = None
    if execute_selected and selected:
        if sandbox:
            execution_report = sandbox
        else:
            execution_report = evaluate_candidate_in_sandbox(
                scenario=scenario,
                condition=condition,
                candidate_id=str(selected["candidate_id"]),
                control_mode=control_mode,
                sandbox_calibration=sandbox_calibration,
            )
            execution_success = bool(execution_report.get("task_success", execution_report.get("success", False)))
    critic_block_rate = float(sandbox_summary.get("critic_block_rate", 0.0) or 0.0) if sandbox_summary else 0.0
    critic_warn_rate = float(sandbox_summary.get("critic_warn_rate", 0.0) or 0.0) if sandbox_summary else 0.0

    return {
        "variant_id": variant_id,
        "variant_label": variant["label"],
        "scenario": scenario,
        "condition": condition,
        "selected_candidate_id": selected_id,
        "baseline_candidate_id": default_id,
        "candidate_changed": bool(selected_id and selected_id != default_id),
        "candidate_changed_by_sandbox": bool(
            variant["use_sandbox"]
            and selected_before_sandbox
            and selected
            and selected_before_sandbox.get("candidate_id") != selected.get("candidate_id")
        ),
        "success": execution_success,
        "executed_selected": bool(execute_selected and selected),
        "execution_report": execution_report,
        "sandbox_calibration_enabled": use_sandbox_calibration,
        "sandbox_calibration": sandbox_calibration or {},
        "selected_calibration_risk_penalty": float((sandbox or execution_report or {}).get("calibration_risk_penalty", 0.0) or 0.0),
        "score_delta": score_delta,
        "risk_score_delta": risk_delta,
        "critic_status": str((sandbox or {}).get("critic_status") or ""),
        "critic_risk_score": float((sandbox or {}).get("critic_risk_score") or 0.0),
        "critic_block_rate": round(critic_block_rate, 4),
        "critic_warn_rate": round(critic_warn_rate, 4),
        "repeated_failure_rate": _repeated_failure_rate(selected),
        "memory_write_count": 0,
        "retrieval_count_delta": round(
            after_snapshot["retrieval_count_distribution"]["mean"]
            - before_snapshot["retrieval_count_distribution"]["mean"],
            4,
        ),
        "selected_before_sandbox": _candidate_summary(selected_before_sandbox),
        "selected": _candidate_summary(selected),
        "baseline": _candidate_summary(baseline),
        "sandbox_summary": sandbox_summary,
        "memory_before": before_snapshot if variant["use_lifecycle"] else {},
        "memory_after": after_snapshot if variant["use_lifecycle"] else {},
        "candidate_ranking": [_candidate_summary(item) for item in candidates],
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for variant_id in sorted({run["variant_id"] for run in runs}):
        group = [run for run in runs if run["variant_id"] == variant_id]
        successes = [run["success"] for run in group if run["success"] is not None]
        summaries.append({
            "variant_id": variant_id,
            "variant_label": VARIANTS[variant_id]["label"],
            "run_count": len(group),
            "success_rate": round(sum(1 for item in successes if item) / len(successes), 4) if successes else None,
            "candidate_changed_rate": round(sum(1 for item in group if item["candidate_changed"]) / len(group), 4),
            "candidate_changed_by_sandbox_rate": round(sum(1 for item in group if item["candidate_changed_by_sandbox"]) / len(group), 4),
            "risk_score_delta_avg": round(mean(float(item["risk_score_delta"]) for item in group), 4),
            "score_delta_avg": round(mean(float(item["score_delta"]) for item in group), 4),
            "critic_block_rate": round(mean(float(item["critic_block_rate"]) for item in group), 4),
            "critic_warn_rate": round(mean(float(item["critic_warn_rate"]) for item in group), 4),
            "repeated_failure_rate_avg": round(mean(float(item["repeated_failure_rate"]) for item in group), 4),
            "memory_write_count": sum(int(item["memory_write_count"]) for item in group),
            "retrieval_count_delta_avg": round(mean(float(item["retrieval_count_delta"]) for item in group), 4),
        })
    return summaries


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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
            "candidate_changed": run["candidate_changed"],
            "candidate_changed_by_sandbox": run["candidate_changed_by_sandbox"],
            "success": run["success"],
            "executed_selected": run["executed_selected"],
            "score_delta": run["score_delta"],
            "risk_score_delta": run["risk_score_delta"],
            "critic_status": run["critic_status"],
            "critic_risk_score": run["critic_risk_score"],
            "critic_block_rate": run["critic_block_rate"],
            "critic_warn_rate": run["critic_warn_rate"],
            "sandbox_calibration_enabled": run["sandbox_calibration_enabled"],
            "selected_calibration_risk_penalty": run["selected_calibration_risk_penalty"],
            "repeated_failure_rate": run["repeated_failure_rate"],
            "retrieval_count_delta": run["retrieval_count_delta"],
            "summary_success_rate": summary["success_rate"],
            "summary_candidate_changed_rate": summary["candidate_changed_rate"],
            "summary_critic_block_rate": summary["critic_block_rate"],
            "summary_repeated_failure_rate_avg": summary["repeated_failure_rate_avg"],
        })
    return rows


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    universal_experience_lib = _path_arg(args, config, "universal_experience_lib")
    if universal_experience_lib is None:
        raise ValueError("--universal-experience-lib is required unless provided by --config")
    policy_calibration_path = _path_arg(args, config, "policy_calibration")
    lesson_lib = _path_arg(args, config, "lesson_lib")
    visual_index_dir = _path_arg(args, config, "visual_index_dir")
    save = _path_arg(args, config, "save")
    save_csv = _path_arg(args, config, "save_csv")
    query_images = [Path(item) for item in (args.query_image or config.get("query_images") or config.get("query_image") or [])]

    top_k = int(config.get("top_k", args.top_k))
    execute_on = str(config.get("execute_on", args.execute_on))
    control_mode = str(config.get("control_mode", args.control_mode))
    sandbox_weight = float(config.get("sandbox_weight", args.sandbox_weight))
    lesson_weight = float(config.get("lesson_weight", args.lesson_weight))
    visual_top_k = int(config.get("visual_top_k", args.visual_top_k))
    visual_weight = float(config.get("visual_weight", args.visual_weight))
    include_risky_candidates = _bool_arg(args, config, "include_risky_candidates")
    use_sandbox_calibration = _bool_arg(args, config, "use_sandbox_calibration")
    execute_selected = _bool_arg(args, config, "execute_selected")

    base_library = ExperienceLibrary.load(universal_experience_lib)
    policy_calibration = load_policy_risk_calibration(policy_calibration_path) if policy_calibration_path else None
    lessons = load_lesson_library(lesson_lib) if lesson_lib else []
    visual_scores = load_visual_scores(visual_index_dir, query_images, top_k=visual_top_k)
    scenarios = _list_arg(args, config, "scenario", ["G3"], config_name="scenarios")
    conditions = _list_arg(args, config, "condition", ["clean", "place_occupied"], config_name="conditions")
    variants = _list_arg(args, config, "variant", list(VARIANTS), config_name="variants")

    runs = [
        _evaluate_variant_run(
            base_library,
            variant_id=variant_id,
            scenario=scenario,
            condition=condition,
            top_k=top_k,
            execute_on=execute_on,
            control_mode=control_mode,
            policy_calibration=policy_calibration,
            lessons=lessons,
            visual_scores=visual_scores,
            visual_weight=visual_weight,
            sandbox_weight=sandbox_weight,
            lesson_weight=lesson_weight,
            include_risky_candidates=include_risky_candidates,
            execute_selected=execute_selected,
            use_sandbox_calibration=use_sandbox_calibration,
        )
        for variant_id in variants
        for scenario in scenarios
        for condition in conditions
    ]
    summaries = _summarize_runs(runs)
    csv_rows = _csv_rows(runs, summaries)
    report = {
        "schema_version": "memory_ablation_report_v1",
        "config": str(args.config) if args.config else "",
        "experience_library": str(universal_experience_lib),
        "policy_calibration": str(policy_calibration_path) if policy_calibration_path else "",
        "lesson_lib": str(lesson_lib) if lesson_lib else "",
        "visual_index_dir": str(visual_index_dir) if visual_index_dir else "",
        "query_images": [str(path) for path in query_images],
        "execute_selected": execute_selected,
        "use_sandbox_calibration": use_sandbox_calibration,
        "scenario_count": len(scenarios),
        "condition_count": len(conditions),
        "variant_count": len(variants),
        "run_count": len(runs),
        "summaries": summaries,
        "runs": runs,
        "csv_rows": csv_rows,
    }
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        save_csv.write_text(_csv_text(csv_rows), encoding="utf-8")
    print(json.dumps({
        "run_count": len(runs),
        "summaries": summaries,
        "save": str(save) if save else "",
        "save_csv": str(save_csv) if save_csv else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
