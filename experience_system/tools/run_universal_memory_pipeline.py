"""Run the universal experience-memory build and smoke pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import R1ProMujocoAdapter, RealEpisodeAdapter, Wrapper1UR5eAdapter
from experience_core import (
    ExperienceLibrary,
    TextSemanticRetrievalIndex,
    apply_stage_planner_guidance,
    apply_critic,
    apply_pair_and_gap,
    apply_sandbox_calibration,
    build_stage_planner_context,
    build_policy_risk_calibration,
    consolidate_experiences,
    consolidate_memory_lifecycle,
    compute_group_calibrations,
    pair_sim_real_experiences,
    run_stage_retrieval,
    semantic_query_text,
    summarize_stage_planner_contexts,
    summarize_stage_retrieval,
    validate_experience_library,
)
from source.import_wrapper1_ur5e_memory import _load_entries as load_wrapper1_entries
from source.build_visual_keyframe_index import build_visual_index
from source.compare_policy_baseline import build_policy_comparison
from source.evaluate_visual_retrieval import evaluate_visual_retrieval
from source.run_r1pro_memory_policy_smoke import evaluate_candidate, candidates_for_scenario, load_visual_scores, object_class_for_scenario, select_candidate, selection_rank
from source.run_r1pro_task_chain import run_task_chain
from source.summarize_universal_experience import build_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run universal Sim-Real experience-memory pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-after", choices=["build", "pair", "critic", "calibration", "policy_calibration", "consolidation", "summary", "visual_index", "policy_smoke"], default="")
    parser.add_argument("--strict-quality", action="store_true", help="fail if final library has quality errors")
    parser.add_argument("--check-refs", action="store_true", help="check raw/keyframe refs during quality validation")
    return parser.parse_args()


def _path(config: dict[str, Any], key: str, default: str) -> Path:
    value = config.get(key) or default
    if not value:
        return Path()
    path = Path(str(value))
    if path.is_absolute():
        return path
    root_path = ROOT / path
    cwd_path = Path.cwd() / path
    if root_path.exists() or not cwd_path.exists():
        return root_path
    return cwd_path


def _output_path(config: dict[str, Any], key: str, default: str) -> Path:
    value = config.get(key) or default
    if not value:
        return Path()
    path = Path(str(value))
    return path if path.is_absolute() else Path.cwd() / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pipeline config must be a JSON object")
    return payload


def _run_r1pro_tasks(library: ExperienceLibrary, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapter = R1ProMujocoAdapter()
    report = []
    for task in tasks:
        scenario = str(task.get("scenario") or task.get("scenario_id") or "")
        condition = str(task.get("condition") or task.get("condition_id") or "")
        control_mode = str(task.get("control_mode") or "physical")
        candidate_id = str(task.get("candidate_id") or "")
        repeat = int(task.get("repeat", 1))
        keyframe_root = task.get("keyframe_dir")
        for index in range(max(repeat, 1)):
            keyframe_dir = None
            if keyframe_root:
                keyframe_dir = _output_path(task, "keyframe_dir", "") / f"{scenario}_{condition}_{candidate_id or 'default'}_{index + 1}"
            result = run_task_chain(scenario, condition, control_mode, candidate_id=candidate_id, keyframe_dir=keyframe_dir)
            entry = adapter.normalize_episode(result)
            write_policy = library.add_with_policy(entry)
            report.append({
                "scenario": scenario,
                "condition": condition,
                "control_mode": control_mode,
                "candidate_id": candidate_id,
                "success": result.success,
                "task_success": result.task_success,
                "experience_id": entry.experience_id,
                "stored_experience_id": write_policy.get("stored_experience_id", ""),
                "keyframe_count": len(entry.keyframes),
                "keyframe_dir": str(keyframe_dir) if keyframe_dir is not None else "",
                "write_policy": write_policy,
            })
    return report


def _run_wrapper1_imports(library: ExperienceLibrary, imports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for item in imports:
        if not item.get("input"):
            continue
        source = _path(item, "input", "")
        raw_entries = load_wrapper1_entries(source)
        limit = int(item.get("limit", 0))
        if limit > 0:
            raw_entries = raw_entries[:limit]
        adapter = Wrapper1UR5eAdapter(robot_id=str(item.get("robot_id") or "ur5e_wrapper1_mujoco"))
        for raw in raw_entries:
            entry = adapter.normalize_entry(raw)
            write_policy = library.add_with_policy(entry)
            report.append({
                "input": str(source),
                "source_experience_id": raw.get("experience_id", ""),
                "experience_id": entry.experience_id,
                "stored_experience_id": write_policy.get("stored_experience_id", ""),
                "scenario_id": entry.scenario_id,
                "condition_id": entry.condition_id,
                "success": entry.result.get("success", False),
                "write_policy": write_policy,
            })
    return report


def _run_real_imports(library: ExperienceLibrary, imports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for item in imports:
        adapter = RealEpisodeAdapter(default_backend=str(item.get("backend") or "real_robot"))
        source_value = str(item.get("source") or "real")
        if item.get("batch_dir"):
            raw_episodes = adapter.collect_batch_sources(_path(item, "batch_dir", ""))
        elif item.get("episode_dir"):
            raw_episodes = [adapter.collect_episode_dir(_path(item, "episode_dir", ""))]
        elif item.get("input"):
            raw_episodes = [adapter.load_episode(_path(item, "input", ""))]
        else:
            continue
        for raw in raw_episodes:
            entry = adapter.normalize_episode(raw, source=source_value)
            write_policy = library.add_with_policy(entry)
            report.append({
                "experience_id": entry.experience_id,
                "stored_experience_id": write_policy.get("stored_experience_id", ""),
                "source": entry.source,
                "backend": entry.backend,
                "scenario_id": entry.scenario_id,
                "condition_id": entry.condition_id,
                "success": entry.result.get("success", False),
                "write_policy": write_policy,
            })
    return report


def _summarize_write_policy(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "decision_counts": {},
        "reason_counts": {},
        "total_candidates": 0,
        "stored_or_merged_count": 0,
    }
    for step_name in ("r1pro_tasks", "wrapper1_imports", "real_episode_imports"):
        for item in report.get("steps", {}).get(step_name, []):
            policy = item.get("write_policy") if isinstance(item, dict) else None
            if not isinstance(policy, dict):
                continue
            decision = str(policy.get("decision") or "unknown")
            reason = str(policy.get("reason") or "unknown")
            summary["decision_counts"][decision] = summary["decision_counts"].get(decision, 0) + 1
            summary["reason_counts"][reason] = summary["reason_counts"].get(reason, 0) + 1
            summary["total_candidates"] += 1
            if decision in {"write", "merge"}:
                summary["stored_or_merged_count"] += 1
    return summary


def _apply_critic(library: ExperienceLibrary, thresholds: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "min_object_lift": float(thresholds.get("min_object_lift", 0.05)),
        "max_place_xy_error": float(thresholds.get("max_place_xy_error", 0.05)),
        "max_place_z_error": float(thresholds.get("max_place_z_error", 0.08)),
        "max_dual_arm_height_mismatch": float(thresholds.get("max_dual_arm_height_mismatch", 0.02)),
        "high_sim_real_gap": float(thresholds.get("high_sim_real_gap", 0.65)),
    }
    summary = {"pass": 0, "warn": 0, "block": 0, "unknown": 0}
    for entry in library.entries:
        apply_critic(entry, thresholds=resolved)
        status = entry.critic_result.overall_status or "unknown"
        summary[status] = summary.get(status, 0) + 1
    return {"thresholds": resolved, "summary": summary}


def _semantic_scores_for_candidate(
    semantic_index: TextSemanticRetrievalIndex | None,
    *,
    scenario: str,
    condition: str,
    object_class: str,
    candidate: Any,
    top_k: int,
) -> tuple[dict[str, float], str]:
    if semantic_index is None:
        return {}, ""
    query_text = semantic_query_text(
        scenario=scenario,
        condition=condition,
        object_class=object_class,
        candidate_id=str(candidate.candidate_id),
        candidate_description=str(candidate.description),
        candidate_steps=list(candidate.steps),
        task_stage="task_chain",
    )
    return semantic_index.search_scores(query_text, top_k=top_k), query_text


def _run_policy_smoke(library: ExperienceLibrary, policy_calibration: dict[str, Any], smoke_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for item in smoke_items:
        scenario = str(item.get("scenario") or item.get("scenario_id") or "")
        condition = str(item.get("condition") or item.get("condition_id") or "")
        top_k = int(item.get("top_k", 5))
        execute_on = str(item.get("execute_on") or "review")
        control_mode = str(item.get("control_mode") or "physical")
        visual_scores = load_visual_scores(
            _path(item, "visual_index_dir", "") if item.get("visual_index_dir") else None,
            [_path({"query_image": image}, "query_image", "") for image in list(item.get("query_images") or [])],
            top_k=int(item.get("visual_top_k", 10)),
        )
        use_semantic = bool(item.get("use_text_semantic_retrieval", item.get("use_semantic_retrieval", False)))
        semantic_index = TextSemanticRetrievalIndex(
            library.entries,
            backend=str(item.get("semantic_backend") or "auto"),
        ) if use_semantic else None
        use_stage_retrieval = bool(item.get("use_stage_retrieval", False) or item.get("use_stage_planner_guidance", False))
        render_stage_context = bool(item.get("render_stage_context", False) or item.get("use_stage_planner_guidance", False))
        object_class = object_class_for_scenario(scenario)
        candidates = []
        semantic_runtime_reports = []
        for candidate in candidates_for_scenario(scenario):
            semantic_scores, semantic_query = _semantic_scores_for_candidate(
                semantic_index,
                scenario=scenario,
                condition=condition,
                object_class=object_class,
                candidate=candidate,
                top_k=int(item.get("semantic_top_k", 10)),
            )
            if semantic_index is not None:
                semantic_runtime_reports.append({
                    "candidate_id": candidate.candidate_id,
                    "query_text": semantic_query,
                    "semantic_score_count": len(semantic_scores),
                    "top_scores": dict(list(semantic_scores.items())[:5]),
                })
            candidates.append(evaluate_candidate(
                library,
                candidate,
                scenario=scenario,
                condition=condition,
                object_class=object_class,
                top_k=top_k,
                risk_aware=bool(item.get("risk_aware", True)),
                policy_calibration=policy_calibration,
                visual_scores=visual_scores,
                visual_weight=float(item.get("visual_weight", 0.12)),
                semantic_scores=semantic_scores,
                semantic_weight=float(item.get("semantic_weight", 0.10)),
                semantic_backend=semantic_index.backend if semantic_index is not None else "",
            ))
        if use_stage_retrieval:
            for candidate_report in candidates:
                stage_report = run_stage_retrieval(
                    library,
                    scenario=scenario,
                    condition=condition,
                    object_class=object_class,
                    candidate_id=str(candidate_report["candidate_id"]),
                    candidate_steps=list(candidate_report["candidate_steps"]),
                    top_k=int(item["stage_top_k"]) if item.get("stage_top_k") is not None else None,
                )
                candidate_report["stage_retrieval"] = stage_report
                if render_stage_context:
                    context = build_stage_planner_context(
                        stage_report,
                        scenario=scenario,
                        condition=condition,
                        candidate_id=str(candidate_report["candidate_id"]),
                        candidate_steps=list(candidate_report["candidate_steps"]),
                        candidate_description=str(candidate_report.get("description") or ""),
                    )
                    candidate_report["stage_planner_context"] = context
                    if bool(item.get("use_stage_planner_guidance", False)):
                        apply_stage_planner_guidance(
                            candidate_report,
                            context,
                            guidance_weight=float(item.get("stage_planner_guidance_weight", 0.10)),
                        )
        ranked = sorted(candidates, key=selection_rank, reverse=True)
        selected = select_candidate(candidates, execute_on)
        executed = False
        execution_success = None
        if selected is not None and bool(selected["executable"]) and bool(item.get("execute", True)):
            result = run_task_chain(scenario, condition, control_mode, candidate_id=str(selected["candidate_id"]))
            executed = True
            execution_success = bool(result.success)
        best = ranked[0] if ranked else None
        stage_contexts = [
            candidate.get("stage_planner_context")
            for candidate in candidates
            if candidate.get("stage_planner_context")
        ]
        stage_guidance_reports = [
            candidate.get("stage_planner_guidance")
            for candidate in candidates
            if candidate.get("stage_planner_guidance")
        ]
        report.append({
            "scenario": scenario,
            "condition": condition,
            "best_candidate": {
                "candidate_id": best["candidate_id"],
                "decision": best["candidate_score"]["decision"],
                "candidate_score": best["candidate_score"]["candidate_score"],
                "risk_score": best["candidate_score"]["risk_score"],
            } if best else None,
            "selected_candidate_id": selected["candidate_id"] if selected is not None else "",
            "executed": executed,
            "execution_success": execution_success,
            "candidate_count": len(candidates),
            "visual_score_count": len(visual_scores),
            "text_semantic_retrieval_enabled": use_semantic,
            "semantic_backend": semantic_index.backend if semantic_index is not None else "",
            "semantic_score_count": sum(item["semantic_score_count"] for item in semantic_runtime_reports),
            "semantic_runtime_reports": semantic_runtime_reports,
            "stage_retrieval_enabled": use_stage_retrieval,
            "stage_retrieval_summary": summarize_stage_retrieval(candidates) if use_stage_retrieval else {},
            "stage_context_enabled": bool(stage_contexts),
            "stage_context_summary": summarize_stage_planner_contexts(stage_contexts),
            "stage_planner_guidance_enabled": bool(item.get("use_stage_planner_guidance", False)),
            "stage_planner_guidance_count": len(stage_guidance_reports),
            "stage_planner_guidance_reports": stage_guidance_reports,
        })
    return report


def _run_policy_comparison(library: ExperienceLibrary, policy_calibration: dict[str, Any], comparison_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for item in comparison_items:
        report.append(
            build_policy_comparison(
                library,
                scenario=str(item.get("scenario") or item.get("scenario_id") or ""),
                condition=str(item.get("condition") or item.get("condition_id") or ""),
                policy_calibration=policy_calibration,
                top_k=int(item.get("top_k", 5)),
                execute_on=str(item.get("execute_on") or "review"),
                control_mode=str(item.get("control_mode") or "physical"),
                execute=bool(item.get("execute", False)),
                visual_scores=load_visual_scores(
                    _path(item, "visual_index_dir", "") if item.get("visual_index_dir") else None,
                    [_path({"query_image": image}, "query_image", "") for image in list(item.get("query_images") or [])],
                    top_k=int(item.get("visual_top_k", 10)),
                ),
                visual_weight=float(item.get("visual_weight", 0.12)),
            )
        )
    return report


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    output_dir = _output_path(config, "output_dir", "results/memory/universal_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    library_path = output_dir / "universal_experience_library.json"
    report_path = output_dir / "pipeline_report.json"
    summary_path = output_dir / "summary_report.json"
    policy_calibration_path = output_dir / "policy_risk_calibration.json"

    library = ExperienceLibrary.load(_path(config, "base_library", "")) if config.get("base_library") else ExperienceLibrary()
    report: dict[str, Any] = {"config": str(args.config), "output_dir": str(output_dir), "steps": {}}

    report["steps"]["r1pro_tasks"] = _run_r1pro_tasks(library, list(config.get("r1pro_tasks") or []))
    report["steps"]["wrapper1_imports"] = _run_wrapper1_imports(library, list(config.get("wrapper1_imports") or []))
    report["steps"]["real_episode_imports"] = _run_real_imports(library, list(config.get("real_episode_imports") or []))
    report["steps"]["write_policy"] = _summarize_write_policy(report)
    _write_json(output_dir / "write_policy_report.json", report["steps"]["write_policy"])
    library.save(library_path)
    if args.stop_after == "build":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path)}, ensure_ascii=False))
        return

    min_pair_score = float(config.get("min_pair_score", 0.55))
    pairs = pair_sim_real_experiences(library.entries, min_pair_score=min_pair_score)
    library.entries = apply_pair_and_gap(library.entries, pairs)
    report["steps"]["pair"] = {"min_pair_score": min_pair_score, "pair_count": len(pairs), "pairs": pairs}
    library.save(library_path)
    if args.stop_after == "pair":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path)}, ensure_ascii=False))
        return

    report["steps"]["critic"] = _apply_critic(library, dict(config.get("critic_thresholds") or {}))
    library.save(library_path)
    if args.stop_after == "critic":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path)}, ensure_ascii=False))
        return

    calibrations = compute_group_calibrations(library.entries)
    library.entries = apply_sandbox_calibration(library.entries)
    report["steps"]["sandbox_calibration"] = {"group_count": len(calibrations)}
    library.save(library_path)
    if args.stop_after == "calibration":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path)}, ensure_ascii=False))
        return

    policy_calibration = build_policy_risk_calibration(library.entries)
    _write_json(policy_calibration_path, policy_calibration)
    report["steps"]["policy_calibration"] = {
        "path": str(policy_calibration_path),
        "group_count": policy_calibration.get("group_count", 0),
    }
    if args.stop_after == "policy_calibration":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path)}, ensure_ascii=False))
        return

    if bool(config.get("consolidate", True)):
        library.entries, consolidation_report = consolidate_experiences(library.entries)
        library.save(library_path)
        _write_json(output_dir / "consolidation_report.json", consolidation_report)
        report["steps"]["consolidation"] = {
            "path": str(output_dir / "consolidation_report.json"),
            "input_count": consolidation_report["input_count"],
            "output_count": consolidation_report["output_count"],
            "removed_count": consolidation_report["removed_count"],
            "merged_group_count": consolidation_report["merged_group_count"],
        }
    else:
        report["steps"]["consolidation"] = {"skipped": True}

    lifecycle_config = dict(config.get("memory_lifecycle") or {})
    if bool(lifecycle_config.get("enabled", False)):
        library.entries, lifecycle_report = consolidate_memory_lifecycle(
            library.entries,
            stm_capacity=int(lifecycle_config.get("stm_capacity", 30)),
            min_retrieval_count=int(lifecycle_config.get("min_retrieval_count", 3)),
            min_write_score=float(lifecycle_config.get("min_write_score", 0.65)),
            promote_real=bool(lifecycle_config.get("promote_real", True)),
            promote_failures=bool(lifecycle_config.get("promote_failures", True)),
            promote_validated_success=bool(lifecycle_config.get("promote_validated_success", True)),
            evict_batch_size=int(lifecycle_config.get("evict_batch_size", 5)),
        )
        library.save(library_path)
        _write_json(output_dir / "memory_lifecycle_report.json", lifecycle_report)
        report["steps"]["memory_lifecycle"] = {
            "path": str(output_dir / "memory_lifecycle_report.json"),
            "input_count": lifecycle_report["input_count"],
            "output_count": lifecycle_report["output_count"],
            "stm_count": lifecycle_report["stm_count"],
            "ltm_count": lifecycle_report["ltm_count"],
            "promoted_count": len(lifecycle_report["promoted"]),
            "evicted_count": len(lifecycle_report["evicted"]),
        }
    else:
        report["steps"]["memory_lifecycle"] = {"skipped": True}
    if args.stop_after == "consolidation":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path)}, ensure_ascii=False))
        return

    summary = build_summary(library, top_k=int(config.get("summary_top_k", 10)))
    quality = validate_experience_library(library.entries, check_refs=bool(args.check_refs or config.get("check_refs", False)))
    _write_json(summary_path, {"input": str(library_path), "summary": summary})
    report["steps"]["summary"] = {"path": str(summary_path), "entry_count": summary["entry_count"], "success_rate": summary["success_rate"]}
    report["steps"]["quality"] = quality
    strict_quality = bool(args.strict_quality or config.get("strict_quality", False))
    if strict_quality and not quality["passed"]:
        _write_json(report_path, report)
        print(json.dumps({"quality_passed": False, "error_count": quality["error_count"], "report": str(report_path)}, ensure_ascii=False))
        raise SystemExit(1)
    if args.stop_after == "summary":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path)}, ensure_ascii=False))
        return

    visual_config = dict(config.get("visual_index") or {})
    if bool(visual_config.get("enabled", False)):
        visual_index_dir = _path(visual_config, "index_dir", str(output_dir / "visual_index"))
        visual_report = build_visual_index(
            library,
            index_dir=visual_index_dir,
            base_dir=_path(visual_config, "base_dir", str(library_path.parent)),
            model_name=visual_config.get("model_name"),
        )
        visual_report = {"input": str(library_path), **visual_report}
        _write_json(_path(visual_config, "report", str(output_dir / "visual_index_report.json")), visual_report)
        report["steps"]["visual_index"] = {
            "index_dir": visual_report["index_dir"],
            "indexed_entry_count": visual_report["indexed_entry_count"],
            "indexed_image_count": visual_report["indexed_image_count"],
            "faiss_size": visual_report["faiss_size"],
        }
        eval_config = dict(visual_config.get("eval") or {})
        if bool(eval_config.get("enabled", False)):
            visual_eval = evaluate_visual_retrieval(
                library,
                index_dir=visual_index_dir,
                base_dir=_path(visual_config, "base_dir", str(library_path.parent)),
                top_k=int(eval_config.get("top_k", 5)),
                visual_weight=float(eval_config.get("visual_weight", 0.12)),
                all_keyframes=bool(eval_config.get("all_keyframes", False)),
            )
            visual_eval = {"library": str(library_path), "index_dir": str(visual_index_dir), **visual_eval}
            _write_json(_path(eval_config, "report", str(output_dir / "visual_retrieval_eval_report.json")), visual_eval)
            report["steps"]["visual_retrieval_eval"] = {
                "query_count": visual_eval["query_count"],
                "top1_self_hit_rate": visual_eval["top1_self_hit_rate"],
                "topk_self_hit_rate": visual_eval["topk_self_hit_rate"],
                "top1_same_condition_rate": visual_eval["top1_same_condition_rate"],
                "topk_same_condition_rate": visual_eval["topk_same_condition_rate"],
                "rank_delta_avg": visual_eval["rank_delta_avg"],
            }
        else:
            report["steps"]["visual_retrieval_eval"] = {"skipped": True}
    else:
        report["steps"]["visual_index"] = {"skipped": True}
        report["steps"]["visual_retrieval_eval"] = {"skipped": True}
    if args.stop_after == "visual_index":
        _write_json(report_path, report)
        print(json.dumps({"stop_after": args.stop_after, "library": str(library_path), "visual_index": report["steps"]["visual_index"]}, ensure_ascii=False))
        return

    report["steps"]["policy_smoke"] = _run_policy_smoke(library, policy_calibration, list(config.get("policy_smoke") or []))
    policy_comparison = _run_policy_comparison(library, policy_calibration, list(config.get("policy_comparison") or []))
    report["steps"]["policy_comparison"] = policy_comparison
    library.save(library_path)
    if policy_comparison:
        _write_json(output_dir / "policy_baseline_vs_memory_report.json", {
            "comparison_count": len(policy_comparison),
            "changed_count": sum(1 for item in policy_comparison if item["candidate_changed"]),
            "comparisons": policy_comparison,
        })
    _write_json(report_path, report)
    print(json.dumps({
        "library": str(library_path),
        "report": str(report_path),
        "summary": str(summary_path),
        "policy_calibration": str(policy_calibration_path),
        "entry_count": len(library.entries),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
