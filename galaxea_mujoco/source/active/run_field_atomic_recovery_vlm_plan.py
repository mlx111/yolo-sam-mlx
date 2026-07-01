from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIENCE_ROOT = ROOT.parent / "experience_system"
for path in (ROOT, EXPERIENCE_ROOT):
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)

from experience_core import (  # noqa: E402
    ExperienceLibrary,
    GALAXEA_R1PRO_TORSO_NAMESPACE,
    VisualRetrievalIndex,
    build_field_atomic_planner_input,
    count_plan_quality_issues_wrapper1_style,
    critic_prefilter_wrapper1_style,
    failed_plan_blocker_matches_wrapper1_style,
    field_atomic_success,
    field_atomic_llm_failure_brief,
    invoke_field_atomic_recovery_vlm,
    memory_usefulness_wrapper1_style,
    mmr_select_wrapper1_style,
    query_field_atomic_experience_matches,
    repeated_failure_wrapper1_style,
    sanitize_field_atomic_parameters,
    score_candidate_plan_wrapper1_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Galaxea field-atomic recovery steps from multimodal anomaly context.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--target-class", default="apple")
    parser.add_argument("--nominal-report", type=Path, required=True)
    parser.add_argument("--replay-state", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, default=None)
    parser.add_argument("--scene-observation", type=Path, default=None)
    parser.add_argument("--universal-experience-lib", type=Path, default=None)
    parser.add_argument("--visual-index-dir", type=Path, default=None)
    parser.add_argument("--use-visual-retrieval", action="store_true")
    parser.add_argument("--provider", choices=["doubao", "openai"], default="doubao")
    parser.add_argument("--model", default="")
    parser.add_argument("--dry-run-llm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--rewrite-attempts", type=int, default=1)
    parser.add_argument("--execute-candidate-validation", action="store_true", help="Execute semantically valid candidates in MuJoCo before selecting the final plan.")
    parser.add_argument("--validation-model-path", type=Path, default=None)
    parser.add_argument("--validation-output-dir", type=Path, default=None)
    parser.add_argument("--validation-settle-before-steps", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=3)
    parser.add_argument("--save-context", type=Path, required=True)
    parser.add_argument("--save-plan", type=Path, required=True)
    parser.add_argument("--save-report", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "_raw_text"}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _execution_steps_view(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": [
            {
                "action": str(step.get("action") or ""),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in plan.get("steps") or []
            if isinstance(step, dict)
        ]
    }


def _mock_recovery_plan(target_class: str) -> dict[str, Any]:
    target_class = str(target_class or "apple")
    return {
        "steps": [
            {"action": "head_camera_rgbd_save", "parameters": {}},
            {"action": "head_camera_grounded_sam2_pose", "parameters": {"target_class": target_class}},
            {
                "action": "plan_cartesian_trajectory",
                "parameters": {
                    "side": "left",
                    "target_class": target_class,
                    "mode": "side_then_in",
                    "pregrasp_offset_x": 0.0,
                    "pregrasp_offset_y": 0.0,
                    "pregrasp_offset_z": 0.08,
                    "side_offset_x": -0.06,
                    "side_offset_y": 0.0,
                    "topdown_mode": "palm_down",
                },
            },
            {"action": "move_to_pregrasp", "parameters": {"side": "left", "target_class": target_class, "pregrasp_offset_x": 0.0, "pregrasp_offset_y": 0.0, "pregrasp_offset_z": 0.08, "topdown_mode": "palm_down"}},
            {"action": "approach_object", "parameters": {"side": "left", "target_class": target_class, "visual_grasp_offset_z": 0.007, "topdown_mode": "palm_down"}},
            {"action": "close_gripper", "parameters": {"side": "left"}},
            {"action": "lift", "parameters": {"side": "left", "target_class": target_class, "lift_height": 0.10}},
        ]
    }


def _mock_bad_recovery_plan(target_class: str) -> dict[str, Any]:
    target_class = str(target_class or "apple")
    return {
        "steps": [
            {"action": "head_camera_rgbd_save", "parameters": {}},
            {"action": "head_camera_grounded_sam2_pose", "parameters": {"target_class": target_class}},
        ]
    }


def _task_history(nominal: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in nominal.get("actions") or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "index": item.get("index"),
            "action": item.get("action", ""),
            "success": bool(item.get("success", False)),
            "status": item.get("status", ""),
            "message": item.get("message", ""),
        })
    return rows


def _failed_action(nominal: dict[str, Any]) -> dict[str, Any]:
    actions = [item for item in nominal.get("actions") or [] if isinstance(item, dict)]
    if nominal.get("object_lift_success") is False:
        for item in reversed(actions):
            if "lift" in str(item.get("action") or ""):
                return item
    for item in actions:
        if not bool(item.get("success", False)):
            return item
    return actions[-1] if actions else {}


def _rule_metrics(nominal: dict[str, Any], failed: dict[str, Any]) -> dict[str, Any]:
    raw = failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {}
    return {
        "task_success": nominal.get("task_success"),
        "object_lift_success": nominal.get("object_lift_success"),
        "object_lift_world": nominal.get("object_lift_world"),
        "object_body": nominal.get("object_body"),
        "final_error": raw.get("final_error"),
        "stage_errors": raw.get("stage_errors"),
        "stage_orientation_errors": raw.get("stage_orientation_errors"),
        "debug_tcp_world": raw.get("debug_tcp_world"),
        "debug_tcp_minus_target_world": raw.get("debug_tcp_minus_target_world"),
        "object_start_world": raw.get("object_start_world") or nominal.get("object_start_world"),
        "object_final_world": raw.get("object_final_world") or nominal.get("object_final_world"),
    }


def _image_paths(nominal: dict[str, Any], replay_state: dict[str, Any]) -> dict[str, str]:
    nominal_frames = [item for item in nominal.get("keyframes") or [] if isinstance(item, dict)]
    replay_frames = [item for item in replay_state.get("keyframes") or [] if isinstance(item, dict)]

    def find(frames: list[dict[str, Any]], prefix: str) -> str:
        for item in frames:
            stage = str(item.get("stage") or "")
            path = str(item.get("image_path") or "")
            if stage.startswith(prefix) and path:
                return path
        return ""

    failed = _failed_action(nominal)

    def failed_index() -> int:
        try:
            return int(failed.get("index"))
        except Exception:
            return 0

    def last_after_failure() -> str:
        return find(nominal_frames, f"after_step_{failed_index():03d}_")

    return {
        "scene_initial": find(nominal_frames, "scene_initial") or find(replay_frames, "replay_initial"),
        "before_anomaly": find(nominal_frames, f"before_step_{failed_index():03d}_"),
        "after_anomaly": last_after_failure(),
        "failed_step": find(nominal_frames, f"failed_step_{failed_index():03d}_"),
        "replay_state": find(replay_frames, "replay_state"),
        "final_state": find(nominal_frames, "final_state"),
    }


def _resolve_image_path(path: str, *, base_dir: Path | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        parts = candidate.parts
        for marker in ("galaxea_mujoco", "experience_system"):
            if marker not in parts:
                continue
            suffix = Path(*parts[parts.index(marker):])
            relocated = (ROOT.parent / suffix).resolve()
            if relocated.exists():
                return str(relocated)
        return ""
    if base_dir is not None:
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    repo_relative = (ROOT.parent / candidate).resolve()
    if repo_relative.exists():
        return str(repo_relative)
    return ""


def _entry_keyframes(entry: Any, *, library_path: Path | None, limit: int = 2) -> list[dict[str, Any]]:
    frames = entry.keyframes if isinstance(getattr(entry, "keyframes", None), list) else []
    if not frames and isinstance(getattr(entry, "metadata", None), dict):
        frames = entry.metadata.get("keyframes") if isinstance(entry.metadata.get("keyframes"), list) else []
    base_dir = library_path.parent if library_path is not None else None
    selected: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_path = _resolve_image_path(str(frame.get("image_path") or ""), base_dir=base_dir)
        if not image_path:
            continue
        selected.append({
            "image_path": image_path,
            "stage": frame.get("stage", ""),
            "description": frame.get("description", ""),
        })
        if len(selected) >= max(0, int(limit)):
            break
    return selected


def _retrieved_experience_context(
    library: ExperienceLibrary,
    *,
    scenario_id: str,
    anomaly_state: dict[str, Any] | None = None,
    task_stage: str = "",
    text_summary: str = "",
    visual_scores: dict[str, float] | None = None,
    library_path: Path | None,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[Any, float, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    available_actions = set()
    for catalog in library.skill_catalogs.values():
        skills = catalog.get("skills") if isinstance(catalog, dict) else {}
        if isinstance(skills, dict):
            available_actions.update(str(name) for name in skills)
    candidates = query_field_atomic_experience_matches(
        library.entries,
        scenario_id=scenario_id,
        skill_namespace=GALAXEA_R1PRO_TORSO_NAMESPACE,
        available_actions=available_actions,
        retrieval_key=anomaly_state or {},
        anomaly_state=anomaly_state,
        task_stage=task_stage,
        text_summary=text_summary,
        visual_scores=visual_scores,
        gap_aware=True,
        risk_aware=True,
        diversity_lambda=0.25,
        limit=max(limit * 3, limit),
    )
    if visual_scores:
        candidates = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]),
                -float(visual_scores.get(item[0].experience_id, 0.0)),
                -int(str(item[0].memory_tags.get("memory_type") or "") == "field_atomic_episode"),
            ),
        )
    candidates = critic_prefilter_wrapper1_style(candidates)
    candidates = mmr_select_wrapper1_style(candidates, top_k=max(0, int(limit)), diversity_lambda=0.25)
    candidates = candidates[: max(0, int(limit))]
    for entry, score, explanation in candidates:
        keyframes = _entry_keyframes(entry, library_path=library_path, limit=2)
        row = field_atomic_llm_failure_brief(entry)
        used_as = "positive" if field_atomic_success(entry) else "negative"
        if _is_gap_risk_entry(entry):
            used_as = "risk"
        row["used_as"] = used_as
        row["retrieval_score"] = round(float(score), 4)
        row["retrieval_explanation"] = explanation
        if visual_scores and entry.experience_id in visual_scores:
            row["visual_similarity"] = round(float(visual_scores.get(entry.experience_id, 0.0)), 4)
        row["keyframes"] = keyframes
        rows.append(row)
        for keyframe in keyframes:
            images.append({
                "experience_id": entry.experience_id,
                **keyframe,
            })
        if len(rows) >= max(0, int(limit)):
            break
    return rows, images, candidates


def _is_gap_risk_entry(entry: Any) -> bool:
    if not field_atomic_success(entry):
        return True
    gap = getattr(entry, "sim_real_gap", None)
    outcome = getattr(gap, "outcome_gap", {}) if gap is not None else {}
    if isinstance(outcome, dict) and str(outcome.get("type") or "") == "sim_success_real_fail":
        return True
    critic = getattr(entry, "critic_result", None)
    if critic is not None and str(getattr(critic, "overall_status", "") or "") in {"warn", "fail"}:
        return True
    return False


def _retrieval_policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support = [row for row in rows if row.get("used_as") == "positive"]
    risk = [row for row in rows if row.get("used_as") in {"negative", "risk"}]
    return {
        "memory_policy": "dual_source_gap_critic",
        "retrieved_count": len(rows),
        "support_count": len(support),
        "risk_count": len(risk),
        "support_retrieved_memories": support,
        "risk_retrieved_memories": risk,
    }


def _visual_scores_for_context(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, float]:
    if not bool(args.use_visual_retrieval) or args.visual_index_dir is None:
        return {}
    image_paths = [str(item) for item in context.get("image_paths") or [] if str(item)]
    if not image_paths:
        return {}
    try:
        index = VisualRetrievalIndex()
        index.load(args.visual_index_dir)
        return dict(index.search(image_paths, top_k=24))
    except Exception as exc:
        context["visual_retrieval_error"] = str(exc)
        return {}


def build_recovery_context(args: argparse.Namespace) -> dict[str, Any]:
    nominal = _read_json(args.nominal_report)
    replay_state = _read_json(args.replay_state)
    replay_report = _read_json(args.replay_report)
    observation = _read_json(args.scene_observation)
    if not isinstance(nominal, dict):
        raise RuntimeError(f"Missing or invalid nominal report: {args.nominal_report}")
    if not isinstance(replay_state, dict):
        raise RuntimeError(f"Missing or invalid replay state: {args.replay_state}")
    failed = _failed_action(nominal)
    metrics = _rule_metrics(nominal, failed)
    stage = str(failed.get("action") or "unknown")
    failure_type = _failure_type(metrics, failed)
    description = _failure_description(metrics, failed)
    query_anomaly_state = _query_anomaly_state(failed, metrics, failure_type=failure_type)
    return {
        "schema_version": "field_atomic_recovery_context_v1",
        "scenario_id": args.scenario_id,
        "goal": args.goal,
        "target_class": args.target_class,
        "task_history": _task_history(nominal),
        "failed_action": {
            "index": failed.get("index"),
            "action": failed.get("action", ""),
            "status": failed.get("status", ""),
            "message": failed.get("message", ""),
            "parameters": failed.get("parameters") if isinstance(failed.get("parameters"), dict) else {},
            "raw_result": failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {},
        },
        "anomaly_summary": {
            "stage": stage,
            "failure_type": failure_type,
            "description": description,
            "rule_metrics": metrics,
            "query_anomaly_state": query_anomaly_state,
        },
        "robot_state": replay_state.get("robot_state") if isinstance(replay_state.get("robot_state"), dict) else {},
        "object_state": replay_state.get("object_state") if isinstance(replay_state.get("object_state"), dict) else {},
        "rule_metrics": metrics,
        "scene_observation": observation if isinstance(observation, dict) else {},
        "replay_report": replay_report if isinstance(replay_report, dict) else {},
        "image_paths": _image_paths(nominal, replay_state),
        "experience_image_paths": [],
        "retrieved_experiences": [],
    }


def _query_anomaly_state(failed: dict[str, Any], metrics: dict[str, Any], *, failure_type: str) -> dict[str, Any]:
    raw = failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {}
    params = failed.get("parameters") if isinstance(failed.get("parameters"), dict) else {}
    target_torso = raw.get("target_torso") if isinstance(raw.get("target_torso"), list) else None
    final_error = raw.get("final_error")
    return {
        "failure_stage": str(failed.get("action") or ""),
        "failure_type": failure_type,
        "target_class": str(params.get("target_class") or raw.get("target_class") or ""),
        "side": str(params.get("side") or raw.get("side") or ""),
        "target_torso_y_sign": _sign_bucket(target_torso[1]) if target_torso and len(target_torso) > 1 else "",
        "final_error_bucket": _error_bucket(final_error),
        "object_lift_bucket": _lift_bucket(metrics.get("object_lift_world")),
    }


def _sign_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric > 0.03:
        return "positive"
    if numeric < -0.03:
        return "negative"
    return "center"


def _error_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric < 0.02:
        return "low"
    if numeric < 0.10:
        return "medium"
    return "high"


def _lift_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric >= 0.095:
        return "success"
    if numeric > 0.01:
        return "partial"
    return "none"


def _failure_type(metrics: dict[str, Any], failed: dict[str, Any]) -> str:
    if metrics.get("object_lift_success") is False:
        return "object_not_lifted"
    if not bool(failed.get("success", False)):
        action = str(failed.get("action") or "")
        raw = failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {}
        if action in {"move_to_pregrasp", "approach_object", "plan_cartesian_trajectory", "move_base_relative"}:
            return "actuation_limit"
        if action == "head_camera_grounded_sam2_pose":
            return "perception_miss"
        if action == "close_gripper":
            return "grasp_miss"
        if action == "lift":
            if raw.get("object_lift_success") is False or raw.get("object_lift_world") is not None:
                return "object_not_lifted"
            return "actuation_limit"
        if action == "transport_to_detected_target":
            return "transport_collision" if raw.get("object_follow_error") is not None else "place_error"
        return "field_atomic_step_failed"
    return "task_incomplete"


def _failure_description(metrics: dict[str, Any], failed: dict[str, Any]) -> str:
    if metrics.get("object_lift_success") is False:
        return "夹爪或机械臂动作完成后，目标物体没有达到要求的提升高度。"
    failure_type = _failure_type(metrics, failed)
    if failure_type == "actuation_limit":
        raw = failed.get("raw_result") if isinstance(failed.get("raw_result"), dict) else {}
        return (
            "机械臂未能到达目标位姿，属于可达性/运动执行受限。"
            f" final_error={raw.get('final_error')}, target_torso={raw.get('target_torso')}"
        )
    if failure_type == "perception_miss":
        return "目标物体位置识别失败，后续运动缺少可靠目标位置。"
    message = str(failed.get("message") or "")
    return message or "技能序列未达到最终任务成功条件。"


def _generate_candidate_raw_plans(
    args: argparse.Namespace,
    context: dict[str, Any],
    planner_input: dict[str, Any],
    library: ExperienceLibrary,
) -> list[dict[str, Any]]:
    count = max(1, int(args.candidate_count))
    if args.dry_run_llm:
        candidates = [_mock_recovery_plan(args.target_class)]
        if count > 1:
            candidates.append(_mock_bad_recovery_plan(args.target_class))
        while len(candidates) < count:
            candidates.append(_mock_recovery_plan(args.target_class))
        return candidates[:count]
    candidates: list[dict[str, Any]] = []
    for index in range(count):
        candidate_context = dict(context)
        candidate_context["candidate_generation"] = {
            "candidate_index": index,
            "candidate_count": count,
            "instruction": (
                "生成一个可独立执行到原始目标完成的恢复候选。多候选之间必须有真实差异："
                "move_base_relative 的 x/y、plan_cartesian_trajectory 或 move_to_pregrasp 的 side、"
                "plan_cartesian_trajectory 的 mode 至少一项不同；禁止多个候选只改 lift_height 或完全重复同一组底盘参数。"
            ),
        }
        candidates.append(invoke_field_atomic_recovery_vlm(
            candidate_context,
            provider=args.provider,
            model=args.model,
            skill_namespace=GALAXEA_R1PRO_TORSO_NAMESPACE,
            skill_catalogs=library.skill_catalogs,
            planner_input=planner_input,
            max_steps=args.max_steps,
        ))
    return candidates


def _generate_rewrite_raw_plans(
    args: argparse.Namespace,
    context: dict[str, Any],
    planner_input: dict[str, Any],
    library: ExperienceLibrary,
    failed_candidate_reports: list[dict[str, Any]],
    *,
    attempt_index: int,
) -> list[dict[str, Any]]:
    count = max(1, int(args.candidate_count))
    if args.dry_run_llm:
        return [_mock_recovery_plan(args.target_class) for _ in range(count)]
    rewrite_context = dict(context)
    blocker_matches = []
    for row in failed_candidate_reports:
        if isinstance(row, dict) and isinstance(row.get("failed_plan_blocker_matches"), list):
            blocker_matches.extend(row.get("failed_plan_blocker_matches") or [])
    rewrite_context["rewrite_feedback"] = {
        "attempt_index": attempt_index,
        "reason": "all recovery candidates failed selection or execution handling",
        "blocker_matches": blocker_matches[:8],
        "failed_candidate_score_history": [_candidate_score_report(row) for row in failed_candidate_reports],
        "failed_candidate_execution": [
            _compact_candidate_failure(row)
            for row in failed_candidate_reports
            if isinstance(row, dict)
        ][:8],
        "instruction": (
            "根据候选失败原因和历史失败经验重写恢复计划。新计划必须从当前异常状态继续执行到原始 goal 完成，"
            "不能只包含观察或重新定位前缀；也不能只重复已经失败的动作参数组合。"
            "如果 blocker_matches 非空，说明上一轮方案与失败经验高度相似，必须改变方案结构来覆盖 llm_critic 中的 missing_phases、failed_predicates 和参数失败信息。"
        ),
    }
    return _generate_candidate_raw_plans(args, rewrite_context, planner_input, library)



def _compact_candidate_failure(row: dict[str, Any]) -> dict[str, Any]:
    plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
    report = row.get("execution_validation") if isinstance(row.get("execution_validation"), dict) else {}
    memory_score = row.get("memory_candidate_score") if isinstance(row.get("memory_candidate_score"), dict) else {}
    return {
        "candidate_index": row.get("candidate_index"),
        "generation_round": row.get("generation_round", "initial"),
        "actions": [
            str(step.get("action") or "")
            for step in plan.get("steps") or []
            if isinstance(step, dict)
        ],
        "memory_decision": memory_score.get("decision", ""),
        "top_failure_risks": memory_score.get("top_failure_risks", []),
        "execution_success": report.get("success") if report else None,
        "execution_report": report.get("report", "") if report else "",
        "object_lift_success": report.get("object_lift_success") if report else None,
        "object_lift_world": report.get("object_lift_world") if report else None,
    }


def _normalize_validate_candidates(
    raw_candidates: list[dict[str, Any]],
    *,
    goal: str,
    planner_input: dict[str, Any],
    library: ExperienceLibrary,
    max_steps: int,
    retrieved_matches: list[tuple[Any, float, dict[str, Any]]] | None = None,
    query_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        raw_text = raw.pop("_raw_text", "") if isinstance(raw, dict) else ""
        row: dict[str, Any] = {
            "candidate_index": index,
            "raw_plan": raw,
            "raw_text": raw_text,
        }
        try:
            plan = raw if isinstance(raw, dict) else {}
            if not isinstance(plan.get("steps"), list) or not plan.get("steps"):
                raise RuntimeError("recovery plan must contain non-empty steps")
            step_count = len(plan.get("steps") or [])
            if step_count > max_steps:
                raise RuntimeError(f"recovery plan has too many steps: {step_count} > {max_steps}")
            plan = {"goal": goal, **plan}
            removed_parameters = _sanitize_llm_plan_parameters(plan)
            available_actions = set()
            for catalog in library.skill_catalogs.values():
                skills = catalog.get("skills") if isinstance(catalog, dict) else {}
                if isinstance(skills, dict):
                    available_actions.update(str(name) for name in skills)
            plan_quality = count_plan_quality_issues_wrapper1_style(
                plan.get("steps") or [],
                allowed_actions=available_actions,
            )
            memory_score = score_candidate_plan_wrapper1_style(
                plan.get("steps") or [],
                retrieved_matches or [],
            )
            row.update({
                "status": "ok",
                "plan": plan,
                "plan_quality": plan_quality,
                "memory_candidate_score": memory_score,
                "selection_score": _candidate_selection_score(plan, memory_score=memory_score),
                "removed_llm_parameters": removed_parameters,
            })
        except Exception as exc:
            row.update({
                "status": "error",
                "error": str(exc),
                "selection_score": -1.0,
            })
        rows.append(row)
    rows.sort(key=lambda item: (-float(item.get("selection_score") or -1.0), int(item.get("candidate_index") or 0)))
    return rows


def _sanitize_llm_plan_parameters(plan: dict[str, Any]) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        parameters = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        clean = sanitize_field_atomic_parameters(action, parameters, llm_view=True)
        dropped = {
            key: value
            for key, value in parameters.items()
            if key not in clean
        }
        if dropped:
            removed.append({
                "step_index": index,
                "action": action,
                "removed_parameters": dropped,
            })
        step["parameters"] = clean
    return removed


def _candidate_selection_score(plan: dict[str, Any], *, memory_score: dict[str, Any] | None = None) -> float:
    base = 1.0
    action_count = len(plan.get("steps") or [])
    memory_value = float((memory_score or {}).get("candidate_score") or 0.5)
    return round(base + 0.35 * memory_value + min(action_count, 10) * 0.01, 4)


def _candidate_score_report(row: dict[str, Any]) -> dict[str, Any]:
    plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
    actions = [
        str(step.get("action") or "")
        for step in plan.get("steps") or []
        if isinstance(step, dict) and str(step.get("action") or "")
    ]
    score = float(row.get("selection_score") or -1.0)
    memory_score = row.get("memory_candidate_score") if isinstance(row.get("memory_candidate_score"), dict) else {}
    plan_quality = row.get("plan_quality") if isinstance(row.get("plan_quality"), dict) else {}
    memory_decision = str(memory_score.get("decision") or "")
    if memory_decision in {"reject", "reject_recommended", "rewrite", "rewrite_recommended"}:
        decision = memory_decision
    elif row.get("status") == "ok":
        decision = "execute_recommended"
    else:
        decision = "reject_recommended"
    return {
        "candidate_index": row.get("candidate_index"),
        "candidate_score": round(score, 4),
        "memory_candidate_score": memory_score.get("candidate_score"),
        "memory_decision": memory_score.get("decision", ""),
        "support_score": memory_score.get("support_score"),
        "risk_score": memory_score.get("risk_score"),
        "failure_overlap_risk": memory_score.get("failure_overlap_risk"),
        "terminal_risk_score": memory_score.get("terminal_risk_score"),
        "failure_risk_penalty": memory_score.get("failure_risk_penalty"),
        "gap_uncertainty": memory_score.get("gap_uncertainty"),
        "critic_risk": memory_score.get("critic_risk"),
        "invalid_plan_count": plan_quality.get("invalid_plan_count", 0),
        "unsafe_gripper_action_count": plan_quality.get("unsafe_gripper_action_count", 0),
        "quality_status": plan_quality.get("quality_status", ""),
        "decision": decision,
        "candidate_actions": actions,
        "issues": [],
    }


def _signature_actions_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    return [
        str(step.get("action") or "")
        for step in steps or []
        if isinstance(step, dict) and str(step.get("action") or "")
    ]


def _lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a == b else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(left), len(right))


def _retrieved_failure_blocker_matches(candidate: dict[str, Any], retrieved_matches: list[tuple[Any, float, dict[str, Any]]] | None) -> list[dict[str, Any]]:
    plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
    candidate_steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    candidate_actions = _signature_actions_from_steps(candidate_steps)
    if not candidate_actions:
        return []
    candidate_set = set(candidate_actions)
    matches: list[dict[str, Any]] = []
    for item in retrieved_matches or []:
        if len(item) < 2:
            continue
        entry = item[0]
        score = float(item[1] or 0.0)
        role = str(getattr(entry, "memory_tags", {}).get("memory_role") or "")
        if "failure" not in role and bool(getattr(entry, "result", {}).get("success", True)):
            continue
        entry_actions = [
            str(step.name)
            for step in getattr(entry, "skill_sequence", []) or []
            if getattr(step, "name", "")
        ]
        if not entry_actions and isinstance(getattr(entry, "action_trace", None), list):
            entry_actions = _signature_actions_from_steps(getattr(entry, "action_trace") or [])
        if not entry_actions:
            continue
        taxonomy = getattr(entry, "failure_taxonomy", {}) if isinstance(getattr(entry, "failure_taxonomy", {}), dict) else {}
        taxonomy = _taxonomy_with_rule_hint(taxonomy, entry)
        overlap = _lcs_ratio(candidate_actions, entry_actions)
        terminal_same = candidate_actions[-1:] == entry_actions[-1:]
        has_mitigation = _candidate_has_failure_mitigation(candidate_set, entry)
        failed_action = str(taxonomy.get("failure_action") or taxonomy.get("failure_stage") or "")
        contains_failed_action = bool(failed_action and failed_action in candidate_set)
        single_skill_failure = len(entry_actions) == 1 and contains_failed_action
        if overlap < 0.72 and not (terminal_same and overlap >= 0.55) and not single_skill_failure:
            continue
        if has_mitigation and overlap < 0.90 and not single_skill_failure:
            continue
        critic = taxonomy.get("llm_critic") if isinstance(taxonomy.get("llm_critic"), dict) else {}
        matches.append({
            "experience_id": getattr(entry, "experience_id", ""),
            "score": round(score, 4),
            "overlap": round(overlap, 4),
            "terminal_same": terminal_same,
            "candidate_actions": candidate_actions,
            "failed_signature": "->".join(entry_actions),
            "failure_type": taxonomy.get("failure_type", ""),
            "failure_stage": taxonomy.get("failure_stage", ""),
            "critic_root_cause": critic.get("root_cause") or "",
            "corrective_direction": critic.get("corrective_direction") or "",
            "missing_phases": critic.get("missing_phases") or [],
            "blocked": True,
        })
    matches.sort(key=lambda row: (-float(row.get("overlap") or 0.0), -float(row.get("score") or 0.0)))
    return matches[:5]


def _taxonomy_with_rule_hint(taxonomy: dict[str, Any], entry: Any) -> dict[str, Any]:
    out = dict(taxonomy)
    failure_type = str(out.get("failure_type") or "")
    failure_stage = str(out.get("failure_stage") or out.get("failure_action") or "")
    return out


def _candidate_has_failure_mitigation(candidate_actions: set[str], entry: Any) -> bool:
    taxonomy = getattr(entry, "failure_taxonomy", {}) if isinstance(getattr(entry, "failure_taxonomy", {}), dict) else {}
    failure_type = str(taxonomy.get("failure_type") or "")
    failure_stage = str(taxonomy.get("failure_stage") or "")
    missing = taxonomy.get("missing_phases") or []
    missing_text = " ".join(str(item) for item in missing)
    if failure_type == "actuation_limit" or failure_stage in {"move_to_pregrasp", "approach_object"} or "可达" in missing_text:
        if ("move_base_relative" in candidate_actions or "set_torso_posture" in candidate_actions) and {"head_camera_rgbd_save", "head_camera_grounded_sam2_pose"}.issubset(candidate_actions):
            return True
    if "重新感知" in missing_text or "感知" in missing_text:
        if {"head_camera_rgbd_save", "head_camera_grounded_sam2_pose"}.issubset(candidate_actions):
            return True
    if failure_type in {"object_not_lifted", "grasp_miss"}:
        if {"approach_object", "close_gripper", "lift"}.issubset(candidate_actions):
            return True
    return False


def _mark_failed_plan_blockers(
    validated_candidates: list[dict[str, Any]],
    retrieved_matches: list[tuple[Any, float, dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for row in validated_candidates:
        plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
        matches = failed_plan_blocker_matches_wrapper1_style(
            plan.get("steps") or [],
            retrieved_matches or [],
            stage=str(row.get("generation_round") or "llm_generated"),
            threshold=0.8,
        )
        if not matches:
            continue
        row["failed_plan_blocker_matches"] = matches
        blockers.extend(matches)
    return blockers


def _select_candidate(validated_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not validated_candidates:
        raise RuntimeError("no recovery candidates generated")
    for row in validated_candidates:
        if row.get("status") == "ok":
            return row
    return validated_candidates[0]


def _execute_candidate_validation(
    args: argparse.Namespace,
    validated_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not bool(args.execute_candidate_validation):
        return []
    if args.validation_model_path is None:
        return [{"enabled": True, "error": "validation_model_path_required"}]
    output_dir = args.validation_output_dir or (args.save_report.parent / "candidate_execution_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    runnable = [
        row for row in validated_candidates
        if isinstance(row.get("plan"), dict)
        and row.get("status") == "ok"
    ][: max(1, int(args.validation_limit))]
    reports: list[dict[str, Any]] = []
    for rank, row in enumerate(runnable):
        candidate_index = int(row.get("candidate_index") or 0)
        generation_round = str(row.get("generation_round") or "initial")
        prefix = f"{generation_round}_candidate_{candidate_index:03d}_rank_{rank:03d}"
        actions_path = output_dir / f"{prefix}_actions.json"
        report_path = output_dir / f"{prefix}_report.json"
        keyframe_dir = output_dir / f"{prefix}_keyframes"
        _write_json(actions_path, _execution_steps_view(row["plan"]))
        cmd = [
            sys.executable,
            "-B",
            "source/active/run_field_atomic_skill_smoke.py",
            "--model-path",
            str(args.validation_model_path),
            "--actions",
            str(actions_path),
            "--scenario-id",
            str(args.scenario_id),
            "--condition-id",
            f"candidate_validation_{generation_round}_{candidate_index}",
            "--goal",
            str(args.goal),
            "--initial-state",
            str(args.replay_state),
            "--settle-before-steps",
            str(max(0, int(args.validation_settle_before_steps))),
            "--save-report",
            str(report_path),
            "--experience-save-mode",
            "none",
            "--keyframe-dir",
            str(keyframe_dir),
            "--keyframe-camera",
            "workspace_overview_camera",
            "--stop-on-failure",
        ]
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False, env=env)
        payload = _read_json(report_path)
        if not isinstance(payload, dict):
            payload = {}
        reports.append({
            "candidate_index": candidate_index,
            "generation_round": generation_round,
            "rank": rank,
            "command": cmd,
            "returncode": int(proc.returncode),
            "success": int(proc.returncode) == 0 and bool(payload.get("task_success", False)),
            "task_success": payload.get("task_success"),
            "failure_count": payload.get("failure_count"),
            "success_count": payload.get("success_count"),
            "object_lift_success": payload.get("object_lift_success"),
            "object_lift_world": payload.get("object_lift_world"),
            "report": str(report_path),
            "actions": str(actions_path),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        })
    return reports


def _select_by_execution_validation(selected_candidate: dict[str, Any], validation_reports: list[dict[str, Any]], validated_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in validation_reports if bool(item.get("success", False))]
    if not successful:
        return selected_candidate
    successful.sort(key=lambda item: (int(item.get("failure_count") or 0), -float(item.get("object_lift_world") or 0.0), int(item.get("rank") or 0)))
    best = successful[0]
    for row in validated_candidates:
        if int(row.get("candidate_index") or 0) == int(best.get("candidate_index") or -1) and str(row.get("generation_round") or "initial") == str(best.get("generation_round") or "initial"):
            row["execution_validation"] = best
            return row
    return selected_candidate


def _attach_execution_validation_reports(validated_candidates: list[dict[str, Any]], validation_reports: list[dict[str, Any]]) -> None:
    for report in validation_reports or []:
        candidate_index = int(report.get("candidate_index") or -1)
        generation_round = str(report.get("generation_round") or "initial")
        for row in validated_candidates:
            if int(row.get("candidate_index") or -2) == candidate_index and str(row.get("generation_round") or "initial") == generation_round:
                row["execution_validation"] = report
                break


def _has_successful_execution_validation(validation_reports: list[dict[str, Any]]) -> bool:
    if not validation_reports:
        return False
    if validation_reports and validation_reports[0].get("error"):
        return False
    return any(bool(item.get("success", False)) for item in validation_reports)


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib) if args.universal_experience_lib else ExperienceLibrary()
    context = build_recovery_context(args)
    planner_input = build_field_atomic_planner_input(
        library.entries,
        scenario_id=args.scenario_id,
        goal=args.goal,
        skill_namespace=GALAXEA_R1PRO_TORSO_NAMESPACE,
    )
    visual_scores = _visual_scores_for_context(args, context)
    query_context = {
        "scenario_id": args.scenario_id,
        "goal": args.goal,
        "target_class": args.target_class,
        "anomaly_state": context.get("anomaly_summary", {}).get("query_anomaly_state") if isinstance(context.get("anomaly_summary"), dict) else {},
        "failed_action": context.get("failed_action") if isinstance(context.get("failed_action"), dict) else {},
        "rule_metrics": context.get("rule_metrics") if isinstance(context.get("rule_metrics"), dict) else {},
    }
    retrieved_experiences, experience_image_paths, retrieved_matches = _retrieved_experience_context(
        library,
        scenario_id=args.scenario_id,
        anomaly_state=context.get("anomaly_summary", {}).get("query_anomaly_state") if isinstance(context.get("anomaly_summary"), dict) else {},
        task_stage=str(context.get("failed_action", {}).get("action") or "") if isinstance(context.get("failed_action"), dict) else "",
        text_summary=str(context.get("anomaly_summary", {}).get("description") or args.goal) if isinstance(context.get("anomaly_summary"), dict) else str(args.goal),
        visual_scores=visual_scores,
        library_path=args.universal_experience_lib,
        limit=8,
    )
    context["retrieved_experiences"] = retrieved_experiences
    context["experience_image_paths"] = experience_image_paths
    context["retrieval_policy"] = _retrieval_policy_summary(retrieved_experiences)
    context["visual_retrieval"] = {
        "enabled": bool(args.use_visual_retrieval),
        "index_dir": str(args.visual_index_dir) if args.visual_index_dir else "",
        "query_image_count": len(context.get("image_paths") or []),
        "match_count": len(visual_scores),
        "top_matches": [
            {"experience_id": experience_id, "visual_similarity": round(float(score), 4)}
            for experience_id, score in sorted(visual_scores.items(), key=lambda item: (-float(item[1]), item[0]))[:8]
        ],
    }
    raw_candidates = _generate_candidate_raw_plans(
        args,
        context,
        planner_input,
        library,
    )
    validated_candidates = _normalize_validate_candidates(
        raw_candidates,
        goal=args.goal,
        planner_input=planner_input,
        library=library,
        max_steps=args.max_steps,
        retrieved_matches=retrieved_matches,
        query_context=query_context,
    )
    for row in validated_candidates:
        row["generation_round"] = "initial"
    _mark_failed_plan_blockers(validated_candidates, retrieved_matches)
    rewrite_reports: list[dict[str, Any]] = []
    selected_candidate: dict[str, Any] | None = None
    rewrite_triggered = False
    rewrite_success = False
    execution_validation_reports: list[dict[str, Any]] = []
    try:
        selected_candidate = _select_candidate(validated_candidates)
    except Exception:
        selected_candidate = None
    if selected_candidate is not None and bool(args.execute_candidate_validation):
        execution_validation_reports = _execute_candidate_validation(args, validated_candidates)
        _attach_execution_validation_reports(validated_candidates, execution_validation_reports)
        if _has_successful_execution_validation(execution_validation_reports):
            selected_candidate = _select_by_execution_validation(selected_candidate, execution_validation_reports, validated_candidates)
    if selected_candidate is None:
        for attempt_index in range(max(0, int(args.rewrite_attempts))):
            rewrite_triggered = True
            rewrite_raw = _generate_rewrite_raw_plans(
                args,
                context,
                planner_input,
                library,
                validated_candidates,
                attempt_index=attempt_index,
            )
            rewrite_validated = _normalize_validate_candidates(
                rewrite_raw,
                goal=args.goal,
                planner_input=planner_input,
                library=library,
                max_steps=args.max_steps,
                retrieved_matches=retrieved_matches,
                query_context=query_context,
            )
            for row in rewrite_validated:
                row["generation_round"] = f"rewrite_{attempt_index}"
            _mark_failed_plan_blockers(rewrite_validated, retrieved_matches)
            rewrite_reports.append({
                "attempt_index": attempt_index,
                "candidate_reports": rewrite_validated,
                "candidate_score_history": [_candidate_score_report(row) for row in rewrite_validated],
            })
            validated_candidates.extend(rewrite_validated)
            try:
                rewrite_selected = _select_candidate(rewrite_validated)
            except Exception:
                continue
            if bool(args.execute_candidate_validation):
                round_execution_reports = _execute_candidate_validation(args, rewrite_validated)
                execution_validation_reports.extend(round_execution_reports)
                _attach_execution_validation_reports(rewrite_validated, round_execution_reports)
                _attach_execution_validation_reports(validated_candidates, round_execution_reports)
                rewrite_reports[-1]["execution_validation_reports"] = round_execution_reports
                selected_candidate = _select_by_execution_validation(rewrite_selected, round_execution_reports, rewrite_validated)
            else:
                selected_candidate = rewrite_selected
            rewrite_success = True
            break
    if selected_candidate is None:
        try:
            selected_candidate = _select_candidate(validated_candidates)
        except Exception as exc:
            candidate_score_history = [_candidate_score_report(row) for row in validated_candidates]
            blocker_count = sum(len(row.get("failed_plan_blocker_matches") or []) for row in validated_candidates if isinstance(row, dict))
            _write_json(args.save_context, context)
            _write_json(args.save_plan, {"steps": []})
            _write_json(args.save_report, {
                "schema_version": "field_atomic_recovery_vlm_plan_report_v1",
                "scenario_id": args.scenario_id,
                "goal": args.goal,
                "target_class": args.target_class,
                "status": "failed",
                "error": "no recovery candidates generated",
                "candidate_score_history": candidate_score_history,
                "failed_plan_blocker_count": blocker_count,
                "rewrite_triggered": rewrite_triggered,
                "rewrite_success": rewrite_success,
                "rewrite_attempts": rewrite_reports,
                "recovery_context": context,
                "planner_input": planner_input,
                "candidate_reports": validated_candidates,
                "execution_plan": {"steps": []},
                "visual_retrieval": {
                    "enabled": bool(args.use_visual_retrieval),
                    "index_dir": str(args.visual_index_dir) if args.visual_index_dir else "",
                    "match_count": len(visual_scores),
                },
            })
            raise
    if not bool(args.execute_candidate_validation):
        execution_validation_reports = _execute_candidate_validation(args, validated_candidates)
        if execution_validation_reports and not execution_validation_reports[0].get("error"):
            selected_candidate = _select_by_execution_validation(selected_candidate, execution_validation_reports, validated_candidates)
    plan = selected_candidate.get("plan") if isinstance(selected_candidate.get("plan"), dict) else {}
    candidate_score_history = [_candidate_score_report(row) for row in validated_candidates]
    memory_usefulness = memory_usefulness_wrapper1_style(plan.get("steps") or [], retrieved_matches)
    repeated_failure = repeated_failure_wrapper1_style(plan.get("steps") or [], retrieved_matches)
    _write_json(args.save_context, context)
    _write_json(args.save_plan, _execution_steps_view(plan))
    _write_json(args.save_report, {
        "schema_version": "field_atomic_recovery_vlm_plan_report_v1",
        "scenario_id": args.scenario_id,
        "goal": args.goal,
        "target_class": args.target_class,
        "dry_run_llm": bool(args.dry_run_llm),
        "candidate_count": max(1, int(args.candidate_count)),
        "selected_candidate_index": selected_candidate.get("candidate_index"),
        "selected_generation_round": selected_candidate.get("generation_round", "initial"),
        "selected_candidate_score": selected_candidate.get("selection_score"),
        "selected_candidate_decision": _candidate_score_report(selected_candidate).get("decision"),
        "candidate_score_history": candidate_score_history,
        "memory_usefulness": memory_usefulness,
        "repeated_failure": repeated_failure,
        "galaxea_recovery_rules": planner_input.get("galaxea_recovery_rules") if isinstance(planner_input, dict) else {},
        "rewrite_triggered": rewrite_triggered,
        "rewrite_success": rewrite_success,
        "rewrite_attempts": rewrite_reports,
        "execution_validation": {
            "enabled": bool(args.execute_candidate_validation),
            "model_path": str(args.validation_model_path) if args.validation_model_path else "",
            "report_count": len(execution_validation_reports),
            "reports": execution_validation_reports,
        },
        "recovery_context": context,
        "planner_input": planner_input,
        "raw_plan": selected_candidate.get("raw_plan", {}),
        "raw_text": selected_candidate.get("raw_text", ""),
        "candidate_reports": validated_candidates,
        "field_atomic_plan": plan,
        "execution_plan": _execution_steps_view(plan),
        "visual_retrieval": {
            "enabled": bool(args.use_visual_retrieval),
            "index_dir": str(args.visual_index_dir) if args.visual_index_dir else "",
            "match_count": len(visual_scores),
        },
    })
    print(json.dumps({
        "step_count": len(plan.get("steps") or []),
        "save_context": str(args.save_context),
        "save_plan": str(args.save_plan),
        "save_report": str(args.save_report),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
