#!/usr/bin/env python3
"""
Experimental runner for ablation-style anomaly recovery experiments.

This file extends ExperimentV4 with method-level switches, memory policies,
read/write-separated experience libraries, and extra experiment metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

WRAPPER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WRAPPER_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WRAPPER_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experience_system"))

from experience_core.llm_provider import JSON_ONLY_LINE, invoke_llm, parse_json_payload
from experience_system.ur5e_core.critic import build_critic_result, critique_ur5e_failure_experience
from experience_system.ur5e_core.llm_runtime import resolve_ur5e_critic_model, resolve_ur5e_provider
from experience_system.ur5e_core.planner import plan_recovery, plan_recovery_candidates
from experiment_runtime.experiment_config import (
    DEFAULT_DIVERSITY_LAMBDA,
    INITIAL_GRASP_ATTACH_MAX_DISTANCE,
    OBJECT_DISPLACED_DX,
    OBJECT_DISPLACED_DY,
    COLLISION_RECOVERY_LIFT_FROM_TABLE,
    COLLISION_RECOVERY_PINCH_DISTANCE,
    OBJECT_DISPLACED_LIFT_FROM_TABLE,
    OBJECT_DISPLACED_PINCH_DISTANCE,
    POLICY_CANDIDATE_TOP_K,
    POLICY_FAILED_TOP_K,
    POLICY_MIXED_TOP_K,
    POLICY_POSITIVE_TOP_K,
    RECOVERY_ATTACH_MAX_DISTANCE,
    TASK_LIFT_Z_CHANGE,
    SLIP_RECOVERY_LIFT_FROM_TABLE,
    SLIP_RECOVERY_PINCH_DISTANCE,
)
from experience_system.memory.v3 import (
    MemoryV3Library,
    build_retrieval_key,
    canonical_action_signature_from_entry,
    canonical_action_signature_from_steps,
)
from experience_system.memory.failure_cluster import FailureClusterer
from experience_system.memory.scoring import critic_risk_score, estimate_gap_uncertainty, score_candidate_plan
from experience_system.memory.visual_retrieval import VisualRetrievalIndex, _image_paths_from_entry
from run_experiment_v4 import ExperimentV4, ROOT, _convert_numpy, _resolve_local_path
from candidate_sandbox import CandidateSandbox
from experiment_runtime.anomaly_conditions import get_condition_spec
from skills.registry import allowed_actions

# Lazy-init singleton for incremental failure clustering
_failure_clusterer: FailureClusterer | None = None

def _get_failure_clusterer() -> FailureClusterer:
    global _failure_clusterer
    if _failure_clusterer is None:
        _failure_clusterer = FailureClusterer()
    return _failure_clusterer


METHOD_DEFAULTS = {
    "direct_llm_weak": {"condition": "direct", "memory_policy": "none"},
    "direct_memory": {"condition": "direct", "memory_policy": "hierarchical"},
    "hierarchical_memory_weak": {"condition": "direct", "memory_policy": "hierarchical"},
    "hierarchical_no_failed": {"condition": "direct", "memory_policy": "no_failed"},
    "dual_source_gap_memory": {"condition": "direct", "memory_policy": "dual_source_gap"},
    "dual_source_gap_critic": {"condition": "direct", "memory_policy": "dual_source_gap_critic"},
}

MEMORY_POLICIES = {
    "none",
    "hierarchical",
    "no_failed",
    "mixed_no_priority",
    "dual_source_gap",
    "dual_source_gap_critic",
}

def _partition(entry: Any) -> str:
    if hasattr(entry, "get_partition"):
        return entry.get_partition()
    return getattr(entry, "memory_partition", "")


def _memory_record(entry: Any, score: float, used_as: str, score_explanation: dict[str, Any] | None = None) -> dict[str, Any]:
    retrieval_key = getattr(entry, "retrieval_key", {}) or {}
    anomaly_state = getattr(entry, "anomaly_state", {}) or {}
    failure_taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
    validation_evidence = getattr(entry, "validation_evidence", {}) or {}
    text_summary = getattr(entry, "text_summary", "") or ""
    score_explanation = score_explanation or {}
    return {
        "experience_id": getattr(entry, "experience_id", ""),
        "partition": _partition(entry),
        "source": getattr(entry, "source", ""),
        "status": getattr(entry, "status", ""),
        "validation_status": getattr(entry, "validation_status", ""),
        "validation_source": getattr(entry, "validation_source", ""),
        "score": float(score),
        "final_score": score_explanation.get("final_score", float(score)),
        "structured_similarity": score_explanation.get("structured_similarity"),
        "validation_score": score_explanation.get("validation_score"),
        "gap_uncertainty": score_explanation.get("gap_uncertainty"),
        "critic_risk": score_explanation.get("critic_risk"),
        "real_validation_bonus": score_explanation.get("real_validation_bonus"),
        "risk_penalty": score_explanation.get("risk_penalty"),
        "trust_bonus": score_explanation.get("trust_bonus"),
        "score_adjustment": score_explanation.get("score_adjustment"),
        "dual_source_adjustment": score_explanation.get("dual_source_adjustment", {}),
        "action_coverage": score_explanation.get("action_coverage"),
        "anomaly_state_similarity": score_explanation.get("anomaly_state_similarity"),
        "retrieval_key_similarity": score_explanation.get("retrieval_key_similarity"),
        "text_summary_similarity": score_explanation.get("text_summary_similarity"),
        "text_score": score_explanation.get("text_score"),
        "embedding_similarity": score_explanation.get("embedding_similarity"),
        "embedding_score": score_explanation.get("embedding_score"),
        "perception_similarity": score_explanation.get("perception_similarity"),
        "scene_embedding_similarity": score_explanation.get("scene_embedding_similarity"),
        "task_stage_match": score_explanation.get("task_stage_match"),
        "object_class_match": score_explanation.get("object_class_match"),
        "condition_match": score_explanation.get("condition_match"),
        "text_summary": text_summary,
        "text_summary_preview": text_summary[:240],
        "used_as": used_as,
        "result_success": bool(getattr(getattr(entry, "result", None), "success", False)),
        "retrieval_key": retrieval_key,
        "validation_evidence": validation_evidence,
        "displacement_bucket": retrieval_key.get("displacement_bucket", ""),
        "displacement_direction": retrieval_key.get("displacement_direction", ""),
        "contact_pattern": retrieval_key.get("contact_pattern", ""),
        "plan_signature": retrieval_key.get("plan_signature", getattr(entry, "plan_signature", "")),
        "anomaly_state": anomaly_state,
        "failure_taxonomy": failure_taxonomy,
        "failure_stage": failure_taxonomy.get("failure_stage", ""),
        "failure_type": failure_taxonomy.get("failure_type", ""),
    }


def _is_failed(entry: Any) -> bool:
    result = getattr(entry, "result", None)
    if result is not None and getattr(result, "task_success", None) is False:
        return True
    if result is not None and not bool(getattr(result, "success", False)):
        return True
    return _partition(entry) == "failed_memory" or getattr(entry, "status", "") == "failure"


def _is_risk_entry(entry: Any) -> bool:
    return _is_failed(entry) or estimate_gap_uncertainty(entry) >= 0.8 or critic_risk_score(entry) >= 0.5


def _step_param(step: dict[str, Any], key: str) -> float | None:
    params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
    if key not in params:
        return None
    try:
        return float(params[key])
    except (TypeError, ValueError):
        return None


def _parameter_signature(steps: list[dict[str, Any]]) -> str:
    payload: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        payload.append(
            {
                "action": str(step.get("action") or ""),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _failed_entry_bad_parameter_buckets(entry: Any) -> dict[str, dict[str, list[float]]]:
    taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
    critic = taxonomy.get("llm_critic") if isinstance(taxonomy, dict) else {}
    summary = critic.get("parameter_failure_summary") if isinstance(critic, dict) else {}
    buckets: dict[str, dict[str, list[float]]] = {}
    for item in (summary.get("items") if isinstance(summary, dict) else []) or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "")
        bad_values = item.get("bad_values") if isinstance(item.get("bad_values"), dict) else {}
        if not action or not bad_values:
            continue
        action_bucket = buckets.setdefault(action, {})
        for key, value in bad_values.items():
            try:
                action_bucket.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return buckets


def _candidate_failure_reject_reasons(entry: Any) -> list[str]:
    taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
    candidate_failure = taxonomy.get("candidate_failure") if isinstance(taxonomy, dict) else {}
    reasons: list[str] = []
    for item in (candidate_failure.get("rejections") if isinstance(candidate_failure, dict) else []) or []:
        if isinstance(item, dict) and item.get("reject_reason"):
            reasons.append(str(item.get("reject_reason")))
    return reasons


def _is_pre_execution_candidate_failure(entry: Any) -> bool:
    """Candidate was rejected before real recovery execution.

    These memories are useful prompt evidence, but their LLM parameter lessons
    are not physical failure evidence and must not hard-block sandbox testing.
    """
    reasons = _candidate_failure_reject_reasons(entry)
    if not reasons:
        return False
    execution_feedback = getattr(entry, "execution_feedback", None)
    virtual_success = getattr(execution_feedback, "virtual_validation_success", None)
    if virtual_success is not None:
        return False
    return True


def _sort_by_score(items: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
    return sorted(items, key=lambda item: -float(item[1]))


def _merge_experience_evidence(*groups: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
    merged: list[tuple[Any, float]] = []
    seen: set[str] = set()
    for group in groups:
        for entry, score in group:
            eid = getattr(entry, "experience_id", "")
            if eid in seen:
                continue
            merged.append((entry, score))
            seen.add(eid)
    return merged


def _failure_dedupe_key(entry: Any) -> str:
    retrieval_key = getattr(entry, "retrieval_key", {}) or {}
    taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
    return "::".join(
        [
            str(getattr(entry, "condition_id", "") or retrieval_key.get("condition_id", "")),
            str(retrieval_key.get("plan_signature", getattr(entry, "plan_signature", ""))),
            str(taxonomy.get("failure_type", getattr(getattr(entry, "result", None), "failure_reason", ""))),
        ]
    )


def _physical_failure_steps_for_critic(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    steps = metrics.get("executed_recovery_steps")
    if isinstance(steps, list) and steps:
        return [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in steps
            if isinstance(step, dict) and step.get("action")
        ]
    steps = metrics.get("llm_recovery_steps")
    if isinstance(steps, list) and steps:
        return [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in steps
            if isinstance(step, dict) and step.get("action")
        ]
    sandbox_results = metrics.get("candidate_sandbox_results")
    if isinstance(sandbox_results, list):
        ranked = sorted(
            [item for item in sandbox_results if isinstance(item, dict)],
            key=lambda item: (
                bool(item.get("selected_for_output")),
                not bool(item.get("task_success")),
                float(item.get("llm_score") or item.get("score") or -1.0),
            ),
            reverse=True,
        )
        for item in ranked:
            trace = item.get("step_trace") if isinstance(item.get("step_trace"), list) else []
            usable = [
                {
                    "action": str(step.get("action", "")),
                    "parameters": step.get("params") if isinstance(step.get("params"), dict) else {},
                }
                for step in trace
                if isinstance(step, dict) and step.get("action")
            ]
            if usable:
                return usable
    for item in metrics.get("candidate_rejections") or []:
        if not isinstance(item, dict):
            continue
        steps = item.get("steps") if isinstance(item.get("steps"), list) else []
        usable = [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in steps
            if isinstance(step, dict) and step.get("action")
        ]
        if usable:
            return usable
    return []


def _candidate_trace_steps_for_critic(result: dict[str, Any]) -> list[dict[str, Any]]:
    trace = result.get("step_trace") if isinstance(result.get("step_trace"), list) else []
    return [
        {
            "action": str(step.get("action", "")),
            "parameters": step.get("params") if isinstance(step.get("params"), dict) else {},
        }
        for step in trace
        if isinstance(step, dict) and step.get("action") and not str(step.get("action", "")).startswith("_")
    ]


def _compact_trace_for_selector(result: dict[str, Any], *, limit: int = 14) -> list[dict[str, Any]]:
    trace = result.get("step_trace") if isinstance(result.get("step_trace"), list) else []
    compact: list[dict[str, Any]] = []
    for step in trace[:limit]:
        if not isinstance(step, dict):
            continue
        before = step.get("before") if isinstance(step.get("before"), dict) else {}
        after = step.get("after") if isinstance(step.get("after"), dict) else {}
        compact.append({
            "action": step.get("action", ""),
            "status": step.get("status", ""),
            "params": step.get("params") if isinstance(step.get("params"), dict) else {},
            "before": {
                "apple_pos": before.get("apple_pos"),
                "apple_z": before.get("apple_z"),
                "pinch_distance": before.get("pinch_distance"),
            },
            "after": {
                "apple_pos": after.get("apple_pos"),
                "apple_z": after.get("apple_z"),
                "pinch_distance": after.get("pinch_distance"),
                "on_plate": after.get("on_plate"),
                "xy_dist_to_plate": after.get("xy_dist_to_plate"),
            },
            "error": step.get("error", ""),
            "keyframe": step.get("keyframe"),
        })
    return compact


def _candidate_task_success(result: dict[str, Any]) -> bool:
    if result.get("task_success") is not None:
        return bool(result.get("task_success"))
    if result.get("on_plate") is not None:
        return bool(result.get("on_plate"))
    return False


def _maybe_load_visual_index(runner: Any) -> None:
    """Load or create VisualRetrievalIndex from the runner's experience directory."""
    visual_dir = _visual_index_dir(runner)
    if visual_dir is None:
        return
    try:
        idx = VisualRetrievalIndex()
        if (visual_dir / "visual_index.faiss").exists():
            idx.load(visual_dir)
        runner.visual_index = idx
    except Exception as exc:
        print(f"  [WARN] 加载视觉索引失败: {exc}")
        runner.visual_index = None


def _visual_index_dir(runner: Any) -> Path | None:
    """Determine where the visual index should be stored.

    With rolling memory the index must live alongside the shared
    rolling_memory.json so all trials share the same FAISS index.
    """
    read_path = getattr(runner, "experience_read_path", None)
    write_path = getattr(runner, "experience_write_path", None)
    if read_path is not None and read_path.exists():
        return read_path.parent / "visual_index"
    if write_path is not None:
        return write_path.parent / "visual_index"
    return None


def _current_keyframe_paths(runner: Any) -> list[str]:
    """Get absolute paths of current trial's keyframe images for visual query context."""
    if runner.keyframe_output_dir is None or not runner.keyframe_output_dir.exists():
        return []
    paths: list[str] = []
    for stage in ("after_anomaly", "before_recovery"):
        p = runner.keyframe_output_dir / f"{stage}.jpg"
        if p.exists():
            paths.append(str(p.resolve()))
    return paths


def _entry_keyframe_abs_paths(entry: Any, runner: Any) -> list[str]:
    """Resolve entry keyframe image paths using the runner's keyframe dir."""
    kf_dir = getattr(runner, "keyframe_output_dir", None)
    if kf_dir is None or not kf_dir.exists():
        return []
    keyframes = getattr(entry, "keyframes", None) or []
    if not keyframes and hasattr(entry, "metadata"):
        keyframes = (entry.metadata or {}).get("keyframes") or []
    paths: list[str] = []
    for kf in keyframes:
        raw = (kf.get("image_path") if isinstance(kf, dict) else getattr(kf, "image_path", None)) or ""
        if not raw:
            continue
        p = kf_dir / Path(raw).name
        if p.exists():
            paths.append(str(p.resolve()))
            continue
        p2 = Path(raw)
        if p2.is_absolute() and p2.exists():
            paths.append(str(p2.resolve()))
    return paths


def _signature_actions(signature: str) -> list[str]:
    try:
        payload = json.loads(signature)
    except (TypeError, json.JSONDecodeError):
        return []
    actions: list[str] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "")
        if action:
            actions.append(action)
    return actions


def _lcs_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a == b else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / max(len(left), len(right))


def _failed_entry_to_llm_steps(entry: Any) -> list[dict[str, Any]]:
    allowed = allowed_actions()
    skill_sequence = getattr(entry, "skill_sequence", None)
    if skill_sequence:
        return [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in skill_sequence
            if isinstance(step, dict) and str(step.get("action", "")) in allowed
        ]
    return []


# Chinese descriptions for deterministic rule-critic flags.
# Each entry maps a rule name → (short_label, explanation) so the LLM prompt
# shows readable Chinese rather than raw English identifiers.
RULE_FLAG_CN: dict[str, tuple[str, str]] = {
    # ── Grasp & Contact ──────────────────────────────────────────────
    "no_contact_detected": (
        "夹爪空夹",
        "夹爪闭合后未检测到任何接触，说明夹爪在目标附近但未触碰到物体，可能因抓取位姿偏差或接近不充分。",
    ),
    "contact_lost_during_lift": (
        "提升中丢失接触",
        "夹爪闭合时有接触但提升后接触消失，说明物体在提升过程中滑脱，可能因夹持力不足或夹爪开合度不当。",
    ),
    "contact_gained_during_lift_unexpected": (
        "提升中异常获得接触",
        "夹爪闭合时无接触但提升后出现接触，说明提升过程中意外碰触到其他物体或环境。",
    ),
    "pinch_too_wide": (
        "夹爪开度过大",
        "夹爪闭合后的间距超出允许阈值，说明夹爪未能有效合拢到夹持位置，可能因物体位置偏差或夹爪被阻挡。",
    ),
    "grasp_not_secured": (
        "夹持未锁定",
        "夹爪闭合后目标未被可靠夹持，夹持锁定标志未置位，说明未形成有效抓取。",
    ),
    "apple_not_tracked": (
        "目标跟踪丢失",
        "苹果在恢复过程中未被持续跟踪，末端与目标的相对位置关系已不可靠。",
    ),
    # ── Lift ─────────────────────────────────────────────────────────
    "object_not_lifted": (
        "物体未抬升",
        "提升后物体离桌面的高度增量低于最低阈值，说明夹爪未能将物体抬离桌面，可能因夹爪未能卡到位或抓取高度不合适。",
    ),
    "insufficient_z_change": (
        "抬升高度变化不足",
        "夹爪提升过程中末端或物体在 Z 轴的高度变化量未达到预期，说明抬升动作未能有效执行。",
    ),
    # ── Plan ─────────────────────────────────────────────────────────
    "plan_blocked_invalid": (
        "规划被无效检测拦截",
        "LLM 生成的规划被无效方案检测器拦截，规划中包含不可执行的步骤组合或参数。",
    ),
    "invalid_skill_steps_in_plan": (
        "方案含无效技能",
        "LLM 输出中包含当前异常条件不可用的技能名称或参数格式错误。",
    ),
    "plan_blocked_by_failed_history": (
        "方案与历史失败模式匹配",
        "LLM 生成的方案与已知的失败经验动作序列高度重叠，被失败记忆拦截器拒绝执行。",
    ),
    # ── Virtual Validation ───────────────────────────────────────────
    "virtual_validation_failed": (
        "MuJoCo候选评估未通过",
        "候选恢复方案在 MuJoCo 中执行失败；该结果作为本轮物理执行证据。",
    ),
    # ── Repeated failure ─────────────────────────────────────────────
    "repeated_failure_pattern": (
        "重复失败模式",
        "当前执行的方案与之前失败的方案具有相同或高度相似的动作模式，说明故障原因未解决。",
    ),
    # ── Perception ───────────────────────────────────────────────────
    "perception_pos_inconsistency": (
        "感知位置不一致",
        "感知系统检测到的目标位置与执行过程中实际观测到的位置存在显著偏差（超过 3cm），说明感知结果不可靠。",
    ),
    "perception_offset_exceeds_grasp_range": (
        "感知偏移超抓取范围",
        "注入的感知位置偏移量（dx/dy 欧氏距离）超过了历史字段 attach_max_distance 表示的物理接触范围，即使夹爪按感知位置移动也无法触碰到物体。这是导致抓取失败的根本原因。",
    ),
}


def deterministic_rule_critic(metrics: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    criteria = metrics.get("task_success_criteria") or {}
    contact_close = metrics.get("contact_after_close") or {}
    contact_lift = metrics.get("contact_after_lift") or {}

    def _has_contact(d: dict[str, Any]) -> bool:
        return bool(d.get("left_contact") or d.get("right_contact") or d.get("contact"))

    close_ok = _has_contact(contact_close)
    lift_ok = _has_contact(contact_lift)

    # ── Grasp & Contact ──────────────────────────────────────────────
    if not close_ok and not lift_ok:
        flags.append({"rule": "no_contact_detected", "stage": "recovery_execution",
                       "evidence": "gripper closed but no contact detected after close or lift"})
    elif close_ok and not lift_ok:
        flags.append({"rule": "contact_lost_during_lift", "stage": "recovery_execution",
                       "evidence": "contact present after close but absent after lift — object likely dropped"})
    elif not close_ok and lift_ok:
        flags.append({"rule": "contact_gained_during_lift_unexpected", "stage": "recovery_execution",
                       "evidence": "no contact after close but contact detected after lift — unexpected"})

    pinch = criteria.get("pinch_distance")
    max_pinch = criteria.get("max_pinch_distance")
    if pinch is not None and max_pinch is not None and pinch > max_pinch:
        flags.append({"rule": "pinch_too_wide", "stage": "recovery_execution",
                       "evidence": f"pinch_distance={pinch:.4f} > max_pinch_distance={max_pinch:.4f}"})

    secured = criteria.get("grasp_secured")
    if secured is False:
        flags.append({"rule": "grasp_not_secured", "stage": "recovery_execution",
                       "evidence": "grasp_secured=False in recovery criteria"})

    # ── Lift ─────────────────────────────────────────────────────────
    lift_table = criteria.get("lift_from_table")
    min_lift = criteria.get("min_lift")
    if lift_table is not None and min_lift is not None and lift_table <= min_lift:
        flags.append({"rule": "object_not_lifted", "stage": "recovery_execution",
                       "evidence": f"lift_from_table={lift_table:.4f} <= min_lift={min_lift:.4f}"})

    z_change = criteria.get("z_change")
    min_z = metrics.get("task_success_z_change", 0.03)
    if z_change is not None and z_change < min_z:
        flags.append({"rule": "insufficient_z_change", "stage": "recovery_execution",
                       "evidence": f"z_change={z_change:.4f} < threshold={min_z}"})

    # ── Plan ─────────────────────────────────────────────────────────
    if metrics.get("recovery_blocked_by_invalid_plan"):
        flags.append({"rule": "plan_blocked_invalid", "stage": "recovery_plan",
                       "evidence": "LLM-generated plan was rejected by invalid-plan detector"})

    invalid_steps = metrics.get("invalid_skill_steps") or []
    if invalid_steps:
        flags.append({"rule": "invalid_skill_steps_in_plan", "stage": "recovery_plan",
                       "evidence": f"plan contained {len(invalid_steps)} invalid step(s): {invalid_steps[:3]}"})

    blocker_matches = metrics.get("failed_plan_blocker_matches") or []
    if blocker_matches:
        flags.append({"rule": "plan_blocked_by_failed_history", "stage": "recovery_plan",
                       "evidence": f"plan matched {len(blocker_matches)} failed plan signature(s) — blocked"})

    # ── Candidate MuJoCo execution ───────────────────────────────────
    vv = metrics.get("virtual_validation_success")
    if vv is False:
        flags.append({"rule": "virtual_validation_failed", "stage": "candidate_mujoco_execution",
                       "evidence": "candidate recovery plan failed in MuJoCo execution"})

    # ── Repeated failure ─────────────────────────────────────────────
    if metrics.get("repeated_failure_detected"):
        matches = metrics.get("repeated_failure_matches") or []
        flags.append({"rule": "repeated_failure_pattern", "stage": "memory_reuse",
                       "evidence": f"executed action signature matches {len(matches)} previous failed attempt(s)"})

    # ── Perception ───────────────────────────────────────────────────
    observed = metrics.get("observed_pos")
    perceived = metrics.get("perceived_position")
    if observed and perceived and len(observed) >= 2 and len(perceived) >= 2:
        dx = abs(float(observed[0]) - float(perceived[0]))
        dy = abs(float(observed[1]) - float(perceived[1]))
        threshold = 0.03
        if dx > threshold or dy > threshold:
            flags.append({"rule": "perception_pos_inconsistency", "stage": "detection",
                           "evidence": f"observed=({observed[0]:.3f},{observed[1]:.3f}) "
                                       f"perceived=({perceived[0]:.3f},{perceived[1]:.3f}) "
                                       f"dx={dx:.3f} dy={dy:.3f}"})

    # ── Injection geometry check ─────────────────────────────────────
    ci = metrics.get("condition_injection") or {}
    ci_params = ci.get("params") or {}
    ci_true = ci.get("true_pos")
    ci_perceived = ci.get("perceived_pos")
    if ci_true and ci_perceived and len(ci_true) >= 2 and len(ci_perceived) >= 2:
        offset_euclidean = float(np.linalg.norm(
            np.array(ci_true[:2]) - np.array(ci_perceived[:2])
        ))
        attach_max = float(ci_params.get("attach_max_distance", 0.045))
        if offset_euclidean > attach_max:
            flags.append({
                "rule": "perception_offset_exceeds_grasp_range",
                "stage": "detection",
                "evidence": f"perception offset {offset_euclidean:.3f}m > physical_contact_range {attach_max:.3f}m, "
                            f"true_pos=({ci_true[0]:.3f},{ci_true[1]:.3f}) "
                            f"perceived_pos=({ci_perceived[0]:.3f},{ci_perceived[1]:.3f})",
            })

    for flag in flags:
        rule_name = flag.get("rule", "")
        if rule_name in RULE_FLAG_CN:
            label_cn, desc_cn = RULE_FLAG_CN[rule_name]
            flag["label_cn"] = label_cn
            flag["description_cn"] = desc_cn

    return {
        "enabled": True,
        "rule_flags": flags,
        "flag_count": len(flags),
    }


class ExperimentMethodRunner(ExperimentV4):
    """ExperimentV4 wrapper with method/memory-policy controls."""

    def __init__(
        self,
        *,
        method: str,
        memory_policy: str,
        experience_read_path: str | None,
        experience_write_path: str | None,
        trial_id: str,
        seed: int | None,
        use_memory_keyframes: bool = False,
        memory_keyframe_top_k: int = 2,
        enable_failed_plan_rewrite: bool = False,
        inject_failed_plan_for_test: bool = False,
        recovery_candidate_count: int = 1,
        execute_recovery_candidate_validation: bool = False,
        failed_memory_hard_block: bool = False,
        dedupe_failure_memory: bool = False,
        memory_index_dir: str | None = None,
        strategy_family: str = "",
        experience_save_mode: str = "all",
        save_path: str | None = None,
        **kwargs: Any,
    ):
        read_path = None if memory_policy == "none" else experience_read_path
        super().__init__(experience_lib_path=read_path, **kwargs)
        self.method = method
        self.memory_policy = memory_policy
        self.experience_read_path = _resolve_local_path(experience_read_path)
        self.experience_write_path = _resolve_local_path(experience_write_path)
        self.trial_id = trial_id
        self.seed = seed
        self.use_memory_keyframes = bool(use_memory_keyframes)
        self.memory_keyframe_top_k = int(memory_keyframe_top_k)
        self.enable_failed_plan_rewrite = bool(enable_failed_plan_rewrite)
        self.inject_failed_plan_for_test = bool(inject_failed_plan_for_test)
        self.recovery_candidate_count = max(1, int(recovery_candidate_count or 1))
        self.execute_recovery_candidate_validation = bool(execute_recovery_candidate_validation)
        self.failed_memory_hard_block = bool(failed_memory_hard_block)
        self.dedupe_failure_memory = bool(dedupe_failure_memory)
        self.memory_index_dir = _resolve_local_path(memory_index_dir)
        self.strategy_family = str(strategy_family or "").strip()
        self.experience_save_mode = str(experience_save_mode or "all")
        self.prompt_profile = "task_list"
        self.save_path = _resolve_local_path(save_path)
        if self.memory_index_dir is not None:
            print("  memory_v3 按 condition_id 检索，忽略旧文本向量索引。")
        if self.experience_write_path is not None:
            self.keyframe_reference_dir = self.experience_write_path.parent
            self.keyframe_output_dir = self.experience_write_path.parent / "keyframes"
        self.visual_index: VisualRetrievalIndex | None = None
        if self.experience_write_path is not None:
            _maybe_load_visual_index(self)
        self._last_retrieved_experiences: list[tuple[Any, float]] = []
        self.metrics.update(
            {
                "method": method,
                "memory_policy": memory_policy,
                "trial_id": trial_id,
                "seed": seed,
                "strategy_family": self.strategy_family,
                "prompt_profile": self.prompt_profile,
                "experience_save_mode": self.experience_save_mode,
                "experience_saved": False,
                "experience_save_skipped_reason": "",
                "retrieved_memories": [],
                "virtual_validation_success": None,
                "virtual_validation_z_before": None,
                "virtual_validation_z_after": None,
                "virtual_validation_z_change": None,
                "invalid_plan_count": 0,
                "unsafe_gripper_action_count": 0,
                "repeated_failure_detected": False,
                "repeated_failure_matches": [],
                "failed_plan_blocked": False,
                "failed_plan_blocker_matches": [],
                "failed_plan_blocker_threshold": 0.8,
                "failed_plan_rewrite_triggered": False,
                "failed_plan_rewrite_success": False,
                "failed_plan_rewrite_error": "",
                "failed_plan_rewrite_steps": [],
                "inject_failed_plan_for_test": self.inject_failed_plan_for_test,
                "injected_failed_plan_for_test": False,
                "injected_failed_experience_id": "",
                "recovery_candidate_count": self.recovery_candidate_count,
                "execute_recovery_candidate_validation": self.execute_recovery_candidate_validation,
                "failed_memory_hard_block": self.failed_memory_hard_block,
                "dedupe_failure_memory": self.dedupe_failure_memory,
                "candidate_plans": [],
                "candidate_rejections": [],
                "candidate_sandbox_results": [],
                "selected_candidate_id": None,
                "executed_plan_signature": "",
                "retrieved_positive_count": 0,
                "retrieved_failed_count": 0,
                "memory_action_overlap_mean": None,
                "memory_action_overlap_max": None,
                "useful_memory_ratio": None,
                "avoided_failed_plan": False,
                "memory_usefulness": [],
                "prompt_memory_keyframes": [],
                "prompt_keyframe_count": 0,
                "executed_plan_source": None,
                "executed_recovery_steps": None,
            }
        )

    def _is_recovery_phase(self) -> bool:
        return self.metrics.get("observed_pos") is not None

    def _evaluate_task_success(self, target_pos: np.ndarray | None) -> bool:
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        apple_z = float(apple_pos[2])

        if self.scenario_id == "U3":
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            contact = self.metrics.get("contact_after_lift") or self._contact_summary()
            contact_secured = bool(contact.get("left_contact") or contact.get("right_contact"))
            min_lift = SLIP_RECOVERY_LIFT_FROM_TABLE
            max_pinch_distance = SLIP_RECOVERY_PINCH_DISTANCE
            grasp_secured = bool(contact_secured or pinch_distance < max_pinch_distance)
            success = bool(lift_from_table > min_lift and grasp_secured)
            self.metrics["task_success_criteria"] = {
                "type": "u3_gripper_recovered_and_lifted",
                "condition_id": self.condition_id,
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "contact_secured": contact_secured,
                "grasp_secured": grasp_secured,
                "min_lift": min_lift,
                "max_pinch_distance": max_pinch_distance,
                "success": success,
            }
            return success

        if self.anomaly_type == "object_displaced":
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            contact = self.metrics.get("contact_after_lift") or self._contact_summary()
            contact_secured = bool(contact.get("left_contact") or contact.get("right_contact"))
            success = lift_from_table > OBJECT_DISPLACED_LIFT_FROM_TABLE and (
                contact_secured or pinch_distance < OBJECT_DISPLACED_PINCH_DISTANCE
            )
            self.metrics["task_success_criteria"] = {
                "type": "object_displaced_lift_from_table",
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "contact_secured": contact_secured,
                "success": bool(success),
            }
            return bool(success)

        if self.anomaly_type in {"slip", "collision"}:
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            contact = self.metrics.get("contact_after_lift") or self._contact_summary()
            contact_secured = bool(contact.get("left_contact") or contact.get("right_contact"))
            if self.anomaly_type == "slip":
                min_lift = SLIP_RECOVERY_LIFT_FROM_TABLE
                max_pinch_distance = SLIP_RECOVERY_PINCH_DISTANCE
                criteria_type = "slip_regrasp_lift_from_table"
            else:
                min_lift = COLLISION_RECOVERY_LIFT_FROM_TABLE
                max_pinch_distance = COLLISION_RECOVERY_PINCH_DISTANCE
                criteria_type = "collision_relocalize_lift_from_table"
            success = lift_from_table > min_lift and (
                contact_secured or pinch_distance < max_pinch_distance
            )
            self.metrics["task_success_criteria"] = {
                "type": criteria_type,
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "contact_secured": contact_secured,
                "min_lift": min_lift,
                "max_pinch_distance": max_pinch_distance,
                "success": bool(success),
            }
            return bool(success)

        if target_pos is None:
            self.metrics["task_success_criteria"] = {
                "type": "observed_z_lift",
                "success": False,
                "reason": "target_position_unavailable",
            }
            return False

        observed_z = float(np.asarray(target_pos, dtype=np.float64).reshape(3)[2])
        z_change = apple_z - observed_z
        success = z_change > TASK_LIFT_Z_CHANGE
        self.metrics["task_success_criteria"] = {
            "type": "observed_z_lift",
            "apple_z": apple_z,
            "observed_z": observed_z,
            "z_change": z_change,
            "success": bool(success),
        }
        return bool(success)

    def _query_experiences_for_policy(self) -> list[tuple[Any, float]]:
        if self.memory_policy == "none" or self.experience_library is None or len(self.experience_library) == 0:
            self.metrics["retrieved_memories"] = []
            return []

        try:
            dual_source_policy = self.memory_policy in {"dual_source_gap", "dual_source_gap_critic"}
            candidates = self.experience_library.query(
                scenario_id=self.scenario_id,
                condition_id=self.condition_id,
                available_actions=allowed_actions(self.scenario_id),
                anomaly_state=self._current_anomaly_state(),
                retrieval_key=self._current_retrieval_key(self._current_anomaly_state()),
                task_stage=self.condition_spec.task_stage if self.condition_spec else "",
                text_summary=f"condition_id={self.condition_id}; scenario_id={self.scenario_id}",
                top_k=max(POLICY_CANDIDATE_TOP_K, len(self.experience_library)) if dual_source_policy else POLICY_CANDIDATE_TOP_K,
                diversity_lambda=0.0 if dual_source_policy else DEFAULT_DIVERSITY_LAMBDA,
                critic_prefilter=True,
                visual_index=self.visual_index,
                visual_context=_current_keyframe_paths(self),
                gap_aware=True,
                risk_aware=True,
            )
            self.metrics["memory_query_condition_id"] = self.condition_id
        except Exception as exc:
            print(f"  [WARN] 经验库检索失败: {exc}")
            self.metrics["retrieved_memories"] = []
            return []

        support_selected: list[tuple[Any, float]] = []
        risk_selected: list[tuple[Any, float]] = []
        if self.memory_policy == "hierarchical":
            positive = [(e, s) for e, s in candidates if not _is_failed(e)][:POLICY_POSITIVE_TOP_K]
            failed = [(e, s) for e, s in candidates if _is_failed(e)][:POLICY_FAILED_TOP_K]
            selected = positive + failed
        elif self.memory_policy == "no_failed":
            selected = [(e, s) for e, s in candidates if not _is_failed(e)][:POLICY_POSITIVE_TOP_K]
        elif self.memory_policy == "mixed_no_priority":
            selected = _sort_by_score(candidates)[:POLICY_MIXED_TOP_K]
        elif self.memory_policy in {"dual_source_gap", "dual_source_gap_critic"}:
            support_selected = [
                (e, s)
                for e, s in candidates
                if not _is_risk_entry(e)
            ][:POLICY_POSITIVE_TOP_K]
            risk_selected = [
                (e, s)
                for e, s in candidates
                if _is_risk_entry(e)
            ][:POLICY_FAILED_TOP_K]
            selected = _merge_experience_evidence(support_selected, risk_selected)
        else:
            selected = []

        records = []
        for entry, score in selected:
            used_as = "negative" if _is_failed(entry) else "positive"
            score_explanation = {}
            if self.experience_library is not None:
                score_explanation = self.experience_library.get_last_score_explanation(getattr(entry, "experience_id", ""))
            records.append(_memory_record(entry, score, used_as, score_explanation))
        self.metrics["retrieved_memories"] = records
        self.metrics["support_retrieved_memories"] = [
            _memory_record(
                entry,
                score,
                "positive",
                self.experience_library.get_last_score_explanation(getattr(entry, "experience_id", ""))
                if self.experience_library is not None
                else {},
            )
            for entry, score in support_selected
        ]
        self.metrics["risk_retrieved_memories"] = [
            _memory_record(
                entry,
                score,
                "negative" if _is_failed(entry) else "risk",
                self.experience_library.get_last_score_explanation(getattr(entry, "experience_id", ""))
                if self.experience_library is not None
                else {},
            )
            for entry, score in risk_selected
        ]
        self.metrics["dual_source_support_count"] = len(support_selected)
        self.metrics["dual_source_risk_count"] = len(risk_selected)
        self._last_retrieved_experiences = selected
        print(f"  经验库策略: {self.memory_policy}, 检索 {len(selected)} 条")
        if self.memory_policy in {"dual_source_gap", "dual_source_gap_critic"}:
            print(
                "  双源证据: "
                f"support={len(support_selected)} risk={len(risk_selected)}"
            )
        for rec in records:
            print(
                "    - {experience_id} partition={partition} score={score:.3f} used_as={used_as}".format(
                    **rec
                )
            )
        return selected

    def _query_recovery_experiences(self) -> list[tuple[Any, float]]:
        return self._query_experiences_for_policy()

    def _count_plan_quality_issues(self, steps: list[dict[str, Any]]) -> None:
        allowed = allowed_actions(getattr(self, "scenario_id", ""))
        invalid = 0
        for step in steps:
            action = step.get("action", "")
            if action not in allowed:
                invalid += 1
        self.metrics["invalid_plan_count"] = invalid
        self.metrics["unsafe_gripper_action_count"] = 0

    def _mark_repeated_failure(self, executed_steps: list[dict[str, Any]]) -> None:
        signature = canonical_action_signature_from_steps(executed_steps)
        self.metrics["executed_plan_signature"] = signature
        if not signature or signature == "[]":
            self.metrics["repeated_failure_detected"] = False
            self.metrics["repeated_failure_matches"] = []
            self._mark_memory_usefulness(signature)
            return

        matches: list[dict[str, Any]] = []
        for entry, score in self._last_retrieved_experiences:
            if not _is_failed(entry):
                continue
            if canonical_action_signature_from_entry(entry) != signature:
                continue
            matches.append(
                {
                    "experience_id": getattr(entry, "experience_id", ""),
                    "partition": _partition(entry),
                    "score": float(score),
                }
            )

        self.metrics["repeated_failure_detected"] = bool(matches)
        self.metrics["repeated_failure_matches"] = matches
        self._mark_memory_usefulness(signature)

    def _mark_failed_plan_blocker(self, candidate_steps: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
        signature = canonical_action_signature_from_steps(candidate_steps)
        if not signature or signature == "[]":
            return []

        threshold = float(self.metrics.get("failed_plan_blocker_threshold") or 0.8)
        candidate_actions = _signature_actions(signature)
        matches: list[dict[str, Any]] = []
        for entry, score in self._last_retrieved_experiences:
            if not _is_failed(entry):
                continue
            failed_signature = canonical_action_signature_from_entry(entry)
            overlap = _lcs_ratio(candidate_actions, _signature_actions(failed_signature))
            exact_match = bool(failed_signature and failed_signature == signature)
            if not exact_match and overlap < threshold:
                continue
            failure_taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
            matches.append(
                {
                    "stage": stage,
                    "experience_id": getattr(entry, "experience_id", ""),
                    "partition": _partition(entry),
                    "score": float(score),
                    "overlap": overlap,
                    "exact_signature_match": exact_match,
                    "candidate_signature": signature,
                    "failed_signature": failed_signature,
                    "failure_stage": failure_taxonomy.get("failure_stage", ""),
                    "failure_type": failure_taxonomy.get("failure_type", ""),
                    "cluster_id": failure_taxonomy.get("cluster_id", ""),
                    "blocked": True,
                    "rewrite_triggered": False,
                }
            )

        if matches:
            existing = list(self.metrics.get("failed_plan_blocker_matches") or [])
            existing.extend(matches)
            self.metrics["failed_plan_blocked"] = True
            self.metrics["failed_plan_blocker_matches"] = existing
        return matches

    def _mark_memory_usefulness(self, executed_signature: str) -> None:
        executed_actions = _signature_actions(executed_signature)
        records: list[dict[str, Any]] = []
        positive_count = 0
        failed_count = 0
        avoided_failed = False

        for entry, score in self._last_retrieved_experiences:
            entry_signature = canonical_action_signature_from_entry(entry)
            is_failed = _is_failed(entry)
            positive_count += int(not is_failed)
            failed_count += int(is_failed)
            overlap = _lcs_ratio(executed_actions, _signature_actions(entry_signature))
            exact_match = bool(executed_signature and entry_signature == executed_signature)
            if is_failed and not exact_match:
                avoided_failed = True
            useful = (not is_failed and overlap >= 0.6) or (is_failed and not exact_match)
            records.append(
                {
                    "experience_id": getattr(entry, "experience_id", ""),
                    "partition": _partition(entry),
                    "score": float(score),
                    "used_as": "negative" if is_failed else "positive",
                    "plan_signature": entry_signature,
                    "action_overlap": overlap,
                    "exact_signature_match": exact_match,
                    "useful_memory": useful,
                }
            )

        overlaps = [r["action_overlap"] for r in records if r["used_as"] == "positive"]
        self.metrics["retrieved_positive_count"] = positive_count
        self.metrics["retrieved_failed_count"] = failed_count
        self.metrics["memory_action_overlap_mean"] = sum(overlaps) / len(overlaps) if overlaps else None
        self.metrics["memory_action_overlap_max"] = max(overlaps) if overlaps else None
        self.metrics["useful_memory_ratio"] = (
            sum(1 for r in records if r["useful_memory"]) / len(records)
            if records
            else None
        )
        self.metrics["avoided_failed_plan"] = bool(failed_count and avoided_failed)
        self.metrics["memory_usefulness"] = records

    def _after_llm_plan_generated(self, llm_steps: list[dict]) -> None:
        self._count_plan_quality_issues(llm_steps)
        candidate_score = score_candidate_plan(llm_steps, self._last_retrieved_experiences)
        self.metrics["candidate_score"] = candidate_score
        self.metrics["dual_source_decision"] = candidate_score.get("decision", "")
        self.metrics["dual_source_rewrite_recommended"] = candidate_score.get("decision") in {
            "rewrite_recommended",
            "reject_recommended",
        }
        self.metrics.setdefault("candidate_score_history", []).append({
            "stage": "llm_generated",
            **candidate_score,
        })
        self._mark_failed_plan_blocker(llm_steps, "llm_generated")

    def _before_llm_plan_finalized(self, llm_steps: list[dict]) -> list[dict]:
        llm_steps = super()._before_llm_plan_finalized(llm_steps)
        if not self.inject_failed_plan_for_test:
            return llm_steps
        if self.memory_policy != "hierarchical":
            return llm_steps
        for entry, _score in self._last_retrieved_experiences:
            if not _is_failed(entry):
                continue
            injected_steps = _failed_entry_to_llm_steps(entry)
            if not injected_steps:
                continue
            self.metrics["injected_failed_plan_for_test"] = True
            self.metrics["injected_failed_experience_id"] = getattr(entry, "experience_id", "")
            print(
                "  [TEST] 注入 failed_memory 方案用于验证 blocker: "
                f"{self.metrics['injected_failed_experience_id']}"
            )
            return injected_steps
        print("  [TEST][WARN] 未找到可注入的 failed_memory 方案。")
        return llm_steps

    def _after_executed_steps_selected(self, executed_steps: list[dict]) -> None:
        candidate_score = score_candidate_plan(executed_steps, self._last_retrieved_experiences)
        self.metrics["selected_candidate_score"] = candidate_score
        self.metrics.setdefault("candidate_score_history", []).append({
            "stage": "executed_selected",
            **candidate_score,
        })
        self._mark_failed_plan_blocker(executed_steps, "executed_selected")
        self._mark_repeated_failure(executed_steps)

    def _generate_recovery_candidates(
        self,
        *,
        task_history: list[dict[str, Any]],
        image_paths: list[str],
        experience_image_paths: list[str],
        experiences: list[tuple[object, float]],
        target_observation_status: str,
        gripper_status: str,
    ) -> list[dict[str, Any]]:
        if self.recovery_candidate_count <= 1:
            steps = plan_recovery(
                task_history=task_history,
                image_paths=image_paths,
                experience_image_paths=experience_image_paths,
                target="apple",
                experiences=experiences,
                condition=self.condition,
                strategy_family=self.strategy_family,
                prompt_profile=self.prompt_profile,
                scenario_id=self.scenario_id,
                condition_id=self.condition_id,
                failure_family=self.failure_family,
                condition_name=self.condition_spec.name if self.condition_spec else "",
                task_stage=self.condition_spec.task_stage if self.condition_spec else "",
                injection_stage=self.condition_spec.injection_stage if self.condition_spec else "",
                success_criteria=self.condition_spec.success_criteria if self.condition_spec else "",
                target_observation_status=target_observation_status,
                gripper_status=gripper_status,
            )
            return [{"candidate_id": 1, "steps": steps}] if steps else []

        candidates = plan_recovery_candidates(
            task_history=task_history,
            image_paths=image_paths,
            experience_image_paths=experience_image_paths,
            target="apple",
            experiences=experiences,
            condition=self.condition,
            strategy_family=self.strategy_family,
            prompt_profile=self.prompt_profile,
            scenario_id=self.scenario_id,
            condition_id=self.condition_id,
            failure_family=self.failure_family,
            condition_name=self.condition_spec.name if self.condition_spec else "",
            task_stage=self.condition_spec.task_stage if self.condition_spec else "",
            injection_stage=self.condition_spec.injection_stage if self.condition_spec else "",
            success_criteria=self.condition_spec.success_criteria if self.condition_spec else "",
            target_observation_status=target_observation_status,
            gripper_status=gripper_status,
            candidate_count=self.recovery_candidate_count,
        )
        self.recovery_candidates_already_selected = True
        return candidates

    def _maybe_rewrite_blocked_plan(
        self,
        *,
        llm_steps: list[dict],
        image_paths: list[str],
        experience_image_paths: list[str],
        experiences: list[tuple[object, float]],
        **_unused: Any,
    ) -> list[dict]:
        dual_source_critic = self.memory_policy == "dual_source_gap_critic"
        if not self.enable_failed_plan_rewrite and not dual_source_critic:
            return llm_steps
        if self.memory_policy not in {"hierarchical", "dual_source_gap_critic"}:
            return llm_steps
        blocker_matches = [
            match
            for match in self.metrics.get("failed_plan_blocker_matches", [])
            if match.get("stage") == "llm_generated"
        ]
        if dual_source_critic and self.metrics.get("dual_source_rewrite_recommended"):
            blocker_matches = blocker_matches or self._dual_source_risk_blocker_matches(
                llm_steps,
                stage="llm_generated",
            )
        if not blocker_matches:
            return llm_steps

        self.metrics["failed_plan_rewrite_triggered"] = True
        if dual_source_critic:
            self.metrics["dual_source_rewrite_triggered"] = True
        for match in blocker_matches:
            match["rewrite_triggered"] = True

        if dual_source_critic:
            print("  双源风险评分建议重写，触发一次 LLM 重写...")
        else:
            print("  failed_memory blocker 命中，触发一次 LLM 重写...")
        try:
            rewritten_steps = plan_recovery(
                task_history=self._task_history,
                image_paths=image_paths,
                experience_image_paths=experience_image_paths,
                target="apple",
                experiences=experiences,
                condition=self.condition,
                blocker_matches=blocker_matches,
                strategy_family=self.strategy_family,
                prompt_profile=self.prompt_profile,
                scenario_id=self.scenario_id,
                condition_id=self.condition_id,
                failure_family=self.failure_family,
                condition_name=self.condition_spec.name if self.condition_spec else "",
                task_stage=self.condition_spec.task_stage if self.condition_spec else "",
                injection_stage=self.condition_spec.injection_stage if self.condition_spec else "",
                success_criteria=self.condition_spec.success_criteria if self.condition_spec else "",
            )
        except Exception as exc:
            self.metrics["failed_plan_rewrite_error"] = str(exc)
            print(f"  [WARN] failed_memory 重写失败: {exc}")
            return llm_steps

        if not rewritten_steps:
            self.metrics["failed_plan_rewrite_error"] = "empty_rewrite_plan"
            print("  [WARN] failed_memory 重写返回空计划，保留原计划。")
            return llm_steps

        post_matches = self._mark_failed_plan_blocker(rewritten_steps, "llm_rewritten")
        rewritten_score = score_candidate_plan(rewritten_steps, self._last_retrieved_experiences)
        self.metrics["rewritten_candidate_score"] = rewritten_score
        self.metrics.setdefault("candidate_score_history", []).append({
            "stage": "llm_rewritten",
            **rewritten_score,
        })
        self.metrics["failed_plan_rewrite_steps"] = rewritten_steps
        self.metrics["failed_plan_rewrite_success"] = not bool(post_matches)
        self._count_plan_quality_issues(rewritten_steps)
        if post_matches:
            print("  [WARN] 重写方案仍与 failed_memory 高度相似，保留重写方案但记录风险。")
        else:
            print("  failed_memory 重写完成，未再次命中 blocker。")
        return rewritten_steps

    def _validate_recovery_candidate_plan(self, steps: list[dict[str, Any]], candidate_id: int) -> dict[str, Any] | None:
        actions = [str(step.get("action") or "") for step in steps if isinstance(step, dict)]
        if not steps:
            return {"candidate_id": candidate_id, "rejected": True, "reject_reason": "empty_candidate"}
        if any(action not in allowed_actions(self.scenario_id) for action in actions):
            return {"candidate_id": candidate_id, "rejected": True, "reject_reason": "invalid_skill"}
        return None

    def _sandbox_score_candidate(
        self,
        *,
        candidate_id: int,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        target_pos = self.data.body(self.apple_body_id).xpos.copy()
        actions = [str(step.get("action") or "") for step in steps if isinstance(step, dict)]
        intended_task_attempt = "move_lifted_object_to" in actions
        wrapper = CandidateSandbox(self.noise_scale, scene_xml=self.scene_xml)
        virtual_scene = None
        z_before = 0.0
        z_after = 0.0
        step_trace: list[dict[str, Any]] = []
        try:
            virtual_scene = wrapper.clone_experiment_scene(self, enable_viewer=False)
            z_before = float(virtual_scene.data.body(virtual_scene.apple_body_id).xpos[2])
            sandbox_success, step_trace = self._execute_steps_in_virtual(
                virtual_scene,
                steps,
                target_pos,
                candidate_id=candidate_id,
            )
            z_after = float(virtual_scene.data.body(virtual_scene.apple_body_id).xpos[2])
            pinch_distance = float(np.linalg.norm(
                virtual_scene.data.body(virtual_scene.apple_body_id).xpos
                - virtual_scene.data.site_xpos[virtual_scene.pinch_site_id]
            ))
            z_change = float(z_after - z_before)
            task_check = next(
                (
                    item.get("after") or {}
                    for item in reversed(step_trace)
                    if isinstance(item, dict) and item.get("action") == "_sandbox_task_success_check"
                ),
                {},
            )
            on_plate = task_check.get("on_plate")
            xy_dist_to_plate = task_check.get("xy_dist_to_plate")
            task_attempted = bool(task_check) or intended_task_attempt
            task_success = bool(on_plate) if on_plate is not None else False
            return {
                "candidate_id": candidate_id,
                "sandbox_success": bool(sandbox_success),
                "success": bool(sandbox_success),
                "task_success": bool(sandbox_success),
                "task_attempted": task_attempted,
                "recovery_only": not task_attempted,
                "z_before": z_before,
                "z_after": z_after,
                "z_change": z_change,
                "pinch_distance": pinch_distance,
                "on_plate": on_plate,
                "xy_dist_to_plate": xy_dist_to_plate,
                "score": None,
                "llm_score": None,
                "target_source": "current_mujoco_state_clone",
                "step_trace": step_trace,
                "keyframes": [
                    item.get("keyframe")
                    for item in step_trace
                    if isinstance(item, dict) and item.get("keyframe")
                ],
            }
        except Exception as exc:
            return {
                "candidate_id": candidate_id,
                "sandbox_success": False,
                "success": False,
                "task_success": False,
                "task_attempted": False,
                "recovery_only": not intended_task_attempt,
                "score": None,
                "llm_score": None,
                "reason": str(exc),
                "z_before": z_before,
                "z_after": z_after,
                "step_trace": step_trace,
            }
        finally:
            if virtual_scene is not None:
                virtual_scene.close()

    def _critic_candidate_failures(
        self,
        *,
        sandbox_results: list[dict[str, Any]],
        accepted: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_candidate = {int(item["candidate_id"]): item.get("steps", []) for item in accepted}
        critic_results: list[dict[str, Any]] = []
        for result in sandbox_results:
            if not isinstance(result, dict):
                continue
            if _candidate_task_success(result):
                continue
            candidate_id = int(result.get("candidate_id") or 0)
            candidate_metrics = dict(self.metrics)
            candidate_metrics.update({
                "candidate_sandbox_results": [result],
                "selected_candidate_id": candidate_id,
                "candidate_sandbox_final_result": result,
                "candidate_sandbox_final_success": bool(result.get("sandbox_success")),
                "virtual_validation_success": bool(result.get("sandbox_success")),
                "task_success": False,
                "executed_plan_source": "candidate_mujoco_evaluation_for_critic",
                "llm_recovery_steps": by_candidate.get(candidate_id, []),
            })
            try:
                critic = critique_ur5e_failure_experience(
                    method=self.method,
                    memory_policy=self.memory_policy,
                    metrics=candidate_metrics,
                    task_history=getattr(self, "_task_history", []),
                    recovery_steps=_candidate_trace_steps_for_critic(result) or by_candidate.get(candidate_id, []),
                    retrieved_memories=self.metrics.get("retrieved_memories") or [],
                )
            except Exception as exc:
                critic = {
                    "enabled": True,
                    "error": str(exc),
                    "failure_stage": "unknown",
                    "failure_type": "候选失败critic异常",
                    "root_cause": str(exc),
                }
            result["candidate_failure_critic"] = critic
            critic_results.append({
                "candidate_id": candidate_id,
                "critic": critic,
            })
        return critic_results

    def _select_candidate_with_llm(
        self,
        *,
        sandbox_results: list[dict[str, Any]],
        accepted: list[dict[str, Any]],
        max_retries: int = 1,
    ) -> dict[str, Any]:
        accepted_steps = {int(item["candidate_id"]): item.get("steps", []) for item in accepted}
        candidate_payload: list[dict[str, Any]] = []
        for result in sandbox_results:
            candidate_id = int(result.get("candidate_id") or 0)
            critic = result.get("candidate_failure_critic") if isinstance(result.get("candidate_failure_critic"), dict) else {}
            candidate_payload.append({
                "candidate_id": candidate_id,
                "steps": accepted_steps.get(candidate_id, []),
                "physical_execution": {
                    "sandbox_success": result.get("sandbox_success"),
                    "success": result.get("success"),
                    "task_attempted": result.get("task_attempted"),
                    "task_success": result.get("task_success"),
                    "recovery_only": result.get("recovery_only"),
                    "z_before": result.get("z_before"),
                    "z_after": result.get("z_after"),
                    "z_change": result.get("z_change"),
                    "pinch_distance": result.get("pinch_distance"),
                    "on_plate": result.get("on_plate"),
                    "xy_dist_to_plate": result.get("xy_dist_to_plate"),
                    "reason": result.get("reason", ""),
                },
                "keyframes": result.get("keyframes") if isinstance(result.get("keyframes"), list) else [],
                "step_trace": _compact_trace_for_selector(result),
                "failure_critic": {
                    "failure_stage": critic.get("failure_stage", ""),
                    "failure_type": critic.get("failure_type", ""),
                    "root_cause": critic.get("root_cause", ""),
                    "corrective_direction": critic.get("corrective_direction", ""),
                    "missing_phases": critic.get("missing_phases", []),
                    "parameter_failure_summary": critic.get("parameter_failure_summary", {}),
                    "error": critic.get("error", ""),
                },
            })

        prompt = f"""
你是 UR5e MuJoCo 候选恢复执行裁判。所有候选已经在 MuJoCo 中实际执行过。
请只根据候选的动作参数、物理执行结果、step_trace、keyframe 路径和候选失败 critic，给每个候选打分并选择最终要采用的 candidate。

总体目标：异常恢复后继续完成任务闭环，使目标物体完成抓取、提升、搬运、释放到目标区域，并让机械臂回到安全状态。不要生成新计划。

硬性要求：
1. 不要按固定技能序列打分，不要因为缺某个手写技能名直接判死。
2. 必须优先考虑最终物理效果和任务闭环，而不是动作数量或动作名称看起来是否完整。
3. recovery_only=true 表示该候选没有完成任务闭环检查，只能作为恢复候选；如果其他候选有更好的完整任务物理证据，应优先完整任务候选。
4. 如果所有候选都失败，也必须选择一个物理效果最好、最接近完成目标、最有继续执行价值的候选。
5. candidate_scores.score 使用 0 到 100 的数字，分数由你根据物理证据判断。
6. 如果存在 task_attempted=true 的候选，不能选择 recovery_only=true 的候选作为最终输出；最终输出必须服务于完整任务闭环，而不是只恢复抓取/提升。
7. success/sandbox_success/task_success 含义完全相同，都只表示完整任务闭环成功；单纯抓取或抬起不能当作成功。

候选执行结果：
{json.dumps(_convert_numpy(candidate_payload), ensure_ascii=False, indent=2)}

只输出 JSON 对象，格式：
{{
  "selected_candidate_id": 1,
  "candidate_scores": [
    {{"candidate_id": 1, "score": 0, "task_success": false, "reason": "中文原因"}}
  ],
  "ranking_reason": "中文说明为什么选择该候选",
  "selection_confidence": 0.0,
  "critic_summary_used": true
}}
"""
        provider = resolve_ur5e_provider()
        model = resolve_ur5e_critic_model()
        raw_responses: list[str] = []
        last_error = ""
        valid_ids = {int(item.get("candidate_id") or 0) for item in sandbox_results if isinstance(item, dict)}
        task_attempted_ids = {
            int(item.get("candidate_id") or 0)
            for item in sandbox_results
            if isinstance(item, dict) and bool(item.get("task_attempted"))
        }
        for attempt in range(max_retries + 1):
            try:
                raw = invoke_llm(
                    prompt,
                    provider=provider,
                    model=model,
                    system_prompt=f"你是候选执行裁判。{JSON_ONLY_LINE}",
                    temperature=0.0,
                )
                raw_responses.append(raw)
                payload = parse_json_payload(raw)
                if not isinstance(payload, dict):
                    raise ValueError("selector_response_not_object")
                selected_id = int(payload.get("selected_candidate_id"))
                if selected_id not in valid_ids:
                    raise ValueError(f"selector_selected_unknown_candidate:{selected_id}")
                selected_result = next(
                    item for item in sandbox_results
                    if isinstance(item, dict) and int(item.get("candidate_id") or 0) == selected_id
                )
                if task_attempted_ids and bool(selected_result.get("recovery_only")):
                    raise ValueError(
                        "selector_selected_recovery_only_despite_task_attempted_candidates"
                    )
                scores = payload.get("candidate_scores")
                if not isinstance(scores, list):
                    raise ValueError("selector_candidate_scores_not_list")
                return {
                    "enabled": True,
                    "provider": provider,
                    "model": model,
                    "attempts": attempt + 1,
                    "selected_candidate_id": selected_id,
                    "candidate_scores": scores,
                    "ranking_reason": str(payload.get("ranking_reason") or ""),
                    "selection_confidence": payload.get("selection_confidence"),
                    "critic_summary_used": bool(payload.get("critic_summary_used", False)),
                    "raw_response": raw,
                }
            except Exception as exc:
                last_error = str(exc)
        return {
            "enabled": True,
            "provider": provider,
            "model": model,
            "attempts": max_retries + 1,
            "error": last_error,
            "raw_responses": raw_responses,
        }

    def _select_recovery_candidate(
        self,
        *,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.metrics["candidate_plans"] = candidates
        accepted: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for fallback_index, candidate in enumerate(candidates, start=1):
            candidate_id = int(candidate.get("candidate_id") or fallback_index)
            steps = candidate.get("steps") if isinstance(candidate.get("steps"), list) else []
            rejection = self._validate_recovery_candidate_plan(steps, candidate_id)
            if rejection:
                rejection["steps"] = steps
                rejection["plan_signature"] = canonical_action_signature_from_steps(steps)
                rejections.append(rejection)
                continue
            accepted.append({"candidate_id": candidate_id, "steps": steps})

        self.metrics["candidate_rejections"] = rejections
        if not accepted:
            self.metrics["candidate_selection_blocked"] = True
            self.metrics["candidate_selection_block_reason"] = "all_candidates_rejected"
            self.metrics["candidate_plan_failure_type"] = "all_candidates_rejected_by_plan_validator"
            self.metrics["candidate_plan_failure_evidence"] = {
                "candidate_count": len(candidates),
                "rejections": rejections,
            }
            self.candidate_selection_blocked = True
            self.candidate_selection_block_reason = "all_candidates_rejected"
            return []

        if not self.execute_recovery_candidate_validation:
            selected = accepted[0]
            self.metrics["selected_candidate_id"] = selected["candidate_id"]
            self.metrics["candidate_selection_reason"] = "first_non_rejected_candidate"
            return selected["steps"]

        sandbox_results = [
            self._sandbox_score_candidate(
                candidate_id=int(candidate["candidate_id"]),
                steps=candidate["steps"],
            )
            for candidate in accepted
        ]
        self.metrics["candidate_sandbox_results"] = sandbox_results
        candidate_failure_critics = self._critic_candidate_failures(
            sandbox_results=sandbox_results,
            accepted=accepted,
        )
        self.metrics["candidate_failure_critics"] = candidate_failure_critics
        selector_result = self._select_candidate_with_llm(
            sandbox_results=sandbox_results,
            accepted=accepted,
        )
        self.metrics["llm_candidate_selector_result"] = selector_result
        if selector_result.get("error"):
            self.metrics["candidate_selection_block_reason"] = "llm_selector_failed"
            self.metrics["candidate_plan_failure_type"] = "llm_selector_failed"
            self.metrics["candidate_plan_failure_evidence"] = {
                "candidate_count": len(candidates),
                "sandbox_results": sandbox_results,
                "rejections": rejections,
                "llm_selector_result": selector_result,
            }
            self.metrics["candidate_selection_blocked"] = True
            self.candidate_selection_blocked = True
            self.candidate_selection_block_reason = "llm_selector_failed"
            self.metrics["executed_plan_source"] = "blocked_by_llm_selector_failure"
            return []

        score_by_candidate: dict[int, dict[str, Any]] = {}
        for item in selector_result.get("candidate_scores") or []:
            if not isinstance(item, dict):
                continue
            try:
                score_by_candidate[int(item.get("candidate_id"))] = item
            except (TypeError, ValueError):
                continue
        for result in sandbox_results:
            candidate_id = int(result.get("candidate_id") or 0)
            llm_score_item = score_by_candidate.get(candidate_id, {})
            result["llm_selector_reason"] = llm_score_item.get("reason", "")
            result["llm_task_success"] = llm_score_item.get("task_success")
            try:
                result["llm_score"] = float(llm_score_item.get("score"))
                result["score"] = result["llm_score"]
            except (TypeError, ValueError):
                result["llm_score"] = None
                result["score"] = None

        selected_id = int(selector_result["selected_candidate_id"])
        selected = next(candidate for candidate in accepted if int(candidate["candidate_id"]) == selected_id)
        selected_result = next(item for item in sandbox_results if int(item.get("candidate_id") or 0) == selected_id)
        selected_result["selected_for_output"] = True
        self.metrics["selected_candidate_id"] = selected_id
        self.metrics["candidate_selection_reason"] = "llm_selector"
        self.metrics["candidate_selection_ranking_reason"] = selector_result.get("ranking_reason", "")
        self.metrics["candidate_sandbox_final_execution"] = True
        self.metrics["candidate_sandbox_final_success"] = bool(selected_result.get("sandbox_success"))
        self.metrics["candidate_sandbox_final_result"] = selected_result
        self.metrics["executed_plan_source"] = (
            "sandbox_candidate_selected_success"
            if selected_result.get("sandbox_success")
            else "sandbox_llm_selected_failed"
        )
        return selected["steps"]

    def _dual_source_risk_blocker_matches(self, candidate_steps: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
        signature = canonical_action_signature_from_steps(candidate_steps)
        if not signature or signature == "[]":
            return []
        candidate_actions = _signature_actions(signature)
        matches: list[dict[str, Any]] = []
        for entry, score in self._last_retrieved_experiences:
            if not _is_risk_entry(entry):
                continue
            entry_signature = canonical_action_signature_from_entry(entry)
            overlap = _lcs_ratio(candidate_actions, _signature_actions(entry_signature))
            if overlap < 0.5 and not _is_failed(entry):
                continue
            failure_taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
            match = {
                "stage": stage,
                "experience_id": getattr(entry, "experience_id", ""),
                "partition": _partition(entry),
                "score": float(score),
                "overlap": overlap,
                "exact_signature_match": bool(entry_signature and entry_signature == signature),
                "candidate_signature": signature,
                "failed_signature": entry_signature,
                "failure_stage": failure_taxonomy.get("failure_stage", ""),
                "failure_type": failure_taxonomy.get("failure_type", getattr(entry.result, "failure_reason", "")),
                "cluster_id": failure_taxonomy.get("cluster_id", ""),
                "blocked": True,
                "rewrite_triggered": False,
                "source": "dual_source_risk",
                "gap_uncertainty": estimate_gap_uncertainty(entry),
                "critic_risk": critic_risk_score(entry),
            }
            matches.append(match)
        if matches:
            existing = list(self.metrics.get("failed_plan_blocker_matches") or [])
            existing.extend(matches)
            self.metrics["failed_plan_blocked"] = True
            self.metrics["failed_plan_blocker_matches"] = existing
        return matches

    def save_experience(self):
        if self.experience_write_path is None:
            return None
        if self.experience_read_path and self.experience_read_path.exists():
            lib = MemoryV3Library.load(self.experience_read_path)
        else:
            lib = MemoryV3Library()
        entry = self._build_experience_entry()
        candidate_failure_critics = self.metrics.get("candidate_failure_critics")
        llm_selector_result = self.metrics.get("llm_candidate_selector_result")
        if candidate_failure_critics or llm_selector_result:
            taxonomy = dict(getattr(entry, "failure_taxonomy", {}) or {})
            candidate_failures = taxonomy.get("candidate_failures") if isinstance(taxonomy.get("candidate_failures"), dict) else {}
            if candidate_failure_critics:
                candidate_failures["physical_failure_critics"] = candidate_failure_critics
            if llm_selector_result:
                candidate_failures["llm_selector"] = {
                    "selected_candidate_id": llm_selector_result.get("selected_candidate_id"),
                    "candidate_scores": llm_selector_result.get("candidate_scores", []),
                    "ranking_reason": llm_selector_result.get("ranking_reason", ""),
                    "selection_confidence": llm_selector_result.get("selection_confidence"),
                    "error": llm_selector_result.get("error", ""),
                }
            taxonomy["candidate_failures"] = candidate_failures
            entry.failure_taxonomy = taxonomy
        is_failure_entry = getattr(entry, "status", "") == "failure" or not bool(getattr(entry.result, "success", False))
        llm_critic_result: dict[str, Any] = {}
        rule_critic_result: dict[str, Any] = {}
        if is_failure_entry:
            try:
                llm_critic_result = critique_ur5e_failure_experience(
                    method=self.method,
                    memory_policy=self.memory_policy,
                    metrics=self.metrics,
                    task_history=getattr(self, "_task_history", []),
                    recovery_steps=_physical_failure_steps_for_critic(self.metrics),
                    retrieved_memories=self.metrics.get("retrieved_memories") or [],
                )
                self.metrics["failure_experience_critic"] = llm_critic_result
                if llm_critic_result.get("enabled"):
                    taxonomy = dict(getattr(entry, "failure_taxonomy", {}) or {})
                    taxonomy["llm_critic"] = llm_critic_result
                    if llm_critic_result.get("failure_stage"):
                        taxonomy["failure_stage"] = llm_critic_result["failure_stage"]
                    if llm_critic_result.get("failure_type"):
                        taxonomy["failure_type"] = llm_critic_result["failure_type"]
                    if llm_critic_result.get("failed_predicates"):
                        taxonomy["failed_predicates"] = llm_critic_result["failed_predicates"]
                    if llm_critic_result.get("failure_evidence"):
                        taxonomy["failure_evidence"] = llm_critic_result["failure_evidence"]
                    if llm_critic_result.get("corrective_direction"):
                        taxonomy["corrective_direction"] = llm_critic_result["corrective_direction"]
                    if llm_critic_result.get("missing_phases"):
                        taxonomy["missing_phases"] = llm_critic_result["missing_phases"]
                    if getattr(entry, "retrieval_key", None) is not None:
                        entry.retrieval_key["failure_type"] = taxonomy.get("failure_type", "")
                    entry.failure_taxonomy = taxonomy
                    if llm_critic_result.get("failure_type") and not entry.result.failure_reason and not llm_critic_result.get("error"):
                        entry.result.failure_reason = llm_critic_result["failure_type"]
            except Exception as exc:
                self.metrics["failure_experience_critic_error"] = str(exc)
                print(f"  [WARN] 失败经验 critic 生成失败: {exc}")
        # Deterministic rule critic — runs for all failed entries, not just hierarchical
        if is_failure_entry:
            try:
                rule_critic_result = deterministic_rule_critic(self.metrics)
                self.metrics["failure_rule_critic"] = rule_critic_result
                if rule_critic_result.get("enabled") and rule_critic_result.get("rule_flags"):
                    taxonomy = dict(getattr(entry, "failure_taxonomy", {}) or {})
                    taxonomy["rule_critic"] = rule_critic_result
                    entry.failure_taxonomy = taxonomy
            except Exception as exc:
                self.metrics["failure_rule_critic_error"] = str(exc)
        critic_result = build_critic_result(
            rule_result=rule_critic_result,
            llm_result=llm_critic_result,
            is_failure=is_failure_entry,
        )
        entry.critic_result = critic_result
        self.metrics["critic_result"] = critic_result
        if getattr(entry, "retrieval_key", None) is not None:
            entry.retrieval_key = build_retrieval_key(entry)
        if self.experience_save_mode == "none":
            self.metrics["experience_saved"] = False
            self.metrics["experience_save_skipped_reason"] = "save_mode_none"
        elif self.experience_save_mode == "success_only" and is_failure_entry:
            self.metrics["experience_saved"] = False
            self.metrics["experience_save_skipped_reason"] = "failure_not_saved_success_only"
        else:
            deduped_existing = None
            if self.dedupe_failure_memory and is_failure_entry:
                new_key = _failure_dedupe_key(entry)
                for existing_entry in list(getattr(lib, "_entries", [])):
                    if _is_failed(existing_entry) and _failure_dedupe_key(existing_entry) == new_key:
                        deduped_existing = existing_entry
                        break
            if deduped_existing is not None:
                taxonomy = dict(getattr(deduped_existing, "failure_taxonomy", {}) or {})
                new_taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
                if isinstance(new_taxonomy, dict):
                    for key in (
                        "llm_critic",
                        "rule_critic",
                        "failure_stage",
                        "failure_type",
                        "failed_predicates",
                        "failure_evidence",
                        "corrective_direction",
                        "missing_phases",
                    ):
                        value = new_taxonomy.get(key)
                        if value:
                            taxonomy[key] = value
                taxonomy["dedupe_count"] = int(taxonomy.get("dedupe_count") or 1) + 1
                taxonomy["last_duplicate_trial_id"] = self.trial_id
                taxonomy["last_duplicate_metrics"] = {
                    "task_success_criteria": self.metrics.get("task_success_criteria", {}),
                    "candidate_rejections": self.metrics.get("candidate_rejections", []),
                    "candidate_sandbox_results": self.metrics.get("candidate_sandbox_results", []),
                    "selected_candidate_id": self.metrics.get("selected_candidate_id"),
                }
                deduped_existing.failure_taxonomy = taxonomy
                deduped_existing.critic_result = getattr(entry, "critic_result", getattr(deduped_existing, "critic_result", {}))
                if getattr(deduped_existing, "retrieval_key", None) is not None:
                    deduped_existing.retrieval_key = build_retrieval_key(deduped_existing)
                self.metrics["experience_saved"] = True
                self.metrics["experience_save_skipped_reason"] = "deduped_failure_memory"
                self.metrics["deduped_failure_experience_id"] = getattr(deduped_existing, "experience_id", "")
            else:
                lib.upsert(entry)
            try:
                consolidation_report = lib.consolidate()
                if consolidation_report.get("action") == "consolidate":
                    self.metrics["stm_consolidation"] = consolidation_report
            except Exception as exc:
                print(f"  [WARN] STM/LTM consolidation 失败: {exc}")
            if is_failure_entry and deduped_existing is None:
                try:
                    existing = lib._entries if hasattr(lib, '_entries') else []
                    _get_failure_clusterer().assign_new(entry, existing)
                except Exception as exc:
                    print(f"  [WARN] 失败聚类分配失败: {exc}")
            if deduped_existing is None:
                self.metrics["experience_saved"] = True
                self.metrics["experience_save_skipped_reason"] = ""
        self.experience_write_path.parent.mkdir(parents=True, exist_ok=True)
        lib.save(self.experience_write_path)
        saved_entry = deduped_existing if "deduped_existing" in locals() and deduped_existing is not None else entry
        taxonomy = getattr(saved_entry, "failure_taxonomy", {}) or {}
        print(
            "  经验写入: "
            f"path={self.experience_write_path} "
            f"status={getattr(saved_entry, 'status', '')} "
            f"partition={getattr(saved_entry, 'memory_partition', '')} "
            f"task_success={getattr(saved_entry.result, 'task_success', None)} "
            f"success={getattr(saved_entry.result, 'success', None)} "
            f"llm_critic={bool(isinstance(taxonomy, dict) and taxonomy.get('llm_critic'))}"
        )
        # Save entry's keyframe images to visual index
        try:
            if self.visual_index is not None and hasattr(entry, "keyframes"):
                visual_dir = _visual_index_dir(self)
                if visual_dir is not None:
                    img_paths = _entry_keyframe_abs_paths(entry, self)
                    if img_paths:
                        self.visual_index.add(entry.experience_id, img_paths)
                        self.visual_index.save(visual_dir)
        except Exception as exc:
            print(f"  [WARN] 保存视觉索引失败: {exc}")
        return entry


def _resolve_method(method: str, condition: str | None, memory_policy: str | None) -> tuple[str, str]:
    defaults = METHOD_DEFAULTS[method]
    return condition or defaults["condition"], memory_policy or defaults["memory_policy"]


def _copy_if_exists(src: Path | None, dst: Path) -> None:
    if src is not None and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-invasive experiment method runner")
    parser.add_argument("--method", choices=sorted(METHOD_DEFAULTS), required=True)
    parser.add_argument("--memory-policy", choices=sorted(MEMORY_POLICIES), default=None)
    parser.add_argument("--condition", choices=["direct"], default=None)
    parser.add_argument("--experience-read", type=str, default=None)
    parser.add_argument("--experience-write", type=str, default=None)
    parser.add_argument("--save", type=str, default="results/method_result.json")
    parser.add_argument("--save-plan", type=str, default="results/method_plan.json")
    parser.add_argument("--scene-xml", type=str, default=None, help="MuJoCo scene XML, default ur5e_mujoco/scene/scene.xml")
    parser.add_argument("--trial-id", type=str, default="trial_000")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--condition-id", type=str, default="", help="UR5E benchmark condition id, e.g. U2-1")
    parser.add_argument("--anomaly", choices=[
        "grasp_miss",
        "slip",
        "incipient_slip",
        "object_displaced",
        "gripper_fail",
        "collision",
        "wrong_object",
    ], default="grasp_miss")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--object-displaced-dx", type=float, default=OBJECT_DISPLACED_DX)
    parser.add_argument("--object-displaced-dy", type=float, default=OBJECT_DISPLACED_DY)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-inject", action="store_true")
    parser.add_argument("--use-memory-keyframes", action="store_true", help="Attach top retrieved experience keyframes to the planning prompt")
    parser.add_argument("--memory-keyframe-top-k", type=int, default=2)
    parser.add_argument("--memory-index-dir", type=str, default=None)
    parser.add_argument("--strategy-family", type=str, default="", help="Prompt-level recovery strategy family for strategy-diverse memory growth")
    parser.add_argument(
        "--experience-save-mode",
        choices=["all", "success_only", "none"],
        default="all",
        help="Control which current-trial experiences are appended to experience-write",
    )
    parser.add_argument(
        "--enable-failed-plan-rewrite",
        action="store_true",
        help="Rewrite once when generated plan matches retrieved failed_memory",
    )
    parser.add_argument(
        "--inject-failed-plan-for-test",
        action="store_true",
        help="Targeted test only: replace initial LLM plan with retrieved failed_memory plan",
    )
    parser.add_argument("--recovery-candidate-count", type=int, default=1)
    parser.add_argument("--execute-recovery-candidate-validation", action="store_true")
    parser.add_argument("--failed-memory-hard-block", action="store_true")
    parser.add_argument("--dedupe-failure-memory", action="store_true")
    args = parser.parse_args()

    condition, memory_policy = _resolve_method(args.method, args.condition, args.memory_policy)
    condition_spec = get_condition_spec(args.condition_id) if args.condition_id else None
    anomaly = condition_spec.legacy_anomaly_type if condition_spec is not None else args.anomaly
    if args.seed is not None:
        np.random.seed(args.seed)

    save_path = _resolve_local_path(args.save)
    plan_path = _resolve_local_path(args.save_plan)
    read_path = _resolve_local_path(args.experience_read)
    write_path = _resolve_local_path(args.experience_write)

    if memory_policy == "none":
        read_arg = None
    else:
        read_arg = str(read_path) if read_path is not None else None

    print("\n" + "=" * 50)
    print("实验方法配置")
    print(f"  method: {args.method}")
    print(f"  condition: {condition}")
    print(f"  memory_policy: {memory_policy}")
    print(f"  anomaly: {anomaly}")
    print(f"  condition_id: {args.condition_id}")
    print(f"  seed: {args.seed}")
    print(f"  object_displaced_dx: {args.object_displaced_dx}")
    print(f"  object_displaced_dy: {args.object_displaced_dy}")
    print(f"  enable_failed_plan_rewrite: {args.enable_failed_plan_rewrite}")
    print(f"  recovery_candidate_count: {args.recovery_candidate_count}")
    print(f"  execute_recovery_candidate_validation: {args.execute_recovery_candidate_validation}")
    print(f"  failed_memory_hard_block: {args.failed_memory_hard_block}")
    print(f"  dedupe_failure_memory: {args.dedupe_failure_memory}")
    print(f"  inject_failed_plan_for_test: {args.inject_failed_plan_for_test}")
    print(f"  memory_index_dir: {args.memory_index_dir}")
    print(f"  strategy_family: {args.strategy_family}")
    print(f"  experience_save_mode: {args.experience_save_mode}")
    print(f"  experience_read: {read_arg}")
    print(f"  experience_write: {write_path}")
    print(f"  scene_xml: {args.scene_xml or 'scene/scene.xml'}")
    print("=" * 50)

    exp = ExperimentMethodRunner(
        method=args.method,
        memory_policy=memory_policy,
        experience_read_path=read_arg,
        experience_write_path=str(write_path) if write_path is not None else None,
        trial_id=args.trial_id,
        seed=args.seed,
        use_memory_keyframes=args.use_memory_keyframes,
        memory_keyframe_top_k=args.memory_keyframe_top_k,
        enable_failed_plan_rewrite=args.enable_failed_plan_rewrite,
        inject_failed_plan_for_test=args.inject_failed_plan_for_test,
        recovery_candidate_count=args.recovery_candidate_count,
        execute_recovery_candidate_validation=args.execute_recovery_candidate_validation,
        failed_memory_hard_block=args.failed_memory_hard_block,
        dedupe_failure_memory=args.dedupe_failure_memory,
        memory_index_dir=args.memory_index_dir,
        strategy_family=args.strategy_family,
        experience_save_mode=args.experience_save_mode,
        save_path=str(save_path) if save_path is not None else None,
        scene_xml=args.scene_xml,
        enable_viewer=not args.no_viewer,
        condition=condition,
        noise_scale=args.noise_scale,
        save_plan=str(plan_path) if plan_path is not None else None,
        anomaly_type=anomaly,
        condition_id=args.condition_id or None,
        object_displaced_dx=args.object_displaced_dx,
        object_displaced_dy=args.object_displaced_dy,
    )
    try:
        metrics = exp.run(inject_anomaly=not args.no_inject)
        try:
            exp.save_experience()
        except Exception as exc:
            print(f"  [WARN] 经验写入失败，但保留本轮实验结果: {exc}")
            metrics["experience_saved"] = False
            metrics["experience_save_error"] = str(exc)
    finally:
        exp.close()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(_convert_numpy(metrics), f, indent=2)
        print(f"\n结果已保存到: {save_path}")
    if read_path is not None and save_path is not None:
        _copy_if_exists(read_path, save_path.parent / "experience_read.json")
    if write_path is not None and save_path is not None and write_path.exists():
        _copy_if_exists(write_path, save_path.parent / "experience_write.json")


if __name__ == "__main__":
    main()
