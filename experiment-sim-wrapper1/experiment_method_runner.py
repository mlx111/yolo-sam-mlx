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
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from llm_handler import critique_failure_experience, plan_recovery
from ur5e.experiment_config import (
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
    RECOVERY_SUCCESS_Z_CHANGE,
    SLIP_RECOVERY_LIFT_FROM_TABLE,
    SLIP_RECOVERY_PINCH_DISTANCE,
)
from memory.v3 import (
    MemoryV3Library,
    build_retrieval_key,
    canonical_action_signature_from_entry,
    canonical_action_signature_from_steps,
)
from memory.critic import build_critic_result
from memory.failure_cluster import FailureClusterer
from memory.scoring import critic_risk_score, estimate_gap_uncertainty, score_candidate_plan
from memory.visual_retrieval import VisualRetrievalIndex, _image_paths_from_entry
from run_experiment_v4 import ExperimentV4, ROOT, _convert_numpy, _resolve_local_path
from ur5e.anomaly_conditions import get_condition_spec
from ur5e.skills.registry import allowed_actions

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
    "sim_only_weak": {"condition": "sim_wrapper", "memory_policy": "none"},
    "sim_memory_weak": {"condition": "sim_wrapper", "memory_policy": "simulation_only"},
    "hierarchical_memory_weak": {"condition": "sim_wrapper", "memory_policy": "hierarchical"},
    "hierarchical_no_failed": {"condition": "sim_wrapper", "memory_policy": "no_failed"},
    "dual_source_gap_memory": {"condition": "sim_wrapper", "memory_policy": "dual_source_gap"},
    "dual_source_gap_critic": {"condition": "sim_wrapper", "memory_policy": "dual_source_gap_critic"},
}

MEMORY_POLICIES = {
    "none",
    "simulation_only",
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
    if result is not None and not bool(getattr(result, "recovery_success", getattr(result, "success", True))):
        return True
    return _partition(entry) == "failed_memory" or getattr(entry, "status", "") == "failure"


def _is_risk_entry(entry: Any) -> bool:
    return _is_failed(entry) or estimate_gap_uncertainty(entry) >= 0.8 or critic_risk_score(entry) >= 0.5


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
        if action == "gripper-action":
            action = f"{action}:{item.get('state', '')}"
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
    skill_sequence = getattr(entry, "skill_sequence", None)
    if skill_sequence:
        return [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in skill_sequence
            if isinstance(step, dict) and step.get("action")
        ]
    steps: list[dict[str, Any]] = []
    recovery_plan = getattr(entry, "recovery_plan", None)
    for step in getattr(recovery_plan, "steps", []) or []:
        step_type = str(getattr(step, "type", "")).strip()
        params = getattr(step, "params", {}) or {}
        if step_type == "gripper":
            command = str(params.get("command", "")).strip().lower()
            if command == "open":
                steps.append({"action": "gripper-action", "parameters": {"state": 0}})
            elif command == "close":
                steps.append({"action": "gripper-action", "parameters": {"state": 1}})
        elif step_type == "cartesian_move":
            label = str(params.get("label", "")).strip().lower()
            mapped = {
                "pregrasp": "move-pregrasp",
                "grasp": "move-grasp",
                "lift": "vertical-grasp",
            }.get(label)
            if mapped:
                steps.append({"action": mapped, "parameters": {}})
    return steps


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
        "LLM 输出中包含当前场景不可用的技能名称或参数格式错误。",
    ),
    "plan_blocked_by_failed_history": (
        "方案与历史失败模式匹配",
        "LLM 生成的方案与已知的失败经验动作序列高度重叠，被失败记忆拦截器拒绝执行。",
    ),
    # ── Virtual Validation ───────────────────────────────────────────
    "virtual_validation_failed": (
        "虚拟验证未通过",
        "恢复方案在影子仿真中执行失败，说明该方案在理想条件下也不可行。",
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
        "注入的感知位置偏移量（dx/dy 欧氏距离）超过了夹爪吸附距离上限（attach_max_distance），即使夹爪按感知位置移动也无法触碰到物体。这是导致抓取失败的根本原因。",
    ),
}


def deterministic_rule_critic(metrics: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    criteria = metrics.get("recovery_success_criteria") or {}
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

    tracked = criteria.get("tracked_apple")
    if tracked is False:
        flags.append({"rule": "apple_not_tracked", "stage": "recovery_execution",
                       "evidence": "tracked_apple=False during recovery execution"})

    # ── Lift ─────────────────────────────────────────────────────────
    lift_table = criteria.get("lift_from_table")
    min_lift = criteria.get("min_lift")
    if lift_table is not None and min_lift is not None and lift_table <= min_lift:
        flags.append({"rule": "object_not_lifted", "stage": "recovery_execution",
                       "evidence": f"lift_from_table={lift_table:.4f} <= min_lift={min_lift:.4f}"})

    z_change = criteria.get("z_change")
    min_z = metrics.get("recovery_success_z_change", 0.03)
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

    # ── Virtual Validation ───────────────────────────────────────────
    vv = metrics.get("virtual_validation_success")
    if vv is False:
        flags.append({"rule": "virtual_validation_failed", "stage": "virtual_validation",
                       "evidence": "recovery plan failed in shadow-simulation verification"})

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
                "evidence": f"perception offset {offset_euclidean:.3f}m > attach_max_distance {attach_max:.3f}m, "
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

    def _attach_body(self, body_name, *, max_distance: float | None = None):
        """Attach only when the gripper is physically close enough to the body.

        The base experiment uses attachment as a simplified grasp model.  For
        object_displaced, unconditional attachment makes a missed grasp look as
        if the object still followed the gripper, which invalidates the anomaly.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found")

        pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
        body_pos = self.data.body(bid).xpos.copy()
        distance = float(np.linalg.norm(body_pos - pinch_pos))
        if max_distance is None:
            max_distance = RECOVERY_ATTACH_MAX_DISTANCE if self._is_recovery_phase() else INITIAL_GRASP_ATTACH_MAX_DISTANCE
        max_distance = float(max_distance)

        self.metrics["last_attach_attempt"] = {
            "body": body_name,
            "distance": distance,
            "max_distance": max_distance,
            "accepted": distance <= max_distance,
            "phase": "recovery" if self._is_recovery_phase() else "initial",
        }

        if distance > max_distance:
            print(
                f"  → 未形成跟踪抓取: {body_name} 与 pinch 距离 {distance:.4f}m "
                f"> {max_distance:.4f}m"
            )
            self._detach_body()
            return False

        return super()._attach_body(body_name, max_distance=max_distance)

    def _evaluate_recovery_success(self, recovery_pos: np.ndarray | None) -> bool:
        if recovery_pos is None:
            self.metrics["recovery_success_criteria"] = {
                "type": "observed_z_lift",
                "success": False,
                "reason": "recovery_target_unavailable",
            }
            return False
        apple_pos = self.data.body(self.apple_body_id).xpos.copy()
        apple_z = float(apple_pos[2])

        if self.scenario_id == "U3":
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            tracked_apple = bool(self.grasp_tracking and self.tracked_body_id == self.apple_body_id)
            min_lift = SLIP_RECOVERY_LIFT_FROM_TABLE
            max_pinch_distance = SLIP_RECOVERY_PINCH_DISTANCE
            grasp_secured = bool(tracked_apple or pinch_distance < max_pinch_distance)
            success = bool(lift_from_table > min_lift and grasp_secured)
            self.metrics["recovery_success_criteria"] = {
                "type": "u3_gripper_recovered_and_lifted",
                "condition_id": self.condition_id,
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "tracked_apple": tracked_apple,
                "grasp_secured": grasp_secured,
                "min_lift": min_lift,
                "max_pinch_distance": max_pinch_distance,
                "success": success,
            }
            return success

        observed_z = float(np.asarray(recovery_pos, dtype=np.float64).reshape(3)[2])
        z_change = apple_z - observed_z

        if self.anomaly_type == "object_displaced":
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            tracked_apple = bool(self.grasp_tracking and self.tracked_body_id == self.apple_body_id)
            success = lift_from_table > OBJECT_DISPLACED_LIFT_FROM_TABLE and (
                tracked_apple or pinch_distance < OBJECT_DISPLACED_PINCH_DISTANCE
            )
            self.metrics["recovery_success_criteria"] = {
                "type": "object_displaced_lift_from_table",
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "tracked_apple": tracked_apple,
                "success": bool(success),
            }
            return bool(success)

        if self.anomaly_type in {"slip", "collision"}:
            baseline_z = float(self.metrics.get("apple_z_before_lift") or self.apple_initial_pos[2])
            lift_from_table = apple_z - baseline_z
            pinch_pos = self.data.site_xpos[self.pinch_site_id].copy()
            pinch_distance = float(np.linalg.norm(apple_pos - pinch_pos))
            tracked_apple = bool(self.grasp_tracking and self.tracked_body_id == self.apple_body_id)
            if self.anomaly_type == "slip":
                min_lift = SLIP_RECOVERY_LIFT_FROM_TABLE
                max_pinch_distance = SLIP_RECOVERY_PINCH_DISTANCE
                criteria_type = "slip_regrasp_lift_from_table"
            else:
                min_lift = COLLISION_RECOVERY_LIFT_FROM_TABLE
                max_pinch_distance = COLLISION_RECOVERY_PINCH_DISTANCE
                criteria_type = "collision_relocalize_lift_from_table"
            success = lift_from_table > min_lift and (
                tracked_apple or pinch_distance < max_pinch_distance
            )
            self.metrics["recovery_success_criteria"] = {
                "type": criteria_type,
                "apple_z": apple_z,
                "baseline_z": baseline_z,
                "lift_from_table": lift_from_table,
                "pinch_distance": pinch_distance,
                "tracked_apple": tracked_apple,
                "min_lift": min_lift,
                "max_pinch_distance": max_pinch_distance,
                "success": bool(success),
            }
            return bool(success)

        success = z_change > RECOVERY_SUCCESS_Z_CHANGE
        self.metrics["recovery_success_criteria"] = {
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
        if self.memory_policy == "simulation_only":
            selected = [
                (e, s)
                for e, s in candidates
                if _partition(e) in {"validated_memory", "simulation_memory"}
            ][:POLICY_POSITIVE_TOP_K]
        elif self.memory_policy == "hierarchical":
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
        unsafe = 0
        seen_close = False
        seen_grasp = False
        seen_vertical = False
        for step in steps:
            action = step.get("action", "")
            params = step.get("parameters", {}) or {}
            if action not in allowed:
                invalid += 1
            if action == "move-grasp":
                seen_grasp = True
            if action == "vertical-grasp":
                seen_vertical = True
                if not seen_close:
                    unsafe += 1
            if action == "gripper-action":
                state = params.get("state")
                if state not in {0, 1}:
                    invalid += 1
                if state == 1:
                    if not seen_grasp:
                        unsafe += 1
                    seen_close = True
                if state == 0 and seen_vertical:
                    unsafe += 1
        self.metrics["invalid_plan_count"] = invalid
        self.metrics["unsafe_gripper_action_count"] = unsafe

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

    def _maybe_rewrite_blocked_plan(
        self,
        *,
        llm_steps: list[dict],
        recovery_pos: np.ndarray,
        image_paths: list[str],
        experience_image_paths: list[str],
        experiences: list[tuple[object, float]],
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
        is_failure_entry = getattr(entry, "status", "") == "failure" or not bool(getattr(entry.result, "success", False))
        llm_critic_result: dict[str, Any] = {}
        rule_critic_result: dict[str, Any] = {}
        if self.memory_policy == "hierarchical" and is_failure_entry:
            try:
                llm_critic_result = critique_failure_experience(
                    method=self.method,
                    memory_policy=self.memory_policy,
                    metrics=self.metrics,
                    task_history=getattr(self, "_task_history", []),
                    recovery_steps=self.metrics.get("executed_recovery_steps")
                    or self.metrics.get("llm_recovery_steps")
                    or [],
                    retrieved_memories=self.metrics.get("retrieved_memories") or [],
                )
                self.metrics["failure_experience_critic"] = llm_critic_result
                if llm_critic_result.get("enabled") and not llm_critic_result.get("error"):
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
                    if llm_critic_result.get("failure_type") and not entry.result.failure_reason:
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
            lib.upsert(entry)
            try:
                consolidation_report = lib.consolidate()
                if consolidation_report.get("action") == "consolidate":
                    self.metrics["stm_consolidation"] = consolidation_report
            except Exception as exc:
                print(f"  [WARN] STM/LTM consolidation 失败: {exc}")
            if is_failure_entry:
                try:
                    existing = lib._entries if hasattr(lib, '_entries') else []
                    _get_failure_clusterer().assign_new(entry, existing)
                except Exception as exc:
                    print(f"  [WARN] 失败聚类分配失败: {exc}")
            self.metrics["experience_saved"] = True
            self.metrics["experience_save_skipped_reason"] = ""
        self.experience_write_path.parent.mkdir(parents=True, exist_ok=True)
        lib.save(self.experience_write_path)
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
    parser.add_argument("--condition", choices=["direct", "sim_wrapper"], default=None)
    parser.add_argument("--experience-read", type=str, default=None)
    parser.add_argument("--experience-write", type=str, default=None)
    parser.add_argument("--save", type=str, default="results/method_result.json")
    parser.add_argument("--save-plan", type=str, default="results/method_plan.json")
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
    print(f"  inject_failed_plan_for_test: {args.inject_failed_plan_for_test}")
    print(f"  memory_index_dir: {args.memory_index_dir}")
    print(f"  strategy_family: {args.strategy_family}")
    print(f"  experience_save_mode: {args.experience_save_mode}")
    print(f"  experience_read: {read_arg}")
    print(f"  experience_write: {write_path}")
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
        memory_index_dir=args.memory_index_dir,
        strategy_family=args.strategy_family,
        experience_save_mode=args.experience_save_mode,
        save_path=str(save_path) if save_path is not None else None,
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
        exp.save_experience()
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
