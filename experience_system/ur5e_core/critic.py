"""UR5e-specific failure critics for the experience system."""

from __future__ import annotations

import json
import os
from typing import Any

from experience_core.llm_provider import JSON_ONLY_LINE, invoke_llm, parse_json_payload

from .field_atomic_plan import ur5e_allowed_parameter_keys
from .llm_runtime import resolve_ur5e_critic_model, resolve_ur5e_provider
from . import runtime_skills


_UNSUPPORTED_CRITIC_TERMS = (
    "多点接触",
    "接触保持验证",
    "验证接触保持",
    "连续力控",
    "力控闭环",
    "稳定性控制器",
    "视觉伺服闭环",
)

_ALLOWED_STAGES = {
    "detection",
    "recovery_plan",
    "recovery_execution",
    "virtual_validation",
    "task_completion",
    "memory_reuse",
    "unknown",
}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _allowed_critic_skills(scenario_id: str | None = "") -> set[str]:
    return runtime_skills.allowed_actions(scenario_id)

_COLLISION_RULES = {
    "unsafe_collision",
    "obstacle_collision",
}
_JOINT_RULES = {
    "joint_limit",
    "unsafe_joint",
}
_GRIPPER_CONTACT_RULES = {
    "no_contact_detected",
    "contact_lost_during_lift",
    "contact_gained_during_lift_unexpected",
    "pinch_too_wide",
    "grasp_not_secured",
    "apple_not_tracked",
}
_END_EFFECTOR_RULES = {
    "object_not_lifted",
    "insufficient_z_change",
    "perception_pos_inconsistency",
    "perception_offset_exceeds_grasp_range",
}


def _compact_step_trace_for_critic(step_trace: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    if not isinstance(step_trace, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in step_trace[:limit]:
        if not isinstance(item, dict):
            continue
        before = item.get("before") if isinstance(item.get("before"), dict) else {}
        after = item.get("after") if isinstance(item.get("after"), dict) else {}
        compact.append({
            "action": item.get("action", ""),
            "status": item.get("status", ""),
            "params": item.get("params") if isinstance(item.get("params"), dict) else {},
            "before": {
                "apple_z": before.get("apple_z", item.get("apple_z")),
                "pinch_distance": before.get("pinch_distance", item.get("pinch_distance")),
                "contact": before.get("contact"),
            },
            "after": {
                "apple_z": after.get("apple_z"),
                "pinch_distance": after.get("pinch_distance"),
                "contact": after.get("contact"),
            },
            "contact": item.get("contact"),
            "keyframe": item.get("keyframe"),
            "error": item.get("error", ""),
        })
    return compact


def _compact_sandbox_physical_failures(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    selected_id = metrics.get("selected_candidate_id")
    compact: list[dict[str, Any]] = []
    for item in metrics.get("candidate_sandbox_results") or []:
        if not isinstance(item, dict) or item.get("sandbox_success"):
            continue
        compact.append({
            "selected_for_execution": selected_id is not None and item.get("candidate_id") == selected_id,
            "score": item.get("score"),
            "z_before": item.get("z_before"),
            "z_after": item.get("z_after"),
            "z_change": item.get("z_change"),
            "pinch_distance": item.get("pinch_distance"),
            "reason": item.get("reason", ""),
            "target_source": item.get("target_source", ""),
            "step_trace": _compact_step_trace_for_critic(item.get("step_trace")),
        })
    return compact


def critique_ur5e_failure_experience(
    *,
    method: str,
    memory_policy: str,
    metrics: dict[str, Any],
    task_history: list[dict[str, Any]],
    recovery_steps: list[dict[str, Any]],
    retrieved_memories: list[dict[str, Any]] | None = None,
    provider: str = "",
    model: str = "",
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Generate UR5e failure-analysis memory fields.

    This is analysis for memory writeback, not an executable recovery plan.
    """

    if enabled is None:
        enabled = _env_bool("EXPERIMENT_FAILURE_CRITIC_ENABLED", default=True)
    if not enabled:
        return {"enabled": False, "skipped_reason": "disabled_by_env"}
    provider = provider or resolve_ur5e_provider()
    model = model or resolve_ur5e_critic_model()

    task_criteria = metrics.get("task_success_criteria") or {}
    scenario_id = str(metrics.get("scenario_id") or "")
    condition_id = str(metrics.get("condition_id") or "")
    compact_metrics = {
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "task_success": bool(metrics.get("task_success", False)),
        "task_success_criteria": task_criteria,
        "apple_z_after_recovery": metrics.get("apple_z_after_recovery"),
        "observed_pos": metrics.get("observed_pos"),
        "contact_after_close": metrics.get("contact_after_close", {}),
        "contact_after_lift": metrics.get("contact_after_lift", {}),
        "virtual_validation_success": metrics.get("virtual_validation_success"),
        "virtual_execution_result": metrics.get("virtual_execution_result"),
        "executed_plan_source": metrics.get("executed_plan_source", ""),
        "invalid_skill_steps": metrics.get("invalid_skill_steps", []),
        "candidate_plan_failure_type": metrics.get("candidate_plan_failure_type", ""),
        "candidate_selection_block_reason": metrics.get("candidate_selection_block_reason", ""),
        "candidate_sandbox_final_success": metrics.get("candidate_sandbox_final_success"),
        "candidate_sandbox_final_result": metrics.get("candidate_sandbox_final_result", {}),
        "sandbox_physical_failures": _compact_sandbox_physical_failures(metrics),
        "candidate_rejections": [
            {
                "candidate_id": item.get("candidate_id"),
                "reject_reason": item.get("reject_reason"),
                "actions": [
                    str(step.get("action", ""))
                    for step in (item.get("steps") if isinstance(item.get("steps"), list) else [])
                    if isinstance(step, dict) and step.get("action")
                ],
                "plan_signature": item.get("plan_signature", ""),
            }
            for item in (metrics.get("candidate_rejections") or [])[:5]
            if isinstance(item, dict)
        ],
        "failure_reason": metrics.get("failure_reason", ""),
        "condition_injection": metrics.get("condition_injection", {}),
    }
    history = [
        {
            "action": item.get("action", ""),
            "status": item.get("status", ""),
            "reason": item.get("reason", ""),
        }
        for item in (task_history or [])[-12:]
        if isinstance(item, dict)
    ]
    compact_memories = [
        {
            "partition": item.get("partition", ""),
            "status": item.get("status", ""),
            "failure_type": item.get("failure_type", ""),
            "plan_signature": item.get("plan_signature", ""),
        }
        for item in (retrieved_memories or [])[:5]
        if isinstance(item, dict)
    ]

    prompt = f"""
你是 UR5e MuJoCo 机械臂异常恢复实验的失败经验 critic。请基于结构化日志分析任务失败原因，并生成可以写入经验库的失败经验字段。

场景：
- method: {method}
- memory_policy: {memory_policy}
- condition_id: {condition_id}
- scenario_id: {scenario_id}

任务说明：
总体目标：异常恢复后使目标物体完成抓取、提升、搬运和放置闭环，并让机械臂回到安全状态，最后需要把苹果放到盘子上面。

可用技能边界：
{_format_ur5e_critic_skill_scope(scenario_id)}

输入日志：
metrics={json.dumps(_json_safe(compact_metrics), ensure_ascii=False)}
task_history={json.dumps(_json_safe(history), ensure_ascii=False)}
recovery_steps={json.dumps(_json_safe(recovery_steps or []), ensure_ascii=False)}
retrieved_memories={json.dumps(_json_safe(compact_memories), ensure_ascii=False)}

要求：
1. 只分析失败原因、缺失流程和参数问题，不要生成可执行技能序列。
2. 不要写死旧异常类别，输出必须基于 condition_id、执行日志和恢复结果。
3. 如果恢复抓取成功但最终任务失败，要分析 task_success_criteria 中哪个谓词失败。
4. 如果有 virtual_execution_result.step_trace，必须利用每一步前后的 apple_z、pinch_distance、接触状态和执行状态。
5. 如果有 condition_injection，必须基于 true_pos/perceived_pos/offset 等字段判断感知或抓取偏差，不要预设结论。
6. 不要建议当前技能无法实现的能力，例如多点接触、接触保持验证、抓取后接触验证、提升前状态确认、连续力控、稳定性控制器、视觉伺服闭环。
7. corrective_direction 必须只描述如何补齐或调整现有技能阶段，不能创造“验证流程/状态确认流程/接触确认流程”等不存在的阶段。
8. missing_phases需要仔细思考需要完成最终的目标还需要那些环节，把所有缺失的环节都补上。
9. parameter_failure_summary 必须根据完整 recovery_steps 和 step_trace 判断，指出真正需要调整参数的技能和参数，不要机械地填写最后失败动作。
10. 如果 sandbox_physical_failures 非空，必须把它当作本次物理执行失败证据，结合 selected_for_execution、z_change、pinch_distance 和 step_trace 判断失败原因。
11. 如果 candidate_rejections 非空，必须结合 reject_reason、候选动作链和执行日志判断失败原因，不能只复述 reject_reason。
12. 对候选计划结构失败，missing_phases 必须列出缺失的现有技能阶段，corrective_direction 必须说明如何围绕现有技能阶段修正，不能写抽象流程名。
13. 如果无法判断具体参数错误，items 中写 action=unknown，bad_keys 为空，并说明证据不足。
14. 必须以完成总体目标为前提来思考失败经验，绝对不能仅仅只是完成异常处理。
只输出 JSON 对象，字段如下：
{{
  "failure_stage": "detection|recovery_plan|recovery_execution|virtual_validation|task_completion|memory_reuse|unknown",
  "failure_type": "中文失败类型名称",
  "root_cause": "中文短句，解释失败根因",
  "corrective_direction": "方向性修正建议，只能围绕现有技能阶段，不要创造新能力",
  "missing_phases": ["registered_action_name"],
  "failed_predicates": ["未满足的谓词或条件"],
  "failure_evidence": {{"关键证据名": "关键证据值"}},
  "parameter_failure_summary": {{
    "items": [
      {{
        "action": "需要调整参数的技能名或 unknown",
        "bad_keys": ["需要调整的参数名"],
        "bad_values": {{"参数名": "本次失败中的值"}},
        "expected_direction": {{"参数名": "increase|decrease|change|keep|unknown"}},
        "reason": "为什么这些参数导致或没有解决失败",
        "impact": "这些参数对失败证据的影响",
        "parameter_lesson": "一句参数层面的经验"
      }}
    ],
    "overall_lesson": "参数层面的总体经验"
  }},
  "task_goal": "当前任务最终目标一句话总结"
}}
{JSON_ONLY_LINE}
"""
    if _env_bool("PRINT_CRITIC_PROMPT", default=False):
        print("\n" + "=" * 24 + " UR5E CRITIC PROMPT BEGIN " + "=" * 24)
        print(prompt.strip())
        print("=" * 25 + " UR5E CRITIC PROMPT END " + "=" * 25 + "\n")

    raw_text = ""
    try:
        raw_text = invoke_llm(
            prompt,
            provider=provider,
            model=model,
            system_prompt="你是 UR5e MuJoCo 异常恢复实验的失败经验 critic。必须只返回 JSON。",
            temperature=0.2,
        )
        payload = parse_json_payload(raw_text, prefer_array=False)
    except Exception as exc:
        return {
            "enabled": True,
            "model": model,
            "error": "critic_json_parse_failed",
            "parse_error_type": type(exc).__name__,
            "raw_text": raw_text,
        }

    if not isinstance(payload, dict):
        return {
            "enabled": True,
            "model": model,
            "error": "critic_response_not_json_object",
            "raw_text": raw_text,
        }
    return sanitize_ur5e_llm_critic(payload, model=model)


def sanitize_ur5e_llm_critic(payload: dict[str, Any], *, model: str = "") -> dict[str, Any]:
    stage = _clean_text(payload.get("failure_stage") or "unknown")
    if stage not in _ALLOWED_STAGES:
        stage = "unknown"
    predicates = payload.get("failed_predicates") if isinstance(payload.get("failed_predicates"), list) else []
    evidence = _clean_json_text_values(payload.get("failure_evidence")) if isinstance(payload.get("failure_evidence"), dict) else {}
    root_cause = _clean_text(payload.get("root_cause") or "")
    corrective_direction = _clean_text(payload.get("corrective_direction") or "")
    failure_type = _clean_text(payload.get("failure_type") or "")
    unsupported_detected = any(
        _contains_unsupported_critic_term(text)
        for text in (root_cause, corrective_direction, failure_type)
    )
    result = {
        "enabled": True,
        "model": model,
        "failure_stage": stage,
        "failure_type": failure_type[:80],
        "root_cause": root_cause[:300],
        "corrective_direction": _sanitize_corrective_direction(corrective_direction)[:200],
        "missing_phases": _sanitize_missing_phases(payload.get("missing_phases")),
        "failed_predicates": [_clean_text(item)[:80] for item in predicates[:8] if _clean_text(item)],
        "failure_evidence": evidence,
        "task_goal": _clean_text(payload.get("task_goal") or "")[:200],
        "parameter_failure_summary": _sanitize_parameter_failure_summary(
            payload.get("parameter_failure_summary")
            if isinstance(payload.get("parameter_failure_summary"), dict)
            else {}
        ),
    }
    if unsupported_detected:
        result["unsupported_terms_removed"] = list(_UNSUPPORTED_CRITIC_TERMS)
        result["critic_warning"] = "critic output contained unsupported capability terms; no recovery instruction was generated by code"
    return result


def build_critic_result(
    *,
    rule_result: dict[str, Any] | None = None,
    llm_result: dict[str, Any] | None = None,
    is_failure: bool = False,
) -> dict[str, Any]:
    """Merge deterministic and LLM critics into CriticResultInfo-compatible dict."""

    rule_result = rule_result if isinstance(rule_result, dict) else {}
    llm_result = llm_result if isinstance(llm_result, dict) else {}
    rule_flags = [
        flag
        for flag in (rule_result.get("rule_flags") or [])
        if isinstance(flag, dict)
    ]
    names = _rule_names(rule_flags)
    llm_enabled = bool(llm_result.get("enabled")) and not bool(llm_result.get("error"))
    llm_failure_type = str(llm_result.get("failure_type") or "")
    llm_root_cause = str(llm_result.get("root_cause") or "")
    llm_stage = str(llm_result.get("failure_stage") or "")

    if rule_flags:
        overall_status = "warn"
    elif is_failure:
        overall_status = "unknown"
    else:
        overall_status = "pass"

    if names & {"plan_blocked_invalid", "invalid_skill_steps_in_plan", "plan_blocked_by_failed_history"}:
        overall_status = "block"
    if llm_result.get("critic_warning"):
        overall_status = "warn"

    risk_score = 0.0
    if is_failure:
        risk_score += 0.25
    risk_score += min(len(rule_flags) * 0.12, 0.55)
    if llm_enabled and (llm_failure_type or llm_root_cause):
        risk_score += 0.15
    if overall_status == "block":
        risk_score = max(risk_score, 0.8)
    elif overall_status == "warn":
        risk_score = max(risk_score, 0.45)
    risk_score = min(risk_score, 1.0)

    feedback_parts: list[str] = []
    if llm_failure_type:
        feedback_parts.append(f"failure_type={llm_failure_type}")
    if llm_stage:
        feedback_parts.append(f"failure_stage={llm_stage}")
    if llm_root_cause:
        feedback_parts.append(f"root_cause={llm_root_cause}")
    if llm_result.get("corrective_direction"):
        feedback_parts.append(f"corrective_direction={llm_result.get('corrective_direction')}")
    if llm_result.get("missing_phases"):
        feedback_parts.append("missing_phases=" + ", ".join(str(x) for x in llm_result.get("missing_phases") or []))

    return {
        "overall_status": overall_status,
        "critic_risk_score": round(risk_score, 4),
        "collision": _flag_summary(rule_flags, _COLLISION_RULES),
        "joint": _flag_summary(rule_flags, _JOINT_RULES),
        "gripper_contact": _flag_summary(rule_flags, _GRIPPER_CONTACT_RULES),
        "end_effector_pose": _flag_summary(rule_flags, _END_EFFECTOR_RULES),
        "rule_flags": rule_flags,
        "feedback_for_rewrite": "；".join(feedback_parts)[:600],
    }


def _format_ur5e_critic_skill_scope(scenario_id: str | None = "") -> str:
    lines = []
    for name in sorted(_allowed_critic_skills(scenario_id)):
        description = runtime_skills.skill_description(name)
        signature = runtime_skills.skill_signature(name)
        params = sorted(ur5e_allowed_parameter_keys(name))
        lines.append(f"- {signature}: {description}; allowed_parameters={params}")
    return "\n".join(lines)


def _sanitize_parameter_failure_summary(summary: dict[str, Any]) -> dict[str, Any]:
    raw_items = summary.get("items") if isinstance(summary.get("items"), list) else []
    items = [
        item
        for item in (_sanitize_parameter_failure_item(raw) for raw in raw_items[:8] if isinstance(raw, dict))
        if item
    ]
    if not items:
        items = [{
            "action": "unknown",
            "bad_keys": [],
            "bad_values": {},
            "expected_direction": {},
            "reason": "不足以从日志判断具体参数错误。",
            "impact": "",
            "parameter_lesson": "",
        }]
    return {
        "items": items,
        "overall_lesson": _clean_text(summary.get("overall_lesson") or "")[:240],
    }


def _sanitize_missing_phases(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    phases: list[str] = []
    for item in value:
        text = _clean_text(item)
        if not text:
            continue
        for phase in _normalize_missing_phases_from_text(text):
            if phase and phase not in phases:
                phases.append(phase)
            if len(phases) >= 4:
                break
        if len(phases) >= 4:
            break
    return phases


def _normalize_missing_phases_from_text(text: str) -> list[str]:
    allowed_missing_phases = _allowed_critic_skills()
    if text in allowed_missing_phases:
        return [text]
    for phase in allowed_missing_phases:
        if phase in text:
            return [phase]
    return []


def _sanitize_corrective_direction(text: str) -> str:
    clean = _clean_text(text)
    if not clean:
        return ""
    replacements = {
        "恢复执行流程": "恢复计划需要补齐现有 UR5E 技能阶段",
    }
    for old, new in replacements.items():
        clean = clean.replace(old, new)
    return clean


def _sanitize_parameter_failure_item(item: dict[str, Any]) -> dict[str, Any]:
    action = _clean_text(item.get("action") or "unknown")
    if action not in _allowed_critic_skills():
        action = "unknown"
    raw_bad_keys = item.get("bad_keys") if isinstance(item.get("bad_keys"), list) else []
    allowed_keys = ur5e_allowed_parameter_keys(action) if action != "unknown" else set()
    bad_keys = [
        _clean_text(key)
        for key in raw_bad_keys[:8]
        if action == "unknown" or _clean_text(key) in allowed_keys
    ]
    bad_values = item.get("bad_values") if isinstance(item.get("bad_values"), dict) else {}
    if action != "unknown":
        bad_values = {_clean_text(key): _clean_json_text_values(value) for key, value in bad_values.items() if _clean_text(key) in allowed_keys}
    else:
        bad_values = _clean_json_text_values(bad_values)
    expected_direction = item.get("expected_direction") if isinstance(item.get("expected_direction"), dict) else {}
    if action != "unknown":
        expected_direction = {_clean_text(key): _clean_text(value) for key, value in expected_direction.items() if _clean_text(key) in allowed_keys}
    else:
        expected_direction = _clean_json_text_values(expected_direction)
    return {
        "action": action,
        "bad_keys": bad_keys,
        "bad_values": bad_values,
        "expected_direction": expected_direction,
        "reason": _clean_text(item.get("reason") or "")[:240],
        "impact": _clean_text(item.get("impact") or "")[:240],
        "parameter_lesson": _clean_text(item.get("parameter_lesson") or "")[:180],
    }


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("```", "").replace("`", "")
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32).strip()


def _clean_json_text_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {_clean_text(key): _clean_json_text_values(val) for key, val in value.items() if _clean_text(key)}
    if isinstance(value, list):
        return [_clean_json_text_values(item) for item in value]
    if isinstance(value, str):
        return _clean_text(value)
    return value


def _rule_names(rule_flags: list[dict[str, Any]]) -> set[str]:
    return {
        str(flag.get("rule") or "")
        for flag in rule_flags
        if isinstance(flag, dict) and flag.get("rule")
    }


def _flag_summary(rule_flags: list[dict[str, Any]], wanted: set[str]) -> dict[str, Any]:
    matched = [
        flag
        for flag in rule_flags
        if isinstance(flag, dict) and str(flag.get("rule") or "") in wanted
    ]
    return {
        "status": "warn" if matched else "pass",
        "flag_count": len(matched),
        "rules": [str(flag.get("rule") or "") for flag in matched],
        "evidence": [
            str(flag.get("evidence") or flag.get("description_cn") or "")[:200]
            for flag in matched[:3]
        ],
    }


def _contains_unsupported_critic_term(text: str) -> bool:
    return any(term in str(text or "") for term in _UNSUPPORTED_CRITIC_TERMS)


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
