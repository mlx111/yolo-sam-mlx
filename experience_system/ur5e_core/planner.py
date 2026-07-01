"""UR5e recovery planner backed by the shared experience-system LLM runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experience_core.llm_provider import parse_json_payload

from .field_atomic_plan import (
    render_ur5e_field_atomic_skill_specs,
)
from .llm_runtime import build_image_block, invoke_ur5e_multimodal, resolve_ur5e_recovery_model
from . import runtime_skills


def plan_recovery(
    task_history: list[dict[str, Any]],
    image_paths: list[Path | str],
    target: str = "apple",
    experiences: list[tuple[Any, float]] | None = None,
    condition: str = "direct",
    experience_image_paths: list[Path | str] | None = None,
    blocker_matches: list[dict[str, Any]] | None = None,
    strategy_family: str = "",
    prompt_profile: str = "strong",
    scenario_id: str = "",
    condition_id: str = "",
    failure_family: str = "",
    condition_name: str = "",
    task_stage: str = "",
    injection_stage: str = "",
    success_criteria: str = "",
    target_observation_status: str = "",
    gripper_status: str = "",
) -> list[dict[str, Any]]:
    """Generate a UR5e field-atomic recovery plan."""
    candidates = plan_recovery_candidates(
        task_history=task_history,
        image_paths=image_paths,
        target=target,
        experiences=experiences,
        condition=condition,
        experience_image_paths=experience_image_paths,
        blocker_matches=blocker_matches,
        strategy_family=strategy_family,
        prompt_profile=prompt_profile,
        scenario_id=scenario_id,
        condition_id=condition_id,
        failure_family=failure_family,
        condition_name=condition_name,
        task_stage=task_stage,
        injection_stage=injection_stage,
        success_criteria=success_criteria,
        target_observation_status=target_observation_status,
        gripper_status=gripper_status,
        candidate_count=1,
    )
    return candidates[0]["steps"] if candidates else []


def plan_recovery_candidates(
    task_history: list[dict[str, Any]],
    image_paths: list[Path | str],
    target: str = "apple",
    experiences: list[tuple[Any, float]] | None = None,
    condition: str = "direct",
    experience_image_paths: list[Path | str] | None = None,
    blocker_matches: list[dict[str, Any]] | None = None,
    strategy_family: str = "",
    prompt_profile: str = "strong",
    scenario_id: str = "",
    condition_id: str = "",
    failure_family: str = "",
    condition_name: str = "",
    task_stage: str = "",
    injection_stage: str = "",
    success_criteria: str = "",
    target_observation_status: str = "",
    gripper_status: str = "",
    candidate_count: int = 3,
) -> list[dict[str, Any]]:
    """Generate one or more UR5e field-atomic recovery candidate plans."""

    del strategy_family, prompt_profile, injection_stage
    candidate_count = max(1, int(candidate_count or 1))
    task_list = [f"{item.get('action')}:{item.get('status')}" for item in task_history or []]
    prompt = _build_recovery_prompt(
        task_list=task_list,
        target=target,
        experiences=experiences or [],
        condition=condition,
        blocker_matches=blocker_matches or [],
        scenario_id=scenario_id,
        condition_id=condition_id,
        failure_family=failure_family,
        condition_name=condition_name,
        task_stage=task_stage,
        success_criteria=success_criteria,
        target_observation_status=target_observation_status,
        gripper_status=gripper_status,
        candidate_count=candidate_count,
    )

    content_blocks: list[dict[str, Any]] = []
    for image_path in (image_paths or [])[:4]:
        content_blocks.append(build_image_block(image_path))
    if experience_image_paths:
        content_blocks.append(
            {
                "type": "text",
                "text": "以下图像来自检索到的历史经验关键帧，仅作为相似异常状态参考，不要把其中坐标当作当前坐标。",
            }
        )
        for image_path in experience_image_paths[:2]:
            content_blocks.append(build_image_block(image_path))
    content_blocks.append({"type": "text", "text": prompt})
    if candidate_count > 1:
        content_blocks.append({"type": "text", "text": "只输出包含 plans 的 JSON 对象，不要解释，不要代码块。"})
    else:
        content_blocks.append({"type": "text", "text": "只输出 JSON 数组，不要解释，不要代码块。"})

    if _env_bool("PRINT_RECOVERY_PROMPT"):
        print("\n" + "=" * 24 + " UR5E RECOVERY PROMPT BEGIN " + "=" * 24)
        print(prompt.strip())
        print("=" * 25 + " UR5E RECOVERY PROMPT END " + "=" * 25 + "\n")

    raw_text = invoke_ur5e_multimodal(
        content_blocks,
        model=resolve_ur5e_recovery_model(),
        system_prompt="你是 UR5e MuJoCo 异常恢复实验的恢复规划器。必须只输出 JSON。",
    )
    payload = parse_json_payload(raw_text, prefer_array=True)
    candidates, diagnostics = _normalize_recovery_candidates_payload(payload, target=target)
    if candidate_count > 1:
        candidates = candidates[:candidate_count]
    if _env_bool("PRINT_RECOVERY_RAW"):
        print("\n" + "=" * 24 + " UR5E 恢复计划原始返回 BEGIN " + "=" * 24)
        print(str(raw_text)[:4000])
        print("=" * 25 + " UR5E 恢复计划原始返回 END " + "=" * 25)
    if _env_bool("PRINT_RECOVERY_RAW") or not candidates:
        print("UR5E 恢复计划解析诊断:", json.dumps(_diagnostics_with_cn(diagnostics), ensure_ascii=False, indent=2)[:4000])
    return candidates


def _build_recovery_prompt(
    *,
    task_list: list[str],
    target: str,
    experiences: list[tuple[Any, float]],
    condition: str,
    blocker_matches: list[dict[str, Any]],
    scenario_id: str,
    condition_id: str,
    failure_family: str,
    condition_name: str,
    task_stage: str,
    success_criteria: str,
    target_observation_status: str,
    gripper_status: str,
    candidate_count: int,
) -> str:
    condition_context = _format_condition_prompt_context(
        scenario_id=scenario_id,
        condition_id=condition_id,
        condition_name=condition_name,
        task_stage=task_stage,
        success_criteria=success_criteria,
        failure_family=failure_family,
    )
    experience_context = _format_experience_context(experiences)
    blocker_context = _format_blocker_context(blocker_matches)
    mode_hint = (
        "direct 模式：候选恢复计划会在 MuJoCo 中执行验证，并用最终物理状态评估。"
        if condition != "sim_wrapper"
        else "sim_wrapper 对比模式：候选恢复计划会在 MuJoCo 中执行验证，并用最终物理状态评估。"
    )
    if candidate_count > 1:
        output_contract = f"""
输出格式：
必须输出一个 JSON 对象，格式为：
{{
  "plans": [
    {{"steps": [{{"action": "...", "parameters": {{...}}}}]}},
    {{"steps": [{{"action": "...", "parameters": {{...}}}}]}}
  ]
}}
plans 数量必须为 {candidate_count}。每个候选的动作结构或关键参数必须不同，禁止输出 {candidate_count} 个近似相同的候选。
"""
    else:
        output_contract = "输出格式：只能输出 JSON 数组；每项只能包含 action 和 parameters。"
    return f"""
你是 UR5e MuJoCo 异常恢复实验的恢复规划器。当前任务是在异常发生后继续完成闭环：正确目标 {target} 被重新抓取、提升、必要时搬运/放置，并让机械臂回到安全状态。

已执行技能状态：
{json.dumps(task_list, ensure_ascii=False, indent=2)}

场景/条件信息：
{condition_context}

当前目标观测状态：
{target_observation_status or "未提供"}

当前夹爪状态：
{gripper_status or "未提供"}

模式：
{mode_hint}

可用技能名：
{json.dumps(sorted(runtime_skills.allowed_actions(scenario_id)), ensure_ascii=False, indent=2)}

技能参数说明：
{render_ur5e_field_atomic_skill_specs()}

历史经验：
{experience_context}

失败模式阻断：
{blocker_context or "无"}

硬性规则：
{output_contract}
1. 每个 step 只能包含 action 和 parameters。
2. action 必须逐字匹配可用技能名。
3. parameters 只能包含该技能允许的参数；禁止输出 stage、duration、work_dir、settle_steps、q、release 等底层参数。
4. 如果历史经验里有失败案例，要利用 critic 根因、corrective_direction 和 missing_phases 修正方案，要仔细思考missing_phases中的缺失的部分技能为什么会缺失，他的作用是什么，能够在技能序列的哪里使用，如何使用，必须把这部分体现在输出的技能序列中。
5. 不要 Markdown，不要解释，只输出 JSON。
6. 不能只进行异常处理 需要彻底完成最开始提到的完整任务 需要输出能够完成整个任务的技能序列，而不是仅仅处理异常的任务序列
"""


def _normalize_recovery_candidates_payload(payload: Any, *, target: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_plans: list[Any]
    if isinstance(payload, dict) and isinstance(payload.get("plans"), list):
        raw_plans = payload["plans"]
    else:
        steps, diagnostics = _normalize_recovery_payload(payload, target=target)
        candidates = [{"candidate_id": 1, "steps": steps}] if steps else []
        diagnostics["candidate_count"] = len(candidates)
        diagnostics["candidate_mode"] = "legacy_single_plan"
        return candidates, diagnostics

    candidates: list[dict[str, Any]] = []
    invalid_candidates: list[dict[str, Any]] = []
    for index, raw_plan in enumerate(raw_plans, start=1):
        if isinstance(raw_plan, dict):
            raw_steps = raw_plan.get("steps") or raw_plan.get("actions") or []
        elif isinstance(raw_plan, list):
            raw_steps = raw_plan
        else:
            invalid_candidates.append({"candidate_id": index, "reason": "candidate_not_object_or_list"})
            continue
        steps, step_diagnostics = _normalize_recovery_payload(raw_steps, target=target)
        if not steps:
            invalid_candidates.append({
                "candidate_id": index,
                "reason": "candidate_has_no_valid_steps",
                "diagnostics": step_diagnostics,
            })
            continue
        candidates.append({"candidate_id": index, "steps": steps})
    return candidates, {
        "payload_type": type(payload).__name__,
        "candidate_mode": "plans",
        "raw_candidate_count": len(raw_plans),
        "candidate_count": len(candidates),
        "invalid_candidates": invalid_candidates[:8],
    }


def _normalize_recovery_payload(payload: Any, *, target: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_steps: list[Any]
    if isinstance(payload, dict):
        value = payload.get("steps") or payload.get("actions") or []
        raw_steps = value if isinstance(value, list) else []
    elif isinstance(payload, list):
        raw_steps = payload
    else:
        return [], {
            "payload_type": type(payload).__name__,
            "raw_step_count": 0,
            "invalid_steps": [
                {
                    "index": None,
                    "reason": "payload_not_list_or_object",
                    "reason_cn": _reason_cn("payload_not_list_or_object"),
                    "raw_step": payload,
                }
            ],
        }

    steps: list[dict[str, Any]] = []
    invalid_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        normalized, reason = _normalize_recovery_step(raw_step, target=target)
        if normalized is None:
            invalid_steps.append({
                "index": index,
                "reason": reason,
                "reason_cn": _reason_cn(reason),
                "raw_step": raw_step,
            })
            continue
        steps.append(normalized)
    fixed_steps = steps
    diagnostics = {
        "payload_type": type(payload).__name__,
        "raw_step_count": len(raw_steps),
        "valid_step_count": len(steps),
        "final_step_count": len(fixed_steps),
        "invalid_steps": invalid_steps[:8],
    }
    return fixed_steps, diagnostics


def _diagnostics_with_cn(diagnostics: dict[str, Any]) -> dict[str, Any]:
    raw_step_count = int(diagnostics.get("raw_step_count") or 0)
    final_step_count = int(diagnostics.get("final_step_count") or 0)
    if raw_step_count == 0:
        summary_cn = "模型返回的 JSON 中没有任何技能步骤。"
    elif final_step_count == 0:
        summary_cn = "模型返回了内容，但所有步骤都被参数/技能名校验过滤掉了。"
    elif diagnostics.get("invalid_steps"):
        summary_cn = "模型返回了部分可执行步骤，另有部分步骤被过滤。"
    else:
        summary_cn = "模型返回的恢复计划解析正常。"
    result = dict(diagnostics)
    result["summary_cn"] = summary_cn
    return result


def _reason_cn(reason: str) -> str:
    reason = str(reason or "")
    if reason.startswith("action_not_allowed:"):
        action = reason.split(":", 1)[1]
        return f"技能名 {action} 不在当前 UR5E 允许技能列表中。"
    mapping = {
        "payload_not_list_or_object": "LLM 返回的 JSON 不是数组，也不是包含 steps/actions 的对象。",
        "step_not_object": "某个步骤不是 JSON 对象。",
        "parameters_not_object": "parameters 字段不是 JSON 对象。",
    }
    if "_missing_required_parameters" in reason:
        return "技能缺少运行时元数据声明的必需参数。"
    if "_invalid_" in reason:
        return "技能参数不符合运行时元数据声明。"
    return mapping.get(reason, f"未知过滤原因：{reason}")


def _normalize_recovery_step(raw_step: Any, *, target: str) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw_step, dict):
        return None, "step_not_object"
    action = str(raw_step.get("action", "")).strip()
    if action not in runtime_skills.allowed_actions():
        return None, f"action_not_allowed:{action}"
    raw_params = raw_step.get("parameters", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        return None, "parameters_not_object"
    params, reason = runtime_skills.normalize_parameters(action, raw_params)
    if params is None:
        return None, reason
    return {"action": action, "parameters": params}, ""


def _normalize_target_class(value: Any, *, fallback: str = "apple") -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    text = str(value or fallback).strip().lower()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list) and parsed:
                text = str(parsed[0]).strip().lower()
        except Exception:
            text = text.strip("[]'\" ")
    return text or fallback


def _format_experience_context(experiences: list[tuple[Any, float]]) -> str:
    if not experiences:
        return "无检索经验。"
    success_blocks: list[str] = []
    failure_blocks: list[str] = []
    for entry, score in experiences:
        converted_steps = _experience_steps_to_llm_actions(entry)
        result = getattr(entry, "result", None)
        is_success = bool(getattr(result, "success", False))
        taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
        block = [
            f"相似度={float(score):.2f}",
            f"来源={getattr(entry, 'source', '')}",
            f"条件={getattr(entry, 'condition_id', '')}",
        ]
        if converted_steps:
            block.append(f"技能序列={json.dumps(converted_steps, ensure_ascii=False)}")
        if is_success:
            block.append(f"成功证据={_format_success_memory_evidence(entry)}")
            if getattr(entry, "summary", ""):
                block.append(f"摘要={getattr(entry, 'summary', '')}")
            success_blocks.append("；".join([item for item in block if item]))
        else:
            llm_critic = taxonomy.get("llm_critic") if isinstance(taxonomy, dict) else {}
            candidate_failure = taxonomy.get("candidate_failure") if isinstance(taxonomy, dict) else {}
            candidate_rejections = (
                candidate_failure.get("rejections")
                if isinstance(candidate_failure, dict)
                else []
            ) or []
            if candidate_rejections:
                compact_rejections = []
                for item in candidate_rejections[:3]:
                    if not isinstance(item, dict):
                        continue
                    steps = item.get("steps") if isinstance(item.get("steps"), list) else []
                    compact_rejections.append(
                        {
                            "candidate_id": item.get("candidate_id"),
                            "reject_reason": item.get("reject_reason"),
                            "actions": [
                                str(step.get("action", ""))
                                for step in steps
                                if isinstance(step, dict) and step.get("action")
                            ],
                        }
                    )
                block.append(f"失败候选={json.dumps(compact_rejections, ensure_ascii=False)}")
                block.append("候选约束=旧结构拒绝仅作为历史参考；当前候选是否可行以 MuJoCo 物理执行结果为准。")
            if isinstance(taxonomy, dict):
                block.append(f"失败类型={taxonomy.get('failure_type', '')}")
                block.append(f"失败阶段={taxonomy.get('failure_stage', '')}")
                block.append(f"修正方向={taxonomy.get('corrective_direction', '')}")
                block.append(f"缺失阶段={json.dumps(taxonomy.get('missing_phases', []), ensure_ascii=False)}")
            parameter_lesson_lines: list[str] = []
            if isinstance(llm_critic, dict):
                block.append(f"critic根因={llm_critic.get('root_cause', '')}")
                block.append(f"critic修正方向={llm_critic.get('corrective_direction', '')}")
                parameter_summary = llm_critic.get("parameter_failure_summary")
                parameter_items = (
                    parameter_summary.get("items")
                    if isinstance(parameter_summary, dict) and isinstance(parameter_summary.get("items"), list)
                    else []
                )
                if parameter_items:
                    for index, item in enumerate(parameter_items[:4], start=1):
                        if not isinstance(item, dict):
                            continue
                        action = item.get("action")
                        bad_keys = item.get("bad_keys")
                        bad_values = item.get("bad_values")
                        expected_direction = item.get("expected_direction")
                        reason = item.get("reason")
                        impact = item.get("impact")
                        parameter_lesson = item.get("parameter_lesson")
                        parameter_lesson_lines.append(
                            "\n".join(
                                [
                                    f"  {index}. 错误技能: {action}",
                                    f"     错误参数: {json.dumps(bad_keys, ensure_ascii=False)}",
                                    f"     错误取值: {json.dumps(bad_values, ensure_ascii=False)}",
                                    f"     期望调整方向: {json.dumps(expected_direction, ensure_ascii=False)}",
                                    f"     失败原因: {reason}",
                                    f"     失败影响: {impact}",
                                    f"     参数教训: {parameter_lesson}",
                                ]
                            )
                        )
            failure_text = "；".join([item for item in block if item])
            if parameter_lesson_lines:
                failure_text += "\n  历史失败参数教训:\n" + "\n".join(parameter_lesson_lines)
            failure_blocks.append(failure_text)
    lines = [f"共检索到 {len(experiences)} 条经验。"]
    if success_blocks:
        lines.append("可参考成功经验：")
        lines.extend(f"- {item}" for item in success_blocks[:5])
    if failure_blocks:
        lines.append("必须避免的失败模式：")
        lines.extend(f"- {item}" for item in failure_blocks[:5])
    return "\n".join(lines)


def _experience_steps_to_llm_actions(entry: Any) -> list[dict[str, Any]]:
    taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
    candidate_failure = taxonomy.get("candidate_failure") if isinstance(taxonomy, dict) else {}
    candidate_rejections = (
        candidate_failure.get("rejections")
        if isinstance(candidate_failure, dict)
        else []
    ) or []
    for item in candidate_rejections:
        if not isinstance(item, dict):
            continue
        steps = item.get("steps") if isinstance(item.get("steps"), list) else []
        converted = [
            {
                "action": str(step.get("action", "")),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
            }
            for step in steps
            if isinstance(step, dict) and step.get("action") in runtime_skills.allowed_actions()
        ]
        if converted:
            return converted

    skill_sequence = getattr(entry, "skill_sequence", None)
    if skill_sequence:
        return [
            {"action": str(step.get("action", "")), "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {}}
            for step in skill_sequence
            if isinstance(step, dict) and step.get("action") in runtime_skills.allowed_actions()
        ]

    converted: list[dict[str, Any]] = []
    recovery_plan = getattr(entry, "recovery_plan", None)
    plan_steps = recovery_plan.get("steps", []) if isinstance(recovery_plan, dict) else getattr(recovery_plan, "steps", []) or []
    for step in plan_steps:
        if isinstance(step, dict):
            step_type = str(step.get("type", "") or step.get("action", "")).strip()
            params = step.get("params") if isinstance(step.get("params"), dict) else step.get("parameters") or {}
        else:
            step_type = str(getattr(step, "type", "")).strip()
            params = getattr(step, "params", {}) or {}
        if step_type in runtime_skills.allowed_actions():
            normalized, _reason = _normalize_recovery_step({"action": step_type, "parameters": params}, target="apple")
            if normalized:
                converted.append(normalized)
    return converted


def _format_success_memory_evidence(entry: Any) -> str:
    feedback = _entry_dict(getattr(entry, "execution_feedback", {}) or {})
    validation = _entry_dict(getattr(entry, "validation_evidence", {}) or {})
    metadata = _entry_dict(getattr(entry, "metadata", {}) or {})
    criteria = feedback.get("task_success_criteria") or validation.get("task_success_criteria") or metadata.get("task_success_criteria") or {}
    parts: list[str] = []
    if isinstance(criteria, dict):
        for key in ("type", "success", "apple_z", "baseline_z", "lift_from_table", "pinch_distance", "z_change", "grasp_secured"):
            if key in criteria:
                parts.append(f"{key}={criteria.get(key)}")
    if feedback.get("apple_z_after_recovery") is not None:
        parts.append(f"apple_z_after_recovery={feedback.get('apple_z_after_recovery')}")
    return ", ".join(parts[:10])


def _entry_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {key: getattr(value, key) for key in getattr(value, "__dataclass_fields__", {})}
    return value if isinstance(value, dict) else {}


def _format_blocker_context(blocker_matches: list[dict[str, Any]]) -> str:
    if not blocker_matches:
        return ""
    lines = ["上一次方案与失败经验高度相似，必须重写，不能复现以下失败动作模式："]
    for index, match in enumerate(blocker_matches[:3], 1):
        lines.append(
            f"{index}. failed_experience_id={match.get('experience_id', '')}; "
            f"overlap={float(match.get('overlap') or 0.0):.3f}; "
            f"failure_type={match.get('failure_type', '')}; "
            f"failed_signature={match.get('failed_signature', '')}"
        )
    return "\n".join(lines)


def _format_condition_prompt_context(
    *,
    scenario_id: str,
    condition_id: str,
    condition_name: str,
    task_stage: str,
    success_criteria: str,
    failure_family: str,
) -> str:
    parts = []
    if scenario_id or condition_id:
        parts.append(f"当前异常场景={condition_id or scenario_id}")
    if condition_name:
        parts.append(f"异常名称={condition_name}")
    if task_stage:
        parts.append(f"发生阶段={task_stage}")
    if success_criteria:
        parts.append(f"恢复成功目标={success_criteria}")
    if failure_family:
        parts.append(f"失败族={failure_family}")
    return "\n".join(parts) if parts else "未指定。"


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}
