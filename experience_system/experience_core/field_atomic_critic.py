"""LLM critic for Galaxea field-atomic failure memories."""

from __future__ import annotations

import json
from typing import Any

from .llm_provider import JSON_ONLY_LINE, build_image_block, invoke_llm, invoke_multimodal_llm, parse_json_payload


_UNSUPPORTED_CRITIC_TERMS = (
    "多点接触",
    "接触保持验证",
    "验证接触保持",
    "连续力控",
    "力控闭环",
    "稳定性控制器",
    "视觉伺服闭环",
)


_DIRECTIVE_MEMORY_TERMS = (
    "应该",
    "应当",
    "应",
    "必须",
    "需要",
    "不要",
    "不可",
    "不能",
    "下一次",
    "下次",
    "先",
    "再",
    "然后",
    "最后",
    "->",
    "→",
)


_GALAXEA_ALLOWED_MISSING_PHASES = {
    "move_base_relative",
    "set_torso_posture",
    "head_camera_rgbd_save",
    "head_camera_grounded_sam2_pose",
    "plan_cartesian_trajectory",
    "move_to_pregrasp",
    "approach_object",
    "close_gripper",
    "lift",
    "transport_to_detected_target",
    "lower_held_object",
    "open_gripper",
}


_GALAXEA_MISSING_PHASE_ALIASES = {
    "可达性条件调整": ["move_base_relative"],
    "底盘调整": ["move_base_relative"],
    "底盘移动": ["move_base_relative"],
    "腰部调整": ["set_torso_posture"],
    "躯干调整": ["set_torso_posture"],
    "调整后重新感知目标": ["head_camera_rgbd_save", "head_camera_grounded_sam2_pose"],
    "重新感知目标位置": ["head_camera_rgbd_save", "head_camera_grounded_sam2_pose"],
    "重新定位目标": ["head_camera_rgbd_save", "head_camera_grounded_sam2_pose"],
    "目标位置感知": ["head_camera_rgbd_save", "head_camera_grounded_sam2_pose"],
    "重新规划预抓取与接近": ["plan_cartesian_trajectory", "move_to_pregrasp", "approach_object"],
    "重新规划预抓取": ["plan_cartesian_trajectory", "move_to_pregrasp"],
    "到达预抓取": ["move_to_pregrasp"],
    "预抓取": ["move_to_pregrasp"],
    "接近抓取点": ["approach_object"],
    "重新接近有效抓取点": ["approach_object"],
    "重新夹持": ["close_gripper"],
    "夹持": ["close_gripper"],
    "抓取": ["close_gripper"],
    "完成抓取闭环": ["approach_object", "close_gripper", "lift"],
    "完成任务闭环": ["close_gripper", "lift"],
    "提升验证": ["lift"],
    "提升目标": ["lift"],
    "放置": ["transport_to_detected_target", "lower_held_object", "open_gripper"],
}


def _contains_unsupported_critic_term(text: str) -> bool:
    return any(term in str(text or "") for term in _UNSUPPORTED_CRITIC_TERMS)


def looks_like_recovery_instruction(text: str) -> bool:
    return any(term in str(text or "") for term in _DIRECTIVE_MEMORY_TERMS)


def critique_field_atomic_failure(
    *,
    goal: str,
    scenario_id: str,
    episode_role: str,
    action_reports: list[dict[str, Any]],
    task_summary: dict[str, Any],
    failure_taxonomy: dict[str, Any],
    image_paths: list[str] | None = None,
    provider: str = "doubao",
    model: str = "",
) -> dict[str, Any]:
    failed_action = _first_failed(action_reports)
    compact_trace = [_compact_action(item) for item in action_reports[-12:]]
    fallback = _fallback_field_atomic_failure_critic(
        goal=goal,
        failed_action=failed_action,
        action_reports=action_reports,
        task_summary=task_summary,
        failure_taxonomy=failure_taxonomy,
    )
    prompt = f"""
你是 Galaxea 机器人 field atomic 异常恢复实验的失败经验归因器。
请基于结构化日志分析本次技能序列为什么失败，并生成可写入经验库的失败经验增强字段。

目标：
{goal}

场景：
- scenario_id: {scenario_id}
- episode_role: {episode_role}
- visual_evidence_count: {len(image_paths or [])}

失败摘要：
failure_taxonomy={json.dumps(failure_taxonomy, ensure_ascii=False)}
task_summary={json.dumps(_compact_task_summary(task_summary), ensure_ascii=False)}
failed_action={json.dumps(failed_action, ensure_ascii=False)}
action_trace={json.dumps(compact_trace, ensure_ascii=False)}

要求：
1. 只分析失败原因和缺失流程，不要生成具体可执行技能序列。
2. corrective_direction 只能是方向性修正建议，不要输出固定动作顺序。
3. missing_phases 只能写 Galaxea 可执行技能名，2 到 4 项；不要写“可达性条件调整/完成闭环”这类抽象阶段。
4. 如果日志中有 final_error、target_torso、pregrasp_torso、final_tcp_minus_pregrasp_torso、object_lift_world 等证据，必须基于这些证据归因。
5. 不要建议当前技能无法实现的能力，例如连续力控、视觉伺服闭环、多点接触控制。
6. 如果失败是预抓取/接近阶段的可达性问题，必须区分“换手臂仍不可达”和“需要改变机器人与目标空间关系后重新建立感知与规划”的缺失流程。
7. 如果 action_trace 中缺少完成目标所需的后续阶段，请在 missing_phases 中指出这些缺失阶段。
8. 如果机械臂pregrasp以及apporach的final_error值过大，说明需要进行底盘的移动或者腰部的移动，也可以是两者的结合,抓取相关的技能final)error阈值需要比较小比如0.01左右，这样才能正确的抓取
9. 如果识别目标位置出现错误 需要调整腰部位置和底盘移动
10. 如果提供了失败关键帧图像，请结合图像判断夹爪/TCP相对目标物体的位置、侧向偏差、过冲或姿态偏差；图像不清楚时以结构化日志为准。
11. missing_phases 只能从以下枚举中选择：
{json.dumps(sorted(_GALAXEA_ALLOWED_MISSING_PHASES), ensure_ascii=False)}
只输出 JSON 对象，字段如下：
{{
  "failure_stage": "perception|planning|pregrasp|approach|grasp|lift|transport|place|task_completion|unknown",
  "failure_type": "中文短语",
  "root_cause": "中文短句，解释根本原因",
  "corrective_direction": "方向性修正建议，不写固定技能序列",
  "missing_phases": ["move_base_relative", "head_camera_rgbd_save"],
  "failed_predicates": ["未满足的条件"],
  "memory_lesson": "一句可检索的经验教训",
  "failure_evidence": {{"final_error": 0.0, "object_lift_world": 0.0, "target_torso": [0,0,0]}},
  "parameter_failure_summary": {{
    "items": [
      {{
        "action": "真正需要调整参数的技能名，而不是机械地填写最后失败动作",
        "bad_keys": ["需要调整的参数名"],
        "bad_values": {{"参数名": "本次失败中的值"}},
        "expected_direction": {{"参数名": "increase|decrease|change|keep|unknown"}},
        "reason": "为什么这些参数导致或没有解决失败",
        "impact": "这些参数对失败证据的影响",
        "parameter_lesson": "一句参数层面的经验"
      }}
    ],
    "required_actions": ["必须补充或重点调整的技能名"],
    "must_relocalize_after": ["执行后必须重新感知的技能名"],
    "forbidden_parameters": {{
      "技能名": {{"参数约束名": "会重复失败的参数范围或上限"}}
    }},
    "suggested_parameter_region": {{
      "技能名": {{"参数名": ["建议下界", "建议上界或枚举值"]}}
    }},
    "overall_lesson": "参数层面的总体经验"
  }},
  "task_goal": "最终任务目标一句话总结"
}}

parameter_failure_summary 要求：
- 必须根据完整 action_trace 判断，不要只看 failed_action。
- 不一定只有一个错误动作；如果底盘、腰部、感知、预抓取参数都可能影响结果，必须在 items 中分别列出多个 action。
- 如果某个动作需要新增或加大调整，也应作为一个 item 写入. 如果你认为某个技能的参数修改可以完成计划就把这个技能加入到里面。
- 如果无法判断具体参数错误，items 中写一个 action=unknown 的 item，bad_keys 为空，reason 写“不足以从日志判断具体参数错误”。
{JSON_ONLY_LINE}
"""
    try:
        valid_images = [str(path) for path in (image_paths or [])[:3] if path]
        if valid_images:
            content = [build_image_block(path) for path in valid_images]
            content.append({"type": "text", "text": prompt})
            raw = invoke_multimodal_llm(
                content,
                provider=provider,
                model=model,
                system_prompt="你是异常处理领域的专家 请你仔细观察对应图像和结构化输入，生成 Galaxea field atomic 失败经验 critic。必须只返回 JSON。",
                temperature=0.2,
            )
        else:
            raw = invoke_llm(
                prompt,
                provider=provider,
                model=model,
                system_prompt="你是异常处理领域的专家 请你仔细观察对应的输入 生成 Galaxea field atomic 失败经验 critic。必须只返回 JSON。",
                temperature=0.2,
            )
        payload = parse_json_payload(raw, prefer_array=False)
    except Exception as exc:
        fallback["error"] = str(exc)
        return fallback
    if not isinstance(payload, dict):
        fallback["error"] = "critic_response_not_json_object"
        fallback["raw_text"] = str(payload)[:500]
        return fallback

    allowed_stages = {"perception", "planning", "pregrasp", "approach", "grasp", "lift", "transport", "place", "task_completion", "unknown"}
    stage = str(payload.get("failure_stage") or "unknown").strip()
    if stage not in allowed_stages:
        stage = "unknown"
    missing = payload.get("missing_phases") if isinstance(payload.get("missing_phases"), list) else []
    predicates = payload.get("failed_predicates") if isinstance(payload.get("failed_predicates"), list) else []
    result = {
        "enabled": True,
        "model": model,
        "failure_stage": stage,
        "failure_type": str(payload.get("failure_type") or "").strip()[:100],
        "root_cause": str(payload.get("root_cause") or "").strip()[:300],
        "corrective_direction": str(payload.get("corrective_direction") or "").strip()[:240],
        "missing_phases": _sanitize_missing_phases(missing),
        "failed_predicates": [str(item).strip()[:100] for item in predicates[:8] if str(item).strip()],
        "memory_lesson": str(payload.get("memory_lesson") or "").strip()[:240],
        "failure_evidence": payload.get("failure_evidence") if isinstance(payload.get("failure_evidence"), dict) else {},
        "task_goal": str(payload.get("task_goal") or "").strip()[:200],
    }
    parameter_summary = payload.get("parameter_failure_summary")
    if isinstance(parameter_summary, dict):
        result["parameter_failure_summary"] = _sanitize_llm_parameter_failure_summary(parameter_summary)
    unsupported_detected = any(
        _contains_unsupported_critic_term(result.get(key, ""))
        for key in ("root_cause", "corrective_direction", "failure_type")
    )
    if unsupported_detected:
        result["unsupported_terms_removed"] = list(_UNSUPPORTED_CRITIC_TERMS)
        result["critic_warning"] = "critic output contained unsupported capability terms; no recovery instruction was generated by code"
    directive_fields = [
        key
        for key in ("root_cause", "corrective_direction", "failure_type")
        if looks_like_recovery_instruction(str(result.get(key) or ""))
    ]
    if directive_fields:
        result["directive_memory_warning"] = {
            "fields": directive_fields,
            "message": "critic output contains directive terms; keep it as failure-analysis memory, not as an executable recovery plan",
        }
    if not result.get("memory_lesson"):
        result["memory_lesson"] = field_atomic_memory_lesson(result)
    if not isinstance(result.get("parameter_failure_summary"), dict) or not result.get("parameter_failure_summary"):
        result["parameter_failure_summary"] = _fallback_parameter_failure_summary(
            failed_action=failed_action,
            action_reports=action_reports,
            task_summary=task_summary,
            failure_evidence=result.get("failure_evidence") if isinstance(result.get("failure_evidence"), dict) else {},
        )
    return _merge_with_fallback_critic(result, fallback)


def field_atomic_memory_lesson(llm_critic: dict[str, Any]) -> str:
    if not isinstance(llm_critic, dict) or llm_critic.get("error"):
        return ""
    if str(llm_critic.get("memory_lesson") or "").strip():
        return str(llm_critic.get("memory_lesson") or "").strip()[:700]
    parts = []
    if llm_critic.get("failure_type"):
        parts.append(f"失败类型：{llm_critic.get('failure_type')}")
    if llm_critic.get("root_cause"):
        parts.append(f"根因：{llm_critic.get('root_cause')}")
    if llm_critic.get("corrective_direction"):
        parts.append(f"修正方向：{llm_critic.get('corrective_direction')}")
    missing = llm_critic.get("missing_phases")
    if isinstance(missing, list) and missing:
        parts.append("缺失阶段：" + "，".join(str(item) for item in missing[:4]))
    evidence = llm_critic.get("failure_evidence") if isinstance(llm_critic.get("failure_evidence"), dict) else {}
    if evidence:
        evidence_bits = []
        for key in ("final_error", "object_lift_world", "target_torso"):
            value = evidence.get(key)
            if value not in (None, "", [], {}):
                evidence_bits.append(f"{key}={value}")
        if evidence_bits:
            parts.append("证据：" + "，".join(evidence_bits[:4]))
    return "；".join(parts)[:700]


def _sanitize_llm_parameter_failure_summary(summary: dict[str, Any]) -> dict[str, Any]:
    raw_items = summary.get("items") if isinstance(summary.get("items"), list) else []
    if not raw_items:
        raw_items = [summary]
    items = [
        item
        for item in (_sanitize_parameter_failure_item(raw) for raw in raw_items[:6] if isinstance(raw, dict))
        if item
    ]
    if not items:
        return {}
    result: dict[str, Any] = {
        "items": items,
        "overall_lesson": str(summary.get("overall_lesson") or summary.get("parameter_lesson") or "").strip()[:260],
    }
    required_actions = _sanitize_action_list(summary.get("required_actions"))
    must_relocalize_after = _sanitize_action_list(summary.get("must_relocalize_after"))
    forbidden = summary.get("forbidden_parameters") if isinstance(summary.get("forbidden_parameters"), dict) else {}
    suggested = summary.get("suggested_parameter_region") if isinstance(summary.get("suggested_parameter_region"), dict) else {}
    derived = _derive_actionable_parameter_guidance(items)
    if not required_actions:
        required_actions = derived.get("required_actions", [])
    if not must_relocalize_after:
        must_relocalize_after = derived.get("must_relocalize_after", [])
    if not forbidden:
        forbidden = derived.get("forbidden_parameters", {})
    if not suggested:
        suggested = derived.get("suggested_parameter_region", {})
    if required_actions:
        result["required_actions"] = required_actions
    if must_relocalize_after:
        result["must_relocalize_after"] = must_relocalize_after
    if forbidden:
        result["forbidden_parameters"] = _sanitize_nested_dict(forbidden)
    if suggested:
        result["suggested_parameter_region"] = _sanitize_nested_dict(suggested)
    if len(items) == 1:
        result.update(items[0])
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _sanitize_parameter_failure_item(summary: dict[str, Any]) -> dict[str, Any]:
    action = str(summary.get("action") or "").strip()
    allowed_actions = {
        "move_base_relative",
        "set_torso_posture",
        "head_camera_grounded_sam2_pose",
        "plan_cartesian_trajectory",
        "move_to_pregrasp",
        "approach_object",
        "close_gripper",
        "lift",
        "transport_to_detected_target",
        "lower_held_object",
        "open_gripper",
        "unknown",
    }
    if action not in allowed_actions:
        action = "unknown"
    bad_keys = summary.get("bad_keys") if isinstance(summary.get("bad_keys"), list) else []
    bad_values = summary.get("bad_values") if isinstance(summary.get("bad_values"), dict) else {}
    expected_direction = summary.get("expected_direction") if isinstance(summary.get("expected_direction"), dict) else {}
    result = {
        "action": action,
        "bad_keys": [str(item).strip()[:80] for item in bad_keys[:8] if str(item).strip()],
        "bad_values": {str(key)[:80]: value for key, value in bad_values.items()},
        "expected_direction": {str(key)[:80]: str(value)[:80] for key, value in expected_direction.items()},
        "severity": str(summary.get("severity") or "").strip()[:80],
        "reason": str(summary.get("reason") or "").strip()[:260],
        "impact": str(summary.get("impact") or "").strip()[:220],
        "parameter_lesson": str(summary.get("parameter_lesson") or "").strip()[:220],
    }
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _sanitize_action_list(value: Any) -> list[str]:
    allowed_actions = {
        "move_base_relative",
        "set_torso_posture",
        "head_camera_rgbd_save",
        "head_camera_grounded_sam2_pose",
        "plan_cartesian_trajectory",
        "move_to_pregrasp",
        "approach_object",
        "close_gripper",
        "lift",
        "transport_to_detected_target",
        "lower_held_object",
        "open_gripper",
    }
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        action = str(item or "").strip()
        if action in allowed_actions and action not in result:
            result.append(action)
    return result[:8]


def _sanitize_missing_phases(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    phases: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        normalized = _normalize_missing_phase_text(text)
        for phase in normalized:
            if phase and phase not in phases:
                phases.append(phase)
            if len(phases) >= 4:
                break
        if len(phases) >= 4:
            break
    return phases


def _normalize_missing_phase_text(text: str) -> list[str]:
    if text in _GALAXEA_ALLOWED_MISSING_PHASES:
        return [text]
    lowered = text.lower()
    if lowered in _GALAXEA_ALLOWED_MISSING_PHASES:
        return [lowered]
    if text in _GALAXEA_MISSING_PHASE_ALIASES:
        return _GALAXEA_MISSING_PHASE_ALIASES[text]
    result: list[str] = []
    for alias, phases in _GALAXEA_MISSING_PHASE_ALIASES.items():
        if alias in text:
            result.extend(phases)
    if result:
        return list(dict.fromkeys(result))
    if "rgb" in lowered or "图像" in text or "相机" in text or "感知" in text or "定位" in text:
        return ["head_camera_rgbd_save", "head_camera_grounded_sam2_pose"]
    if "规划" in text or "轨迹" in text:
        return ["plan_cartesian_trajectory"]
    if "预抓" in text:
        return ["move_to_pregrasp"]
    if "接近" in text:
        return ["approach_object"]
    if "夹" in text or "抓取" in text:
        return ["close_gripper"]
    if "提升" in text or "lift" in lowered:
        return ["lift"]
    if "底盘" in text or "base" in lowered:
        return ["move_base_relative"]
    if "腰" in text or "躯干" in text or "torso" in lowered:
        return ["set_torso_posture"]
    return []


def _sanitize_nested_dict(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key or "").strip()[:80]
        if not name:
            continue
        if isinstance(item, dict):
            cleaned = {str(k)[:80]: v for k, v in item.items() if v not in (None, "", [], {})}
            if cleaned:
                result[name] = cleaned
        elif item not in (None, "", [], {}):
            result[name] = item
    return result


def _derive_actionable_parameter_guidance(items: list[dict[str, Any]]) -> dict[str, Any]:
    required_actions: list[str] = []
    must_relocalize_after: list[str] = []
    forbidden: dict[str, dict[str, Any]] = {}
    suggested: dict[str, dict[str, Any]] = {}
    for item in items:
        action = str(item.get("action") or "")
        bad_values = item.get("bad_values") if isinstance(item.get("bad_values"), dict) else {}
        expected = item.get("expected_direction") if isinstance(item.get("expected_direction"), dict) else {}
        if action in {"move_base_relative", "set_torso_posture"} and action not in required_actions:
            required_actions.append(action)
        if action in {"move_base_relative", "set_torso_posture"} and action not in must_relocalize_after:
            must_relocalize_after.append(action)
        if action == "move_base_relative":
            limits: dict[str, Any] = {}
            region: dict[str, Any] = {}
            for axis in ("x", "y"):
                direction = str(expected.get(axis) or "")
                value = bad_values.get(axis)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = None
                if direction == "increase":
                    if numeric is not None:
                        limits[f"{axis}_max"] = round(numeric, 6)
                        low = min(max(numeric + 0.1, -0.1), 0.4)
                        region[axis] = [round(low, 6), 0.4]
                    else:
                        region[axis] = [0.3, 0.4]
            if limits:
                forbidden[action] = limits
            if region:
                suggested[action] = region
        elif action == "set_torso_posture":
            suggested.setdefault(action, {"level": "high"})
    if "move_base_relative" in required_actions and "set_torso_posture" not in required_actions:
        required_actions.append("set_torso_posture")
    if "move_base_relative" in must_relocalize_after:
        for action in ("head_camera_rgbd_save", "head_camera_grounded_sam2_pose"):
            if action not in required_actions:
                required_actions.append(action)
    return {
        "required_actions": required_actions,
        "must_relocalize_after": must_relocalize_after,
        "forbidden_parameters": forbidden,
        "suggested_parameter_region": suggested,
    }


def _fallback_parameter_failure_summary(
    *,
    failed_action: dict[str, Any],
    action_reports: list[dict[str, Any]],
    task_summary: dict[str, Any],
    failure_evidence: dict[str, Any],
) -> dict[str, Any]:
    raw = failed_action.get("raw_result") if isinstance(failed_action.get("raw_result"), dict) else {}
    evidence = failure_evidence if isinstance(failure_evidence, dict) else {}
    final_error = evidence.get("final_error") or raw.get("final_error")
    last_base = _last_action_parameters(action_reports, "move_base_relative")
    last_torso = _last_action_parameters(action_reports, "set_torso_posture")
    summary = {
        "items": [
            {
                "action": "unknown",
                "reason": "LLM critic did not provide parameter_failure_summary; do not infer bad parameters mechanically from the final failed action.",
                "impact": f"final_error={final_error}" if final_error is not None else "",
                "parameter_lesson": "无法仅凭最后失败动作判断具体参数错误；需要结合完整动作序列、底盘/腰部调整、重新感知和最终误差由 LLM 归因。",
            }
        ],
        "overall_lesson": "无法仅凭最后失败动作判断具体参数错误；需要结合完整动作序列、底盘/腰部调整、重新感知和最终误差由 LLM 归因。",
    }
    try:
        numeric_error = float(final_error)
    except (TypeError, ValueError):
        numeric_error = 0.0
    if numeric_error >= 0.1:
        items: list[dict[str, Any]] = []
        if last_base:
            bad_values = {
                key: last_base.get(key)
                for key in ("x", "y")
                if last_base.get(key) is not None
            }
            items.append({
                "action": "move_base_relative",
                "bad_keys": list(bad_values),
                "bad_values": bad_values,
                "expected_direction": {key: "increase" for key in bad_values},
                "severity": "blocking",
                "reason": "预抓取 final_error 仍然过大，当前底盘相对移动没有充分改善机器人与目标空间关系。",
                "impact": f"move_to_pregrasp final_error={final_error}，超过抓取精度阈值。",
                "parameter_lesson": "当底盘小幅移动后预抓取仍不可达，应增大底盘移动量并重新感知规划。",
            })
        else:
            items.append({
                "action": "move_base_relative",
                "bad_keys": ["x", "y"],
                "expected_direction": {"x": "increase", "y": "increase"},
                "severity": "blocking",
                "reason": "预抓取 final_error 过大且序列中没有有效底盘空间关系调整。",
                "impact": f"move_to_pregrasp final_error={final_error}，超过抓取精度阈值。",
                "parameter_lesson": "可达性失败时应加入底盘移动并重新感知规划。",
            })
        torso_level = str(last_torso.get("level") or "") if last_torso else ""
        if torso_level != "high":
            items.append({
                "action": "set_torso_posture",
                "bad_keys": ["level"] if torso_level else [],
                "bad_values": {"level": torso_level} if torso_level else {},
                "expected_direction": {"level": "change"},
                "severity": "blocking",
                "reason": "预抓取可达性失败需要同时考虑腰部高度对相机与机械臂工作空间的影响。",
                "impact": "只调整预抓取动作不能可靠改变机器人与目标的空间关系。",
                "parameter_lesson": "底盘移动仍不可达时，应尝试 set_torso_posture=high 并重新感知目标。",
            })
        summary["items"] = items
        summary["required_actions"] = ["move_base_relative", "set_torso_posture", "head_camera_rgbd_save", "head_camera_grounded_sam2_pose"]
        summary["must_relocalize_after"] = ["move_base_relative", "set_torso_posture"]
        summary["suggested_parameter_region"] = {
            "move_base_relative": {"x": [0.3, 0.4], "y": [0.2, 0.4]},
            "set_torso_posture": {"level": "high"},
        }
        if last_base:
            forbidden: dict[str, Any] = {}
            for key in ("x", "y"):
                if last_base.get(key) is not None:
                    forbidden[f"{key}_max"] = last_base.get(key)
            if forbidden:
                summary["forbidden_parameters"] = {"move_base_relative": forbidden}
    return {key: value for key, value in summary.items() if value not in (None, "", [], {})}


def _last_action_parameters(action_reports: list[dict[str, Any]], action: str) -> dict[str, Any]:
    for item in reversed(action_reports or []):
        if not isinstance(item, dict) or str(item.get("action") or "") != action:
            continue
        params = item.get("parameters")
        if isinstance(params, dict):
            return {str(key): value for key, value in params.items() if not str(key).startswith("_")}
    return {}


def score_field_atomic_recovery_quality(
    *,
    goal: str,
    task_history: list[dict[str, Any]],
    image_paths: list[str],
    provider: str = "doubao",
    model: str = "",
) -> dict[str, Any]:
    """Wrapper1-style multimodal score for a Galaxea recovery result."""
    task_list = [
        f"{item.get('action')}:{item.get('status')}"
        for item in task_history or []
        if isinstance(item, dict)
    ]
    summary_lines = []
    for index, item in enumerate(task_history or [], 1):
        if not isinstance(item, dict):
            continue
        summary_lines.append(
            f"{index}. action={item.get('action')} status={item.get('status')} "
            f"success={item.get('success')} message={item.get('message', '')}"
        )
    prompt = f"""
你是 Mujoco/Galaxea 仿真实验评估器，评估异常恢复后的最终效果。

目标：{goal}
技能流程及状态：{task_list}
动作历史摘要：
{chr(10).join(summary_lines)}

根据图像和技能流程判断异常处理是否成功，给出 0-10 分。
评分标准：
1. 目标任务完成且关键物体状态满足要求，才算 success。
2. 越早发现异常并恢复，分数越高。
3. 如果只是重新观察、重新定位，但没有完成目标闭环，不能判为 success。
4. 如果抓取任务中物体没有被提升或没有随夹爪移动，应判为 failure。

只输出 JSON 对象：
{{"status":"success/failure","score":0,"reason":"中文原因"}}
{JSON_ONLY_LINE}
"""
    content: list[dict[str, Any]] = []
    for image_path in (image_paths or [])[:3]:
        if image_path:
            content.append(build_image_block(str(image_path)))
    content.append({"type": "text", "text": prompt})
    try:
        raw = invoke_multimodal_llm(
            content,
            provider=provider,
            model=model,
            system_prompt="你是 Galaxea 异常恢复结果评估器。必须只返回 JSON。",
            temperature=0.0,
        )
        payload = parse_json_payload(raw, prefer_array=False)
    except Exception as exc:
        return {"enabled": True, "status": "failure", "score": 0, "reason": str(exc), "error": str(exc)}
    if not isinstance(payload, dict):
        return {"enabled": True, "status": "failure", "score": 0, "reason": "score response is not a JSON object"}
    try:
        score = float(payload.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "enabled": True,
        "status": str(payload.get("status") or "failure"),
        "score": max(0.0, min(score, 10.0)),
        "reason": str(payload.get("reason") or ""),
    }


def verify_field_atomic_anomaly(
    *,
    goal: str,
    image_before: str,
    image_after: str,
    rule_summary: dict[str, Any] | None = None,
    provider: str = "doubao",
    model: str = "",
) -> dict[str, Any]:
    """Wrapper1-style VLM anomaly verification for Galaxea."""
    rule_summary = rule_summary if isinstance(rule_summary, dict) else {}
    prompt = f"""
你是 Mujoco/Galaxea 异常实验核验器。请根据前后两张图和结构化规则结果，判断当前是否确实发生异常。

目标：{goal}
规则摘要：
{json.dumps(rule_summary, ensure_ascii=False)}

判断标准：
1. 如果目标物体没有达到任务要求，或者技能失败导致任务中断，判为 FAILURE。
2. 如果目标物体已经满足任务要求，判为 SUCCESS。
3. 如果图像不清楚，以结构化规则摘要为主要依据，并说明不确定性。

只输出 JSON 对象：
{{"status":"SUCCESS/FAILURE","reason":"中文原因","consider":"后续是否需要恢复处理"}}
{JSON_ONLY_LINE}
"""
    content = [
        build_image_block(str(image_before)),
        build_image_block(str(image_after)),
        {"type": "text", "text": prompt},
    ]
    try:
        raw = invoke_multimodal_llm(
            content,
            provider=provider,
            model=model,
            system_prompt="你是 Galaxea 异常核验器。必须只返回 JSON。",
            temperature=0.0,
        )
        payload = parse_json_payload(raw, prefer_array=False)
    except Exception as exc:
        return {"enabled": True, "status": "FAILURE", "reason": str(exc), "consider": "vlm_error", "error": str(exc)}
    if not isinstance(payload, dict):
        return {"enabled": True, "status": "FAILURE", "reason": "verify response is not a JSON object", "consider": "parse_error"}
    status = str(payload.get("status") or "FAILURE").upper()
    if status not in {"SUCCESS", "FAILURE"}:
        status = "FAILURE"
    return {
        "enabled": True,
        "status": status,
        "reason": str(payload.get("reason") or ""),
        "consider": str(payload.get("consider") or ""),
    }


def _merge_with_fallback_critic(result: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(result)
    for key in ("failure_stage", "failure_type", "root_cause", "corrective_direction", "task_goal"):
        if not str(merged.get(key) or "").strip() and fallback.get(key):
            merged[key] = fallback.get(key)
    for key in ("missing_phases", "failed_predicates"):
        current = merged.get(key) if isinstance(merged.get(key), list) else []
        backup = fallback.get(key) if isinstance(fallback.get(key), list) else []
        combined = [*(str(item) for item in current if str(item)), *(str(item) for item in backup if str(item))]
        merged[key] = _sanitize_missing_phases(combined) if key == "missing_phases" else list(dict.fromkeys(combined))[:8]
    if fallback.get("rule_fallback"):
        merged["rule_fallback"] = fallback.get("rule_fallback")
    return merged


def _fallback_field_atomic_failure_critic(
    *,
    goal: str,
    failed_action: dict[str, Any],
    action_reports: list[dict[str, Any]],
    task_summary: dict[str, Any],
    failure_taxonomy: dict[str, Any],
) -> dict[str, Any]:
    action = str(failed_action.get("action") or failure_taxonomy.get("failure_action") or "").strip()
    failure_type = str(failure_taxonomy.get("failure_type") or "").strip()
    evidence = failed_action.get("evidence") if isinstance(failed_action.get("evidence"), dict) else {}
    final_error = evidence.get("final_error") or failure_taxonomy.get("final_error")
    predicates = list((task_summary.get("task_success_criteria") or {}).get("failed_predicates") or []) if isinstance(task_summary.get("task_success_criteria"), dict) else []
    actions = [str(item.get("action") or "") for item in action_reports if isinstance(item, dict)]
    parameter_failure_summary = _fallback_parameter_failure_summary(
        failed_action=failed_action,
        action_reports=action_reports,
        task_summary=task_summary,
        failure_evidence=evidence,
    )

    if action in {"move_to_pregrasp", "approach_object"} or failure_type == "actuation_limit":
        stage = "pregrasp" if action == "move_to_pregrasp" else "approach"
        if action not in {"move_to_pregrasp", "approach_object"}:
            stage = "pregrasp"
        return {
            "enabled": True,
            "rule_fallback": True,
            "failure_stage": stage,
            "failure_type": "可达性受限",
            "root_cause": f"当前机器人与目标的空间关系下，{action or '机械臂接近'} 未能到达目标位姿，final_error={final_error}。",
            "corrective_direction": "需要先改变可达性条件，并在几何关系变化后重新建立目标位置和接近规划，避免只重复原来的跨侧或远距离接近。",
            "missing_phases": ["move_base_relative", "head_camera_rgbd_save", "head_camera_grounded_sam2_pose", "plan_cartesian_trajectory"],
            "failed_predicates": list(dict.fromkeys([*predicates, "pregrasp_pose_reachable=false", "target_pose_reached=false"])),
            "parameter_failure_summary": parameter_failure_summary,
            "task_goal": str(goal or "完成目标抓取与提升"),
        }

    if action == "lift" or failure_type == "object_not_lifted" or task_summary.get("object_lift_success") is False:
        return {
            "enabled": True,
            "rule_fallback": True,
            "failure_stage": "lift",
            "failure_type": "物体未被稳定提升",
            "root_cause": "提升动作完成但物体提升高度不足，说明夹持没有稳定带住目标物体。",
            "corrective_direction": "需要重新建立可靠夹持关系，再执行提升并确认目标随夹爪移动。",
            "missing_phases": ["head_camera_rgbd_save", "head_camera_grounded_sam2_pose", "approach_object", "close_gripper"],
            "failed_predicates": list(dict.fromkeys([*predicates, "object_lift_success=false", "grasp_secured=false"])),
            "parameter_failure_summary": parameter_failure_summary,
            "task_goal": str(goal or "完成目标抓取与提升"),
        }

    missing = []
    required_order = [
        ("head_camera_grounded_sam2_pose", "目标位置感知"),
        ("move_to_pregrasp", "到达预抓取"),
        ("approach_object", "接近抓取点"),
        ("close_gripper", "闭合夹爪"),
        ("lift", "提升目标"),
    ]
    for skill, label in required_order:
        if skill not in actions:
            missing.append(skill)
    return {
        "enabled": True,
        "rule_fallback": True,
        "failure_stage": str(failure_taxonomy.get("failure_stage") or "unknown"),
        "failure_type": str(failure_type or "未知失败"),
        "root_cause": str(failure_taxonomy.get("failure_reason") or "技能序列没有满足任务成功条件。")[:300],
        "corrective_direction": "需要补齐失败后缺失的任务闭环阶段，并避免重复已经失败的动作参数组合。",
        "missing_phases": _sanitize_missing_phases(missing[:4] or ["close_gripper", "lift"]),
        "failed_predicates": list(dict.fromkeys([*predicates, f"{action}_success=false" if action else "task_success=false"])),
        "parameter_failure_summary": parameter_failure_summary,
        "task_goal": str(goal or "完成目标任务闭环"),
    }


def _first_failed(action_reports: list[dict[str, Any]]) -> dict[str, Any]:
    for item in action_reports:
        if isinstance(item, dict) and not bool(item.get("success", False)):
            return _compact_action(item)
    return {}


def _compact_action(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("raw_result") if isinstance(item.get("raw_result"), dict) else {}
    params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
    keys = (
        "target_torso",
        "pregrasp_torso",
        "grasp_torso",
        "start_torso",
        "target_source_torso",
        "final_error",
        "object_body",
        "object_lift_world",
        "object_lift_success",
        "min_object_lift",
        "object_follow_error",
        "valid_depth_count",
        "bbox_xyxy",
    )
    evidence = {key: raw.get(key) for key in keys if raw.get(key) not in (None, "", [], {})}
    public_params = {str(key): value for key, value in params.items() if not str(key).startswith("_")}
    return {
        "index": item.get("index"),
        "action": item.get("action"),
        "success": bool(item.get("success", False)),
        "status": item.get("status"),
        "message": item.get("message"),
        "parameters": public_params,
        "evidence": evidence,
    }


def _compact_task_summary(task_summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task_success",
        "all_action_success",
        "lift_action_success",
        "object_lift_success",
        "object_body",
        "object_lift_world",
        "min_object_lift",
        "task_success_reason",
        "scenario_id",
        "condition_id",
        "episode_role",
    )
    return {key: task_summary.get(key) for key in keys if task_summary.get(key) not in (None, "", [], {})}
