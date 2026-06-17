"""
LLM handler for simulation anomaly handling.

Wraps doubao (豆包) VLM for anomaly verification, recovery planning, and scoring.
Adapts MuJoCo rendered images to VLM input format.

Three entry points mirror the root system's pipeline:
  verify_anomaly()   → corresponding to doubao.gen_content() + _action_prompt("提升物体")
  plan_recovery()    → corresponding to doubao.fault_recover()
  score_recovery()   → corresponding to doubao.get_sorce()
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env from experiment-sim-wrapper directory BEFORE importing doubao
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

import numpy as np
from ur5e.skills.registry import (
    GENERIC_ACTIONS,
    U1_ACTIONS,
    U2_ACTIONS,
    U3_ACTIONS,
    U4_ACTIONS,
    U5_ACTIONS,
    SCENARIO_BASE_ACTIONS,
    allowed_actions,
    skill_description,
    skill_signature,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# doubao module initialises Ark client at import time; needs ARK_API_KEY in env.
from doubao import (  # noqa: E402
    JSON_ONLY_LINE,
    ALLOWED_RECOVERY_ACTIONS,
    _action_prompt,
    _dump_json,
    _history_images,
    _history_records,
    _history_summary_text,
    _invoke_raw,
    _normalize_action_result,
    _normalize_recovery_step,
    _parse_json_payload,
    _sanitize_image_items,
    encode_file,
)

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


_SCENARIO_ACTIONS = {
    "U1": U1_ACTIONS,
    "U2": U2_ACTIONS,
    "U3": U3_ACTIONS,
    "U4": U4_ACTIONS,
    "U5": U5_ACTIONS,
}

def _format_skill_lines(actions: set[str]) -> str:
    return "\n".join(
        f"- {skill_signature(action)}：{skill_description(action)}"
        for action in sorted(actions)
    )


def _format_available_skills(scenario_id: str) -> str:
    scenario = (scenario_id or "").upper()
    full_actions = allowed_actions(scenario)
    base = SCENARIO_BASE_ACTIONS.get(scenario, allowed_actions("legacy"))
    generic_base = base & GENERIC_ACTIONS
    specialist = base - GENERIC_ACTIONS
    lines: list[str] = []
    if scenario in _SCENARIO_ACTIONS:
        lines.extend([f"当前 {scenario} 场景专项技能：", _format_skill_lines(_SCENARIO_ACTIONS[scenario])])
    if specialist:
        lines.extend(["放置阶段技能：", _format_skill_lines(specialist)])
    lines.extend(["基础运动与感知技能：", _format_skill_lines(generic_base)])
    return "\n".join(lines)


def _format_critic_skill_scope(scenario_id: str) -> str:
    """Skill scope for failure-memory critic output.

    The critic writes memory hints, not executable plans, but those hints must
    still be expressible by currently registered skills. Otherwise the memory
    library teaches the planner to ask for unsupported controller behavior.
    """
    scenario = (scenario_id or "").upper()
    actions = allowed_actions(scenario)
    skill_lines = _format_skill_lines(actions)
    virtual_motion_actions = [
        "detect-object",
        "create-grasp",
        "move-pregrasp",
        "move-grasp",
        "gripper-action",
        "vertical-grasp",
    ]
    return (
        f"当前场景可引用技能名={json.dumps(sorted(actions), ensure_ascii=False)}\n"
        f"技能说明：\n{skill_lines}\n"
        "在当前 sim_wrapper 虚拟验证中会直接改变运动/夹爪状态的技能："
        f"{', '.join(action for action in virtual_motion_actions if action in actions)}。\n"
        "检查类或参考修复类技能只能作为状态检查/内部参考更新来描述，不能替代实际运动、闭合或提升。"
    )


_UNSUPPORTED_CRITIC_TERMS = (
    "多点接触",
    "接触保持验证",
    "验证接触保持",
    "连续力控",
    "力控闭环",
    "稳定性控制器",
    "视觉伺服闭环",
)


def _contains_unsupported_critic_term(text: str) -> bool:
    return any(term in text for term in _UNSUPPORTED_CRITIC_TERMS)


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


def _looks_like_recovery_instruction(text: str) -> bool:
    if not text:
        return False
    return any(term in text for term in _DIRECTIVE_MEMORY_TERMS)


_SCENARIO_FOCUS = {
    "U1": "感知/目标确认异常，重点是恢复正确目标 apple 的感知、位姿或分割结果。",
    "U2": "抓取几何异常，重点是修正抓取位姿、预抓位姿或接近关系。",
    "U3": "夹爪/夹持保持异常，重点是确认夹爪状态、恢复稳定夹持或处理滑移。",
    "U4": "运输/放置异常，重点是让目标最终回到 plate 上的正确放置状态。",
    "U5": "路径/策略异常，重点是避开异常路径、切换策略或产生有效进展。",
}

_SUCCESS_CRITERIA_TEXT = {
    "redetect_correct_target_and_task": "重新确认正确目标，并继续完成最终任务闭环。",
    "redetect_unoccluded_target_and_task": "恢复可靠感知，并继续完成最终任务闭环。",
    "redetect_current_pose_and_task": "重新定位当前目标位置，并继续完成最终任务闭环。",
    "reestimate_orientation_and_task": "重新估计目标位姿，并继续完成最终任务闭环。",
    "resegment_clean_target_and_task": "重新获得干净目标分割，并继续完成最终任务闭环。",
    "lift_and_task": "恢复抓取并继续完成最终任务闭环。",
    "regrasp_lift_and_task": "重新稳定夹持/提升，并继续完成最终任务闭环。",
    "relocalize_lift_and_task": "重新定位目标、恢复提升，并继续完成最终任务闭环。",
    "replace_on_plate": "让目标最终位于 plate 上的正确位置。",
    "replace_on_plate_with_orientation": "让目标最终位于 plate 上，并尽量恢复正确姿态。",
    "replan_via_safe_waypoint": "通过安全路径或中间点恢复任务进展，并完成最终任务闭环。",
    "plan_reordered_and_task": "修正恢复计划中的动作逻辑问题，并完成最终任务闭环。",
    "switch_strategy_and_task": "在无进展时切换恢复策略，并完成最终任务闭环。",
}

_CONDITION_DESCRIPTIONS = {
    "U1-1": "当前异常是相似物体导致的目标混淆：系统可能把非目标物体当成 apple，需要重新确认正确目标并继续任务。",
    "U1-2": "当前异常是目标被部分遮挡：原始感知可能不完整。需要重新检测获取可靠位置后继续任务，使其最终回到 plate 上。",
    "U1-3": "当前异常是目标位姿过期：apple 的真实位置已经变化，原先抓取参考可能不再有效，需要重新定位当前目标。",
    "U1-4": "当前异常是目标姿态估计错误：apple 的朝向或抓取参考存在偏差，需要重新估计位姿后继续任务。",
    "U1-5": "当前异常是目标边界/背景混淆：分割结果可能包含背景或缺失目标区域。需要重新检测获取可靠位置后继续任务，使其最终回到 plate 上。",
    "U2-1": "当前异常是抓取位姿存在横向偏移：夹爪接近位置相对 apple 偏离，需要修正抓取位置后再执行抓取。",
    "U2-2": "当前异常是抓取高度偏移：夹爪高度不适合稳定接触 apple，需要修正抓取高度。",
    "U2-3": "当前异常是抓取姿态旋转偏差：夹爪朝向与目标不匹配，需要修正抓取姿态。",
    "U2-4": "当前异常是预抓位过近：接近阶段安全间隙不足，可能导致碰撞或错误接近，需要修正预抓位。",
    "U2-5": "当前异常是预抓位过远：接近阶段距离目标过远，后续抓取可能无法形成有效接触，需要修正预抓/抓取参考。",
    "U3-1": "当前异常是夹爪闭合失败：夹爪没有形成有效闭合，apple 未被稳定夹持。",
    "U3-2": "当前异常是夹爪只部分闭合：夹持力或闭合程度不足，apple 在提升时可能失稳。",
    "U3-3": "当前异常是夹爪过早闭合：apple 被提前闭合的夹爪推离了原始位置。",
    "U3-4": "当前异常是提升初期滑落：apple 在提升开始后不久从夹爪中脱落。",
    "U3-5": "当前异常是提升过程中的渐发滑移：apple 在提升过程中逐渐从夹爪中滑落。",
    "U4-1": "当前异常是运输阶段掉落：apple 在抓起后运输过程中脱离夹爪，需要重新定位、重新抓取并继续放置任务。",
    "U4-2": "当前异常是运输阶段目标位置变化：apple 已经偏离预期运输状态，需要重新定位目标并继续完成放置。",
    "U4-3": "当前异常是放置位置错误：apple 没有落在 plate 的正确区域（距 plate 中心超过12cm）。恢复需要经过：重新检测 apple 位置 → 抓取并提升 apple → 检测 plate 目标位置 → 将 apple 运输到 plate 上方 → 在 plate 上释放 apple → 验证放置结果。恢复成功的标志是：apple 最终位于 plate 上（距 plate 中心 < 12cm）、夹爪松开释放目标、机械臂回到 home 位置。",
    "U4-4": "当前异常是过早释放：apple 在到达正确放置位置前被释放，落在 plate 以外。恢复需要经过：重新检测 apple 位置 → 抓取并提升 apple → 检测 plate 目标位置 → 将 apple 运输到 plate 上方 → 在 plate 上释放 apple → 验证放置结果。恢复成功的标志是：apple 最终位于 plate 上、夹爪松开释放目标、机械臂回到 home 位置。",
    "U4-5": "当前异常是放置姿态错误：apple 位于 plate 上但朝向不正确。恢复需要经过：重新检测 apple 位置 → 抓取并提升 apple → 检测 plate 目标位置 → 将 apple 运输到 plate 上方 → 在 plate 上释放 apple → 验证放置结果。恢复成功的标志是：apple 位于 plate 上、朝向正确（四元数误差 < 0.25）、夹爪松开释放目标、机械臂回到 home 位置。",
    "U5-1": "当前异常是直线路径被阻挡：直接接近路径不可用，需要通过安全中间点或绕行策略恢复任务。",
    "U5-2": "当前异常是接近阶段可能碰撞邻近物体：需要避开风险区域并重新建立有效抓取/运输路径。",
    "U5-3": "当前异常是夹爪可能与桌面碰撞：当前接近路径高度或姿态不安全，需要调整路径后继续任务。",
    "U5-4": "当前异常是恢复动作顺序不合理：已有计划的动作逻辑可能无法完成闭环，需要重新组织恢复动作。",
    "U5-5": "当前异常是重复尝试没有进展：当前策略不能推进任务，需要切换恢复策略并形成有效进展。",
}

_STAGE_TEXT = {
    "perception": "异常发生在感知/目标确认阶段。",
    "move_pregrasp": "异常发生在移动到预抓位阶段。",
    "move_grasp": "异常发生在移动到抓取位阶段。",
    "gripper_close": "异常发生在夹爪闭合阶段。",
    "lift": "异常发生在提升阶段。",
    "transport": "异常发生在运输阶段。",
    "place": "异常发生在放置阶段。",
    "recovery_plan": "异常发生在恢复计划生成或执行逻辑阶段。",
}


def _format_condition_prompt_context(
    *,
    scenario_id: str,
    condition_id: str,
    condition_name: str,
    task_stage: str,
    injection_stage: str,
    success_criteria: str,
    failure_family: str,
) -> str:
    scenario = (scenario_id or "").upper()
    if not (scenario or condition_id or condition_name or success_criteria):
        return ""
    focus = _SCENARIO_FOCUS.get(scenario, "根据当前异常状态选择可用技能恢复任务。")
    success_text = _SUCCESS_CRITERIA_TEXT.get(success_criteria, "完成当前异常条件对应的最终物理闭环。")
    condition_text = _CONDITION_DESCRIPTIONS.get(
        condition_id,
        f"当前异常条件为 {condition_name or condition_id or scenario or '未知条件'}，需要结合已执行状态选择恢复技能。",
    )
    stage_text = _STAGE_TEXT.get(task_stage, "异常发生阶段需要从已执行技能状态中判断。")
    return (
        f"\n当前异常场景：{condition_id or scenario or '未指定'}。\n"
        f"异常说明：{condition_text}\n"
        f"发生阶段：{stage_text}\n"
        f"异常关注点：{focus}\n"
        f"该条件的恢复目标：{success_text}\n"
        "注意：以上描述只说明异常含义和最终物理目标，不规定固定技能顺序。"
    )


def _env_model(*names: str, default: str = "doubao-seed-1-6-vision-250815") -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return default


VERIFY_MODEL = _env_model(
    "EXPERIMENT_LLM_VERIFY_MODEL",
    "EXPERIMENT_LLM_MODEL",
    "DOUBAO_VERIFY_MODEL",
    "DOUBAO_MODEL_NAME",
)
RECOVERY_MODEL = _env_model(
    "EXPERIMENT_LLM_RECOVERY_MODEL",
    "EXPERIMENT_LLM_MODEL",
    "DOUBAO_RECOVERY_MODEL",
    "DOUBAO_MODEL_NAME",
)
SCORE_MODEL = _env_model(
    "EXPERIMENT_LLM_SCORE_MODEL",
    "EXPERIMENT_LLM_MODEL",
    "DOUBAO_SCORE_MODEL",
    "DOUBAO_MODEL_NAME",
)
FAILURE_CRITIC_MODEL = RECOVERY_MODEL


def _env_str(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    value = _env_str(*names)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _invoke_failure_critic_raw(content_blocks: list[dict[str, Any]]) -> str:
    return _invoke_raw(content_blocks, model=FAILURE_CRITIC_MODEL)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _encode_image_b64(image_path: Path | str) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = "png" if suffix == ".png" else "jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{data}"


def _build_image_block(image_path: Path | str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": _encode_image_b64(image_path)},
    }


def _load_images_from_dir(work_dir: Path | str, max_images: int = 3) -> list[dict[str, Any]]:
    """Load rendered images from a perception work directory."""
    d = Path(work_dir)
    blocks: list[dict[str, Any]] = []
    for name in sorted(d.iterdir()):
        if name.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            blocks.append(_build_image_block(name))
            if len(blocks) >= max_images:
                break
    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_anomaly(
    image_before: Path | str,
    image_after: Path | str,
    rule_z_change: float,
    rule_contact: Optional[dict] = None,
    perceived_z_before: Optional[float] = None,
    perceived_z_after: Optional[float] = None,
) -> dict[str, Any]:
    """VLM verification: is this really an anomaly?

    Sends before/after images to VLM to confirm whether the object
    was successfully lifted or an anomaly occurred.  Mirrors the
    root system's gen_content() → _action_prompt("提升物体") flow.

    Returns {"status": "SUCCESS"|"FAILURE", "reason": "...", "consider": "..."}
    """
    content_blocks: list[dict[str, Any]] = []
    content_blocks.append(_build_image_block(image_before))
    content_blocks.append(_build_image_block(image_after))

    context_parts = []
    if perceived_z_before is not None and perceived_z_after is not None:
        context_parts.append(
            f"规则检测: 提起前感知Z={perceived_z_before:.4f}m, "
            f"提起后感知Z={perceived_z_after:.4f}m, "
            f"ΔZ={rule_z_change:.4f}m (阈值0.03m)"
        )
    if rule_contact:
        context_parts.append(
            f"接触状态: left_pad={rule_contact.get('left_contact')}, "
            f"right_pad={rule_contact.get('right_contact')}"
        )
    if context_parts:
        content_blocks.append({"type": "text", "text": "\n".join(context_parts)})

    content_blocks.append({"type": "text", "text": _action_prompt("提升物体")})
    content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})

    raw_text = _invoke_raw(content_blocks, model=VERIFY_MODEL)
    payload = _normalize_action_result(_parse_json_payload(raw_text, prefer_array=False))
    return payload


def plan_recovery(
    task_history: list[dict[str, Any]],
    image_paths: list[Path | str],
    target: str = "apple",
    experiences: Optional[list[tuple[Any, float]]] = None,
    condition: str = "direct",
    experience_image_paths: Optional[list[Path | str]] = None,
    blocker_matches: Optional[list[dict[str, Any]]] = None,
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
    """LLM generates a recovery plan.

    Sends task state + images + optional memory context to JSON recovery plan.

    Args:
        task_history: list of {"action": str, "status": str, "reason": str} records
        image_paths: rendered images for VLM context (paths or base64 strings)
        target: target object class
        experiences: retrieved similar experiences from library, as list of
                     (MemoryV3Entry, similarity_score) tuples
        condition: "direct" or "sim_wrapper"

    Returns:
        list of recovery steps [{"action": str, "parameters": {...}}, ...]
    """
    task_list = [f"{h['action']}:{h['status']}" for h in task_history]
    scenario = (scenario_id or "").upper()
    action_labels = _format_available_skills(scenario)
    condition_context = _format_condition_prompt_context(
        scenario_id=scenario,
        condition_id=condition_id,
        condition_name=condition_name,
        task_stage=task_stage,
        injection_stage=injection_stage,
        success_criteria=success_criteria,
        failure_family=failure_family,
    )
    observation_context = ""
    if target_observation_status:
        observation_context = f"\n当前目标观测状态：{target_observation_status}\n"
    gripper_context = f"\n当前夹爪状态：{gripper_status}\n" if gripper_status else ""
    physics_constraint = "\n物理约束说明：\n- 夹爪闭合状态：在夹爪闭合的情况下移动末端靠近物体，会因夹爪占据物体空间而将物体推离原位，导致无法形成有效夹持。\n"

    # Experience context
    exp_context = ""
    if experiences:
        lines = [
            "\n以下是历史相似异常经验。成功案例中的步骤已转换为当前可用技能名；"
            "失败案例只作为反例，不要模仿其动作序列（尤其注意失败类型和错误动作序列）："
        ]
        succ_blocks: list[str] = []
        fail_blocks: list[str] = []
        for entry, score in experiences:
            converted_steps = _experience_steps_to_llm_actions(entry)
            is_success = bool(getattr(entry.result, "recovery_success", getattr(entry.result, "success", False)))
            if is_success:
                block: list[str] = []
                block.append(f"\n成功案例 (相似度={score:.2f}, 来源={entry.source}):")
                block.append(f"  条件编号: {getattr(entry, 'condition_id', '')}")
                validation_status = getattr(entry, "validation_status", "") or "unknown"
                if validation_status:
                    block.append(f"  验证状态: {validation_status}")
                dual_source_hint = _format_dual_source_memory_hint(entry)
                if dual_source_hint:
                    block.append(f"  双源经验提示: {dual_source_hint}")
                if converted_steps:
                    block.append(f"  可复用技能序列: {json.dumps(converted_steps, ensure_ascii=False)}")
                success_evidence = _format_success_memory_evidence(entry)
                if success_evidence:
                    block.append(f"  成功物理证据: {success_evidence}")
                keyframes = getattr(entry, "keyframes", []) or []
                if keyframes:
                    stages = [
                        str(frame.get("stage", "") if isinstance(frame, dict) else getattr(frame, "stage", ""))
                        for frame in keyframes[:4]
                    ]
                    stages = [stage for stage in stages if stage]
                    block.append(f"  关键帧: {len(keyframes)} 张" + (f", stages={json.dumps(stages, ensure_ascii=False)}" if stages else ""))
                if entry.summary:
                    block.append(f"  摘要: {entry.summary}")
                succ_blocks.extend(block)
            else:
                failure_taxonomy = getattr(entry, "failure_taxonomy", {}) or {}
                failure_type = failure_taxonomy.get("failure_type", "")
                block = [f"\n⚠ 失败案例 (相似度={score:.2f}, 来源={entry.source}):"]
                block.append(f"  条件编号: {getattr(entry, 'condition_id', '')}")
                dual_source_hint = _format_dual_source_memory_hint(entry)
                if dual_source_hint:
                    block.append(f"  双源经验提示: {dual_source_hint}")
                if failure_type:
                    block.append(f"  失败类型: {_ftype_cn(failure_type)}")
                if failure_taxonomy.get("failure_stage"):
                    block.append(f"  失败阶段: {failure_taxonomy.get('failure_stage')}")
                if getattr(entry.result, 'failure_reason', ''):
                    block.append(f"  失败原因: {getattr(entry.result, 'failure_reason', '')}")
                llm_critic = failure_taxonomy.get("llm_critic") or {}
                if isinstance(llm_critic, dict) and llm_critic.get("root_cause"):
                    block.append(f"  critic根因: {llm_critic.get('root_cause')}")
                corrective_direction = failure_taxonomy.get("corrective_direction") or (isinstance(llm_critic, dict) and llm_critic.get("corrective_direction")) or ""
                if corrective_direction:
                    block.append(f"  修正方向: {corrective_direction}")
                missing_phases = failure_taxonomy.get("missing_phases") or (isinstance(llm_critic, dict) and llm_critic.get("missing_phases")) or []
                if missing_phases:
                    block.append(f"  缺失阶段: {json.dumps(missing_phases, ensure_ascii=False)}")
                rule_critic = failure_taxonomy.get("rule_critic") or {}
                if isinstance(rule_critic, dict):
                    rule_flags = rule_critic.get("rule_flags") or []
                    if rule_flags:
                        flags_by_stage: dict[str, list[str]] = {}
                        for f in rule_flags[:6]:
                            if not isinstance(f, dict):
                                continue
                            stage = f.get("stage", "")
                            label = f.get("label_cn") or f.get("rule", "")
                            flags_by_stage.setdefault(stage, []).append(label)
                        stage_summary = "; ".join(
                            f"{s}: {', '.join(flags)}" for s, flags in flags_by_stage.items()
                        )
                        block.append(f"  规则检测: {len(rule_flags)} 个异常 —— {stage_summary}")
                if converted_steps:
                    block.append(f"  执行的错误动作序列: {json.dumps(converted_steps, ensure_ascii=False)}")
                if entry.summary:
                    block.append(f"  摘要: {entry.summary}")
                fail_blocks.extend(block)

        n_fail = len([e for e, _ in experiences if not bool(getattr(e.result, "recovery_success", getattr(e.result, "success", False)))])
        n_succ = len(experiences) - n_fail
        exp_lines = [f"\n当前共检索到 {len(experiences)} 条历史经验：{n_succ} 条成功，{n_fail} 条失败。"]
        if succ_blocks:
            exp_lines.append("\n─── 可参考的成功经验 ───")
            exp_lines.extend(succ_blocks)
        if fail_blocks:
            exp_lines.append("\n─── 必须避免的失败模式 ───")
            exp_lines.extend(fail_blocks)
        exp_context = "\n".join(exp_lines)

    # Condition hint
    if condition == "sim_wrapper":
        condition_hint = (
            "\n当前使用 sim_wrapper 模式：生成完整动作序列，系统会对计划进行虚拟仿真验证。"
        )
    else:
        condition_hint = (
            "\n当前使用 direct 模式：直接在仿真中执行恢复动作。"
        )

    blocker_context = ""
    if blocker_matches:
        blocker_lines = [
            "\n重要：你上一次生成的方案与失败经验高度相似，必须重写，不能复现以下失败动作模式："
        ]
        for index, match in enumerate(blocker_matches[:3], 1):
            blocker_lines.append(f"{index}. failed_experience_id={match.get('experience_id', '')}")
            blocker_lines.append(f"   overlap={float(match.get('overlap') or 0.0):.3f}")
            blocker_lines.append(f"   failure_type={match.get('failure_type', '')}")
            blocker_lines.append(f"   failed_signature={match.get('failed_signature', '')}")
        blocker_lines.append("请生成一个不同的恢复方案，并避免复现上述失败动作模式。")
        blocker_context = "\n".join(blocker_lines)

    prompt = f"""
你是mujoco仿真中的机械臂控制助手。当前任务是在异常发生后继续完成任务闭环,到达最终期望状态：正确目标{target}位于plate上，机械臂回到安全/结束状态。
	已执行的技能及状态：{task_list}
	场景/条件信息 :{condition_context}
	 当前目标观测状态:{observation_context}
	当前夹爪状态 :{gripper_context}
	物理约束 :{physics_constraint}
	模式提示:{condition_hint}
	检索到的历史经验 — 包含成功和失败案例:{exp_context}
    警告：{blocker_context}
	
	可用技能：
	{action_labels}
	
	⚠ 关键要求：仔细分析上面列出的失败案例，尤其是"失败类型"和"critic根因"和"corrective_direction"以及missing_phases。你的方案必须和失败案例中的错误动作序列不同，不能简单重复同样的步骤。
生成方案前，先分析给出的历史经验：critic根因意味着什么；corrective_direction中需要补充或者增强的是什么，missing_phases中少的是什么，为什么少了这些内容会失败。总结这些内容，仔细想想看为什么之前的方案会失败？
你的方案和失败方案有什么不同？确保不会犯同样的错误的同时，完成任务的闭环。注意你要仔细的观察exp_context中列出的失败经验。

请根据以上内容生成一个能够使异常恢复并完成任务闭环的方案，任务闭环需要包含所有必要的步骤，尽可能确保处理方案能够使任务闭环成功。
	
	输出要求：
	1. JSON 数组格式，每项含 action 和 parameters
	2. detect-object 参数为 target_class
	3. gripper-action 参数为 state (0=张开, 1=闭合)
	4. 每个 action 必须逐字选自“可用技能”中列出的技能名，不能输出未列出的技能
	5. 如果某个技能没有出现在上面的可用技能列表中，即使你认为有用，也禁止输出
	6. 不要 Markdown，只输出 JSON
	"""

    content_blocks: list[dict[str, Any]] = []
    for img_path in image_paths[:4]:
        content_blocks.append(_build_image_block(img_path))
    if experience_image_paths:
        content_blocks.append(
            {
                "type": "text",
                "text": "以下图像来自检索到的历史经验关键帧，仅作为相似异常状态参考，不要把其中的具体坐标当作当前坐标。",
            }
        )
        for img_path in experience_image_paths[:2]:
            content_blocks.append(_build_image_block(img_path))
    content_blocks.append({"type": "text", "text": prompt})
    content_blocks.append({"type": "text", "text": "只输出 JSON 数组，不要解释，不要代码块。"})

    if os.getenv("PRINT_RECOVERY_PROMPT", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("\n" + "=" * 24 + " RECOVERY PROMPT BEGIN " + "=" * 24)
        print(prompt.strip())
        print("=" * 25 + " RECOVERY PROMPT END " + "=" * 25 + "\n")

    raw_text = _invoke_raw(content_blocks, model=RECOVERY_MODEL)
    payload = _parse_json_payload(raw_text, prefer_array=True)

    steps: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_steps = payload.get("steps") or payload.get("actions") or []
        if isinstance(raw_steps, list):
            for raw_step in raw_steps:
                normalized = _normalize_recovery_step(raw_step, target=target)
                if normalized is not None:
                    steps.append(normalized)
    elif isinstance(payload, list):
        for raw_step in payload:
            normalized = _normalize_recovery_step(raw_step, target=target)
            if normalized is not None:
                steps.append(normalized)

    return steps


_FTYPE_CN_MAP: dict[str, str] = {
    "grasp_lift_failure_and_insufficient_recovery_steps": "抓取抬升失败，恢复步骤不足（缺少抓取或抬升动作）",
    "insufficient_recovery_steps_for_placement_and_lift": "恢复步骤不足（缺少放置和抬升动作）",
    "grasp_not_secured": "抓取未锁定（夹爪闭合但未抓住目标）",
    "no_grasp_attempted": "未尝试抓取（方案中没有抓取动作）",
    "grasp_position_error": "抓取位置偏差（预抓取位姿离目标过远）",
    "virtual_validation_failure": "虚拟仿真验证未通过",
    "recovery_execution_failure": "恢复执行失败",
}

def _ftype_cn(ftype: str) -> str:
    if not ftype:
        return ""
    if ftype in _FTYPE_CN_MAP:
        return _FTYPE_CN_MAP[ftype]
    return ftype.replace("_", " ")


def _experience_steps_to_llm_actions(entry: Any) -> list[dict[str, Any]]:
    """Convert replay-plan style experience steps to current LLM action schema.

    Experience entries are stored for replay as `cartesian_move` / `gripper`.
    The planner prompt, however, only accepts high-level skills such as
    `move-grasp` and `vertical-grasp`. Exposing replay action names directly
    makes the VLM mix action schemas and drop motion steps.
    """
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

    converted: list[dict[str, Any]] = []
    recovery_plan = getattr(entry, "recovery_plan", None)
    if isinstance(recovery_plan, dict):
        plan_steps = recovery_plan.get("steps", []) or []
    else:
        plan_steps = getattr(recovery_plan, "steps", []) or []
    for step in plan_steps:
        if isinstance(step, dict):
            step_type = str(step.get("type", "") or step.get("action", "")).strip()
            params = step.get("params") if isinstance(step.get("params"), dict) else step.get("parameters") or {}
        else:
            step_type = str(getattr(step, "type", "")).strip()
            params = getattr(step, "params", {}) or {}
        if step_type == "gripper":
            command = str(params.get("command", "")).strip().lower()
            if command == "open":
                converted.append({"action": "gripper-action", "parameters": {"state": 0}})
            elif command == "close":
                converted.append({"action": "gripper-action", "parameters": {"state": 1}})
            continue

        if step_type == "cartesian_move":
            label = str(params.get("label", "")).strip().lower()
            if label == "pregrasp":
                converted.append({"action": "move-pregrasp", "parameters": {}})
            elif label == "grasp":
                converted.append({"action": "move-grasp", "parameters": {}})
            elif label == "lift":
                converted.append({"action": "vertical-grasp", "parameters": {}})
            elif label == "place":
                # Placement is handled by the main experiment Step 9, not by recovery.
                pass
            continue

        if step_type == "joint_move":
            label = str(params.get("label", "")).strip().lower()
            if label in {"home", "init", "execute-init"}:
                # Returning home is handled by the main experiment Step 9.
                pass

    for index, step in enumerate(converted):
        if step.get("action") == "vertical-grasp":
            return converted[:index + 1]
    return converted


def _format_success_memory_evidence(entry: Any) -> str:
    feedback = getattr(entry, "execution_feedback", {}) or {}
    validation = getattr(entry, "validation_evidence", {}) or {}
    metadata = getattr(entry, "metadata", {}) or {}
    if hasattr(feedback, "__dataclass_fields__"):
        feedback = {
            key: getattr(feedback, key)
            for key in getattr(feedback, "__dataclass_fields__", {})
        }
    if not isinstance(feedback, dict):
        feedback = {}
    if not isinstance(validation, dict):
        validation = {}
    if not isinstance(metadata, dict):
        metadata = {}

    criteria = (
        feedback.get("recovery_success_criteria")
        or validation.get("recovery_success_criteria")
        or metadata.get("recovery_success_criteria")
        or {}
    )
    parts: list[str] = []

    # 1. 成功判据类型
    criteria_type = criteria.get("type") if isinstance(criteria, dict) else None
    if criteria_type:
        parts.append(f"成功判据={criteria_type}")

    # 2. 测量值与阈值对比
    measured: list[str] = []

    # apple_z: 优先从 criteria 取，fallback 到 apple_z_after_recovery
    apple_z = criteria.get("apple_z") if isinstance(criteria, dict) else None
    if apple_z is None:
        apple_z = feedback.get("apple_z_after_recovery") or metadata.get("apple_z_after_recovery")
    if apple_z is not None:
        measured.append(f"apple_z={apple_z:.4f}")

    baseline_z = criteria.get("baseline_z") if isinstance(criteria, dict) else None
    if baseline_z is not None:
        measured.append(f"baseline_z={baseline_z:.4f}")

    lift_from_table = criteria.get("lift_from_table") if isinstance(criteria, dict) else None
    if lift_from_table is not None:
        min_lift = criteria.get("min_lift") if isinstance(criteria, dict) else None
        if min_lift is not None:
            tag = "✅" if lift_from_table > min_lift else "❌"
            measured.append(f"lift_from_table={lift_from_table:.4f} (>{min_lift}{tag})")
        else:
            measured.append(f"lift_from_table={lift_from_table:.4f}")

    pinch_distance = criteria.get("pinch_distance") if isinstance(criteria, dict) else None
    if pinch_distance is not None:
        max_pinch = criteria.get("max_pinch_distance") if isinstance(criteria, dict) else None
        if max_pinch is not None:
            tag = "✅" if pinch_distance < max_pinch else "❌"
            measured.append(f"pinch_distance={pinch_distance:.4f} (<{max_pinch}{tag})")
        else:
            measured.append(f"pinch_distance={pinch_distance:.4f}")

    z_change = criteria.get("z_change") if isinstance(criteria, dict) else None
    if z_change is not None:
        measured.append(f"z_change={z_change:.4f}")

    tracked_apple = criteria.get("tracked_apple") if isinstance(criteria, dict) else None
    if tracked_apple is not None:
        measured.append(f"tracked_apple={tracked_apple}")

    grasp_secured = criteria.get("grasp_secured") if isinstance(criteria, dict) else None
    if grasp_secured is not None:
        measured.append(f"grasp_secured={grasp_secured}")

    success = criteria.get("success") if isinstance(criteria, dict) else None
    if success is not None:
        measured.append(f"success={success}")

    if measured:
        parts.append("测量值: " + ", ".join(measured))

    # 3. 接触状态
    contact_after_close = feedback.get("contact_after_close") or metadata.get("contact_after_close")
    contact_after_lift = feedback.get("contact_after_lift") or metadata.get("contact_after_lift")
    contact_parts: list[str] = []
    if contact_after_close:
        close_str = "+".join(k for k, v in contact_after_close.items() if v) or "none"
        contact_parts.append(f"close={close_str}")
    if contact_after_lift:
        lift_str = "+".join(k for k, v in contact_after_lift.items() if v) or "none"
        contact_parts.append(f"lift={lift_str}")
    if contact_parts:
        # compute contact pattern locally
        def _local_has_contact(d: dict) -> bool:
            return bool(d.get("left_contact") or d.get("right_contact") or d.get("contact"))
        close_ok = _local_has_contact(contact_after_close) if isinstance(contact_after_close, dict) else False
        lift_ok = _local_has_contact(contact_after_lift) if isinstance(contact_after_lift, dict) else False
        if close_ok and lift_ok:
            contact_parts.append("pattern=contact_close_and_lift")
        elif close_ok:
            contact_parts.append("pattern=contact_after_close_only")
        elif lift_ok:
            contact_parts.append("pattern=contact_after_lift_only")
        else:
            contact_parts.append("pattern=no_contact")
        parts.append("接触: " + ", ".join(contact_parts))

    # 4. 虚拟验证状态
    virtual_validation_success = validation.get("virtual_validation_success")
    if virtual_validation_success is not None:
        parts.append(f"虚拟验证={virtual_validation_success}")

    return "; ".join(parts[:15])


def _entry_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {key: getattr(value, key) for key in getattr(value, "__dataclass_fields__", {})}
    return value if isinstance(value, dict) else {}


def _format_dual_source_memory_hint(entry: Any) -> str:
    """Compact real/gap trust hint for the recovery prompt."""

    source = str(getattr(entry, "source", "") or "")
    validation_status = str(getattr(entry, "validation_status", "") or "")
    gap = _entry_dict(getattr(entry, "sim_real_gap", {}) or {})
    outcome = gap.get("outcome_gap") if isinstance(gap.get("outcome_gap"), dict) else {}
    outcome_type = str(outcome.get("type") or "")
    memory_tags = getattr(entry, "memory_tags", {}) if isinstance(getattr(entry, "memory_tags", {}), dict) else {}

    parts: list[str] = []
    if source or validation_status:
        parts.append(f"来源={source or 'unknown'}, 验证={validation_status or 'unknown'}")
    if memory_tags.get("memory_role"):
        parts.append(f"角色={memory_tags.get('memory_role')}")
    if outcome_type:
        gap_bits = [f"outcome={outcome_type}"]
        if gap.get("gap_score") is not None:
            gap_bits.append(f"gap_score={float(gap.get('gap_score') or 0.0):.2f}")
        if gap.get("uncertainty") is not None:
            gap_bits.append(f"uncertainty={float(gap.get('uncertainty') or 0.0):.2f}")
        parts.append("Sim-Real Gap: " + ", ".join(gap_bits))
        if outcome_type == "sim_success_real_fail":
            parts.append("风险提示: 该方案仿真成功但真实/伪真机失败，不能直接照搬，应调整接触、夹爪或感知步骤")
        elif outcome_type == "matched_success" and source in {"real", "pseudo_real"}:
            parts.append("可信提示: 该经验与真实/伪真机成功结果一致，可优先参考")
    elif validation_status in {"real_executed", "real_validated"} and getattr(getattr(entry, "result", None), "recovery_success", False):
        parts.append("可信提示: 真机/伪真机执行成功经验，可优先参考")
    return "; ".join(parts[:4])


def score_recovery(
    task_history: list[dict[str, Any]],
    image_paths: list[Path | str],
) -> dict[str, Any]:
    """LLM scores the recovery quality (0-10).

    Mirrors doubao.get_sorce().
    """
    task_list = [f"{h['action']}:{h['status']}" for h in task_history]

    summary_lines = []
    for i, h in enumerate(task_history, 1):
        summary_lines.append(
            f"{i}. action={h.get('action')} status={h.get('status')}"
        )
    history_text = "\n".join(summary_lines)

    prompt = f"""
你是mujoco仿真专家，评估机械臂异常处理后的最终效果。
技能流程及状态：{task_list}
动作历史摘要：
{history_text}

根据图像和技能流程判断异常处理是否成功，给出 0-10 分。
标准：
1. 物体被放置到指定位置，且 execute-init 回到初始状态 → 成功
2. 越早发现异常并处理，分数越高
3. task_list 越长，分数越低
JSON 格式输出：{{"status":"success/failure","score":"...","reason":"..."}}
"""

    content_blocks: list[dict[str, Any]] = []
    for img_path in image_paths[:3]:
        content_blocks.append(_build_image_block(img_path))
    content_blocks.append({"type": "text", "text": prompt})
    content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})

    raw_text = _invoke_raw(content_blocks, model=SCORE_MODEL)
    payload = _parse_json_payload(raw_text, prefer_array=True)
    if not isinstance(payload, dict):
        return {"status": "failure", "score": 0, "reason": "score response is not a JSON object"}
    return {
        "status": payload.get("status", "failure"),
        "score": payload.get("score", 0),
        "reason": payload.get("reason", ""),
    }


def critique_failure_experience(
    *,
    method: str,
    memory_policy: str,
    metrics: dict[str, Any],
    task_history: list[dict[str, Any]],
    recovery_steps: list[dict[str, Any]],
    retrieved_memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a compact failed-memory explanation with a separately configured LLM.

    This is intended for hierarchical/mixed experience libraries only. Baselines
    should not call it, otherwise the extra model becomes part of the baseline.
    """
    if not _env_bool("EXPERIMENT_FAILURE_CRITIC_ENABLED", default=True):
        return {"enabled": False, "skipped_reason": "disabled_by_env"}

    task_criteria = metrics.get("task_success_criteria") or {}
    scenario_id = str(metrics.get("scenario_id") or "")
    condition_id = str(metrics.get("condition_id") or "")
    skill_scope = _format_critic_skill_scope(scenario_id)
    compact_metrics = {
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "recovery_success": bool(metrics.get("recovery_success", False)),
        "task_success": bool(metrics.get("task_success", False)),
        "task_success_criteria": task_criteria,
        "recovery_success_criteria": metrics.get("recovery_success_criteria", {}),
        "apple_z_after_recovery": metrics.get("apple_z_after_recovery"),
        "observed_pos": metrics.get("observed_pos"),
        "contact_after_close": metrics.get("contact_after_close", {}),
        "contact_after_lift": metrics.get("contact_after_lift", {}),
        "virtual_validation_success": metrics.get("virtual_validation_success"),
        "virtual_execution_result": metrics.get("virtual_execution_result"),
        "executed_plan_source": metrics.get("executed_plan_source", ""),
        "invalid_skill_steps": metrics.get("invalid_skill_steps", []),
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
    compact_memories = []
    for item in (retrieved_memories or [])[:5]:
        if not isinstance(item, dict):
            continue
        compact_memories.append(
            {
                "partition": item.get("partition", ""),
                "status": item.get("status", ""),
                "failure_type": item.get("failure_type", ""),
                "plan_signature": item.get("plan_signature", ""),
            }
        )

    prompt = f"""
你是机械臂异常恢复实验的失败经验归因器。请基于结构化日志分析任务失败原因。

场景：
- method: {method}
- memory_policy: {memory_policy}
- condition_id: {condition_id}
- scenario_id: {scenario_id}

任务说明：
当前场景为 {scenario_id}（抓取任务异常恢复），条件为 {condition_id}。
总体目标：异常恢复后使 apple 回到 plate 上，完成抓取-提拉-放置闭环。

可用技能边界：
{skill_scope}

输入日志：
metrics={json.dumps(compact_metrics, ensure_ascii=False)}
task_history={json.dumps(history, ensure_ascii=False)}
recovery_steps={json.dumps(recovery_steps or [], ensure_ascii=False)}
retrieved_memories={json.dumps(compact_memories, ensure_ascii=False)}

要求：
1. 不要写死旧异常类别，输出应基于 condition_id、执行日志和恢复结果。
2. 如果恢复抓取成功但最终任务失败，要分析 task_success_criteria 中哪个谓词失败。
3. 分析失败原因时，对比 recovery_steps（已执行的恢复步骤）与完成任务实际需要的动作序列，指出缺失或错误的步骤。
4. 如有 virtual_execution_result.step_trace（虚拟仿真逐步骤状态记录），请使用时序信息：每一步前后的 apple_z、grasp_tracking、pinch_distance 变化。尤其注意：如果 grasp_tracking 曾经为 true 但最终为 false，说明夹爪曾抓住目标但后续释放了，失败原因可能不是”没抓到”而是”抓住后不当释放”。
5. 如有 condition_injection 字段，其中可能包含注入参数供分析：
   - injector: 注入类型（如 occlusion_noise、boundary_confusion、stale_pose、grasp_pose_offset 等）
   - true_pos / perceived_pos: 目标真实位置与感知位置（如有），对比两者差异可判断感知偏移量
   - params.attach_max_distance: 夹爪吸附距离上限（如有），与感知偏移量对比可判断抓取是否物理可行
   - 请基于上述数据自行分析注入对失败的影响，不要预设结论。
6. 不要建议当前技能无法实现的能力，例如”多点接触””接触保持验证””连续力控””稳定性控制器””视觉伺服闭环”等。
7. 请根据要求的实验闭环，分析该方案为什么无法成功执行任务，提供一个方向性的修正建议。禁止在corrective_direction中出现任何具体技能名（如create-grasp、move-grasp、detect-object等），
只能描述缺失的流程阶段。
8. 根据分析生成 missing_phases：列出该方案缺失了哪些关键流程阶段（2-4 项），用自然语言描述，禁止包含具体技能名。

只输出 JSON 对象，字段如下：
{{
  “failure_stage”: “detection|recovery_plan|recovery_execution|virtual_validation|task_completion|memory_reuse|unknown”,
  “failure_type”: “中文失败类型名称（如：抓取未执行、抓取位置偏差、抬升不足、步骤缺失等）”,
  “root_cause”: “中文短句，解释失败的根本原因”,
  “corrective_direction”: “方向性修正建议，说明该方案缺失了哪些内容导致的失败，禁止包含任何具体技能名，禁止输出过于具体的建议”,
  “missing_phases”: [“缺失阶段1”, “缺失阶段2”],
  “failed_predicates”: [“...”],
  “task_goal”: “当前任务的最终目标的一句话总结”
}}
"""
    raw_text = _invoke_failure_critic_raw(
        [
            {"type": "text", "text": prompt},
            {"type": "text", "text": JSON_ONLY_LINE},
        ]
    )
    payload = _parse_json_payload(raw_text, prefer_array=False)
    if not isinstance(payload, dict):
        return {
            "enabled": True,
            "error": "critic_response_not_json_object",
            "raw_text": str(raw_text)[:500],
        }

    allowed_stages = {
        "detection",
        "recovery_plan",
        "recovery_execution",
        "virtual_validation",
        "task_completion",
        "memory_reuse",
        "unknown",
    }
    stage = str(payload.get("failure_stage") or "unknown").strip()
    if stage not in allowed_stages:
        stage = "unknown"
    predicates = payload.get("failed_predicates")
    if not isinstance(predicates, list):
        predicates = []
    evidence = payload.get("failure_evidence")
    if not isinstance(evidence, list):
        evidence = []
    unsupported_detected = False
    root_cause = str(payload.get("root_cause") or "").strip()
    if _contains_unsupported_critic_term(root_cause):
        unsupported_detected = True

    result = {
        "enabled": True,
        "model": FAILURE_CRITIC_MODEL,
        "failure_stage": stage,
        "failure_type": str(payload.get("failure_type") or "").strip()[:80],
        "root_cause": root_cause[:300],
        "corrective_direction": str(payload.get("corrective_direction") or "").strip()[:200],
        "missing_phases": [str(item).strip()[:60] for item in (payload.get("missing_phases") or [])[:4]],
        "failed_predicates": [str(item)[:80] for item in predicates[:8]],
    }
    if unsupported_detected:
        result["unsupported_terms_removed"] = list(_UNSUPPORTED_CRITIC_TERMS)
        result["critic_warning"] = "critic output contained unsupported capability terms; no recovery instruction was generated by code"
    return result
