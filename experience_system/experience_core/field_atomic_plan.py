"""LLM planning helpers for field-style atomic robot actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .llm_provider import JSON_ONLY_LINE, build_image_block, invoke_llm, invoke_multimodal_llm, parse_json_payload
from .schema import (
    GALAXEA_R1PRO_TORSO_NAMESPACE,
    SkillCatalog,
    canonical_skill_action,
    coerce_skill_catalogs,
    default_skill_catalogs,
)


FIELD_ATOMIC_PARAMETER_GUIDANCE: dict[str, dict[str, Any]] = {
    "move_base_relative": {
        "x": [-0.1, 0.5],
        "y": [-0.1, 0.5],
    },
    "set_torso_posture": {
        "level": ["mid", "high"],
    },
    "head_camera_rgbd_save": {},
    "head_camera_grounded_sam2_pose": {
        "target_class": ["apple", "red sphere", "red cube", "red box", "green box"],
    },
    "plan_cartesian_trajectory": {
        "side": ["left", "right"],
        "target_class": ["apple", "red sphere", "red cube", "red box", "green box"],
        "mode": ["straight", "top_then_down", "side_then_in"],
    },
    "move_to_pregrasp": {
        "side": ["left", "right"],
        "target_class": ["apple", "red sphere", "red cube", "red box", "green box"],
    },
    "approach_object": {
        "side": ["left", "right"],
        "target_class": ["apple", "red sphere", "red cube", "red box", "green box"],
    },
    "close_gripper": {
        "side": ["left", "right"],
    },
    "open_gripper": {
        "side": ["left", "right"],
    },
    "lift": {
        "side": ["left", "right"],
        "target_class": ["apple", "red sphere", "red cube", "red box", "green box"],
        "lift_height": [0.02, 0.25],
    },
    "lower_held_object": {
        "side": ["left", "right"],
        "lower_distance": [0.0, 0.08],
    },
    "transport_to_detected_target": {
        "side": ["left", "right"],
        "target_class": ["apple", "red sphere", "red cube", "red box", "green box"],
        "place_offset_x": [-0.02, 0.02],
        "place_offset_y": [-0.02, 0.02],
    },
}

FIELD_ATOMIC_LLM_PARAMETER_GUIDANCE = FIELD_ATOMIC_PARAMETER_GUIDANCE

FIELD_ATOMIC_PARAMETER_NOTES: dict[str, dict[str, str]] = {
    "move_base_relative": {
        "x": "底盘沿当前坐标系 x 轴的相对位移，单位米。移动底盘会改变机器人、相机和 torso_link4 相对物体的位置；后续如果依赖物体位置，必须重新执行 。",
        "y": "底盘沿当前坐标系 y 轴的相对位移，单位米。移动底盘会让此前保存的物体 torso 坐标失效；不要在底盘移动后直接复用旧感知位置规划抓取。注意x和y不一定要数值保持一致",
    },
    "set_torso_posture": {
        "level": "躯干高度档位。只能使用 mid 或 high；high 对应真机直立目标 [0.0, 0.0, 0.0, 0.0]，mid 对应真机中位目标 [0.87, -1.35, -0.48, 0.0]。仿真内部会按 MuJoCo 关节方向映射。腰部移动后必须重新感知物体位置。",
    },
    "head_camera_rgbd_save": {},
    "head_camera_grounded_sam2_pose": {
        "target_class": "要识别的物体类别。输出的位置 JSON 会作为后续运动技能的输入。",
    },
    "plan_cartesian_trajectory": {
        "side": "执行轨迹的机械臂侧别：left 或 right。通常 torso_link4 坐标下目标 y>0 更适合 left，y<0 更适合 right；如果历史失败显示跨侧抓取不可达，应切换到目标同侧手臂。",
        "target_class": "读取其最新感知位置的物体类别。",
        "pregrasp_offset_x": "从抓取点到预抓点在 torso_link4 坐标系下的 x 偏移。",
        "pregrasp_offset_y": "从抓取点到预抓点在 torso_link4 坐标系下的 y 偏移。",
        "pregrasp_offset_z": "从抓取点到预抓点在 torso_link4 坐标系下的 z 偏移。z 为正表示向上抬高。",
        "mode": "轨迹形态。straight 表示直接到预抓点；top_then_down 表示先抬高默认 clearance_z，再平移到预抓上方，最后下降；side_then_in 表示使用默认侧向入口点再进入预抓点。",
        "side_offset_x": "side_then_in 模式下的侧向入口 x 偏移。入口点 = pregrasp + [side_offset_x, side_offset_y, 0]。",
        "side_offset_y": "side_then_in 模式下的侧向入口 y 偏移。入口点 = pregrasp + [side_offset_x, side_offset_y, 0]。",
        "clearance_z": "top_then_down 模式下的抬高距离。值越大，先抬得越高再靠近目标。",
        "topdown_mode": "轨迹规划时使用的末端朝向预设。",
    },
    "move_to_pregrasp": {
        "side": "执行移动的机械臂侧别：left 或 right。通常 torso_link4 坐标下目标 y>0 更适合 left，y<0 更适合 right；如果跨侧抓取导致 move_to_pregrasp 失败，应考虑同侧手臂。",
        "target_class": "作为抓取锚点的物体类别，其最新位置会被读取。",
        "pregrasp_offset_x": "从抓取点到预抓点在 torso_link4 坐标系下的 x 偏移。",
        "pregrasp_offset_y": "从抓取点到预抓点在 torso_link4 坐标系下的 y 偏移。",
        "pregrasp_offset_z": "从抓取点到预抓点在 torso_link4 坐标系下的 z 偏移。z 为正表示向上抬高。",
        "topdown_mode": "移动过程中使用的末端朝向预设。",
    },
    "approach_object": {
        "side": "执行最终接近的机械臂侧别：left 或 right。应和 plan_cartesian_trajectory、move_to_pregrasp 使用同一侧；目标在 torso y<0 时优先考虑 right，y>0 时优先考虑 left。",
        "target_class": "作为抓取锚点的物体类别，其最新位置会被读取。",
        "visual_grasp_offset_z": "在识别到的抓取点基础上额外增加的 z 修正量。",
        "topdown_mode": "最终接近时使用的末端朝向预设。",
    },
    "close_gripper": {
        "side": "要闭合的夹爪侧别。",
    },
    "open_gripper": {
        "side": "要张开的夹爪侧别。",
    },
    "lift": {
        "side": "抓取后执行提升的机械臂侧别。",
        "target_class": "用于解析被抓取物体并进行提升评估的物体类别。",
        "lift_height": "沿 torso 正 z 方向提升的距离，单位米。",
    },
    "lower_held_object": {
        "side": "搬运后仍夹住物体的机械臂侧别。",
        "lower_distance": "保持夹爪闭合，沿 torso_link4 负 z 方向下降的距离，限制在 0 到 0.08m。",
    },
    "transport_to_detected_target": {
        "side": "搬运抓取物体的机械臂侧别。",
        "target_class": "搬运目标参考物体类别。该技能读取前面感知技能保存的 target_class 位置 JSON 作为放置参考。",
        "place_offset_x": "相对于检测到的目标位置，在 torso_link4 坐标系 x 方向的小放置偏移，限制在正负 0.02m 内。",
        "place_offset_y": "相对于检测到的目标位置，在 torso_link4 坐标系 y 方向的小放置偏移，限制在正负 0.02m 内。",
    },
}


def field_atomic_recovery_prompt(
    *,
    recovery_context: dict[str, Any],
    planner_input: dict[str, Any] | None = None,
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    skill_catalogs: dict[str, Any] | None = None,
    max_steps: int = 8,
) -> str:
    guidance = _guidance_for_namespace(skill_namespace, skill_catalogs, llm_view=True)
    skill_specs = _render_field_atomic_skill_specs(guidance, skill_namespace=skill_namespace)
    memory_context = _render_field_atomic_memory_context(planner_input or {})
    context_for_prompt = dict(recovery_context)
    candidate_generation = context_for_prompt.get("candidate_generation") if isinstance(context_for_prompt.get("candidate_generation"), dict) else {}
    candidate_count = int(candidate_generation.get("candidate_count") or 1) if candidate_generation else 1
    candidate_diversity_rule = ""
    if candidate_count > 1:
        candidate_diversity_rule = (
            "- 当前是多候选生成模式；本候选必须和其他候选有真实差异。"
            "move_base_relative 的 x/y、plan_cartesian_trajectory 或 move_to_pregrasp 的 side、"
            "plan_cartesian_trajectory 的 mode 至少一项应不同。"
            "禁止多个候选完全相同，禁止只改 lift_height 但底盘/手臂/轨迹策略完全相同。\n"
        )
    context_for_prompt.pop("image_paths", None)
    context_for_prompt.pop("experience_image_paths", None)
    context_for_prompt.pop("retrieved_experiences", None)
    context_for_prompt.pop("retrieval_policy", None)
    context_for_prompt.pop("visual_retrieval", None)
    return f"""
你是 Galaxea 机器人异常恢复实验中的多模态恢复规划助手。
你会看到 MuJoCo 仿真关键帧，并收到结构化异常日志。请根据图像、异常状态、执行历史和历史经验，输出可以直接执行的 Galaxea field atomic 恢复技能序列。

当前只能使用以下技能名：
{json.dumps(sorted(guidance), ensure_ascii=False, indent=2)}

技能参数说明：
{skill_specs}

恢复规划规则：
- 不要输出固定模板，要根据异常现场图像和日志判断下一步恢复动作。
- 成功经验要优先提取可复用的动作顺序、参数趋势和成功证据，不要只看动作名。
- 失败经验只能作为反例，不能照抄失败技能序列。
- 对失败经验必须像逐条分析：failure_type 表示失败类别，missing_phases 表示缺失的流程阶段，failed_predicates 表示未满足条件,parameter_failure_summary的items表示出现错误参数的动作和需要修改的参数。
- 如果当前候选与失败经验的错误动作序列相似，必须改变方案结构或关键参数，不能只微调一个已经失败的参数后重复同一失败模式。
- 参数错误归因必须按 parameter_failure_summary 判断：仔细思考，parameter_failure_summary中的技能参数为什么错误，如何修改来避免这些错误
- 如果 recovery_context 中存在 rewrite_feedback 或 blocker_matches，说明上一轮候选已经被历史失败或 MuJoCo 验证否定；新方案必须针对其中的失败证据做结构性改变，并继续完成原始 goal。
{candidate_diversity_rule}

历史经验摘要：
{memory_context}

	⚠ 关键要求：仔细分析上面列出的失败案例，尤其是"failure_type","parameter_failure_summary"和"missing_phases"。你的方案必须和失败案例中的错误动作序列和动作参数不同，不能简单重复同样的步骤。
生成方案前，先分析给出的历史经验：missing_phases 中少的是什么，为什么少了这些内容会失败。总结这些内容，仔细想想看为什么之前的方案会失败？parameter_failure_summary每个错误的技能动作中参数到底哪里不对，为什么会导致失败,是哪里还不足参数的调整需要比较细致，reason中为什么出现错误也需要仔细看，根据这些内容来调整技能序列和技能的参数。
你的方案和失败方案有什么不同？确保不会犯同样的错误的同时，完成任务的闭环。注意你要仔细的观察exp_context中列出的失败经验。


当前现场上下文：
{json.dumps(context_for_prompt, ensure_ascii=False, indent=2)}




输出要求：
- 只返回一个 JSON 对象。
- 顶层 JSON 只能包含 steps 字段。
- 每个 step 只能包含 action 和 parameters 字段。
- steps 必须包含 1 到 {max_steps} 个技能。
- action 必须逐字匹配允许技能名。
- parameters 只能包含该技能允许的参数。
- 只输出“技能参数说明”里列出的参数；不要输出未列出的底层控制参数或轨迹微调参数。
- plan_cartesian_trajectory、move_to_pregrasp、approach_object 未列出的 pregrasp_offset_x/y/z、topdown_mode、clearance_z、side_offset_x/y、visual_grasp_offset_z 会由执行器使用稳定默认值，不要自行输出。
- 如果失败归因指向底盘/腰部空间关系，优先改变 move_base_relative 和 set_torso_posture；不要把主要变化放在 pregrasp_offset 或 topdown_mode 上。
- 所有会影响实验条件且已列出的参数必须显式输出，例如 mode、lift_height、place_offset_x/y。
- 只输出 JSON，不要 Markdown，不要解释文字，不要代码块。

生成前内部检查：
- 失败案例中的“执行的错误动作序列/plan_signature”不能被简单复现。
- 如果 missing_phases 指出缺失了某个流程阶段，输出 steps 必须体现该阶段对应的可用原子技能。
- 如果 failed_predicates 指出某个目标条件没有满足，输出 steps 必须覆盖该条件直到任务闭环完成。

JSON 格式：
{{
  "steps": [
    {{
      "action": "一个允许的 field atomic 技能名",
      "parameters": {{}}
    }}
  ]
}}
{JSON_ONLY_LINE}
"""


def invoke_field_atomic_recovery_vlm(
    recovery_context: dict[str, Any],
    *,
    planner_input: dict[str, Any] | None = None,
    provider: str = "doubao",
    model: str = "",
    skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE,
    skill_catalogs: dict[str, Any] | None = None,
    max_steps: int = 8,
) -> dict[str, Any]:
    prompt = field_atomic_recovery_prompt(
        recovery_context=recovery_context,
        planner_input=planner_input,
        skill_namespace=skill_namespace,
        skill_catalogs=skill_catalogs,
        max_steps=max_steps,
    )
    content_blocks: list[dict[str, Any]] = []
    for path in _select_recovery_images(recovery_context.get("image_paths"), limit=4):
        content_blocks.append(build_image_block(path))
    experience_images = _select_recovery_images(recovery_context.get("experience_image_paths"), limit=2)
    if experience_images:
        content_blocks.append({
            "type": "text",
            "text": "以下图像来自检索到的历史经验关键帧，仅作为相似异常状态参考，不要把其中坐标当作当前坐标。",
        })
        for path in experience_images:
            content_blocks.append(build_image_block(path))
    content_blocks.append({"type": "text", "text": prompt})
    content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})
    raw = invoke_multimodal_llm(
        content_blocks,
        provider=provider,
        model=model,
        system_prompt="你生成 Galaxea 多模态异常恢复技能序列。必须只返回 JSON。",
        temperature=0.25,
    )
    payload = parse_json_payload(raw, prefer_array=False)
    if not isinstance(payload, dict):
        raise RuntimeError("field atomic recovery response must be a JSON object")
    payload["_raw_text"] = raw
    return payload


def sanitize_field_atomic_parameters(
    action: str,
    parameters: dict[str, Any],
    *,
    guidance: dict[str, dict[str, Any]] | None = None,
    llm_view: bool = False,
) -> dict[str, Any]:
    source = guidance or (FIELD_ATOMIC_LLM_PARAMETER_GUIDANCE if llm_view else FIELD_ATOMIC_PARAMETER_GUIDANCE)
    guidance = source.get(action, {})
    clean: dict[str, Any] = {}
    for key, value in parameters.items():
        key = str(key)
        if key not in guidance:
            continue
        bounded = _coerce_parameter(guidance[key], value)
        if bounded is not None:
            clean[key] = bounded
    return clean


def _render_field_atomic_skill_specs(guidance: dict[str, dict[str, Any]], *, skill_namespace: str = GALAXEA_R1PRO_TORSO_NAMESPACE) -> str:
    lines: list[str] = []
    preferred_order = [
        "move_base_relative",
        "set_torso_posture",
        "head_camera_rgbd_save",
        "head_camera_grounded_sam2_pose",
        "plan_cartesian_trajectory",
        "move_to_pregrasp",
        "approach_object",
        "close_gripper",
        "lift",
        "lower_held_object",
        "transport_to_detected_target",
        "open_gripper",
    ]
    ordered_actions = [name for name in preferred_order if name in guidance]
    ordered_actions.extend(name for name in sorted(guidance) if name not in set(ordered_actions))
    for action in ordered_actions:
        params = guidance.get(action, {})
        notes = FIELD_ATOMIC_PARAMETER_NOTES.get(action, {})
        lines.append(f"- {action}")
        lines.append(f"  parameters: {json.dumps(params, ensure_ascii=False)}")
        if notes:
            lines.append("  meanings:")
            for key in params:
                lines.append(f"    - {key}: {notes.get(key, '')}")
        lines.extend(_action_usage_notes(action))
    return "\n".join(lines)


def _render_field_atomic_memory_context(planner_input: dict[str, Any]) -> str:
    if not planner_input:
        return "当前没有提供历史经验。按静态技能参数说明生成保守方案。"

    lines: list[str] = []
    scenario_id = str(planner_input.get("scenario_id") or "")
    if scenario_id:
        lines.append(f"- 当前检索场景：scenario_id={scenario_id}。")

    priors = planner_input.get("field_atomic_parameter_priors")
    if isinstance(priors, dict):
        total = int(priors.get("field_atomic_entry_count") or 0)
        succ = int(priors.get("field_atomic_success_count") or 0)
        fail = int(priors.get("field_atomic_failure_count") or 0)
        lines.append(f"- 历史 field_atomic 经验数量：总计 {total} 条，成功 {succ} 条，失败 {fail} 条。")
        semantic_failure_summary = priors.get("semantic_plan_failure_summary")
        if isinstance(semantic_failure_summary, dict) and semantic_failure_summary:
            lines.append(
                "- 前置语义失败经验："
                f"count={semantic_failure_summary.get('entry_count', 0)}, "
                f"failure_ids={_compact_json(semantic_failure_summary.get('failure_ids'))}, "
                f"failure_types={_compact_json(semantic_failure_summary.get('failure_types'))}, "
                f"memory_lessons={_compact_json(semantic_failure_summary.get('memory_lessons'))}"
            )
        by_action = priors.get("by_action")
        if isinstance(by_action, dict) and by_action:
            lines.append("- 按技能整理的参数经验：")
            for action, summary in sorted(by_action.items()):
                if not isinstance(summary, dict):
                    continue
                rate = summary.get("success_rate", 0.0)
                lines.append(f"  - {action}: success_rate={rate}, success={summary.get('success_count', 0)}, failure={summary.get('failure_count', 0)}")
                success_params = summary.get("recommended_from_success")
                if isinstance(success_params, dict) and success_params:
                    lines.append(f"    成功参数正例: {_compact_json(success_params)}")
                failure_params = summary.get("avoid_from_failure")
                if isinstance(failure_params, dict) and failure_params:
                    lines.append(f"    失败参数反例: {_compact_json(failure_params)}")
                parameter_failures = summary.get("parameter_failure_summary")
                if isinstance(parameter_failures, dict) and parameter_failures:
                    lines.append(f"    参数错误归因: {_compact_json(parameter_failures)}")
                success_ids = summary.get("success_ids")
                if isinstance(success_ids, list) and success_ids:
                    lines.append(f"    成功证据ID: {', '.join(str(item) for item in success_ids[:6])}")
                failure_ids = summary.get("failure_ids")
                if isinstance(failure_ids, list) and failure_ids:
                    lines.append(f"    失败证据ID: {', '.join(str(item) for item in failure_ids[:6])}")

    recovery_rules = planner_input.get("galaxea_recovery_rules")
    if isinstance(recovery_rules, dict) and recovery_rules.get("rule_count"):
        lines.append("- Galaxea 恢复规则摘要（由失败经验聚合得到）：")
        required_actions = recovery_rules.get("required_actions")
        if isinstance(required_actions, list) and required_actions:
            lines.append(f"  必须考虑的技能: {_compact_json(required_actions)}")
        relocalize = recovery_rules.get("must_relocalize_after")
        if isinstance(relocalize, list) and relocalize:
            lines.append(f"  执行后必须重新感知的技能: {_compact_json(relocalize)}")
        forbidden = recovery_rules.get("forbidden_patterns")
        if isinstance(forbidden, list) and forbidden:
            lines.append(f"  已重复失败/应避免的参数模式: {_compact_json(forbidden)}")
        suggested = recovery_rules.get("suggested_parameter_region")
        if isinstance(suggested, dict) and suggested:
            lines.append(f"  建议参数区域: {_compact_json(suggested)}")
        rules = recovery_rules.get("rules")
        if isinstance(rules, list) and rules:
            compact_rules = [
                {
                    "rule_id": item.get("rule_id"),
                    "failure_stage": item.get("failure_stage"),
                    "failure_type": item.get("failure_type"),
                    "support_count": item.get("support_count"),
                    "forbidden_patterns": item.get("forbidden_patterns"),
                    "required_actions": item.get("required_actions"),
                    "suggested_parameter_region": item.get("suggested_parameter_region"),
                }
                for item in rules[:4]
                if isinstance(item, dict)
            ]
            lines.append(f"  规则证据: {_compact_json(compact_rules)}")

    success_items = planner_input.get("recent_field_atomic_successes")
    failure_items = planner_input.get("recent_field_atomic_failures")
    mixed_items = planner_input.get("recent_field_atomic_experiences")
    retrieved_items = planner_input.get("retrieved_experiences")
    rewrite_feedback = planner_input.get("rewrite_feedback") if isinstance(planner_input.get("rewrite_feedback"), dict) else {}

    def _append_success_items(items: Any, title: str) -> None:
        if not isinstance(items, list) or not items:
            return
        lines.append(title)
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"id={item.get('experience_id', '')}, "
                f"role={item.get('memory_role', '')}, "
                f"action={item.get('action', '')}, "
                f"plan={item.get('plan_signature', '')}, "
                f"parameters={_compact_json(item.get('parameters') if isinstance(item.get('parameters'), dict) else {})}, "
                f"success_evidence={_compact_json(item.get('success_evidence')) if item.get('success_evidence') is not None else ''}, "
                f"success_reason={item.get('success_reason', '')}, "
                f"text_summary={item.get('text_summary', '')}"
            )

    def _append_failure_items(items: Any, title: str) -> None:
        if not isinstance(items, list) or not items:
            return
        lines.append(title)
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"id={item.get('experience_id', '')}, "
                f"role={item.get('memory_role', '')}, "
                f"action={item.get('action', '')}, "
                f"parameters={_compact_json(item.get('parameters') if isinstance(item.get('parameters'), dict) else {})}, "
                f"llm_critic={_compact_json(item.get('llm_critic'))}"
            )

    def _append_parameter_lessons(items: Any, title: str) -> None:
        lessons = _collect_failure_parameter_lessons(items)
        if not lessons:
            return
        lines.append(title)
        for lesson in lessons[:8]:
            lines.append(lesson)

    lines.append("- 经验展示顺序必须是：先成功经验，再失败经验。")
    _append_success_items(success_items, "- 最近成功经验，可直接参考其动作顺序、参数趋势和成功证据：")
    _append_parameter_lessons(failure_items, "- 历史失败参数教训（优先看这里，再看普通失败经验）：")
    _append_failure_items(failure_items, "- 最近失败经验，只能作为反例，避免重复相同技能和参数组合：")
    if isinstance(retrieved_items, list) and retrieved_items:
        retrieved_success = [item for item in retrieved_items if isinstance(item, dict) and bool(item.get("success"))]
        retrieved_failure = [item for item in retrieved_items if isinstance(item, dict) and not bool(item.get("success"))]
        _append_success_items(retrieved_success, "- 检索到的成功经验：")
        _append_parameter_lessons(retrieved_failure, "- 检索到的失败参数教训：")
        _append_failure_items(retrieved_failure, "- 检索到的失败经验：")
    if not success_items and not failure_items and isinstance(mixed_items, list) and mixed_items:
        success_fallback = [item for item in mixed_items if isinstance(item, dict) and bool(item.get("success"))]
        failure_fallback = [item for item in mixed_items if isinstance(item, dict) and not bool(item.get("success"))]
        _append_success_items(success_fallback, "- 最近成功经验，可直接参考其动作顺序、参数趋势和成功证据：")
        _append_parameter_lessons(failure_fallback, "- 历史失败参数教训（优先看这里，再看普通失败经验）：")
        _append_failure_items(failure_fallback, "- 最近失败经验，只能作为反例，避免重复相同技能和参数组合：")

    if rewrite_feedback:
        lines.append("- 上轮重写反馈：")
        blocker_matches = rewrite_feedback.get("blocker_matches")
        if isinstance(blocker_matches, list) and blocker_matches:
            lines.append(f"  blocker_matches={_compact_json(blocker_matches)}")
        failed_candidate_score_history = rewrite_feedback.get("failed_candidate_score_history")
        if isinstance(failed_candidate_score_history, list) and failed_candidate_score_history:
            lines.append(f"  failed_candidate_score_history={_compact_json(failed_candidate_score_history)}")

    if not lines:
        lines.append("当前 planner_input 中没有可解析的 field_atomic 历史经验。按静态技能参数说明生成保守方案。")
    return "\n".join(lines)


def _compact_json(value: Any, *, max_chars: int = 900) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _collect_failure_parameter_lessons(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    lessons: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        experience_id = str(item.get("experience_id") or "")
        action = str(item.get("action") or "")
        critic = item.get("llm_critic") if isinstance(item.get("llm_critic"), dict) else {}
        summary = critic.get("parameter_failure_summary") if isinstance(critic.get("parameter_failure_summary"), dict) else {}
        parameter_items = summary.get("items") if isinstance(summary.get("items"), list) else []
        if not parameter_items and summary:
            parameter_items = [summary]
        for index, parameter_item in enumerate(parameter_items[:4], start=1):
            if not isinstance(parameter_item, dict):
                continue
            lesson = _format_parameter_failure_lesson(
                parameter_item,
                experience_id=experience_id,
                source_action=action,
                index=index,
            )
            if lesson:
                lessons.append(lesson)
        axis_hint = _format_pregrasp_axis_error_lesson(item, experience_id=experience_id)
        if axis_hint:
            lessons.append(axis_hint)
    return lessons


def _format_parameter_failure_lesson(
    item: dict[str, Any],
    *,
    experience_id: str,
    source_action: str,
    index: int,
) -> str:
    action = str(item.get("action") or "unknown")
    bad_keys = item.get("bad_keys") if isinstance(item.get("bad_keys"), list) else []
    bad_values = item.get("bad_values") if isinstance(item.get("bad_values"), dict) else {}
    expected_direction = item.get("expected_direction") if isinstance(item.get("expected_direction"), dict) else {}
    reason = str(item.get("reason") or "")
    impact = str(item.get("impact") or "")
    parameter_lesson = str(item.get("parameter_lesson") or "")
    if not any([bad_keys, bad_values, expected_direction, reason, impact, parameter_lesson]):
        return ""
    return "\n".join([
        f"  - id={experience_id or 'unknown'} #{index}",
        f"    错误技能: {action or source_action or 'unknown'}",
        f"    错误参数: {_compact_json(bad_keys, max_chars=220)}",
        f"    错误取值: {_compact_json(bad_values, max_chars=260)}",
        f"    期望调整方向: {_compact_json(expected_direction, max_chars=260)}",
        f"    失败原因: {reason[:220]}",
        f"    失败影响: {impact[:220]}",
        f"    参数教训: {parameter_lesson[:220]}",
    ])


def _format_pregrasp_axis_error_lesson(item: dict[str, Any], *, experience_id: str) -> str:
    vector = item.get("pregrasp_error_vector_torso")
    frame = "torso"
    if not isinstance(vector, list) or len(vector) < 3:
        vector = item.get("pregrasp_error_vector_world")
        frame = "world"
    values = _numeric_vector3(vector)
    if values is None:
        return ""
    norm = _numeric_scalar(item.get("pregrasp_error_norm"))
    axes = ("x", "y", "z")
    dominant_axis, dominant_value = max(zip(axes, values), key=lambda pair: abs(pair[1]))
    if abs(dominant_value) < 0.03:
        return ""
    axis_bits = ", ".join(f"{axis}{value:+.3f}m" for axis, value in zip(axes, values))
    local_hint = _pregrasp_axis_adjustment_hint(dominant_axis, dominant_value, values)
    return "\n".join([
        f"  - id={experience_id or 'unknown'} pregrasp主轴误差",
        "    错误技能: move_base_relative / plan_cartesian_trajectory / move_to_pregrasp",
        f"    错误参数: [\"move_base_relative.x\", \"move_base_relative.y\", \"side\", \"mode\"]",
        f"    错误取值: final_tcp_minus_pregrasp_{frame}=[{axis_bits}], norm={norm if norm is not None else 'unknown'}",
        f"    期望调整方向: {local_hint}",
        "    失败原因: 预抓取失败已经有实际 TCP 偏差证据，应按主轴误差做局部修正，而不是只套用粗粒度“继续增大底盘”经验。",
        "    失败影响: 如果 y/z 已对齐但 x 仍偏差，继续同时增大 x/y 会造成重复失败或侧向过冲。",
        "    参数教训: 优先保持已对齐轴稳定，只围绕主误差轴微调底盘或换 side/mode；不要把 final_tcp 偏差误当成 pregrasp_offset 参数。",
    ])


def _pregrasp_axis_adjustment_hint(axis: str, value: float, values: list[float]) -> str:
    aligned = [
        name
        for name, component in zip(("x", "y", "z"), values)
        if name != axis and abs(component) < 0.02
    ]
    sign_text = "正向过冲" if value > 0 else "负向欠到/反向偏差"
    if axis == "x":
        direction = "move_base_relative.x 不应继续同向增大；尝试小幅回退或减小 x 步长"
    elif axis == "y":
        direction = "move_base_relative.y 不应继续同向增大；尝试小幅回退 y 或换手臂侧向策略"
    else:
        direction = "优先检查 torso 高度/轨迹 mode；不要用底盘 x/y 代替 z 方向修正"
    if aligned:
        direction += f"；保持已对齐轴 {','.join(aligned)} 基本不变"
    return f"{axis} 轴{sign_text} {value:+.3f}m，{direction}"


def _numeric_vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None


def _numeric_scalar(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_recovery_images(value: Any, *, limit: int) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key in ("scene_initial", "before_anomaly", "after_anomaly", "failed_step", "replay_state", "final_state"):
            item = value.get(key)
            if isinstance(item, str) and item:
                paths.append(item)
        for item in value.values():
            if isinstance(item, str) and item and item not in paths:
                paths.append(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                paths.append(item)
            elif isinstance(item, dict):
                path = item.get("image_path")
                if isinstance(path, str) and path:
                    paths.append(path)
    existing = [path for path in paths if Path(path).exists()]
    return existing[: max(0, int(limit))]


def _action_usage_notes(action: str) -> list[str]:
    notes: dict[str, list[str]] = {
        "move_base_relative": [
            "  使用场景：目标超出机械臂舒适工作范围时，用底盘相对移动改善可达性，参数可以使用限制范围内的所有参数",
            "  避免：只需要微调机械臂位姿时不要使用。",
        ],
        "set_torso_posture": [
            "  使用场景：需要改变相机高度或抓取高度范围时使用。",
            "  参数：level 只能为 mid 或 high；high时机器人躯干伸直，mid机器人躯干会降低，当只移动底盘无法到达有效位置的时候可以使用",
        ],
        "head_camera_rgbd_save": [
            "  使用场景：任何物体位姿估计之前都要先保存头部 RGB-D。",
        ],
        "head_camera_grounded_sam2_pose": [
            "  使用场景：估计 target_class 的最新位置，供后续技能读取。",
            "  输出连接：后续同名 target_class 技能会自动读取这个位置文件。",
        ],
        "plan_cartesian_trajectory": [
            "  使用场景：选择到预抓取点的轨迹形态。",
            "  LLM 只选择 side、target_class、mode；pregrasp_offset、topdown_mode、clearance_z、side_offset_x/y 使用执行器默认值。",
            "  如果历史失败是可达性问题，优先调整底盘/腰部，而不是修改轨迹微调参数。",
        ],
        "move_to_pregrasp": [
            "  使用场景：执行规划轨迹或直接移动到预抓取点。",
            "  LLM 只输出 side、target_class；预抓取偏移和末端朝向使用执行器默认值或前面生成的轨迹。",
        ],
        "approach_object": [
            "  使用场景：从预抓取点进入最终抓取点。",
            "  LLM 只输出 side、target_class；visual_grasp_offset_z 和末端朝向使用执行器默认值。",
        ],
        "close_gripper": [
            "  使用场景：approach_object 到达抓取点之后闭合夹爪。",
        ],
        "lift": [
            "  使用场景：close_gripper 之后提升被抓物体。",
            "  重要参数：lift_height 是实验参数，必须显式给出。",
        ],
        "lower_held_object": [
            "  使用场景：transport_to_detected_target 之后、open_gripper 之前，夹住物体向下靠近放置目标。",
            "  重要参数：lower_distance 控制下降距离，范围 0 到 0.08m。",
            "  限制：该技能只负责 TCP 下降并保持夹爪闭合，不负责判断放置是否稳定。",
        ],
        "transport_to_detected_target": [
            "  使用场景：提升成功后，把被抓物体移动到目标物体或参考物体附近。",
            "  重要参数：读取前面 target_class 感知生成的位置 JSON，可用 place_offset_x/y 在正负 0.02m 内微调放置点。",
            "  限制：不接受 place_offset_z；z 高度保持当前 TCP 高度。",
            "  失败条件：如果 target_class 位置 JSON 不存在，该技能失败且不移动。",
        ],
        "open_gripper": [
            "  使用场景：运输完成后释放物体，或任务要求失败后松开夹爪。",
        ],
    }
    return notes.get(action, [])


def _guidance_for_namespace(
    skill_namespace: str,
    skill_catalogs: dict[str, Any] | None = None,
    *,
    llm_view: bool = False,
) -> dict[str, dict[str, Any]]:
    catalogs = coerce_skill_catalogs(skill_catalogs or default_skill_catalogs())
    catalog = catalogs.get(skill_namespace)
    if catalog is None:
        return {}
    allowed = set(catalog.allowed_skills)
    source = FIELD_ATOMIC_LLM_PARAMETER_GUIDANCE if llm_view else FIELD_ATOMIC_PARAMETER_GUIDANCE
    return {name: source[name] for name in sorted(allowed) if name in source}


def _coerce_parameter(bounds: Any, value: Any) -> Any:
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, bool) for item in bounds):
        return bool(value)
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, int) and not isinstance(item, bool) for item in bounds):
        if not isinstance(value, (int, float)):
            return None
        return int(max(int(bounds[0]), min(int(value), int(bounds[1]))))
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in bounds):
        if not isinstance(value, (int, float)):
            return None
        return max(float(bounds[0]), min(float(value), float(bounds[1])))
    if isinstance(bounds, list) and len(bounds) == 3 and bounds[2] == 4:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        out = []
        for item in value:
            if not isinstance(item, (int, float)):
                return None
            out.append(float(item))
        return out
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(item, list) for item in bounds):
        if not isinstance(value, list) or len(value) != len(bounds[0]) or len(value) != len(bounds[1]):
            return None
        out = []
        for index, item in enumerate(value):
            if not isinstance(item, (int, float)):
                return None
            out.append(max(float(bounds[0][index]), min(float(item), float(bounds[1][index]))))
        return out
    return value
