"""UR5e anomaly verification and final-effect scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_core.llm_provider import parse_json_payload

from .llm_runtime import build_image_block, env_str, invoke_ur5e_multimodal


JSON_ONLY_LINE = "只输出一行 JSON，不要解释，不要 Markdown，不要代码块。"


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _verify_model() -> str:
    return env_str(
        "EXPERIMENT_LLM_VERIFY_MODEL",
        "EXPERIMENT_LLM_MODEL",
        "EXPERIENCE_LLM_MODEL",
        "DOUBAO_VERIFY_MODEL",
        "DOUBAO_MODEL_NAME",
        default="",
    ) or ""


def _score_model() -> str:
    return env_str(
        "EXPERIMENT_LLM_SCORE_MODEL",
        "EXPERIMENT_LLM_MODEL",
        "EXPERIENCE_LLM_MODEL",
        "DOUBAO_SCORE_MODEL",
        "DOUBAO_MODEL_NAME",
        default="",
    ) or ""


def _action_prompt(action: str) -> str:
    prompts = {
        "提升物体": """
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[提升物体]技能。
请严格按照以下顺序分析，并在 consider 中分步写出结论：
1. 先检查结构化摘要中的 task_tail，判断最近是否出现“夹爪闭合:SUCCESS”。
2. 再检查结构化摘要中的 gripper_signal。如果 gripper_signal 为 0、0.0 或明显接近 0，说明夹爪没有闭合到可抓取状态。
3. 只有当前两项都满足时，才允许继续看图像，判断目标物体是否被夹爪稳定夹持，并且是否随夹爪一起离开桌面。
4. 最后判断提升过程中物体是否掉落，或一开始就是空抓。

强制判定规则：
- 若 task_tail 最近没有“夹爪闭合:SUCCESS”，必须返回 FAILURE。
- 若 gripper_signal 为 0、0.0 或明显接近 0，不得返回 SUCCESS，必须返回 FAILURE。
- 若图像中夹爪中间没有物体、物体没有被带起、或提升过程中掉落，必须返回 FAILURE。
- 只有在“前置条件满足”且“物体被稳定夹持并成功带起”时，才可以返回 SUCCESS。

输出要求：
- consider 必须按“前置条件检查 -> 抓持状态检查 -> 提升结果检查”的顺序书写。
- reason 用一句话概括最主要的失败原因，例如“缺少夹爪闭合前置条件”或“提升时物体未被夹持”。
请以 JSON 格式输出：{"status":"SUCCESS/FAILURE","reason":"...","consider":"..."}。
""",
    }
    return prompts.get(
        action,
        f"""
你是一名mujoco仿真环境中的机械臂监控专家。请根据图像和结构化摘要判断动作[{action}]是否执行正确。
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE","reason":"..."}}。
""",
    )


def _normalize_action_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "FAILURE", "reason": "模型返回不是 JSON 对象"}

    raw_status = str(payload.get("status", "FAILURE")).strip().upper()
    if raw_status in {"SUCCESS", "FAILURE", "UNCERTAIN"}:
        status = raw_status
    elif raw_status == "FINISHED":
        status = "SUCCESS"
    else:
        status = "FAILURE"

    result = {
        "status": status,
        "reason": str(payload.get("reason", "")),
    }
    if "consider" in payload and payload.get("consider") is not None:
        result["consider"] = str(payload.get("consider"))
    return result


def verify_anomaly(
    image_before: Path | str,
    image_after: Path | str,
    rule_z_change: float,
    rule_contact: dict[str, Any] | None = None,
    perceived_z_before: float | None = None,
    perceived_z_after: float | None = None,
) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = [
        build_image_block(image_before),
        build_image_block(image_after),
    ]

    context_parts: list[str] = []
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

    raw_text = invoke_ur5e_multimodal(content_blocks, model=_verify_model())
    return _normalize_action_result(parse_json_payload(raw_text, prefer_array=False))


def score_recovery(
    task_history: list[dict[str, Any]],
    image_paths: list[Path | str],
) -> dict[str, Any]:
    task_list = [f"{h['action']}:{h['status']}" for h in task_history]
    history_text = "\n".join(
        f"{index}. action={item.get('action')} status={item.get('status')}"
        for index, item in enumerate(task_history, start=1)
    )

    prompt = f"""
你是mujoco仿真专家，评估机械臂异常处理后的最终效果。
技能流程及状态：{_dump_json(task_list)}
动作历史摘要：
{history_text}

根据图像和技能流程判断异常处理是否成功，给出 0-10 分。
标准：
1. 物体被放置到指定位置，且机械臂回到安全/结束状态 → 成功
2. 越早发现异常并处理，分数越高
3. task_list 越长，分数越低
JSON 格式输出：{{"status":"success/failure","score":"...","reason":"..."}}
"""

    content_blocks: list[dict[str, Any]] = []
    for image_path in image_paths[:3]:
        content_blocks.append(build_image_block(image_path))
    content_blocks.append({"type": "text", "text": prompt})
    content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})

    raw_text = invoke_ur5e_multimodal(content_blocks, model=_score_model())
    payload = parse_json_payload(raw_text, prefer_array=True)
    if not isinstance(payload, dict):
        return {"status": "failure", "score": 0, "reason": "score response is not a JSON object"}
    return {
        "status": payload.get("status", "failure"),
        "score": payload.get("score", 0),
        "reason": payload.get("reason", ""),
    }
