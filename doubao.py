from __future__ import annotations

import ast
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from volcenginesdkarkruntime import Ark


ROOT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "doubao-seed-1-6-vision-250815"
ALLOWED_RECOVERY_ACTIONS = {
    "camera-image",
    "detect-object",
    "create-cloud",
    "create-grasp",
    "move-pregrasp",
    "move-grasp",
    "vertical-grasp",
    "gripper-action",
    "execute-grasp2",
    "execute-init",
}
JSON_ONLY_LINE = "只输出一行 JSON，不要解释，不要 Markdown，不要代码块。"

client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("ARK_API_KEY"),
)


def encode_file(file_path: str | os.PathLike[str]) -> str:
    with open(file_path, "rb") as read_file:
        return base64.b64encode(read_file.read()).decode("utf-8")


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _sample_evenly(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    selected: list[Any] = []
    seen: set[int] = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(items[idx])
    return selected


def _sanitize_image_items(image_items: Any, *, max_images: int | None = None) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in image_items or []:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image_url")
        if not isinstance(image_url, dict):
            continue
        url = image_url.get("url")
        if not isinstance(url, str) or not url:
            continue
        sanitized.append({"type": "image_url", "image_url": {"url": url}})
    if max_images is not None:
        sanitized = _sample_evenly(sanitized, max_images)
    return sanitized


def _extract_json_text(raw_text: str, *, prefer_array: bool) -> str:
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, raw_text)
    if match:
        return match.group(1).strip()
    if prefer_array:
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start != -1 and end != -1:
            return raw_text[start:end + 1].strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        return raw_text[start:end + 1].strip()
    return raw_text.strip()


def _parse_json_payload(raw_text: str, *, prefer_array: bool) -> Any:
    candidate = _extract_json_text(raw_text, prefer_array=prefer_array)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        payload = ast.literal_eval(candidate)
        if isinstance(payload, (dict, list)):
            return payload
        raise ValueError(f"unexpected payload type: {type(payload)!r}")


def _invoke_raw(content_blocks: list[dict[str, Any]]) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content_blocks}],
    )
    return completion.choices[0].message.content


def _history_records(content_list_all: Any) -> list[dict[str, Any]]:
    return [
        item for item in (content_list_all or [])
        if isinstance(item, dict) and item.get("record_type") == "action_evidence"
    ]


def _history_summary_text(history_records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, record in enumerate(history_records, start=1):
        summary = record.get("summary", {})
        lines.append(
            f"{idx}. action={record.get('action')} status={record.get('status')} "
            f"reason={record.get('reason', '')} frames={summary.get('frame_count', 0)} "
            f"task_tail={summary.get('task_tail', [])} gripper_signal={summary.get('gripper_signal', '')}"
        )
    return "\n".join(lines) if lines else "无历史动作摘要。"


def _history_images(history_records: list[dict[str, Any]], *, tail_records: int, per_record: int, max_images: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    initial_record = next((record for record in history_records if record.get("action") == "获取相机图像"), None)
    if initial_record:
        selected.extend(initial_record.get("history_images", [])[:1])
    for record in history_records[-tail_records:]:
        selected.extend(record.get("history_images", [])[:per_record])
    return _sanitize_image_items(selected, max_images=max_images)


def _action_prompt(action: str) -> str:
    prompts = {
        "移动到预抓取位置": """
你是一名mujoco仿真环境中的机械臂监控专家。
当前机械臂正在执行的技能是[移动到预抓取位置]。
请分析提供的 MuJoCo 仿真图像，判断技能执行是否正确。
技能执行正确的条件是机械臂夹爪到达待抓取物体附近，若没有则执行错误。
请以 JSON 格式输出：{"status":"SUCCESS/FAILURE","reason":"..."}。
""",
        "移动到抓取位置": """
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[移动到抓取位置]技能。
该机械臂由 ur5e 机械臂与黑色的 2f85 夹爪构成。
请按照以下步骤分析，并在 consider 中输出结论：
1. 观察物体中心位置。
2. 观察黑色夹爪左右两个指尖位置。
3. 判断两个指尖形成的闭合空间是否覆盖物体中心点。
4. 判断夹爪两指是否位于物体两侧并具备下一步抓取条件。
5. 只评估当前这一步“移动到抓取位置”是否完成，不要因为历史上的提升失败、抓取失败或其它动作失败而直接判当前动作失败。
6. 若夹爪没有到达物体两侧，或已经碰触物体，则执行失败。
请以 JSON 格式输出：{"status":"SUCCESS/FAILURE","reason":"...","consider":"..."}。
""",
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
        "移动到预放置位置": """
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[移动到预放置位置]技能。
这里的“移动到预放置位置”表示将物体移动到指定放置位置，
正确执行的标准是机械臂持续抓取物体，并将物体稳定移动到指定目标位置上方或附近。
如果没有抓取物体就移动，或移动过程中物体掉落，或明显没有到达目标放置区域，则失败。
请以 JSON 格式输出：{"status":"SUCCESS/FAILURE","reason":"..."}。
""",
        "回到初始位置": """
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[回到初始位置]技能。
正确执行的标准是机械臂恢复到初始竖直状态，且夹爪不再夹取着物体。
请以 JSON 格式输出：{"status":"SUCCESS/FAILURE","reason":"..."}。
""",
        "夹爪闭合": """
你是一名专业的mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[夹爪闭合]技能。
请分析：
1. 夹爪两指是否正确包围物体。
2. 是否存在空抓。
3. 夹爪是否正确贴合物体。
请以 JSON 格式输出：{"status":"SUCCESS/FAILURE","reason":"...","consider":"..."}。
""",
        "夹爪开启": """
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[夹爪开启]技能。
请判断夹爪是否已经松开物体。
请以 JSON 格式输出：{"status":"SUCCESS/FAILURE","reason":"..."}。
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


def gen_content(evidence: Any, action: str, tier: str = "light") -> str:
    content_blocks: list[dict[str, Any]] = []
    if isinstance(evidence, dict) and "summary_text" in evidence:
        image_items = evidence.get("full_images", []) if tier == "full" else evidence.get("light_images", [])
        content_blocks.extend(_sanitize_image_items(image_items, max_images=6 if tier == "full" else 3))
        summary_text = evidence.get("summary_text")
        if summary_text:
            content_blocks.append({"type": "text", "text": str(summary_text)})
    elif isinstance(evidence, list):
        content_blocks.extend(_sanitize_image_items(evidence, max_images=6 if tier == "full" else 3))
    content_blocks.append({"type": "text", "text": _action_prompt(action)})
    content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})

    raw_text = _invoke_raw(content_blocks)
    payload = _normalize_action_result(_parse_json_payload(raw_text, prefer_array=False))
    content = _dump_json(payload)
    print(content)
    return content


def get_obj() -> str:
    content_blocks: list[dict[str, Any]] = []
    for flag in ("left", "right"):
        img_data = encode_file(ROOT_DIR / "scenes" / f"c{flag}001.png")
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data}"},
            }
        )
    content_blocks.append(
        {
            "type": "text",
            "text": """
这里有两张图片，观察这两张图片的黑色传送带上有哪些物体并返回。
不要返回传送带上的方块与按钮等物体，只需要黑色传送带上的物体。
白色基座上的物体不需要，也不需要返回颜色。
输出要求：
1. 使用数组格式返回
2. 物体名称使用英文
""",
        }
    )
    content_blocks.append({"type": "text", "text": "只输出一个英文 JSON 数组，不要解释，不要 Markdown。"})
    raw_text = _invoke_raw(content_blocks)
    payload = _parse_json_payload(raw_text, prefer_array=True)
    if not isinstance(payload, list):
        payload = []
    content = _dump_json([str(item).strip() for item in payload if str(item).strip()])
    print(content)
    return content


def _last_failed_action(task_list: list[Any]) -> str | None:
    for item in reversed(task_list):
        text = str(item)
        if ":FAILURE" not in text:
            continue
        return text.split(":FAILURE", 1)[0].strip() or None
    return None


def _normalize_place_target_xy(place_target_xy: Any) -> dict[str, float] | None:
    if not isinstance(place_target_xy, (list, tuple)) or len(place_target_xy) != 2:
        return None
    try:
        return {"x": float(place_target_xy[0]), "y": float(place_target_xy[1])}
    except (TypeError, ValueError):
        return None


def _build_execute_grasp2_step(place_target_xy: Any) -> dict[str, Any] | None:
    normalized = _normalize_place_target_xy(place_target_xy)
    if normalized is None:
        return None
    return {"action": "execute-grasp2", "parameters": dict(normalized)}


def _completion_tail(place_target_xy: Any, *, include_place_move: bool) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if include_place_move:
        place_step = _build_execute_grasp2_step(place_target_xy)
        if place_step is not None:
            steps.append(place_step)
        else:
            return steps
    steps.append({"action": "gripper-action", "parameters": {"state": 0}})
    steps.append({"action": "execute-init", "parameters": {}})
    return steps


def _vertical_grasp_recovery_fallback(place_target_xy: Any = None) -> list[dict[str, Any]]:
    steps = [
        {"action": "gripper-action", "parameters": {"state": 0}},
        {"action": "move-grasp", "parameters": {}},
        {"action": "gripper-action", "parameters": {"state": 1}},
        {"action": "vertical-grasp", "parameters": {}},
    ]
    steps.extend(_completion_tail(place_target_xy, include_place_move=True))
    return steps


def _is_vertical_grasp_prereq_failure(history_records: list[dict[str, Any]], task_list: list[Any]) -> bool:
    if _last_failed_action(task_list) != "提升物体":
        return False

    last_record = history_records[-1] if history_records else {}
    if last_record.get("action") != "提升物体":
        return False

    summary = last_record.get("summary", {})
    gripper_signal = summary.get("gripper_signal")
    if gripper_signal in {0, 0.0, "0", "0.0"}:
        return True

    reason_text = " ".join(
        str(part)
        for part in (
            last_record.get("reason", ""),
            last_record.get("message", ""),
            task_list[-1] if task_list else "",
        )
        if part
    )
    failure_hints = (
        "未闭合夹爪",
        "夹爪未闭合",
        "gripper signal=0.0",
        "gripper_signal is 0.0",
        "missing prerequisite",
        "缺少前置条件",
    )
    return any(hint in reason_text for hint in failure_hints)


def _normalize_recovery_step(
    raw_step: Any,
    *,
    target: str | None = None,
    place_target_xy: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(raw_step, dict):
        return None
    action = str(raw_step.get("action", "")).strip()
    if action not in ALLOWED_RECOVERY_ACTIONS:
        return None

    raw_params = raw_step.get("parameters", {})
    if not isinstance(raw_params, dict):
        raw_params = {}

    if action == "detect-object":
        target_class = raw_params.get("target_class") or raw_params.get("object") or raw_params.get("target") or target
        if not target_class:
            return None
        return {"action": action, "parameters": {"target_class": str(target_class)}}

    if action == "gripper-action":
        raw_state = raw_params.get("state", raw_params.get("flag"))
        try:
            state = int(raw_state)
        except (TypeError, ValueError):
            return None
        if state not in {0, 1}:
            return None
        return {"action": action, "parameters": {"state": state}}

    if action == "execute-grasp2":
        normalized_target = _normalize_place_target_xy(
            [raw_params.get("x"), raw_params.get("y")]
            if "x" in raw_params or "y" in raw_params
            else place_target_xy
        )
        if normalized_target is None:
            return None
        return {"action": action, "parameters": dict(normalized_target)}

    return {"action": action, "parameters": {}}


def _is_valid_vertical_grasp_recovery_plan(steps: list[dict[str, Any]]) -> bool:
    seen_vertical = False
    seen_close_before_vertical = False

    for step in steps:
        action = step.get("action")
        if action == "vertical-grasp":
            seen_vertical = True
            continue
        if seen_vertical:
            continue
        if action == "execute-grasp2":
            return False
        if action != "gripper-action":
            if action == "move-grasp" and seen_close_before_vertical:
                return False
            continue

        raw_state = step.get("parameters", {}).get("state")
        try:
            state = int(raw_state)
        except (TypeError, ValueError):
            return False
        if state == 1:
            seen_close_before_vertical = True
            continue
        if state == 0 and seen_close_before_vertical:
            return False

    return True


def _finalize_recovery_steps(
    steps: list[dict[str, Any]],
    *,
    history_records: list[dict[str, Any]],
    task_list: list[Any],
    place_target_xy: Any = None,
) -> list[dict[str, Any]]:
    if _is_vertical_grasp_prereq_failure(history_records, task_list):
        return _vertical_grasp_recovery_fallback(place_target_xy)
    if _last_failed_action(task_list) == "提升物体" and not _is_valid_vertical_grasp_recovery_plan(steps):
        return _vertical_grasp_recovery_fallback(place_target_xy)

    normalized_target = _normalize_place_target_xy(place_target_xy)
    failed_action = _last_failed_action(task_list)
    has_place_move = any(step.get("action") == "execute-grasp2" for step in steps)
    if normalized_target is not None:
        for step in steps:
            if step.get("action") == "execute-grasp2":
                step["parameters"] = dict(normalized_target)

    if failed_action in {"提升物体", "移动到预放置位置"} and normalized_target is not None and not has_place_move:
        steps.append({"action": "execute-grasp2", "parameters": dict(normalized_target)})
        has_place_move = True

    execute_grasp2_index = next(
        (index for index, step in enumerate(steps) if step.get("action") == "execute-grasp2"),
        None,
    )
    if execute_grasp2_index is not None:
        tail_steps = steps[execute_grasp2_index + 1:]
        has_release_after_place = any(
            step.get("action") == "gripper-action" and step.get("parameters", {}).get("state") == 0
            for step in tail_steps
        )
        has_init_after_place = any(step.get("action") == "execute-init" for step in tail_steps)
        if not has_release_after_place:
            steps.append({"action": "gripper-action", "parameters": {"state": 0}})
        if not has_init_after_place:
            steps.append({"action": "execute-init", "parameters": {}})
    elif failed_action == "回到初始位置" and not any(step.get("action") == "execute-init" for step in steps):
        steps.append({"action": "execute-init", "parameters": {}})
    return steps


def fault_recover(
    content_list_all: Any,
    task_list: list[Any],
    target: str | None = None,
    place_target_xy: Any = None,
) -> str:
    history_records = _history_records(content_list_all)
    if _is_vertical_grasp_prereq_failure(history_records, task_list):
        content = _dump_json(_vertical_grasp_recovery_fallback(place_target_xy))
        print(content)
        with open("yichang.json", "w", encoding="utf-8") as file_obj:
            file_obj.write(content)
        return content

    target_hint = ""
    normalized_place_target = _normalize_place_target_xy(place_target_xy)
    if normalized_place_target is not None:
        target_hint = (
            f"\n当前指定放置坐标为 x={normalized_place_target['x']:.4f}, "
            f"y={normalized_place_target['y']:.4f}。"
            "\n若恢复计划包含 execute-grasp2，必须使用这组 x/y，"
            "并在成功放置后继续执行 gripper-action(state=0) 和 execute-init。"
        )

    prompt = f"""
你是mujoco仿真领域专家。机械臂在执行[抓取物体{target}并移动到指定位置]任务时出现了错误。
这是目前已经执行的技能及其状态：{task_list}
以下是压缩后的动作历史摘要：
{_history_summary_text(history_records)}
{target_hint}

这是所有可用技能：
camera-image, detect-object, create-cloud, create-grasp, move-grasp,
vertical-grasp, gripper-action(state=0/1), execute-grasp2, execute-init

请输出如何使用已有技能进行恢复，使最后失败的技能能够正确执行，并最终把物体放置到指定位置，然后让机械臂回到初始状态。
输出要求：
1. 使用 JSON 数组格式输出指令序列
2. 技能字段名为 action
3. detect-object 的参数名必须为 target_class
4. gripper-action 的参数名必须为 state，且只能是 0 或 1
5. 不要输出 Markdown，只输出 JSON
6. vertical-grasp 失败时必须先回到 move-grasp 附近，禁止在空中直接闭合夹爪
7. 如果物体位置变化明显，需要回到初始位置重新获取抓取姿势
8. 在准备抓取物体前必须确保夹爪开启
9. 拍摄图像前必须保证机械臂处于初始位置
10. 在重新成功执行 vertical-grasp 之前，禁止输出 execute-grasp2
11. 任务完成标准是：物体放置到指定位置，随后夹爪打开，并执行 execute-init 回到初始状态
"""

    content_blocks: list[dict[str, Any]] = []
    content_blocks.extend(_history_images(history_records, tail_records=2, per_record=2, max_images=4))
    content_blocks.append({"type": "text", "text": prompt})
    content_blocks.append({"type": "text", "text": "只输出 JSON 数组，不要解释，不要代码块。"})

    raw_text = _invoke_raw(content_blocks)
    payload = _parse_json_payload(raw_text, prefer_array=True)
    steps: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for raw_step in payload:
            normalized = _normalize_recovery_step(raw_step, target=target, place_target_xy=place_target_xy)
            if normalized is not None:
                steps.append(normalized)
    steps = _finalize_recovery_steps(
        steps,
        history_records=history_records,
        task_list=task_list,
        place_target_xy=place_target_xy,
    )
    content = _dump_json(steps)
    print(content)
    with open("yichang.json", "w", encoding="utf-8") as file_obj:
        file_obj.write(content)
    return content


def get_sorce(content_list_all: Any, task_list: list[Any], place_target_xy: Any = None) -> dict[str, Any]:
    history_records = _history_records(content_list_all)
    target_hint = ""
    normalized_place_target = _normalize_place_target_xy(place_target_xy)
    if normalized_place_target is not None:
        target_hint = (
            f"\n指定放置坐标：x={normalized_place_target['x']:.4f}, "
            f"y={normalized_place_target['y']:.4f}。"
        )
    prompt = f"""
你是一名mujoco仿真专家，负责评估机械臂异常处理后的最终效果。
当前任务是将目标物体抓取后放置到指定位置，并让机械臂回到初始状态。
技能流程及状态如下：{task_list}
以下是压缩后的动作历史摘要：
{_history_summary_text(history_records)}
{target_hint}

请根据最终图像证据和技能流程判断异常处理是否成功，并给出 0-10 分评分。
标准：
1. 成功标准：目标物体被放置到指定位置附近，且机械臂最终执行 execute-init 回到初始状态。
2. 如果异常处理失败，status 返回 failure；成功则返回 success。
3. 越早发现异常并处理，分数越高。
4. task_list 越长，分数越低。
请以 JSON 格式输出：{{"status":"success/failure","score":"...","reason":"..."}}。
"""

    content_blocks: list[dict[str, Any]] = []
    content_blocks.extend(_history_images(history_records, tail_records=3, per_record=2, max_images=5))
    content_blocks.append({"type": "text", "text": prompt})
    content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})

    raw_text = _invoke_raw(content_blocks)
    payload = _parse_json_payload(raw_text, prefer_array=False)
    if not isinstance(payload, dict):
        payload = {"status": "failure", "score": "0", "reason": "score response is not a JSON object"}

    status = str(payload.get("status", "failure")).strip().lower()
    if status not in {"success", "failure"}:
        status = "failure"
    result = {
        "status": status,
        "score": str(payload.get("score", "0")),
        "reason": str(payload.get("reason", "")),
    }
    content = _dump_json(result)
    print(content)
    with open("sorce.txt", "a", encoding="utf-8") as file_obj:
        file_obj.write("\n")
        file_obj.write(content + "\n")
        file_obj.write(",".join(map(str, task_list)))
        file_obj.write("\n")
    return result


__all__ = [
    "encode_file",
    "fault_recover",
    "gen_content",
    "get_obj",
    "get_sorce",
]
