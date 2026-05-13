from __future__ import annotations

import json
from typing import Any, Dict

from doubao import (
    JSON_ONLY_LINE,
    _action_prompt,
    _dump_json,
    _history_images,
    _history_records,
    _history_summary_text,
    _invoke_raw,
    _normalize_action_result,
    _parse_json_payload,
    _sanitize_image_items,
    fault_recover,
    get_sorce,
)


def recheck_action(evidence: Any, action_name: str, rule_result: Dict[str, Any], *, tier: str = "full") -> Dict[str, Any]:
    content_blocks: list[dict[str, Any]] = []
    if isinstance(evidence, dict) and "summary_text" in evidence:
        image_items = evidence.get("full_images", []) if tier == "full" else evidence.get("light_images", [])
        content_blocks.extend(_sanitize_image_items(image_items, max_images=6 if tier == "full" else 3))
        summary_text = evidence.get("summary_text")
        if summary_text:
            content_blocks.append({"type": "text", "text": str(summary_text)})

    rule_text = _dump_json(
        {
            "rule_status": rule_result.get("rule_status"),
            "reason": rule_result.get("reason", ""),
            "metrics": rule_result.get("metrics", {}),
        }
    )
    content_blocks.append(
        {
            "type": "text",
            "text": (
                "下面是固定规则检测结果，请重点复核是否真的需要进入异常处理。\n"
                f"{rule_text}\n"
                "如果规则只是处于边界条件，请根据图像和状态摘要做最终判断。"
            ),
        }
    )
    content_blocks.append({"type": "text", "text": _action_prompt(action_name)})
    content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})

    raw_text = _invoke_raw(content_blocks)
    payload = _normalize_action_result(_parse_json_payload(raw_text, prefer_array=False))
    return {
        "status": payload.get("status", "FAILURE"),
        "reason": str(payload.get("reason", "")),
        "consider": str(payload.get("consider", "")) if payload.get("consider") is not None else "",
        "raw_text": raw_text,
        "used_tier": tier,
    }


def generate_recovery(
    content_list_all: Any,
    task_list: list[Any],
    target: str | None = None,
    place_target_xy: Any = None,
) -> str:
    return fault_recover(
        content_list_all,
        task_list,
        target=target,
        place_target_xy=place_target_xy,
    )


def score_recovery(content_list_all: Any, task_list: list[Any], place_target_xy: Any = None) -> dict[str, Any]:
    return get_sorce(content_list_all, task_list, place_target_xy=place_target_xy)


def recovery_context_summary(content_list_all: Any) -> Dict[str, Any]:
    history_records = _history_records(content_list_all)
    return {
        "history_summary": _history_summary_text(history_records),
        "history_images": _history_images(history_records, tail_records=3, per_record=2, max_images=5),
    }
