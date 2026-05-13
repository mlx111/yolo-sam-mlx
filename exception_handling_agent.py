from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, PrivateAttr
from volcenginesdkarkruntime import Ark


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
LEGACY_DASHSCOPE_API_KEY = "sk-c60f21de762148a990e9ec13041b7fe1"
JSON_ONLY_LINE = "只输出一行 JSON，不要解释，不要 Markdown，不要代码块。"
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
_AGENT_CACHE: dict[str, "ExceptionHandlingAgent"] = {}


def _env_float(*names: str, default: float) -> float:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def _env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def _env_str(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _model_validate(model_cls: type[BaseModel], payload: Any) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _model_dump(instance: BaseModel) -> dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(exclude_none=True)
    return instance.dict(exclude_none=True)


def _base_message_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, AIMessage):
        return "assistant"
    return "user"


def _normalize_message_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": "text", "text": item})
                continue
            if isinstance(item, dict):
                normalized.append(item)
        return normalized
    return str(content)


def _base_message_to_payload(message: BaseMessage) -> dict[str, Any]:
    return {
        "role": _base_message_role(message),
        "content": _normalize_message_content(message.content),
    }


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


def encode_file(file_path: str | os.PathLike[str]) -> str:
    with open(file_path, "rb") as read_file:
        return base64.b64encode(read_file.read()).decode("utf-8")


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


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


class ActionReviewResult(BaseModel):
    status: Literal["SUCCESS", "FAILURE", "UNCERTAIN"]
    reason: str
    consider: str | None = None

    def to_json_text(self) -> str:
        return _json_dumps(_model_dump(self))


class RecoveryStep(BaseModel):
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return _model_dump(self)


class RecoveryScoreResult(BaseModel):
    status: Literal["success", "failure"]
    score: str
    reason: str

    def to_json_text(self) -> str:
        return _json_dumps(_model_dump(self))


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    base_url: str
    api_key: str
    client_timeout_seconds: float
    light_timeout_seconds: float
    full_timeout_seconds: float
    recovery_timeout_seconds: float
    score_timeout_seconds: float
    request_retry_count: int
    max_action_tokens: int
    max_recovery_tokens: int
    max_score_tokens: int
    max_object_tokens: int
    max_light_images: int
    max_full_images: int
    max_recovery_images: int
    max_score_images: int
    light_model: str
    full_model: str
    recovery_model: str
    score_model: str
    object_model: str


def _load_provider_settings() -> ProviderSettings:
    provider = (_env_str("EXCEPTION_LLM_PROVIDER", default="qwen") or "qwen").strip().lower()
    if provider not in {"qwen", "doubao"}:
        raise ValueError(f"Unsupported EXCEPTION_LLM_PROVIDER={provider!r}")

    if provider == "qwen":
        api_key = _env_str("EXCEPTION_LLM_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY", default=LEGACY_DASHSCOPE_API_KEY)
        base_url = _env_str("EXCEPTION_LLM_BASE_URL", "DASHSCOPE_BASE_URL", default=DEFAULT_QWEN_BASE_URL)
        light_model = _env_str("EXCEPTION_LLM_LIGHT_MODEL", "QWEN_LIGHT_MODEL", default="qwen3.5-flash") or "qwen3.5-flash"
        full_model = _env_str("EXCEPTION_LLM_FULL_MODEL", "QWEN_FULL_MODEL", default="qwen3.6-plus") or "qwen3.6-plus"
        recovery_model = _env_str("EXCEPTION_LLM_RECOVERY_MODEL", "QWEN_RECOVERY_MODEL", default="qwen3-vl-flash") or "qwen3-vl-flash"
        score_model = _env_str("EXCEPTION_LLM_SCORE_MODEL", "QWEN_SCORE_MODEL", default="qwen3-vl-flash") or "qwen3-vl-flash"
        object_model = _env_str("EXCEPTION_LLM_OBJECT_MODEL", "QWEN_OBJECT_MODEL", default=recovery_model) or recovery_model
    else:
        api_key = _env_str("EXCEPTION_LLM_API_KEY", "ARK_API_KEY")
        base_url = _env_str("EXCEPTION_LLM_BASE_URL", default=DEFAULT_DOUBAO_BASE_URL)
        light_model = _env_str("EXCEPTION_LLM_LIGHT_MODEL", default="doubao-seed-1-6-vision-250815") or "doubao-seed-1-6-vision-250815"
        full_model = _env_str("EXCEPTION_LLM_FULL_MODEL", default=light_model) or light_model
        recovery_model = _env_str("EXCEPTION_LLM_RECOVERY_MODEL", default=light_model) or light_model
        score_model = _env_str("EXCEPTION_LLM_SCORE_MODEL", default=light_model) or light_model
        object_model = _env_str("EXCEPTION_LLM_OBJECT_MODEL", default=light_model) or light_model

    if not api_key:
        raise ValueError(f"Missing API key for provider {provider}.")

    return ProviderSettings(
        provider=provider,
        base_url=base_url or (DEFAULT_QWEN_BASE_URL if provider == "qwen" else DEFAULT_DOUBAO_BASE_URL),
        api_key=api_key,
        client_timeout_seconds=_env_float("EXCEPTION_LLM_CLIENT_TIMEOUT_SECONDS", "QWEN_CLIENT_TIMEOUT_SECONDS", default=30.0),
        light_timeout_seconds=_env_float("EXCEPTION_LLM_LIGHT_TIMEOUT_SECONDS", "QWEN_LIGHT_TIMEOUT_SECONDS", default=15.0),
        full_timeout_seconds=_env_float("EXCEPTION_LLM_FULL_TIMEOUT_SECONDS", "QWEN_FULL_TIMEOUT_SECONDS", default=25.0),
        recovery_timeout_seconds=_env_float("EXCEPTION_LLM_RECOVERY_TIMEOUT_SECONDS", "QWEN_RECOVERY_TIMEOUT_SECONDS", default=20.0),
        score_timeout_seconds=_env_float("EXCEPTION_LLM_SCORE_TIMEOUT_SECONDS", "QWEN_SCORE_TIMEOUT_SECONDS", default=15.0),
        request_retry_count=max(0, _env_int("EXCEPTION_LLM_REQUEST_RETRY_COUNT", "QWEN_REQUEST_RETRY_COUNT", default=1)),
        max_action_tokens=max(32, _env_int("EXCEPTION_LLM_ACTION_MAX_TOKENS", "QWEN_ACTION_MAX_TOKENS", default=120)),
        max_recovery_tokens=max(64, _env_int("EXCEPTION_LLM_RECOVERY_MAX_TOKENS", "QWEN_RECOVERY_MAX_TOKENS", default=300)),
        max_score_tokens=max(32, _env_int("EXCEPTION_LLM_SCORE_MAX_TOKENS", "QWEN_SCORE_MAX_TOKENS", default=120)),
        max_object_tokens=max(32, _env_int("EXCEPTION_LLM_OBJECT_MAX_TOKENS", "QWEN_OBJECT_MAX_TOKENS", default=120)),
        max_light_images=max(1, _env_int("EXCEPTION_LLM_MAX_LIGHT_IMAGES", "QWEN_MAX_LIGHT_IMAGES", default=2)),
        max_full_images=max(1, _env_int("EXCEPTION_LLM_MAX_FULL_IMAGES", "QWEN_MAX_FULL_IMAGES", default=3)),
        max_recovery_images=max(1, _env_int("EXCEPTION_LLM_MAX_RECOVERY_IMAGES", "QWEN_MAX_RECOVERY_IMAGES", default=2)),
        max_score_images=max(1, _env_int("EXCEPTION_LLM_MAX_SCORE_IMAGES", "QWEN_MAX_SCORE_IMAGES", default=3)),
        light_model=light_model,
        full_model=full_model,
        recovery_model=recovery_model,
        score_model=score_model,
        object_model=object_model,
    )


class OpenAICompatibleChatModel(SimpleChatModel):
    provider_name: str
    model_name: str
    base_url: str
    api_key: str
    timeout_seconds: float = 30.0
    max_tokens: int = 256
    request_retry_count: int = 1
    temperature: float = 0.0
    _client: OpenAI | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return f"{self.provider_name}-openai-compatible"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "base_url": self.base_url,
        }

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        payload_messages = [_base_message_to_payload(message) for message in messages]
        request_payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": payload_messages,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "timeout": float(kwargs.get("timeout", self.timeout_seconds)),
        }
        if stop:
            request_payload["stop"] = stop

        last_error: Exception | None = None
        for attempt in range(self.request_retry_count + 1):
            try:
                completion = self._get_client().chat.completions.create(**request_payload)
                content = completion.choices[0].message.content
                if isinstance(content, str):
                    return content
                if content is None:
                    return ""
                return _json_dumps(content)
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
                print(
                    f"[WARN] {self.provider_name} request failed on attempt "
                    f"{attempt + 1}/{self.request_retry_count + 1}: {exc}"
                )
                continue
            except APIError as exc:
                last_error = exc
                print(f"[WARN] {self.provider_name} api error: {exc}")
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[WARN] {self.provider_name} unexpected error: {exc}")
                break
        raise RuntimeError(f"{self.provider_name} invoke failed: {last_error}")


class ArkChatModel(SimpleChatModel):
    provider_name: str
    model_name: str
    base_url: str
    api_key: str
    timeout_seconds: float = 30.0
    max_tokens: int = 256
    request_retry_count: int = 1
    temperature: float = 0.0
    _client: Ark | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return f"{self.provider_name}-ark"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "base_url": self.base_url,
        }

    def _get_client(self) -> Ark:
        if self._client is None:
            self._client = Ark(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        payload_messages = [_base_message_to_payload(message) for message in messages]
        request_payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": payload_messages,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
        }
        if stop:
            request_payload["stop"] = stop

        last_error: Exception | None = None
        for attempt in range(self.request_retry_count + 1):
            try:
                completion = self._get_client().chat.completions.create(**request_payload)
                content = completion.choices[0].message.content
                if isinstance(content, str):
                    return content
                if content is None:
                    return ""
                return _json_dumps(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(
                    f"[WARN] {self.provider_name} request failed on attempt "
                    f"{attempt + 1}/{self.request_retry_count + 1}: {exc}"
                )
                continue
        raise RuntimeError(f"{self.provider_name} invoke failed: {last_error}")


class ProviderRegistry:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings

    def create_chat_model(
        self,
        *,
        model_name: str,
        timeout_seconds: float,
        max_tokens: int,
    ) -> SimpleChatModel:
        common_kwargs = {
            "provider_name": self.settings.provider,
            "model_name": model_name,
            "base_url": self.settings.base_url,
            "api_key": self.settings.api_key,
            "timeout_seconds": timeout_seconds,
            "max_tokens": max_tokens,
            "request_retry_count": self.settings.request_retry_count,
        }
        if self.settings.provider == "doubao":
            return ArkChatModel(**common_kwargs)
        return OpenAICompatibleChatModel(**common_kwargs)


def _sanitize_image_items(image_items: Any, *, max_images: Optional[int] = None) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in image_items or []:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image_url")
        if not isinstance(image_url, dict) or "url" not in image_url:
            continue
        sanitized.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url["url"]},
            }
        )
    if max_images is not None:
        sanitized = _sample_evenly(sanitized, max_images)
    return sanitized


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
            f"reason={record.get('reason', '')} tier={record.get('used_tier', '')} "
            f"frames={summary.get('frame_count', 0)} cameras={summary.get('camera_counts', {})}"
        )
    return "\n".join(lines) if lines else "无历史动作摘要。"


def _history_images_for_recovery(history_records: list[dict[str, Any]], settings: ProviderSettings) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    initial_record = next((record for record in history_records if record.get("action") == "获取相机图像"), None)
    if initial_record:
        selected.extend(initial_record.get("history_images", [])[:1])
    for record in history_records[-2:]:
        selected.extend(record.get("history_images", [])[:2])
    return _sanitize_image_items(selected, max_images=settings.max_recovery_images)


def _history_images_for_score(history_records: list[dict[str, Any]], settings: ProviderSettings) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    initial_record = next((record for record in history_records if record.get("action") == "获取相机图像"), None)
    if initial_record:
        selected.extend(initial_record.get("history_images", [])[:1])
    for record in history_records[-3:]:
        selected.extend(record.get("history_images", [])[:2])
    return _sanitize_image_items(selected, max_images=settings.max_score_images)


def _action_prompt(action: str, tier: str) -> str:
    uncertainty_line = (
        "如果根据当前证据无法可靠判断，请输出 JSON：{'status':'UNCERTAIN','reason':'...'}。"
        if tier == "light"
        else "只有在证据仍不足以可靠判断时才允许输出 UNCERTAIN。"
    )
    prompts = {
        "移动到预抓取位置": f"""
你是一名mujoco仿真环境中的机械臂监控专家。
当前机械臂正在执行的技能是[移动到预抓取位置]。
请分析提供的 MuJoCo 仿真图像和结构化摘要,判断技能执行的是否正确。
预抓取位置距离抓取位置在z轴方向上约0.1米，移动成功后机械臂夹爪应该在物体上方。
如果机械臂夹爪没有到达物体上方，则执行失败。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"..."}}。只输出一次 json。""",
        "移动到抓取位置": f"""
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[移动到抓取位置]技能。
该机械臂由 ur5e 与黑色的 2f85 夹爪构成。
请结合图像和结构化摘要检查：夹爪两指是否到达目标物体两侧、是否碰触物体、是否具备下一步闭合抓取条件。
只评估当前这一步“移动到抓取位置”是否完成，不要因为 task_tail 中历史上的提升失败、抓取失败或其它动作失败而直接判当前动作失败。
若夹爪没有到达目标两侧，或已经碰触物体，则执行失败。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"...","consider":"..."}}。只输出一次 json。""",
        "提升物体": f"""
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[提升物体]技能。
请严格按照以下顺序分析，并在 consider 中分步写出结论：
1. 先检查结构化摘要中的 task_tail，判断最近是否出现“夹爪闭合:SUCCESS”。
2. 再检查结构化摘要中的 gripper_signal。如果 gripper_signal 为 0、0.0 或明显接近 0，说明夹爪没有闭合到可抓取状态。
3. 只有当前两项都满足时，才允许继续看图像，判断目标物体是否被夹爪稳定夹持，并且是否随夹爪一起离开桌面。
4. 最后判断提升过程中物体是否掉落，或一开始就是空抓。

强制判定规则：
- 若 task_tail 最近没有“夹爪闭合:SUCCESS”，必须返回 FAILURE，不能返回 SUCCESS 或 UNCERTAIN。
- 若 gripper_signal 为 0、0.0 或明显接近 0，必须返回 FAILURE，不能返回 SUCCESS 或 UNCERTAIN。
- 若图像中夹爪中间没有物体、物体没有被带起、或提升过程中掉落，必须返回 FAILURE。
- 只有在“前置条件满足”且“物体被稳定夹持并成功带起”时，才可以返回 SUCCESS。
- 只有在前置条件满足、且图像证据不足以判断抓持与提升结果时，才允许返回 UNCERTAIN。

输出要求：
- consider 必须按“前置条件检查 -> 抓持状态检查 -> 提升结果检查”的顺序书写。
- reason 用一句话概括最主要的失败原因，例如“缺少夹爪闭合前置条件”或“提升时物体未被夹持”。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"...","consider":"..."}}。只输出一次 json。""",
        "移动到预放置位置": f"""
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[移动到预放置位置]技能。
请结合图像和结构化摘要判断：
1. 机械臂在移动过程中是否持续抓取着物体。
2. 机械臂夹爪是否正确移动到了指定放置位置附近。
如果任一条件不满足，则执行失败。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"..."}}。只输出一次 json。""",
        "回到初始位置": f"""
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[回到初始位置]技能。
请结合图像和结构化摘要判断机械臂是否恢复到初始竖直状态，以及夹爪是否仍夹取着物体。
如果没有回到初始状态或仍夹着物体，则执行失败。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"..."}}。只输出一次 json。""",
        "夹爪闭合": f"""
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[夹爪闭合]技能。
请重点观察黑色夹爪的两指是否包围并贴合目标物体，避免空抓或夹偏。
如果夹爪闭合后中间没有物体，或未正确贴合物体，则失败。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"...","consider":"..."}}。只输出一次 json。""",
        "夹爪开启": f"""
你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[夹爪开启]技能。
请结合图像和摘要判断夹爪是否已经松开物体。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"..."}}。只输出一次 json。""",
    }
    return prompts.get(
        action,
        f"""
你是一名mujoco仿真环境中的机械臂监控专家。请根据图像和结构化摘要判断动作[{action}]是否执行正确。
{uncertainty_line}
请以 JSON 格式输出：{{"status":"SUCCESS/FAILURE/UNCERTAIN","reason":"..."}}。只输出一次 json。""",
    )


def _action_fallback(action: str, tier: str) -> ActionReviewResult:
    status = "UNCERTAIN" if tier == "light" else "FAILURE"
    return ActionReviewResult(
        status=status,
        reason=f"{action} llm request timeout or invalid response",
    )


def _recovery_fallback() -> list[RecoveryStep]:
    return []


def _score_fallback() -> RecoveryScoreResult:
    return RecoveryScoreResult(
        status="failure",
        score="0",
        reason="score timeout or invalid response",
    )


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


def _build_execute_grasp2_step(place_target_xy: Any) -> RecoveryStep | None:
    normalized = _normalize_place_target_xy(place_target_xy)
    if normalized is None:
        return None
    return RecoveryStep(action="execute-grasp2", parameters=dict(normalized))


def _completion_tail(place_target_xy: Any, *, include_place_move: bool) -> list[RecoveryStep]:
    steps: list[RecoveryStep] = []
    if include_place_move:
        place_step = _build_execute_grasp2_step(place_target_xy)
        if place_step is not None:
            steps.append(place_step)
        else:
            return steps
    steps.append(RecoveryStep(action="gripper-action", parameters={"state": 0}))
    steps.append(RecoveryStep(action="execute-init", parameters={}))
    return steps


def _vertical_grasp_recovery_fallback(place_target_xy: Any = None) -> list[RecoveryStep]:
    steps = [
        RecoveryStep(action="gripper-action", parameters={"state": 0}),
        RecoveryStep(action="move-grasp", parameters={}),
        RecoveryStep(action="gripper-action", parameters={"state": 1}),
        RecoveryStep(action="vertical-grasp", parameters={}),
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
        str(part) for part in (
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


def _is_valid_vertical_grasp_recovery_plan(steps: list[RecoveryStep]) -> bool:
    seen_vertical = False
    seen_close_before_vertical = False

    for step in steps:
        action = step.action
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

        raw_state = step.parameters.get("state", step.parameters.get("flag"))
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
    steps: list[RecoveryStep],
    *,
    history_records: list[dict[str, Any]],
    task_list: list[Any],
    place_target_xy: Any = None,
) -> list[RecoveryStep]:
    if _is_vertical_grasp_prereq_failure(history_records, task_list):
        return _vertical_grasp_recovery_fallback(place_target_xy)

    if _last_failed_action(task_list) == "提升物体" and not _is_valid_vertical_grasp_recovery_plan(steps):
        return _vertical_grasp_recovery_fallback(place_target_xy)

    normalized_target = _normalize_place_target_xy(place_target_xy)
    failed_action = _last_failed_action(task_list)
    has_place_move = any(step.action == "execute-grasp2" for step in steps)
    if normalized_target is not None:
        for step in steps:
            if step.action == "execute-grasp2":
                step.parameters = dict(normalized_target)

    if failed_action in {"提升物体", "移动到预放置位置"} and normalized_target is not None and not has_place_move:
        steps.append(RecoveryStep(action="execute-grasp2", parameters=dict(normalized_target)))
        has_place_move = True

    execute_grasp2_index = next(
        (index for index, step in enumerate(steps) if step.action == "execute-grasp2"),
        None,
    )
    if execute_grasp2_index is not None:
        tail_steps = steps[execute_grasp2_index + 1:]
        has_release_after_place = any(
            step.action == "gripper-action" and step.parameters.get("state") == 0
            for step in tail_steps
        )
        has_init_after_place = any(step.action == "execute-init" for step in tail_steps)
        if not has_release_after_place:
            steps.append(RecoveryStep(action="gripper-action", parameters={"state": 0}))
        if not has_init_after_place:
            steps.append(RecoveryStep(action="execute-init", parameters={}))
    elif failed_action == "回到初始位置" and not any(step.action == "execute-init" for step in steps):
        steps.append(RecoveryStep(action="execute-init", parameters={}))

    return steps


def _normalize_recovery_step(
    raw_step: Any,
    *,
    target: str | None = None,
    place_target_xy: Any = None,
) -> RecoveryStep | None:
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
        return RecoveryStep(action=action, parameters={"target_class": str(target_class)})

    if action == "gripper-action":
        raw_flag = raw_params.get("flag", raw_params.get("state"))
        try:
            flag = int(raw_flag)
        except (TypeError, ValueError):
            return None
        if flag not in {0, 1}:
            return None
        return RecoveryStep(action=action, parameters={"state": flag})

    if action == "execute-grasp2":
        normalized_target = _normalize_place_target_xy(
            [raw_params.get("x"), raw_params.get("y")]
            if "x" in raw_params or "y" in raw_params
            else place_target_xy
        )
        if normalized_target is None:
            return None
        return RecoveryStep(action=action, parameters=dict(normalized_target))

    return RecoveryStep(action=action, parameters={})


class ExceptionHandlingAgent:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.registry = ProviderRegistry(settings)
        self.json_parser = JsonOutputParser()
        self._chat_models: dict[tuple[str, float, int], SimpleChatModel] = {}

    def _get_chat_model(
        self,
        *,
        model_name: str,
        timeout_seconds: float,
        max_tokens: int,
    ) -> SimpleChatModel:
        cache_key = (model_name, timeout_seconds, max_tokens)
        cached = self._chat_models.get(cache_key)
        if cached is not None:
            return cached
        model = self.registry.create_chat_model(
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
        self._chat_models[cache_key] = model
        return model

    def _invoke_json(
        self,
        *,
        content_blocks: list[dict[str, Any]],
        model_name: str,
        timeout_seconds: float,
        max_tokens: int,
        prefer_array: bool,
    ) -> Any:
        chat_model = self._get_chat_model(
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
        response = chat_model.invoke(
            [HumanMessage(content=content_blocks)],
            max_tokens=max_tokens,
            timeout=timeout_seconds,
        )
        raw_text = response.content if isinstance(response.content, str) else _json_dumps(response.content)
        normalized_text = _extract_json_text(raw_text, prefer_array=prefer_array)
        return self.json_parser.parse(normalized_text)

    def review_action(self, evidence: Any, action: str, tier: str = "light") -> ActionReviewResult:
        content_list: list[dict[str, Any]] = []
        max_images = self.settings.max_full_images if tier == "full" else self.settings.max_light_images

        if isinstance(evidence, dict) and "summary_text" in evidence:
            image_items = evidence.get("full_images", []) if tier == "full" else evidence.get("light_images", [])
            sanitized_images = _sanitize_image_items(image_items, max_images=max_images)
            content_list.extend(sanitized_images)
            summary_text = evidence.get("summary_text", "")
            if summary_text:
                content_list.append({"type": "text", "text": summary_text})
            content_list.append(
                {
                    "type": "text",
                    "text": f"当前判定层级: {tier}，当前关键图数量: {len(sanitized_images)}",
                }
            )
        elif isinstance(evidence, list):
            content_list.extend(_sanitize_image_items(evidence, max_images=max_images))

        content_list.append({"type": "text", "text": _action_prompt(action, tier)})
        content_list.append({"type": "text", "text": JSON_ONLY_LINE})

        try:
            payload = self._invoke_json(
                content_blocks=content_list,
                model_name=self.settings.full_model if tier == "full" else self.settings.light_model,
                timeout_seconds=self.settings.full_timeout_seconds if tier == "full" else self.settings.light_timeout_seconds,
                max_tokens=self.settings.max_action_tokens,
                prefer_array=False,
            )
            return _model_validate(ActionReviewResult, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] review_action[{action}][{tier}] fallback due to: {exc}")
            return _action_fallback(action, tier)

    def identify_objects(self) -> str:
        try:
            content_list: list[dict[str, Any]] = []
            for flag in ("left", "right"):
                img_data = encode_file(ROOT_DIR / "scenes" / f"c{flag}001.png")
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    }
                )
            content_list.append(
                {
                    "type": "text",
                    "text": """
这里有两张图片,观察这两张图片的黑色传送带上有哪些物体并返回,注意不需返回传送带上的方块与按钮等物体,只需要黑色传送带上的物体,白色基座上的物体不需要,也不需要返回颜色
以下是输出要求：
1.使用数组格式返回
2.物体名称是英文,不要比中文输出多出来物体
""",
                }
            )
            content_list.append({"type": "text", "text": "只输出一个英文数组，不要解释，不要 Markdown。"})
            payload = self._invoke_json(
                content_blocks=content_list,
                model_name=self.settings.object_model,
                timeout_seconds=self.settings.light_timeout_seconds,
                max_tokens=self.settings.max_object_tokens,
                prefer_array=True,
            )
            if isinstance(payload, list):
                cleaned = [str(item).strip() for item in payload if str(item).strip()]
                return _json_dumps(cleaned)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] identify_objects fallback due to: {exc}")
        return "[]"

    def plan_recovery(
        self,
        content_list_all: Any,
        task_list: list[Any],
        target: str | None = None,
        place_target_xy: Any = None,
    ) -> list[RecoveryStep]:
        history_records = _history_records(content_list_all)
        if _is_vertical_grasp_prereq_failure(history_records, task_list):
            return _vertical_grasp_recovery_fallback(place_target_xy)

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
你是mujoco仿真领域专家。机械臂执行[抓取物体{target}并移动到指定位置]任务时出现了错误。
目前已执行技能及状态：{task_list}
以下是压缩后的动作历史摘要：
{_history_summary_text(history_records)}
{target_hint}

这是所有可用技能：
camera-image, detect-object, create-cloud, create-grasp, move-pregrasp, move-grasp,
vertical-grasp, gripper-action(state=0/1), execute-grasp2, execute-init

请输出应该如何使用已有技能进行恢复操作，使最后失败的技能能够正确执行，并最终完成放置和回初始位。
输出要求：
1. 使用 json 数组格式输出指令序列
2. 技能字段名为 action
3. detect-object 的参数名必须为 target_class
4. gripper-action 的参数名必须为 state，且只能是 0 或 1
5. 严禁输出 Markdown 标记
6. 如果物体位置变化明显，需要回到初始位置重新获取抓取姿势
7. 如果物体位置没有变化，优先局部回退，不要无谓重拍整套图像
8. 在准备抓取前必须保证夹爪开启
9. 拍摄图像前必须保证机械臂处于初始位置
10. 若 vertical-grasp 暴露抓空，必须先撤回到 move-grasp 附近，而不能直接空中闭合夹爪
11. 若最后失败动作为 提升物体，且失败原因是夹爪未闭合或缺少抓取前置条件，则优先输出：
    gripper-action(state=0) -> move-grasp -> gripper-action(state=1) -> vertical-grasp
12. 在重新成功执行 vertical-grasp 之前，禁止输出 execute-grasp2
13. 任务完成标准是：物体放置到指定位置，随后夹爪打开，并执行 execute-init 回到初始状态
每一个动作类似于 {{"action":"camera-image","parameters":{{}}}}
"""
        content_blocks: list[dict[str, Any]] = []
        content_blocks.extend(_history_images_for_recovery(history_records, self.settings))
        content_blocks.append({"type": "text", "text": prompt})
        content_blocks.append({"type": "text", "text": "只输出 JSON 数组，不要解释，不要 Markdown，不要代码块。"})

        try:
            payload = self._invoke_json(
                content_blocks=content_blocks,
                model_name=self.settings.recovery_model,
                timeout_seconds=self.settings.recovery_timeout_seconds,
                max_tokens=self.settings.max_recovery_tokens,
                prefer_array=True,
            )
            if not isinstance(payload, list):
                return _recovery_fallback()
            steps: list[RecoveryStep] = []
            for raw_step in payload:
                normalized = _normalize_recovery_step(raw_step, target=target, place_target_xy=place_target_xy)
                if normalized is not None:
                    steps.append(normalized)
            return _finalize_recovery_steps(
                steps,
                history_records=history_records,
                task_list=task_list,
                place_target_xy=place_target_xy,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] plan_recovery fallback due to: {exc}")
            return _finalize_recovery_steps(
                _recovery_fallback(),
                history_records=history_records,
                task_list=task_list,
                place_target_xy=place_target_xy,
            )

    def score_recovery(
        self,
        content_list_all: Any,
        task_list: list[Any],
        place_target_xy: Any = None,
    ) -> RecoveryScoreResult:
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
        content_blocks.extend(_history_images_for_score(history_records, self.settings))
        content_blocks.append({"type": "text", "text": prompt})
        content_blocks.append({"type": "text", "text": JSON_ONLY_LINE})

        try:
            payload = self._invoke_json(
                content_blocks=content_blocks,
                model_name=self.settings.score_model,
                timeout_seconds=self.settings.score_timeout_seconds,
                max_tokens=self.settings.max_score_tokens,
                prefer_array=False,
            )
            return _model_validate(RecoveryScoreResult, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] score_recovery fallback due to: {exc}")
            return _score_fallback()

    def run(self, task_kind: str, **payload: Any) -> Any:
        if task_kind == "review_action":
            return self.review_action(
                payload.get("evidence"),
                str(payload.get("action", "")),
                tier=str(payload.get("tier", "light")),
            )
        if task_kind == "plan_recovery":
            return self.plan_recovery(
                payload.get("content_list_all"),
                payload.get("task_list") or [],
                target=payload.get("target"),
                place_target_xy=payload.get("place_target_xy"),
            )
        if task_kind == "score_recovery":
            return self.score_recovery(
                payload.get("content_list_all"),
                payload.get("task_list") or [],
                place_target_xy=payload.get("place_target_xy"),
            )
        if task_kind == "identify_objects":
            return self.identify_objects()
        raise ValueError(f"Unsupported task_kind={task_kind!r}")


def get_exception_handling_agent() -> ExceptionHandlingAgent:
    settings = _load_provider_settings()
    cache_key = _json_dumps(settings.__dict__)
    cached = _AGENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    agent = ExceptionHandlingAgent(settings)
    _AGENT_CACHE.clear()
    _AGENT_CACHE[cache_key] = agent
    return agent


def gen_content(evidence: Any, action: str, tier: str = "light") -> str:
    result = get_exception_handling_agent().review_action(evidence, action, tier=tier)
    print(result.to_json_text())
    return result.to_json_text()


def get_obj() -> str:
    content = get_exception_handling_agent().identify_objects()
    print(content)
    return content


def fault_recover(
    content_list_all: Any,
    task_list: list[Any],
    target: str | None = None,
    place_target_xy: Any = None,
) -> str:
    steps = get_exception_handling_agent().plan_recovery(
        content_list_all,
        task_list,
        target=target,
        place_target_xy=place_target_xy,
    )
    content = _json_dumps([step.to_payload() for step in steps])
    print(content)
    with open("yichang.json", "w", encoding="utf-8") as file_obj:
        file_obj.write(content)
    return content


def get_sorce(content_list_all: Any, task_list: list[Any], place_target_xy: Any = None) -> dict[str, Any]:
    result = get_exception_handling_agent().score_recovery(
        content_list_all,
        task_list,
        place_target_xy=place_target_xy,
    )
    content = result.to_json_text()
    print(content)
    with open("sorce.txt", "a", encoding="utf-8") as file_obj:
        file_obj.write("\n")
        file_obj.write(content + "\n")
        file_obj.write(",".join(map(str, task_list)))
        file_obj.write("\n")
    return _model_dump(result)
