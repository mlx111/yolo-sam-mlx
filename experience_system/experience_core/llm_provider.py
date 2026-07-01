"""Local LLM provider for experience-system generation tasks."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request


JSON_ONLY_LINE = "Return only valid JSON. No Markdown, no code fences, no commentary."
_ENV_LOADED = False


def _experience_system_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _unquote_env_value(value)


def load_experience_env() -> None:
    """Load experience-system .env without overriding shell env."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _load_env_file(_experience_system_root() / ".env")
    _ENV_LOADED = True


def _env_str(*names: str, default: str | None = None) -> str | None:
    load_experience_env()
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


def provider_config(provider: str) -> dict[str, str]:
    provider = (provider or "doubao").strip().lower()
    default_base_url = "https://api.openai.com/v1" if provider == "openai" else "https://ark.cn-beijing.volces.com/api/v3"
    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": _env_str("EXPERIENCE_LLM_API_KEY", default="") or "",
            "base_url": _env_str("EXPERIENCE_LLM_BASE_URL", default=default_base_url) or default_base_url,
            "model": _env_str("EXPERIENCE_LLM_MODEL", default="") or "",
        }
    return {
        "provider": "doubao",
        "api_key": _env_str("EXPERIENCE_LLM_API_KEY", default="") or "",
        "base_url": _env_str("EXPERIENCE_LLM_BASE_URL", default=default_base_url) or default_base_url,
        "model": _env_str("EXPERIENCE_LLM_MODEL", default="") or "",
    }


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_json_payload(text: str, *, prefer_array: bool = False) -> Any:
    candidate_texts = []
    stripped = (text or "").strip()
    if stripped:
        candidate_texts.append(stripped)
        extracted = _extract_json_block(stripped)
        if extracted != stripped:
            candidate_texts.append(extracted)
    for candidate in candidate_texts:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if prefer_array and isinstance(payload, dict):
            for key in ("steps", "actions", "lessons"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return payload
    raise ValueError(f"Unable to parse JSON payload: {str(text)[:500]}")


def _request_payload(messages: list[dict[str, Any]], *, model: str, temperature: float = 0.2) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }


def encode_image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with image_path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def build_image_block(path: str | Path) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": encode_image_data_url(path)},
    }


def _chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def invoke_llm(
    prompt: str,
    *,
    provider: str = "doubao",
    model: str = "",
    system_prompt: str = "",
    temperature: float = 0.2,
) -> str:
    config = provider_config(provider)
    api_key = config["api_key"]
    if not api_key:
        raise RuntimeError(f"Missing API key for provider {config['provider']}")
    final_model = model or config["model"]
    if not final_model:
        raise RuntimeError(f"Missing model for provider {config['provider']}")

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = _request_payload(messages, model=final_model, temperature=temperature)
    req = request.Request(
        _chat_completions_url(config["base_url"]),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with request.urlopen(req, timeout=int(_env_str("EXPERIENCE_LLM_TIMEOUT", default="120") or "120")) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        response = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"LLM response was not valid JSON: {body[:500]}") from exc

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response missing choices: {response}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM response missing message content: {response}")
    return content


def invoke_multimodal_llm(
    content_blocks: list[dict[str, Any]],
    *,
    provider: str = "doubao",
    model: str = "",
    system_prompt: str = "",
    temperature: float = 0.2,
) -> str:
    config = provider_config(provider)
    api_key = config["api_key"]
    if not api_key:
        raise RuntimeError(f"Missing API key for provider {config['provider']}")
    final_model = model or config["model"]
    if not final_model:
        raise RuntimeError(f"Missing model for provider {config['provider']}")

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_blocks})

    payload = _request_payload(messages, model=final_model, temperature=temperature)
    req = request.Request(
        _chat_completions_url(config["base_url"]),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with request.urlopen(req, timeout=int(_env_str("EXPERIENCE_LLM_TIMEOUT", default="120") or "120")) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        response = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"LLM response was not valid JSON: {body[:500]}") from exc

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response missing choices: {response}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM response missing message content: {response}")
    return content
