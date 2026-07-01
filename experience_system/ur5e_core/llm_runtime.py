"""UR5e LLM runtime helpers shared by planner and critic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experience_core.llm_provider import (
    build_image_block as _provider_build_image_block,
    invoke_multimodal_llm,
    load_experience_env,
)


def bridge_ur5e_legacy_env() -> None:
    """Map wrapper-era env names onto the shared experience LLM provider."""

    load_experience_env()
    env_pairs = (
        ("EXPERIENCE_LLM_API_KEY", ("ARK_API_KEY", "DOUBAO_API_KEY", "EXPERIMENT_LLM_API_KEY")),
        ("EXPERIENCE_LLM_BASE_URL", ("ARK_BASE_URL", "DOUBAO_BASE_URL", "EXPERIMENT_LLM_BASE_URL")),
        ("EXPERIENCE_LLM_MODEL", ("EXPERIMENT_LLM_MODEL", "DOUBAO_MODEL_NAME")),
    )
    for target_name, source_names in env_pairs:
        if os.getenv(target_name):
            continue
        for source_name in source_names:
            value = os.getenv(source_name)
            if value and value.strip():
                os.environ[target_name] = value.strip()
                break


def env_str(*names: str, default: str | None = None) -> str | None:
    bridge_ur5e_legacy_env()
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return default


def env_bool(*names: str, default: bool = False) -> bool:
    value = env_str(*names)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_float(*names: str, default: float = 0.2) -> float:
    value = env_str(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_ur5e_provider() -> str:
    return env_str("EXPERIMENT_LLM_PROVIDER", "EXPERIENCE_LLM_PROVIDER", default="doubao") or "doubao"


def resolve_ur5e_recovery_model() -> str:
    return env_str(
        "EXPERIMENT_LLM_RECOVERY_MODEL",
        "EXPERIMENT_LLM_MODEL",
        "EXPERIENCE_LLM_MODEL",
        "DOUBAO_RECOVERY_MODEL",
        "DOUBAO_MODEL_NAME",
        default="",
    ) or ""


def resolve_ur5e_critic_model() -> str:
    return env_str(
        "EXPERIMENT_LLM_CRITIC_MODEL",
        "EXPERIMENT_LLM_RECOVERY_MODEL",
        "EXPERIMENT_LLM_MODEL",
        "EXPERIENCE_LLM_MODEL",
        "DOUBAO_RECOVERY_MODEL",
        "DOUBAO_MODEL_NAME",
        default="",
    ) or ""


def build_image_block(path: str | Path) -> dict[str, Any]:
    return _provider_build_image_block(path)


def invoke_ur5e_multimodal(
    content_blocks: list[dict[str, Any]],
    *,
    model: str = "",
    system_prompt: str = "",
    temperature: float | None = None,
) -> str:
    return invoke_multimodal_llm(
        content_blocks,
        provider=resolve_ur5e_provider(),
        model=model or resolve_ur5e_recovery_model(),
        system_prompt=system_prompt,
        temperature=env_float("EXPERIMENT_LLM_TEMPERATURE", default=0.2) if temperature is None else temperature,
    )
