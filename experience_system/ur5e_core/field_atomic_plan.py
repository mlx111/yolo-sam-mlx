"""UR5e field-atomic planning guidance rendered from runtime skill metadata."""

from __future__ import annotations

import json
from typing import Any

from experience_core.llm_provider import JSON_ONLY_LINE

from . import runtime_skills


def ur5e_field_atomic_guidance(*, llm_view: bool = False) -> dict[str, dict[str, Any]]:
    del llm_view
    return {
        action: runtime_skills.parameter_schema(action)
        for action in sorted(runtime_skills.allowed_actions())
    }


UR5E_FIELD_ATOMIC_PARAMETER_GUIDANCE = ur5e_field_atomic_guidance()
UR5E_FIELD_ATOMIC_LLM_PARAMETER_GUIDANCE = UR5E_FIELD_ATOMIC_PARAMETER_GUIDANCE
UR5E_FIELD_ATOMIC_PARAMETER_NOTES: dict[str, dict[str, str]] = {}


def ur5e_allowed_parameter_keys(action: str, *, llm_view: bool = True) -> set[str]:
    del llm_view
    return runtime_skills.parameter_keys(str(action))


def render_ur5e_field_atomic_skill_specs(
    guidance: dict[str, dict[str, Any]] | None = None,
) -> str:
    if guidance is None:
        return runtime_skills.render_skill_specs()
    lines: list[str] = []
    for action in sorted(guidance):
        lines.append(f"- {runtime_skills.skill_signature(action)}")
        lines.append(f"  description: {runtime_skills.skill_description(action)}")
        lines.append(f"  parameters: {json.dumps(guidance.get(action, {}), ensure_ascii=False)}")
    return "\n".join(lines)


def build_ur5e_field_atomic_recovery_prompt(
    *,
    goal: str,
    recovery_context: dict[str, Any] | None = None,
    memory_context: Any = None,
    max_steps: int = 8,
) -> str:
    recovery_context = recovery_context or {}
    return f"""
你是 UR5e MuJoCo 实验中的异常恢复规划助手。请根据现场状态、执行历史和经验库内容，输出可以直接执行的 UR5e 技能序列。

目标：
{goal}

当前只能使用以下技能名：
{json.dumps(sorted(runtime_skills.allowed_actions()), ensure_ascii=False, indent=2)}

技能参数说明：
{render_ur5e_field_atomic_skill_specs()}

现场上下文：
{json.dumps(recovery_context, ensure_ascii=False, indent=2)}

经验库上下文：
{json.dumps(memory_context, ensure_ascii=False, indent=2)}

输出要求：
- 只返回 JSON。
- 可以返回 JSON 数组，也可以返回顶层包含 steps 的 JSON 对象。
- 每个 step 只能包含 action 和 parameters 字段。
- steps 必须包含 1 到 {max_steps} 个技能。
- action 必须逐字匹配允许技能名。
- parameters 只能包含该技能允许的参数。
- 不需要大模型输出的内部控制参数禁止出现，例如 stage、duration、work_dir、settle_steps、q、release。
{JSON_ONLY_LINE}
"""


def sanitize_ur5e_field_atomic_parameters(action: str, parameters: dict[str, Any]) -> dict[str, Any]:
    normalized, _reason = runtime_skills.normalize_parameters(str(action), parameters or {})
    return normalized or {}
