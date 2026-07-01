"""Runtime UR5e skill metadata adapter.

All UR5e planner/critic code should read skill names and parameter contracts
through this module instead of keeping local copies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _runtime_registry() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    ur5e_root = repo_root / "ur5e_mujoco"
    ur5e_root_text = str(ur5e_root)
    if ur5e_root.exists() and ur5e_root_text not in sys.path:
        sys.path.insert(0, ur5e_root_text)
    from skills import registry  # type: ignore

    return registry


def allowed_actions(scenario_id: str | None = "") -> set[str]:
    return {str(action) for action in _runtime_registry().allowed_actions(scenario_id)}


def skill_specs(scenario_id: str | None = "") -> dict[str, Any]:
    registry = _runtime_registry()
    raw = registry.field_atomic_skill_registry()
    allowed = allowed_actions(scenario_id)
    return {action: spec for action, spec in raw.items() if action in allowed}


def skill_description(action: str) -> str:
    return str(_runtime_registry().skill_description(action))


def skill_signature(action: str) -> str:
    return str(_runtime_registry().skill_signature(action))


def parameter_schema(action: str) -> dict[str, Any]:
    return dict(_runtime_registry().skill_parameter_schema(action))


def parameter_keys(action: str) -> set[str]:
    return set(parameter_schema(action))


def normalize_parameters(action: str, params: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    return _runtime_registry().normalize_parameters(action, params)


def render_skill_specs(scenario_id: str | None = "") -> str:
    lines: list[str] = []
    for action in sorted(allowed_actions(scenario_id)):
        schema = parameter_schema(action)
        lines.append(f"- {skill_signature(action)}")
        lines.append(f"  description: {skill_description(action)}")
        lines.append(f"  parameters: {schema}")
    return "\n".join(lines)
