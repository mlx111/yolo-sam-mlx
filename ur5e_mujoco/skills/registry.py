"""UR5e field-atomic skill registry.

Only the current experiment skill set is public here.  Historical action names
are handled by the migration tool, not by the runtime registry.
"""

from __future__ import annotations

from typing import Any, Callable

from skills.context import Ur5eSkillContext
from skills.field_atomic.atomic_executor import Ur5eFieldAtomicSkillExecutor
from skills.field_atomic.atomic_registry import field_atomic_skill_registry

def allowed_actions(scenario_id: str | None = "") -> set[str]:
    del scenario_id
    return {
        action
        for action, spec in field_atomic_skill_registry().items()
        if bool(getattr(spec, "llm_visible", True))
    }


def skill_description(action: str) -> str:
    spec = field_atomic_skill_registry().get(str(action))
    return spec.description if spec is not None else "UR5e MuJoCo recovery skill."


def skill_signature(action: str) -> str:
    action = str(action)
    spec = field_atomic_skill_registry().get(action)
    if spec is None:
        return f"{action}()"
    params = list(getattr(spec, "parameter_schema", {}) or {})
    if not params:
        return f"{action}()"
    return f"{action}(" + ", ".join(f"{key}=<value>" for key in params) + ")"


def skill_parameter_schema(action: str) -> dict[str, Any]:
    spec = field_atomic_skill_registry().get(str(action))
    return dict(getattr(spec, "parameter_schema", {}) or {}) if spec is not None else {}


def skill_required_any(action: str) -> tuple[tuple[str, ...], ...]:
    spec = field_atomic_skill_registry().get(str(action))
    return tuple(getattr(spec, "required_any", ()) or ()) if spec is not None else ()


def skill_parameter_ranges(action: str) -> dict[str, tuple[float, float]]:
    spec = field_atomic_skill_registry().get(str(action))
    return dict(getattr(spec, "parameter_ranges", {}) or {}) if spec is not None else {}


def skill_parameter_enums(action: str) -> dict[str, list[str]]:
    spec = field_atomic_skill_registry().get(str(action))
    return dict(getattr(spec, "parameter_enums", {}) or {}) if spec is not None else {}


def skill_parameter_vector_lengths(action: str) -> dict[str, int]:
    spec = field_atomic_skill_registry().get(str(action))
    return dict(getattr(spec, "parameter_vector_lengths", {}) or {}) if spec is not None else {}


def normalize_parameters(action: str, params: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    action = str(action)
    if action not in allowed_actions():
        return None, f"action_not_allowed:{action}"
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return None, "parameters_not_object"
    schema = skill_parameter_schema(action)
    ranges = skill_parameter_ranges(action)
    enums = skill_parameter_enums(action)
    vector_lengths = skill_parameter_vector_lengths(action)
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if key not in schema:
            continue
        if key in enums:
            text = str(value).strip().lower()
            if text not in {str(item).lower() for item in enums[key]}:
                return None, f"{action}_invalid_{key}"
            cleaned[key] = text
            continue
        if key in vector_lengths:
            length = int(vector_lengths[key])
            if not isinstance(value, (list, tuple)) or len(value) < length:
                return None, f"{action}_invalid_{key}"
            cleaned[key] = [float(value[index]) for index in range(length)]
            continue
        if key in ranges:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None, f"{action}_invalid_{key}"
            low, high = ranges[key]
            cleaned[key] = float(min(max(numeric, low), high))
            continue
        cleaned[key] = value
    required_any = skill_required_any(action)
    if required_any and not any(all(key in cleaned for key in group) for group in required_any):
        return None, f"{action}_missing_required_parameters"
    return cleaned, ""


def build_action_map(experiment: Any, default_pregrasp_height: float = 0.127, scenario_id: str | None = "") -> dict[str, Callable[[dict], Any]]:
    context = Ur5eSkillContext(experiment, default_pregrasp_height=default_pregrasp_height)
    executor = Ur5eFieldAtomicSkillExecutor(context, default_pregrasp_height=default_pregrasp_height)
    allowed = allowed_actions(scenario_id)

    def _handler(action: str) -> Callable[[dict], Any]:
        def run(params: dict) -> Any:
            result = executor.execute(action, params)
            if not result.success:
                raise RuntimeError(result.message)
            return result

        return run

    return {action: _handler(action) for action in allowed}
