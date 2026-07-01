from __future__ import annotations

from typing import Any

from .atomic_schema import Ur5eFieldAtomicSkillSpec, default_field_atomic_skill_specs


def field_atomic_skill_registry() -> dict[str, Ur5eFieldAtomicSkillSpec]:
    return {spec.action: spec for spec in default_field_atomic_skill_specs()}


def field_atomic_action_names() -> list[str]:
    return list(field_atomic_skill_registry().keys())


def field_atomic_action_descriptions() -> dict[str, dict[str, Any]]:
    return {
        action: {
            "name": spec.name,
            "action": spec.action,
            "description": spec.description,
            "parameter_schema": dict(spec.parameter_schema),
            "source_skill": spec.source_skill,
        }
        for action, spec in field_atomic_skill_registry().items()
    }

