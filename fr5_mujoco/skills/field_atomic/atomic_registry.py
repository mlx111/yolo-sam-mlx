from __future__ import annotations

from .atomic_schema import FR5FieldAtomicSkillSpec, default_field_atomic_skill_specs


def field_atomic_skill_registry() -> dict[str, FR5FieldAtomicSkillSpec]:
    return {spec.action: spec for spec in default_field_atomic_skill_specs()}
