"""UR5e skill semantic compatibility wrappers.

The UR5e runtime does not keep an independent handwritten precondition/effect
table. Skill names and descriptions come from the runtime skill registry.
"""

from __future__ import annotations

from typing import Any

from experience_core.skill_semantics import SkillSemantics, _semantics, validate_skill_semantic_plan

from . import runtime_skills


def default_ur5e_skill_semantics() -> dict[str, SkillSemantics]:
    return {
        name: _semantics(name, description=runtime_skills.skill_description(name))
        for name in sorted(runtime_skills.allowed_actions())
    }


def validate_ur5e_skill_semantic_plan(
    plan: dict[str, Any],
    *,
    initial_facts: set[str] | None = None,
) -> dict[str, Any]:
    return validate_skill_semantic_plan(plan, skill_semantics=default_ur5e_skill_semantics(), initial_facts=initial_facts)
