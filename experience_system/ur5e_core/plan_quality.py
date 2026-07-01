"""Plan-quality checks for UR5e skill sequences using runtime skill metadata."""

from __future__ import annotations

from typing import Any

from experience_core.schema import canonical_skill_action

from . import runtime_skills


def count_ur5e_plan_quality_issues(
    steps: list[dict[str, Any]],
    *,
    allowed_actions: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    allowed = {
        canonical_skill_action(str(item))
        for item in (allowed_actions or runtime_skills.allowed_actions())
        if str(item)
    }
    invalid = 0
    actions: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = canonical_skill_action(str(step.get("action") or step.get("name") or step.get("skill") or ""))
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if not action:
            continue
        actions.append(action)
        if allowed and action not in allowed:
            invalid += 1
            continue
        normalized, _reason = runtime_skills.normalize_parameters(action, params)
        if normalized is None:
            invalid += 1
    return {
        "invalid_plan_count": invalid,
        "unsafe_gripper_action_count": 0,
        "candidate_actions": actions,
        "quality_status": "fail" if invalid else "pass",
    }
