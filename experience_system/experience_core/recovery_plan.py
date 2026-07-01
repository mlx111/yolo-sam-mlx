"""LLM-generated, structured recovery plans for robot anomaly handling."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .llm_provider import JSON_ONLY_LINE, invoke_llm, parse_json_payload
from .schema import GALAXEA_R1PRO_TORSO_NAMESPACE
from .skill_semantics import default_galaxea_field_atomic_skill_semantics, validate_skill_semantic_plan


def stable_plan_id(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return "recovery_plan_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def allowed_skill_set(candidates: list[Any]) -> set[str]:
    skills: set[str] = set()
    for candidate in candidates:
        for step in getattr(candidate, "steps", []) or []:
            if str(step):
                skills.add(str(step))
    return skills


def recovery_skill_parameter_guidance() -> dict[str, Any]:
    return {
        "move_to_pregrasp": {
            "pregrasp_distance": [0.02, 0.08],
            "grasp_offset_z": [0.0, 0.02],
            "steps": [200, 900],
        },
        "approach_object": {
            "approach_velocity_limit": [0.15, 0.6],
            "approach_segment_count": [4, 12],
        },
        "reposition_base_for_reach": {
            "lateral_offset": [-0.08, 0.08],
            "forward_offset": [-0.08, 0.02],
            "yaw_delta": [-0.15, 0.15],
        },
        "adjust_torso_for_reach": {
            "height_level": ["low", "mid", "high"],
        },
    }


def recovery_plan_prompt(
    *,
    scenario: str,
    condition: str,
    planner_input: dict[str, Any],
    candidate: Any,
    candidates: list[Any],
    max_steps: int = 16,
) -> str:
    allowed = sorted(allowed_skill_set(candidates))
    candidate_payload = {
        "candidate_id": getattr(candidate, "candidate_id", ""),
        "description": getattr(candidate, "description", ""),
        "steps": list(getattr(candidate, "steps", []) or []),
    }
    return f"""
You generate a compact, executable robot anomaly recovery plan.

Use only the allowed skill names listed below. Do not invent robot abilities,
hidden sensors, coordinates, or controller APIs. The plan is not final until
it passes deterministic validation, MuJoCo sandbox rollout, and critic checks.

Scenario: {scenario}
Condition: {condition}

Seed candidate:
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

Allowed skills:
{json.dumps(allowed, ensure_ascii=False, indent=2)}

Recovery skill parameter guidance:
{json.dumps(recovery_skill_parameter_guidance(), ensure_ascii=False, indent=2)}

Planner input:
{json.dumps(planner_input, ensure_ascii=False, indent=2)}

Return only one JSON object with this schema:
{{
  "goal": "short physical recovery goal",
  "steps": [
    {{
      "stage": "candidate_generation|candidate_ranking|sandbox_rewrite|execution_writeback|execution",
      "action": "one allowed skill name",
      "parameters": {{
        "use only scalar numbers, short strings, or small flat arrays when needed"
      }},
      "reason": "short reason grounded in planner_input"
    }}
  ],
  "constraints": ["short safety or executability constraint"],
  "risk_notes": ["short risk note grounded in planner_input"],
  "evidence_ids": ["experience ids copied from planner_input"],
  "confidence": 0.0
}}

Constraints:
- steps must contain 1 to {max_steps} items.
- every action must exactly match one allowed skill.
- parameters must stay inside the recovery skill parameter guidance whenever the action is a recovery skill.
- evidence_ids must be copied from planner_input memory ids only.
- confidence must be a number from 0 to 1.
- Return JSON only, no Markdown.
"""


def invoke_recovery_plan_llm(
    prompt: str,
    *,
    provider: str,
    model: str = "",
) -> dict[str, Any]:
    raw_text = invoke_llm(
        f"{prompt}\n\n{JSON_ONLY_LINE}",
        provider=provider,
        model=model,
        system_prompt="You return JSON only.",
    )
    payload = parse_json_payload(raw_text, prefer_array=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"LLM recovery plan response was not a JSON object: {str(raw_text)[:500]}")
    return payload


def planner_input_evidence_ids(planner_input: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for section, key in (
        ("generation_guidance", "positive_memory_ids"),
        ("ranking_guidance", "risk_memory_ids"),
        ("ranking_guidance", "gap_memory_ids"),
        ("rewrite_guidance", "critic_warning_ids"),
        ("writeback_guidance", "recent_execution_ids"),
    ):
        value = (planner_input.get(section) or {}).get(key) or []
        for item in value:
            if str(item):
                ids.add(str(item))
    return ids


def _coerce_bounded_parameter(action: str, key: str, value: Any) -> Any:
    guidance = recovery_skill_parameter_guidance().get(action)
    if not isinstance(guidance, dict) or key not in guidance:
        return None
    bounds = guidance[key]
    if not isinstance(bounds, list) or not bounds:
        return None
    if all(isinstance(item, (int, float)) for item in bounds) and len(bounds) == 2:
        if not isinstance(value, (int, float)):
            return None
        low = float(bounds[0])
        high = float(bounds[1])
        return max(low, min(float(value), high))
    allowed = {str(item) for item in bounds}
    text = str(value)
    return text if text in allowed else None


def sanitize_recovery_parameters(action: str, parameters: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    if not isinstance(parameters, dict):
        return clean
    for key, value in parameters.items():
        bounded = _coerce_bounded_parameter(action, str(key), value)
        if bounded is not None:
            clean[str(key)] = bounded
    return clean


def normalize_recovery_plan(
    raw_plan: dict[str, Any],
    *,
    scenario: str,
    condition: str,
    candidate: Any,
    candidates: list[Any],
    planner_input: dict[str, Any],
    provider: str = "",
    model: str = "",
    max_steps: int = 16,
) -> dict[str, Any]:
    if not isinstance(raw_plan, dict):
        raise RuntimeError("recovery plan must be a JSON object")

    allowed = allowed_skill_set(candidates)
    evidence_allowed = planner_input_evidence_ids(planner_input)
    raw_steps = raw_plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RuntimeError("recovery plan must contain non-empty steps")
    if len(raw_steps) > max_steps:
        raise RuntimeError(f"recovery plan has too many steps: {len(raw_steps)} > {max_steps}")

    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise RuntimeError(f"step[{index}] is not an object")
        action = str(raw_step.get("action") or "").strip()
        if action not in allowed:
            raise RuntimeError(f"step[{index}] action is not allowed: {action}")
        raw_parameters = raw_step.get("parameters") if isinstance(raw_step.get("parameters"), dict) else {}
        parameters = sanitize_recovery_parameters(action, raw_parameters)
        steps.append({
            "stage": str(raw_step.get("stage") or "execution").strip() or "execution",
            "action": action,
            "parameters": parameters,
            "reason": str(raw_step.get("reason") or "").strip(),
        })

    evidence_ids = []
    for item in raw_plan.get("evidence_ids") or []:
        evidence_id = str(item)
        if evidence_id not in evidence_allowed:
            raise RuntimeError(f"recovery plan uses unknown evidence_id: {evidence_id}")
        evidence_ids.append(evidence_id)

    confidence = raw_plan.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise RuntimeError("recovery plan confidence must be 0..1")

    action_signature = [step["action"] for step in steps]
    return {
        "schema_version": "llm_recovery_plan_v1",
        "plan_id": stable_plan_id(scenario, condition, getattr(candidate, "candidate_id", ""), action_signature),
        "scenario": scenario,
        "condition": condition,
        "source_candidate_id": str(getattr(candidate, "candidate_id", "")),
        "goal": str(raw_plan.get("goal") or "").strip(),
        "steps": steps,
        "candidate_steps": action_signature,
        "constraints": [str(item) for item in raw_plan.get("constraints") or [] if str(item)],
        "risk_notes": [str(item) for item in raw_plan.get("risk_notes") or [] if str(item)],
        "evidence_ids": evidence_ids,
        "confidence": round(float(confidence), 4),
        "llm_provider": provider,
        "llm_model": model,
        "planner_input": planner_input,
        "validation": {
            "schema_status": "pass",
            "allowed_skill_status": "pass",
            "sandbox_status": "not_run",
            "critic_status": "not_run",
        },
    }


def validate_recovery_plan_semantics(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a recovery plan through skill precondition/effect semantics."""

    skill_namespace = str(plan.get("skill_namespace") or "") if isinstance(plan, dict) else ""
    semantics = default_galaxea_field_atomic_skill_semantics() if skill_namespace == GALAXEA_R1PRO_TORSO_NAMESPACE else None
    result = validate_skill_semantic_plan(plan, skill_semantics=semantics)
    _apply_goal_completion_checks(result, plan)
    result["schema_version"] = "recovery_plan_semantic_validation_v2"
    result["scenario"] = str(plan.get("scenario") or "").upper() if isinstance(plan, dict) else ""
    result["validator"] = "skill_precondition_effect_graph"
    result["skill_namespace"] = skill_namespace
    return result


def _apply_goal_completion_checks(result: dict[str, Any], plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        return
    goal = str(plan.get("goal") or "").lower()
    final_facts = {str(item) for item in result.get("final_facts") or []}
    issues = result.setdefault("issues", [])
    required: list[tuple[str, str]] = []
    if any(token in goal for token in ("lift", "提升", "抬起", "抓取")):
        required.append(("lift_attempted", "goal requires grasp-and-lift completion, but plan never reaches lift."))
    if any(token in goal for token in ("place", "放置", "放到", "放在")):
        required.extend([
            ("transport_attempted", "goal requires moving toward the placement target, but plan never transports."),
            ("place_lowered", "goal requires lowering before release, but plan never lowers the held object."),
            ("object_released", "goal requires releasing the object at placement, but plan never opens the gripper."),
        ])
    for fact, message in required:
        if fact in final_facts:
            continue
        issues.append({
            "severity": "fatal",
            "code": "goal_completion_missing",
            "message": message,
            "missing_fact": fact,
        })
    fatal_count = sum(1 for issue in issues if issue.get("severity") == "fatal")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    result["fatal_count"] = fatal_count
    result["warning_count"] = warning_count
    result["status"] = "fail" if fatal_count else "warn" if warning_count else "pass"


def recovery_plan_to_candidate(
    plan: dict[str, Any],
    *,
    base_candidate: Any,
    candidates: list[Any],
    candidate_cls: type,
) -> Any:
    steps = list(plan.get("candidate_steps") or [])
    matched = None
    for candidate in candidates:
        if list(getattr(candidate, "steps", []) or []) == steps:
            matched = candidate
            break
    if matched is not None:
        candidate_id = str(getattr(matched, "candidate_id", ""))
        description = str(getattr(matched, "description", ""))
        executable = bool(getattr(matched, "executable", False))
    else:
        candidate_id = str(plan.get("plan_id") or stable_plan_id(steps))
        description = str(plan.get("goal") or "LLM-generated recovery plan")
        executable = False
    return candidate_cls(
        candidate_id=candidate_id,
        description=description,
        steps=steps,
        executable=executable,
        planner_generated=True,
        planner_source_id=str(getattr(base_candidate, "candidate_id", "")),
        planner_reason="llm_recovery_plan",
    )


def build_validated_robot_plan(
    *,
    scenario: str,
    condition: str,
    selected_candidate_id: str,
    selected_steps: list[str],
    sandbox_report: dict[str, Any] | None = None,
    fused_score: dict[str, Any] | None = None,
    recovery_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sandbox_report = sandbox_report or {}
    fused_score = fused_score or {}
    if recovery_plan:
        steps = [
            {
                "action": str(step.get("action") or ""),
                "parameters": step.get("parameters") if isinstance(step.get("parameters"), dict) else {},
                "stage": str(step.get("stage") or ""),
                "reason": str(step.get("reason") or ""),
            }
            for step in recovery_plan.get("steps") or []
            if isinstance(step, dict) and str(step.get("action") or "")
        ]
        plan_id = str(recovery_plan.get("plan_id") or stable_plan_id(scenario, condition, selected_candidate_id, selected_steps))
        goal = str(recovery_plan.get("goal") or "")
        constraints = [str(item) for item in recovery_plan.get("constraints") or [] if str(item)]
        risk_notes = [str(item) for item in recovery_plan.get("risk_notes") or [] if str(item)]
        confidence = float(recovery_plan.get("confidence") or 0.0)
    else:
        steps = [
            {
                "action": str(action),
                "parameters": {},
                "stage": "execution",
                "reason": "",
            }
            for action in selected_steps
            if str(action)
        ]
        plan_id = stable_plan_id(scenario, condition, selected_candidate_id, selected_steps)
        goal = ""
        constraints = []
        risk_notes = []
        confidence = 0.0
    return {
        "schema_version": "validated_robot_plan_v1",
        "plan_id": plan_id,
        "scenario": scenario,
        "condition": condition,
        "selected_candidate_id": selected_candidate_id,
        "goal": goal,
        "steps": steps,
        "constraints": constraints,
        "risk_notes": risk_notes,
        "confidence": round(confidence, 4),
        "validation": {
            "plan_semantic_validation": recovery_plan.get("semantic_validation", {}) if recovery_plan else {},
            "sandbox_status": str(sandbox_report.get("critic_status") or "not_run"),
            "task_success": bool(sandbox_report.get("task_success", False)),
            "sandbox_score": float(sandbox_report.get("sandbox_score") or 0.0),
            "decision": str(fused_score.get("decision") or ""),
            "combined_score": float(fused_score.get("combined_score") or 0.0),
        },
    }
